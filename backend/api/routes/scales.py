"""
心理量表 API — 题目获取 + 自评提交 + 结果查询 + AI交叉验证
"""

from __future__ import annotations
import os
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/scales", tags=["心理量表"])

SCALES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "scales")
os.makedirs(SCALES_DIR, exist_ok=True)


class ScaleSubmitRequest(BaseModel):
    student_id: int = Field(..., description="学生ID")
    scale_type: str = Field(..., description="量表类型: SAS/SDS/SCL-90")
    answers: list[int] = Field(..., description="答案列表（选项序号 1-4 或 1-5）")


def _load_scale(scale_type: str) -> dict:
    path = os.path.join(SCALES_DIR, f"{scale_type.upper()}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"量表 {scale_type} 不存在")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _score_scale(scale: dict, answers: list[int]) -> dict:
    scoring = scale["scoring"]
    reverse_items = scoring.get("reverse_items", [])
    n = len(scale["questions"])

    if len(answers) != n:
        raise HTTPException(status_code=400, detail=f"答案数量应为 {n}，实际收到 {len(answers)}")

    raw = 0
    for i, ans in enumerate(answers):
        qid = scale["questions"][i]["id"]
        if qid in reverse_items:
            ans = scoring["options"][-1] + scoring["options"][0] - ans
        raw += ans

    # 标准分
    if "standard_score_formula" in scoring:
        std = int(raw * 1.25)
    else:
        std = raw

    # 等级
    cutoff = scoring.get("cutoff", {})
    level = "normal"
    thresholds = ["normal", "mild", "moderate", "severe"]
    for lv in thresholds:
        if lv in cutoff:
            lo, hi = cutoff[lv]
            if lo <= std <= hi:
                level = lv
                break

    # 维度得分（SCL-90）
    dim_scores = {}
    if "dimensions" in scoring:
        for dim_name, item_ids in scoring["dimensions"].items():
            total = 0
            count = 0
            for i, ans in enumerate(answers):
                qid = scale["questions"][i]["id"]
                rev = qid in reverse_items
                a = scoring["options"][-1] + scoring["options"][0] - ans if rev else ans
                if qid in item_ids:
                    total += a
                    count += 1
            dim_scores[dim_name] = round(total / max(count, 1), 2)

    return {
        "raw_score": raw,
        "standard_score": std,
        "level": level,
        "dimension_scores": dim_scores,
    }


# =================== API ===================

@router.get("/{scale_type}", summary="获取量表题目")
async def get_scale(scale_type: str):
    """返回量表题目、选项标签和计分规则，前端据此渲染测验页面。"""
    scale = _load_scale(scale_type)
    return {
        "success": True,
        "data": {
            "code": scale["code"],
            "name": scale["name"],
            "description": scale["description"],
            "questions": scale["questions"],
            "options": scale["scoring"]["labels"],
            "question_count": scale["questions_count"],
        },
    }


@router.get("/list/all", summary="列出所有可用量表")
async def list_scales():
    scales = []
    for fname in sorted(os.listdir(SCALES_DIR)):
        if fname.endswith(".json"):
            path = os.path.join(SCALES_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
            scales.append({
                "code": s["code"], "name": s["name"],
                "description": s["description"], "question_count": s["questions_count"],
            })
    return {"success": True, "data": scales}


@router.post("/submit", summary="提交量表自评")
async def submit_scale(req: ScaleSubmitRequest):
    """提交量表自评答案，系统自动计分并存储。"""
    scale = _load_scale(req.scale_type)
    scoring = _score_scale(scale, req.answers)

    from backend.database import SessionLocal
    from backend.models.scale_result import ScaleResult

    db = SessionLocal()
    try:
        record = ScaleResult(
            student_id=req.student_id,
            scale_type=scale["code"],
            raw_score=scoring["raw_score"],
            standard_score=scoring["standard_score"],
            level=scoring["level"],
            dimension_scores=json.dumps(scoring["dimension_scores"], ensure_ascii=False),
            answers=json.dumps(req.answers),
            submitted_at=datetime.now().isoformat(),
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "success": True,
            "data": {
                "id": record.id,
                "scale_type": scale["code"],
                "standard_score": scoring["standard_score"],
                "level": scoring["level"],
                "dimension_scores": scoring["dimension_scores"],
                "cutoff": scale["scoring"].get("cutoff", {}),
            },
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/results/{student_id}", summary="查询学生量表结果")
async def get_results(student_id: int):
    from backend.database import SessionLocal
    from backend.models.scale_result import ScaleResult

    db = SessionLocal()
    try:
        records = db.query(ScaleResult).filter(
            ScaleResult.student_id == student_id
        ).order_by(ScaleResult.submitted_at.desc()).all()
        return {
            "success": True,
            "data": [
                {
                    "id": r.id, "scale_type": r.scale_type,
                    "standard_score": r.standard_score, "level": r.level,
                    "dimension_scores": json.loads(r.dimension_scores),
                    "submitted_at": r.submitted_at,
                }
                for r in records
            ],
        }
    finally:
        db.close()


@router.get("/crosscheck/{student_id}", summary="量表与AI情绪交叉验证")
async def crosscheck(student_id: int):
    """将量表结果与AI情绪数据对比，输出一致性报告。"""
    from backend.database import SessionLocal
    from backend.models.scale_result import ScaleResult
    from backend.models.emotion_record import EmotionRecord

    db = SessionLocal()
    try:
        scale_records = db.query(ScaleResult).filter(
            ScaleResult.student_id == student_id
        ).order_by(ScaleResult.submitted_at.desc()).limit(3).all()

        emotion_records = db.query(EmotionRecord).filter(
            EmotionRecord.student_id == student_id
        ).order_by(EmotionRecord.recorded_at.desc()).limit(30).all()

        if not emotion_records:
            return {"success": False, "detail": "该学生暂无AI情绪数据"}

        avg_score = sum(r.fused_score for r in emotion_records) / len(emotion_records)
        neg_ratio = sum(1 for r in emotion_records if r.fused_emotion in {"悲伤","焦虑","愤怒","恐惧"}) / len(emotion_records)

        checks = []
        for sr in scale_records:
            if sr.scale_type == "SAS":
                sas_severe = sr.standard_score >= 60
                ai_severe = avg_score < 0.4 and neg_ratio > 0.5
                checks.append({
                    "scale": "SAS（焦虑自评）",
                    "scale_score": sr.standard_score,
                    "scale_level": sr.level,
                    "ai_avg_score": round(avg_score, 2),
                    "ai_negative_ratio": round(neg_ratio, 2),
                    "consistent": sas_severe == ai_severe,
                    "detail": "量表与AI一致：均提示高风险" if (sas_severe and ai_severe) else
                              "量表与AI一致：均正常" if (not sas_severe and not ai_severe) else
                              "量表与AI不一致：建议复核",
                })
            elif sr.scale_type == "SDS":
                sds_severe = sr.standard_score >= 60
                ai_severe = avg_score < 0.4 and neg_ratio > 0.5
                checks.append({
                    "scale": "SDS（抑郁自评）",
                    "scale_score": sr.standard_score,
                    "scale_level": sr.level,
                    "ai_avg_score": round(avg_score, 2),
                    "ai_negative_ratio": round(neg_ratio, 2),
                    "consistent": sds_severe == ai_severe,
                    "detail": "量表与AI一致" if sds_severe == ai_severe else "量表与AI不一致：建议复核",
                })

        return {"success": True, "data": {"checks": checks, "ai_summary": {"avg_score": round(avg_score, 2), "negative_ratio": round(neg_ratio, 2)}}}
    finally:
        db.close()
