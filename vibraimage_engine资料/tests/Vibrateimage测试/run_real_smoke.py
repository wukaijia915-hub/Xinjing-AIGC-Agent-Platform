#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实视频冒烟测试: 对任意命名的视频直接跑引擎, 输出 E1-E12 + stability + K,
并对照常模 (NORMAL_NORMS ± 3SD) 标记越界参数。

与 run_validation.py 的区别: 不做统计检验 (ANOVA/CV/ICC 需要重复试次),
仅验证引擎在真实视频上是否产出合理量级、且参数方向是否符合情绪语义。

用法:
    python run_real_smoke.py [--video-dir <目录>]
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ENGINE_ROOT = BASE.parent
REPO_ROOT = BASE.parent.parent
# 真实视频默认位于引擎上一级的 data/ 目录
DEFAULT_VIDEO_DIR = ENGINE_ROOT.parent / "data"
sys.path.insert(0, str(REPO_ROOT))

from backend.vibraimage.pipeline.engine import VibraImageEngine          # noqa: E402
from backend.vibraimage.pipeline.face_detector import FaceDetector       # noqa: E402
from backend.vibraimage.utils.constants import NORMAL_NORMS, NORMAL_SDS  # noqa: E402

logger = logging.getLogger("run_real_smoke")

EMOTION_PARAMS = [
    "aggression", "stress", "tension", "suspect", "balance", "charm", "energy",
    "self_regulation", "inhibition", "neuroticism", "depression", "happiness",
]
CSV_COLS = ["video", "label"] + EMOTION_PARAMS + ["stability", "K_value"]

# 中文名 -> 可读标签 (含大致情绪方向, 用于定性判断)
VIDEO_LABELS = {
    "开心平静": "开心平静(happy/calm)",
    "委屈哭泣": "委屈哭泣(crying)",
    "失望": "失望(disappointment)",
    "激动": "激动(excited)",
}


def probe(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def build_engine(window_frames, stride, face_detector):
    return VibraImageEngine(
        window_frames=window_frames,
        window_stride=stride,
        freq_method="zerocross",
        face_detector=face_detector,
    )


def process_one(path: Path, label: str, face_detector) -> dict:
    fps, n_frames = probe(path)
    # 短于默认 100 帧窗口的视频: 用整段作为单窗, 保证能产出参数
    if n_frames < 100:
        win, stride = max(n_frames, 2), max(n_frames, 2)
        logger.warning(f"{path.name}: 仅 {n_frames} 帧 (<100), 整段单窗处理")
    else:
        win, stride = 100, 50

    engine = build_engine(win, stride, face_detector)
    try:
        result = engine.process_video(str(path))
    except Exception as e:
        logger.error(f"{path.name}: 处理失败 {e}")
        return {"video": path.name, "label": label, "error": str(e)}

    em = result.to_dict()["emotions"]
    return {
        "video": path.name,
        "label": label,
        "aggression": em["aggression"], "stress": em["stress"], "tension": em["tension"],
        "suspect": em["suspect"], "balance": em["balance"], "charm": em["charm"],
        "energy": em["energy"], "self_regulation": em["self_regulation"],
        "inhibition": em["inhibition"], "neuroticism": em["neuroticism"],
        "depression": em["depression"], "happiness": em["happiness"],
        "stability": result.to_dict()["psychophysiological"]["stability"],
        "K_value": result.K_value,
        "n_frames": n_frames, "fps": fps,
    }


def _norm_key(param: str) -> str:
    return "suspicious" if param == "suspect" else param


def main():
    parser = argparse.ArgumentParser(description="真实视频冒烟测试")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--output", default=str(BASE / "results" / "real_smoke.csv"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    video_dir = Path(args.video_dir).resolve()
    videos = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
    )
    logger.info(f"发现 {len(videos)} 个视频: {video_dir}")
    for v in videos:
        logger.info(f"  - {v.name}")

    model_path = ENGINE_ROOT / "yolov8n.pt"
    face_detector = FaceDetector(model_path=str(model_path)) if model_path.exists() else None

    rows = []
    for vp in videos:
        label = VIDEO_LABELS.get(vp.stem, vp.stem)
        logger.info(f"处理: {vp.name} -> {label}")
        row = process_one(vp, label, face_detector)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # 仅保留稳定列顺序, 丢弃 error/内联信息列到 CSV (error 列存在时单独保留)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"结果已保存: {out}")

    # ---- 打印对照表 (参数 × 视频) ----
    print("\n" + "=" * 90)
    print("参数对照表 (越界标记: * = 超出常模±3SD)")
    print("=" * 90)
    header = "参数".ljust(16) + "".join(f"{r['label'][:12]:>14}" for r in rows) + f"{'常模±3SD':>22}"
    print(header)
    print("-" * len(header))

    for param in EMOTION_PARAMS:
        key = _norm_key(param)
        norm = NORMAL_NORMS.get(key)
        sd = NORMAL_SDS.get(key)
        line = param.ljust(16)
        for r in rows:
            v = r.get(param, float("nan"))
            if isinstance(v, (int, float)) and not np.isnan(v):
                flag = "*" if (norm and sd and not (norm - 3 * sd <= v <= norm + 3 * sd)) else " "
                line += f"{v:>13.1f}{flag}"
            else:
                line += f"{'—':>14}"
        if norm and sd:
            line += f"  [{norm - 3 * sd:.1f}, {norm + 3 * sd:.1f}]"
        print(line)

    print("-" * len(header))
    for extra in ["stability", "K_value"]:
        line = extra.ljust(16)
        for r in rows:
            v = r.get(extra, float("nan"))
            if isinstance(v, (int, float)) and not np.isnan(v):
                line += f"{v:>14.1f}"
            else:
                line += f"{'—':>14}"
        line += f"  {'(无常模)':>18}"
        print(line)

    # 帧数/时长
    line = "帧数/时长".ljust(16)
    for r in rows:
        n = r.get("n_frames", 0)
        fps = r.get("fps", 0) or 1
        tag = "单窗" if n < 100 else ""
        line += f"{n}帧{tag}({n / fps:.1f}s)".rjust(14)
    print(line)
    logger.info("冒烟测试完成")


if __name__ == "__main__":
    main()
