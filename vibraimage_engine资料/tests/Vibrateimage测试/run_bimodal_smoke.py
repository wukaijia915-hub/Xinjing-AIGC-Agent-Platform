#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地端到端冒烟测试：双模态情绪识别（前庭 + 面部融合）。

背景：
  双模态 = 面部(DeepFace) + 前庭(VibraImage) 融合。本地未安装 deepface/tensorflow
  （面部模态依赖）以及 pydantic-settings/sqlalchemy（backend.tools 包 __init__ 依赖），
  故本脚本只覆盖「前庭模态 + 融合逻辑」这两段可离线验证的生产同款代码：

    - 用真实的 backend/vibraimage 引擎（生产同款）跑视频 -> E1-E12 + 窗口序列 + K
    - 用真实的 EmotionMapper 做 E1-E12 -> V/A 映射（生产同款）
    - 用真实的 MultiModalEmotionFuser 融合逻辑（按文件直接加载，绕过 backend/tools/__init__）
    - 面部模态用「中性 stub」占位（uniform 概率 -> 熵置信度接近 0，融合自动偏向前庭）

验证目标（对应验收标准）：
    - fused_valence / fused_arousal ∈ [-1, 1]
    - fused_emotion_7 ∈ 7 类集合
    - 7 类结果与视频内容（calm/happy/sad）方向大致吻合
"""
import sys
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # -> Xinjing-AIGC-Agent-Platform
sys.path.insert(0, str(REPO))

from backend.vibraimage.pipeline.engine import VibraImageEngine
from backend.vibraimage.pipeline.face_detector import FaceDetector
from backend.vibraimage.mapping.emotion_mapper import EmotionMapper

# 直接加载 multi_modal_fusion.py，绕过 backend/tools/__init__.py（其依赖 config/obs SDK 未安装）
_fusion_path = REPO / "backend" / "tools" / "multi_modal_fusion.py"
_spec = importlib.util.spec_from_file_location("multi_modal_fusion_local", _fusion_path)
_fusion = importlib.util.module_from_spec(_spec)
sys.modules["multi_modal_fusion_local"] = _fusion  # dataclass 需要模块在 sys.modules 中
_spec.loader.exec_module(_fusion)
MultiModalEmotionFuser = _fusion.MultiModalEmotionFuser
EMOTION_7_CLASSES = _fusion.EMOTION_7_CLASSES

MODEL = REPO / "vibraimage_engine" / "yolov8n.pt"
VIDEO_DIR = REPO / "vibraimage_engine" / "Vibrateimage测试" / "data" / "s01"

FACIAL_NEUTRAL_STUB = {
    "facial_valence": 0.0,
    "facial_arousal": 0.0,
    "emotion_probs": {k: 1.0 / 7.0 for k in EMOTION_7_CLASSES},
}


def run_one(path: Path, mapper: EmotionMapper, fuser: MultiModalEmotionFuser):
    engine = VibraImageEngine(
        window_frames=100,
        window_stride=50,
        freq_method="zerocross",
        face_detector=FaceDetector(
            model_path=str(MODEL), roi_size=(224, 224), conf_threshold=0.5
        ),
    )
    session = engine.process_video(str(path))
    d = session.to_dict()
    em = d["emotions"]
    windows = d.get("windows", [])
    K = d.get("K_value", 0.0)

    mapped = mapper.map(em, K=K)
    vest = {
        "valence": mapped.valence,
        "arousal": mapped.arousal,
        "window_results": windows,
    }
    fused = fuser.fuse(dict(FACIAL_NEUTRAL_STUB), vest)
    return em, mapped, fused, K, d


def main():
    mapper = EmotionMapper()
    fuser = MultiModalEmotionFuser()

    for name in ["calm_01", "happy_01", "sad_01"]:
        p = VIDEO_DIR / (name + ".mp4")
        if not p.exists():
            print(f"[SKIP] {p} 不存在")
            continue

        print("=" * 88)
        print(f"视频: {name}.mp4  (标签: {name.split('_')[0]})")
        try:
            em, mapped, fused, K, d = run_one(p, mapper, fuser)
        except Exception as e:
            print(f"  [FAIL] 前庭引擎运行异常: {type(e).__name__}: {e}")
            continue

        n_win = d.get("n_windows", 0)
        print(f"  窗口数={n_win}  时长={d.get('duration_sec', 0):.1f}s  K={K:.2f}")
        print(
            f"  E1-E12: agg={em['aggression']:.1f} stress={em['stress']:.1f} "
            f"tension={em['tension']:.1f} balance={em['balance']:.1f} "
            f"charm={em['charm']:.1f} energy={em['energy']:.1f} "
            f"self_reg={em['self_regulation']:.1f} inhib={em['inhibition']:.1f} "
            f"neuro={em['neuroticism']:.1f} dep={em['depression']:.1f} "
            f"hap={em['happiness']:.1f}"
        )
        print(
            f"  前庭V/A(mapper): valence={mapped.valence:+.3f} "
            f"arousal={mapped.arousal:+.3f} intensity={mapped.intensity:.3f} "
            f"10类={mapped.emotion_label}"
        )

        if fused is None:
            print("  [FAIL] 融合返回 None")
            continue

        va_ok = -1.0 <= fused.fused_valence <= 1.0 and -1.0 <= fused.fused_arousal <= 1.0
        cls_ok = fused.fused_emotion_7 in EMOTION_7_CLASSES
        print(
            f"  融合V/A: fused_valence={fused.fused_valence:+.3f} "
            f"fused_arousal={fused.fused_arousal:+.3f} fused_conf={fused.fused_conf:.3f}"
        )
        print(
            f"  7类={fused.fused_emotion_7}  10类={fused.fused_emotion_10}  "
            f"权重 w_face={fused.w_face:.3f} w_vi={fused.w_vi:.3f}"
        )
        print(
            f"  [验收] V/A∈[-1,1]: {'PASS' if va_ok else 'FAIL'} | "
            f"7类∈集合: {'PASS' if cls_ok else 'FAIL'}"
        )

    print("=" * 88)


if __name__ == "__main__":
    main()
