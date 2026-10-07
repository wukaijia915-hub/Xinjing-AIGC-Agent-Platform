"""自适应时间窗测试：短片段置信度衰减与真实时长计算。"""
import numpy as np
import pytest

from vibraimage.pipeline.engine import VibraImageEngine
from vibraimage.utils.constants import (
    DEFAULT_WINDOW_FRAMES, MIN_WINDOW_FRAMES, MIN_CONFIDENCE,
)


def _make_frames(n: int, h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """生成 n 帧带运动噪声的灰度帧序列（模拟人脸 ROI 区域）。"""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        base = np.full((h, w), 120.0, dtype=np.float32)
        # 模拟头部微振动: 缓慢平移 + 高频抖动
        shift = int(2 * np.sin(i / 4.0))
        jitter = rng.normal(0, 1.2, (h, w)).astype(np.float32)
        if shift >= 0:
            base[shift:, :] = 120.0
        frames.append(base + jitter)
    return np.array(frames)


class TestAdaptiveWindow:
    def test_full_window_confidence_one(self):
        """满窗口（100 帧）置信度应为 1.0。"""
        engine = VibraImageEngine()
        frames = _make_frames(DEFAULT_WINDOW_FRAMES)
        session = engine.process_frames(frames)
        assert session.n_windows >= 1
        assert session.confidence == pytest.approx(1.0, abs=1e-6)
        assert all(w.confidence == pytest.approx(1.0) for w in session.window_results)

    def test_short_window_confidence_decay(self):
        """16 帧短片段：置信度按 sqrt(16/100)≈0.4 衰减，且能成功产出结果。"""
        engine = VibraImageEngine()
        frames = _make_frames(16)
        session = engine.process_frames(frames)
        assert session.n_windows == 1
        expected = max(MIN_CONFIDENCE, (16 / DEFAULT_WINDOW_FRAMES) ** 0.5)
        assert session.confidence == pytest.approx(expected, abs=1e-6)
        assert session.window_results[0].frames_in_window == 16

    def test_short_window_duration_correct(self):
        """短窗口时长应基于实际帧数（16/30≈0.53s），而非按 100 帧高估。"""
        engine = VibraImageEngine()
        session = engine.process_frames(_make_frames(16))
        assert session.duration_sec == pytest.approx(16 / 30.0, abs=1e-3)

    def test_min_confidence_floor(self):
        """8 帧（下限）置信度应等于 MIN_CONFIDENCE。"""
        engine = VibraImageEngine()
        session = engine.process_frames(_make_frames(MIN_WINDOW_FRAMES))
        assert session.confidence == pytest.approx(MIN_CONFIDENCE, abs=1e-6)

    def test_below_min_raises(self):
        """低于最小窗口（<8 帧）应明确报错而非静默产出空结果。"""
        engine = VibraImageEngine()
        with pytest.raises(ValueError):
            engine.process_frames(_make_frames(4))

    def test_window_result_dict_includes_confidence(self):
        """to_dict 应输出 confidence 与 frames_in_window。"""
        engine = VibraImageEngine()
        session = engine.process_frames(_make_frames(16))
        d = session.window_results[0].to_dict()
        assert "confidence" in d and "frames_in_window" in d
        assert d["frames_in_window"] == 16

    def test_session_dict_includes_confidence(self):
        """会话 to_dict 应输出聚合置信度。"""
        engine = VibraImageEngine()
        session = engine.process_frames(_make_frames(DEFAULT_WINDOW_FRAMES))
        d = session.to_dict()
        assert "confidence" in d
        assert d["confidence"] == pytest.approx(1.0, abs=1e-6)

    def test_long_video_multiple_windows(self):
        """250 帧视频应产生多个满置信度窗口（stride 50 → 4 窗）。"""
        engine = VibraImageEngine()
        session = engine.process_frames(_make_frames(250))
        assert session.n_windows == 4
        assert session.confidence == pytest.approx(1.0, abs=1e-6)
