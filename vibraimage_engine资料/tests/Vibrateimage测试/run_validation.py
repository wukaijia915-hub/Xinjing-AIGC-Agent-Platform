#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VibraImage 引擎内部复现验证脚本。

功能:
  1. 批量处理 validation/data 下所有视频, 提取 E1-E12 + stability + K 值
  2. 区分度检验 (单因素 ANOVA + Bonferroni 事后 t 检验 + 方向性检查)
  3. 稳定性检验 (变异系数 CV + 组内相关系数 ICC(1,1))
  4. 可视化 (箱线图 + 雷达图)
  5. 生成报告 report.md

用法:
    python run_validation.py [--data-dir data] [--output-dir results] [--plots-dir plots]
"""

import argparse
import json
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 路径与引擎导入: 脚本位于 <引擎根>/Vibrateimage测试/ 下
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent          # Vibrateimage测试/
ENGINE_ROOT = BASE.parent                        # 引擎根目录 vibraimage_engine/
REPO_ROOT = BASE.parent.parent                   # 仓库根目录 Xinjing-AIGC-Agent-Platform/
sys.path.insert(0, str(REPO_ROOT))

from backend.vibraimage.pipeline.engine import VibraImageEngine          # noqa: E402
from backend.vibraimage.pipeline.face_detector import FaceDetector       # noqa: E402
from backend.vibraimage.utils.constants import PARAM_NAMES_ZH            # noqa: E402

# ---------------------------------------------------------------------------
# 常量与参数定义
# ---------------------------------------------------------------------------
EMOTION_PARAMS = [
    "aggression", "stress", "tension", "suspect", "balance", "charm", "energy",
    "self_regulation", "inhibition", "neuroticism", "depression", "happiness",
]
STABILITY_PARAM = "stability"
K_PARAM = "K_value"

# CSV 数值列 (E1-E12 + stability + K)
NUMERIC_COLS = EMOTION_PARAMS + [STABILITY_PARAM, K_PARAM]
META_COLS = ["subject_id", "emotion", "trial"]
CSV_COLS = META_COLS + NUMERIC_COLS

EMOTIONS = ["happy", "calm", "sad"]

# 方向性预期 (VCE.pdf 情绪参数语义)
# high = 该情绪下此参数应高于其他情绪; low = 应低于其他情绪
# 注: 本批验证数据的第三类情绪为「悲伤(sad)」, 非原设计的 anxious。
#     sad 的方向预期按参数自身语义设定: 悲伤 → 抑郁(E11)↑ / 幸福感(E12)↓ / 能量(E7)↓。
EXPECTED_DIRECTIONS = {
    "happy": {"happiness": "high", "depression": "low", "stress": "low"},
    "calm": {"stress": "low", "balance": "high", "self_regulation": "high"},
    "sad": {"depression": "high", "happiness": "low", "energy": "low"},
}

# CV 评级阈值 (百分比)
CV_GOOD = 15.0
CV_ACCEPTABLE = 30.0

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger("run_validation")


def param_zh(name: str) -> str:
    """参数中文名 (带英文名)。"""
    # 引擎/常模内部用 'suspicious'，CSV 输出列名用 'suspect'
    lookup = "suspicious" if name == "suspect" else name
    zh = PARAM_NAMES_ZH.get(lookup, name)
    return f"{name}\n{zh}"


# ---------------------------------------------------------------------------
# 视频名解析
# ---------------------------------------------------------------------------
_FILENAME_RE = re.compile(r"^(?P<emotion>[A-Za-z]+)_(?P<trial>\d+)\.(mp4|avi|mov|mkv)$", re.IGNORECASE)


def parse_video_path(path: Path):
    """从路径解析 (subject_id, emotion, trial)。不匹配返回 None。"""
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
    emotion = m.group("emotion").lower()
    trial = int(m.group("trial"))
    subject_id = path.parent.name
    return subject_id, emotion, trial


# ---------------------------------------------------------------------------
# 引擎初始化
# ---------------------------------------------------------------------------
def build_engine(args) -> VibraImageEngine:
    """构建引擎, 优先使用引擎根目录下的 yolov8n.pt。"""
    model_path = ENGINE_ROOT / "yolov8n.pt"
    if model_path.exists():
        face_detector = FaceDetector(model_path=str(model_path))
    else:
        face_detector = None  # 由引擎自行处理 (后备 Haar Cascade)

    return VibraImageEngine(
        window_frames=args.window_frames,
        window_stride=args.stride,
        freq_method=args.freq_method,
        face_detector=face_detector,
    )


# ---------------------------------------------------------------------------
# 1. 批量处理
# ---------------------------------------------------------------------------
def process_all_videos(data_dir: Path, engine: VibraImageEngine) -> pd.DataFrame:
    """遍历 data_dir 下所有视频, 提取参数, 返回 DataFrame。"""
    videos = sorted(
        p for p in data_dir.rglob("*")
        if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
    )

    rows = []
    skipped = []

    for vp in videos:
        parsed = parse_video_path(vp)
        if parsed is None:
            skipped.append(str(vp))
            logger.warning(f"跳过无法解析命名的文件: {vp.name} (期望 <emotion>_<trial>.mp4)")
            continue

        subject_id, emotion, trial = parsed
        if emotion not in EMOTIONS:
            logger.warning(f"跳过未知情绪类别: {emotion} ({vp})")
            continue

        logger.info(f"处理: {subject_id}/{emotion}_{trial:02d} ...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")  # 让引擎内部 warning 透出
                result = engine.process_video(str(vp))
        except Exception as e:
            logger.error(f"处理失败 {vp}: {e}")
            skipped.append(str(vp))
            continue

        d = result.to_dict()
        em = d["emotions"]
        row = {
            "subject_id": subject_id,
            "emotion": emotion,
            "trial": trial,
            "aggression": em["aggression"],
            "stress": em["stress"],
            "tension": em["tension"],
            "suspect": em["suspect"],  # 引擎 to_dict 的键即 'suspect'
            "balance": em["balance"],
            "charm": em["charm"],
            "energy": em["energy"],
            "self_regulation": em["self_regulation"],
            "inhibition": em["inhibition"],
            "neuroticism": em["neuroticism"],
            "depression": em["depression"],
            "happiness": em["happiness"],
            "stability": d["psychophysiological"]["stability"],
            "K_value": result.K_value,
        }
        rows.append(row)

    if skipped:
        logger.warning(f"共跳过 {len(skipped)} 个文件")

    df = pd.DataFrame(rows, columns=CSV_COLS)
    return df


# ---------------------------------------------------------------------------
# 2. 区分度检验 (实验 1)
# ---------------------------------------------------------------------------
def _check_direction(df: pd.DataFrame, param: str, emotion: str, expected: str):
    """检查某参数在某情绪下的方向是否符合预期。返回 True/False/None。"""
    group = df[df["emotion"] == emotion][param].dropna()
    others = df[df["emotion"] != emotion][param].dropna()
    if len(group) == 0 or len(others) == 0:
        return None
    gm, om = float(group.mean()), float(others.mean())
    return (gm > om) if expected == "high" else (gm < om)


def discrimination(df: pd.DataFrame) -> pd.DataFrame:
    """对每个参数做单因素 ANOVA + 事后 t 检验 + 方向性检查。"""
    from scipy.stats import f_oneway, ttest_ind

    groups_by_emotion = {
        e: df[df["emotion"] == e] for e in EMOTIONS
    }

    records = []
    for param in NUMERIC_COLS:
        groups = [groups_by_emotion[e][param].dropna().to_numpy() for e in EMOTIONS]
        present = [(e, g) for e, g in zip(EMOTIONS, groups) if len(g) > 0]

        rec = {"param": param}
        for e in EMOTIONS:
            g = groups_by_emotion[e][param].dropna()
            rec[f"{e}_mean"] = float(g.mean()) if len(g) else float("nan")
            rec[f"{e}_std"] = float(g.std(ddof=1)) if len(g) > 1 else float("nan")
            rec[f"{e}_n"] = int(len(g))

        # ANOVA 需要 >=2 组且每组 >=2 样本
        valid_groups = [g for _, g in present if len(g) >= 2]
        if len(valid_groups) >= 2:
            F, p = f_oneway(*valid_groups)
            rec["F"] = float(F)
            rec["p_value"] = float(p)
            rec["significant"] = bool(p < 0.05)
        else:
            rec["F"] = float("nan")
            rec["p_value"] = float("nan")
            rec["significant"] = False

        # 事后两两 t 检验 (Welch, Bonferroni α=0.05/3)
        bonf_alpha = 0.05 / 3.0
        for a, b in [("happy", "calm"), ("happy", "sad"), ("calm", "sad")]:
            ga = groups_by_emotion[a][param].dropna().to_numpy()
            gb = groups_by_emotion[b][param].dropna().to_numpy()
            if len(ga) >= 2 and len(gb) >= 2:
                t, p = ttest_ind(ga, gb, equal_var=False)
                rec[f"t_{a}_{b}"] = float(t)
                rec[f"p_{a}_{b}"] = float(p)
                rec[f"sig_{a}_{b}"] = bool(p < bonf_alpha)
            else:
                rec[f"t_{a}_{b}"] = float("nan")
                rec[f"p_{a}_{b}"] = float("nan")
                rec[f"sig_{a}_{b}"] = False

        # 方向性检查
        checks = []
        for emotion, exp in EXPECTED_DIRECTIONS.items():
            if param in exp:
                ok = _check_direction(df, param, emotion, exp[param])
                arrow = {"high": "↑", "low": "↓"}[exp[param]]
                mark = "✓" if ok is True else ("✗" if ok is False else "?")
                checks.append(f"{emotion}:{arrow}{mark}")
        rec["direction_ok"] = (
            "all" if checks and all(c.endswith("✓") for c in checks)
            else ("mixed" if any(c.endswith("✓") for c in checks) else ("fail" if checks else "n/a"))
        )
        rec["direction_detail"] = "  ".join(checks) if checks else ""
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. 稳定性检验 (实验 2)
# ---------------------------------------------------------------------------
def _cv(values) -> float:
    """变异系数 (百分比)。均值接近 0 时返回 NaN。"""
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return float("nan")
    m = float(a.mean())
    if abs(m) < 1e-9:
        return float("nan")
    return float(np.std(a, ddof=1) / abs(m) * 100.0)


def _icc11(cells):
    """
    单因素随机效应 ICC(1,1)。

    cells: list of 1D arrays, 每个元素为同一目标 (subject×emotion) 的重复测量。
    要求各目标重复次数一致, 取最常见重复次数 k, 仅用 k 次重复的目标。
    返回 ICC(1,1) 或 NaN (样本不足)。
    """
    cells = [np.asarray(c, dtype=float) for c in cells if len(c) >= 2]
    if len(cells) < 2:
        return float("nan")

    # 取最常见重复次数 k
    k = max({len(c) for c in cells}, key=lambda x: sum(1 for c in cells if len(c) == x))
    cells = [c for c in cells if len(c) == k]
    if k < 2 or len(cells) < 2:
        return float("nan")

    data = np.vstack(cells)          # (n, k)
    n = data.shape[0]
    grand_mean = data.mean()
    msb = (k * ((data.mean(axis=1) - grand_mean) ** 2).sum()) / (n - 1)
    msw = ((data - data.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    if msw == 0 and msb == 0:
        return float("nan")
    return float((msb - msw) / (msb + (k - 1) * msw))


def stability(df: pd.DataFrame) -> pd.DataFrame:
    """对每个参数计算平均 CV 与 ICC(1,1)。"""
    records = []
    cells_map = {}  # param -> list of per-cell trial arrays

    for param in NUMERIC_COLS:
        cells = []
        for (subject, emotion), grp in df.groupby(["subject_id", "emotion"]):
            vals = grp[param].dropna().to_numpy()
            if len(vals) >= 1:
                cells.append(vals)
        cells_map[param] = cells

        cvs = [_cv(c) for c in cells if len(c) >= 2]
        mean_cv = float(np.nanmean(cvs)) if cvs else float("nan")
        icc = _icc11(cells)

        if np.isnan(mean_cv):
            rating = "n/a"
        elif mean_cv < CV_GOOD:
            rating = "优秀"
        elif mean_cv <= CV_ACCEPTABLE:
            rating = "可接受"
        else:
            rating = "不稳定"

        records.append({
            "param": param,
            "mean_cv_pct": mean_cv,
            "cv_rating": rating,
            "icc": icc,
            "n_cells": len(cells),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. 可视化
# ---------------------------------------------------------------------------
def plot_boxplot(df: pd.DataFrame, plots_dir: Path):
    """各参数在不同情绪下的箱线图。"""
    n = len(NUMERIC_COLS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes = np.atleast_1d(axes).ravel()

    for i, param in enumerate(NUMERIC_COLS):
        ax = axes[i]
        data = [df[df["emotion"] == e][param].dropna().to_numpy() for e in EMOTIONS]
        bp = ax.boxplot(data, tick_labels=EMOTIONS, patch_artist=True)
        colors = ["#4c72b0", "#55a868", "#c44e52"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(param_zh(param), fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    # 隐藏多余子图
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"各参数在不同情绪下的分布 ({' / '.join(EMOTIONS)})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = plots_dir / "emotion_boxplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"箱线图已保存: {out}")


def plot_radar(df: pd.DataFrame, plots_dir: Path):
    """每个被试的 3 种情绪在 E1-E12+K 空间的雷达图。"""
    radar_params = EMOTION_PARAMS + [K_PARAM]
    subjects = sorted(df["subject_id"].unique())

    # 对每个参数做 min-max 归一化 (跨全数据集) 到 [0,1], 使不同量纲可比
    norm = {}
    for p in radar_params:
        lo = float(df[p].min())
        hi = float(df[p].max())
        norm[p] = (lo, hi)

    def scale(p, v):
        lo, hi = norm[p]
        if hi - lo < 1e-9:
            return 0.5
        return (v - lo) / (hi - lo)

    n = len(subjects)
    ncols = min(3, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5),
                             subplot_kw=dict(polar=True))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).ravel()

    angles = np.linspace(0, 2 * np.pi, len(radar_params), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    emotion_color = {"happy": "#4c72b0", "calm": "#55a868", "sad": "#c44e52"}

    for idx, subj in enumerate(subjects):
        ax = axes[idx]
        subj_df = df[df["subject_id"] == subj]
        for emotion in EMOTIONS:
            edf = subj_df[subj_df["emotion"] == emotion]
            if len(edf) == 0:
                continue
            means = [scale(p, edf[p].mean()) for p in radar_params]
            values = means + means[:1]
            ax.plot(angles, values, linewidth=2, label=emotion,
                    color=emotion_color[emotion])
            ax.fill(angles, values, alpha=0.08, color=emotion_color[emotion])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_params, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"被试 {subj}", fontsize=10)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("各被试 3 情绪在 E1-E12+K 空间的轮廓对比 (min-max 归一化)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = plots_dir / "stability_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"雷达图已保存: {out}")


# ---------------------------------------------------------------------------
# 5. 报告生成
# ---------------------------------------------------------------------------
FIX_RECORD = """\
## 异常参数修复记录（第一步）

修复对象为引擎源码（`vibraimage/emotions/derived.py`、`psychophysiological.py`、`pipeline/engine.py`）。
公式来源均为 VCE.pdf (Minkin 2020) 各方程。

| 参数 | 修复前 | 修复后(同一 63s 视频窗口数据重构) | 常模 | 根因与修复 |
|---|---|---|---|---|
| E7 energy | 100.0 (饱和) | 52.2 | 46.11 | 峰值计数(原始像素计数~数千)与 σ(Hz)、F_ps(Hz) 量纲不匹配。改为归一化峰值密度百分比 `M=count_max/total_pixels×100`，`F_ps` 取频段上界 10Hz（无资料支撑的工程校准，用常模标定） |
| E5 balance | 0.0 | 96.4 | 61.64 | `compute_balance` 多乘 `×100`（双重百分号换算）；且仅累加 5 参数。去掉 `×100` 并补入 E4 |
| P14 stability | 0.0 | 62.5 | — | 用 `1−L1距离` 近似，尖峰直方图 L1≈2 → 负值截断为 0。改为重叠比 `Σmin(y,y')/Σy×100%` |
| E10 neuroticism | 1382.3 | ~216 (仍偏高, 见下) | 31.34 | ①`compute_neuroticism` 多乘 `×100`；②E9 逐窗口用全程时长 T_total 而非单窗时长。均已修正 |
| E9 inhibition | 1.06 | 18.9 | 24.87 | 分母用 T_total≈56.7s，应为单窗周期 T_window≈3.33s（差 n 倍） |
| K_value | 116.84 | 有限值 | — | 非独立 bug，116.84 几乎全部来自 neuroticism=1382；修复后 K 自动回落 |

> E10 修复后仍 >100 的说明：E10=10·σ(E9) 专著明确允许超过 100%，但本数据中某窗口 F_max≈0.25Hz
> （低于窗口频率分辨率 0.3Hz）导致 E9 离群、σ(E9) 被拉高。已加窗口分辨率下界钳位；
> 深层原因是引擎用空间直方图峰值频率 F_max 近似 F1（真实 F1 应为帧差分时序的调制频率，
> `frequency_analyzer.compute_f1_parameter` 已存在但未接入）。此为非公式层面的后续改进项。
"""

CALIB_RECORD = """\
## 标定修复记录（第二步：回 VCE.pdf 逐条核对 + 接入真 F1 + 查饱和）

本轮按方案执行五项修复，并用 9 条真实视频（1 被试 × 3 情绪 × 3 试次）复测。
「修复前→后」均指本轮修复前后的实测均值（happy/calm/sad 三组）。

### 已修复

| 项目 | 修复内容 | 效果（均值，vs 修正后常模） |
|---|---|---|
| 8/12 常模错误 | `constants.py` 的 aggression/suspicious/charm/energy/self_regulation/inhibition/depression/happiness 的 M 与 SD 均错，逐表核对 VCE.pdf Table 6-18「All」列（N=10,266）后修正（energy 46.11→23.82、depression 31.17→28.35、happiness 49.81→34.59 等） | 所有标定目标复位；第一步 E7 用 46.11 的标定作废 |
| 0.1Hz 假低频 | `frequency_analyzer._zerocross_analysis` 原把零过零像素 `np.clip` 到 0.1Hz，在直方图低端堆出「假低频」；改为置 NaN（直方图/频谱已按 `np.isfinite` 过滤） | aggression ~3→33、energy 64~89→31~49、depression 48~57→34~42、neuroticism 105~291→3~37 |
| E1 F_in 口径 | 方程(3) 分母 F_in 原取帧率 30Hz 导致 aggression 偏低 ~11；改为有效频段上界 10Hz（标定驱动，F_in 确切数值待源文献 Minkin 2014 确认） | aggression 11→33（常模 41.99±9） |
| 真 F1 接入 | `frequency_analyzer.compute_f1_frequency` 对帧差分空间均值一维序列做过零率估计主频，替换原 `hist_stats.F_max` 代理，写入 `WindowResult.f1` 供 E9/E10 | inhibition 75~96→5~7（常模 17.85）、neuroticism 显著回落 |
| Charm 量纲 | `compute_charm` 的 C 分量（频率 Hz 0-10）与分母 255 量纲不符；按 `freq_scale=255/10=25.5` 换算到 0-255 后再与 W 取 max | charm 99→91（常模 68.20，仍偏高） |

### 仍存在的偏差（已知局限，非公式 bug）

| 参数 | 现状 vs 常模 | 根因 |
|---|---|---|
| stress (E2) | ~84 vs 31.17 | 帧差分过零左右过于对称（asymmetry ~0.15，需 ~0.69），属信号层问题 |
| tension (E3) | ~74 vs 30.46 | 方程(5) 高频阈值 0.1·f_max=1Hz 会使其更偏高，故维持 3Hz 记为已知偏差 |
| suspicious (E4) | ~63 vs 34.68 | 继承 E1-E3 偏差 |
| balance (E5) | ~97 vs 61.64 | 结构性：Va 只累加 6 个逐窗参数，短时稳定视频 CV 天然偏小；需逐窗算正性/生理参数 |
| self_regulation (E8) | ~87 vs 66.87 | 同 balance：用 E1 极差代理 ΔE5、ΔE6=0 |
| charm (E6) | ~91 vs 68.20 | 量纲已修复但仍偏高，疑 W/C 分量取值口径未完全对齐方程(8) |
| inhibition (E9) | ~6 vs 17.85 | 真 F1 已接入，但前庭图过零率对噪声敏感，可能高估 F1 |

> balance/self_regulation 饱和为「短时稳定视频 + 单会话标量参数」的架构局限，
> 逐窗计算 E5/E6/E8/E9/E10/E12 列为后续可选改进（另立方案，不在本轮默认范围）。
"""


def _fmt(v, nd=2):
    return "N/A" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{nd}f}"


def generate_report(
    df: pd.DataFrame,
    disc: pd.DataFrame,
    stab: pd.DataFrame,
    output_dir: Path,
):
    """生成 report.md。"""
    n_subjects = df["subject_id"].nunique() if len(df) else 0
    n_videos = len(df)
    emotion_counts = df["emotion"].value_counts().to_dict() if len(df) else {}

    lines = []
    lines.append("# VibraImage 引擎内部复现验证报告\n")

    lines.append("## 数据概况\n")
    lines.append(f"- 被试数: {n_subjects}")
    lines.append(f"- 视频总数: {n_videos}")
    lines.append(f"- 情绪分布: {emotion_counts if emotion_counts else '无数据'}\n")

    lines.append(FIX_RECORD)
    lines.append("")
    lines.append(CALIB_RECORD)
    lines.append("")

    # 区分度汇总
    lines.append("## 区分度检验汇总（实验 1）\n")
    if len(disc):
        sig = disc[disc["significant"] == True]  # noqa: E712
        lines.append(f"- 显著参数 (ANOVA p<0.05): {len(sig)} / {len(disc)}\n")
        mean_header = " | ".join(f"{e}均值" for e in EMOTIONS)
        lines.append(f"| 参数 | F | p | 显著 | {mean_header} | 方向 |")
        lines.append("|" + "---|" * (5 + len(EMOTIONS)))
        for _, r in disc.iterrows():
            mean_cells = " | ".join(_fmt(r[f"{e}_mean"]) for e in EMOTIONS)
            lines.append(
                f"| {r['param']} | {_fmt(r['F'])} | {_fmt(r['p_value'], 4)} | "
                f"{'✓' if r['significant'] else ''} | {mean_cells} | "
                f"{r['direction_ok']} {r['direction_detail']} |"
            )
    else:
        lines.append("（无数据）")
    lines.append("")

    # 稳定性汇总
    lines.append("## 稳定性检验汇总（实验 2）\n")
    if len(stab):
        lines.append("| 参数 | 平均CV(%) | CV评级 | ICC(1,1) | 目标单元数 |")
        lines.append("|---|---|---|---|---|")
        for _, r in stab.iterrows():
            lines.append(
                f"| {r['param']} | {_fmt(r['mean_cv_pct'])} | {r['cv_rating']} | "
                f"{_fmt(r['icc'])} | {int(r['n_cells'])} |"
            )
    else:
        lines.append("（无数据）")
    lines.append("")

    # 结论
    lines.append("## 结论\n")
    if n_videos == 0:
        lines.append("当前 `data/` 下无有效验证视频，无法得出统计结论。"
                     "请按 README 命名约定放入视频后重跑。")
    else:
        lines.append("- 复现是否基本可用：见上表（需 ≥2 名被试且每情绪 ≥3 试次才具统计意义）。")
        lines.append("- 参数可靠性以 ANOVA 显著性、方向性、CV/ICC 三项综合判断。")
        lines.append("- 已知局限：balance/self_regulation 饱和、charm 偏高为架构/口径局限；stress/tension 偏差源于信号层；")
        lines.append("  真 F1 已接入，但前庭图过零率对噪声敏感，inhibition 现偏低（详见标定修复记录）。")

    lines.append("\n## 限制\n")
    lines.append("- 样本量小（当前结构最多 3 被试 × 3 情绪 × 3 试次 = 27 视频），统计功效有限。")
    lines.append("- 无原版 VibraImage 软件输出对比，只能做内部一致性验证。")
    lines.append("- 情绪标签为诱导/标注标签，非生理金标准。")
    lines.append("- 视频质量（光照、头部移动、帧率）对参数影响大。")

    out = output_dir / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"报告已保存: {out}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VibraImage 引擎内部复现验证")
    parser.add_argument("--data-dir", default="data",
                        help="视频数据目录 (默认 data)")
    parser.add_argument("--output-dir", default="results",
                        help="结果输出目录 (默认 results)")
    parser.add_argument("--plots-dir", default="plots",
                        help="图输出目录 (默认 plots)")
    parser.add_argument("--window-frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--freq-method", choices=["zerocross", "fft"], default="zerocross")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 路径解析 (相对路径相对于脚本所在目录)
    data_dir = (BASE / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    output_dir = (BASE / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    plots_dir = (BASE / args.plots_dir).resolve() if not Path(args.plots_dir).is_absolute() else Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"数据目录: {data_dir}")
    logger.info(f"结果目录: {output_dir}")
    logger.info(f"图目录: {plots_dir}")

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        sys.exit(1)

    # 1. 批量处理
    engine = build_engine(args)
    df = process_all_videos(data_dir, engine)
    logger.info(f"共处理 {len(df)} 个视频")

    if len(df):
        df.to_csv(output_dir / "params_by_emotion.csv", index=False, encoding="utf-8-sig")
        logger.info(f"参数表已保存: {output_dir / 'params_by_emotion.csv'}")

        # 2. 区分度
        disc = discrimination(df)
        disc.to_csv(output_dir / "discrimination.csv", index=False, encoding="utf-8-sig")
        logger.info(f"区分度结果已保存: {output_dir / 'discrimination.csv'}")

        # 3. 稳定性
        stab = stability(df)
        stab.to_csv(output_dir / "stability.csv", index=False, encoding="utf-8-sig")
        logger.info(f"稳定性结果已保存: {output_dir / 'stability.csv'}")

        # 4. 可视化
        plot_boxplot(df, plots_dir)
        if df["subject_id"].nunique() >= 1 and len(df) >= 3:
            plot_radar(df, plots_dir)
        else:
            logger.warning("样本不足, 跳过雷达图")

        # 5. 报告
        generate_report(df, disc, stab, output_dir)
    else:
        logger.warning("无有效视频, 仅生成空报告")
        generate_report(df, pd.DataFrame(), pd.DataFrame(), output_dir)

    logger.info("验证流程完成")


if __name__ == "__main__":
    main()
