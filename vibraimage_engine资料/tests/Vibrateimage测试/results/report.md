# VibraImage 引擎内部复现验证报告

## 数据概况

- 被试数: 1
- 视频总数: 9
- 情绪分布: {'calm': 3, 'happy': 3, 'sad': 3}

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


## 区分度检验汇总（实验 1）

- 显著参数 (ANOVA p<0.05): 1 / 14

| 参数 | F | p | 显著 | happy均值 | calm均值 | sad均值 | 方向 |
|---|---|---|---|---|---|---|---|
| aggression | 0.05 | 0.9538 |  | 34.07 | 30.29 | 33.35 | n/a  |
| stress | 5.09 | 0.0510 |  | 86.80 | 87.94 | 76.48 | fail happy:↓✗  calm:↓✗ |
| tension | 0.37 | 0.7035 |  | 79.02 | 73.40 | 70.95 | n/a  |
| suspect | 0.35 | 0.7163 |  | 66.63 | 63.88 | 60.26 | n/a  |
| balance | 2.63 | 0.1516 |  | 97.69 | 96.57 | 98.74 | fail calm:↑✗ |
| charm | 0.17 | 0.8510 |  | 91.63 | 92.11 | 91.34 | n/a  |
| energy | 0.34 | 0.7222 |  | 31.14 | 42.23 | 49.49 | fail sad:↓✗ |
| self_regulation | 49.24 | 0.0002 | ✓ | 83.45 | 78.66 | 98.55 | fail calm:↑✗ |
| inhibition | 0.54 | 0.6103 |  | 6.60 | 6.46 | 4.90 | n/a  |
| neuroticism | 0.86 | 0.4680 |  | 37.41 | 26.07 | 2.81 | n/a  |
| depression | 1.50 | 0.2962 |  | 39.78 | 42.10 | 34.18 | fail happy:↓✗  sad:↑✗ |
| happiness | 1.64 | 0.2710 |  | 40.69 | 39.24 | 53.32 | fail happy:↑✗  sad:↓✗ |
| stability | 1.06 | 0.4033 |  | 54.71 | 64.01 | 51.49 | n/a  |
| K_value | 4.78 | 0.0573 |  | 26.83 | 25.50 | 23.53 | n/a  |

## 稳定性检验汇总（实验 2）

| 参数 | 平均CV(%) | CV评级 | ICC(1,1) | 目标单元数 |
|---|---|---|---|---|
| aggression | 47.23 | 不稳定 | -0.47 | 3 |
| stress | 5.02 | 优秀 | 0.58 | 3 |
| tension | 15.28 | 可接受 | -0.26 | 3 |
| suspect | 14.32 | 优秀 | -0.27 | 3 |
| balance | 1.17 | 优秀 | 0.35 | 3 |
| charm | 1.70 | 优秀 | -0.39 | 3 |
| energy | 50.95 | 不稳定 | -0.28 | 3 |
| self_regulation | 2.33 | 优秀 | 0.94 | 3 |
| inhibition | 30.85 | 不稳定 | -0.18 | 3 |
| neuroticism | 113.67 | 不稳定 | -0.05 | 3 |
| depression | 14.02 | 优秀 | 0.14 | 3 |
| happiness | 14.26 | 优秀 | 0.17 | 3 |
| stability | 15.18 | 可接受 | 0.02 | 3 |
| K_value | 4.53 | 优秀 | 0.56 | 3 |

## 结论

- 复现是否基本可用：见上表（需 ≥2 名被试且每情绪 ≥3 试次才具统计意义）。
- 参数可靠性以 ANOVA 显著性、方向性、CV/ICC 三项综合判断。
- 已知局限：balance/self_regulation 饱和、charm 偏高为架构/口径局限；stress/tension 偏差源于信号层；
  真 F1 已接入，但前庭图过零率对噪声敏感，inhibition 现偏低（详见标定修复记录）。

## 限制

- 样本量小（当前结构最多 3 被试 × 3 情绪 × 3 试次 = 27 视频），统计功效有限。
- 无原版 VibraImage 软件输出对比，只能做内部一致性验证。
- 情绪标签为诱导/标注标签，非生理金标准。
- 视频质量（光照、头部移动、帧率）对参数影响大。
