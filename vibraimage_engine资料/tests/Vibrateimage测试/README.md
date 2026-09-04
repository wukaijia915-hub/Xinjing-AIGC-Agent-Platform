# Vibrateimage测试 — VibraImage 引擎内部复现验证

本目录用于验证 VibraImage 前庭振动情绪识别引擎复现是否基本可用。

## 目录结构

```
Vibrateimage测试/
├── data/                 # 验证视频数据（按被试分目录，需自行填充）
│   ├── s01/              # 被试1：3 情绪 × 3 试次 = 9 个视频
│   ├── s02/              # 被试2
│   └── s03/              # 被试3
├── run_validation.py     # 验证主脚本（批量处理 + 统计检验 + 可视化 + 报告）
├── results/              # 脚本生成的结果（CSV + report.md）
└── plots/                # 脚本生成的图（箱线图 + 雷达图）
```

## 数据命名约定

每个被试目录下放置 9 个视频，命名格式为 `<emotion>_<trial>.mp4`：

- `emotion` ∈ `happy` / `calm` / `sad`
- `trial` ∈ `01` / `02` / `03`

示例：

```
data/s01/happy_01.mp4   happy_02.mp4   happy_03.mp4
data/s01/calm_01.mp4    calm_02.mp4    calm_03.mp4
data/s01/sad_01.mp4     sad_02.mp4     sad_03.mp4
```

脚本会扫描 `data/` 下所有 `s*/<emotion>_<trial>.mp4`，缺失的视频自动跳过并警告，
支持不完整数据（部分被试 / 部分情绪 / 部分试次）。

## 运行方式

在引擎根目录 `vibraimage_engine/` 下执行：

```bash
# 用默认路径（相对于本脚本所在目录）
python "Vibrateimage测试/run_validation.py"

# 或指定路径
python "Vibrateimage测试/run_validation.py" \
    --data-dir "Vibrateimage测试/data" \
    --output-dir "Vibrateimage测试/results" \
    --plots-dir "Vibrateimage测试/plots"
```

可选参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data-dir` | `data` | 视频数据目录 |
| `--output-dir` | `results` | 结果输出目录 |
| `--plots-dir` | `plots` | 图输出目录 |
| `--window-frames` | `100` | 每窗口帧数 |
| `--stride` | `50` | 窗口步长 |
| `--freq-method` | `zerocross` | 频率分析方法（zerocross / fft） |

## 输出文件

| 文件 | 说明 |
|---|---|
| `results/params_by_emotion.csv` | 每个视频的 E1-E12 + stability + K 值 |
| `results/discrimination.csv` | 区分度检验（ANOVA + 事后 t 检验 + 方向性） |
| `results/stability.csv` | 稳定性检验（CV + ICC(1,1)） |
| `results/report.md` | 验证报告 |
| `plots/emotion_boxplot.png` | 各参数在不同情绪下的箱线图 |
| `plots/stability_radar.png` | 每个被试 3 情绪在 E1-E12+K 空间的轮廓对比 |
