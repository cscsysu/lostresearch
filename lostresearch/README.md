# InfoDyn - Lost Research

Tracking how the "correct answer signal" evolves across Transformer layers.
Full experiment on Qwen3-8B with TriviaQA + HotpotQA + GSM8K.

## Quick Start

```bash
bash run_pilot.sh
```

## What It Does (7 stages)

1. **Load model**: Qwen3-8B on A40 (non-thinking mode)
2. **Load data**: TriviaQA 100 + HotpotQA 50 + GSM8K 50 = 200 samples
3. **Collect trajectories**: per-layer hidden states + logit lens + CIS
4. **Sanity check**: final-layer argmax == generate() first token
5. **Save data**: full trajectory JSON
6. **Analysis**: signal loss statistics + pattern classification + visualization
7. **Prediction task**: 5 baselines (Random/Persistence/Linear/MLP) predict signal decay

## Key Files

```
config.py              # 配置
data_loader.py         # 多数据集加载 + token 对齐
trajectory_collector.py # Hook + CIS + generated 对照
analyze.py             # 信号丢失判据 + 模式分类 + 可视化
prediction.py          # 信息动态预测任务
negatives.py           # 负对照 (随机标签/答案置换/层序打乱)
run_full.py            # 主实验脚本
run_pilot.sh           # 一键运行
```

## Output

```
outputs/
├── data/
│   ├── full_results_Qwen3-8B.json        # 完整轨迹数据
│   ├── prediction_results_Qwen3-8B.json   # 预测任务结果
│   ├── negative_controls_Qwen3-8B.json    # 负对照结果
│   └── sanity_Qwen3-8B.json               # sanity check
└── figures/
    ├── trajectory_comparison_Qwen3-8B.png  # 正确 vs 错误对比
    ├── pattern_distribution_Qwen3-8B.png  # 模式分布
    └── single_*.png                       # 单样本轨迹
```

## Key Metrics

- **CIS** = log P(correct) - log P(generated) per layer
- **Signal Lost**: mid rank ≤ 5 but final rank > 10
- **Pattern**: Absent / Early-Decay / Late-Emergent / Gradual-Buildup / Fluctuating
- **"Know but didn't say"**: mid rank ≤ 5 + final CIS < 0
