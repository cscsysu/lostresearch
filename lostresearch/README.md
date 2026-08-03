# InfoDyn - Lost Research

Tracking how the "correct answer signal" evolves across Transformer layers.
Pilot experiment on Qwen3-8B with TriviaQA.

## Quick Start

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行 pilot (50 题, 约 5-10 分钟)
bash run_pilot.sh
```

## File Structure

```
lostresearch/
├── config.py                  # 配置: 模型路径、GPU、数据集等
├── data_loader.py             # 加载 TriviaQA, 构造 prompt, tokenize 答案
├── trajectory_collector.py    # Hook 采集每层 hidden state + CIS 计算
├── visualize.py               # 画轨迹图, 正确 vs 错误对比
├── run_pilot.py               # 主实验脚本
├── run_pilot.sh               # 一键运行
└── outputs/
    ├── data/                  # 原始轨迹数据 (JSON)
    └── figures/               # 可视化图
```

## What This Pilot Tests

**核心问题**: 模型答错时, 中间层是否曾经出现过正确答案的信号?

**测量方法**:
1. 对每层 hidden state 用 logit lens (final norm + LM head) 解码
2. 计算正确答案 token 的 log probability 和 rank
3. 画出从第 0 层到第 L 层的信号轨迹
4. 比较正确样本 vs 错误样本的轨迹

**预期发现**:
- 正确样本: 信号从弱到强, 最终层 log prob 高
- 错误样本: 中间层信号曾达到峰值, 但最终层衰减
- 这就是 "知道但没说" (Lost in Transmission) 现象

## Key Config

Edit `config.py` to change:
- `NUM_SAMPLES`: 样本数量 (pilot=50, 正式=1000+)
- `DEVICE`: GPU 设备
- `ENABLE_THINKING`: Qwen3 thinking mode (默认关闭)
