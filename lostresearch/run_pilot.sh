#!/bin/bash
# InfoDyn Full Experiment - 一键运行
# 用法: bash run_pilot.sh

set -e

echo "============================================"
echo "InfoDyn Full Experiment (Qwen3-8B)"
echo "============================================"
echo ""

# 检查模型路径
MODEL_PATH="/data2/css2025/models/Qwen/Qwen3-8B"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi
echo "✓ 模型路径存在"

# GPU 状态
echo ""
echo "GPU 状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo ""

# 服务器 GPU 映射 (CUDA ordinal):
#   0 = A40 (46GB)  ← 选这个
#   1-4 = RTX 3090 (24GB)
export CUDA_VISIBLE_DEVICES=0
echo "✓ 使用 GPU 0 (A40), 程序内映射为 cuda:0"
echo ""

# 运行完整实验
echo "开始运行..."
python run_full.py

echo ""
echo "============================================"
echo "实验完成!"
echo "结果保存在: outputs/"
echo "============================================"
