#!/bin/bash
# InfoDyn Pilot 实验 - 一键运行脚本
# 用法: bash run_pilot.sh

set -e

echo "============================================"
echo "InfoDyn Pilot Experiment"
echo "============================================"
echo ""

# 检查模型路径
MODEL_PATH="/data2/css2025/models/Qwen/Qwen3-8B"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi
echo "✓ 模型路径存在"

# 检查 GPU
echo ""
echo "GPU 状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo ""

# 服务器 GPU 映射 (按 nvidia-smi 的 index):
#   0 = RTX 3090 (24GB, 被占 16GB)
#   1 = RTX 3090 (24GB, 空闲)
#   2 = RTX 3090 (24GB, 空闲)
#   3 = RTX 3090 (24GB, 空闲)
#   4 = A40      (46GB, 空闲)  ← 选这个
# 用 index 4 的 A40, 程序内它会被映射成 cuda:0
export CUDA_VISIBLE_DEVICES=4
echo "✓ 使用 GPU index 4 (A40 46GB), 程序内映射为 cuda:0"
echo ""

# 运行实验
echo "开始运行..."
python run_pilot.py

echo ""
echo "============================================"
echo "实验完成!"
echo "结果保存在: outputs/"
echo "============================================"
