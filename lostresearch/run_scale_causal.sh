#!/bin/bash
# 规模扩展 + 因果补强实验
# 1. 跨模型轨迹 (Qwen3-4B, Qwen3-14B, Qwen2.5-7B)
# 2. Effect-matched control (真正的 Δz 匹配) 各模型
# 3. MLP layer-35 patching 各模型

set -e
cd "$(dirname "$0")"

LOG="outputs/scale_causal.log"
echo "Starting scale + causal experiments..." | tee "$LOG"
echo "Log: $LOG"

nohup bash -c '
echo "=== Scale: Qwen3-4B trajectory ==="
python run_cross_model.py --model qwen4b

echo "=== Scale: Qwen3-14B trajectory ==="
python run_cross_model.py --model qwen14b

echo "=== Scale: Qwen2.5-7B trajectory ==="
python run_cross_model.py --model qwen25_7b

echo "=== Effect-matched v2: Qwen3-8B ==="
python run_effect_matched_v2.py --model qwen --n 20

echo "=== Effect-matched v2: Qwen3-4B ==="
python run_effect_matched_v2.py --model qwen4b --n 20

echo "=== Effect-matched v2: Qwen3-14B ==="
python run_effect_matched_v2.py --model qwen14b --n 20

echo "=== MLP patch: Qwen3-8B ==="
python run_mlp_patch.py --model qwen --n 40

echo "=== ALL DONE ==="
date
' >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > outputs/scale_causal.pid
echo "PID: $PID"
echo "View: tail -f $LOG"
