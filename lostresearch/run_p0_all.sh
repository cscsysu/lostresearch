#!/bin/bash
# P0 实验一键运行 (Qwen3-8B)
# 用法: bash run_p0_all.sh
# 日志: outputs/p0.log

set -e
cd "$(dirname "$0")"

LOG="outputs/p0.log"

echo "Starting P0 experiments..." | tee "$LOG"
echo "Log: $LOG"
echo ""

nohup bash -c '
  echo "=== P0-1+2: Top-k Sensitivity (no model needed) ==="
  python run_p0_sensitivity.py
  echo ""
  echo "=== P0-3+5: Strong Baselines + Cross-task Transfer (no model) ==="
  python run_p0_baselines.py
  echo ""
  echo "=== P0-1: Multi-token Sequence-level CIS (needs model) ==="
  python run_p0_multitoken.py
  echo ""
  echo "=== P0-4: Tuned Lens Comparison (needs model, slow) ==="
  python run_p0_tuned_lens.py
  echo ""
  echo "=== P0-5+6: Strict Patching Controls + Mediation (needs model) ==="
  python run_p0_strict_patching.py
  echo ""
  echo "=== ALL P0 DONE ==="
  date
' >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > outputs/p0.pid
echo "PID: $PID"
echo "Started at: $(date)"
echo ""
echo "View progress:  tail -f $LOG"
echo "Check process: ps -p $PID"
