#!/bin/bash
# 后台运行 Step 1 (1000 题前向采集) + Step 2 (因果干预)
# 用法: bash run_all_background.sh
# 日志: outputs/experiment.log

set -e
cd "$(dirname "$0")"

LOG="outputs/experiment.log"

echo "Starting full experiment (Step 1 + Step 2) in background..."
echo "Log: $LOG"
echo ""

nohup bash -c '
  echo "=== Step 1: Trajectory Collection ==="
  python run_step1_collect.py
  echo ""
  echo "=== Step 2: Causal Intervention ==="
  python run_step2_intervention.py
  echo ""
  echo "=== ALL DONE ==="
  date
' > "$LOG" 2>&1 &

PID=$!
echo "$PID" > outputs/experiment.pid
echo "PID: $PID"
echo "Started at: $(date)"
echo ""
echo "View progress:  tail -f $LOG"
echo "Check process: ps -p $PID"
echo "Stop:           kill $PID"
