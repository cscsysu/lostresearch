#!/bin/bash
# InfoDyn 完整实验 - 后台运行
# 用法: bash run_background.sh
# 日志: outputs/experiment.log
# 查看进度: tail -f outputs/experiment.log

set -e

cd "$(dirname "$0")"

LOG_FILE="outputs/experiment.log"

echo "Starting InfoDyn full experiment in background..."
echo "Log: $LOG_FILE"
echo "View progress: tail -f $LOG_FILE"
echo ""

# 后台运行, 输出到 log 文件
nohup python run_full.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "PID: $PID"
echo "Started at: $(date)"
echo ""
echo "Commands:"
echo "  View progress:  tail -f $LOG_FILE"
echo "  Check process: ps -p $PID"
echo "  Stop:           kill $PID"
echo ""

# 把 PID 写到文件方便后续管理
echo "$PID" > outputs/experiment.pid
