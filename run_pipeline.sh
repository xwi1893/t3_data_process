#!/bin/bash
# 场景清洗与提取 - 后台运行脚本
# 用法:
#   nohup bash run_pipeline.sh &
#   nohup bash run_pipeline.sh --phase 1 &
#   nohup bash run_pipeline.sh --phase 2 &

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/output_data/pipeline_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$DIR/output_data"
cd "$DIR"
python main.py "$@" > "$LOG" 2>&1

echo "[$(date)] Done, exit=$?" >> "$LOG"
echo "日志: $LOG"
