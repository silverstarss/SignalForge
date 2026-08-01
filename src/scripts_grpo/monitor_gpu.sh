#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-/workspace/outputs/gpu_monitor.csv}
INTERVAL=${INTERVAL:-5}

mkdir -p "$(dirname "${OUT}")"
echo "timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu" > "${OUT}"

while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
        --format=csv,noheader,nounits >> "${OUT}"
    sleep "${INTERVAL}"
done

