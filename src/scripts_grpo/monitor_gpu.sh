#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-${OUTPUT_ROOT:-./outputs}/gpu_monitor.csv}
INTERVAL=${INTERVAL:-5}

mkdir -p "$(dirname -- "${OUT}")"
echo "timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu" > "${OUT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "# nvidia-smi unavailable" >> "${OUT}"
    exit 0
fi

while true; do
    if ! nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
        --format=csv,noheader,nounits >> "${OUT}" 2>/dev/null; then
        echo "# nvidia-smi failed" >> "${OUT}"
        exit 0
    fi
    sleep "${INTERVAL}"
done
