#!/usr/bin/env bash
# Run on the WSL host from /home/tutu/grpo. Packages verl plus the new Qwen 3B experiment scripts.
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$HOME/grpo}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-${ROOT_DIR}/grpo_qwen3b_ab_autodl_${STAMP}.tar.gz}

cd "${ROOT_DIR}"

tar \
    --exclude='./.venv' \
    --exclude='./.venv-vllm' \
    --exclude='./data' \
    --exclude='./outputs' \
    --exclude='./verl/outputs' \
    --exclude='./verl/checkpoints' \
    --exclude='./verl/.git' \
    --exclude='./verl/.pytest_cache' \
    --exclude='./verl/**/__pycache__' \
    -czf "${OUT}" \
    scripts \
    scripts_grpo \
    docs_grpo \
    verl

ls -lh "${OUT}"
echo "Created ${OUT}"
