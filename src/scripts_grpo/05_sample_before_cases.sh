#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace}
VENV_DIR=${VENV_DIR:-${ROOT_DIR}/.venv-vllm}

if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
DATA_FILE=${DATA_FILE:-${ROOT_DIR}/data/gsm8k/test.parquet}
OUT=${OUT:-${ROOT_DIR}/outputs/cases/qwen25_1p5b_before.jsonl}
LIMIT=${LIMIT:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

python "${ROOT_DIR}/scripts_grpo/sample_gsm8k_cases.py" \
    --model "${MODEL_PATH}" \
    --data "${DATA_FILE}" \
    --out "${OUT}" \
    --limit "${LIMIT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}"

