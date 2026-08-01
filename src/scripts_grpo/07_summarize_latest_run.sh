#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace}
VENV_DIR=${VENV_DIR:-${ROOT_DIR}/.venv-vllm}

if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

RUN_DIR=${RUN_DIR:-${ROOT_DIR}/outputs/grpo_baseline_gsm8k/qwen25_1p5b_gsm8k_grpo}

python "${ROOT_DIR}/scripts_grpo/summarize_run.py" "${RUN_DIR}"

