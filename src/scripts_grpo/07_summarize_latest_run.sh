#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
A0_SCRIPT_DIR=$(cd -- "${SCRIPT_DIR}/../scripts_a0" && pwd)
# shellcheck source=/dev/null
source "${A0_SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${A0_SCRIPT_DIR}"

if [ -x "${VENV_DIR}/bin/python" ]; then
    export PATH="${VENV_DIR}/bin:${PATH}"
fi
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

RUN_DIR=${RUN_DIR:-${OUTPUT_ROOT}/grpo_baseline_gsm8k/qwen25_1p5b_gsm8k_grpo}

python "${SIGNAL_FORGE_SRC}/scripts_grpo/summarize_run.py" "${RUN_DIR}"

