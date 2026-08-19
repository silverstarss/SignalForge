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

export PYTHONPATH="${SIGNAL_FORGE_SRC}:${VENDOR_PYTHON:+${VENDOR_PYTHON}:}${REWARDSCOPE_SRC}:${VERL_DIR}:${PYTHONPATH:-}"

DATASET_OUTPUT_DIR=${DATASET_OUTPUT_DIR:-${DATA_ROOT}/gsm8k_boxed}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
TEST_MAX_SAMPLES=${TEST_MAX_SAMPLES:--1}

python -m signal_forge.data.prepare_gsm8k_runtime \
    --output-dir "${DATASET_OUTPUT_DIR}" \
    --train-max-samples "${TRAIN_MAX_SAMPLES}" \
    --test-max-samples "${TEST_MAX_SAMPLES}"
