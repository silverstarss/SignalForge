#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${SCRIPT_DIR}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_model_paths.sh"

export A0_DATA_DIR=${A0_DATA_DIR:-${DATA_ROOT}/signal_forge_a0_0p5b_regression}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH:-$(choose_qwen25_0p5b_path)}}

ARGS=(
    --train-gsm8k "${TRAIN_GSM8K:-18}"
    --train-math "${TRAIN_MATH:-12}"
    --val-gsm8k "${VAL_GSM8K:-8}"
    --val-math "${VAL_MATH:-8}"
)

exec bash "${SCRIPT_DIR}/00_prepare_a0_data.sh" "${ARGS[@]}" "$@"
