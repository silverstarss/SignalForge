#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIGNAL_FORGE_SRC=${SIGNAL_FORGE_SRC:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
ROOT_DIR=${ROOT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_model_paths.sh"

export A0_DATA_DIR=${A0_DATA_DIR:-${ROOT_DIR}/data/signal_forge_a0_0p5b_regression}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH:-$(choose_qwen25_0p5b_path)}}

ARGS=(
    --train-gsm8k "${TRAIN_GSM8K:-18}"
    --train-math "${TRAIN_MATH:-12}"
    --val-gsm8k "${VAL_GSM8K:-8}"
    --val-math "${VAL_MATH:-8}"
)

exec bash "${SCRIPT_DIR}/00_prepare_a0_data.sh" "${ARGS[@]}" "$@"
