#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${SCRIPT_DIR}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_model_paths.sh"

export PROJECT_NAME=${PROJECT_NAME:-signal_forge_a0_0p5b}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-A0_0p5b_regression_50step}
export MODEL_PATH=${MODEL_PATH:-$(choose_qwen25_0p5b_path)}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/signal_forge_a0_0p5b_regression/train.parquet}
export TEST_FILE=${TEST_FILE:-${DATA_ROOT}/signal_forge_a0_0p5b_regression/val.parquet}

exec bash "${SCRIPT_DIR}/04_reload_a0_checkpoint.sh" "$@"
