#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${SCRIPT_DIR}"

export PYTHONPATH="${SIGNAL_FORGE_SRC}:${VENDOR_PYTHON:+${VENDOR_PYTHON}:}${REWARDSCOPE_SRC}:${VERL_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_model_paths.sh"

TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH:-$(choose_qwen25_default_path)}}
MATH_TEST_DIR=${MATH_TEST_DIR:-${SIGNAL_FORGE_SRC}/RewardScope/raw/competition_math/test}

EXTRA_ARGS=()
if [ "${CHECK_VERL_LOADER:-0}" = "1" ]; then
    EXTRA_ARGS+=(--check-verl-loader)
fi

python -m signal_forge.data.prepare_a0_data \
    --output-dir "${A0_DATA_DIR:-${DATA_ROOT}/signal_forge_a0}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --math-test-dir "${MATH_TEST_DIR}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH:-512}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
