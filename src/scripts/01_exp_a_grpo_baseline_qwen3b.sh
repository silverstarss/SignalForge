#!/usr/bin/env bash
# Experiment A: GRPO baseline. Random sampling, standard GRPO.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export EXP_ID=${EXP_ID:-A}
export PROJECT_NAME=${PROJECT_NAME:-qwen3b_grpo_fair_gsm8k}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-A_grpo_baseline_qwen25_3b_gsm8k}
export FILTER_GROUPS_ENABLE=False
exec bash "${SCRIPT_DIR}/00_common_qwen3b_grpo.sh" "$@"
