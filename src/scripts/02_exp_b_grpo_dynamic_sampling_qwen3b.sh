#!/usr/bin/env bash
# Experiment B: GRPO + Dynamic Sampling. Drop all-correct/all-wrong rollout groups before update.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export EXP_ID=${EXP_ID:-B}
export PROJECT_NAME=${PROJECT_NAME:-qwen3b_grpo_fair_gsm8k}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-B_grpo_dynamic_sampling_qwen25_3b_gsm8k}
export FILTER_GROUPS_ENABLE=True
export FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC:-acc}
export FILTER_GROUPS_MAX_NUM_GEN_BATCHES=${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-1}
exec bash "${SCRIPT_DIR}/00_common_qwen3b_grpo.sh" "$@"
