#!/usr/bin/env bash
# Stage 1: Qwen2.5-1.5B A800 preflight only. This script must not start training.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
A0_SCRIPT_DIR=$(cd -- "${SCRIPT_DIR}/../scripts_a0" && pwd)

# shellcheck source=/dev/null
source "${A0_SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${A0_SCRIPT_DIR}"

# shellcheck source=/dev/null
source "${A0_SCRIPT_DIR}/_model_paths.sh"

export QWEN25_DEFAULT_SIZE=${QWEN25_DEFAULT_SIZE:-1.5B}
export PROJECT_NAME=${PROJECT_NAME:-signal_forge_a_1p5b}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-A_1p5b_stage1_preflight}
export MODEL_PATH=${MODEL_PATH:-$(choose_qwen25_1p5b_path)}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/processed/signal_forge_v1/train.parquet}
export TEST_FILE=${TEST_FILE:-${DATA_ROOT}/processed/signal_forge_v1/validation_id.parquet}

# Keep frozen scientific settings from the Stage 0 protocol.
export ROLLOUT_N=${ROLLOUT_N:-8}
export VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-768}

# Conservative engineering defaults for the 1.5B A800 preflight/smoke path.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-5}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-3}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-5}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-1280}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}

export PREFLIGHT_ONLY=1
export PREFLIGHT_MODE=${PREFLIGHT_MODE:-deep}
export PREFLIGHT_FORMAL=${PREFLIGHT_FORMAL:-1}
export PREFLIGHT_LAUNCH_SCRIPT=${PREFLIGHT_LAUNCH_SCRIPT:-${SCRIPT_DIR}/01_preflight_1p5b_a800.sh}

exec bash "${A0_SCRIPT_DIR}/03_run_a0_grpo_smoke.sh" --preflight-only --preflight-deep "$@"
