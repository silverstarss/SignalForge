#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${SCRIPT_DIR}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_model_paths.sh"

export PROJECT_NAME=${PROJECT_NAME:-signal_forge_a0_0p5b}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-A0_0p5b_regression_40step}
export MODEL_PATH=${MODEL_PATH:-$(choose_qwen25_0p5b_path)}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/signal_forge_a0_0p5b_regression/train.parquet}
export TEST_FILE=${TEST_FILE:-${DATA_ROOT}/signal_forge_a0_0p5b_regression/val.parquet}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-5}
export ROLLOUT_N=${ROLLOUT_N:-8}
export VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-768}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-40}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-7}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-5}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-10}
export ROLLOUT_DUMP_INTERVAL=${ROLLOUT_DUMP_INTERVAL:-10}
export ROLLOUT_DUMP_MAX_RECORDS=${ROLLOUT_DUMP_MAX_RECORDS:-40}
export VALIDATION_DUMP_MAX_RECORDS=${VALIDATION_DUMP_MAX_RECORDS:-128}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
export MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-1}
export LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-16}

exec bash "${SCRIPT_DIR}/03_run_a0_grpo_smoke.sh" "$@"
