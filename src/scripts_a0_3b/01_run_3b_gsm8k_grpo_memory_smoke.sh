#!/usr/bin/env bash
# Qwen2.5-3B vanilla GRPO memory smoke on boxed GSM8K.
# This validates memory/runtime only; it is not formal A/B evidence.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
A0_SCRIPT_DIR=$(cd -- "${SCRIPT_DIR}/../scripts_a0" && pwd)
SCRIPT_1P5B_DIR=$(cd -- "${SCRIPT_DIR}/../scripts_a0_1p5b" && pwd)

# shellcheck source=/dev/null
source "${A0_SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${A0_SCRIPT_DIR}"

# shellcheck source=/dev/null
source "${A0_SCRIPT_DIR}/_model_paths.sh"

export PROJECT_NAME=${PROJECT_NAME:-signal_forge_memory_smoke_3b}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25_3b_gsm8k_vanilla_grpo_memory_smoke}
export MODEL_PATH=${MODEL_PATH:-$(choose_qwen25_3b_path)}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/gsm8k_boxed/train.parquet}
export TEST_FILE=${TEST_FILE:-${DATA_ROOT}/gsm8k_boxed/test.parquet}

PREFLIGHT_ONLY_REQUESTED=0
for arg in "$@"; do
    if [ "${arg}" = "--preflight-only" ]; then
        PREFLIGHT_ONLY_REQUESTED=1
    fi
done

PREFETCH_MODEL=${PREFETCH_MODEL:-1}
if [ "${PREFETCH_MODEL}" = "1" ] && [ "${PREFLIGHT_ONLY_REQUESTED}" != "1" ] && [[ "${MODEL_PATH}" == Qwen/* ]]; then
    QWEN25_3B_LOCAL_DIR=${QWEN25_3B_LOCAL_DIR:-${MODEL_ROOT}/Qwen/Qwen2.5-3B-Instruct} \
        "${SCRIPT_DIR}/00_prefetch_qwen25_3b.sh"
    export MODEL_PATH=${QWEN25_3B_LOCAL_DIR:-${MODEL_ROOT}/Qwen/Qwen2.5-3B-Instruct}
fi

if { [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${TEST_FILE}" ]; } && [ "${SKIP_DATA_PREPARE:-0}" != "1" ]; then
    DATASET_OUTPUT_DIR=$(dirname -- "${TRAIN_FILE}") "${SCRIPT_1P5B_DIR}/00_prepare_1p5b_gsm8k_runtime_data.sh"
fi

export QWEN25_DEFAULT_SIZE=3B
export ROLLOUT_N=${ROLLOUT_N:-8}
export VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-768}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
export GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-4096}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-1280}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-4}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
export MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-1}
export PREFLIGHT_LAUNCH_SCRIPT=${PREFLIGHT_LAUNCH_SCRIPT:-${SCRIPT_DIR}/01_run_3b_gsm8k_grpo_memory_smoke.sh}
export PREFLIGHT_MODE=${PREFLIGHT_MODE:-fast}
export ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-1}
export FILTER_GROUPS_ENABLE=False
export TARGET_RESPONSE_TOKENS=${TARGET_RESPONSE_TOKENS:-0}

exec "${A0_SCRIPT_DIR}/03_run_a0_grpo_smoke.sh" "$@"
