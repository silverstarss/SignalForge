#!/usr/bin/env bash
# RTX6000D migration smoke: Qwen2.5-1.5B-Instruct on boxed GSM8K only.
# This is an environment validation path, not formal Experiment A/B evidence.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
A0_SCRIPT_DIR=$(cd -- "${SCRIPT_DIR}/../scripts_a0" && pwd)

# shellcheck source=/dev/null
source "${A0_SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${A0_SCRIPT_DIR}"

# shellcheck source=/dev/null
source "${A0_SCRIPT_DIR}/_model_paths.sh"

export PROJECT_NAME=${PROJECT_NAME:-signal_forge_migration_rtx6000d}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25_1p5b_gsm8k_migration_smoke}
export MODEL_PATH=${MODEL_PATH:-$(choose_qwen25_1p5b_path)}
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
    QWEN25_1P5B_LOCAL_DIR=${QWEN25_1P5B_LOCAL_DIR:-${MODEL_ROOT}/Qwen/Qwen2.5-1.5B-Instruct} \
        "${SCRIPT_DIR}/00_prefetch_qwen25_1p5b.sh"
    export MODEL_PATH=${QWEN25_1P5B_LOCAL_DIR:-${MODEL_ROOT}/Qwen/Qwen2.5-1.5B-Instruct}
fi

if { [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${TEST_FILE}" ]; } && [ "${SKIP_DATA_PREPARE:-0}" != "1" ]; then
    DATASET_OUTPUT_DIR=$(dirname -- "${TRAIN_FILE}") "${SCRIPT_DIR}/00_prepare_1p5b_gsm8k_runtime_data.sh"
fi

export QWEN25_DEFAULT_SIZE=1.5B
export ROLLOUT_N=${ROLLOUT_N:-8}
export VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-768}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
export GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export TEST_FREQ=${TEST_FREQ:-1}
export SAVE_FREQ=${SAVE_FREQ:-1}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
export MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-2}
export PREFLIGHT_LAUNCH_SCRIPT=${PREFLIGHT_LAUNCH_SCRIPT:-${SCRIPT_DIR}/08_run_1p5b_gsm8k_migration_smoke.sh}
export PREFLIGHT_MODE=${PREFLIGHT_MODE:-fast}
export ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-1}
export FILTER_GROUPS_ENABLE=False
export TARGET_RESPONSE_TOKENS=${TARGET_RESPONSE_TOKENS:-0}

exec "${A0_SCRIPT_DIR}/03_run_a0_grpo_smoke.sh" "$@"
