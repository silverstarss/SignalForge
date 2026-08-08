#!/usr/bin/env bash
# Stage 4: resume the 1.5B 40-step regression checkpoint and continue training.
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
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-A_1p5b_regression_40step}
export MODEL_PATH=${MODEL_PATH:-$(choose_qwen25_1p5b_path)}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/processed/signal_forge_v1/train.parquet}
export TEST_FILE=${TEST_FILE:-${DATA_ROOT}/processed/signal_forge_v1/validation_id.parquet}

CKPT_DIR=${CKPT_DIR:-${CHECKPOINT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LATEST_FILE=${LATEST_FILE:-${CKPT_DIR}/latest_checkpointed_iteration.txt}

if [ ! -f "${LATEST_FILE}" ]; then
    echo "ERROR: latest checkpoint marker not found: ${LATEST_FILE}" >&2
    exit 2
fi

STEP=$(tr -d '[:space:]' < "${LATEST_FILE}")
if ! [[ "${STEP}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: latest checkpoint marker is not an integer: ${LATEST_FILE} -> ${STEP}" >&2
    exit 2
fi

RESUME_PATH=${RESUME_PATH:-${CKPT_DIR}/global_step_${STEP}}
if [ ! -d "${RESUME_PATH}" ]; then
    echo "ERROR: checkpoint directory not found: ${RESUME_PATH}" >&2
    exit 2
fi

RESUME_EXTRA_STEPS=${RESUME_EXTRA_STEPS:-5}
if ! [[ "${RESUME_EXTRA_STEPS}" =~ ^[0-9]+$ ]] || [ "${RESUME_EXTRA_STEPS}" -le 0 ]; then
    echo "ERROR: RESUME_EXTRA_STEPS must be a positive integer, got: ${RESUME_EXTRA_STEPS}" >&2
    exit 2
fi

# Keep frozen scientific settings from Stage 0.
export ROLLOUT_N=${ROLLOUT_N:-8}
export VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-768}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-5}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-$((STEP + RESUME_EXTRA_STEPS))}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TEST_FREQ=${TEST_FREQ:-5}
export ROLLOUT_DUMP_INTERVAL=${ROLLOUT_DUMP_INTERVAL:-5}
export ROLLOUT_DUMP_MAX_RECORDS=${ROLLOUT_DUMP_MAX_RECORDS:-40}
export VALIDATION_DUMP_MAX_RECORDS=${VALIDATION_DUMP_MAX_RECORDS:-128}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
export MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-2}
export LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-16}

# Local veRL interprets ppo_mini_batch_size at prompt level before multiplying by rollout.n.
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-5}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

# Conservative A800 engineering defaults. Adjust these before changing n or max_response_length.
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-1280}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}

export RESUME_MODE=resume_path
export ALLOW_EXISTING_OUTPUT=1
export PREFLIGHT_MODE=${PREFLIGHT_MODE:-deep}
export PREFLIGHT_FORMAL=${PREFLIGHT_FORMAL:-1}
export PREFLIGHT_LAUNCH_SCRIPT=${PREFLIGHT_LAUNCH_SCRIPT:-${SCRIPT_DIR}/04_reload_1p5b_checkpoint.sh}

echo "Resuming ${EXPERIMENT_NAME} from ${RESUME_PATH}; target total_training_steps=${TOTAL_TRAINING_STEPS}" >&2

exec bash "${A0_SCRIPT_DIR}/03_run_a0_grpo_smoke.sh" \
    --allow-existing-output \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${RESUME_PATH}" \
    trainer.val_before_train=True \
    "$@"
