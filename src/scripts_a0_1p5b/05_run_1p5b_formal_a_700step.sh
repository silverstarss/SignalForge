#!/usr/bin/env bash
# Formal Experiment A: Qwen2.5-1.5B standard GRPO baseline, fixed 700-step budget.
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
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-A_1p5b_formal_a_700step}
export MODEL_PATH=${MODEL_PATH:-$(choose_qwen25_1p5b_path)}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/processed/signal_forge_v1/train.parquet}
export TEST_FILE=${TEST_FILE:-${DATA_ROOT}/processed/signal_forge_v1/validation_id_effective_498.parquet}

# Frozen Experiment A scientific settings.
export ROLLOUT_N=${ROLLOUT_N:-8}
export VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-768}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
export ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0}
export VAL_TOP_P=${VAL_TOP_P:-1.0}
export VAL_TOP_K=${VAL_TOP_K:--1}
export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-False}

# Formal A budget and evaluation cadence.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-5}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-700}
# The effective train dataloader is ~693 steps at batch size 5, so use 2 epochs to guarantee 700 optimizer steps.
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
export TEST_FREQ=${TEST_FREQ:-20}
export SAVE_FREQ=${SAVE_FREQ:-100}
export ROLLOUT_DUMP_INTERVAL=${ROLLOUT_DUMP_INTERVAL:-20}
export ROLLOUT_DUMP_MAX_RECORDS=${ROLLOUT_DUMP_MAX_RECORDS:-40}
export VALIDATION_DUMP_MAX_RECORDS=${VALIDATION_DUMP_MAX_RECORDS:-498}
export LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-16}

# Preserve all possible formal checkpoints for this 700-step run: scheduled 100-step checkpoints plus any new-best checkpoints.
# Full checkpoints are about 19GB each in the current 1.5B FSDP path, so expand the data disk before running.
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-64}
export MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-64}

# Local veRL interprets ppo_mini_batch_size at prompt level before multiplying by rollout.n.
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-5}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

# Conservative single-GPU engineering defaults validated by the 40-step pilot.
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-1280}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}

export RESUME_MODE=${RESUME_MODE:-disable}
export PREFLIGHT_MODE=${PREFLIGHT_MODE:-deep}
export PREFLIGHT_FORMAL=${PREFLIGHT_FORMAL:-1}
export PREFLIGHT_LAUNCH_SCRIPT=${PREFLIGHT_LAUNCH_SCRIPT:-${SCRIPT_DIR}/05_run_1p5b_formal_a_700step.sh}

exec bash "${A0_SCRIPT_DIR}/03_run_a0_grpo_smoke.sh" \
    +trainer.best_checkpoint_metric=val/pass_at_1 \
    +trainer.best_checkpoint_save_on_update=True \
    "$@"
