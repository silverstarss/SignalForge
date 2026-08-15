#!/usr/bin/env bash
# Formal Experiment B: Experiment A + Dynamic Sampling.
#
# B keeps A's model, data, reward, rollout shape, optimizer family, validation
# manifest, and fixed response-token budget. It enables group filtering based
# only on raw binary correctness: reject all-correct and all-wrong n=8 groups.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export PROJECT_NAME=${PROJECT_NAME:-signal_forge_b_1p5b}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-B_1p5b_dynamic_sampling_formal}

# Same A scientific protocol.
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

# Use a conservative optimizer-step ceiling; the formal stop condition is A's
# generated response-token budget, enforced inside the trainer after each step.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-5}
export GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
export TARGET_RESPONSE_TOKENS=${TARGET_RESPONSE_TOKENS:-9605733}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-700}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-16}
export TEST_FREQ=${TEST_FREQ:-20}
export SAVE_FREQ=${SAVE_FREQ:-50}
export ROLLOUT_DUMP_INTERVAL=${ROLLOUT_DUMP_INTERVAL:-20}
export ROLLOUT_DUMP_MAX_RECORDS=${ROLLOUT_DUMP_MAX_RECORDS:-40}
export VALIDATION_DUMP_MAX_RECORDS=${VALIDATION_DUMP_MAX_RECORDS:-498}
export LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-16}

export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-64}
export MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-64}
# Keep every scheduled 50-step checkpoint. For unscheduled best-validation
# checkpoints, keep only the latest best to avoid A-style checkpoint growth.
export BEST_CHECKPOINT_KEEP_LATEST_UNSCHEDULED=${BEST_CHECKPOINT_KEEP_LATEST_UNSCHEDULED:-True}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-5}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

# Dynamic Sampling. A candidate generation batch has TRAIN_BATCH_SIZE prompt
# groups; if too few groups are mixed, the trainer may generate additional
# candidate batches up to this cap before the update.
export FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE:-True}
export FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC:-raw_correctness}
export FILTER_GROUPS_MAX_NUM_GEN_BATCHES=${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-8}

export PREFLIGHT_MODE=${PREFLIGHT_MODE:-deep}
export PREFLIGHT_FORMAL=${PREFLIGHT_FORMAL:-1}
export PREFLIGHT_LAUNCH_SCRIPT=${PREFLIGHT_LAUNCH_SCRIPT:-${SCRIPT_DIR}/06_run_1p5b_formal_b_dynamic_sampling.sh}

exec bash "${SCRIPT_DIR}/05_run_1p5b_formal_a_700step.sh" "$@"
