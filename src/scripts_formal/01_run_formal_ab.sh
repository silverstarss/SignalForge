#!/usr/bin/env bash
# Frozen single-GPU SignalForge v2 formal A/B launcher.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 {a|b} [approved operational arguments...]" >&2
    exit 2
fi

MODE=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
shift
case "${MODE}" in
    a)
        HIVE_ENABLE=False
        EXPERIMENT_SUFFIX=formal_a_grpo
        ;;
    b)
        HIVE_ENABLE=True
        EXPERIMENT_SUFFIX=formal_b_hive
        ;;
    *)
        echo "ERROR: formal mode must be 'a' (GRPO) or 'b' (HIVE)." >&2
        exit 2
        ;;
esac

# The formal launcher owns every semantic parameter. Only operational controls
# needed for preflight and exact-path resume may pass through to the base
# launcher; all other overrides require a protocol amendment and a new run ID.
FORWARD_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        --preflight-only|--preflight-fast|--preflight-deep|--allow-existing-output)
            FORWARD_ARGS+=("${arg}")
            ;;
        trainer.resume_from_path=*|trainer.resume_mode=resume_path|trainer.del_local_ckpt_after_load=True)
            FORWARD_ARGS+=("${arg}")
            ;;
        trainer.val_before_train=False|trainer.val_only=True|trainer.save_freq=-1|trainer.test_freq=-1)
            FORWARD_ARGS+=("${arg}")
            ;;
        +trainer.require_hive_signal_counters=True|+trainer.update_best_checkpoint_metadata=False)
            FORWARD_ARGS+=("${arg}")
            ;;
        +trainer.extra_save_steps=\[*\]|+trainer.validation_label=pilot_diagnostic_step80)
            FORWARD_ARGS+=("${arg}")
            ;;
        *)
            echo "ERROR: frozen formal launcher rejects semantic or unapproved override: ${arg}" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
BASE_LAUNCHER="${ROOT_DIR}/src/scripts_a0/03_run_a0_grpo_smoke.sh"
TRAIN_FILE_DEFAULT="${ROOT_DIR}/artifacts/formal_data/hive_math75_dapo25_seed42_validation_clean_max_exact_3to1/train.parquet"
VAL_FILE_DEFAULT="${ROOT_DIR}/artifacts/validation_data/qwen_math_a45202bd_math500_6e4ed1a2/formal_validation.parquet"
MODEL_PATH_DEFAULT="${ROOT_DIR}/artifacts/models/Qwen/Qwen2.5-3B-Instruct"

TRAIN_SHA256="94c4d168cf911797a6694a6be2c4ebc3c4ae677b51c0e03b7988227e0946de5f"
VAL_SHA256="cff36876612e3e55bb963e1f05a33b60c86cb7907d18befa90e38718566a4301"

for required in "${BASE_LAUNCHER}" "${TRAIN_FILE:-${TRAIN_FILE_DEFAULT}}" \
    "${TEST_FILE:-${VAL_FILE_DEFAULT}}" "${MODEL_PATH:-${MODEL_PATH_DEFAULT}}"; do
    if [ ! -e "${required}" ]; then
        echo "ERROR: frozen formal input is missing: ${required}" >&2
        exit 2
    fi
done

check_sha256() {
    local label=$1
    local expected=$2
    local path=$3
    local actual
    actual=$(sha256sum "${path}" | cut -d " " -f 1)
    if [ "${actual}" != "${expected}" ]; then
        echo "ERROR: ${label} SHA-256 mismatch: expected=${expected} actual=${actual} path=${path}" >&2
        exit 2
    fi
}

check_sha256 "formal training parquet" "${TRAIN_SHA256}" "${TRAIN_FILE:-${TRAIN_FILE_DEFAULT}}"
check_sha256 "formal validation parquet" "${VAL_SHA256}" "${TEST_FILE:-${VAL_FILE_DEFAULT}}"


export PROJECT_NAME=${PROJECT_NAME:-signal_forge_v2_formal}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25_3b_${EXPERIMENT_SUFFIX}_math75_dapo25_seed42}
export MODEL_PATH=${MODEL_PATH:-${MODEL_PATH_DEFAULT}}
export TRAIN_FILE=${TRAIN_FILE:-${TRAIN_FILE_DEFAULT}}
export TEST_FILE=${TEST_FILE:-${VAL_FILE_DEFAULT}}

export TRAIN_BATCH_SIZE=32
export GEN_BATCH_SIZE=32
export ROLLOUT_N=8
export VAL_ROLLOUT_N=1
FORMAL_SEGMENT_END_STEP=${FORMAL_SEGMENT_END_STEP:-300}
if [ "${FORMAL_SEGMENT_END_STEP}" != "100" ] && [ "${FORMAL_SEGMENT_END_STEP}" != "300" ]; then
    echo "ERROR: FORMAL_SEGMENT_END_STEP must be 100 or 300." >&2
    exit 2
fi
export TOTAL_TRAINING_STEPS=${FORMAL_SEGMENT_END_STEP}
export TOTAL_EPOCHS=300
export MAX_PROMPT_LENGTH=1792
export MAX_RESPONSE_LENGTH=1536

export ACTOR_LR=1e-6
export PPO_MINI_BATCH_SIZE=32
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export PPO_MAX_TOKEN_LEN_PER_GPU=4096
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096

export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
export ROLLOUT_TOP_K=-1
export VAL_TEMPERATURE=0
export VAL_TOP_P=1.0
export VAL_TOP_K=-1
export VAL_DO_SAMPLE=False
export ROLLOUT_MAX_MODEL_LEN=3328
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096
export ROLLOUT_MAX_NUM_SEQS=8
export ROLLOUT_GPU_MEM_UTIL=0.45

export FILTER_GROUPS_ENABLE=False
export SAVE_FREQ=50
export TEST_FREQ=50
export LOG_VAL_GENERATIONS=0
export VALIDATION_DUMP_MAX_RECORDS=1902
export HIVE_ROUND_DUMP_ENABLED=False
export MAX_ACTOR_CKPT_TO_KEEP=6
export MAX_CRITIC_CKPT_TO_KEEP=6
export RESUME_MODE=${RESUME_MODE:-disable}

# Only a hard process timeout is converted to the frozen 0.0 infrastructure
# fallback. All other verifier failures remain fail-fast.
export VERIFY_TIMEOUT_MODE=process
export VERIFY_TIMEOUT_SECONDS=120
export VERIFY_TIMEOUT_FALLBACK=True
export VERIFY_TIMEOUT_FALLBACK_SCORE=0.0
export VERIFY_TIMEOUT_DIAGNOSTICS_PATH="${ROOT_DIR}/artifacts/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}/logs/verifier_timeouts.jsonl"

export PREFLIGHT_FORMAL=1
export PREFLIGHT_STRICT=1
export PREFLIGHT_LAUNCH_SCRIPT="${SCRIPT_DIR}/01_run_formal_ab.sh"

exec "${BASE_LAUNCHER}" \
    --formal \
    --strict \
    algorithm.hive.enable="${HIVE_ENABLE}" \
    algorithm.hive.group_size=8 \
    algorithm.hive.seed=42 \
    algorithm.hive.p_easy_initial=0.5 \
    algorithm.hive.p_hard_initial=0.5 \
    algorithm.hive.p_default=0.5 \
    algorithm.hive.lambda_weight=1.0 \
    algorithm.hive.epsilon_p=0.01 \
    algorithm.hive.alpha_total=0.25 \
    algorithm.hive.delta_p=0.01 \
    algorithm.hive.p_min=0.05 \
    algorithm.hive.p_max=0.95 \
    algorithm.hive.upper_trim_ratio=0.25 \
    algorithm.hive.keep_ratio=0.50 \
    algorithm.hive.prompt_entropy_micro_batch_size=1 \
    +algorithm.hive.b_min=8 \
    +algorithm.hive.eta=1.25 \
    +algorithm.hive.max_topup_rounds=4 \
    +algorithm.hive.survival_epsilon=1e-6 \
    data.seed=42 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.optim.optimizer=AdamW \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    'actor_rollout_ref.actor.optim.betas=[0.9,0.999]' \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.use_kl_in_reward=False \
    +trainer.best_checkpoint_metric=val/six_benchmark_mean_accuracy \
    +trainer.best_checkpoint_save_on_update=False \
    trainer.val_before_train=True \
    "${FORWARD_ARGS[@]}"
