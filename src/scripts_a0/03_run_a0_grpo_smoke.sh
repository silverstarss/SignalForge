#!/usr/bin/env bash
# Experiment A0: 2-3 step GRPO smoke test with the frozen Experiment A protocol.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${SCRIPT_DIR}"

PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
PREFLIGHT_MODE=${PREFLIGHT_MODE:-fast}
PREFLIGHT_STRICT=${PREFLIGHT_STRICT:-0}
PREFLIGHT_FORMAL=${PREFLIGHT_FORMAL:-0}
ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-0}
TRAIN_ARGS=()

for override in "$@"; do
    case "${override}" in
        --preflight-only)
            PREFLIGHT_ONLY=1
            ;;
        --preflight-fast)
            PREFLIGHT_MODE=fast
            ;;
        --preflight-deep)
            PREFLIGHT_MODE=deep
            ;;
        --strict)
            PREFLIGHT_STRICT=1
            ;;
        --formal)
            PREFLIGHT_FORMAL=1
            ;;
        --allow-existing-output)
            ALLOW_EXISTING_OUTPUT=1
            ;;
        trainer.use_v1=True|trainer.use_v1=true|+trainer.use_v1=True|+trainer.use_v1=true|transfer_queue.enable=True|transfer_queue.enable=true|+transfer_queue.enable=True|+transfer_queue.enable=true)
            echo "ERROR: A0 must use trainer.use_v1=False; v1 imports transfer_queue in this local veRL checkout." >&2
            echo "Remove '${override}' from the command line." >&2
            exit 2
            ;;
        *)
            TRAIN_ARGS+=("${override}")
            ;;
    esac
done

if [ -x "${VENV_DIR}/bin/python" ]; then
    export PATH="${VENV_DIR}/bin:${PATH}"
fi
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

cd "${VERL_DIR}"

export PYTHONPATH="${SIGNAL_FORGE_SRC}:${VENDOR_PYTHON:+${VENDOR_PYTHON}:}${REWARDSCOPE_SRC}:${VERL_DIR}:${PYTHONPATH:-}"
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export WANDB_DIR=${WANDB_DIR:-${WANDB_ROOT}}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${CACHE_ROOT}/wandb}

if [ "${ENABLE_WANDB:-0}" = "1" ]; then
    TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
else
    TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
fi

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_model_paths.sh"

PROJECT_NAME=${PROJECT_NAME:-signal_forge_a0}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-A0_grpo_math_verify_qwen25_${QWEN25_DEFAULT_SIZE}}
MODEL_PATH=${MODEL_PATH:-$(choose_qwen25_default_path)}
TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/signal_forge_a0/train.parquet}
TEST_FILE=${TEST_FILE:-${DATA_ROOT}/signal_forge_a0/val.parquet}
OUT_DIR=${OUT_DIR:-${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOG_DIR=${LOG_DIR:-${OUT_DIR}/logs}
CKPT_DIR=${CKPT_DIR:-${CHECKPOINT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}}
ROLLOUT_DIR=${ROLLOUT_DIR:-${OUT_DIR}/rollout_data}
VAL_DIR=${VAL_DIR:-${OUT_DIR}/validation_data}
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${ROLLOUT_DIR}" "${VAL_DIR}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-5}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-3}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-768}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-20}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-8192}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-1280}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
ROLLOUT_TP=${ROLLOUT_TP:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
RAW_ROLLOUTS_PER_STEP=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
RAW_ROLLOUTS_TOTAL=$((RAW_ROLLOUTS_PER_STEP * TOTAL_TRAINING_STEPS))
start_time=$(date +%Y%m%d_%H%M%S)

MONITOR_PID=""
GPU_LOG=${GPU_LOG:-${LOG_DIR}/gpu_${start_time}.csv}
MONITOR_SCRIPT=""
if [ -x "${ROOT_DIR}/scripts_grpo/monitor_gpu.sh" ]; then
    MONITOR_SCRIPT="${ROOT_DIR}/scripts_grpo/monitor_gpu.sh"
elif [ -x "${SIGNAL_FORGE_SRC}/scripts_grpo/monitor_gpu.sh" ]; then
    MONITOR_SCRIPT="${SIGNAL_FORGE_SRC}/scripts_grpo/monitor_gpu.sh"
fi
if [ "${PREFLIGHT_ONLY}" != "1" ] && [ -n "${MONITOR_SCRIPT}" ]; then
    "${MONITOR_SCRIPT}" "${GPU_LOG}" &
    MONITOR_PID=$!
    trap 'if [ -n "${MONITOR_PID}" ]; then kill "${MONITOR_PID}" 2>/dev/null || true; fi' EXIT
fi

if [ "${PREFLIGHT_ONLY}" != "1" ]; then
python - <<'PY' | tee "${LOG_DIR}/env_${start_time}.json"
import importlib.metadata as metadata
import json
import platform
import torch

from signal_forge.rewards.math_verify_adapter import compute_score

payload = {
    "platform": platform.platform(),
    "hostname": platform.node(),
    "signal_forge_env_name": "${SIGNAL_FORGE_ENV_NAME:-unknown}",
    "python": platform.python_version(),
    "hf_home": "${HF_HOME:-}",
    "hf_hub_cache": "${HF_HUB_CACHE:-}",
    "hf_datasets_cache": "${HF_DATASETS_CACHE:-}",
    "torch_home": "${TORCH_HOME:-}",
    "wandb_dir": "${WANDB_DIR:-}",
    "wandb_mode": "${WANDB_MODE:-}",
    "cuda_available": bool(torch.cuda.is_available()),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "gpus": [],
}
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    payload["gpus"].append({"index": index, "name": props.name, "total_memory_gib": round(props.total_memory / 1024**3, 2)})
for package in ["verl", "vllm", "transformers", "rewardscope", "math-verify"]:
    try:
        payload[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        payload[package] = None
payload["reward_adapter_probe"] = compute_score("gsm8k", "The final answer is \\boxed{2}.", "2")
print(json.dumps(payload, indent=2, sort_keys=True))
if not payload["cuda_available"]:
    raise SystemExit("CUDA is unavailable; A0 forbids silent CPU fallback.")
PY

cat > "${LOG_DIR}/run_config_${start_time}.txt" <<EOF
project_name=${PROJECT_NAME}
experiment_name=${EXPERIMENT_NAME}
model_path=${MODEL_PATH}
train_file=${TRAIN_FILE}
test_file=${TEST_FILE}
train_batch_size=${TRAIN_BATCH_SIZE}
gen_batch_size=${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
rollout_n=${ROLLOUT_N}
val_rollout_n=${VAL_ROLLOUT_N}
total_training_steps=${TOTAL_TRAINING_STEPS}
max_prompt_length=${MAX_PROMPT_LENGTH}
max_response_length=${MAX_RESPONSE_LENGTH}
validation_decoding=deterministic_temperature_0_n_1
raw_rollouts_per_step=${RAW_ROLLOUTS_PER_STEP}
raw_rollouts_total=${RAW_ROLLOUTS_TOTAL}
raw_rollout_tokens_budget_upper_bound=$((RAW_ROLLOUTS_TOTAL * MAX_RESPONSE_LENGTH))
target_response_tokens=${TARGET_RESPONSE_TOKENS:-0}
dynamic_sampling=${FILTER_GROUPS_ENABLE:-False}
output_dir=${OUT_DIR}
checkpoint_dir=${CKPT_DIR}
cache_root=${CACHE_ROOT}
hf_home=${HF_HOME}
wandb_dir=${WANDB_DIR}
wandb_mode=${WANDB_MODE}
trainer_logger=${TRAINER_LOGGER}
clip_higher=false
overlong_reward_shaping=false
curriculum_sampling=false
trainer_use_v1=false
transfer_queue_enable=false
checkpoint_rule=A0 smoke requires at least one checkpoint and one reload check; formal A uses fixed validation schedule and reports best-validation plus final step.
EOF
fi

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO:-True}
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    +data.gen_batch_size=${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.trust_remote_code=${TRUST_REMOTE_CODE:-True}
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF:-0}
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO:-0.2}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.2}
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE:-token-mean}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-1.0}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-1.0}
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K:--1}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-1.0}
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K:--1}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE:-False}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER:-False}
    actor_rollout_ref.rollout.enable_chunked_prefill=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}
    actor_rollout_ref.rollout.enable_prefix_caching=${ROLLOUT_ENABLE_PREFIX_CACHING:-True}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    reward.reward_manager.source=register
    reward.reward_manager.name=naive
    reward.custom_reward_function.path="${SIGNAL_FORGE_SRC}/signal_forge/rewards/math_verify_adapter.py"
    reward.custom_reward_function.name=compute_score
    reward.reward_model.rollout.name="${ROLLOUT_NAME:-vllm}"
    +reward.custom_reward_function.reward_kwargs.verify_timeout_mode=${VERIFY_TIMEOUT_MODE:-process}
    +reward.custom_reward_function.reward_kwargs.verify_timeout_seconds=${VERIFY_TIMEOUT_SECONDS:-10}
    +reward.custom_reward_function.reward_kwargs.verify_timeout_fallback=${VERIFY_TIMEOUT_FALLBACK:-True}
    +reward.custom_reward_function.reward_kwargs.verify_timeout_fallback_score=${VERIFY_TIMEOUT_FALLBACK_SCORE:-0.0}
    +reward.custom_reward_function.reward_kwargs.verifier_max_input_chars=${VERIFIER_MAX_INPUT_CHARS:-200000}
)

FILTER_GROUPS=(
    +algorithm.filter_groups.enable=${FILTER_GROUPS_ENABLE:-False}
    +algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC:-acc}
    +algorithm.filter_groups.max_num_gen_batches=${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-1}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=${TRAINER_LOGGER}
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.default_local_dir="${CKPT_DIR}"
    trainer.rollout_data_dir="${ROLLOUT_DIR}"
    trainer.validation_data_dir="${VAL_DIR}"
    +trainer.rollout_dump_interval=${ROLLOUT_DUMP_INTERVAL:-1}
    +trainer.rollout_dump_max_records=${ROLLOUT_DUMP_MAX_RECORDS:-40}
    +trainer.validation_dump_max_records=${VALIDATION_DUMP_MAX_RECORDS:-128}
    trainer.log_val_generations=${LOG_VAL_GENERATIONS:-16}
    trainer.val_before_train=True
    trainer.save_freq=${SAVE_FREQ:-1}
    trainer.test_freq=${TEST_FREQ:-1}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2}
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-2}
    +trainer.target_response_tokens=${TARGET_RESPONSE_TOKENS:-0}
    +trainer.best_checkpoint_keep_latest_unscheduled=${BEST_CHECKPOINT_KEEP_LATEST_UNSCHEDULED:-False}
    trainer.total_epochs=${TOTAL_EPOCHS:-1}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.resume_mode=${RESUME_MODE:-disable}
    trainer.use_v1=False
)

PREFLIGHT_ARGS=(
    --project-root "${SIGNAL_FORGE_ROOT}"
    --launch-script "${PREFLIGHT_LAUNCH_SCRIPT:-${SCRIPT_DIR}/08_run_a0_0p5b_regression.sh}"
    --resolved-config "${LOG_DIR}/resolved_config_${start_time}.yaml"
    --mode "${PREFLIGHT_MODE}"
    --json-report "${LOG_DIR}/preflight_${start_time}.json"
)
if [ "${PREFLIGHT_STRICT}" = "1" ]; then
    PREFLIGHT_ARGS+=(--strict)
fi
if [ "${PREFLIGHT_FORMAL}" = "1" ]; then
    PREFLIGHT_ARGS+=(--formal)
fi
if [ "${ALLOW_EXISTING_OUTPUT}" = "1" ]; then
    PREFLIGHT_ARGS+=(--allow-existing-output)
fi
if [ "${PREFLIGHT_MODE}" = "deep" ]; then
    PREFLIGHT_ARGS+=(--benchmark-parser)
fi

if [ "${SKIP_PREFLIGHT:-0}" = "1" ]; then
    echo "========================================================================" >&2
    echo "WARNING: SKIP_PREFLIGHT=1 is set. Training will start without preflight." >&2
    echo "Use only for emergency debugging; this bypasses A0 safety checks." >&2
    echo "========================================================================" >&2
else
    RESOLVED_CONFIG_TIMEOUT=${RESOLVED_CONFIG_TIMEOUT:-60}
    RESOLVE_CONFIG_WITH_HYDRA=${RESOLVE_CONFIG_WITH_HYDRA:-1}
    if [ "${PREFLIGHT_ONLY}" = "1" ] \
        && [ "${ALLOW_NO_GPU_BOOT:-1}" = "1" ] \
        && [ "${REQUIRE_GPU_FOR_PREFLIGHT:-0}" != "1" ] \
        && ! nvidia-smi -L >/dev/null 2>&1; then
        RESOLVE_CONFIG_WITH_HYDRA=0
    fi
    if [ "${RESOLVE_CONFIG_WITH_HYDRA}" = "1" ]; then
        if ! timeout "${RESOLVED_CONFIG_TIMEOUT}" python -m verl.trainer.main_ppo --cfg job --resolve \
            "${DATA[@]}" \
            "${MODEL[@]}" \
            "${ACTOR[@]}" \
            "${ROLLOUT[@]}" \
            "${REF[@]}" \
            "${REWARD[@]}" \
            "${FILTER_GROUPS[@]}" \
            "${TRAINER[@]}" \
            "${TRAIN_ARGS[@]}" > "${LOG_DIR}/resolved_config_${start_time}.yaml"; then
            echo "WARNING: Hydra --cfg job --resolve failed in this environment; writing launch-array fallback config for preflight." >&2
            python "${SIGNAL_FORGE_SRC}/tools/preflight_check.py" \
                --write-fallback-config "${LOG_DIR}/resolved_config_${start_time}.yaml" \
                $(printf ' --override %q' "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${ROLLOUT[@]}" "${REF[@]}" "${REWARD[@]}" "${FILTER_GROUPS[@]}" "${TRAINER[@]}" "${TRAIN_ARGS[@]}")
        fi
    else
        echo "INFO: no-GPU preflight-only mode; writing launch-array fallback config without importing veRL trainer." >&2
        python "${SIGNAL_FORGE_SRC}/tools/preflight_check.py" \
            --write-fallback-config "${LOG_DIR}/resolved_config_${start_time}.yaml" \
            $(printf ' --override %q' "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${ROLLOUT[@]}" "${REF[@]}" "${REWARD[@]}" "${FILTER_GROUPS[@]}" "${TRAINER[@]}" "${TRAIN_ARGS[@]}")
    fi
    python "${SIGNAL_FORGE_SRC}/tools/preflight_check.py" "${PREFLIGHT_ARGS[@]}"
fi

if [ "${PREFLIGHT_ONLY}" = "1" ]; then
    echo "Preflight-only mode complete; training was not started."
    exit 0
fi

if command -v ray >/dev/null 2>&1; then
    ray stop --force || true
fi

python -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${FILTER_GROUPS[@]}" \
    "${TRAINER[@]}" \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/train_${start_time}.log"
