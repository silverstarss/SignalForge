#!/usr/bin/env bash
# Shared launcher for Qwen 3B GSM8K GRPO experiments A/B.
set -xeuo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace}
VERL_DIR=${VERL_DIR:-${ROOT_DIR}/verl}
VENV_DIR=${VENV_DIR:-${ROOT_DIR}/.venv-vllm}

if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

cd "${VERL_DIR}"

export VLLM_USE_V1=${VLLM_USE_V1:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}

ray stop --force || true

for override in "$@"; do
    case "${override}" in
        trainer.use_v1=True|trainer.use_v1=true|+trainer.use_v1=True|+trainer.use_v1=true|transfer_queue.enable=True|transfer_queue.enable=true|+transfer_queue.enable=True|+transfer_queue.enable=true)
            echo "ERROR: this project uses trainer.use_v1=False because the AutoDL veRL image lacks transfer_queue." >&2
            echo "Remove '${override}' from the command line." >&2
            exit 2
            ;;
    esac
done

EXP_ID=${EXP_ID:-A}
PROJECT_NAME=${PROJECT_NAME:-qwen3b_grpo_fair_gsm8k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${EXP_ID}_qwen25_3b_gsm8k}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-${ROOT_DIR}/data/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-${ROOT_DIR}/data/gsm8k/test.parquet}
OUT_DIR=${OUT_DIR:-${ROOT_DIR}/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOG_DIR=${LOG_DIR:-${OUT_DIR}/logs}
CKPT_DIR=${CKPT_DIR:-${OUT_DIR}/checkpoints}
ROLLOUT_DIR=${ROLLOUT_DIR:-${OUT_DIR}/rollout_data}
VAL_DIR=${VAL_DIR:-${OUT_DIR}/validation_data}
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${ROLLOUT_DIR}" "${VAL_DIR}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-8}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-8192}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-1536}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
ROLLOUT_TP=${ROLLOUT_TP:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
RAW_ROLLOUTS_PER_STEP=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
RAW_ROLLOUTS_TOTAL=$((RAW_ROLLOUTS_PER_STEP * TOTAL_TRAINING_STEPS))

GPU_LOG=${GPU_LOG:-${LOG_DIR}/gpu.csv}
"${ROOT_DIR}/scripts_grpo/monitor_gpu.sh" "${GPU_LOG}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

start_time=$(date +%Y%m%d_%H%M%S)
cat > "${LOG_DIR}/run_config_${start_time}.txt" <<EOF
exp_id=${EXP_ID}
project_name=${PROJECT_NAME}
experiment_name=${EXPERIMENT_NAME}
model_path=${MODEL_PATH}
train_file=${TRAIN_FILE}
test_file=${TEST_FILE}
train_batch_size=${TRAIN_BATCH_SIZE}
rollout_n=${ROLLOUT_N}
val_rollout_n=${VAL_ROLLOUT_N}
total_training_steps=${TOTAL_TRAINING_STEPS}
raw_rollouts_per_step=${RAW_ROLLOUTS_PER_STEP}
raw_rollouts_total=${RAW_ROLLOUTS_TOTAL}
filter_groups_enable=${FILTER_GROUPS_ENABLE:-False}
filter_groups_metric=${FILTER_GROUPS_METRIC:-acc}
checkpoint_rule=use the checkpoint with the best validation reward/pass@1 under the same validation schedule; report final-step separately.
EOF

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO:-True}
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
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
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.7}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.95}
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K:--1}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE:-True}
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

FILTER_GROUPS=(
    +algorithm.filter_groups.enable=${FILTER_GROUPS_ENABLE:-False}
    +algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC:-acc}
    +algorithm.filter_groups.max_num_gen_batches=${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-1}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.default_local_dir="${CKPT_DIR}"
    trainer.rollout_data_dir="${ROLLOUT_DIR}"
    trainer.validation_data_dir="${VAL_DIR}"
    trainer.log_val_generations=${LOG_VAL_GENERATIONS:-16}
    trainer.val_before_train=True
    trainer.save_freq=${SAVE_FREQ:-50}
    trainer.test_freq=${TEST_FREQ:-25}
    trainer.total_epochs=${TOTAL_EPOCHS:-1}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.use_v1=False
)

python -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${FILTER_GROUPS[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/train_${start_time}.log"
