#!/usr/bin/env bash
# Local WSL smoke test. It is intentionally tiny and is not meant for results.
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
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-1}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}

ray stop --force || true
pkill -f vllm || true

PROJECT_NAME=${PROJECT_NAME:-local_grpo_smoke}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25_0p5b_gsm8k_tiny}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-${ROOT_DIR}/data/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-${ROOT_DIR}/data/gsm8k/test.parquet}
OUT_DIR=${OUT_DIR:-${ROOT_DIR}/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOG_DIR=${LOG_DIR:-${OUT_DIR}/logs}
CKPT_DIR=${CKPT_DIR:-${OUT_DIR}/checkpoints}
ROLLOUT_DIR=${ROLLOUT_DIR:-${OUT_DIR}/rollout_data}
VAL_DIR=${VAL_DIR:-${OUT_DIR}/validation_data}
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${ROLLOUT_DIR}" "${VAL_DIR}"

GPU_LOG=${GPU_LOG:-${LOG_DIR}/gpu.csv}
MONITOR_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
    "${ROOT_DIR}/scripts_grpo/monitor_gpu.sh" "${GPU_LOG}" &
    MONITOR_PID=$!
fi
cleanup() {
    if [ -n "${MONITOR_PID}" ]; then
        kill "${MONITOR_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_time=$(date +%Y%m%d_%H%M%S)

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE:-1}
    data.max_prompt_length=${MAX_PROMPT_LENGTH:-128}
    data.max_response_length=${MAX_RESPONSE_LENGTH:-64}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-1}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-256}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF:-0}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${ROLLOUT_N:-1}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-1}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.35}
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN:-256}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-256}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-1}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-256}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-256}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE:-1}
    trainer.nnodes=${NNODES:-1}
    trainer.default_local_dir="${CKPT_DIR}"
    trainer.rollout_data_dir=null
    trainer.validation_data_dir=null
    trainer.log_val_generations=${LOG_VAL_GENERATIONS:-0}
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ:--1}
    trainer.test_freq=${TEST_FREQ:--1}
    trainer.total_epochs=${TOTAL_EPOCHS:-1}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-1}
    trainer.use_v1=False
)

python -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/train_${start_time}.log"

