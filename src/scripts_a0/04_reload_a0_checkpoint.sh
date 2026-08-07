#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${SCRIPT_DIR}"

PROJECT_NAME=${PROJECT_NAME:-signal_forge_a0}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-A0_grpo_math_verify_qwen25_${QWEN25_DEFAULT_SIZE}}
OUT_DIR=${OUT_DIR:-${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}}
CKPT_DIR=${CKPT_DIR:-${CHECKPOINT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LATEST_FILE=${LATEST_FILE:-${CKPT_DIR}/latest_checkpointed_iteration.txt}

if [ ! -f "${LATEST_FILE}" ]; then
    echo "ERROR: latest checkpoint marker not found: ${LATEST_FILE}" >&2
    exit 2
fi

STEP=$(tr -d '[:space:]' < "${LATEST_FILE}")
RESUME_PATH=${RESUME_PATH:-${CKPT_DIR}/global_step_${STEP}}
if [ ! -d "${RESUME_PATH}" ]; then
    echo "ERROR: checkpoint directory not found: ${RESUME_PATH}" >&2
    exit 2
fi

export RESUME_MODE=resume_path
exec bash "${SCRIPT_DIR}/03_run_a0_grpo_smoke.sh" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${RESUME_PATH}" \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    --allow-existing-output \
    "$@"
