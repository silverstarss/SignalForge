#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIGNAL_FORGE_SRC=${SIGNAL_FORGE_SRC:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
ROOT_DIR=${ROOT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
VENDOR_PYTHON=${VENDOR_PYTHON:-${SIGNAL_FORGE_SRC}/vendor_python}
PROJECT_NAME=${PROJECT_NAME:-signal_forge_a0}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-A0_grpo_math_verify_qwen25_1p5b}
OUT_DIR=${OUT_DIR:-${ROOT_DIR}/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}
CKPT_DIR=${CKPT_DIR:-${OUT_DIR}/checkpoints}
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
    "$@"
