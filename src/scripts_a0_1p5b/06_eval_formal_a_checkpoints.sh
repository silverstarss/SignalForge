#!/usr/bin/env bash
# Unified final evaluator for Formal A checkpoints.
#
# This script intentionally runs validation-only jobs. It does not train and
# does not save new checkpoints. Use TARGET=base|640|700|all.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
A0_SCRIPT_DIR=$(cd -- "${SCRIPT_DIR}/../scripts_a0" && pwd)

# shellcheck source=/dev/null
source "${A0_SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${A0_SCRIPT_DIR}"

FORMAL_PROJECT=${FORMAL_PROJECT:-signal_forge_a_1p5b}
FORMAL_EXPERIMENT=${FORMAL_EXPERIMENT:-A_1p5b_formal_a_700step}
EVAL_PROJECT=${EVAL_PROJECT:-signal_forge_a_1p5b_eval}
TARGET=${TARGET:-all}
DRY_RUN=${DRY_RUN:-0}

VALIDATION_FILE=${VALIDATION_FILE:-${DATA_ROOT}/processed/signal_forge_v1/validation_id_effective_498.parquet}
FORMAL_CKPT_DIR=${FORMAL_CKPT_DIR:-${CHECKPOINT_ROOT}/${FORMAL_PROJECT}/${FORMAL_EXPERIMENT}}

COMMON_ENV=(
    "TEST_FILE=${VALIDATION_FILE}"
    "TOTAL_TRAINING_STEPS=1"
    "TOTAL_EPOCHS=1"
    "SAVE_FREQ=-1"
    "TEST_FREQ=-1"
    "VALIDATION_DUMP_MAX_RECORDS=498"
    "LOG_VAL_GENERATIONS=0"
    "ROLLOUT_DUMP_INTERVAL=1000000"
    "PREFLIGHT_MODE=fast"
    "PREFLIGHT_FORMAL=1"
)

run_eval() {
    local label=$1
    local resume_path=${2:-}
    local experiment_name="final_eval_${label}_498_vllm_greedy"
    local -a cmd=(
        env
        "PROJECT_NAME=${EVAL_PROJECT}"
        "EXPERIMENT_NAME=${experiment_name}"
        "${COMMON_ENV[@]}"
        bash "${SCRIPT_DIR}/05_run_1p5b_formal_a_700step.sh"
        trainer.val_only=True
        trainer.save_freq=-1
        --allow-existing-output
    )

    if [ -n "${resume_path}" ]; then
        cmd+=(
            trainer.resume_mode=resume_path
            "trainer.resume_from_path=${resume_path}"
        )
    else
        cmd+=(trainer.resume_mode=disable)
    fi

    if [ "${DRY_RUN}" = "1" ]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
}

case "${TARGET}" in
    base)
        run_eval base
        ;;
    640|step640)
        run_eval step640 "${FORMAL_CKPT_DIR}/global_step_640"
        ;;
    700|step700)
        run_eval step700 "${FORMAL_CKPT_DIR}/global_step_700"
        ;;
    all)
        run_eval base
        run_eval step640 "${FORMAL_CKPT_DIR}/global_step_640"
        run_eval step700 "${FORMAL_CKPT_DIR}/global_step_700"
        ;;
    *)
        echo "ERROR: TARGET must be one of base, 640, 700, all; got ${TARGET}" >&2
        exit 2
        ;;
esac
