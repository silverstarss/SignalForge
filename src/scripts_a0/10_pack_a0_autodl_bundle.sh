#!/usr/bin/env bash
# Run on the WSL/host side. Creates an AutoDL-ready bundle with code, veRL,
# vendored Math-Verify runtime, local validation data, and optional model tars.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${SCRIPT_DIR}"

ROOT_DIR=${ROOT_DIR:-${SIGNAL_FORGE_ROOT}}
SRC_DIR=${SRC_DIR:-${SIGNAL_FORGE_SRC}}
VERL_DIR=${VERL_DIR:-${ROOT_DIR}/verl}
OUT_DIR=${OUT_DIR:-${OUTPUT_ROOT}/autodl_bundles}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
BUNDLE_NAME=${BUNDLE_NAME:-signal_forge_a0_autodl_${STAMP}}
STAGE_DIR=${STAGE_DIR:-${OUT_DIR}/${BUNDLE_NAME}}
OUT_TAR=${OUT_TAR:-${OUT_DIR}/${BUNDLE_NAME}.tar.gz}
INCLUDE_0P5B=${INCLUDE_0P5B:-1}
INCLUDE_1P5B=${INCLUDE_1P5B:-0}
QWEN_0P5B_TAR=${QWEN_0P5B_TAR:-${SIGNAL_FORGE_ROOT}/qwen25_0p5b_instruct.tar.gz}
QWEN_1P5B_TAR=${QWEN_1P5B_TAR:-${SIGNAL_FORGE_ROOT}/qwen25_1p5b_instruct.tar.gz}

if [ ! -d "${SRC_DIR}" ]; then
    echo "ERROR: SRC_DIR not found: ${SRC_DIR}" >&2
    exit 2
fi
if [ ! -d "${VERL_DIR}" ]; then
    echo "ERROR: VERL_DIR not found: ${VERL_DIR}" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

rsync -a --delete \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.log' \
    --exclude='*.tar.gz' \
    --exclude='*.zip' \
    --exclude='*Zone.Identifier' \
    "${SRC_DIR}/" "${STAGE_DIR}/src/"

rsync -a --delete \
    --exclude='.git/' \
    --exclude='.pytest_cache/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='core.*' \
    --exclude='outputs/' \
    --exclude='checkpoints/' \
    --exclude='wandb/' \
    --exclude='*Zone.Identifier' \
    "${VERL_DIR}/" "${STAGE_DIR}/verl/"

if [ "${INCLUDE_0P5B}" = "1" ]; then
    if [ ! -f "${QWEN_0P5B_TAR}" ]; then
        echo "ERROR: INCLUDE_0P5B=1 but tar not found: ${QWEN_0P5B_TAR}" >&2
        exit 2
    fi
    cp -a "${QWEN_0P5B_TAR}" "${STAGE_DIR}/qwen25_0p5b_instruct.tar.gz"
fi

if [ "${INCLUDE_1P5B}" = "1" ]; then
    if [ ! -f "${QWEN_1P5B_TAR}" ]; then
        echo "ERROR: INCLUDE_1P5B=1 but tar not found: ${QWEN_1P5B_TAR}" >&2
        exit 2
    fi
    cp -a "${QWEN_1P5B_TAR}" "${STAGE_DIR}/qwen25_1p5b_instruct.tar.gz"
fi

cat > "${STAGE_DIR}/AUTODL_BUNDLE_README.txt" <<EOF
Signal Forge A0 AutoDL bundle
created_at=${STAMP}

Extract on AutoDL, then run:
  bash src/scripts_a0/12_autodl_nogpu_setup.sh

Recommended no-GPU checks from the extracted bundle root:
  bash src/scripts_a0/00_prepare_a0_data.sh
  bash src/scripts_a0/01_check_reward_equivalence.sh
  bash src/scripts_a0/02_check_verl_reward_manager.sh

4090 0.5B regression from the extracted bundle root:
  bash src/scripts_a0/05_unpack_qwen25_0p5b_model.sh
  bash src/scripts_a0/06_prepare_a0_0p5b_regression_data.sh
  bash src/scripts_a0/07_run_a0_0p5b_short.sh
  bash src/scripts_a0/08_run_a0_0p5b_regression.sh
  bash src/scripts_a0/09_reload_a0_0p5b_regression_checkpoint.sh
EOF

(
    cd "${STAGE_DIR}"
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
    find . -type f -name '*Zone.Identifier' -delete
)

tar -C "${OUT_DIR}" -czf "${OUT_TAR}" "${BUNDLE_NAME}"
(
    cd "$(dirname "${OUT_TAR}")"
    sha256sum "$(basename "${OUT_TAR}")" > "$(basename "${OUT_TAR}").sha256"
)

ls -lh "${OUT_TAR}" "${OUT_TAR}.sha256"
echo "Created bundle: ${OUT_TAR}"
echo "Stage dir kept at: ${STAGE_DIR}"
echo "Upload example: AUTODL_HOST=... bash ${SRC_DIR}/scripts_a0/11_upload_a0_bundle_to_autodl.sh ${OUT_TAR}"
