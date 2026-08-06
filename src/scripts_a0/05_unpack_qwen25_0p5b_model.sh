#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${SCRIPT_DIR}"
MODEL_ROOT=${QWEN_MODEL_ROOT:-${MODEL_ROOT}/Qwen}
TARBALL=${TARBALL:-${SIGNAL_FORGE_ROOT}/qwen25_0p5b_instruct.tar.gz}
DEST=${DEST:-${MODEL_ROOT}/Qwen2___5-0___5B-Instruct}
LINK=${LINK:-${MODEL_ROOT}/Qwen2.5-0.5B-Instruct}

if [ -f "${DEST}/model.safetensors" ]; then
    echo "0.5B model already exists: ${DEST}"
else
    if [ ! -f "${TARBALL}" ]; then
        echo "ERROR: 0.5B tarball not found: ${TARBALL}" >&2
        echo "Set TARBALL=/path/to/qwen25_0p5b_instruct.tar.gz or copy it to ${SIGNAL_FORGE_ROOT}." >&2
        exit 2
    fi
    mkdir -p "${DEST}"
    tar -xzf "${TARBALL}" -C "${DEST}"
fi

ln -sfn "${DEST}" "${LINK}"
echo "0.5B model ready: ${DEST}"
echo "symlink: ${LINK} -> ${DEST}"
