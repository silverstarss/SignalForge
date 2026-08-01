#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace}
MODEL_ROOT=${MODEL_ROOT:-${ROOT_DIR}/models/Qwen}
TARBALL=${TARBALL:-${ROOT_DIR}/qwen25_0p5b_instruct.tar.gz}
DEST=${DEST:-${MODEL_ROOT}/Qwen2___5-0___5B-Instruct}
LINK=${LINK:-${MODEL_ROOT}/Qwen2.5-0.5B-Instruct}

if [ -f "${DEST}/model.safetensors" ]; then
    echo "0.5B model already exists: ${DEST}"
else
    if [ ! -f "${TARBALL}" ]; then
        echo "ERROR: 0.5B tarball not found: ${TARBALL}" >&2
        echo "Set TARBALL=/path/to/qwen25_0p5b_instruct.tar.gz or copy it to ${ROOT_DIR}." >&2
        exit 2
    fi
    mkdir -p "${DEST}"
    tar -xzf "${TARBALL}" -C "${DEST}"
fi

ln -sfn "${DEST}" "${LINK}"
echo "0.5B model ready: ${DEST}"
echo "symlink: ${LINK} -> ${DEST}"
