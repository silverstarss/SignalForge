#!/usr/bin/env bash

load_signal_forge_paths() {
    local script_dir="$1"
    local inferred_src
    local inferred_root

    inferred_src=$(cd -- "${script_dir}/.." && pwd)
    inferred_root=$(cd -- "${script_dir}/../.." && pwd)

    SIGNAL_FORGE_ROOT=${SIGNAL_FORGE_ROOT:-${inferred_root}}
    SIGNAL_FORGE_CONFIG=${SIGNAL_FORGE_CONFIG:-${inferred_root}/config/signal_forge.env}

    if [ -f "${SIGNAL_FORGE_CONFIG}" ]; then
        # shellcheck source=/dev/null
        source "${SIGNAL_FORGE_CONFIG}"
    fi

    SIGNAL_FORGE_ROOT=${SIGNAL_FORGE_ROOT:-${inferred_root}}
    SIGNAL_FORGE_SRC=${SIGNAL_FORGE_SRC:-${SIGNAL_FORGE_ROOT}/src}
    if [ -d "${inferred_src}/signal_forge" ]; then
        SIGNAL_FORGE_ROOT=${inferred_root}
        SIGNAL_FORGE_SRC=${inferred_src}
    fi
    SIGNAL_FORGE_ARTIFACT_ROOT=${SIGNAL_FORGE_ARTIFACT_ROOT:-${SIGNAL_FORGE_ROOT}/artifacts}

    MODEL_ROOT=${MODEL_ROOT:-${SIGNAL_FORGE_ARTIFACT_ROOT}/models}
    CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${SIGNAL_FORGE_ARTIFACT_ROOT}/checkpoints}
    OUTPUT_ROOT=${OUTPUT_ROOT:-${SIGNAL_FORGE_ARTIFACT_ROOT}/outputs}
    WANDB_ROOT=${WANDB_ROOT:-${SIGNAL_FORGE_ARTIFACT_ROOT}/wandb}
    WHEEL_ROOT=${WHEEL_ROOT:-${SIGNAL_FORGE_ARTIFACT_ROOT}/wheels}
    DATA_ROOT=${DATA_ROOT:-${SIGNAL_FORGE_ROOT}/data}
    CACHE_ROOT=${CACHE_ROOT:-${SIGNAL_FORGE_ARTIFACT_ROOT}/cache}
    HF_HOME=${HF_HOME:-${CACHE_ROOT}/huggingface}
    HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
    HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
    TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
    TORCH_HOME=${TORCH_HOME:-${CACHE_ROOT}/torch}
    XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}
    WANDB_DIR=${WANDB_DIR:-${WANDB_ROOT}}
    WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${CACHE_ROOT}/wandb}
    WANDB_PROJECT=${WANDB_PROJECT:-signal_forge}
    WANDB_MODE=${WANDB_MODE:-offline}
    ENABLE_WANDB=${ENABLE_WANDB:-0}
    ALLOW_NO_GPU_BOOT=${ALLOW_NO_GPU_BOOT:-1}
    REQUIRE_GPU_FOR_PREFLIGHT=${REQUIRE_GPU_FOR_PREFLIGHT:-0}
    QWEN25_1P5B_REPO_ID=${QWEN25_1P5B_REPO_ID:-Qwen/Qwen2.5-1.5B-Instruct}
    QWEN25_3B_REPO_ID=${QWEN25_3B_REPO_ID:-Qwen/Qwen2.5-3B-Instruct}
    ALLOW_HF_MODEL_DOWNLOAD=${ALLOW_HF_MODEL_DOWNLOAD:-1}
    VENDOR_PYTHON=${VENDOR_PYTHON:-}
    VERL_DIR=${VERL_DIR:-${SIGNAL_FORGE_ROOT}/verl}
    REWARDSCOPE_SRC=${REWARDSCOPE_SRC:-${SIGNAL_FORGE_SRC}/RewardScope/src}
    VENV_DIR=${VENV_DIR:-${SIGNAL_FORGE_ROOT}/.venv-vllm}
    QWEN25_DEFAULT_SIZE=${QWEN25_DEFAULT_SIZE:-1.5B}

    ROOT_DIR=${SIGNAL_FORGE_ROOT}

    export SIGNAL_FORGE_ROOT SIGNAL_FORGE_SRC SIGNAL_FORGE_ARTIFACT_ROOT
    export MODEL_ROOT CHECKPOINT_ROOT OUTPUT_ROOT WANDB_ROOT WHEEL_ROOT DATA_ROOT CACHE_ROOT
    export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE TORCH_HOME XDG_CACHE_HOME
    export WANDB_DIR WANDB_CACHE_DIR WANDB_PROJECT WANDB_MODE ENABLE_WANDB
    export ALLOW_NO_GPU_BOOT REQUIRE_GPU_FOR_PREFLIGHT
    export VENDOR_PYTHON VERL_DIR REWARDSCOPE_SRC VENV_DIR ROOT_DIR QWEN25_DEFAULT_SIZE QWEN25_1P5B_REPO_ID QWEN25_3B_REPO_ID ALLOW_HF_MODEL_DOWNLOAD
}

find_flash_attn_wheel() {
    if [ -n "${FLASH_ATTN_WHEEL:-}" ]; then
        printf '%s\n' "${FLASH_ATTN_WHEEL}"
        return 0
    fi

    local wheel
    wheel=$(find "${WHEEL_ROOT}" -maxdepth 1 -type f -name 'flash_attn*.whl' -print -quit 2>/dev/null || true)
    if [ -n "${wheel}" ]; then
        printf '%s\n' "${wheel}"
        return 0
    fi

    return 1
}
