#!/usr/bin/env bash

if ! declare -F load_signal_forge_paths >/dev/null 2>&1; then
    SCRIPT_DIR=${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/_paths.sh"
    load_signal_forge_paths "${SCRIPT_DIR}"
fi

choose_qwen25_model_path() {
    local size="$1"
    local dotted
    local escaped
    local repo_id

    case "${size}" in
        0.5B|0.5b|0p5b|0p5B)
            dotted="Qwen2.5-0.5B-Instruct"
            escaped="Qwen2___5-0___5B-Instruct"
            repo_id="${QWEN25_0P5B_REPO_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
            ;;
        1.5B|1.5b|1p5b|1p5B)
            dotted="Qwen2.5-1.5B-Instruct"
            escaped="Qwen2___5-1___5B-Instruct"
            repo_id="${QWEN25_1P5B_REPO_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
            ;;
        *)
            echo "ERROR: unsupported Qwen2.5 model size: ${size}" >&2
            return 2
            ;;
    esac

    local candidates=(
        "${MODEL_ROOT}/Qwen/${dotted}"
        "${MODEL_ROOT}/Qwen/${escaped}"
        "${MODEL_ROOT}/${dotted}"
        "${MODEL_ROOT}/${escaped}"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -e "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    printf '%s\n' "${repo_id}"
}

choose_qwen25_0p5b_path() {
    choose_qwen25_model_path "0.5B"
}

choose_qwen25_1p5b_path() {
    choose_qwen25_model_path "1.5B"
}

choose_qwen25_default_path() {
    choose_qwen25_model_path "${QWEN25_DEFAULT_SIZE:-1.5B}"
}
