#!/usr/bin/env bash

choose_qwen25_model_path() {
    local size="$1"
    local dotted
    local escaped

    case "${size}" in
        0.5B|0.5b|0p5b|0p5B)
            dotted="Qwen2.5-0.5B-Instruct"
            escaped="Qwen2___5-0___5B-Instruct"
            ;;
        1.5B|1.5b|1p5b|1p5B)
            dotted="Qwen2.5-1.5B-Instruct"
            escaped="Qwen2___5-1___5B-Instruct"
            ;;
        *)
            echo "ERROR: unsupported Qwen2.5 model size: ${size}" >&2
            return 2
            ;;
    esac

    local candidates=(
        "${ROOT_DIR}/models/Qwen/${dotted}"
        "${ROOT_DIR}/models/Qwen/${escaped}"
        "/home/tutu/tinyvr/models/Qwen/${dotted}"
        "/home/tutu/tinyvr/models/Qwen/${escaped}"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -e "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    case "${size}" in
        0.5B|0.5b|0p5b|0p5B)
            printf '%s\n' "Qwen/Qwen2.5-0.5B-Instruct"
            ;;
        *)
            printf '%s\n' "Qwen/Qwen2.5-1.5B-Instruct"
            ;;
    esac
}

choose_qwen25_0p5b_path() {
    choose_qwen25_model_path "0.5B"
}

choose_qwen25_1p5b_path() {
    choose_qwen25_model_path "1.5B"
}
