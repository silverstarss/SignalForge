#!/usr/bin/env bash
# Run inside the extracted bundle on an AutoDL no-GPU machine.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=${BUNDLE_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${SCRIPT_DIR}"
WORKSPACE=${WORKSPACE:-${SIGNAL_FORGE_ROOT}}
WANDB_PROJECT=${WANDB_PROJECT:-signal_forge_a0}
RUN_NO_GPU_CHECKS=${RUN_NO_GPU_CHECKS:-1}
UNPACK_0P5B=${UNPACK_0P5B:-1}
UNPACK_1P5B=${UNPACK_1P5B:-0}
INSTALL_TMUX=${INSTALL_TMUX:-0}

mkdir -p "${BUNDLE_ROOT}/outputs" "${BUNDLE_ROOT}/data"
chmod +x "${BUNDLE_ROOT}"/src/scripts_a0/*.sh "${BUNDLE_ROOT}"/src/scripts_grpo/*.sh 2>/dev/null || true

if [ ! -e "${WORKSPACE}" ] && [ ! -L "${WORKSPACE}" ]; then
    ln -s "${BUNDLE_ROOT}" "${WORKSPACE}"
else
    mkdir -p "${WORKSPACE}"
    for item in src verl qwen25_0p5b_instruct.tar.gz qwen25_1p5b_instruct.tar.gz; do
        if [ -e "${BUNDLE_ROOT}/${item}" ] && [ ! -e "${WORKSPACE}/${item}" ]; then
            ln -s "${BUNDLE_ROOT}/${item}" "${WORKSPACE}/${item}"
        fi
    done
fi

if [ ! -d "${WORKSPACE}/src" ] || [ ! -d "${WORKSPACE}/verl" ]; then
    echo "ERROR: ${WORKSPACE}/src and ${WORKSPACE}/verl must exist after setup." >&2
    echo "       Try extracting the bundle under /workspace, or set WORKSPACE=${BUNDLE_ROOT}." >&2
    exit 2
fi

if [ "${INSTALL_TMUX}" = "1" ] && ! command -v tmux >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update && apt-get install -y tmux
    else
        echo "WARNING: tmux not found and apt-get unavailable." >&2
    fi
fi

cat > "${HOME}/.tmux.conf" <<'EOF'
set -g history-limit 50000
set -g mouse on
setw -g mode-keys vi
set -g status-interval 5
EOF

BASHRC_MARKER_BEGIN="# >>> signal_forge_a0 >>>"
BASHRC_MARKER_END="# <<< signal_forge_a0 <<<"
if ! grep -q "${BASHRC_MARKER_BEGIN}" "${HOME}/.bashrc" 2>/dev/null; then
    cat >> "${HOME}/.bashrc" <<EOF
${BASHRC_MARKER_BEGIN}
export ROOT_DIR=${WORKSPACE}
export PYTHONPATH=${WORKSPACE}/src:${WORKSPACE}/src/vendor_python:${WORKSPACE}/src/RewardScope/src:${WORKSPACE}/verl:\${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export WANDB_PROJECT=${WANDB_PROJECT}
cd ${WORKSPACE}
${BASHRC_MARKER_END}
EOF
fi

export ROOT_DIR="${WORKSPACE}"
export PYTHONPATH="${WORKSPACE}/src:${WORKSPACE}/src/vendor_python:${WORKSPACE}/src/RewardScope/src:${WORKSPACE}/verl:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export WANDB_PROJECT="${WANDB_PROJECT}"

if command -v wandb >/dev/null 2>&1; then
    if [ -n "${WANDB_API_KEY:-}" ]; then
        wandb login --relogin "${WANDB_API_KEY}"
    else
        echo "wandb is installed. To login later: wandb login <your_key>"
    fi
else
    echo "wandb command not found. veRL console logging still works; install/login later if needed."
fi

python - <<'PY'
from math_verify import parse, verify
from rewardscope.verification.math_verify import MathVerifyNumericVerifier
print("offline Math-Verify import ok")
print(MathVerifyNumericVerifier(mode="training").verify(r"\boxed{2}", "2"))
PY

if [ "${UNPACK_0P5B}" = "1" ] && [ -f "${WORKSPACE}/qwen25_0p5b_instruct.tar.gz" ]; then
    bash "${WORKSPACE}/src/scripts_a0/05_unpack_qwen25_0p5b_model.sh"
fi

if [ "${UNPACK_1P5B}" = "1" ] && [ -f "${WORKSPACE}/qwen25_1p5b_instruct.tar.gz" ]; then
    MODEL_ROOT=${WORKSPACE}/models/Qwen
    DEST=${MODEL_ROOT}/Qwen2___5-1___5B-Instruct
    LINK=${MODEL_ROOT}/Qwen2.5-1.5B-Instruct
    if [ ! -f "${DEST}/model.safetensors" ]; then
        mkdir -p "${DEST}"
        tar -xzf "${WORKSPACE}/qwen25_1p5b_instruct.tar.gz" -C "${DEST}"
    fi
    ln -sfn "${DEST}" "${LINK}"
    echo "1.5B model ready: ${DEST}"
fi

if [ "${RUN_NO_GPU_CHECKS}" = "1" ]; then
    bash "${WORKSPACE}/src/scripts_a0/00_prepare_a0_data.sh"
    bash "${WORKSPACE}/src/scripts_a0/01_check_reward_equivalence.sh"
    if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("ray") else 1)
PY
    then
        bash "${WORKSPACE}/src/scripts_a0/02_check_verl_reward_manager.sh"
    else
        echo "Skipping 02_check_verl_reward_manager.sh because ray is not importable in this no-GPU environment."
    fi
fi

cat > "${WORKSPACE}/AUTODL_NEXT_COMMANDS.txt" <<'EOF'
# 4090 0.5B regression
bash /workspace/src/scripts_a0/06_prepare_a0_0p5b_regression_data.sh
bash /workspace/src/scripts_a0/07_run_a0_0p5b_short.sh
bash /workspace/src/scripts_a0/08_run_a0_0p5b_regression.sh
bash /workspace/src/scripts_a0/09_reload_a0_0p5b_regression_checkpoint.sh

# A800 1.5B smoke
bash /workspace/src/scripts_a0/00_prepare_a0_data.sh
bash /workspace/src/scripts_a0/01_check_reward_equivalence.sh
bash /workspace/src/scripts_a0/02_check_verl_reward_manager.sh
bash /workspace/src/scripts_a0/03_run_a0_grpo_smoke.sh
bash /workspace/src/scripts_a0/04_reload_a0_checkpoint.sh

# tmux examples
tmux new -s a0_0p5b
tmux attach -t a0_0p5b
EOF

echo "No-GPU setup complete. Next commands: ${WORKSPACE}/AUTODL_NEXT_COMMANDS.txt"
