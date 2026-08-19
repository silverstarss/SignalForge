#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
A0_SCRIPT_DIR=$(cd -- "${SCRIPT_DIR}/../scripts_a0" && pwd)
# shellcheck source=/dev/null
source "${A0_SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${A0_SCRIPT_DIR}"

echo "[check] root: ${ROOT_DIR}"
echo "[check] verl: ${VERL_DIR}"

if [ -x "${VENV_DIR}/bin/python" ]; then
    export PATH="${VENV_DIR}/bin:${PATH}"
fi
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    echo "[check] activated venv: ${VENV_DIR}"
else
    echo "[check] venv not found at ${VENV_DIR}; using current python"
fi

cd "${VERL_DIR}"

echo "[check] python: $(which python)"
python - <<'PY'
import importlib
import sys

print("python", sys.version)
for name in ["torch", "vllm", "ray", "verl", "transformers", "datasets"]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name} ok {version}")
    except Exception as exc:
        print(f"{name} FAIL: {exc}")
        raise
PY

echo "[check] nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
else
    echo "[check] nvidia-smi unavailable; no-GPU boot is allowed for setup/preflight"
fi

echo "[check] veRL git"
git rev-parse --short HEAD || true
git status --short || true

