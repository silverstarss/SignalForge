#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace}
VERL_DIR=${VERL_DIR:-${ROOT_DIR}/verl}
VENV_DIR=${VENV_DIR:-${ROOT_DIR}/.venv-vllm}

echo "[check] root: ${ROOT_DIR}"
echo "[check] verl: ${VERL_DIR}"

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
nvidia-smi

echo "[check] veRL git"
git rev-parse --short HEAD || true
git status --short || true

