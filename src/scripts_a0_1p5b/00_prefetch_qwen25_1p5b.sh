#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
A0_SCRIPT_DIR=$(cd -- "${SCRIPT_DIR}/../scripts_a0" && pwd)

# shellcheck source=/dev/null
source "${A0_SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${A0_SCRIPT_DIR}"

REPO_ID=${QWEN25_1P5B_REPO_ID:-Qwen/Qwen2.5-1.5B-Instruct}
LOCAL_DIR=${QWEN25_1P5B_LOCAL_DIR:-${MODEL_ROOT}/Qwen/Qwen2.5-1.5B-Instruct}

if [ -f "${LOCAL_DIR}/config.json" ] && [ -f "${LOCAL_DIR}/tokenizer_config.json" ]; then
    echo "[model] using cached local model: ${LOCAL_DIR}"
    exit 0
fi

mkdir -p "$(dirname -- "${LOCAL_DIR}")" "${HF_HOME}" "${HF_HUB_CACHE}"

echo "[model] downloading ${REPO_ID} to ${LOCAL_DIR}"
echo "[model] HF_HOME=${HF_HOME}"
python - <<'PY'
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception as exc:  # pragma: no cover - runtime setup check
    raise SystemExit(f"huggingface_hub is required to download the model: {exc}")

repo_id = os.environ.get("QWEN25_1P5B_REPO_ID", "Qwen/Qwen2.5-1.5B-Instruct")
local_dir = Path(os.environ.get("QWEN25_1P5B_LOCAL_DIR", "")).expanduser()
if not str(local_dir):
    model_root = Path(os.environ["MODEL_ROOT"])
    local_dir = model_root / "Qwen" / "Qwen2.5-1.5B-Instruct"
local_dir.parent.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(local_dir)
PY
