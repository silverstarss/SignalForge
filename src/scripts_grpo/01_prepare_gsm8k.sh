#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace}
VERL_DIR=${VERL_DIR:-${ROOT_DIR}/verl}
VENV_DIR=${VENV_DIR:-${ROOT_DIR}/.venv-vllm}
DATA_DIR=${DATA_DIR:-${ROOT_DIR}/data/gsm8k}

if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

mkdir -p "${DATA_DIR}"
cd "${VERL_DIR}"

python examples/data_preprocess/gsm8k.py --local_save_dir "${DATA_DIR}"

ls -lh "${DATA_DIR}"
python - <<PY
from pathlib import Path
import pandas as pd

data_dir = Path("${DATA_DIR}")
for name in ["train.parquet", "test.parquet"]:
    path = data_dir / name
    df = pd.read_parquet(path)
    print(name, len(df), list(df.columns))
    print(df.iloc[0]["prompt"])
    print(df.iloc[0]["reward_model"])
PY

