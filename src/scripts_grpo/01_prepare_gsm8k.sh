#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
A0_SCRIPT_DIR=$(cd -- "${SCRIPT_DIR}/../scripts_a0" && pwd)
# shellcheck source=/dev/null
source "${A0_SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${A0_SCRIPT_DIR}"
DATA_DIR=${DATA_DIR:-${DATA_ROOT}/gsm8k}

if [ -x "${VENV_DIR}/bin/python" ]; then
    export PATH="${VENV_DIR}/bin:${PATH}"
fi
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

