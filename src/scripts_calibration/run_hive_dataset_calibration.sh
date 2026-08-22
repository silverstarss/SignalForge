#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${WORKSPACE}/src:${WORKSPACE}/src/RewardScope/src:${WORKSPACE}/verl:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" -m signal_forge.calibration.hive_dataset "$@"
