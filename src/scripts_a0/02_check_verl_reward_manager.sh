#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${SCRIPT_DIR}"

export PYTHONPATH="${SIGNAL_FORGE_SRC}:${VENDOR_PYTHON:+${VENDOR_PYTHON}:}${REWARDSCOPE_SRC}:${VERL_DIR}:${PYTHONPATH:-}"
python -m signal_forge.rewards.check_verl_reward_manager "$@"
