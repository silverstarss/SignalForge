#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIGNAL_FORGE_SRC=${SIGNAL_FORGE_SRC:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
ROOT_DIR=${ROOT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
VENDOR_PYTHON=${VENDOR_PYTHON:-${SIGNAL_FORGE_SRC}/vendor_python}
REWARDSCOPE_SRC=${REWARDSCOPE_SRC:-${SIGNAL_FORGE_SRC}/RewardScope/src}
if [ ! -d "${REWARDSCOPE_SRC}" ] && [ -d "${ROOT_DIR}/RewardScope/src" ]; then
    REWARDSCOPE_SRC=${ROOT_DIR}/RewardScope/src
fi
if [ ! -d "${REWARDSCOPE_SRC}" ] && [ -d /home/tutu/grpo/src/RewardScope/src ]; then
    REWARDSCOPE_SRC=/home/tutu/grpo/src/RewardScope/src
fi
VERL_DIR=${VERL_DIR:-${ROOT_DIR}/verl}
if [ ! -d "${VERL_DIR}" ] && [ -d /home/tutu/grpo/verl ]; then
    VERL_DIR=/home/tutu/grpo/verl
fi

export PYTHONPATH="${SIGNAL_FORGE_SRC}:${VENDOR_PYTHON}:${REWARDSCOPE_SRC}:${VERL_DIR}:${PYTHONPATH:-}"
python -m signal_forge.rewards.check_verl_reward_manager "$@"
