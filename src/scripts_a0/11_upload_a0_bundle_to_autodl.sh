#!/usr/bin/env bash
# Upload a prepared A0 bundle to an AutoDL no-GPU machine by scp.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${SCRIPT_DIR}"

BUNDLE=${1:-}
AUTODL_HOST=${AUTODL_HOST:-}
AUTODL_USER=${AUTODL_USER:-root}
AUTODL_PORT=${AUTODL_PORT:-22}
REMOTE_DIR=${REMOTE_DIR:-${OUTPUT_ROOT}/signal_forge_uploads}

if [ -z "${BUNDLE}" ]; then
    echo "Usage: AUTODL_HOST=<host> [AUTODL_PORT=port] bash $0 /path/to/bundle.tar.gz" >&2
    exit 2
fi
if [ ! -f "${BUNDLE}" ]; then
    echo "ERROR: bundle not found: ${BUNDLE}" >&2
    exit 2
fi
if [ -z "${AUTODL_HOST}" ]; then
    echo "ERROR: set AUTODL_HOST, for example AUTODL_HOST=connect.westb.seetacloud.com" >&2
    exit 2
fi

ssh -p "${AUTODL_PORT}" "${AUTODL_USER}@${AUTODL_HOST}" "mkdir -p '${REMOTE_DIR}'"
scp -P "${AUTODL_PORT}" "${BUNDLE}" "${BUNDLE}.sha256" "${AUTODL_USER}@${AUTODL_HOST}:${REMOTE_DIR}/"

REMOTE_BUNDLE="${REMOTE_DIR}/$(basename "${BUNDLE}")"
cat <<EOF
Uploaded:
  ${AUTODL_USER}@${AUTODL_HOST}:${REMOTE_BUNDLE}

On AutoDL no-GPU machine:
  cd ${REMOTE_DIR}
  sha256sum -c $(basename "${BUNDLE}").sha256
  tar -xzf $(basename "${BUNDLE}")
  cd $(basename "${BUNDLE}" .tar.gz)
  bash src/scripts_a0/12_autodl_nogpu_setup.sh
EOF
