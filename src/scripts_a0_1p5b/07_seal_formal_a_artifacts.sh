#!/usr/bin/env bash
# Write a lightweight manifest for the completed Formal A run.
#
# The default mode records paths, sizes, mtimes, git state, and known primary /
# secondary checkpoints without hashing multi-GB checkpoint tensors. Set
# HASH_LARGE=1 if you explicitly want SHA256 for every file.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
A0_SCRIPT_DIR=$(cd -- "${SCRIPT_DIR}/../scripts_a0" && pwd)

# shellcheck source=/dev/null
source "${A0_SCRIPT_DIR}/_paths.sh"
load_signal_forge_paths "${A0_SCRIPT_DIR}"

PROJECT_NAME=${PROJECT_NAME:-signal_forge_a_1p5b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-A_1p5b_formal_a_700step}
RUN_DIR=${RUN_DIR:-${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}}
CKPT_DIR=${CKPT_DIR:-${CHECKPOINT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}}
SEAL_DIR=${SEAL_DIR:-${RUN_DIR}/sealed_artifacts}
HASH_LARGE=${HASH_LARGE:-0}

mkdir -p "${SEAL_DIR}"

MANIFEST="${SEAL_DIR}/formal_a_artifact_manifest.json"
SUMMARY="${RUN_DIR}/reports/formal_a_summary.json"
BEST="${CKPT_DIR}/best_checkpoint.json"

python - "$RUN_DIR" "$CKPT_DIR" "$MANIFEST" "$SUMMARY" "$BEST" "$HASH_LARGE" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
ckpt_dir = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
best_path = Path(sys.argv[5])
hash_large = sys.argv[6] == "1"


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def sha256_file(path: Path) -> str | None:
    if not hash_large and path.stat().st_size > 64 * 1024 * 1024:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, root: Path) -> dict:
    st = path.stat()
    rec = {
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "bytes": st.st_size,
        "mtime_unix": st.st_mtime,
    }
    digest = sha256_file(path)
    if digest:
        rec["sha256"] = digest
    else:
        rec["sha256"] = None
        rec["sha256_skipped_reason"] = "large_file_set_HASH_LARGE_1_to_hash"
    return rec


def collect_files(root: Path) -> list[dict]:
    if not root.exists():
        return []
    return [file_record(path, root) for path in sorted(p for p in root.rglob("*") if p.is_file())]


best = json.loads(best_path.read_text()) if best_path.exists() else {}
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

payload = {
    "sealed_at_unix": time.time(),
    "sealed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "experiment": {
        "project_name": "signal_forge_a_1p5b",
        "experiment_name": "A_1p5b_formal_a_700step",
        "validation_manifest": "data/processed/signal_forge_v1/validation_id_effective_498.parquet",
        "primary_checkpoint": str(ckpt_dir / "global_step_700"),
        "secondary_best_checkpoint": best.get("checkpoint_path") or str(ckpt_dir / "global_step_640"),
        "best_checkpoint_metadata": best,
        "summary": summary,
    },
    "git": {
        "head": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": run_git(["status", "--short"]),
        "remote": run_git(["remote", "-v"]),
    },
    "files": {
        "logs": collect_files(run_dir / "logs"),
        "reports": collect_files(run_dir / "reports"),
        "validation_data": collect_files(run_dir / "validation_data"),
        "rollout_data": collect_files(run_dir / "rollout_data"),
    },
    "checkpoints": {
        "primary_step700": collect_files(ckpt_dir / "global_step_700"),
        "secondary_step640": collect_files(ckpt_dir / "global_step_640"),
        "best_checkpoint_json": file_record(best_path, ckpt_dir) if best_path.exists() else None,
        "latest_checkpoint_marker": file_record(ckpt_dir / "latest_checkpointed_iteration.txt", ckpt_dir)
        if (ckpt_dir / "latest_checkpointed_iteration.txt").exists()
        else None,
    },
}

manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(manifest_path)
PY

echo "Formal A artifact manifest written to ${MANIFEST}"
