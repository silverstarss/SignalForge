#!/usr/bin/env python3
"""Extract first-pass evidence from a veRL run directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


KEYWORDS = [
    "reward",
    "acc",
    "score",
    "length",
    "response",
    "entropy",
    "kl",
    "throughput",
    "tokens/s",
    "time",
    "timing",
    "perf",
]


def latest_log(log_dir: Path) -> Path | None:
    logs = sorted(log_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def extract_metric_lines(log_path: Path, max_lines: int) -> list[str]:
    rows: list[str] = []
    pattern = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.IGNORECASE)
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if pattern.search(line):
                rows.append(line.rstrip())
    return rows[-max_lines:]


def summarize_gpu(gpu_csv: Path) -> dict[str, object]:
    if not gpu_csv.exists():
        return {"available": False}

    max_used = 0.0
    max_total = 0.0
    max_util = 0.0
    rows = 0
    with gpu_csv.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            used = float(row["memory.used"].strip())
            total = float(row["memory.total"].strip())
            util = float(row["utilization.gpu"].strip())
            max_used = max(max_used, used)
            max_total = max(max_total, total)
            max_util = max(max_util, util)

    return {
        "available": True,
        "rows": rows,
        "max_memory_used_mib": max_used,
        "max_memory_total_mib": max_total,
        "max_gpu_util_percent": max_util,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Run directory, e.g. /workspace/outputs/project/experiment")
    parser.add_argument("--max-lines", type=int, default=120)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_dir = run_dir / "logs"
    log_path = latest_log(log_dir)
    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "log_path": str(log_path) if log_path else None,
        "gpu": summarize_gpu(log_dir / "gpu.csv"),
        "metric_lines": [],
    }
    if log_path:
        summary["metric_lines"] = extract_metric_lines(log_path, args.max_lines)

    out_path = run_dir / "run_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

