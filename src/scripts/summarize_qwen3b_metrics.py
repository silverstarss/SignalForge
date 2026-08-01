#!/usr/bin/env python3
"""Summarize Qwen 3B GRPO experiment outputs from rollout/validation dumps and logs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, variance

METRIC_RE = re.compile(r"['\"]([^'\"]+)['\"]\s*:\s*([-+0-9.eE]+)")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_generation_dir(path: Path) -> dict[str, float]:
    rows = []
    for file in sorted(path.glob("*.jsonl")):
        rows.extend(iter_jsonl(file))
    if not rows:
        return {}

    rewards = [float(r.get("score", r.get("reward", 0.0))) for r in rows]
    lengths = [len(str(r.get("output", "")).split()) for r in rows]
    by_uid: dict[str, list[float]] = defaultdict(list)
    for r, reward in zip(rows, rewards, strict=True):
        by_uid[str(r.get("uid", r.get("input", len(by_uid))))].append(reward)

    group_pass1 = []
    group_passk = []
    group_vars = []
    all_correct = 0
    all_wrong = 0
    for vals in by_uid.values():
        correct = [v > 0.0 for v in vals]
        group_pass1.append(float(correct[0]))
        group_passk.append(float(any(correct)))
        if len(vals) > 1:
            group_vars.append(variance(vals))
        all_correct += int(all(correct))
        all_wrong += int(not any(correct))

    return {
        "samples": float(len(rows)),
        "groups": float(len(by_uid)),
        "pass@1": mean(group_pass1) if group_pass1 else math.nan,
        "pass@k": mean(group_passk) if group_passk else math.nan,
        "reward_mean": mean(rewards),
        "reward_var": variance(rewards) if len(rewards) > 1 else 0.0,
        "group_reward_var_mean": mean(group_vars) if group_vars else 0.0,
        "response_length_word_mean": mean(lengths),
        "all_correct_group_ratio": all_correct / len(by_uid) if by_uid else math.nan,
        "all_wrong_group_ratio": all_wrong / len(by_uid) if by_uid else math.nan,
    }


def summarize_logs(path: Path) -> dict[str, float]:
    latest = {}
    for file in sorted(path.glob("train_*.log")):
        for line in file.read_text(encoding="utf-8", errors="ignore").splitlines():
            for key, value in METRIC_RE.findall(line):
                if key.startswith(("actor/", "critic/", "response_length/", "dynamic_sampling/", "perf/", "training/")):
                    try:
                        latest[key] = float(value)
                    except ValueError:
                        pass
    return latest


def summarize_gpu(path: Path) -> dict[str, float]:
    gpu_file = path / "gpu.csv"
    if not gpu_file.exists():
        return {}
    rows = list(csv.DictReader(gpu_file.open("r", encoding="utf-8", errors="ignore")))
    if not rows:
        return {}
    util_keys = [k for k in rows[0] if "util" in k.lower()]
    mem_keys = [k for k in rows[0] if "mem" in k.lower()]
    out = {"gpu_log_rows": float(len(rows))}
    for keys, prefix in [(util_keys, "gpu_util"), (mem_keys, "gpu_mem")]:
        vals = []
        for row in rows:
            for key in keys:
                text = row.get(key, "").replace("%", "").replace("MiB", "").strip()
                try:
                    vals.append(float(text))
                except ValueError:
                    pass
        if vals:
            out[f"{prefix}_mean"] = mean(vals)
            out[f"{prefix}_max"] = max(vals)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Experiment output dir, e.g. /workspace/outputs/.../A_...")
    args = parser.parse_args()
    run_dir = args.run_dir
    summary = {}
    for name in ["rollout_data", "validation_data"]:
        metrics = summarize_generation_dir(run_dir / name)
        summary.update({f"{name}/{k}": v for k, v in metrics.items()})
    summary.update(summarize_logs(run_dir / "logs"))
    summary.update(summarize_gpu(run_dir / "logs"))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
