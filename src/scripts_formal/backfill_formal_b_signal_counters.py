#!/usr/bin/env python3
"""Backfill exact candidate signal counters for Formal B steps 1-50.

The old logs retain complete candidate-pool metrics but not the arrival order
needed to reconstruct the final 32 training groups.  Training-slice signal
observations therefore start strictly after this checkpoint.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from signal_forge.hive.signal_metrics import (
    HIVE_SIGNAL_COUNTERS_FILENAME,
    HiveSignalCounters,
)


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_PATTERN = re.compile(r"\bstep:(\d+)\s+-\s+")


def _metric(line: str, name: str) -> float:
    match = re.search(rf"(?:^| - ){re.escape(name)}:([^ ]+)", line)
    if match is None:
        raise ValueError(f"step log line is missing {name!r}")
    return float(match.group(1))


def build_step50_counters(log_path: Path, *, expected_step: int = 50) -> HiveSignalCounters:
    rows: dict[int, str] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_ESCAPE.sub("", raw_line)
            match = STEP_PATTERN.search(line)
            if match is None or "hive/generated_prompt_groups:" not in line:
                continue
            step = int(match.group(1))
            if step <= expected_step:
                if step in rows:
                    raise ValueError(f"duplicate HIVE metric row for step {step}")
                rows[step] = line
    expected = set(range(1, expected_step + 1))
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        raise ValueError(f"HIVE log does not contain exactly steps 1..{expected_step}; missing={missing}")

    generated_groups = 0
    scalar_effective = 0
    raw_mixed = 0
    topup_groups = 0
    for step in range(1, expected_step + 1):
        line = rows[step]
        generated_groups += int(_metric(line, "hive/generated_prompt_groups"))
        scalar_effective += int(_metric(line, "hive/effective_prompt_groups"))
        raw_mixed += int(_metric(line, "group/mixed_count"))
        topup_groups += int(_metric(line, "hive/generated_groups_topup"))
    final_line = rows[expected_step]
    cumulative_groups = int(_metric(final_line, "compute/generated_prompt_groups"))
    cumulative_effective = int(_metric(final_line, "compute/effective_prompt_groups"))
    response_tokens = int(_metric(final_line, "compute/generated_response_tokens"))
    if generated_groups != cumulative_groups or scalar_effective != cumulative_effective:
        raise ValueError("summed HIVE log rows disagree with the step-50 cumulative compute counters")
    if raw_mixed > scalar_effective:
        raise ValueError("candidate raw-correctness mixed groups exceed scalar-effective groups")

    return HiveSignalCounters(
        global_step=expected_step,
        candidate_observation_start_step=0,
        training_observation_start_step=expected_step,
        candidate_observed_updates=expected_step,
        candidate_groups=generated_groups,
        candidate_optimization_effective=scalar_effective,
        candidate_raw_correctness_mixed=raw_mixed,
        candidate_extraction_only_effective=scalar_effective - raw_mixed,
        candidate_generated_response_tokens=response_tokens,
        topup_groups=topup_groups,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    step_match = re.fullmatch(r"global_step_(\d+)", args.checkpoint.name)
    if step_match is None:
        raise ValueError("checkpoint path must end in global_step_<N>")
    checkpoint_step = int(step_match.group(1))
    counters = build_step50_counters(args.log, expected_step=checkpoint_step)
    destination = args.checkpoint / HIVE_SIGNAL_COUNTERS_FILENAME
    print(counters.to_dict())
    if args.write:
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing counter sidecar: {destination}")
        counters.save_checkpoint(args.checkpoint)
        print(f"wrote {destination}")
    else:
        print(f"dry run; would write {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
