"""Build the frozen six-benchmark validation parquet for formal A/B runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from rewardscope.io.atomic import atomic_write_json
from signal_forge.calibration.hive_dataset import format_canonical_math_prompt
from signal_forge.data.validation_decontamination import (
    EXPECTED_BENCHMARK_COUNTS,
    MATH500_REVISION,
    QWEN_EVAL_REVISION,
    ValidationProblem,
    load_validation_suite,
)
from verl.utils.tokenizer.chat_template import preprocess_chat_prompt


FORMAL_VALIDATION_FILENAME = "formal_validation.parquet"
FORMAL_VALIDATION_MANIFEST = "formal_validation_manifest.json"
EXPECTED_TOTAL = sum(EXPECTED_BENCHMARK_COUNTS.values())


def build_formal_validation_rows(
    problems: Sequence[ValidationProblem],
) -> list[dict[str, Any]]:
    identities = [(item.benchmark, item.benchmark_id) for item in problems]
    if len(identities) != len(set(identities)):
        raise ValueError("formal validation contains duplicate benchmark-qualified IDs")

    counts = Counter(item.benchmark for item in problems)
    expected = Counter(EXPECTED_BENCHMARK_COUNTS)
    if counts != expected:
        raise ValueError(f"formal validation benchmark counts differ: {dict(counts)} != {dict(expected)}")

    rows: list[dict[str, Any]] = []
    for problem in problems:
        canonical = format_canonical_math_prompt(problem.question)
        rows.append(
            {
                "data_source": problem.benchmark,
                "ability": "math",
                "prompt": [{"role": "user", "content": canonical}],
                "reward_model": {"style": "rule", "ground_truth": problem.ground_truth},
                "extra_info": {
                    "prompt_id": f"validation:{problem.benchmark}:{problem.benchmark_id}",
                    "dataset_source": problem.benchmark,
                    "source_dataset": "math",
                    "benchmark": problem.benchmark,
                    "benchmark_id": problem.benchmark_id,
                    "benchmark_row_index": int(problem.row_index),
                    "formal_partition": "validation",
                    "canonical_prompt": canonical,
                    "raw_question": problem.raw_question,
                    "source_answer_json": json.dumps(problem.source_answer, ensure_ascii=True),
                    "source_ground_truth": problem.ground_truth,
                    "is_multiple_answer": bool(problem.is_multiple_answer),
                    "multiple_answers_json": json.dumps(
                        list(problem.multiple_answers), ensure_ascii=True
                    ),
                },
            }
        )
    prompt_ids = [row["extra_info"]["prompt_id"] for row in rows]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise AssertionError("formal validation stable prompt IDs are not unique")
    return rows


def scan_prompt_lengths(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer,
) -> dict[str, Any]:
    values: list[tuple[int, str, str]] = []
    for row in rows:
        extra = row["extra_info"]
        canonical = preprocess_chat_prompt(
            tokenizer,
            list(row["prompt"]),
            max_prompt_length=None,
        )
        values.append(
            (
                int(canonical.untruncated_token_count),
                str(extra["benchmark"]),
                str(extra["benchmark_id"]),
            )
        )
    by_source: dict[str, list[int]] = {}
    for length, benchmark, _ in values:
        by_source.setdefault(benchmark, []).append(length)
    return {
        "overall": _length_summary([item[0] for item in values]),
        "by_benchmark": {
            benchmark: _length_summary(lengths)
            for benchmark, lengths in sorted(by_source.items())
        },
        "longest_prompts": [
            {"prompt_token_count": length, "benchmark": benchmark, "benchmark_id": benchmark_id}
            for length, benchmark, benchmark_id in sorted(
                values, key=lambda item: (-item[0], item[1], item[2])
            )[:25]
        ],
    }


def write_formal_validation(
    output_dir: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    length_report: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    parquet_path = destination / FORMAL_VALIDATION_FILENAME
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    round_trip = pd.read_parquet(parquet_path)
    if len(round_trip) != EXPECTED_TOTAL:
        raise AssertionError(f"formal validation parquet row count mismatch: {len(round_trip)}")
    source_counts = Counter(str(value) for value in round_trip["data_source"].tolist())
    if source_counts != Counter(EXPECTED_BENCHMARK_COUNTS):
        raise AssertionError("formal validation parquet source counts changed during round trip")

    manifest = {
        "schema_version": 1,
        "row_count": len(rows),
        "benchmark_counts": dict(sorted(source_counts.items())),
        "qwen_eval_revision": QWEN_EVAL_REVISION,
        "math500_revision": MATH500_REVISION,
        "parquet_filename": FORMAL_VALIDATION_FILENAME,
        "parquet_sha256": _sha256(parquet_path),
        "prompt_template": format_canonical_math_prompt("{problem}"),
        "stable_id_format": "validation:<benchmark>:<benchmark_id>",
        "multiple_answer_semantics": "source-declared top-level answers are an unordered set",
        "prompt_token_lengths": dict(length_report or {}),
    }
    manifest_path = destination / FORMAL_VALIDATION_MANIFEST
    atomic_write_json(manifest_path, manifest)
    return parquet_path, manifest_path


def _length_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise ValueError("prompt-length summary requires values")
    ordered = sorted(int(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": _nearest_rank(ordered, 0.50),
        "p90": _nearest_rank(ordered, 0.90),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "max": ordered[-1],
    }


def _nearest_rank(ordered: Sequence[int], quantile: float) -> int:
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    problems = load_validation_suite(args.validation_data_dir)
    rows = build_formal_validation_rows(problems)
    length_report: dict[str, Any] | None = None
    if args.model_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        length_report = scan_prompt_lengths(rows, tokenizer=tokenizer)
    parquet_path, manifest_path = write_formal_validation(
        args.output_dir,
        rows=rows,
        length_report=length_report,
    )
    print(f"wrote {len(rows)} validation prompts to {parquet_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
