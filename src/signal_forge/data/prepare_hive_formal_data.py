"""Build the maximum exact 3:1 MATH/DAPO formal pool and scan prompt lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from rewardscope.io.atomic import atomic_write_json, atomic_write_jsonl
from rewardscope.verification import extract_final_boxed_latex_gold
from signal_forge.calibration.hive_dataset import (
    CalibrationPrompt,
    load_math_calibration_prompts,
    normalize_dapo_row,
    select_fixed_subset,
)
from signal_forge.data.validation_decontamination import (
    MATH500_REVISION,
    QWEN_EVAL_REVISION,
    audit_training_pool,
    load_review_config,
    load_validation_suite,
    remove_confirmed_overlaps,
    semantic_tokens,
)
from verl.utils.tokenizer.chat_template import preprocess_chat_prompt


DEFAULT_SEED = 42
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
DAPO_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
DAPO_SNAPSHOT_RELATIVE_PATH = Path("data/dapo-math-17k.parquet")
DAPO_PARQUET_SHA256 = "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941"
DAPO_UNIQUE_PROMPTS = 17_917
DAPO_REPEAT_FACTOR = 100
LENGTH_THRESHOLDS = (256, 384, 512, 640, 768, 1024, 1280, 1536)


def maximum_exact_ratio_counts(
    math_available: int,
    dapo_available: int,
    *,
    math_weight: int = 3,
    dapo_weight: int = 1,
) -> tuple[int, int]:
    """Return the largest source counts satisfying an exact integer ratio."""
    for name, value in (
        ("math_available", math_available),
        ("dapo_available", dapo_available),
        ("math_weight", math_weight),
        ("dapo_weight", dapo_weight),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if math_weight <= 0 or dapo_weight <= 0:
        raise ValueError("ratio weights must be positive")
    units = min(math_available // math_weight, dapo_available // dapo_weight)
    if units <= 0:
        raise ValueError("available prompts cannot form one complete ratio unit")
    return units * math_weight, units * dapo_weight


def validate_dapo_ground_truths(
    prompts: Sequence[CalibrationPrompt],
) -> tuple[tuple[CalibrationPrompt, ...], tuple[str, ...]]:
    """Validate every DAPO gold under the frozen boxed-LaTeX contract."""
    valid: list[CalibrationPrompt] = []
    rejected: list[str] = []
    for prompt in prompts:
        parsed = extract_final_boxed_latex_gold(prompt.ground_truth)
        if parsed is None:
            rejected.append(prompt.prompt_id)
        else:
            valid.append(replace(prompt, ground_truth=parsed))
    return tuple(valid), tuple(rejected)


def load_dapo_prompts_streaming(
    parquet_path: str | Path,
    *,
    batch_size: int = 1024,
) -> tuple[tuple[CalibrationPrompt, ...], dict[str, int]]:
    """Read the anomalously repeated DAPO parquet with bounded memory.

    Every duplicate is fingerprinted and compared so conflicting source rows
    still fail rather than being silently discarded.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"DAPO parquet not found: {path}")

    import pyarrow.parquet as pq

    by_id: dict[str, CalibrationPrompt] = {}
    fingerprints: dict[str, bytes] = {}
    source_rows = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            source_rows += 1
            row_id = _dapo_row_id(row)
            fingerprint = _row_fingerprint(row)
            previous = fingerprints.get(row_id)
            if previous is not None:
                if previous != fingerprint:
                    raise ValueError(f"DAPO duplicate row id has conflicting content: dapo:{row_id}")
                continue
            prompt = normalize_dapo_row(row)
            fingerprints[row_id] = fingerprint
            by_id[row_id] = prompt

    prompts = tuple(by_id[row_id] for row_id in sorted(by_id))
    return prompts, {
        "source_rows": source_rows,
        "unique_prompts": len(prompts),
        "duplicate_rows_removed": source_rows - len(prompts),
    }


def load_frozen_dapo_repeated_snapshot(
    parquet_path: str | Path,
    *,
    expected_sha256: str = DAPO_PARQUET_SHA256,
    unique_prompts: int = DAPO_UNIQUE_PROMPTS,
    repeat_factor: int = DAPO_REPEAT_FACTOR,
    batch_size: int = 1024,
) -> tuple[tuple[CalibrationPrompt, ...], dict[str, int]]:
    """Load the frozen DAPO snapshot without materializing 100 text copies."""
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"DAPO parquet not found: {path}")
    if _file_sha256(path) != expected_sha256:
        raise ValueError("DAPO parquet SHA-256 does not match the frozen audited snapshot")
    if unique_prompts <= 0 or repeat_factor <= 0:
        raise ValueError("unique_prompts and repeat_factor must be positive")

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path, pre_buffer=False)
    expected_rows = unique_prompts * repeat_factor
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"DAPO repeated snapshot row count mismatch: {parquet.metadata.num_rows} != "
            f"{expected_rows}"
        )
    first_cycle: list[str] = []
    source_index = 0
    for batch in parquet.iter_batches(
        batch_size=max(batch_size, 65_536),
        columns=["extra_info"],
        use_threads=False,
    ):
        for row in batch.to_pylist():
            row_id = _dapo_row_id(row)
            if source_index < unique_prompts:
                first_cycle.append(row_id)
            elif row_id != first_cycle[source_index % unique_prompts]:
                raise ValueError(
                    "DAPO repeated snapshot stable-ID cycles differ at physical row "
                    f"{source_index}"
                )
            source_index += 1
    if len(first_cycle) != len(set(first_cycle)):
        raise ValueError("DAPO first repeated cycle contains duplicate stable IDs")

    prompts: list[CalibrationPrompt] = []
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["prompt", "reward_model", "extra_info"],
        use_threads=False,
    ):
        for row in batch.to_pylist():
            prompts.append(normalize_dapo_row(row))
            if len(prompts) == unique_prompts:
                break
        if len(prompts) == unique_prompts:
            break
    if [prompt.source_row_id for prompt in prompts] != first_cycle:
        raise AssertionError("DAPO materialized first cycle does not match audited ID cycle")
    prompts.sort(key=lambda prompt: prompt.source_row_id)
    return tuple(prompts), {
        "source_rows": expected_rows,
        "unique_prompts": unique_prompts,
        "duplicate_rows_removed": expected_rows - unique_prompts,
        "verified_repeat_factor": repeat_factor,
    }


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def select_maximum_exact_three_to_one(
    math_prompts: Sequence[CalibrationPrompt],
    dapo_prompts: Sequence[CalibrationPrompt],
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[CalibrationPrompt, ...]:
    math_count, dapo_count = maximum_exact_ratio_counts(
        len(math_prompts), len(dapo_prompts)
    )
    selected_math = select_fixed_subset(math_prompts, sample_size=math_count, seed=seed)
    selected_dapo = select_fixed_subset(dapo_prompts, sample_size=dapo_count, seed=seed)
    selected = [*selected_math, *selected_dapo]
    random.Random(seed).shuffle(selected)
    ids = [prompt.prompt_id for prompt in selected]
    if len(ids) != len(set(ids)):
        raise AssertionError("formal pool contains duplicate stable prompt IDs")
    return tuple(selected)


def scan_prompt_token_lengths(
    prompts: Sequence[CalibrationPrompt],
    token_count: Callable[[CalibrationPrompt], int],
) -> tuple[dict[str, int], dict[str, Any]]:
    if not prompts:
        raise ValueError("prompt length scan requires at least one prompt")
    counts: dict[str, int] = {}
    for prompt in prompts:
        count = token_count(prompt)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"invalid token count for {prompt.prompt_id}: {count!r}")
        counts[prompt.prompt_id] = count
    if len(counts) != len(prompts):
        raise ValueError("prompt length scan received duplicate stable prompt IDs")

    report = {
        "overall": _length_summary([counts[prompt.prompt_id] for prompt in prompts]),
        "by_dataset_source": {
            source: _length_summary(
                [counts[prompt.prompt_id] for prompt in prompts if prompt.dataset_source == source]
            )
            for source in sorted({prompt.dataset_source for prompt in prompts})
        },
        "longest_prompts": [
            {
                "prompt_id": prompt.prompt_id,
                "dataset_source": prompt.dataset_source,
                "prompt_token_count": counts[prompt.prompt_id],
            }
            for prompt in sorted(
                prompts,
                key=lambda item: (-counts[item.prompt_id], item.prompt_id),
            )[:25]
        ],
    }
    return counts, report


def build_verl_rows(
    prompts: Sequence[CalibrationPrompt],
    prompt_token_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        rows.append(
            {
                # Both sources use the shared math verifier/reward adapter.
                "data_source": "math",
                "ability": "math",
                "prompt": list(prompt.messages),
                "reward_model": {"style": "rule", "ground_truth": prompt.ground_truth},
                "extra_info": {
                    "prompt_id": prompt.prompt_id,
                    "dataset_source": prompt.dataset_source,
                    "source_row_id": prompt.source_row_id,
                    "source_dataset": "math",
                    "source_dataset_id": (
                        "EleutherAI/hendrycks_math"
                        if prompt.dataset_source == "math"
                        else "BytedTsinghua-SIA/DAPO-Math-17k"
                    ),
                    "dataset_revision": (
                        MATH_REVISION if prompt.dataset_source == "math" else DAPO_REVISION
                    ),
                    "split": "train",
                    "formal_partition": "train",
                    "canonical_prompt": prompt.canonical_prompt,
                    "raw_prompt_json": json.dumps(list(prompt.raw_prompt), ensure_ascii=True),
                    "source_ground_truth": prompt.source_ground_truth,
                    "prompt_token_count": int(prompt_token_counts[prompt.prompt_id]),
                },
            }
        )
    return rows


def write_formal_pool(
    output_dir: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    length_report: Mapping[str, Any],
    decontamination_records: Sequence[Mapping[str, Any]] = (),
    decontamination_summary: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    parquet_path = destination / "train.parquet"
    frame.to_parquet(parquet_path, index=False)
    if len(pd.read_parquet(parquet_path, columns=["data_source"])) != len(rows):
        raise AssertionError("formal parquet round-trip row count mismatch")
    atomic_write_json(destination / "selection_manifest.json", dict(manifest))
    atomic_write_json(destination / "prompt_token_length_report.json", dict(length_report))
    if decontamination_records:
        atomic_write_jsonl(
            destination / "decontamination_manifest.jsonl",
            [dict(record) for record in decontamination_records],
        )
    if decontamination_summary is not None:
        atomic_write_json(
            destination / "decontamination_summary.json",
            dict(decontamination_summary),
        )


def _length_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise ValueError("length summary requires values")
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    return {
        "count": count,
        "min": ordered[0],
        "mean": mean,
        "p50": _nearest_rank(ordered, 0.50),
        "p75": _nearest_rank(ordered, 0.75),
        "p90": _nearest_rank(ordered, 0.90),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "p99_5": _nearest_rank(ordered, 0.995),
        "p99_9": _nearest_rank(ordered, 0.999),
        "max": ordered[-1],
        "thresholds": {
            str(threshold): {
                "count_above": sum(value > threshold for value in ordered),
                "ratio_above": sum(value > threshold for value in ordered) / count,
            }
            for threshold in LENGTH_THRESHOLDS
        },
    }


def _nearest_rank(ordered: Sequence[int], quantile: float) -> int:
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _dapo_row_id(row: Mapping[str, Any]) -> str:
    extra_info = row.get("extra_info")
    if not isinstance(extra_info, Mapping):
        raise ValueError("DAPO row is missing extra_info")
    row_id = extra_info.get("index")
    if row_id is None or not str(row_id).strip():
        raise ValueError("DAPO row is missing stable extra_info.index")
    return str(row_id).strip()


def _row_fingerprint(row: Mapping[str, Any]) -> bytes:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _resolve_default_dapo_parquet() -> Path:
    return (
        Path.home()
        / ".cache/huggingface/hub"
        / "datasets--BytedTsinghua-SIA--DAPO-Math-17k"
        / "snapshots"
        / DAPO_REVISION
        / DAPO_SNAPSHOT_RELATIVE_PATH
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the maximum validated exact 3:1 MATH/DAPO formal pool."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/formal_data/"
            "hive_math75_dapo25_seed42_validation_clean_max_exact_3to1"
        ),
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dapo-parquet", type=Path, default=_resolve_default_dapo_parquet())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dapo-batch-size", type=int, default=1024)
    parser.add_argument(
        "--validation-data-dir",
        type=Path,
        default=Path(
            "artifacts/validation_data/"
            "qwen_math_a45202bd_math500_6e4ed1a2"
        ),
    )
    parser.add_argument(
        "--decontamination-review",
        type=Path,
        default=Path("docs/hive/HIVE_VALIDATION_DECONTAMINATION_REVIEW.json"),
    )
    return parser


def _cross_benchmark_duplicates(
    validation_problems: Sequence[Any], *, left: str, right: str
) -> list[dict[str, str]]:
    left_by_text: dict[tuple[str, ...], list[str]] = {}
    for problem in validation_problems:
        if problem.benchmark == left:
            left_by_text.setdefault(semantic_tokens(problem.question), []).append(
                problem.benchmark_id
            )
    pairs: list[dict[str, str]] = []
    for problem in validation_problems:
        if problem.benchmark != right:
            continue
        for left_id in left_by_text.get(semantic_tokens(problem.question), []):
            pairs.append({left: left_id, right: problem.benchmark_id})
    return sorted(pairs, key=lambda item: (item[left], item[right]))


def _legacy_exact_review(
    review_config: Mapping[str, Any], records: Sequence[Any]
) -> list[dict[str, Any]]:
    by_id: dict[str, list[Any]] = {}
    for record in records:
        by_id.setdefault(record.train_prompt_id, []).append(record)
    result: list[dict[str, Any]] = []
    for prompt_id in review_config.get("legacy_reported_exact_train_ids", []):
        matches = by_id.get(str(prompt_id), [])
        removed = any(record.decision == "remove" for record in matches)
        result.append(
            {
                "train_prompt_id": str(prompt_id),
                "decision": "remove" if removed else "keep",
                "match_types": sorted({record.match_type for record in matches}),
                "benchmark_matches": [
                    {
                        "benchmark": record.benchmark,
                        "benchmark_id": record.benchmark_id,
                        "match_type": record.match_type,
                        "similarity": record.similarity,
                        "decision": record.decision,
                        "reason": record.reason,
                    }
                    for record in matches
                ],
                "reason": (
                    "confirmed validation overlap"
                    if removed
                    else "prior operator-dropping normalization false positive; operator-preserving audit keeps it"
                ),
            }
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed < 0:
        raise ValueError("seed must be non-negative")

    dapo_prompts, dapo_metadata = load_frozen_dapo_repeated_snapshot(
        args.dapo_parquet,
        batch_size=args.dapo_batch_size,
    )
    math_prompts, math_metadata = load_math_calibration_prompts(split="train")
    valid_dapo, rejected_dapo_ids = validate_dapo_ground_truths(dapo_prompts)
    validation_problems = load_validation_suite(args.validation_data_dir)
    review_config = load_review_config(args.decontamination_review)
    source_prompts = (*math_prompts, *valid_dapo)
    decontamination_records = audit_training_pool(
        source_prompts,
        validation_problems,
        review_config,
    )
    clean_prompts, removed_prompt_ids = remove_confirmed_overlaps(
        source_prompts,
        decontamination_records,
    )
    clean_math = tuple(prompt for prompt in clean_prompts if prompt.dataset_source == "math")
    clean_dapo = tuple(prompt for prompt in clean_prompts if prompt.dataset_source == "dapo")
    selected = select_maximum_exact_three_to_one(clean_math, clean_dapo, seed=args.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    def count_tokens(prompt: CalibrationPrompt) -> int:
        canonical = preprocess_chat_prompt(
            tokenizer,
            list(prompt.messages),
            max_prompt_length=None,
        )
        return canonical.untruncated_token_count

    prompt_token_counts, length_report = scan_prompt_token_lengths(selected, count_tokens)
    rows = build_verl_rows(selected, prompt_token_counts)
    source_counts = Counter(prompt.dataset_source for prompt in selected)
    if source_counts["math"] != 3 * source_counts["dapo"]:
        raise AssertionError("formal pool is not exact 3:1 MATH:DAPO")

    exact_remove_ids = {
        record.train_prompt_id
        for record in decontamination_records
        if record.match_type == "A" and record.decision == "remove"
    }
    near_remove_ids = {
        record.train_prompt_id
        for record in decontamination_records
        if record.match_type == "B" and record.decision == "remove"
    } - exact_remove_ids
    if set(removed_prompt_ids) != exact_remove_ids | near_remove_ids:
        raise AssertionError("decontamination removal accounting mismatch")
    removed_by_source = Counter(prompt_id.split(":", 1)[0] for prompt_id in removed_prompt_ids)
    candidate_benchmark_rows = {
        (record.benchmark, record.benchmark_id) for record in decontamination_records
    }
    validation_cross_duplicates = _cross_benchmark_duplicates(
        validation_problems,
        left="amc23",
        right="gaokao2023en",
    )
    declared_cross_duplicate_count = 8
    legacy_exact_review = _legacy_exact_review(
        review_config,
        decontamination_records,
    )
    decontamination_summary = {
        "schema_version": 1,
        "validation_policy": "validation rows are immutable; remove A/B matches from training only",
        "candidate_generation": {
            "normalized_word_ngram_size": int(review_config["ngram_size"]),
            "candidate_pairs": len(decontamination_records),
            "candidate_benchmark_rows": len(candidate_benchmark_rows),
            "manual_review_lcs_threshold": review_config["manual_review_lcs_threshold"],
            "manual_review_char_threshold": review_config["manual_review_char_threshold"],
            "manual_review_pairs": sum(
                record.similarity >= float(review_config["manual_review_lcs_threshold"])
                or record.normalized_char_similarity
                >= float(review_config["manual_review_char_threshold"])
                for record in decontamination_records
            ),
        },
        "pair_match_type_counts": dict(
            Counter(record.match_type for record in decontamination_records)
        ),
        "removed_unique_prompt_counts": {
            "exact_A": len(exact_remove_ids),
            "near_duplicate_B_excluding_A": len(near_remove_ids),
            "total": len(removed_prompt_ids),
            "by_dataset_source": dict(removed_by_source),
        },
        "removed_prompt_ids": list(removed_prompt_ids),
        "legacy_selected_pool_exact9_review": legacy_exact_review,
        "validation_cross_benchmark_duplicates_retained": {
            "benchmarks": ["amc23", "gaokao2023en"],
            "declared_count": declared_cross_duplicate_count,
            "observed_count": len(validation_cross_duplicates),
            "count_conflict": len(validation_cross_duplicates) != declared_cross_duplicate_count,
            "pairs": validation_cross_duplicates,
            "reporting": "retain both suites; report each and the preregistered six-suite average",
        },
        "clean_source_counts": {"math": len(clean_math), "dapo": len(clean_dapo)},
        "selected_counts": dict(source_counts),
        "selected_total": len(selected),
        "selected_stable_prompt_id_unique": len({p.prompt_id for p in selected}) == len(selected),
    }

    manifest = {
        "schema_version": 2,
        "seed": args.seed,
        "ratio": {"math": 3, "dapo": 1},
        "selection_rule": (
            "remove confirmed validation A/B overlaps from complete validated source pools, "
            "then select the maximum exact integer 3:1 ratio without prompt-length filtering"
        ),
        "tokenization": {
            "model_path": str(args.model_path),
            "canonical_rollout_helper": "verl.utils.tokenizer.chat_template.preprocess_chat_prompt",
            "add_generation_prompt": True,
            "truncation": None,
        },
        "validation_snapshot": {
            "path": str(args.validation_data_dir),
            "qwen_math_revision": QWEN_EVAL_REVISION,
            "math500_revision": MATH500_REVISION,
            "benchmark_counts": dict(Counter(p.benchmark for p in validation_problems)),
            "rows_removed_from_validation": 0,
        },
        "decontamination": {
            "review_config": str(args.decontamination_review),
            "manifest": "decontamination_manifest.jsonl",
            "summary": "decontamination_summary.json",
            "removed_exact_unique_prompts": len(exact_remove_ids),
            "removed_near_duplicate_unique_prompts": len(near_remove_ids),
        },
        "sources": {
            "math": {
                "dataset_id": "EleutherAI/hendrycks_math",
                "revision": MATH_REVISION,
                "split": "train",
                "source_rows": math_metadata["source_rows"],
                "gold_parse_failures": math_metadata["gold_parse_failures"],
                "validated_unique_prompts": len(math_prompts),
                "removed_validation_overlaps": removed_by_source["math"],
                "clean_unique_prompts": len(clean_math),
                "selected_prompts": source_counts["math"],
            },
            "dapo": {
                "dataset_id": "BytedTsinghua-SIA/DAPO-Math-17k",
                "revision": DAPO_REVISION,
                "split": "train",
                **dapo_metadata,
                "gold_parse_failures": len(rejected_dapo_ids),
                "rejected_prompt_ids": list(rejected_dapo_ids),
                "validated_unique_prompts": len(valid_dapo),
                "removed_validation_overlaps": removed_by_source["dapo"],
                "clean_unique_prompts": len(clean_dapo),
                "selected_prompts": source_counts["dapo"],
            },
        },
        "total_selected_prompts": len(selected),
        "prompt_ids": [prompt.prompt_id for prompt in selected],
    }
    write_formal_pool(
        args.output_dir,
        rows=rows,
        manifest=manifest,
        length_report=length_report,
        decontamination_records=[record.to_dict() for record in decontamination_records],
        decontamination_summary=decontamination_summary,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "source_counts": dict(source_counts),
                "total_selected_prompts": len(selected),
                "decontamination": decontamination_summary,
                "prompt_lengths": length_report,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
