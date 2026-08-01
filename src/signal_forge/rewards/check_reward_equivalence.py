"""Check RewardScope verifier and veRL reward adapter equivalence."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from signal_forge.rewards.math_verify_adapter import compute_score

_SIGNAL_FORGE_SRC = Path(os.environ.get("SIGNAL_FORGE_SRC", Path(__file__).resolve().parents[2]))
_REWARDSCOPE_OUTPUTS = _SIGNAL_FORGE_SRC / "RewardScope" / "outputs"

DEFAULT_GSM8K_ROLLOUTS = _REWARDSCOPE_OUTPUTS / "gsm8k-qwen-grpo-train-zero-shot-boxed-128" / "rollouts.jsonl"
DEFAULT_MATH_ROLLOUTS = _REWARDSCOPE_OUTPUTS / "math-qwen-grpo-train-level-3-64-max768" / "rollouts.jsonl"
LOCAL_GSM8K_ROLLOUTS = DEFAULT_GSM8K_ROLLOUTS
LOCAL_MATH_ROLLOUTS = DEFAULT_MATH_ROLLOUTS


def _existing(default_path: Path, local_path: Path) -> Path:
    if default_path.exists():
        return default_path
    return local_path


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _source(record: dict[str, Any]) -> str:
    dataset_name = record.get("dataset_name")
    if dataset_name == "gsm8k":
        return "gsm8k"
    if dataset_name == "math":
        return "math_level_3"
    raise ValueError(f"Unsupported rollout dataset_name: {dataset_name!r}")


def _direct_reward(record: dict[str, Any]) -> dict[str, Any]:
    from rewardscope.verification.math_verify import MathVerifyLatexVerifier, MathVerifyNumericVerifier

    source = _source(record)
    if source == "gsm8k":
        result = MathVerifyNumericVerifier(mode="training").verify(
            response=record["response"], ground_truth=record["ground_truth"]
        )
    else:
        result = MathVerifyLatexVerifier(mode="training").verify(
            response=record["response"], ground_truth=record["ground_truth"]
        )
    extraction = result.extraction
    score = float(result.is_correct)
    return {
        "score": score,
        "raw_correctness": score,
        "extraction_ok": bool(extraction.extraction_ok),
        "format_ok": bool(extraction.format_ok),
        "verification_status": extraction.extraction_status.value,
        "verification_error_type": result.error_type or "",
    }


def _adapter_reward(record: dict[str, Any]) -> dict[str, Any]:
    return compute_score(
        data_source=_source(record),
        solution_str=record["response"],
        ground_truth=record["ground_truth"],
        extra_info={"prompt_id": record.get("prompt_id"), "source_dataset": record.get("dataset_name")},
    )


def _category(record: dict[str, Any]) -> set[str]:
    verification = record.get("verification") or {}
    extraction = verification.get("extraction") or {}
    status = str(extraction.get("extraction_status") or "")
    categories = {_source(record)}
    categories.add("correct" if bool(verification.get("is_correct")) else "wrong")
    if status in {"missing", "parse_error", "ambiguous"}:
        categories.add("missing_or_parse")
    if bool(record.get("hit_max_length")) or record.get("finish_reason") == "length":
        categories.add("truncated")
    if len(extraction.get("valid_candidates") or []) > 1:
        categories.add("multi_value")
    return categories


def _pick_diverse(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    wanted = ["gsm8k", "math_level_3", "correct", "wrong", "missing_or_parse", "truncated", "multi_value"]
    picked: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for category in _category(record):
            buckets[category].append(record)

    for category in wanted:
        for record in buckets.get(category, [])[:8]:
            key = (record["prompt_id"], int(record["sample_id"]))
            if key not in seen_keys:
                picked.append(record)
                seen_keys.add(key)

    for record in records:
        if len(picked) >= limit:
            break
        key = (record["prompt_id"], int(record["sample_id"]))
        if key not in seen_keys:
            picked.append(record)
            seen_keys.add(key)
    return picked[:limit]


def _compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key in [
        "score",
        "raw_correctness",
        "extraction_ok",
        "format_ok",
        "verification_status",
        "verification_error_type",
    ]:
        if expected[key] != actual[key]:
            mismatches.append(f"{key}: expected={expected[key]!r} actual={actual[key]!r}")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k-rollouts", type=Path, default=_existing(DEFAULT_GSM8K_ROLLOUTS, LOCAL_GSM8K_ROLLOUTS))
    parser.add_argument("--math-rollouts", type=Path, default=_existing(DEFAULT_MATH_ROLLOUTS, LOCAL_MATH_ROLLOUTS))
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--require-categories", action="store_true")
    args = parser.parse_args()

    records = list(_iter_jsonl(args.gsm8k_rollouts)) + list(_iter_jsonl(args.math_rollouts))
    if len(records) < 100:
        raise AssertionError(f"Need at least 100 saved rollouts, found {len(records)}")

    picked = _pick_diverse(records, max(args.limit, 100))
    category_counts = Counter(category for record in picked for category in _category(record))

    if args.require_categories:
        required = {"gsm8k", "math_level_3", "correct", "wrong", "missing_or_parse", "truncated"}
        missing = sorted(category for category in required if category_counts[category] == 0)
        if missing:
            raise AssertionError(f"Selected rollouts miss required categories: {missing}; counts={dict(category_counts)}")

    mismatches: list[str] = []
    start = time.perf_counter()
    for index, record in enumerate(picked):
        expected = _direct_reward(record)
        actual = _adapter_reward(record)
        diff = _compare(expected, actual)
        if diff:
            mismatches.append(
                f"#{index} {record['prompt_id']} sample={record['sample_id']} source={_source(record)}: "
                + "; ".join(diff)
            )
    elapsed = time.perf_counter() - start

    if mismatches:
        preview = "\n".join(mismatches[:20])
        raise AssertionError(f"Reward equivalence failed for {len(mismatches)} rows:\n{preview}")

    print(
        json.dumps(
            {
                "checked": len(picked),
                "elapsed_sec": round(elapsed, 3),
                "rows_per_sec": round(len(picked) / elapsed, 3) if elapsed else None,
                "categories": dict(sorted(category_counts.items())),
                "gsm8k_rollouts": str(args.gsm8k_rollouts),
                "math_rollouts": str(args.math_rollouts),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
