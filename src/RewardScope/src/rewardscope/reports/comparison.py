"""Offline, prompt-aligned comparisons between two persisted rollout files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rewardscope.io import read_rollouts_jsonl
from rewardscope.io.atomic import atomic_write_json, atomic_write_jsonl


@dataclass(frozen=True)
class RolloutComparisonArtifacts:
    """Files produced by an offline prompt-aligned rollout comparison."""

    output_dir: Path
    sample_comparison_jsonl: Path
    summary_json: Path


def compare_rollouts_jsonl(
    baseline_path: str | Path,
    candidate_path: str | Path,
    output_dir: str | Path,
) -> RolloutComparisonArtifacts:
    """Compare candidate samples against matching baseline prompt/sample pairs.

    The candidate file may contain a subset of baseline prompts. Both files must
    contain at most one row per ``(prompt_id, sample_id)`` pair.
    """
    baseline_rows = read_rollouts_jsonl(baseline_path)
    candidate_rows = read_rollouts_jsonl(candidate_path)
    if not candidate_rows:
        raise ValueError("Candidate rollout file is empty.")
    destination = Path(output_dir)
    _require_empty_destination(destination)
    destination.mkdir(parents=True)

    baseline_by_key = _index_rows(baseline_rows, "baseline")
    candidate_by_key = _index_rows(candidate_rows, "candidate")
    comparisons: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    for candidate in candidate_rows:
        key = _row_key(candidate)
        baseline = baseline_by_key.get(key)
        if baseline is None:
            raise ValueError(f"Candidate sample {key} is missing from the baseline rollout file.")
        if _require_str(baseline, "ground_truth") != _require_str(candidate, "ground_truth"):
            raise ValueError(f"Ground truth differs for prompt/sample pair {key}.")
        comparison = _comparison_row(baseline, candidate)
        comparisons.append(comparison)
        transitions[comparison["correctness_transition"]] += 1

    summary = {
        "sample_count": len(comparisons),
        "baseline_accuracy": _accuracy(baseline_by_key, candidate_by_key),
        "candidate_accuracy": _accuracy(candidate_by_key, candidate_by_key),
        "correctness_transition_counts": dict(sorted(transitions.items())),
        "baseline_response_tokens_total": sum(
            row["baseline_response_tokens"] for row in comparisons
        ),
        "candidate_response_tokens_total": sum(
            row["candidate_response_tokens"] for row in comparisons
        ),
        "baseline_response_tokens_mean": sum(
            row["baseline_response_tokens"] for row in comparisons
        ) / len(comparisons),
        "candidate_response_tokens_mean": sum(
            row["candidate_response_tokens"] for row in comparisons
        ) / len(comparisons),
    }
    summary["response_tokens_delta_total"] = (
        summary["candidate_response_tokens_total"]
        - summary["baseline_response_tokens_total"]
    )

    samples_path = destination / "sample_comparison.jsonl"
    summary_path = destination / "summary.json"
    atomic_write_jsonl(samples_path, comparisons)
    atomic_write_json(summary_path, summary)
    return RolloutComparisonArtifacts(destination, samples_path, summary_path)


def _index_rows(rows: list[dict[str, Any]], label: str) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        if key in indexed:
            raise ValueError(f"{label} rollout file contains duplicate prompt/sample pair {key}.")
        indexed[key] = row
    return indexed


def _comparison_row(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_correct = _raw_correctness(baseline)
    candidate_correct = _raw_correctness(candidate)
    return {
        "prompt_id": _require_str(candidate, "prompt_id"),
        "sample_id": _require_non_negative_int(candidate, "sample_id"),
        "ground_truth": _require_str(candidate, "ground_truth"),
        "baseline_raw_correctness": baseline_correct,
        "candidate_raw_correctness": candidate_correct,
        "correctness_transition": _correctness_transition(baseline_correct, candidate_correct),
        "baseline_response_tokens": _require_non_negative_int(baseline, "response_tokens"),
        "candidate_response_tokens": _require_non_negative_int(candidate, "response_tokens"),
        "response_tokens_delta": (
            _require_non_negative_int(candidate, "response_tokens")
            - _require_non_negative_int(baseline, "response_tokens")
        ),
        "baseline_extraction_status": _extraction_status(baseline),
        "candidate_extraction_status": _extraction_status(candidate),
        "baseline_response": _require_str(baseline, "response"),
        "candidate_response": _require_str(candidate, "response"),
    }


def _accuracy(
    rows: dict[tuple[str, int], dict[str, Any]],
    selected_rows: dict[tuple[str, int], dict[str, Any]],
) -> float:
    return sum(_raw_correctness(rows[key]) for key in selected_rows) / len(selected_rows)


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return _require_str(row, "prompt_id"), _require_non_negative_int(row, "sample_id")


def _raw_correctness(row: dict[str, Any]) -> bool:
    verification = _require_mapping(row.get("verification"), "verification")
    value = verification.get("is_correct")
    if not isinstance(value, bool):
        raise ValueError("verification.is_correct must be a boolean.")
    return value


def _extraction_status(row: dict[str, Any]) -> str:
    verification = _require_mapping(row.get("verification"), "verification")
    extraction = _require_mapping(verification.get("extraction"), "verification.extraction")
    return _require_str(extraction, "extraction_status")


def _correctness_transition(baseline: bool, candidate: bool) -> str:
    if baseline and candidate:
        return "correct_to_correct"
    if baseline and not candidate:
        return "correct_to_incorrect"
    if not baseline and candidate:
        return "incorrect_to_correct"
    return "incorrect_to_incorrect"


def _require_empty_destination(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"Comparison destination must be absent or empty: {path}")
        path.rmdir()


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _require_str(row: dict[str, Any], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    return value


def _require_non_negative_int(row: dict[str, Any], name: str) -> int:
    value = row.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value
