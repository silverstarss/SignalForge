"""Offline numeric-extractor rescoring for completed rollout artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rewardscope.extraction import NumericExtractionConfig
from rewardscope.io import read_rollouts_jsonl, write_rollouts_jsonl
from rewardscope.io.atomic import atomic_write_json, atomic_write_jsonl
from rewardscope.metrics import PromptGroupMetricsResult, PromptGroupSummary
from rewardscope.reports.analysis import analyze_rollouts_jsonl, write_analysis_report
from rewardscope.rewards import RewardConfig
from rewardscope.rollouts import (
    RolloutInput,
    build_math_verify_numeric_rollout,
    build_numeric_rollout,
)


@dataclass(frozen=True)
class OfflineRescoreArtifacts:
    """Artifacts and metrics from recomputing extraction without model sampling."""

    output_dir: Path
    rescored_rollouts_jsonl: Path
    comparison_json: Path
    changed_samples_jsonl: Path
    migration_json: Path
    before_summary: PromptGroupSummary
    after_summary: PromptGroupSummary
    before_result: PromptGroupMetricsResult
    after_result: PromptGroupMetricsResult


def rescore_completed_run(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
) -> OfflineRescoreArtifacts:
    """Re-score a completed run's saved responses using the current extractor only."""
    source = Path(run_dir)
    rows = read_rollouts_jsonl(source / "rollouts.jsonl")
    if not rows:
        raise ValueError("Cannot rescore an empty rollout file.")
    snapshot = _read_json(source / "config_snapshot.json")
    resolved = _require_mapping(snapshot.get("resolved"), "config_snapshot.resolved")
    destination = (
        Path(output_dir)
        if output_dir is not None
        else source.parent / f"{source.name}-extractor-rescore"
    )
    _require_empty_destination(destination)
    destination.mkdir(parents=True)

    reward_config = RewardConfig(**_require_mapping(resolved.get("reward"), "resolved.reward"))
    sampling = _require_mapping(resolved.get("sampling"), "resolved.sampling")
    analysis = _require_mapping(resolved.get("analysis"), "resolved.analysis")
    dataset = _require_mapping(resolved.get("dataset"), "resolved.dataset")
    percentage_policy = "literal" if dataset.get("name", "").lower() == "gsm8k" else "reject"
    extraction_config = NumericExtractionConfig(percentage_policy=percentage_policy)

    expected_group_size = _require_positive_int(sampling, "num_samples")
    k_values = tuple(_require_positive_int_value(value, "analysis.k_values") for value in analysis.get("k_values", (1, 4, 8)))
    strict = analysis.get("strict", False)
    before_result, before_summary = analyze_rollouts_jsonl(
        source / "rollouts.jsonl", expected_group_size=expected_group_size, strict=strict, k_values=k_values
    )
    if before_summary is None:
        raise ValueError("Completed run has no prompt-group summary.")

    rescored_records = [
        build_numeric_rollout(
            _rollout_input_from_row(row), reward_config=reward_config, extraction_config=extraction_config
        )
        for row in rows
    ]
    rescored_path = destination / "rescored_rollouts.jsonl"
    write_rollouts_jsonl(rescored_path, rescored_records)
    after_result, after_summary = analyze_rollouts_jsonl(
        rescored_path, expected_group_size=expected_group_size, strict=strict, k_values=k_values
    )
    if after_summary is None:
        raise ValueError("Rescoring produced no prompt-group summary.")
    write_analysis_report(destination / "analysis", after_result, after_summary)

    rescored_rows = read_rollouts_jsonl(rescored_path)
    changed_rows = _changed_samples(rows, rescored_rows)
    changed_path = destination / "changed_samples.jsonl"
    atomic_write_jsonl(changed_path, changed_rows)
    migration_path = destination / "migration.json"
    atomic_write_json(
        migration_path,
        _build_migration(rows, rescored_rows, before_result, after_result),
    )
    comparison_path = destination / "comparison.json"
    atomic_write_json(comparison_path, {
        "percentage_policy": percentage_policy,
        "before": _metric_snapshot(rows, before_summary),
        "after": _metric_snapshot(rescored_rows, after_summary),
        "changed_sample_count": len(changed_rows),
    })
    return OfflineRescoreArtifacts(
        output_dir=destination,
        rescored_rollouts_jsonl=rescored_path,
        comparison_json=comparison_path,
        changed_samples_jsonl=changed_path,
        migration_json=migration_path,
        before_summary=before_summary,
        after_summary=after_summary,
        before_result=before_result,
        after_result=after_result,
    )


def rescore_completed_run_with_math_verify(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    mode: str = "evaluation",
) -> OfflineRescoreArtifacts:
    """Re-score saved outputs with Math-Verify without generating new responses."""
    source = Path(run_dir)
    rows = read_rollouts_jsonl(source / "rollouts.jsonl")
    if not rows:
        raise ValueError("Cannot rescore an empty rollout file.")
    snapshot = _read_json(source / "config_snapshot.json")
    resolved = _require_mapping(snapshot.get("resolved"), "config_snapshot.resolved")
    destination = (
        Path(output_dir)
        if output_dir is not None
        else source.parent / f"{source.name}-math-verify-rescore"
    )
    _require_empty_destination(destination)
    destination.mkdir(parents=True)

    reward_config = RewardConfig(**_require_mapping(resolved.get("reward"), "resolved.reward"))
    sampling = _require_mapping(resolved.get("sampling"), "resolved.sampling")
    analysis = _require_mapping(resolved.get("analysis"), "resolved.analysis")
    expected_group_size = _require_positive_int(sampling, "num_samples")
    k_values = tuple(_require_positive_int_value(value, "analysis.k_values") for value in analysis.get("k_values", (1, 4, 8)))
    strict = analysis.get("strict", False)
    before_result, before_summary = analyze_rollouts_jsonl(
        source / "rollouts.jsonl", expected_group_size=expected_group_size, strict=strict, k_values=k_values
    )
    if before_summary is None:
        raise ValueError("Completed run has no prompt-group summary.")

    rescored_records = [
        build_math_verify_numeric_rollout(
            _rollout_input_from_row(row), reward_config=reward_config, mode=mode
        )
        for row in rows
    ]
    rescored_path = destination / "rescored_rollouts.jsonl"
    write_rollouts_jsonl(rescored_path, rescored_records)
    after_result, after_summary = analyze_rollouts_jsonl(
        rescored_path, expected_group_size=expected_group_size, strict=strict, k_values=k_values
    )
    if after_summary is None:
        raise ValueError("Rescoring produced no prompt-group summary.")
    write_analysis_report(destination / "analysis", after_result, after_summary)

    rescored_rows = read_rollouts_jsonl(rescored_path)
    changed_rows = _changed_samples(rows, rescored_rows)
    changed_path = destination / "changed_samples.jsonl"
    atomic_write_jsonl(changed_path, changed_rows)
    migration_path = destination / "migration.json"
    atomic_write_json(
        migration_path,
        _build_migration(rows, rescored_rows, before_result, after_result),
    )
    comparison_path = destination / "comparison.json"
    atomic_write_json(comparison_path, {
        "verifier": "math_verify",
        "mode": mode,
        "before": _metric_snapshot(rows, before_summary),
        "after": _metric_snapshot(rescored_rows, after_summary),
        "changed_sample_count": len(changed_rows),
    })
    return OfflineRescoreArtifacts(
        output_dir=destination,
        rescored_rollouts_jsonl=rescored_path,
        comparison_json=comparison_path,
        changed_samples_jsonl=changed_path,
        migration_json=migration_path,
        before_summary=before_summary,
        after_summary=after_summary,
        before_result=before_result,
        after_result=after_result,
    )


def _rollout_input_from_row(row: dict[str, Any]) -> RolloutInput:
    return RolloutInput(
        run_id=_require_str(row, "run_id"), prompt_id=_require_str(row, "prompt_id"),
        sample_id=_require_non_negative_int(row, "sample_id"), model_name=_require_str(row, "model_name"),
        dataset_name=_require_str(row, "dataset_name"), split=_require_str(row, "split"),
        generation_seed=_require_non_negative_int(row, "generation_seed"),
        temperature=_require_number(row, "temperature"), top_p=_require_number(row, "top_p"),
        max_new_tokens=_require_positive_int(row, "max_new_tokens"), batch_size=_require_positive_int(row, "batch_size"),
        prompt=_require_str(row, "prompt"), response=_require_str(row, "response"),
        ground_truth=_require_str(row, "ground_truth"), prompt_tokens=_require_non_negative_int(row, "prompt_tokens"),
        response_tokens=_require_non_negative_int(row, "response_tokens"), hit_max_length=_require_bool(row, "hit_max_length"),
        finish_reason=_finish_reason_from_row(row),
    )


def _finish_reason_from_row(row: dict[str, Any]) -> str:
    value = row.get("finish_reason")
    if value is None:
        return "length" if _require_bool(row, "hit_max_length") else "eos"
    if value not in {"eos", "length"}:
        raise ValueError("finish_reason must be eos or length.")
    return value


def _changed_samples(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    for old, new in zip(before, after, strict=True):
        old_extraction = old["verification"]["extraction"]
        new_extraction = new["verification"]["extraction"]
        if (
            old_extraction["extraction_status"], old_extraction["normalized_answer"], old_extraction["format_ok"], old["verification"]["is_correct"]
        ) != (
            new_extraction["extraction_status"], new_extraction["normalized_answer"], new_extraction["format_ok"], new["verification"]["is_correct"]
        ):
            changed.append({
                "prompt_id": new["prompt_id"], "sample_id": new["sample_id"], "response": new["response"],
                "ground_truth": new["ground_truth"], "before": {"extraction": old_extraction, "raw_correctness": old["verification"]["is_correct"]},
                "after": {"extraction": new_extraction, "raw_correctness": new["verification"]["is_correct"]},
            })
    return changed


def _metric_snapshot(rows: list[dict[str, Any]], summary: PromptGroupSummary) -> dict[str, Any]:
    return {
        "extraction_failure_rate": summary.extraction_failure_rate,
        "format_error_rate": summary.format_error_rate,
        "accuracy": sum(row["verification"]["is_correct"] for row in rows) / len(rows),
        "all_wrong_rate": summary.all_wrong_rate,
        "mixed_rate": summary.mixed_rate,
        "all_correct_rate": summary.all_correct_rate,
        "pass_at_k": summary.pass_at_k,
    }


def _build_migration(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    before_result: PromptGroupMetricsResult,
    after_result: PromptGroupMetricsResult,
) -> dict[str, Any]:
    """Summarize extractor-only changes at both sample and prompt-group levels."""
    if len(before_rows) != len(after_rows):
        raise ValueError("Rescored rollout count does not match the source rollout count.")

    sample_transitions: Counter[tuple[tuple[object, ...], tuple[object, ...]]] = Counter()
    for before, after in zip(before_rows, after_rows, strict=True):
        if _sample_key(before) != _sample_key(after):
            raise ValueError("Rescored rollout order or sample identifiers changed.")
        sample_transitions[(_sample_decision(before), _sample_decision(after))] += 1

    before_groups = {
        (group.run_id, group.prompt_id): group for group in before_result.groups
    }
    after_groups = {
        (group.run_id, group.prompt_id): group for group in after_result.groups
    }
    if before_groups.keys() != after_groups.keys():
        raise ValueError("Rescored prompt groups do not match the source prompt groups.")

    group_transitions: Counter[tuple[str, str]] = Counter()
    group_details: list[dict[str, Any]] = []
    for key in sorted(before_groups):
        before_group = before_groups[key]
        after_group = after_groups[key]
        before_outcome = _group_outcome(before_group)
        after_outcome = _group_outcome(after_group)
        group_transitions[(before_outcome, after_outcome)] += 1
        group_details.append(
            {
                "run_id": key[0],
                "prompt_id": key[1],
                "before": _group_decision(before_group, before_outcome),
                "after": _group_decision(after_group, after_outcome),
            }
        )

    return {
        "sample": {
            "sample_count": len(before_rows),
            "decision_transitions": [
                {
                    "before": _decision_to_dict(before),
                    "after": _decision_to_dict(after),
                    "count": count,
                }
                for (before, after), count in sorted(
                    sample_transitions.items(), key=lambda item: (item[0], item[1])
                )
            ],
        },
        "group": {
            "group_count": len(before_groups),
            "outcome_transitions": [
                {"before": before, "after": after, "count": count}
                for (before, after), count in sorted(group_transitions.items())
            ],
            "groups": group_details,
        },
    }


def _sample_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        _require_str(row, "run_id"),
        _require_str(row, "prompt_id"),
        _require_non_negative_int(row, "sample_id"),
    )


def _sample_decision(row: dict[str, Any]) -> tuple[object, ...]:
    extraction = _require_mapping(
        _require_mapping(row.get("verification"), "verification").get("extraction"),
        "verification.extraction",
    )
    verification = _require_mapping(row.get("verification"), "verification")
    return (
        _require_str(extraction, "extraction_status"),
        _require_bool(extraction, "format_ok"),
        _require_bool(verification, "is_correct"),
    )


def _decision_to_dict(decision: tuple[object, ...]) -> dict[str, object]:
    return {
        "extraction_status": decision[0],
        "format_ok": decision[1],
        "raw_correctness": decision[2],
    }


def _group_outcome(group: Any) -> str:
    if group.all_wrong:
        return "all_wrong"
    if group.mixed:
        return "mixed"
    return "all_correct"


def _group_decision(group: Any, outcome: str) -> dict[str, object]:
    return {
        "outcome": outcome,
        "correct_count": group.correct_count,
        "sample_count": group.sample_count,
        "extraction_failure_count": group.extraction_failure_count,
        "format_error_count": group.format_error_count,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        value = json.load(input_file)
    return _require_mapping(value, str(path))


def _require_empty_destination(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"Offline rescore destination must be absent or empty: {path}")
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


def _require_bool(row: dict[str, Any], name: str) -> bool:
    value = row.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _require_number(row: dict[str, Any], name: str) -> float:
    value = row.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    return float(value)


def _require_non_negative_int(row: dict[str, Any], name: str) -> int:
    value = row.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _require_positive_int(row: dict[str, Any], name: str) -> int:
    value = _require_non_negative_int(row, name)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _require_positive_int_value(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must contain positive integers.")
    return value
