"""Online metric reducers for Signal Forge A0/A training.

These helpers intentionally compute only cheap per-step aggregates. RewardScope
remains the source of truth for offline reports, plots, and pass@k suites.
"""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Iterable


def _as_list(values) -> list:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _float_values(values) -> list[float]:
    out = []
    for value in _as_list(values):
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _population_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float(sum((value - mean) ** 2 for value in values) / len(values))


def _std(values: list[float]) -> float:
    return sqrt(_population_variance(values)) if values else 0.0


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def compute_reward_extra_metrics(reward_extra_infos: dict) -> dict[str, float]:
    """Aggregate reward adapter extra fields without changing reward semantics."""
    raw = _float_values(reward_extra_infos.get("raw_correctness", reward_extra_infos.get("acc", [])))
    score = _float_values(reward_extra_infos.get("score", raw))
    extraction_ok = _float_values(reward_extra_infos.get("extraction_ok", []))
    format_ok = _float_values(reward_extra_infos.get("format_ok", []))

    metrics = {}
    if raw:
        metrics["reward/raw_correctness_mean"] = _mean(raw)
    if score:
        metrics["reward/final_score_mean"] = _mean(score)
    if extraction_ok:
        metrics["reward/extraction_ok_ratio"] = _mean(extraction_ok)
        metrics["reward/extraction_failure_ratio"] = 1.0 - _mean(extraction_ok)
    if format_ok:
        metrics["reward/format_ok_ratio"] = _mean(format_ok)
    metrics["reward/format_reward_mean"] = 0.0
    metrics["reward/length_penalty_mean"] = 0.0
    return metrics


def compute_group_metrics(
    *,
    uids: Iterable | None,
    raw_correctness: Iterable | None,
    expected_group_size: int | None = None,
    prefix: str = "group/",
) -> dict[str, float]:
    """Compute cheap prompt-group diagnostics from raw binary correctness."""
    uid_values = _as_list(uids)
    correctness_values = _float_values(raw_correctness)
    n = min(len(uid_values), len(correctness_values))

    groups: dict[object, list[float]] = defaultdict(list)
    missing_uid_count = 0
    for uid, value in zip(uid_values[:n], correctness_values[:n], strict=True):
        if uid in (None, ""):
            missing_uid_count += 1
            continue
        groups[uid].append(1.0 if value >= 0.5 else 0.0)

    all_correct = 0
    all_wrong = 0
    mixed = 0
    incomplete = 0
    oversized = 0
    pass_rates = []
    variances = []

    valid_group_count = 0
    for values in groups.values():
        if expected_group_size and len(values) != expected_group_size:
            incomplete += int(len(values) < expected_group_size)
            oversized += int(len(values) > expected_group_size)
            continue
        valid_group_count += 1
        group_sum = sum(values)
        all_correct += int(group_sum == len(values))
        all_wrong += int(group_sum == 0)
        mixed += int(0 < group_sum < len(values))
        pass_rates.append(_mean(values))
        variances.append(_population_variance(values))

    num_groups = len(groups)
    denom = max(valid_group_count, 1)
    nonzero_variances = [value for value in variances if value > 0.0]

    return {
        f"{prefix}all_correct_count": float(all_correct),
        f"{prefix}all_wrong_count": float(all_wrong),
        f"{prefix}mixed_count": float(mixed),
        f"{prefix}all_correct_ratio": float(all_correct / denom),
        f"{prefix}all_wrong_ratio": float(all_wrong / denom),
        f"{prefix}mixed_ratio": float(mixed / denom),
        f"{prefix}pass_rate_mean": _mean(pass_rates),
        f"{prefix}pass_rate_std": _std(pass_rates),
        f"{prefix}raw_reward_variance_mean": _mean(variances),
        f"{prefix}raw_reward_variance_nonzero_ratio": float(len(nonzero_variances) / denom),
        f"{prefix}num_groups": float(num_groups),
        f"{prefix}group_size_mean": _mean([float(len(values)) for values in groups.values()]),
        f"{prefix}incomplete_group_count": float(incomplete),
        f"{prefix}oversized_group_count": float(oversized),
        f"{prefix}missing_uid_count": float(missing_uid_count),
    }


def compute_length_metrics(
    *,
    response_lengths: Iterable,
    raw_correctness: Iterable | None,
    max_response_length: int,
    prefix: str = "length/",
) -> dict[str, float]:
    lengths = sorted(_float_values(response_lengths))
    correctness = _float_values(raw_correctness)
    correct_lengths = []
    incorrect_lengths = []
    for length, correct in zip(_float_values(response_lengths), correctness, strict=False):
        if correct >= 0.5:
            correct_lengths.append(length)
        else:
            incorrect_lengths.append(length)

    truncated = [1.0 if length >= float(max_response_length) else 0.0 for length in lengths]
    return {
        f"{prefix}response_p50": _percentile(lengths, 0.50),
        f"{prefix}response_p90": _percentile(lengths, 0.90),
        f"{prefix}response_p95": _percentile(lengths, 0.95),
        f"{prefix}truncated_ratio": _mean(truncated),
        f"{prefix}correct_response_mean": _mean(correct_lengths),
        f"{prefix}incorrect_response_mean": _mean(incorrect_lengths),
    }


def compute_validation_alias_metrics(data_sources: Iterable, reward_extra_infos: dict) -> dict[str, float]:
    """Add stable Signal Forge validation aliases over veRL native metrics."""
    sources = _as_list(data_sources)
    acc_values = _float_values(reward_extra_infos.get("acc", reward_extra_infos.get("reward", [])))
    raw_values = _float_values(reward_extra_infos.get("raw_correctness", acc_values))
    extraction_ok = _float_values(reward_extra_infos.get("extraction_ok", []))
    format_ok = _float_values(reward_extra_infos.get("format_ok", []))

    metrics: dict[str, float] = {}
    if acc_values:
        metrics["val/pass_at_1"] = _mean(acc_values)
    if raw_values:
        metrics["val/boxed_pass_at_1"] = _mean(raw_values)
    if extraction_ok:
        metrics["val/extraction_ok_ratio"] = _mean(extraction_ok)
    if format_ok:
        metrics["val/format_ok_ratio"] = _mean(format_ok)
    metrics["val/num_prompts"] = float(len(acc_values))

    per_source_indices: dict[str, list[int]] = defaultdict(list)
    for idx, source in enumerate(sources):
        per_source_indices[str(source)].append(idx)

    for source, indices in per_source_indices.items():
        safe_source = "math_l3" if source == "math_level_3" else source.replace("/", "_")
        source_acc = [acc_values[i] for i in indices if i < len(acc_values)]
        source_raw = [raw_values[i] for i in indices if i < len(raw_values)]
        source_extraction = [extraction_ok[i] for i in indices if i < len(extraction_ok)]
        source_format = [format_ok[i] for i in indices if i < len(format_ok)]
        if source_acc:
            metrics[f"val/{safe_source}/pass_at_1"] = _mean(source_acc)
        if source_raw:
            metrics[f"val/{safe_source}/boxed_pass_at_1"] = _mean(source_raw)
        if source_extraction:
            metrics[f"val/{safe_source}/extraction_ok_ratio"] = _mean(source_extraction)
        if source_format:
            metrics[f"val/{safe_source}/format_ok_ratio"] = _mean(source_format)
        metrics[f"val/{safe_source}/num_prompts"] = float(len(indices))

    return metrics
