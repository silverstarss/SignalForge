"""Prompt-level rollout diagnostics for group-relative optimization experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import comb, isfinite, sqrt
from typing import Any


_SUCCESSFUL_EXTRACTION_STATUSES = frozenset(
    {"explicit_final", "boxed", "implicit_terminal"}
)


@dataclass(frozen=True)
class MetricsIssue:
    """A malformed row or group-consistency problem found during analysis."""

    code: str
    message: str
    row_index: int | None
    run_id: str | None
    prompt_id: str | None


@dataclass(frozen=True)
class PromptGroupMetrics:
    """Diagnostics for all valid samples sharing one ``(run_id, prompt_id)`` key."""

    run_id: str
    prompt_id: str
    sample_count: int
    any_correct: bool
    all_correct: bool
    all_wrong: bool
    mixed: bool
    correct_count: int
    extraction_failure_count: int
    extraction_failure_rate: float
    format_error_count: int
    format_error_rate: float
    unique_valid_answer_count: int
    raw_reward_mean: float
    raw_reward_variance: float
    raw_reward_std: float
    raw_reward_range: float
    final_reward_mean: float
    final_reward_variance: float
    final_reward_std: float
    final_reward_range: float
    response_tokens_total: int
    response_tokens_mean: float
    hit_max_length_count: int
    hit_max_length_rate: float
    bad_case_tags: tuple[str, ...]


@dataclass(frozen=True)
class PromptGroupMetricsResult:
    """Computed group metrics plus any rows or groups excluded from analysis."""

    groups: tuple[PromptGroupMetrics, ...]
    issues: tuple[MetricsIssue, ...]


@dataclass(frozen=True)
class PromptGroupSummary:
    """Run-level aggregate diagnostics derived from prompt-group metrics."""

    group_count: int
    all_wrong_rate: float
    all_correct_rate: float
    mixed_rate: float
    pass_at_k: dict[int, float | None]
    pass_at_k_eligible_group_count: dict[int, int]
    mean_raw_reward_variance: float
    mean_raw_reward_range: float
    mean_final_reward_variance: float
    mean_final_reward_range: float
    format_error_rate: float
    extraction_failure_rate: float
    hit_max_length_count: int
    hit_max_length_rate: float
    groups_with_hit_max_length_count: int
    total_response_tokens: int
    effective_response_tokens: int
    effective_token_ratio: float
    token_cost_per_mixed_prompt: float | None
    bad_case_counts: dict[str, int]
    issue_count: int


@dataclass(frozen=True)
class _RolloutObservation:
    row_index: int
    run_id: str
    prompt_id: str
    sample_id: int
    ground_truth: str
    model_name: str
    dataset_name: str
    split: str
    generation_seed: int
    temperature: float
    top_p: float
    max_new_tokens: int
    batch_size: int
    raw_correctness: bool
    extraction_ok: bool
    normalized_answer: str | None
    format_ok: bool
    raw_reward: float
    final_reward: float
    response_tokens: int
    hit_max_length: bool


class _MalformedRowError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def compute_prompt_group_metrics(
    rollout_rows: Iterable[Mapping[str, Any]],
    *,
    expected_group_size: int | None = None,
    strict: bool = False,
) -> PromptGroupMetricsResult:
    """Compute diagnostics by ``(run_id, prompt_id)`` from serialized rollout rows.

    In non-strict mode malformed rows and conflicting group members are excluded
    and returned as issues. Strict mode raises as soon as such a problem appears.
    """
    _require_optional_positive_int("expected_group_size", expected_group_size)
    _require_bool_value("strict", strict)

    grouped_rows: dict[tuple[str, str], list[_RolloutObservation]] = defaultdict(list)
    issues: list[MetricsIssue] = []

    for row_index, row in enumerate(rollout_rows):
        try:
            observation = _parse_rollout_row(row, row_index)
        except _MalformedRowError as error:
            issue = MetricsIssue(
                code=error.code,
                message=str(error),
                row_index=row_index,
                run_id=None,
                prompt_id=None,
            )
            if strict:
                raise ValueError(_format_issue(issue)) from error
            issues.append(issue)
            continue
        grouped_rows[(observation.run_id, observation.prompt_id)].append(observation)

    metrics: list[PromptGroupMetrics] = []
    for group_key in sorted(grouped_rows):
        selected_rows, group_tags, group_issues = _select_consistent_group_rows(
            grouped_rows[group_key], expected_group_size
        )
        if strict and group_issues:
            raise ValueError(_format_issue(group_issues[0]))
        issues.extend(group_issues)
        if selected_rows:
            metrics.append(_compute_group_metrics(selected_rows, group_tags))

    return PromptGroupMetricsResult(groups=tuple(metrics), issues=tuple(issues))


def summarize_prompt_group_metrics(
    result: PromptGroupMetricsResult,
    *,
    k_values: tuple[int, ...] = (1, 4, 8),
) -> PromptGroupSummary | None:
    """Summarize prompt groups, including standard macro-averaged pass@k."""
    if not isinstance(result, PromptGroupMetricsResult):
        raise TypeError("result must be a PromptGroupMetricsResult.")
    _validate_k_values(k_values)

    groups = result.groups
    if not groups:
        return None

    group_count = len(groups)
    total_samples = sum(group.sample_count for group in groups)
    total_response_tokens = sum(group.response_tokens_total for group in groups)
    hit_max_length_count = sum(group.hit_max_length_count for group in groups)
    mixed_groups = [group for group in groups if group.mixed]
    effective_response_tokens = sum(
        group.response_tokens_total for group in mixed_groups
    )
    bad_case_counts = dict(
        Counter(tag for group in groups for tag in group.bad_case_tags)
    )

    pass_at_k: dict[int, float | None] = {}
    eligible_counts: dict[int, int] = {}
    for k in k_values:
        eligible_groups = [group for group in groups if group.sample_count >= k]
        eligible_counts[k] = len(eligible_groups)
        pass_at_k[k] = (
            sum(_estimate_pass_at_k(group.sample_count, group.correct_count, k) for group in eligible_groups)
            / len(eligible_groups)
            if eligible_groups
            else None
        )

    return PromptGroupSummary(
        group_count=group_count,
        all_wrong_rate=sum(group.all_wrong for group in groups) / group_count,
        all_correct_rate=sum(group.all_correct for group in groups) / group_count,
        mixed_rate=sum(group.mixed for group in groups) / group_count,
        pass_at_k=pass_at_k,
        pass_at_k_eligible_group_count=eligible_counts,
        mean_raw_reward_variance=(
            sum(group.raw_reward_variance for group in groups) / group_count
        ),
        mean_raw_reward_range=(
            sum(group.raw_reward_range for group in groups) / group_count
        ),
        mean_final_reward_variance=(
            sum(group.final_reward_variance for group in groups) / group_count
        ),
        mean_final_reward_range=(
            sum(group.final_reward_range for group in groups) / group_count
        ),
        format_error_rate=(
            sum(group.format_error_count for group in groups) / total_samples
        ),
        extraction_failure_rate=(
            sum(group.extraction_failure_count for group in groups) / total_samples
        ),
        hit_max_length_count=hit_max_length_count,
        hit_max_length_rate=hit_max_length_count / total_samples,
        groups_with_hit_max_length_count=sum(
            group.hit_max_length_count > 0 for group in groups
        ),
        total_response_tokens=total_response_tokens,
        effective_response_tokens=effective_response_tokens,
        effective_token_ratio=(
            effective_response_tokens / total_response_tokens
            if total_response_tokens
            else 0.0
        ),
        token_cost_per_mixed_prompt=(
            total_response_tokens / len(mixed_groups) if mixed_groups else None
        ),
        bad_case_counts=bad_case_counts,
        issue_count=len(result.issues),
    )


def _parse_rollout_row(row: Mapping[str, Any], row_index: int) -> _RolloutObservation:
    if not isinstance(row, Mapping):
        raise _MalformedRowError("row_not_mapping", "Rollout row must be a mapping.")

    verification = _require_mapping(row, "verification")
    extraction = _require_mapping(verification, "extraction")
    reward = _require_mapping(row, "reward")
    extraction_status = _require_str(extraction, "extraction_status")
    if extraction_status not in {
        "explicit_final",
        "boxed",
        "implicit_terminal",
        "ambiguous",
        "missing",
        "parse_error",
    }:
        raise _MalformedRowError(
            "invalid_extraction_status",
            f"Unsupported extraction_status {extraction_status!r}.",
        )

    normalized_answer = extraction.get("normalized_answer")
    extraction_ok = extraction_status in _SUCCESSFUL_EXTRACTION_STATUSES
    if extraction_ok and (not isinstance(normalized_answer, str) or not normalized_answer):
        raise _MalformedRowError(
            "invalid_normalized_answer",
            "Successful extraction requires a non-empty normalized_answer.",
        )
    if not extraction_ok and normalized_answer is not None:
        raise _MalformedRowError(
            "invalid_normalized_answer",
            "Failed extraction must not include a normalized_answer.",
        )

    return _RolloutObservation(
        row_index=row_index,
        run_id=_require_non_empty_str(row, "run_id"),
        prompt_id=_require_non_empty_str(row, "prompt_id"),
        sample_id=_require_non_negative_int(row, "sample_id"),
        ground_truth=_require_str(row, "ground_truth"),
        model_name=_require_non_empty_str(row, "model_name"),
        dataset_name=_require_non_empty_str(row, "dataset_name"),
        split=_require_non_empty_str(row, "split"),
        generation_seed=_require_non_negative_int(row, "generation_seed"),
        temperature=_require_non_negative_number(row, "temperature"),
        top_p=_require_probability(row, "top_p"),
        max_new_tokens=_require_positive_int(row, "max_new_tokens"),
        batch_size=_require_positive_int(row, "batch_size"),
        raw_correctness=_require_row_bool(verification, "is_correct"),
        extraction_ok=extraction_ok,
        normalized_answer=normalized_answer,
        format_ok=_require_row_bool(extraction, "format_ok"),
        raw_reward=_require_finite_number(reward, "correctness_reward"),
        final_reward=_require_finite_number(reward, "final_reward"),
        response_tokens=_require_non_negative_int(row, "response_tokens"),
        hit_max_length=_require_row_bool(row, "hit_max_length"),
    )


def _select_consistent_group_rows(
    rows: list[_RolloutObservation], expected_group_size: int | None
) -> tuple[list[_RolloutObservation], list[str], list[MetricsIssue]]:
    canonical = rows[0]
    selected_rows: list[_RolloutObservation] = [canonical]
    seen_sample_ids = {canonical.sample_id}
    tags: list[str] = []
    issues: list[MetricsIssue] = []

    for row in rows[1:]:
        mismatch_tag = _find_group_mismatch(canonical, row)
        if mismatch_tag is not None:
            tags.append(mismatch_tag)
            issues.append(
                _group_issue(
                    mismatch_tag,
                    row,
                    f"Row conflicts with the canonical {mismatch_tag.removeprefix('inconsistent_')} value.",
                )
            )
            continue
        if row.sample_id in seen_sample_ids:
            tags.append("duplicate_sample_id")
            issues.append(
                _group_issue(
                    "duplicate_sample_id",
                    row,
                    f"sample_id {row.sample_id} appears more than once in the group.",
                )
            )
            continue
        seen_sample_ids.add(row.sample_id)
        selected_rows.append(row)

    if expected_group_size is not None and len(selected_rows) != expected_group_size:
        tags.append("unexpected_group_size")
        issues.append(
            MetricsIssue(
                code="unexpected_group_size",
                message=(
                    f"Expected {expected_group_size} samples but retained "
                    f"{len(selected_rows)} valid samples."
                ),
                row_index=None,
                run_id=canonical.run_id,
                prompt_id=canonical.prompt_id,
            )
        )

    return selected_rows, _deduplicate_preserving_order(tags), issues


def _find_group_mismatch(
    canonical: _RolloutObservation, row: _RolloutObservation
) -> str | None:
    if row.ground_truth != canonical.ground_truth:
        return "inconsistent_ground_truth"
    if row.model_name != canonical.model_name:
        return "inconsistent_model"
    if row.dataset_name != canonical.dataset_name or row.split != canonical.split:
        return "inconsistent_dataset"
    if (
        row.generation_seed,
        row.temperature,
        row.top_p,
        row.max_new_tokens,
        row.batch_size,
    ) != (
        canonical.generation_seed,
        canonical.temperature,
        canonical.top_p,
        canonical.max_new_tokens,
        canonical.batch_size,
    ):
        return "inconsistent_generation_config"
    return None


def _compute_group_metrics(
    rows: list[_RolloutObservation], group_tags: list[str]
) -> PromptGroupMetrics:
    sample_count = len(rows)
    correct_count = sum(row.raw_correctness for row in rows)
    extraction_failure_count = sum(not row.extraction_ok for row in rows)
    format_error_count = sum(not row.format_ok for row in rows)
    hit_max_length_count = sum(row.hit_max_length for row in rows)
    raw_stats = _compute_distribution([row.raw_reward for row in rows])
    final_stats = _compute_distribution([row.final_reward for row in rows])
    all_correct = correct_count == sample_count
    all_wrong = correct_count == 0
    tags = list(group_tags)

    if all_wrong:
        tags.append("all_wrong")
    if extraction_failure_count:
        tags.append("extraction_failures")
    if format_error_count:
        tags.append("format_errors")
    unique_valid_answer_count = len(
        {
            row.normalized_answer
            for row in rows
            if row.extraction_ok and row.normalized_answer is not None
        }
    )
    if unique_valid_answer_count > 1:
        tags.append("multiple_valid_answers")
    if hit_max_length_count:
        tags.append("hit_max_length")
    if sample_count > 1 and raw_stats[1] == 0.0:
        tags.append("zero_raw_reward_variance")

    response_tokens_total = sum(row.response_tokens for row in rows)
    return PromptGroupMetrics(
        run_id=rows[0].run_id,
        prompt_id=rows[0].prompt_id,
        sample_count=sample_count,
        any_correct=correct_count > 0,
        all_correct=all_correct,
        all_wrong=all_wrong,
        mixed=not all_correct and not all_wrong,
        correct_count=correct_count,
        extraction_failure_count=extraction_failure_count,
        extraction_failure_rate=extraction_failure_count / sample_count,
        format_error_count=format_error_count,
        format_error_rate=format_error_count / sample_count,
        unique_valid_answer_count=unique_valid_answer_count,
        raw_reward_mean=raw_stats[0],
        raw_reward_variance=raw_stats[1],
        raw_reward_std=raw_stats[2],
        raw_reward_range=raw_stats[3],
        final_reward_mean=final_stats[0],
        final_reward_variance=final_stats[1],
        final_reward_std=final_stats[2],
        final_reward_range=final_stats[3],
        response_tokens_total=response_tokens_total,
        response_tokens_mean=response_tokens_total / sample_count,
        hit_max_length_count=hit_max_length_count,
        hit_max_length_rate=hit_max_length_count / sample_count,
        bad_case_tags=tuple(_deduplicate_preserving_order(tags)),
    )


def _compute_distribution(values: list[float]) -> tuple[float, float, float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance, sqrt(variance), max(values) - min(values)


def _estimate_pass_at_k(sample_count: int, correct_count: int, k: int) -> float:
    return 1 - comb(sample_count - correct_count, k) / comb(sample_count, k)


def _require_mapping(container: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = container.get(name)
    if not isinstance(value, Mapping):
        raise _MalformedRowError("missing_or_invalid_field", f"{name} must be a mapping.")
    return value


def _require_str(container: Mapping[str, Any], name: str) -> str:
    value = container.get(name)
    if not isinstance(value, str):
        raise _MalformedRowError("missing_or_invalid_field", f"{name} must be a string.")
    return value


def _require_non_empty_str(container: Mapping[str, Any], name: str) -> str:
    value = _require_str(container, name)
    if not value.strip():
        raise _MalformedRowError(
            "missing_or_invalid_field", f"{name} must be a non-empty string."
        )
    return value


def _require_row_bool(container: Mapping[str, Any], name: str) -> bool:
    value = container.get(name)
    if not isinstance(value, bool):
        raise _MalformedRowError("missing_or_invalid_field", f"{name} must be a boolean.")
    return value


def _require_non_negative_int(container: Mapping[str, Any], name: str) -> int:
    value = container.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _MalformedRowError(
            "missing_or_invalid_field", f"{name} must be a non-negative integer."
        )
    return value


def _require_positive_int(container: Mapping[str, Any], name: str) -> int:
    value = container.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _MalformedRowError(
            "missing_or_invalid_field", f"{name} must be a positive integer."
        )
    return value


def _require_finite_number(container: Mapping[str, Any], name: str) -> float:
    value = container.get(name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
    ):
        raise _MalformedRowError(
            "missing_or_invalid_field", f"{name} must be a finite number."
        )
    return float(value)


def _require_non_negative_number(container: Mapping[str, Any], name: str) -> float:
    value = _require_finite_number(container, name)
    if value < 0:
        raise _MalformedRowError(
            "missing_or_invalid_field", f"{name} must be non-negative."
        )
    return value


def _require_probability(container: Mapping[str, Any], name: str) -> float:
    value = _require_finite_number(container, name)
    if not 0 < value <= 1:
        raise _MalformedRowError(
            "missing_or_invalid_field", f"{name} must be in the interval (0, 1]."
        )
    return value


def _require_optional_positive_int(name: str, value: int | None) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer or None.")


def _require_bool_value(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")


def _validate_k_values(k_values: tuple[int, ...]) -> None:
    if not isinstance(k_values, tuple) or not k_values:
        raise ValueError("k_values must be a non-empty tuple of positive integers.")
    if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in k_values):
        raise ValueError("k_values must be a non-empty tuple of positive integers.")
    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values must not contain duplicates.")


def _group_issue(code: str, row: _RolloutObservation, message: str) -> MetricsIssue:
    return MetricsIssue(
        code=code,
        message=message,
        row_index=row.row_index,
        run_id=row.run_id,
        prompt_id=row.prompt_id,
    )


def _format_issue(issue: MetricsIssue) -> str:
    location = f"row {issue.row_index}" if issue.row_index is not None else "group"
    return f"{issue.code} at {location}: {issue.message}"


def _deduplicate_preserving_order(tags: list[str]) -> list[str]:
    return list(dict.fromkeys(tags))
