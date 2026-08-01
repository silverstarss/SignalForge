"""Export prompt-group diagnostics to CSV, JSON, and JSONL artifacts."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from rewardscope.io import read_rollouts_jsonl
from rewardscope.io.atomic import atomic_write_json, atomic_write_jsonl, _atomic_write
from rewardscope.metrics import (
    PromptGroupMetrics,
    PromptGroupMetricsResult,
    PromptGroupSummary,
    compute_prompt_group_metrics,
    summarize_prompt_group_metrics,
)


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Filesystem locations and counts produced by an analysis report write."""

    prompt_group_metrics_csv: Path
    summary_json: Path
    issues_jsonl: Path
    group_count: int
    issue_count: int


def analyze_rollouts_jsonl(
    input_path: str | Path,
    *,
    expected_group_size: int | None = None,
    strict: bool = False,
    k_values: tuple[int, ...] = (1, 4, 8),
) -> tuple[PromptGroupMetricsResult, PromptGroupSummary | None]:
    """Load rollout JSONL and compute its prompt-group diagnostics and summary."""
    rows = read_rollouts_jsonl(input_path)
    result = compute_prompt_group_metrics(
        rows, expected_group_size=expected_group_size, strict=strict
    )
    return result, summarize_prompt_group_metrics(result, k_values=k_values)


def write_analysis_report(
    output_dir: str | Path,
    result: PromptGroupMetricsResult,
    summary: PromptGroupSummary | None,
) -> AnalysisArtifacts:
    """Write a flat group CSV, JSON summary, and JSONL issue log."""
    if not isinstance(result, PromptGroupMetricsResult):
        raise TypeError("result must be a PromptGroupMetricsResult.")
    if summary is not None and not isinstance(summary, PromptGroupSummary):
        raise TypeError("summary must be a PromptGroupSummary or None.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    group_metrics_path = destination / "prompt_group_metrics.csv"
    summary_path = destination / "summary.json"
    issues_path = destination / "issues.jsonl"

    _write_group_metrics_csv(group_metrics_path, result.groups)
    _write_json(summary_path, None if summary is None else asdict(summary))
    _write_issues_jsonl(issues_path, result)

    return AnalysisArtifacts(
        prompt_group_metrics_csv=group_metrics_path,
        summary_json=summary_path,
        issues_jsonl=issues_path,
        group_count=len(result.groups),
        issue_count=len(result.issues),
    )


def _write_group_metrics_csv(
    path: Path, groups: tuple[PromptGroupMetrics, ...]
) -> None:
    field_names = [field.name for field in fields(PromptGroupMetrics)]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=field_names)
    writer.writeheader()
    for group in groups:
        row = asdict(group)
        row["bad_case_tags"] = ";".join(group.bad_case_tags)
        writer.writerow(row)
    _atomic_write(path, output.getvalue())


def _write_json(path: Path, value: object) -> None:
    atomic_write_json(path, value)


def _write_issues_jsonl(path: Path, result: PromptGroupMetricsResult) -> None:
    atomic_write_jsonl(path, [asdict(issue) for issue in result.issues])
