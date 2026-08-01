"""Optional matplotlib visualizations for prompt-group diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
import os
from pathlib import Path
import tempfile

from rewardscope.metrics import (
    PromptGroupMetrics,
    PromptGroupMetricsResult,
    summarize_prompt_group_metrics,
)


_OUTCOME_ORDER = ("all-wrong", "mixed", "all-correct")
_OUTCOME_COLORS = {
    "all-wrong": "#d95f02",
    "mixed": "#1b9e77",
    "all-correct": "#7570b3",
}


@dataclass(frozen=True)
class AnalysisPlotArtifacts:
    """Paths of the optional PNG figures generated for one analysis result."""

    outcome_distribution_png: Path | None
    prompt_pass_rate_distribution_png: Path | None
    reward_variance_png: Path | None
    token_efficiency_png: Path | None


def write_analysis_plots(
    output_dir: str | Path,
    result: PromptGroupMetricsResult,
) -> AnalysisPlotArtifacts:
    """Generate four diagnostic PNGs from one internally consistent metric result."""
    if not isinstance(result, PromptGroupMetricsResult):
        raise TypeError("result must be a PromptGroupMetricsResult.")
    if not result.groups:
        return AnalysisPlotArtifacts(None, None, None, None)

    plt = _load_pyplot()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = summarize_prompt_group_metrics(result)
    assert summary is not None

    outcome_path = destination / "outcome_distribution.png"
    pass_rate_path = destination / "prompt_pass_rate_distribution.png"
    variance_path = destination / "reward_variance.png"
    token_path = destination / "token_efficiency.png"

    _plot_outcome_distribution(plt, outcome_path, result.groups)
    _plot_prompt_pass_rate_distribution(plt, pass_rate_path, result.groups)
    _plot_reward_variance(plt, variance_path, result.groups)
    _plot_token_efficiency(plt, token_path, result.groups, summary)

    return AnalysisPlotArtifacts(
        outcome_distribution_png=outcome_path,
        prompt_pass_rate_distribution_png=pass_rate_path,
        reward_variance_png=variance_path,
        token_efficiency_png=token_path,
    )


def _load_pyplot():
    try:
        import matplotlib
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'Plotting requires the optional analysis dependency. Run: pip install -e ".[analysis]"'
        ) from error

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return plt


def _plot_outcome_distribution(plt, path: Path, groups: tuple[PromptGroupMetrics, ...]) -> None:
    counts = _outcome_group_counts(groups)
    total = len(groups)
    labels = list(_OUTCOME_ORDER)
    values = [counts[label] for label in labels]
    colors = [_OUTCOME_COLORS[label] for label in labels]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Prompt groups")
    ax.set_title("Prompt-group outcomes")
    ax.set_ylim(0, max(values) * 1.2 + 0.5)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value} ({value / total:.1%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )
    _save_and_close(plt, fig, path)


def _plot_prompt_pass_rate_distribution(
    plt, path: Path, groups: tuple[PromptGroupMetrics, ...]
) -> None:
    group_sizes = {group.sample_count for group in groups}
    fig, ax = plt.subplots()

    if len(group_sizes) == 1:
        group_size = next(iter(group_sizes))
        counts = Counter(group.correct_count for group in groups)
        x_values = list(range(group_size + 1))
        labels = [f"{correct_count}/{group_size}" for correct_count in x_values]
        values = [counts[correct_count] for correct_count in x_values]
        annotation = f"prompt count: {len(groups)} | group size: {group_size}"
        ax.set_xlabel("Correct samples per prompt")
    else:
        counts = Counter(
            Fraction(group.correct_count, group.sample_count) for group in groups
        )
        rates = sorted(counts)
        x_values = list(range(len(rates)))
        labels = [f"{float(rate):.0%}" for rate in rates]
        values = [counts[rate] for rate in rates]
        annotation = (
            f"prompt count: {len(groups)} | group sizes vary: "
            f"{min(group_sizes)}-{max(group_sizes)}"
        )
        ax.set_xlabel("Correct rate")

    ax.bar(x_values, values, color="#4c78a8")
    ax.set_xticks(x_values, labels)
    ax.set_ylabel("Prompt count")
    ax.set_title("Prompt pass-rate distribution")
    ax.text(0.5, 1.02, annotation, ha="center", transform=ax.transAxes)
    _save_and_close(plt, fig, path)


def _plot_reward_variance(
    plt, path: Path, groups: tuple[PromptGroupMetrics, ...]
) -> None:
    grouped_coordinates = Counter(
        (
            group.raw_reward_variance,
            group.final_reward_variance,
            _outcome_name(group),
        )
        for group in groups
    )
    max_variance = max(
        max(group.raw_reward_variance, group.final_reward_variance) for group in groups
    )
    reference_limit = max_variance * 1.1 if max_variance > 0 else 1.0
    shaped_only_count = sum(
        group.raw_reward_variance == 0.0 and group.final_reward_variance > 0.0
        for group in groups
    )

    fig, ax = plt.subplots()
    for outcome in _OUTCOME_ORDER:
        coordinates = [
            (raw_variance, final_variance, count)
            for (raw_variance, final_variance, point_outcome), count in grouped_coordinates.items()
            if point_outcome == outcome
        ]
        if not coordinates:
            continue
        ax.scatter(
            [coordinate[0] for coordinate in coordinates],
            [coordinate[1] for coordinate in coordinates],
            s=[80 + 40 * sqrt(coordinate[2]) for coordinate in coordinates],
            color=_OUTCOME_COLORS[outcome],
            alpha=0.8,
            label=outcome,
        )

    ax.plot([0, reference_limit], [0, reference_limit], "--", color="#555555", label="y = x")
    ax.set_xlim(0, reference_limit)
    ax.set_ylim(0, reference_limit)
    ax.set_xlabel("Raw reward variance")
    ax.set_ylabel("Final reward variance")
    ax.set_title("Raw vs final reward variance")
    ax.legend()
    ax.text(
        0.02,
        0.98,
        f"raw=0, final>0 groups: {shaped_only_count}",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    _save_and_close(plt, fig, path)


def _plot_token_efficiency(plt, path: Path, groups, summary) -> None:
    token_totals = {
        outcome: sum(
            group.response_tokens_total
            for group in groups
            if _outcome_name(group) == outcome
        )
        for outcome in _OUTCOME_ORDER
    }
    total_tokens = summary.total_response_tokens
    percentages = {
        outcome: token_totals[outcome] / total_tokens * 100 if total_tokens else 0.0
        for outcome in _OUTCOME_ORDER
    }

    fig, ax = plt.subplots()
    bottom = 0.0
    for outcome in _OUTCOME_ORDER:
        percentage = percentages[outcome]
        ax.bar(
            ["Response tokens"],
            [percentage],
            bottom=[bottom],
            color=_OUTCOME_COLORS[outcome],
            label=outcome,
        )
        if percentage:
            ax.text(
                0,
                bottom + percentage / 2,
                f"{outcome}\n{percentage:.1f}%\n{token_totals[outcome]} tokens",
                ha="center",
                va="center",
            )
        bottom += percentage

    cost = (
        f"{summary.token_cost_per_mixed_prompt:.1f}"
        if summary.token_cost_per_mixed_prompt is not None
        else "N/A (no mixed group)"
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Response tokens (%)")
    ax.set_title("Token efficiency by prompt-group outcome")
    ax.legend(loc="upper right")
    ax.text(
        0.5,
        1.02,
        "signal_token_ratio: "
        f"{summary.effective_token_ratio:.1%} | total_response_tokens: "
        f"{total_tokens} | token_cost_per_mixed_prompt: {cost}",
        ha="center",
        transform=ax.transAxes,
    )
    _save_and_close(plt, fig, path)


def _outcome_group_counts(groups: tuple[PromptGroupMetrics, ...]) -> Counter[str]:
    return Counter(_outcome_name(group) for group in groups)


def _outcome_name(group: PromptGroupMetrics) -> str:
    if group.all_wrong:
        return "all-wrong"
    if group.mixed:
        return "mixed"
    return "all-correct"


def _save_and_close(plt, fig, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fig.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)
