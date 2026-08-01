"""Analysis report generation utilities."""

from rewardscope.reports.analysis import (
    AnalysisArtifacts,
    analyze_rollouts_jsonl,
    write_analysis_report,
)
from rewardscope.reports.plots import AnalysisPlotArtifacts, write_analysis_plots
from rewardscope.reports.rescore import (
    OfflineRescoreArtifacts,
    rescore_completed_run,
    rescore_completed_run_with_math_verify,
)
from rewardscope.reports.comparison import RolloutComparisonArtifacts, compare_rollouts_jsonl

__all__ = [
    "AnalysisArtifacts",
    "analyze_rollouts_jsonl",
    "write_analysis_report",
    "AnalysisPlotArtifacts",
    "write_analysis_plots",
    "OfflineRescoreArtifacts",
    "rescore_completed_run",
    "rescore_completed_run_with_math_verify",
    "RolloutComparisonArtifacts",
    "compare_rollouts_jsonl",
]
