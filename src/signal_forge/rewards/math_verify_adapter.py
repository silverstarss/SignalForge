"""Thin veRL reward adapter over the frozen RewardScope Math-Verify verifier.

The adapter intentionally owns no grading logic. It only maps veRL's reward
function signature to RewardScope's frozen training verifier and returns the
flat dict format consumed by this repository's NaiveRewardManager.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


_GSM8K_SOURCES = frozenset({"gsm8k", "openai/gsm8k"})
_MATH_LEVEL_3_SOURCES = frozenset({"math_level_3", "math", "hendrycks_math", "competition_math"})


@lru_cache(maxsize=1)
def _numeric_verifier():
    from rewardscope.verification.math_verify import MathVerifyNumericVerifier

    return MathVerifyNumericVerifier(mode="training")


@lru_cache(maxsize=1)
def _latex_verifier():
    from rewardscope.verification.math_verify import MathVerifyLatexVerifier

    return MathVerifyLatexVerifier(mode="training")


def _canonical_source(data_source: str, extra_info: dict[str, Any] | None) -> str:
    source = str(data_source)
    if source in _GSM8K_SOURCES:
        return "gsm8k"
    if source in _MATH_LEVEL_3_SOURCES:
        return "math_level_3"

    source_dataset = ""
    if extra_info:
        source_dataset = str(extra_info.get("source_dataset") or extra_info.get("dataset_name") or "")
    if source_dataset == "gsm8k":
        return "gsm8k"
    if source_dataset == "math":
        return "math_level_3"

    raise ValueError(f"Unsupported data_source for Math-Verify reward: {data_source!r}")


def _verify(source: str, solution_str: str, ground_truth: str):
    if source == "gsm8k":
        return _numeric_verifier().verify(response=solution_str, ground_truth=ground_truth)
    if source == "math_level_3":
        return _latex_verifier().verify(response=solution_str, ground_truth=ground_truth)
    raise AssertionError(f"Unexpected canonical source: {source}")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a binary Math-Verify reward and diagnostic fields for veRL.

    Parser and verifier exceptions intentionally propagate. A verifier/library
    failure is a training correctness bug, not an ordinary wrong answer.
    """
    source = _canonical_source(data_source, extra_info)
    result = _verify(source=source, solution_str=solution_str, ground_truth=ground_truth)
    extraction = result.extraction
    raw_correctness = float(result.is_correct)

    return {
        "score": raw_correctness,
        "acc": raw_correctness,
        "raw_correctness": raw_correctness,
        "extraction_ok": bool(extraction.extraction_ok),
        "format_ok": bool(extraction.format_ok),
        "verification_status": extraction.extraction_status.value,
        "verification_error_type": result.error_type or "",
        "predicted_answer": extraction.normalized_answer or "",
        "raw_answer": extraction.raw_answer or "",
        "reward_source": source,
    }
