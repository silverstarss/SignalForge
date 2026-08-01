"""Exact numeric verification for extracted rollout answers."""

from __future__ import annotations

from rewardscope.extraction import (
    NumericExtractionConfig,
    extract_numeric_answer,
    parse_numeric_value,
)
from rewardscope.schemas import ExtractionResult, ExtractionStatus, VerificationResult


_EXTRACTION_ERROR_TYPES = {
    ExtractionStatus.AMBIGUOUS: "ambiguous_answer",
    ExtractionStatus.MISSING: "missing_answer",
    ExtractionStatus.PARSE_ERROR: "answer_parse_error",
}


def verify_numeric_answer(
    response: str,
    ground_truth: str,
    *,
    extraction_config: NumericExtractionConfig = NumericExtractionConfig(),
) -> VerificationResult:
    """Extract and exactly compare a model response with a numeric ground truth."""
    extraction = extract_numeric_answer(response, config=extraction_config)
    return verify_extracted_numeric_answer(
        extraction,
        ground_truth,
        percentage_policy=extraction_config.percentage_policy,
    )


def verify_extracted_numeric_answer(
    extraction: ExtractionResult,
    ground_truth: str,
    *,
    percentage_policy: str = "reject",
) -> VerificationResult:
    """Compare an existing numeric extraction result with a ground truth value."""
    if not isinstance(extraction, ExtractionResult):
        raise TypeError("extraction must be an ExtractionResult.")
    if not isinstance(ground_truth, str):
        raise TypeError("ground_truth must be a string.")

    expected_value = parse_numeric_value(
        ground_truth, percentage_policy=percentage_policy
    )
    if expected_value is None:
        return VerificationResult(
            extraction=extraction,
            is_correct=False,
            error_type="invalid_ground_truth",
        )

    if not extraction.extraction_ok:
        return VerificationResult(
            extraction=extraction,
            is_correct=False,
            error_type=_EXTRACTION_ERROR_TYPES.get(
                extraction.extraction_status, "extraction_failed"
            ),
        )

    is_correct = extraction.parsed_value == expected_value
    return VerificationResult(
        extraction=extraction,
        is_correct=is_correct,
        error_type=None if is_correct else "wrong_answer",
    )
