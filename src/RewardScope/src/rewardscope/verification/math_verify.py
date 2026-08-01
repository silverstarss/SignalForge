"""Math-Verify-backed verification for GSM8K numeric and MATH LaTeX answers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from rewardscope.schemas import (
    ExtractionCandidate,
    ExtractionResult,
    ExtractionStatus,
    VerificationResult,
)


VerificationMode = Literal["evaluation", "training"]


@dataclass(frozen=True)
class MathVerifyNumericVerifier:
    """Verify numerical expressions with Math-Verify under an explicit protocol."""

    mode: VerificationMode = "evaluation"

    def __post_init__(self) -> None:
        if self.mode not in {"evaluation", "training"}:
            raise ValueError("mode must be evaluation or training.")

    def verify(self, response: str, ground_truth: str) -> VerificationResult:
        """Extract one mathematical prediction and compare it with a numeric gold value."""
        if not isinstance(response, str):
            raise TypeError("response must be a string.")
        if not isinstance(ground_truth, str):
            raise TypeError("ground_truth must be a string.")

        backend = _load_math_verify_backend()
        gold = _parse_gold_expression(backend, ground_truth)
        extraction, prediction = _extract_prediction(backend, response, self.mode)
        if not extraction.extraction_ok or prediction is None:
            return VerificationResult(
                extraction=extraction,
                is_correct=False,
                error_type=_failure_error_type(extraction.extraction_status, self.mode),
            )

        # Math-Verify owns equivalence; exceptions intentionally propagate.
        is_correct = bool(backend.verify(gold, prediction,timeout_seconds=None))
        return VerificationResult(
            extraction=extraction,
            is_correct=is_correct,
            error_type=None if is_correct else "wrong_answer",
        )


def verify_math_verify_numeric_answer(
    response: str,
    ground_truth: str,
    *,
    mode: VerificationMode = "evaluation",
) -> VerificationResult:
    """Convenience wrapper for one Math-Verify-backed numeric decision."""
    return MathVerifyNumericVerifier(mode=mode).verify(response, ground_truth)


@dataclass(frozen=True)
class MathVerifyLatexVerifier:
    """Strict boxed-answer verifier for LaTeX MATH gold answers."""

    mode: VerificationMode = "training"

    def __post_init__(self) -> None:
        if self.mode not in {"evaluation", "training"}:
            raise ValueError("mode must be evaluation or training.")

    def verify(self, response: str, ground_truth: str) -> VerificationResult:
        """Compare a boxed model answer with a clean boxed LaTeX gold value."""
        if not isinstance(response, str):
            raise TypeError("response must be a string.")
        if not isinstance(ground_truth, str):
            raise TypeError("ground_truth must be a string.")

        backend = _load_math_verify_backend()
        gold = _parse_latex_expression(backend, ground_truth, context="MATH ground_truth")
        extraction, prediction = _extract_latex_boxed_prediction(backend, response)
        if not extraction.extraction_ok or prediction is None:
            return VerificationResult(
                extraction=extraction,
                is_correct=False,
                error_type=_failure_error_type(extraction.extraction_status, "training"),
            )
        is_correct = bool(backend.verify(gold, prediction,timeout_seconds=None))
        return VerificationResult(
            extraction=extraction,
            is_correct=is_correct,
            error_type=None if is_correct else "wrong_answer",
        )


def verify_math_verify_latex_answer(
    response: str,
    ground_truth: str,
    *,
    mode: VerificationMode = "training",
) -> VerificationResult:
    """Verify a boxed model answer against a clean LaTeX MATH gold answer."""
    return MathVerifyLatexVerifier(mode=mode).verify(response, ground_truth)


@dataclass(frozen=True)
class _MathVerifyBackend:
    parse: Any
    verify: Any
    LatexExtractionConfig: Any
    ExprExtractionConfig: Any


def _load_math_verify_backend() -> _MathVerifyBackend:
    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'Math-Verify verification requires the optional dependency. Run: pip install -e ".[math]"'
        ) from error
    return _MathVerifyBackend(
        parse=parse,
        verify=verify,
        LatexExtractionConfig=LatexExtractionConfig,
        ExprExtractionConfig=ExprExtractionConfig,
    )


def _parse_gold_expression(backend: _MathVerifyBackend, ground_truth: str) -> list[Any]:
    expressions = backend.parse(
        ground_truth,
        extraction_config=[backend.ExprExtractionConfig()],
        parsing_timeout=None,
        raise_on_error=True,
    )
    if not expressions:
        raise ValueError("GSM8K ground_truth must be a parseable pure numeric expression.")
    if _fraction_from_expression(expressions[0]) is None:
        raise ValueError("GSM8K ground_truth must be a rational numeric expression.")
    return expressions


def extract_final_boxed_latex_gold(solution: str) -> str | None:
    """Return the final parseable boxed MATH answer, or ``None`` when unusable.

    MATH solutions often contain intermediate boxes. Gold is deliberately taken
    from the final box only, so malformed endings are filtered instead of
    accidentally rewarding an earlier intermediate result.
    """
    if not isinstance(solution, str):
        raise TypeError("solution must be a string.")
    boxed = _boxed_contents(solution)
    if not boxed:
        return None
    raw_answer = boxed[-1]
    backend = _load_math_verify_backend()
    try:
        _parse_latex_expression(backend, rf"\boxed{{{raw_answer}}}", context="MATH solution")
    except Exception:
        return None
    return rf"\boxed{{{raw_answer}}}"


def _parse_latex_expression(
    backend: _MathVerifyBackend, latex: str, *, context: str
) -> list[Any]:
    expressions = backend.parse(
        latex,
        extraction_config=[backend.LatexExtractionConfig(boxed_match_priority=0)],
        parsing_timeout=None,
        raise_on_error=True,
    )
    if not expressions:
        raise ValueError(f"{context} must contain a parseable LaTeX expression.")
    return expressions


def _extract_prediction(
    backend: _MathVerifyBackend,
    response: str,
    mode: VerificationMode,
) -> tuple[ExtractionResult, list[Any] | None]:
    boxed = _last_parseable_boxed_expression(backend, response)
    if boxed is not None:
        raw_answer, prediction = boxed
        return _successful_extraction(
            raw_answer=raw_answer,
            expression=prediction[0],
            status=ExtractionStatus.BOXED,
            format_ok=True,
        ), prediction

    if mode == "training":
        status = ExtractionStatus.PARSE_ERROR if _boxed_contents(response) else ExtractionStatus.MISSING
        return _failed_extraction(status), None

    expressions = backend.parse(
        response,
        extraction_config=[
            backend.LatexExtractionConfig(boxed_match_priority=0),
            backend.ExprExtractionConfig(),
        ],
        parsing_timeout=None,
        raise_on_error=True,
    )
    if not expressions:
        return _failed_extraction(ExtractionStatus.MISSING), None
    fraction = _fraction_from_expression(expressions[0])
    if fraction is None:
        return _failed_extraction(ExtractionStatus.PARSE_ERROR), None
    return _successful_extraction(
        raw_answer=response,
        expression=expressions[0],
        status=ExtractionStatus.EXPLICIT_FINAL,
        format_ok=False,
    ), expressions


def _last_parseable_boxed_expression(
    backend: _MathVerifyBackend, response: str
) -> tuple[str, list[Any]] | None:
    for raw_answer in reversed(_boxed_contents(response)):
        expressions = backend.parse(
            rf"\boxed{{{raw_answer}}}",
            extraction_config=[backend.LatexExtractionConfig(boxed_match_priority=0)],
            parsing_timeout=None,
            raise_on_error=True,
        )
        if expressions and _fraction_from_expression(expressions[0]) is not None:
            return raw_answer, expressions
    return None


def _extract_latex_boxed_prediction(
    backend: _MathVerifyBackend, response: str
) -> tuple[ExtractionResult, list[Any] | None]:
    for raw_answer in reversed(_boxed_contents(response)):
        try:
            expressions = _parse_latex_expression(
                backend, rf"\boxed{{{raw_answer}}}", context="boxed prediction"
            )
        except ValueError:
            continue
        if expressions:
            expression = expressions[0]
            normalized = str(expression)
            candidate = ExtractionCandidate(
                candidate_type="boxed",
                raw_answer=raw_answer,
                span=(0, len(raw_answer)),
                normalized_answer=normalized,
                parsed_value=normalized,
                format_marker_ok=True,
                format_ok=True,
            )
            return ExtractionResult(
                raw_answer=raw_answer,
                normalized_answer=normalized,
                parsed_value=normalized,
                extraction_status=ExtractionStatus.BOXED,
                format_ok=True,
                all_candidates=(candidate,),
                valid_candidates=(candidate,),
                selected_candidate=candidate,
                selected_candidate_type="boxed",
                selected_span=candidate.span,
            ), expressions
    status = ExtractionStatus.PARSE_ERROR if _boxed_contents(response) else ExtractionStatus.MISSING
    return _failed_extraction(status), None


def _boxed_contents(response: str) -> list[str]:
    contents: list[str] = []
    cursor = 0
    marker = r"\boxed{"
    while (start := response.find(marker, cursor)) != -1:
        content_start = start + len(marker)
        depth = 1
        index = content_start
        while index < len(response) and depth:
            if response[index] == "{":
                depth += 1
            elif response[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            contents.append(response[content_start : index - 1])
            cursor = index
        else:
            cursor = len(response)
    return contents


def _successful_extraction(
    *, raw_answer: str,
    expression: Any,
    status: ExtractionStatus,
    format_ok: bool,
) -> ExtractionResult:
    parsed_value = _fraction_from_expression(expression)
    if parsed_value is None:
        raise ValueError("Math-Verify prediction must be a rational numeric expression.")
    candidate_type = "boxed" if status is ExtractionStatus.BOXED else "explicit_final"
    candidate = ExtractionCandidate(
        candidate_type=candidate_type,
        raw_answer=raw_answer,
        span=(0, len(raw_answer)),
        normalized_answer=str(expression),
        parsed_value=parsed_value,
        format_marker_ok=format_ok,
        format_ok=format_ok,
    )
    return ExtractionResult(
        raw_answer=raw_answer,
        normalized_answer=str(expression),
        parsed_value=parsed_value,
        extraction_status=status,
        format_ok=format_ok,
        all_candidates=(candidate,),
        valid_candidates=(candidate,),
        selected_candidate=candidate,
        selected_candidate_type=candidate_type,
        selected_span=candidate.span,
    )


def _failed_extraction(status: ExtractionStatus) -> ExtractionResult:
    return ExtractionResult(
        raw_answer=None,
        normalized_answer=None,
        parsed_value=None,
        extraction_status=status,
        format_ok=False,
    )


def _fraction_from_expression(expression: Any) -> Fraction | None:
    if isinstance(expression, Fraction):
        return expression
    if isinstance(expression, int) and not isinstance(expression, bool):
        return Fraction(expression)
    try:
        numerator, denominator = expression.as_numer_denom()
        if getattr(numerator, "is_Integer", False) and getattr(denominator, "is_Integer", False):
            return Fraction(int(numerator), int(denominator))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        pass
    try:
        return Fraction(str(expression))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _failure_error_type(status: ExtractionStatus, mode: VerificationMode) -> str:
    if status is ExtractionStatus.PARSE_ERROR:
        return "boxed_answer_parse_error" if mode == "training" else "answer_parse_error"
    return "missing_boxed_answer" if mode == "training" else "missing_answer"
