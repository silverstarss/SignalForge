"""Shared data schemas for RewardScope rollout diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from fractions import Fraction
from math import isfinite
from typing import Any, Literal


# Numeric verifier paths use exact Fractions. MATH LaTeX verifier paths store a
# stable expression string because valid answers may be sets, intervals, tuples,
# or symbolic expressions rather than one scalar number.
ParsedMathValue = Fraction | str


class ExtractionStatus(str, Enum):
    """How an answer candidate was identified in a model response."""

    EXPLICIT_FINAL = "explicit_final"
    BOXED = "boxed"
    IMPLICIT_TERMINAL = "implicit_terminal"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"
    PARSE_ERROR = "parse_error"


@dataclass(frozen=True)
class ExtractionCandidate:
    """One answer-like span discovered during numeric extraction."""

    candidate_type: str
    raw_answer: str
    span: tuple[int, int]
    normalized_answer: str | None = None
    parsed_value: ParsedMathValue | None = None
    format_marker_ok: bool = False
    format_ok: bool = False
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.candidate_type not in {"explicit_final", "boxed", "implicit_terminal"}:
            raise ValueError("candidate_type is unsupported.")
        _require_str("raw_answer", self.raw_answer)
        if (
            not isinstance(self.span, tuple)
            or len(self.span) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.span)
            or self.span[0] > self.span[1]
        ):
            raise ValueError("span must be a non-negative (start, end) tuple.")
        _require_bool("format_ok", self.format_ok)
        _require_bool("format_marker_ok", self.format_marker_ok)
        _require_optional_non_empty_str("normalized_answer", self.normalized_answer)
        _require_optional_non_empty_str("rejection_reason", self.rejection_reason)
        if self.parsed_value is None:
            if self.normalized_answer is not None:
                raise ValueError("Rejected candidates cannot have a normalized_answer.")
            if self.format_ok:
                raise ValueError("Rejected candidates cannot be format-compliant.")
        elif not isinstance(self.parsed_value, (Fraction, str)) or (
            isinstance(self.parsed_value, str) and not self.parsed_value
        ):
            raise ValueError("parsed_value must be a Fraction, a non-empty math expression, or None.")


_SUCCESSFUL_EXTRACTION_STATUSES = frozenset(
    {
        ExtractionStatus.EXPLICIT_FINAL,
        ExtractionStatus.BOXED,
        ExtractionStatus.IMPLICIT_TERMINAL,
    }
)


@dataclass(frozen=True)
class ExtractionResult:
    """A candidate answer, its normalized form, and parsed numeric or LaTeX value."""

    raw_answer: str | None
    normalized_answer: str | None
    parsed_value: ParsedMathValue | None
    extraction_status: ExtractionStatus
    format_ok: bool
    all_candidates: tuple[ExtractionCandidate, ...] = ()
    valid_candidates: tuple[ExtractionCandidate, ...] = ()
    rejected_candidates: tuple[ExtractionCandidate, ...] = ()
    selected_candidate: ExtractionCandidate | None = None
    selected_candidate_type: str | None = None
    selected_span: tuple[int, int] | None = None
    ambiguity_reason: str | None = None

    @property
    def extraction_ok(self) -> bool:
        """Whether extraction produced one usable answer candidate."""
        return self.extraction_status in _SUCCESSFUL_EXTRACTION_STATUSES

    def __post_init__(self) -> None:
        _require_bool("format_ok", self.format_ok)

        if self.extraction_ok:
            _require_non_empty_str("raw_answer", self.raw_answer)
            _require_non_empty_str("normalized_answer", self.normalized_answer)
            if not isinstance(self.parsed_value, (Fraction, str)) or (
                isinstance(self.parsed_value, str) and not self.parsed_value
            ):
                raise ValueError(
                    "Successful extraction requires a Fraction or non-empty math expression parsed_value."
                )
        else:
            if self.parsed_value is not None:
                raise ValueError("Failed extraction cannot have a parsed_value.")
            if self.normalized_answer is not None:
                raise ValueError("Failed extraction cannot have a normalized_answer.")

        if self.format_ok and not self.extraction_ok:
            raise ValueError("Failed extraction cannot be format-compliant.")
        for name in ("all_candidates", "valid_candidates", "rejected_candidates"):
            candidates = getattr(self, name)
            if not isinstance(candidates, tuple) or any(
                not isinstance(candidate, ExtractionCandidate) for candidate in candidates
            ):
                raise ValueError(f"{name} must be a tuple of ExtractionCandidate objects.")
        if self.selected_candidate is not None and not isinstance(
            self.selected_candidate, ExtractionCandidate
        ):
            raise ValueError("selected_candidate must be an ExtractionCandidate or None.")
        if self.selected_candidate_type is not None and self.selected_candidate_type not in {
            "explicit_final", "boxed", "implicit_terminal"
        }:
            raise ValueError("selected_candidate_type is unsupported.")
        if self.selected_span is not None:
            ExtractionCandidate("explicit_final", "", self.selected_span)
        _require_optional_non_empty_str("ambiguity_reason", self.ambiguity_reason)
        if self.extraction_ok and self.selected_candidate is not None:
            if self.selected_candidate_type != self.selected_candidate.candidate_type:
                raise ValueError("selected candidate type must match selected_candidate.")
            if self.selected_span != self.selected_candidate.span:
                raise ValueError("selected span must match selected_candidate.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return _to_json_dict(self)


@dataclass(frozen=True)
class VerificationResult:
    """The verifier's correctness decision for one extracted answer."""

    extraction: ExtractionResult
    is_correct: bool
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.extraction, ExtractionResult):
            raise ValueError("extraction must be an ExtractionResult.")
        _require_bool("is_correct", self.is_correct)
        _require_optional_non_empty_str("error_type", self.error_type)

        if self.is_correct and not self.extraction.extraction_ok:
            raise ValueError("A response cannot be correct when extraction failed.")
        if self.is_correct and self.error_type is not None:
            raise ValueError("A correct response cannot have an error_type.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return _to_json_dict(self)


@dataclass(frozen=True)
class RewardBreakdown:
    """Reward components assigned to one rollout."""

    correctness_reward: float
    format_reward: float
    length_penalty: float
    final_reward: float

    def __post_init__(self) -> None:
        _require_finite_number("correctness_reward", self.correctness_reward)
        _require_finite_number("format_reward", self.format_reward)
        _require_finite_number("length_penalty", self.length_penalty)
        _require_finite_number("final_reward", self.final_reward)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return _to_json_dict(self)


@dataclass(frozen=True)
class RolloutRecord:
    """One generated response and its verification, reward, and token metadata."""

    run_id: str
    prompt_id: str
    sample_id: int
    model_name: str
    dataset_name: str
    split: str
    generation_seed: int
    temperature: float
    top_p: float
    max_new_tokens: int
    batch_size: int
    prompt: str
    response: str
    ground_truth: str
    verification: VerificationResult
    reward: RewardBreakdown
    prompt_tokens: int
    response_tokens: int
    hit_max_length: bool
    finish_reason: Literal["eos", "length"] | None = None

    def __post_init__(self) -> None:
        """Validate identifiers, generation settings, and token accounting."""
        for name in ("run_id", "prompt_id", "model_name", "dataset_name", "split"):
            _require_non_empty_str(name, getattr(self, name))
        for name in ("prompt", "response", "ground_truth"):
            _require_str(name, getattr(self, name))

        _require_non_negative_int("sample_id", self.sample_id)
        _require_non_negative_int("generation_seed", self.generation_seed)
        _require_positive_int("max_new_tokens", self.max_new_tokens)
        _require_positive_int("batch_size", self.batch_size)
        _require_non_negative_int("prompt_tokens", self.prompt_tokens)
        _require_non_negative_int("response_tokens", self.response_tokens)
        _require_non_negative_float("temperature", self.temperature)
        _require_probability("top_p", self.top_p)
        _require_bool("hit_max_length", self.hit_max_length)
        finish_reason = self.finish_reason
        if finish_reason is None:
            finish_reason = "length" if self.hit_max_length else "eos"
            object.__setattr__(self, "finish_reason", finish_reason)
        if finish_reason not in {"eos", "length"}:
            raise ValueError("finish_reason must be eos or length.")
        if self.hit_max_length != (finish_reason == "length"):
            raise ValueError("hit_max_length must agree with finish_reason.")

        if not isinstance(self.verification, VerificationResult):
            raise ValueError("verification must be a VerificationResult.")
        if not isinstance(self.reward, RewardBreakdown):
            raise ValueError("reward must be a RewardBreakdown.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return _to_json_dict(self)


def _to_json_dict(instance: object) -> dict[str, Any]:
    return {
        field.name: _to_json_value(getattr(instance, field.name))
        for field in fields(instance)
    }


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return _to_json_dict(value)
    if isinstance(value, tuple | list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json_value(item) for key, item in value.items()}
    return value


def _require_str(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")


def _require_non_empty_str(name: str, value: object) -> None:
    _require_str(name, value)
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_optional_non_empty_str(name: str, value: object) -> None:
    if value is not None:
        _require_non_empty_str(name, value)


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")


def _require_non_negative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_finite_number(name: str, value: object) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number.")


def _require_non_negative_float(name: str, value: object) -> None:
    _require_finite_number(name, value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative number.")


def _require_probability(name: str, value: object) -> None:
    _require_finite_number(name, value)
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be a number in the interval (0, 1].")
