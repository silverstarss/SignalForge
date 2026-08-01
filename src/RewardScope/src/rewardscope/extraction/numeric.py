"""Exact numeric extraction with auditable candidate selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from rewardscope.schemas import (
    ExtractionCandidate,
    ExtractionResult,
    ExtractionStatus,
)


PercentagePolicy = Literal["literal", "fraction", "reject"]


@dataclass(frozen=True)
class NumericExtractionConfig:
    """Parsing policy for answer decorations with potentially ambiguous meaning."""

    percentage_policy: PercentagePolicy = "reject"

    def __post_init__(self) -> None:
        if self.percentage_policy not in {"literal", "fraction", "reject"}:
            raise ValueError("percentage_policy must be literal, fraction, or reject.")


_EXPLICIT_PATTERNS = (
    re.compile(r"(?i)^\s*(?:the\s+)?final\s+answer\s*(?:is|:|=)\s*(?P<answer>.+?)\s*$"),
    re.compile(r"(?i)^\s*(?:the\s+)?answer\s*(?:is|:|=)\s*(?P<answer>.+?)\s*$"),
    re.compile(r"^\s*####\s*(?P<answer>.+?)\s*$"),
)
_TERMINAL_MARKED_ANSWER_PATTERN = re.compile(
    r"(?i)(?:^|[.!?]\s+|,\s+)(?P<marker>(?:the\s+)?(?:final\s+)?answer)\s*"
    r"(?P<separator>is|:|=)\s*(?P<answer>.+?)\s*$"
)
_TERMINAL_HASH_ANSWER_PATTERN = re.compile(r"####\s*(?P<answer>.+?)\s*$")
_LATEX_FRACTION_PATTERN = re.compile(r"\\(?:d?frac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_NUMBER_PATTERN = re.compile(_NUMBER)
_GROUPED_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+)(?:\.\d*)?")
_SIMPLE_FRACTION_PATTERN = re.compile(rf"(?P<numerator>{_NUMBER})\s*/\s*(?P<denominator>{_NUMBER})")
_NUMERIC_CANDIDATE_PATTERN = re.compile(rf"(?<![\w.])({_NUMBER}(?:\s*/\s*{_NUMBER})?)(?![\w.])")
_UNIT_PATTERN = re.compile(
    r"(?i)\s+(?:"
    r"cubic\s+inches?|miles?(?:\s+per\s+hour)?|"
    r"meters?|minutes?|hours?|seconds?|days?|weeks?|months?|years?|"
    r"pounds?|lbs?|liters?|cups?|dollars?|eggs?|glasses?|"
    r"bolts?|dozens?|boxes?|points?|oranges?|bags?|tomatoes?|"
    r"containers?|sheets?|thorns?|gb"
    r")(?:\s+of\s+[a-z][a-z -]*)?\s*$"
)


def extract_numeric_answer(
    response: str, *, config: NumericExtractionConfig = NumericExtractionConfig()
) -> ExtractionResult:
    """Collect, parse, and select answer candidates without early parse failure."""
    if not isinstance(response, str):
        raise TypeError("response must be a string.")
    if not isinstance(config, NumericExtractionConfig):
        raise TypeError("config must be a NumericExtractionConfig.")

    candidates = _collect_candidates(response)
    parsed_candidates = tuple(_parse_candidate(candidate, config) for candidate in candidates)
    valid = tuple(candidate for candidate in parsed_candidates if candidate.parsed_value is not None)
    rejected = tuple(candidate for candidate in parsed_candidates if candidate.parsed_value is None)

    for candidate_type, status in (
        ("explicit_final", ExtractionStatus.EXPLICIT_FINAL),
        ("boxed", ExtractionStatus.BOXED),
        ("implicit_terminal", ExtractionStatus.IMPLICIT_TERMINAL),
    ):
        tier = tuple(candidate for candidate in valid if candidate.candidate_type == candidate_type)
        if not tier:
            continue
        values = {candidate.parsed_value for candidate in tier}
        if len(values) > 1:
            return _failed_result(
                ExtractionStatus.AMBIGUOUS,
                all_candidates=parsed_candidates,
                valid_candidates=valid,
                rejected_candidates=rejected,
                ambiguity_reason=f"conflicting_{candidate_type}_candidates",
            )
        selected = max(tier, key=lambda candidate: candidate.span[1])
        return ExtractionResult(
            raw_answer=selected.raw_answer,
            normalized_answer=selected.normalized_answer,
            parsed_value=selected.parsed_value,
            extraction_status=status,
            format_ok=selected.format_ok,
            all_candidates=parsed_candidates,
            valid_candidates=valid,
            rejected_candidates=rejected,
            selected_candidate=selected,
            selected_candidate_type=selected.candidate_type,
            selected_span=selected.span,
        )

    if rejected:
        return _failed_result(
            ExtractionStatus.PARSE_ERROR,
            raw_answer=rejected[-1].raw_answer,
            all_candidates=parsed_candidates,
            valid_candidates=valid,
            rejected_candidates=rejected,
        )
    if len(_NUMERIC_CANDIDATE_PATTERN.findall(response)) >= 2:
        return _failed_result(
            ExtractionStatus.AMBIGUOUS,
            all_candidates=parsed_candidates,
            valid_candidates=valid,
            rejected_candidates=rejected,
            ambiguity_reason="multiple_unmarked_numeric_values",
        )
    return _failed_result(
        ExtractionStatus.MISSING,
        all_candidates=parsed_candidates,
        valid_candidates=valid,
        rejected_candidates=rejected,
    )


def parse_numeric_value(
    raw_answer: str, *, percentage_policy: PercentagePolicy = "reject"
) -> Fraction | None:
    """Parse one conservative scalar numeric candidate into an exact value."""
    if not isinstance(raw_answer, str):
        raise TypeError("raw_answer must be a string.")
    return _parse_raw_answer(raw_answer, percentage_policy)[0]


def _collect_candidates(response: str) -> tuple[ExtractionCandidate, ...]:
    candidates: list[ExtractionCandidate] = []
    offset = 0
    for line in response.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        for pattern_index, pattern in enumerate(_EXPLICIT_PATTERNS):
            match = pattern.search(content)
            if match:
                start, end = match.span("answer")
                candidates.append(
                    ExtractionCandidate(
                        "explicit_final",
                        match.group("answer"),
                        (offset + start, offset + end),
                        format_marker_ok=_explicit_pattern_is_format_compliant(
                            pattern_index, content
                        ),
                    )
                )
                break
        offset += len(line)
    candidates.extend(_collect_boxed_candidates(response))
    terminal = _last_non_empty_line_with_span(response)
    if terminal is not None:
        line, start, end = terminal
        candidates.extend(_collect_terminal_marked_candidates(line, start))
        if line.count("=") == 1:
            _, right_hand_side = line.split("=", maxsplit=1)
            candidate = right_hand_side.strip()
            if candidate and _is_terminal_candidate_shape(candidate):
                candidate_start = start + line.rfind(candidate)
                candidates.append(ExtractionCandidate("implicit_terminal", candidate, (candidate_start, candidate_start + len(candidate))))
        elif _is_terminal_candidate_shape(line):
            candidates.append(ExtractionCandidate("implicit_terminal", line, (start, end)))
    return tuple(
        sorted(
            {
                (candidate.candidate_type, candidate.raw_answer, candidate.span): candidate
                for candidate in candidates
            }.values(),
            key=lambda candidate: candidate.span,
        )
    )


def _collect_terminal_marked_candidates(
    line: str, line_start: int
) -> list[ExtractionCandidate]:
    """Recognize a marked final answer only at the end of the final response line."""
    candidates: list[ExtractionCandidate] = []
    explicit_match = _TERMINAL_MARKED_ANSWER_PATTERN.search(line)
    if explicit_match:
        start, end = explicit_match.span("answer")
        candidates.append(
            ExtractionCandidate(
                "explicit_final",
                explicit_match.group("answer"),
                (line_start + start, line_start + end),
                format_marker_ok=(
                    explicit_match.group("marker").strip().lower() != "the answer"
                ),
            )
        )
    hash_match = _TERMINAL_HASH_ANSWER_PATTERN.search(line)
    if hash_match:
        start, end = hash_match.span("answer")
        candidates.append(
            ExtractionCandidate(
                "explicit_final",
                hash_match.group("answer"),
                (line_start + start, line_start + end),
                format_marker_ok=True,
            )
        )
    return candidates


def _collect_boxed_candidates(response: str) -> list[ExtractionCandidate]:
    candidates: list[ExtractionCandidate] = []
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
        content_end = index - 1 if depth == 0 else len(response)
        candidates.append(
            ExtractionCandidate(
                "boxed",
                response[content_start:content_end],
                (content_start, content_end),
                format_marker_ok=True,
            )
        )
        cursor = index if depth == 0 else len(response)
    return candidates


def _parse_candidate(candidate: ExtractionCandidate, config: NumericExtractionConfig) -> ExtractionCandidate:
    value, normalized, surface_format_ok, rejection_reason = _parse_raw_answer(
        candidate.raw_answer, config.percentage_policy
    )
    if value is None:
        return ExtractionCandidate(
            candidate.candidate_type, candidate.raw_answer, candidate.span,
            format_marker_ok=candidate.format_marker_ok,
            rejection_reason=rejection_reason,
        )
    return ExtractionCandidate(
        candidate.candidate_type, candidate.raw_answer, candidate.span,
        normalized_answer=normalized, parsed_value=value,
        format_marker_ok=candidate.format_marker_ok,
        format_ok=(candidate.format_marker_ok and surface_format_ok),
    )


def _explicit_pattern_is_format_compliant(pattern_index: int, line: str) -> bool:
    """Accept established final markers, but not the informal "The answer is" form."""
    if pattern_index in {0, 2}:
        return True
    return not re.match(r"(?i)^\s*the\s+answer\s+is\b", line)


def _parse_raw_answer(
    raw_answer: str, percentage_policy: PercentagePolicy
) -> tuple[Fraction | None, str | None, bool, str | None]:
    if percentage_policy not in {"literal", "fraction", "reject"}:
        raise ValueError("percentage_policy must be literal, fraction, or reject.")
    candidate = _strip_terminal_period(raw_answer.strip())
    if not candidate:
        return None, None, False, "empty_candidate"

    candidate, has_math_delimiters = _unwrap_math_delimiters(candidate)
    surface_format_ok = not has_math_delimiters
    has_percent = candidate.endswith("%")
    if has_percent:
        if percentage_policy == "reject":
            return None, None, False, "percentage_rejected"
        candidate = candidate[:-1].strip()
        surface_format_ok = False

    unit_match = _UNIT_PATTERN.search(candidate)
    if unit_match:
        candidate = candidate[: unit_match.start()].strip()
        surface_format_ok = False

    if candidate.startswith(r"\$"):
        candidate = candidate[2:].strip()
        surface_format_ok = False
    elif candidate.startswith("$"):
        candidate = candidate[1:].strip()
        surface_format_ok = False

    if _GROUPED_NUMBER_PATTERN.fullmatch(candidate):
        candidate = candidate.replace(",", "")
        surface_format_ok = False

    value = _parse_plain_scalar(candidate)
    if value is None:
        return None, None, False, "unsupported_numeric_surface"
    if has_percent and percentage_policy == "fraction":
        value /= 100
    return value, str(value), surface_format_ok, None


def _parse_plain_scalar(candidate: str) -> Fraction | None:
    latex_fraction = _LATEX_FRACTION_PATTERN.fullmatch(candidate)
    if latex_fraction:
        numerator, denominator = latex_fraction.groups()
        try:
            return Fraction(int(numerator), int(denominator))
        except ZeroDivisionError:
            return None
    simple_fraction = _SIMPLE_FRACTION_PATTERN.fullmatch(candidate)
    if simple_fraction:
        numerator = Fraction(simple_fraction.group("numerator"))
        denominator = Fraction(simple_fraction.group("denominator"))
        return None if denominator == 0 else numerator / denominator
    return Fraction(candidate) if _NUMBER_PATTERN.fullmatch(candidate) else None


def _is_terminal_candidate_shape(candidate: str) -> bool:
    value, _, _, _ = _parse_raw_answer(candidate, "literal")
    return value is not None


def _last_non_empty_line_with_span(response: str) -> tuple[str, int, int] | None:
    cursor = len(response)
    for line in reversed(response.splitlines(keepends=True)):
        cursor -= len(line)
        content = line.rstrip("\r\n")
        if content.strip():
            start = cursor + len(content) - len(content.lstrip())
            stripped = content.strip()
            return stripped, start, start + len(stripped)
    return None


def _strip_terminal_period(candidate: str) -> str:
    return candidate[:-1].rstrip() if candidate.endswith(".") else candidate


def _unwrap_math_delimiters(candidate: str) -> tuple[str, bool]:
    wrappers = ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"))
    for prefix, suffix in wrappers:
        if candidate.startswith(prefix) and candidate.endswith(suffix):
            return candidate[len(prefix) : -len(suffix)].strip(), True
    return candidate, False


def _failed_result(
    extraction_status: ExtractionStatus,
    *,
    raw_answer: str | None = None,
    all_candidates: tuple[ExtractionCandidate, ...],
    valid_candidates: tuple[ExtractionCandidate, ...],
    rejected_candidates: tuple[ExtractionCandidate, ...],
    ambiguity_reason: str | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        raw_answer=raw_answer,
        normalized_answer=None,
        parsed_value=None,
        extraction_status=extraction_status,
        format_ok=False,
        all_candidates=all_candidates,
        valid_candidates=valid_candidates,
        rejected_candidates=rejected_candidates,
        ambiguity_reason=ambiguity_reason,
    )
