"""Pure HIVE Stage-2 entropy-band selection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Generic, Protocol, TypeVar


class EntropyScoredPrompt(Protocol):
    """Structural input contract shared with PromptEntropyRecord."""

    @property
    def prompt_id(self) -> str: ...

    @property
    def entropy(self) -> float: ...


EntropyRecordT = TypeVar("EntropyRecordT", bound=EntropyScoredPrompt)


@dataclass(frozen=True)
class Stage2PromptRecord:
    prompt_id: str
    entropy: float


@dataclass(frozen=True)
class Stage2Config:
    upper_trim_ratio: float = 0.25
    keep_ratio: float = 0.50
    group_size: int = 8

    def __post_init__(self) -> None:
        upper_trim_ratio = _validate_ratio(
            "upper_trim_ratio",
            self.upper_trim_ratio,
            allow_zero=True,
            allow_one=False,
        )
        keep_ratio = _validate_ratio(
            "keep_ratio",
            self.keep_ratio,
            allow_zero=False,
            allow_one=True,
        )
        if upper_trim_ratio + keep_ratio > 1.0:
            raise ValueError("upper_trim_ratio + keep_ratio must be at most 1")
        if isinstance(self.group_size, bool) or not isinstance(self.group_size, int) or self.group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        object.__setattr__(self, "upper_trim_ratio", upper_trim_ratio)
        object.__setattr__(self, "keep_ratio", keep_ratio)


@dataclass(frozen=True)
class Stage2Diagnostics:
    input_count: int
    requested_upper_trim_count: int
    actual_upper_trim_count: int
    requested_keep_count: int
    pre_round_keep_count: int
    post_round_keep_count: int
    rounding_dropped_count: int
    low_entropy_reject_count: int


@dataclass(frozen=True)
class Stage2BatchResult(Generic[EntropyRecordT]):
    kept: tuple[EntropyRecordT, ...]
    upper_trimmed: tuple[EntropyRecordT, ...]
    low_entropy_rejected: tuple[EntropyRecordT, ...]
    rounding_dropped: tuple[EntropyRecordT, ...]
    diagnostics: Stage2Diagnostics

    @property
    def pre_round_kept(self) -> tuple[EntropyRecordT, ...]:
        return self.kept + self.rounding_dropped


def compute_stage2_counts(
    input_count: int,
    config: Stage2Config | None = None,
) -> Stage2Diagnostics:
    """Centralize floor-based band counts and paper-faithful G rounding."""
    if isinstance(input_count, bool) or not isinstance(input_count, int) or input_count < 0:
        raise ValueError("input_count must be a non-negative integer")
    resolved_config = config if config is not None else Stage2Config()
    if not isinstance(resolved_config, Stage2Config):
        raise TypeError("config must be a Stage2Config")

    requested_upper_trim_count = math.floor(input_count * resolved_config.upper_trim_ratio)
    requested_keep_count = math.floor(input_count * resolved_config.keep_ratio)
    actual_upper_trim_count = min(requested_upper_trim_count, input_count)
    available_after_upper_trim = input_count - actual_upper_trim_count
    pre_round_keep_count = min(requested_keep_count, available_after_upper_trim)
    post_round_keep_count = (
        pre_round_keep_count // resolved_config.group_size
    ) * resolved_config.group_size
    rounding_dropped_count = pre_round_keep_count - post_round_keep_count
    low_entropy_reject_count = input_count - actual_upper_trim_count - pre_round_keep_count

    return Stage2Diagnostics(
        input_count=input_count,
        requested_upper_trim_count=requested_upper_trim_count,
        actual_upper_trim_count=actual_upper_trim_count,
        requested_keep_count=requested_keep_count,
        pre_round_keep_count=pre_round_keep_count,
        post_round_keep_count=post_round_keep_count,
        rounding_dropped_count=rounding_dropped_count,
        low_entropy_reject_count=low_entropy_reject_count,
    )


class Stage2Selector:
    """Select the deterministic middle-high prompt-entropy band."""

    def __init__(self, config: Stage2Config | None = None):
        self.config = config if config is not None else Stage2Config()
        if not isinstance(self.config, Stage2Config):
            raise TypeError("config must be a Stage2Config")

    def select(
        self,
        records: Sequence[EntropyRecordT],
    ) -> Stage2BatchResult[EntropyRecordT]:
        validated_records = _validate_records(records)
        diagnostics = compute_stage2_counts(len(validated_records), self.config)
        ranked = tuple(sorted(validated_records, key=lambda record: (-float(record.entropy), record.prompt_id)))

        upper_end = diagnostics.actual_upper_trim_count
        band_end = upper_end + diagnostics.pre_round_keep_count
        upper_trimmed = ranked[:upper_end]
        retained_band = ranked[upper_end:band_end]
        kept = retained_band[: diagnostics.post_round_keep_count]
        rounding_dropped = retained_band[diagnostics.post_round_keep_count :]
        low_entropy_rejected = ranked[band_end:]

        return Stage2BatchResult(
            kept=kept,
            upper_trimmed=upper_trimmed,
            low_entropy_rejected=low_entropy_rejected,
            rounding_dropped=rounding_dropped,
            diagnostics=diagnostics,
        )


def _validate_ratio(
    name: str,
    value: Real,
    *,
    allow_zero: bool,
    allow_one: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    lower_valid = normalized >= 0.0 if allow_zero else normalized > 0.0
    upper_valid = normalized <= 1.0 if allow_one else normalized < 1.0
    if not lower_valid or not upper_valid:
        raise ValueError(f"{name} is outside its allowed unit interval")
    return normalized


def _validate_records(records: Sequence[EntropyRecordT]) -> tuple[EntropyRecordT, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of entropy-scored prompts")
    normalized = tuple(records)
    seen_prompt_ids: set[str] = set()
    for index, record in enumerate(normalized):
        prompt_id = getattr(record, "prompt_id", None)
        entropy = getattr(record, "entropy", None)
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompt_id at index {index} must be a non-empty string")
        if prompt_id in seen_prompt_ids:
            raise ValueError(f"duplicate prompt_id {prompt_id!r} in Stage-2 input")
        seen_prompt_ids.add(prompt_id)
        if isinstance(entropy, bool) or not isinstance(entropy, Real):
            raise ValueError(f"entropy for prompt {prompt_id!r} must be a real number")
        if not math.isfinite(float(entropy)):
            raise ValueError(f"entropy for prompt {prompt_id!r} must be finite")
    return normalized
