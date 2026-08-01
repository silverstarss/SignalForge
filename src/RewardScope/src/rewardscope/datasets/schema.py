"""Normalized dataset example contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ChatMessage:
    """One validated role/content pair for a chat-formatted model input."""

    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("role must be system, user, or assistant.")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be a non-empty string.")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class DatasetExample:
    """One dataset item ready to become one prompt's rollout group."""

    dataset_name: str
    split: str
    source_index: int
    prompt_id: str
    question: str
    prompt: str
    ground_truth: str
    reference_solution: str
    messages: tuple[ChatMessage, ...] | None = None

    def __post_init__(self) -> None:
        for name in (
            "dataset_name",
            "split",
            "prompt_id",
            "question",
            "prompt",
            "ground_truth",
            "reference_solution",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if (
            not isinstance(self.source_index, int)
            or isinstance(self.source_index, bool)
            or self.source_index < 0
        ):
            raise ValueError("source_index must be a non-negative integer.")
        if self.messages is not None:
            if not isinstance(self.messages, tuple) or not self.messages:
                raise ValueError("messages must be a non-empty tuple of ChatMessage objects or None.")
            if any(not isinstance(message, ChatMessage) for message in self.messages):
                raise ValueError("messages must be a non-empty tuple of ChatMessage objects or None.")
            if self.messages[-1].role != "user":
                raise ValueError("messages must end with a user message for generation.")


@dataclass(frozen=True)
class DatasetLoadResult:
    """Selected examples plus reproducibility metadata from a dataset adapter."""

    examples: tuple[DatasetExample, ...]
    source_count: int
    fingerprint: str | None
    gold_parse_attempt_count: int | None = None
    gold_parse_failure_count: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_count, int)
            or isinstance(self.source_count, bool)
            or self.source_count < 0
        ):
            raise ValueError("source_count must be a non-negative integer.")
        if self.fingerprint is not None and (
            not isinstance(self.fingerprint, str) or not self.fingerprint
        ):
            raise ValueError("fingerprint must be a non-empty string or None.")
        for name in ("gold_parse_attempt_count", "gold_parse_failure_count"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")
        if (
            self.gold_parse_failure_count is not None
            and self.gold_parse_attempt_count is None
        ):
            raise ValueError("gold_parse_attempt_count is required when gold_parse_failure_count is set.")
        if (
            self.gold_parse_attempt_count is not None
            and self.gold_parse_failure_count is not None
            and self.gold_parse_failure_count > self.gold_parse_attempt_count
        ):
            raise ValueError("gold_parse_failure_count cannot exceed gold_parse_attempt_count.")

    @property
    def gold_parse_failure_rate(self) -> float | None:
        if self.gold_parse_attempt_count is None:
            return None
        if self.gold_parse_attempt_count == 0:
            return 0.0
        assert self.gold_parse_failure_count is not None
        return self.gold_parse_failure_count / self.gold_parse_attempt_count
