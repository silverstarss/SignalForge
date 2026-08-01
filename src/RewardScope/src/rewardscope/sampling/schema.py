"""Model-generation response contracts independent of any model runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FinishReason = Literal["eos", "length"]


@dataclass(frozen=True)
class GeneratedResponse:
    """One generated continuation in prompt-major, sample-minor order."""

    prompt_index: int
    sample_index: int
    response: str
    prompt_tokens: int
    response_tokens: int
    finish_reason: FinishReason

    def __post_init__(self) -> None:
        _require_non_negative_int("prompt_index", self.prompt_index)
        _require_non_negative_int("sample_index", self.sample_index)
        if not isinstance(self.response, str):
            raise ValueError("response must be a string.")
        _require_non_negative_int("prompt_tokens", self.prompt_tokens)
        _require_non_negative_int("response_tokens", self.response_tokens)
        if self.finish_reason not in {"eos", "length"}:
            raise ValueError("finish_reason must be either 'eos' or 'length'.")

    @property
    def hit_max_length(self) -> bool:
        """Whether generation stopped without an observed EOS token."""
        return self.finish_reason == "length"


def _require_non_negative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
