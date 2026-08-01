"""Composable baseline reward calculation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from rewardscope.schemas import RewardBreakdown, VerificationResult


def _require_finite_number(name: str, value: object) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number.")


def _require_non_negative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


@dataclass(frozen=True)
class RewardConfig:
    """Weights used to translate verification facts into training rewards."""

    correct_answer_reward: float = 1.0
    incorrect_answer_reward: float = 0.0
    format_compliance_reward: float = 0.0
    length_penalty_per_token: float = 0.0

    def __post_init__(self) -> None:
        _require_finite_number("correct_answer_reward", self.correct_answer_reward)
        _require_finite_number("incorrect_answer_reward", self.incorrect_answer_reward)
        _require_finite_number(
            "format_compliance_reward", self.format_compliance_reward
        )
        _require_finite_number(
            "length_penalty_per_token", self.length_penalty_per_token
        )
        if self.length_penalty_per_token < 0:
            raise ValueError("length_penalty_per_token must be non-negative.")


def compute_reward(
    verification: VerificationResult,
    response_tokens: int,
    config: RewardConfig = RewardConfig(),
) -> RewardBreakdown:
    """Convert verification facts and response length into reward components."""
    if not isinstance(verification, VerificationResult):
        raise TypeError("verification must be a VerificationResult.")
    if not isinstance(config, RewardConfig):
        raise TypeError("config must be a RewardConfig.")
    _require_non_negative_int("response_tokens", response_tokens)

    correctness_reward = (
        config.correct_answer_reward
        if verification.is_correct
        else config.incorrect_answer_reward
    )
    format_reward = (
        config.format_compliance_reward
        if verification.extraction.format_ok
        else 0.0
    )
    length_penalty = -config.length_penalty_per_token * response_tokens
    final_reward = correctness_reward + format_reward + length_penalty

    return RewardBreakdown(
        correctness_reward=correctness_reward,
        format_reward=format_reward,
        length_penalty=length_penalty,
        final_reward=final_reward,
    )
