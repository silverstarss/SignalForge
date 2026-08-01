"""Build complete rollout records from generated model responses."""

from __future__ import annotations

from dataclasses import dataclass

from rewardscope.extraction import NumericExtractionConfig
from rewardscope.rewards import RewardConfig, compute_reward
from rewardscope.schemas import RolloutRecord
from rewardscope.verification import (
    MathVerifyLatexVerifier,
    MathVerifyNumericVerifier,
    verify_numeric_answer,
)


@dataclass(frozen=True)
class RolloutInput:
    """Generation facts available before verification and reward calculation."""

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
    prompt_tokens: int
    response_tokens: int
    hit_max_length: bool
    finish_reason: str | None = None


def build_numeric_rollout(
    rollout_input: RolloutInput,
    reward_config: RewardConfig = RewardConfig(),
    extraction_config: NumericExtractionConfig = NumericExtractionConfig(),
) -> RolloutRecord:
    """Verify, reward, and package one numeric rollout into a complete record."""
    if not isinstance(rollout_input, RolloutInput):
        raise TypeError("rollout_input must be a RolloutInput.")

    verification = verify_numeric_answer(
        rollout_input.response,
        rollout_input.ground_truth,
        extraction_config=extraction_config,
    )
    return _complete_rollout(rollout_input, verification, reward_config)


def build_math_verify_numeric_rollout(
    rollout_input: RolloutInput,
    reward_config: RewardConfig = RewardConfig(),
    *,
    mode: str = "evaluation",
) -> RolloutRecord:
    """Verify one rollout using Math-Verify and package its reward details."""
    if not isinstance(rollout_input, RolloutInput):
        raise TypeError("rollout_input must be a RolloutInput.")
    verification = MathVerifyNumericVerifier(mode=mode).verify(
        rollout_input.response,
        rollout_input.ground_truth,
    )
    return _complete_rollout(rollout_input, verification, reward_config)


def build_math_verify_latex_rollout(
    rollout_input: RolloutInput,
    reward_config: RewardConfig = RewardConfig(),
    *,
    mode: str = "training",
) -> RolloutRecord:
    """Verify one MATH rollout with LaTeX gold and strict boxed prediction parsing."""
    if not isinstance(rollout_input, RolloutInput):
        raise TypeError("rollout_input must be a RolloutInput.")
    verification = MathVerifyLatexVerifier(mode=mode).verify(
        rollout_input.response,
        rollout_input.ground_truth,
    )
    return _complete_rollout(rollout_input, verification, reward_config)


def _complete_rollout(rollout_input: RolloutInput, verification, reward_config: RewardConfig) -> RolloutRecord:
    reward = compute_reward(verification, response_tokens=rollout_input.response_tokens, config=reward_config)
    return RolloutRecord(
        run_id=rollout_input.run_id,
        prompt_id=rollout_input.prompt_id,
        sample_id=rollout_input.sample_id,
        model_name=rollout_input.model_name,
        dataset_name=rollout_input.dataset_name,
        split=rollout_input.split,
        generation_seed=rollout_input.generation_seed,
        temperature=rollout_input.temperature,
        top_p=rollout_input.top_p,
        max_new_tokens=rollout_input.max_new_tokens,
        batch_size=rollout_input.batch_size,
        prompt=rollout_input.prompt,
        response=rollout_input.response,
        ground_truth=rollout_input.ground_truth,
        verification=verification,
        reward=reward,
        prompt_tokens=rollout_input.prompt_tokens,
        response_tokens=rollout_input.response_tokens,
        finish_reason=(
            rollout_input.finish_reason
            if rollout_input.finish_reason is not None
            else "length" if rollout_input.hit_max_length else "eos"
        ),
        hit_max_length=rollout_input.hit_max_length,
    )
