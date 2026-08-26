"""Rollout-budget and compute accounting shared by formal GRPO and HIVE."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROLLOUT_BUDGET_FILENAME = "rollout_budget_tracker.json"
ROLLOUT_BUDGET_SCHEMA_VERSION = 1


def _as_list(values) -> list:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def count_prompt_groups(uids: Iterable | None) -> int:
    values = [uid for uid in _as_list(uids) if uid not in (None, "")]
    return len(set(values))


@dataclass
class RolloutBudgetTracker:
    """Accumulate identical generated/effective compute counters for A and B."""

    start_time: float = field(default_factory=time.monotonic)
    candidate_prompt_groups: int = 0
    accepted_prompt_groups: int = 0
    rejected_prompt_groups: int = 0
    responses_generated: int = 0
    prompt_tokens_generated: int = 0
    response_tokens_generated: int = 0
    rollout_tokens_generated: int = 0
    effective_prompt_groups: int = 0
    effective_responses: int = 0
    effective_response_tokens: int = 0
    rollout_wall_time_seconds: float = 0.0
    restored_wall_time_seconds: float = 0.0
    optimizer_steps: int = 0

    def update(
        self,
        *,
        candidate_prompt_groups_step: int,
        accepted_prompt_groups_step: int,
        responses_generated_step: int,
        prompt_tokens_generated_step: int,
        response_tokens_generated_step: int,
        optimizer_steps_step: int,
        n_gpus: int,
        effective_prompt_groups_step: int | None = None,
        effective_responses_step: int | None = None,
        effective_response_tokens_step: int | None = None,
        rollout_time_seconds_step: float = 0.0,
    ) -> dict[str, float]:
        rejected_prompt_groups_step = max(candidate_prompt_groups_step - accepted_prompt_groups_step, 0)
        rollout_tokens_generated_step = prompt_tokens_generated_step + response_tokens_generated_step

        self.candidate_prompt_groups += int(candidate_prompt_groups_step)
        self.accepted_prompt_groups += int(accepted_prompt_groups_step)
        self.rejected_prompt_groups += int(rejected_prompt_groups_step)
        self.responses_generated += int(responses_generated_step)
        self.prompt_tokens_generated += int(prompt_tokens_generated_step)
        self.response_tokens_generated += int(response_tokens_generated_step)
        self.rollout_tokens_generated += int(rollout_tokens_generated_step)
        self.effective_prompt_groups += int(
            accepted_prompt_groups_step
            if effective_prompt_groups_step is None
            else effective_prompt_groups_step
        )
        self.effective_responses += int(
            responses_generated_step if effective_responses_step is None else effective_responses_step
        )
        self.effective_response_tokens += int(
            response_tokens_generated_step
            if effective_response_tokens_step is None
            else effective_response_tokens_step
        )
        self.rollout_wall_time_seconds += float(rollout_time_seconds_step)
        self.optimizer_steps += int(optimizer_steps_step)

        return self.snapshot(
            n_gpus=n_gpus,
            step_metrics={
                "budget/candidate_prompt_groups_step": float(candidate_prompt_groups_step),
                "budget/accepted_prompt_groups_step": float(accepted_prompt_groups_step),
                "budget/rejected_prompt_groups_step": float(rejected_prompt_groups_step),
                "budget/responses_generated_step": float(responses_generated_step),
                "budget/prompt_tokens_generated_step": float(prompt_tokens_generated_step),
                "budget/response_tokens_generated_step": float(response_tokens_generated_step),
                "budget/rollout_tokens_generated_step": float(rollout_tokens_generated_step),
            },
        )

    def snapshot(
        self,
        *,
        n_gpus: int,
        step_metrics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        wall_time = self.restored_wall_time_seconds + max(time.monotonic() - self.start_time, 0.0)
        optimizer_steps = max(self.optimizer_steps, 1)
        generated_groups = self.candidate_prompt_groups
        generated_responses = self.responses_generated
        generated_tokens = self.response_tokens_generated
        metrics = dict(step_metrics or {})
        metrics.update(
            {
                "budget/candidate_prompt_groups_cumulative": float(generated_groups),
                "budget/accepted_prompt_groups_cumulative": float(self.accepted_prompt_groups),
                "budget/rejected_prompt_groups_cumulative": float(self.rejected_prompt_groups),
                "budget/responses_generated_cumulative": float(generated_responses),
                "budget/prompt_tokens_generated_cumulative": float(self.prompt_tokens_generated),
                "budget/response_tokens_generated_cumulative": float(generated_tokens),
                "budget/rollout_tokens_generated_cumulative": float(self.rollout_tokens_generated),
                "budget/optimizer_steps": float(self.optimizer_steps),
                "budget/wall_time_seconds_cumulative": float(wall_time),
                "budget/gpu_hours_estimate": float(wall_time * max(int(n_gpus), 0) / 3600.0),
                "budget/responses_per_optimizer_step": float(generated_responses / optimizer_steps),
                "budget/response_tokens_per_optimizer_step": float(generated_tokens / optimizer_steps),
                "compute/generated_prompt_groups": float(generated_groups),
                "compute/generated_responses": float(generated_responses),
                "compute/generated_response_tokens": float(generated_tokens),
                "compute/effective_prompt_groups": float(self.effective_prompt_groups),
                "compute/effective_responses": float(self.effective_responses),
                "compute/effective_response_tokens": float(self.effective_response_tokens),
                "compute/effective_prompt_group_ratio": _ratio(
                    self.effective_prompt_groups, generated_groups
                ),
                "compute/effective_response_ratio": _ratio(
                    self.effective_responses, generated_responses
                ),
                "compute/effective_training_token_ratio": _ratio(
                    self.effective_response_tokens, generated_tokens
                ),
                "compute/effective_prompt_groups_per_1m_generated_response_tokens": (
                    float(self.effective_prompt_groups * 1_000_000 / generated_tokens)
                    if generated_tokens
                    else 0.0
                ),
                "time/rollout_wall_clock_cumulative": float(self.rollout_wall_time_seconds),
                "time/total_wall_clock_cumulative": float(wall_time),
            }
        )
        return metrics

    def to_dict(self) -> dict:
        return {
            "schema_version": ROLLOUT_BUDGET_SCHEMA_VERSION,
            "candidate_prompt_groups": self.candidate_prompt_groups,
            "accepted_prompt_groups": self.accepted_prompt_groups,
            "rejected_prompt_groups": self.rejected_prompt_groups,
            "responses_generated": self.responses_generated,
            "prompt_tokens_generated": self.prompt_tokens_generated,
            "response_tokens_generated": self.response_tokens_generated,
            "rollout_tokens_generated": self.rollout_tokens_generated,
            "effective_prompt_groups": self.effective_prompt_groups,
            "effective_responses": self.effective_responses,
            "effective_response_tokens": self.effective_response_tokens,
            "rollout_wall_time_seconds": self.rollout_wall_time_seconds,
            "wall_time_seconds": self.restored_wall_time_seconds
            + max(time.monotonic() - self.start_time, 0.0),
            "optimizer_steps": self.optimizer_steps,
        }

    def save_checkpoint(self, checkpoint_dir: str | os.PathLike[str]) -> Path:
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / ROLLOUT_BUDGET_FILENAME
        descriptor, temporary_name = tempfile.mkstemp(
            dir=directory, prefix=f".{ROLLOUT_BUDGET_FILENAME}.", suffix=".tmp"
        )
        os.close(descriptor)
        try:
            with open(temporary_name, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, sort_keys=True, separators=(",", ":"))
            os.replace(temporary_name, destination)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: str | os.PathLike[str],
        *,
        expected_optimizer_steps: int | None = None,
    ) -> "RolloutBudgetTracker":
        with open(Path(checkpoint_dir) / ROLLOUT_BUDGET_FILENAME, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != ROLLOUT_BUDGET_SCHEMA_VERSION:
            raise ValueError("unsupported rollout budget tracker schema_version")
        integer_names = (
            "candidate_prompt_groups",
            "accepted_prompt_groups",
            "rejected_prompt_groups",
            "responses_generated",
            "prompt_tokens_generated",
            "response_tokens_generated",
            "rollout_tokens_generated",
            "effective_prompt_groups",
            "effective_responses",
            "effective_response_tokens",
            "optimizer_steps",
        )
        values = {name: _nonnegative_int(name, payload.get(name)) for name in integer_names}
        if expected_optimizer_steps is not None and values["optimizer_steps"] != expected_optimizer_steps:
            raise ValueError(
                "rollout budget optimizer_steps does not match trainer checkpoint step: "
                f"{values['optimizer_steps']} != {expected_optimizer_steps}"
            )
        return cls(
            **values,
            rollout_wall_time_seconds=_nonnegative_float(
                "rollout_wall_time_seconds", payload.get("rollout_wall_time_seconds")
            ),
            restored_wall_time_seconds=_nonnegative_float(
                "wall_time_seconds", payload.get("wall_time_seconds")
            ),
        )


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _nonnegative_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonnegative_float(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)
