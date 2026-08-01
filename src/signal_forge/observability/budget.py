"""Rollout-budget accounting for fair A-E comparisons.

The primary budget is generated candidate rollout tokens. Prompt tokens are
counted once per generated response, not once per unique prompt, so the metric
tracks the actual generated request volume seen by the rollout backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable


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
    """Accumulate generated-rollout budget counters inside the trainer."""

    start_time: float = field(default_factory=time.monotonic)
    candidate_prompt_groups: int = 0
    accepted_prompt_groups: int = 0
    rejected_prompt_groups: int = 0
    responses_generated: int = 0
    prompt_tokens_generated: int = 0
    response_tokens_generated: int = 0
    rollout_tokens_generated: int = 0
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
        self.optimizer_steps += int(optimizer_steps_step)

        wall_time = max(time.monotonic() - self.start_time, 0.0)
        optimizer_steps = max(self.optimizer_steps, 1)
        gpu_hours = wall_time * max(int(n_gpus), 0) / 3600.0

        return {
            "budget/candidate_prompt_groups_step": float(candidate_prompt_groups_step),
            "budget/accepted_prompt_groups_step": float(accepted_prompt_groups_step),
            "budget/rejected_prompt_groups_step": float(rejected_prompt_groups_step),
            "budget/responses_generated_step": float(responses_generated_step),
            "budget/prompt_tokens_generated_step": float(prompt_tokens_generated_step),
            "budget/response_tokens_generated_step": float(response_tokens_generated_step),
            "budget/rollout_tokens_generated_step": float(rollout_tokens_generated_step),
            "budget/candidate_prompt_groups_cumulative": float(self.candidate_prompt_groups),
            "budget/accepted_prompt_groups_cumulative": float(self.accepted_prompt_groups),
            "budget/rejected_prompt_groups_cumulative": float(self.rejected_prompt_groups),
            "budget/responses_generated_cumulative": float(self.responses_generated),
            "budget/prompt_tokens_generated_cumulative": float(self.prompt_tokens_generated),
            "budget/response_tokens_generated_cumulative": float(self.response_tokens_generated),
            "budget/rollout_tokens_generated_cumulative": float(self.rollout_tokens_generated),
            "budget/optimizer_steps": float(self.optimizer_steps),
            "budget/wall_time_seconds_cumulative": float(wall_time),
            "budget/gpu_hours_estimate": float(gpu_hours),
            "budget/responses_per_optimizer_step": float(self.responses_generated / optimizer_steps),
            "budget/response_tokens_per_optimizer_step": float(self.response_tokens_generated / optimizer_steps),
        }
