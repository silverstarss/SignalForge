"""Guards for checkpoint-isolated diagnostic validation runs."""

from __future__ import annotations

import os
import math
from numbers import Real
from pathlib import Path


PILOT_DIAGNOSTIC_STEP80_LABEL = "pilot_diagnostic_step80"


def build_validation_compute_metrics(
    *,
    generated_responses: int,
    generated_prompt_tokens: int,
    generated_response_tokens: int,
    validation_n: int,
    wall_time_seconds: Real,
    label: str | None = None,
) -> dict[str, float]:
    for name, value in (
        ("generated_responses", generated_responses),
        ("generated_prompt_tokens", generated_prompt_tokens),
        ("generated_response_tokens", generated_response_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(validation_n, bool) or not isinstance(validation_n, int) or validation_n <= 0:
        raise ValueError("validation_n must be a positive integer")
    if generated_responses % validation_n:
        raise ValueError("generated_responses must be divisible by validation_n")
    wall = float(wall_time_seconds)
    if not math.isfinite(wall) or wall < 0:
        raise ValueError("wall_time_seconds must be finite and non-negative")
    rollout_tokens = generated_prompt_tokens + generated_response_tokens
    metrics = {
        "validation/generated_prompts": float(generated_responses // validation_n),
        "validation/generated_prompt_tokens": float(generated_prompt_tokens),
        "validation/generated_responses": float(generated_responses),
        "validation/generated_response_tokens": float(generated_response_tokens),
        "validation/rollout_tokens": float(rollout_tokens),
        "validation/wall_time_seconds": wall,
        "physical_compute/validation_generated_prompt_tokens": float(generated_prompt_tokens),
        "physical_compute/validation_generated_response_tokens": float(generated_response_tokens),
        "physical_compute/validation_rollout_tokens": float(rollout_tokens),
        "physical_compute/validation_wall_time_seconds": wall,
    }
    if label:
        metrics[f"diagnostic/{label}"] = 1.0
    return metrics


def validate_diagnostic_validation_contract(config) -> None:
    """Require diagnostic validation to run in a separate, terminating process."""
    label = config.trainer.get("validation_label", None)
    if label is None:
        return
    if label != PILOT_DIAGNOSTIC_STEP80_LABEL:
        raise ValueError(f"unsupported diagnostic validation label: {label!r}")
    if not config.trainer.get("val_only", False):
        raise ValueError("diagnostic validation requires trainer.val_only=True")
    if not config.trainer.get("val_before_train", True):
        raise ValueError("diagnostic validation requires trainer.val_before_train=True")
    if config.trainer.get("update_best_checkpoint_metadata", True):
        raise ValueError("diagnostic validation must disable best-checkpoint metadata updates")
    if config.trainer.get("resume_mode") != "resume_path":
        raise ValueError("diagnostic validation requires an explicit checkpoint resume path")
    resume_path = config.trainer.get("resume_from_path", None)
    if not isinstance(resume_path, str) or "global_step_80" not in resume_path:
        raise ValueError("pilot_diagnostic_step80 must load an explicit global_step_80 checkpoint")
    output_root = Path(os.path.abspath(str(config.trainer.default_local_dir)))
    checkpoint_root = Path(os.path.abspath(resume_path)).parent
    if output_root == checkpoint_root:
        raise ValueError("diagnostic validation output must be separate from Formal B checkpoints")
    if config.trainer.get("del_local_ckpt_after_load", False):
        raise ValueError("diagnostic validation must not delete the source checkpoint after loading")
