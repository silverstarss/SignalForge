"""Validation and resume loading for formal best-checkpoint metadata."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BestCheckpointMetadata:
    metric_name: str
    metric_value: float
    global_step: int
    checkpoint_path: str | None
    checkpoint_saved_on_update: bool


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"best checkpoint {field} must be an integer")
    return value


def load_best_checkpoint_metadata(
    checkpoint_root: str,
    *,
    expected_metric_name: str,
    resume_global_step: int,
) -> BestCheckpointMetadata | None:
    """Load and validate metadata whose observations precede a resumed step."""
    metadata_path = os.path.join(checkpoint_root, "best_checkpoint.json")
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("best_checkpoint.json must contain a JSON object")

    metric_name = payload.get("metric_name")
    if metric_name != expected_metric_name:
        raise ValueError(
            "best checkpoint metric mismatch: "
            f"checkpoint={metric_name!r}, configured={expected_metric_name!r}"
        )

    try:
        metric_value = float(payload["metric_value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("best checkpoint metric_value must be numeric") from exc
    if not math.isfinite(metric_value):
        raise ValueError("best checkpoint metric_value must be finite")

    best_step = _require_int(payload.get("global_step"), field="global_step")
    if best_step < 0 or best_step > resume_global_step:
        raise ValueError(
            "best checkpoint global_step must be between zero and the resume step: "
            f"best={best_step}, resume={resume_global_step}"
        )

    raw_checkpoint_path = payload.get("checkpoint_path")
    checkpoint_path: str | None
    if raw_checkpoint_path is None:
        checkpoint_path = None
    elif not isinstance(raw_checkpoint_path, str) or not raw_checkpoint_path:
        raise ValueError("best checkpoint checkpoint_path must be null or a non-empty string")
    else:
        checkpoint_path = os.path.realpath(os.path.expanduser(raw_checkpoint_path))
        expected_path = os.path.realpath(os.path.join(checkpoint_root, f"global_step_{best_step}"))
        if checkpoint_path != expected_path:
            raise ValueError(
                "best checkpoint path does not match its global_step: "
                f"path={checkpoint_path!r}, expected={expected_path!r}"
            )
        if not os.path.isdir(checkpoint_path):
            raise ValueError(f"best checkpoint path does not exist: {checkpoint_path}")

    checkpoint_saved_on_update = payload.get("checkpoint_saved_on_update", False)
    if not isinstance(checkpoint_saved_on_update, bool):
        raise ValueError("best checkpoint checkpoint_saved_on_update must be boolean")

    return BestCheckpointMetadata(
        metric_name=metric_name,
        metric_value=metric_value,
        global_step=best_step,
        checkpoint_path=checkpoint_path,
        checkpoint_saved_on_update=checkpoint_saved_on_update,
    )
