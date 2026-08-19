"""Persistent Phase 1 state for the HIVE rollout selector."""

from __future__ import annotations

import copy
import gzip
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from pathlib import Path
from statistics import pvariance
from typing import Any, Iterable, Mapping

import numpy as np


HIVE_STATE_SCHEMA_VERSION = 1
HIVE_STATE_FILENAME = "hive_selector_state.json.gz"


class ZeroVarianceType(str, Enum):
    EASY = "easy"
    HARD = "hard"
    OTHER = "other"


@dataclass(frozen=True)
class RewardGroupClassification:
    rewards: tuple[float, ...]
    reward_variance: float
    zero_variance: bool
    zero_variance_type: ZeroVarianceType | None


def _normalize_rewards(rewards: Iterable[Real], *, group_size: int) -> tuple[float, ...]:
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")

    values = tuple(rewards)
    if len(values) != group_size:
        raise ValueError(f"reward group must contain exactly {group_size} values; got {len(values)}")

    normalized: list[float] = []
    for index, reward in enumerate(values):
        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise ValueError(f"reward at index {index} must be a real number")
        value = float(reward)
        if not math.isfinite(value):
            raise ValueError(f"reward at index {index} must be finite")
        normalized.append(value)
    return tuple(normalized)


def classify_zero_variance(
    rewards: Iterable[Real], *, group_size: int
) -> RewardGroupClassification:
    """Classify a complete reward group using the exact HIVE reward semantics."""
    values = _normalize_rewards(rewards, group_size=group_size)
    zero_variance = all(value == values[0] for value in values[1:])
    reward_variance = 0.0 if zero_variance else float(pvariance(values))

    zero_variance_type: ZeroVarianceType | None = None
    if zero_variance:
        if all(value == 1.0 for value in values):
            zero_variance_type = ZeroVarianceType.EASY
        elif all(value == 0.1 for value in values):
            zero_variance_type = ZeroVarianceType.HARD
        else:
            zero_variance_type = ZeroVarianceType.OTHER

    return RewardGroupClassification(
        rewards=values,
        reward_variance=reward_variance,
        zero_variance=zero_variance,
        zero_variance_type=zero_variance_type,
    )


@dataclass(frozen=True)
class PromptVisit:
    step: int
    rewards: tuple[float, ...]
    reward_variance: float
    zero_variance: bool
    zero_variance_type: ZeroVarianceType | None
    response_entropy: float | None = None

    @classmethod
    def from_rewards(
        cls,
        *,
        step: int,
        rewards: Iterable[Real],
        group_size: int,
        response_entropy: float | None = None,
    ) -> "PromptVisit":
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("visit step must be a non-negative integer")
        if response_entropy is not None:
            if isinstance(response_entropy, bool) or not isinstance(response_entropy, Real):
                raise ValueError("response_entropy must be a real number or None")
            response_entropy = float(response_entropy)
            if not math.isfinite(response_entropy):
                raise ValueError("response_entropy must be finite")

        classification = classify_zero_variance(rewards, group_size=group_size)
        return cls(
            step=step,
            rewards=classification.rewards,
            reward_variance=classification.reward_variance,
            zero_variance=classification.zero_variance,
            zero_variance_type=classification.zero_variance_type,
            response_entropy=response_entropy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "rewards": list(self.rewards),
            "reward_variance": self.reward_variance,
            "zero_variance": self.zero_variance,
            "zero_variance_type": self.zero_variance_type.value if self.zero_variance_type is not None else None,
            "response_entropy": self.response_entropy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, group_size: int) -> "PromptVisit":
        visit = cls.from_rewards(
            step=payload["step"],
            rewards=payload["rewards"],
            group_size=group_size,
            response_entropy=payload.get("response_entropy"),
        )
        expected = visit.to_dict()
        for field_name in ("reward_variance", "zero_variance", "zero_variance_type"):
            if payload.get(field_name) != expected[field_name]:
                raise ValueError(f"checkpoint PromptVisit {field_name} does not match rewards")
        return visit


@dataclass
class PromptHistory:
    prompt_id: str
    visits: list[PromptVisit] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id.strip():
            raise ValueError("PromptHistory prompt_id must be a non-empty string")
        previous_step = -1
        for visit in self.visits:
            if visit.step < previous_step:
                raise ValueError("PromptVisit steps must be nondecreasing")
            previous_step = visit.step

    def append(self, visit: PromptVisit) -> None:
        if self.visits and visit.step < self.visits[-1].step:
            raise ValueError("PromptVisit steps must be nondecreasing")
        self.visits.append(visit)

    def trailing_zero_variance_streak(self) -> int:
        streak = 0
        for visit in reversed(self.visits):
            if not visit.zero_variance:
                break
            streak += 1
        return streak

    def to_dict(self) -> dict[str, Any]:
        return {"prompt_id": self.prompt_id, "visits": [visit.to_dict() for visit in self.visits]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, group_size: int) -> "PromptHistory":
        return cls(
            prompt_id=payload["prompt_id"],
            visits=[PromptVisit.from_dict(visit, group_size=group_size) for visit in payload["visits"]],
        )


def _validate_probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return normalized


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


@dataclass
class HiveSelectorState:
    group_size: int
    prompt_history: dict[str, PromptHistory]
    p_easy: float
    p_hard: float
    p_default: float
    global_step: int
    selector_rng_state: dict[str, Any]
    configuration: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.group_size, bool) or not isinstance(self.group_size, int) or self.group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        self.p_easy = _validate_probability("p_easy", self.p_easy)
        self.p_hard = _validate_probability("p_hard", self.p_hard)
        self.p_default = _validate_probability("p_default", self.p_default)
        if isinstance(self.global_step, bool) or not isinstance(self.global_step, int) or self.global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        for prompt_id, history in self.prompt_history.items():
            if prompt_id != history.prompt_id:
                raise ValueError("prompt_history mapping key must match PromptHistory.prompt_id")
            for visit in history.visits:
                if len(visit.rewards) != self.group_size:
                    raise ValueError("PromptVisit reward group does not match selector group_size")
        self.configuration = copy.deepcopy(dict(self.configuration))
        self.selector_rng_state = copy.deepcopy(dict(self.selector_rng_state))
        self.restore_rng()

    @classmethod
    def create(
        cls,
        *,
        group_size: int,
        seed: int,
        p_easy: float = 0.5,
        p_hard: float = 0.5,
        p_default: float = 0.5,
        configuration: Mapping[str, Any] | None = None,
    ) -> "HiveSelectorState":
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("selector seed must be a non-negative integer")
        rng = np.random.default_rng(seed)
        return cls(
            group_size=group_size,
            prompt_history={},
            p_easy=p_easy,
            p_hard=p_hard,
            p_default=p_default,
            global_step=0,
            selector_rng_state=copy.deepcopy(rng.bit_generator.state),
            configuration=dict(configuration or {}),
        )

    def append_visit(self, prompt_id: str, visit: PromptVisit) -> None:
        if len(visit.rewards) != self.group_size:
            raise ValueError("PromptVisit reward group does not match selector group_size")
        history = self.prompt_history.setdefault(prompt_id, PromptHistory(prompt_id=prompt_id))
        history.append(visit)

    def trailing_zero_variance_streak(self, prompt_id: str) -> int:
        history = self.prompt_history.get(prompt_id)
        return 0 if history is None else history.trailing_zero_variance_streak()

    def restore_rng(self) -> np.random.Generator:
        bit_generator_name = self.selector_rng_state.get("bit_generator")
        if bit_generator_name != "PCG64":
            raise ValueError(f"unsupported selector RNG bit generator: {bit_generator_name!r}")
        bit_generator = np.random.PCG64()
        try:
            bit_generator.state = copy.deepcopy(self.selector_rng_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid selector_rng_state") from exc
        return np.random.Generator(bit_generator)

    def capture_rng(self, rng: np.random.Generator) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if rng.bit_generator.__class__.__name__ != "PCG64":
            raise ValueError("selector RNG must use PCG64")
        self.selector_rng_state = copy.deepcopy(rng.bit_generator.state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HIVE_STATE_SCHEMA_VERSION,
            "state": {
                "group_size": self.group_size,
                "prompt_history": {
                    prompt_id: history.to_dict() for prompt_id, history in sorted(self.prompt_history.items())
                },
                "p_easy": self.p_easy,
                "p_hard": self.p_hard,
                "p_default": self.p_default,
                "global_step": self.global_step,
                "selector_rng_state": _json_compatible(self.selector_rng_state),
                "configuration": _json_compatible(self.configuration),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HiveSelectorState":
        schema_version = payload.get("schema_version")
        if schema_version != HIVE_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported HIVE selector state schema_version {schema_version!r}; "
                f"expected {HIVE_STATE_SCHEMA_VERSION}"
            )
        state_payload = payload.get("state")
        if not isinstance(state_payload, Mapping):
            raise ValueError("HIVE selector checkpoint is missing state")
        group_size = state_payload["group_size"]
        histories = {
            prompt_id: PromptHistory.from_dict(history, group_size=group_size)
            for prompt_id, history in state_payload["prompt_history"].items()
        }
        return cls(
            group_size=group_size,
            prompt_history=histories,
            p_easy=state_payload["p_easy"],
            p_hard=state_payload["p_hard"],
            p_default=state_payload["p_default"],
            global_step=state_payload["global_step"],
            selector_rng_state=state_payload["selector_rng_state"],
            configuration=state_payload["configuration"],
        )

    def save_checkpoint(self, checkpoint_dir: str | os.PathLike[str]) -> Path:
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / HIVE_STATE_FILENAME
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=directory,
            prefix=f".{HIVE_STATE_FILENAME}.",
            suffix=".tmp",
        )
        os.close(file_descriptor)
        try:
            with gzip.open(temporary_name, "wt", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            os.replace(temporary_name, destination)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str | os.PathLike[str]) -> "HiveSelectorState":
        checkpoint_path = Path(checkpoint_dir) / HIVE_STATE_FILENAME
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"HIVE selector checkpoint not found: {checkpoint_path}")
        with gzip.open(checkpoint_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)
