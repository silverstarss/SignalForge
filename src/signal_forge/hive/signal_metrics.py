"""Checkpointed HIVE learning-signal observability.

This module classifies complete rollout groups without changing filtering or
training behavior.  Candidate metrics cover every generated group; training
metrics cover only the final fixed-size GRPO batch.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping

from signal_forge.hive.state import classify_zero_variance


HIVE_SIGNAL_COUNTERS_FILENAME = "hive_signal_counters.json"
HIVE_SIGNAL_COUNTERS_SCHEMA_VERSION = 1


def _as_list(values: Iterable) -> list:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _per_million(count: int, generated_response_tokens: int) -> float:
    return float(count * 1_000_000 / generated_response_tokens) if generated_response_tokens else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


@dataclass(frozen=True)
class HiveGroupSignalCounts:
    group_count: int
    optimization_effective: int
    raw_correctness_mixed: int
    extraction_only_effective: int

    def __post_init__(self) -> None:
        for name in (
            "group_count",
            "optimization_effective",
            "raw_correctness_mixed",
            "extraction_only_effective",
        ):
            _nonnegative_integer(name, getattr(self, name))
        if self.optimization_effective > self.group_count:
            raise ValueError("optimization_effective cannot exceed group_count")
        if self.raw_correctness_mixed > self.optimization_effective:
            raise ValueError("correctness-mixed groups must be optimization-effective")
        if self.extraction_only_effective != self.optimization_effective - self.raw_correctness_mixed:
            raise ValueError(
                "extraction_only_effective must equal optimization_effective - raw_correctness_mixed"
            )


def compute_hive_group_signal_counts(
    *,
    uids: Iterable,
    scalar_rewards: Iterable,
    raw_correctness: Iterable,
    group_size: int,
) -> HiveGroupSignalCounts:
    """Classify complete groups under the frozen three-state reward semantics."""
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")

    uid_values = _as_list(uids)
    reward_values = _as_list(scalar_rewards)
    correctness_values = _as_list(raw_correctness)
    if not (len(uid_values) == len(reward_values) == len(correctness_values)):
        raise ValueError("uids, scalar_rewards, and raw_correctness must have identical lengths")

    groups: dict[object, list[tuple[float, bool]]] = defaultdict(list)
    for index, (uid, reward, correct) in enumerate(
        zip(uid_values, reward_values, correctness_values, strict=True)
    ):
        if uid in (None, ""):
            raise ValueError(f"group uid is missing at response index {index}")
        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise ValueError(f"scalar reward at response index {index} must be a real number")
        normalized_reward = float(reward)
        if not math.isfinite(normalized_reward) or normalized_reward not in (0.0, 0.1, 1.0):
            raise ValueError(f"scalar reward at response index {index} violates frozen HIVE semantics")
        if isinstance(correct, bool):
            normalized_correct = correct
        elif isinstance(correct, Real) and float(correct) in (0.0, 1.0):
            normalized_correct = bool(float(correct))
        else:
            raise ValueError(f"raw correctness at response index {index} must be binary")
        if normalized_correct != (normalized_reward == 1.0):
            raise ValueError(
                f"raw correctness at response index {index} does not match the frozen reward semantics"
            )
        groups[uid].append((normalized_reward, normalized_correct))

    optimization_effective = 0
    raw_correctness_mixed = 0
    extraction_only_effective = 0
    for uid, values in groups.items():
        if len(values) != group_size:
            raise ValueError(
                f"group {uid!r} must contain exactly {group_size} responses; got {len(values)}"
            )
        rewards = [value[0] for value in values]
        correctness = [value[1] for value in values]
        scalar_effective = not classify_zero_variance(rewards, group_size=group_size).zero_variance
        correctness_mixed = any(correctness) and not all(correctness)
        if correctness_mixed and not scalar_effective:
            raise ValueError("correctness-mixed group is unexpectedly scalar zero-variance")
        optimization_effective += int(scalar_effective)
        raw_correctness_mixed += int(correctness_mixed)
        extraction_only_effective += int(scalar_effective and not correctness_mixed)

    return HiveGroupSignalCounts(
        group_count=len(groups),
        optimization_effective=optimization_effective,
        raw_correctness_mixed=raw_correctness_mixed,
        extraction_only_effective=extraction_only_effective,
    )


@dataclass(frozen=True)
class HiveSignalStepCounts:
    candidate: HiveGroupSignalCounts
    training: HiveGroupSignalCounts
    generated_response_tokens: int
    topup_groups: int

    def __post_init__(self) -> None:
        _nonnegative_integer("generated_response_tokens", self.generated_response_tokens)
        _nonnegative_integer("topup_groups", self.topup_groups)
        if self.training.group_count > self.candidate.group_count:
            raise ValueError("training groups cannot exceed candidate groups")
        if self.topup_groups > self.candidate.group_count:
            raise ValueError("topup groups cannot exceed candidate groups")


@dataclass
class HiveSignalCounters:
    global_step: int = 0
    candidate_observation_start_step: int = 0
    training_observation_start_step: int = 0
    candidate_observed_updates: int = 0
    training_observed_updates: int = 0
    candidate_groups: int = 0
    candidate_optimization_effective: int = 0
    candidate_raw_correctness_mixed: int = 0
    candidate_extraction_only_effective: int = 0
    candidate_generated_response_tokens: int = 0
    training_groups: int = 0
    training_optimization_effective: int = 0
    training_raw_correctness_mixed: int = 0
    training_extraction_only_effective: int = 0
    training_generated_response_tokens: int = 0
    topup_groups: int = 0
    _pending_step: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "global_step",
            "candidate_observation_start_step",
            "training_observation_start_step",
            "candidate_observed_updates",
            "training_observed_updates",
            "candidate_groups",
            "candidate_optimization_effective",
            "candidate_raw_correctness_mixed",
            "candidate_extraction_only_effective",
            "candidate_generated_response_tokens",
            "training_groups",
            "training_optimization_effective",
            "training_raw_correctness_mixed",
            "training_extraction_only_effective",
            "training_generated_response_tokens",
            "topup_groups",
        ):
            _nonnegative_integer(name, getattr(self, name))
        if self.candidate_observation_start_step > self.global_step:
            raise ValueError("candidate_observation_start_step cannot exceed global_step")
        if self.training_observation_start_step > self.global_step:
            raise ValueError("training_observation_start_step cannot exceed global_step")
        HiveGroupSignalCounts(
            self.candidate_groups,
            self.candidate_optimization_effective,
            self.candidate_raw_correctness_mixed,
            self.candidate_extraction_only_effective,
        )
        HiveGroupSignalCounts(
            self.training_groups,
            self.training_optimization_effective,
            self.training_raw_correctness_mixed,
            self.training_extraction_only_effective,
        )

    def update(self, counts: HiveSignalStepCounts) -> dict[str, float]:
        if self._pending_step:
            raise RuntimeError("HIVE signal counters already contain an uncommitted step")
        self.candidate_observed_updates += 1
        self.training_observed_updates += 1
        self.candidate_groups += counts.candidate.group_count
        self.candidate_optimization_effective += counts.candidate.optimization_effective
        self.candidate_raw_correctness_mixed += counts.candidate.raw_correctness_mixed
        self.candidate_extraction_only_effective += counts.candidate.extraction_only_effective
        self.candidate_generated_response_tokens += counts.generated_response_tokens
        self.training_groups += counts.training.group_count
        self.training_optimization_effective += counts.training.optimization_effective
        self.training_raw_correctness_mixed += counts.training.raw_correctness_mixed
        self.training_extraction_only_effective += counts.training.extraction_only_effective
        self.training_generated_response_tokens += counts.generated_response_tokens
        self.topup_groups += counts.topup_groups
        self._pending_step = True
        return self._metrics(counts)

    def mark_step_complete(self, global_step: int) -> None:
        step = _nonnegative_integer("global_step", global_step)
        if not self._pending_step:
            raise RuntimeError("HIVE signal counters have no pending step to commit")
        if step != self.global_step + 1:
            raise ValueError(
                f"HIVE signal counters require consecutive steps; current={self.global_step}, next={step}"
            )
        self.global_step = step
        self._pending_step = False

    def _metrics(self, counts: HiveSignalStepCounts) -> dict[str, float]:
        metrics: dict[str, float] = {
            "candidate/optimization_effective": float(counts.candidate.optimization_effective),
            "candidate/raw_correctness_mixed": float(counts.candidate.raw_correctness_mixed),
            "candidate/extraction_only_effective": float(counts.candidate.extraction_only_effective),
            "candidate/raw_correctness_mixed_fraction": _ratio(
                counts.candidate.raw_correctness_mixed, counts.candidate.group_count
            ),
            "candidate/extraction_only_effective_fraction": _ratio(
                counts.candidate.extraction_only_effective, counts.candidate.group_count
            ),
            "training/optimization_effective": float(counts.training.optimization_effective),
            "training/raw_correctness_mixed": float(counts.training.raw_correctness_mixed),
            "training/extraction_only_effective": float(counts.training.extraction_only_effective),
            "training/raw_correctness_mixed_fraction": _ratio(
                counts.training.raw_correctness_mixed, counts.training.group_count
            ),
            "training/extraction_only_effective_fraction": _ratio(
                counts.training.extraction_only_effective, counts.training.group_count
            ),
            "candidate/raw_correctness_mixed_per_1m_generated_response_tokens": _per_million(
                counts.candidate.raw_correctness_mixed, counts.generated_response_tokens
            ),
            "candidate/extraction_only_effective_per_1m_generated_response_tokens": _per_million(
                counts.candidate.extraction_only_effective, counts.generated_response_tokens
            ),
            "training/raw_correctness_mixed_per_1m_generated_response_tokens": _per_million(
                counts.training.raw_correctness_mixed, counts.generated_response_tokens
            ),
            "training/extraction_only_effective_per_1m_generated_response_tokens": _per_million(
                counts.training.extraction_only_effective, counts.generated_response_tokens
            ),
            "candidate/raw_correctness_mixed_cumulative": float(self.candidate_raw_correctness_mixed),
            "candidate/extraction_only_effective_cumulative": float(
                self.candidate_extraction_only_effective
            ),
            "training/raw_correctness_mixed_cumulative": float(self.training_raw_correctness_mixed),
            "training/extraction_only_effective_cumulative": float(
                self.training_extraction_only_effective
            ),
            "candidate/raw_correctness_mixed_cumulative_per_1m_generated_response_tokens": _per_million(
                self.candidate_raw_correctness_mixed, self.candidate_generated_response_tokens
            ),
            "candidate/extraction_only_effective_cumulative_per_1m_generated_response_tokens": _per_million(
                self.candidate_extraction_only_effective, self.candidate_generated_response_tokens
            ),
            "training/raw_correctness_mixed_cumulative_per_1m_generated_response_tokens": _per_million(
                self.training_raw_correctness_mixed, self.training_generated_response_tokens
            ),
            "training/extraction_only_effective_cumulative_per_1m_generated_response_tokens": _per_million(
                self.training_extraction_only_effective, self.training_generated_response_tokens
            ),
            "candidate/observation_start_step": float(self.candidate_observation_start_step),
            "training/observation_start_step": float(self.training_observation_start_step),
            "candidate/observed_updates_cumulative": float(self.candidate_observed_updates),
            "training/observed_updates_cumulative": float(self.training_observed_updates),
            "efficiency/generated_groups_per_update": float(counts.candidate.group_count),
            "efficiency/generated_response_tokens_per_update": float(counts.generated_response_tokens),
            "efficiency/topup_groups_per_update": float(counts.topup_groups),
            "efficiency/scalar_zero_var_ratio": 1.0
            - _ratio(counts.candidate.optimization_effective, counts.candidate.group_count),
            "efficiency/raw_correctness_zero_var_ratio": 1.0
            - _ratio(counts.candidate.raw_correctness_mixed, counts.candidate.group_count),
            "efficiency/generated_groups_per_update_cumulative": _ratio(
                self.candidate_groups, self.candidate_observed_updates
            ),
            "efficiency/generated_response_tokens_per_update_cumulative": _ratio(
                self.candidate_generated_response_tokens, self.candidate_observed_updates
            ),
            "efficiency/topup_groups_per_update_cumulative": _ratio(
                self.topup_groups, self.candidate_observed_updates
            ),
            "efficiency/scalar_zero_var_ratio_cumulative": 1.0
            - _ratio(self.candidate_optimization_effective, self.candidate_groups),
            "efficiency/raw_correctness_zero_var_ratio_cumulative": 1.0
            - _ratio(self.candidate_raw_correctness_mixed, self.candidate_groups),
        }
        return metrics

    def to_dict(self) -> dict[str, Any]:
        if self._pending_step:
            raise RuntimeError("cannot checkpoint HIVE signal counters before step commit")
        return {
            "schema_version": HIVE_SIGNAL_COUNTERS_SCHEMA_VERSION,
            **{
                name: getattr(self, name)
                for name in (
                    "global_step",
                    "candidate_observation_start_step",
                    "training_observation_start_step",
                    "candidate_observed_updates",
                    "training_observed_updates",
                    "candidate_groups",
                    "candidate_optimization_effective",
                    "candidate_raw_correctness_mixed",
                    "candidate_extraction_only_effective",
                    "candidate_generated_response_tokens",
                    "training_groups",
                    "training_optimization_effective",
                    "training_raw_correctness_mixed",
                    "training_extraction_only_effective",
                    "training_generated_response_tokens",
                    "topup_groups",
                )
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HiveSignalCounters":
        if payload.get("schema_version") != HIVE_SIGNAL_COUNTERS_SCHEMA_VERSION:
            raise ValueError("unsupported HIVE signal counters schema_version")
        names = (
            "global_step",
            "candidate_observation_start_step",
            "training_observation_start_step",
            "candidate_observed_updates",
            "training_observed_updates",
            "candidate_groups",
            "candidate_optimization_effective",
            "candidate_raw_correctness_mixed",
            "candidate_extraction_only_effective",
            "candidate_generated_response_tokens",
            "training_groups",
            "training_optimization_effective",
            "training_raw_correctness_mixed",
            "training_extraction_only_effective",
            "training_generated_response_tokens",
            "topup_groups",
        )
        return cls(**{name: _nonnegative_integer(name, payload.get(name)) for name in names})

    def save_checkpoint(self, checkpoint_dir: str | os.PathLike[str]) -> Path:
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / HIVE_SIGNAL_COUNTERS_FILENAME
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=directory, prefix=f".{HIVE_SIGNAL_COUNTERS_FILENAME}.", suffix=".tmp"
        )
        os.close(file_descriptor)
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
        expected_global_step: int | None = None,
    ) -> "HiveSignalCounters":
        checkpoint_path = Path(checkpoint_dir) / HIVE_SIGNAL_COUNTERS_FILENAME
        with open(checkpoint_path, "r", encoding="utf-8") as handle:
            counters = cls.from_dict(json.load(handle))
        if expected_global_step is not None and counters.global_step != expected_global_step:
            raise ValueError(
                f"HIVE signal counters global_step {counters.global_step} does not match "
                f"trainer checkpoint step {expected_global_step}"
            )
        return counters
