"""Pure HIVE post-rollout interpretation and atomic step-state publication."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from signal_forge.hive.stage1 import (
    ExplorationControllerConfig,
    ExplorationUpdate,
    Stage1StepSelector,
    apply_exploration_update,
    compute_exploration_update,
)
from signal_forge.hive.state import HiveSelectorSnapshot, HiveSelectorState, PromptVisit, ZeroVarianceType
from verl import DataProto


HIVE_COMPUTE_COUNTERS_FILENAME = "hive_compute_counters.json"
HIVE_COMPUTE_COUNTERS_SCHEMA_VERSION = 1


class HiveInsufficientEffectiveGroupsError(RuntimeError):
    """Phase-6 stop condition used until adaptive post-rollout top-up exists."""


@dataclass(frozen=True)
class HivePostRolloutConfig:
    effective_batch_size: int
    group_size: int = 8
    controller: ExplorationControllerConfig = ExplorationControllerConfig()

    def __post_init__(self) -> None:
        _positive_integer("effective_batch_size", self.effective_batch_size)
        _positive_integer("group_size", self.group_size)


@dataclass(frozen=True)
class PendingPromptVisit:
    prompt_id: str
    visit: PromptVisit


@dataclass(frozen=True)
class HivePostRolloutDiagnostics:
    generated_prompt_groups: int
    generated_responses: int
    generated_prompt_tokens: int
    generated_response_tokens: int
    easy_zero_var_groups: int
    hard_zero_var_groups: int
    other_zero_var_groups: int
    total_zero_var_groups: int
    effective_prompt_groups: int
    effective_responses: int
    effective_response_tokens: int
    training_prompt_groups: int
    training_responses: int
    discarded_zero_var_groups: int
    effective_but_not_trained_groups: int


@dataclass
class HiveComputeCounters:
    generated_prompt_groups: int = 0
    generated_responses: int = 0
    generated_response_tokens: int = 0
    effective_prompt_groups: int = 0
    effective_responses: int = 0
    effective_response_tokens: int = 0

    def update(self, diagnostics: HivePostRolloutDiagnostics) -> dict[str, float]:
        self.generated_prompt_groups += diagnostics.generated_prompt_groups
        self.generated_responses += diagnostics.generated_responses
        self.generated_response_tokens += diagnostics.generated_response_tokens
        self.effective_prompt_groups += diagnostics.effective_prompt_groups
        self.effective_responses += diagnostics.effective_responses
        self.effective_response_tokens += diagnostics.effective_response_tokens
        return {
            "compute/generated_prompt_groups": float(self.generated_prompt_groups),
            "compute/generated_responses": float(self.generated_responses),
            "compute/generated_response_tokens": float(self.generated_response_tokens),
            "compute/effective_prompt_groups": float(self.effective_prompt_groups),
            "compute/effective_responses": float(self.effective_responses),
            "compute/effective_response_tokens": float(self.effective_response_tokens),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HIVE_COMPUTE_COUNTERS_SCHEMA_VERSION,
            "generated_prompt_groups": self.generated_prompt_groups,
            "generated_responses": self.generated_responses,
            "generated_response_tokens": self.generated_response_tokens,
            "effective_prompt_groups": self.effective_prompt_groups,
            "effective_responses": self.effective_responses,
            "effective_response_tokens": self.effective_response_tokens,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HiveComputeCounters":
        if payload.get("schema_version") != HIVE_COMPUTE_COUNTERS_SCHEMA_VERSION:
            raise ValueError("unsupported HIVE compute counters schema_version")
        names = (
            "generated_prompt_groups",
            "generated_responses",
            "generated_response_tokens",
            "effective_prompt_groups",
            "effective_responses",
            "effective_response_tokens",
        )
        values = {name: _nonnegative_integer(name, payload.get(name)) for name in names}
        return cls(**values)

    def save_checkpoint(self, checkpoint_dir: str | os.PathLike[str]) -> Path:
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / HIVE_COMPUTE_COUNTERS_FILENAME
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=directory, prefix=f".{HIVE_COMPUTE_COUNTERS_FILENAME}.", suffix=".tmp"
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
    def load_checkpoint(cls, checkpoint_dir: str | os.PathLike[str]) -> "HiveComputeCounters":
        checkpoint_path = Path(checkpoint_dir) / HIVE_COMPUTE_COUNTERS_FILENAME
        with open(checkpoint_path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


class HiveStepPendingCommit:
    """Pending visits/controller update that remain invisible until explicit commit."""

    def __init__(
        self,
        *,
        selector_snapshot: HiveSelectorSnapshot,
        step: int,
        visits: Sequence[PendingPromptVisit],
        exploration_update: ExplorationUpdate,
    ) -> None:
        self.selector_snapshot = selector_snapshot
        self.step = step
        self.visits = tuple(visits)
        self.exploration_update = exploration_update
        self._committed = False

    def commit(self, state: HiveSelectorState, selector: Stage1StepSelector) -> dict[str, float]:
        """Validate on a state copy, then publish all step-local selector changes together."""
        if self._committed:
            raise RuntimeError("HIVE step observations have already been committed")
        if selector.snapshot != self.selector_snapshot:
            raise RuntimeError("Stage-1 selector is not bound to the pending commit snapshot")
        _validate_live_state_matches_snapshot(state, self.selector_snapshot)

        candidate = HiveSelectorState.from_dict(state.to_dict())
        selector.commit_rng_state(candidate)
        for pending in self.visits:
            candidate.append_visit(pending.prompt_id, pending.visit)
        apply_exploration_update(candidate, self.exploration_update)
        candidate.global_step = self.step

        state.prompt_history = candidate.prompt_history
        state.p_easy = candidate.p_easy
        state.p_hard = candidate.p_hard
        state.p_default = candidate.p_default
        state.global_step = candidate.global_step
        state.selector_rng_state = candidate.selector_rng_state
        state.configuration = candidate.configuration
        self._committed = True
        return {
            "hive/p_easy_before": self.exploration_update.p_easy_before,
            "hive/p_easy_after": self.exploration_update.p_easy_after,
            "hive/p_hard_before": self.exploration_update.p_hard_before,
            "hive/p_hard_after": self.exploration_update.p_hard_after,
            "hive/history_visits_committed": float(len(self.visits)),
        }


@dataclass(frozen=True)
class HivePostRolloutResult:
    training_batch: DataProto | None
    training_reward_tensor: torch.Tensor | None
    training_reward_extra_infos: dict[str, Any]
    pending_commit: HiveStepPendingCommit
    diagnostics: HivePostRolloutDiagnostics
    metrics: dict[str, float]

    def require_training_batch(self) -> tuple[DataProto, torch.Tensor, dict[str, Any]]:
        if self.training_batch is None or self.training_reward_tensor is None:
            raise HiveInsufficientEffectiveGroupsError(
                "HIVE Phase 6 produced fewer effective prompt groups than B_t; "
                "adaptive top-up is not implemented in this phase "
                f"(effective={self.diagnostics.effective_prompt_groups}, "
                f"required={self.metrics['hive/required_prompt_groups']:.0f})"
            )
        return self.training_batch, self.training_reward_tensor, self.training_reward_extra_infos


class HivePostRolloutInterpreter:
    """Interpret one complete pre-rollout candidate pool after reward computation."""

    def __init__(
        self,
        *,
        selector_snapshot: HiveSelectorSnapshot,
        config: HivePostRolloutConfig,
    ) -> None:
        if not isinstance(selector_snapshot, HiveSelectorSnapshot):
            raise TypeError("selector_snapshot must be a HiveSelectorSnapshot")
        if not isinstance(config, HivePostRolloutConfig):
            raise TypeError("config must be a HivePostRolloutConfig")
        if selector_snapshot.group_size != config.group_size:
            raise ValueError("post-rollout group_size must match the selector snapshot")
        self.selector_snapshot = selector_snapshot
        self.config = config

    def interpret(
        self,
        *,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos: Mapping[str, Any],
        candidate_prompt_ids: Sequence[str],
        step: int,
    ) -> HivePostRolloutResult:
        if not isinstance(batch, DataProto):
            raise TypeError("batch must be a DataProto")
        if isinstance(step, bool) or not isinstance(step, int) or step <= self.selector_snapshot.global_step:
            raise ValueError("post-rollout step must be greater than the selector snapshot global_step")
        prompt_order = _validate_prompt_order(candidate_prompt_ids)
        prompt_ids = _aligned_object_values(batch, "prompt_id")
        temporary_uids = _aligned_object_values(batch, "uid")
        outcomes = _structured_reward_outcomes(reward_extra_infos, len(batch))
        _validate_reward_tensor(reward_tensor, outcomes, len(batch))

        grouped_indices: dict[str, list[int]] = {prompt_id: [] for prompt_id in prompt_order}
        for index, prompt_id in enumerate(prompt_ids):
            if prompt_id not in grouped_indices:
                raise ValueError(f"rollout returned unexpected stable prompt_id {prompt_id!r}")
            grouped_indices[prompt_id].append(index)
        if set(prompt_ids) != set(prompt_order):
            raise ValueError("rollout stable prompt_ids do not match the candidate prompt set")

        pending_visits: list[PendingPromptVisit] = []
        effective_group_indices: list[list[int]] = []
        seen_uids: set[object] = set()
        for prompt_id in prompt_order:
            indices = grouped_indices[prompt_id]
            if len(indices) != self.config.group_size:
                raise ValueError(
                    f"prompt {prompt_id!r} must contain exactly {self.config.group_size} rollout responses; "
                    f"got {len(indices)}"
                )
            group_uids = {temporary_uids[index] for index in indices}
            if len(group_uids) != 1 or None in group_uids or "" in group_uids:
                raise ValueError(f"prompt {prompt_id!r} does not have one valid temporary rollout uid")
            temporary_uid = next(iter(group_uids))
            if temporary_uid in seen_uids:
                raise ValueError("temporary rollout uid is shared by multiple stable prompt_ids")
            seen_uids.add(temporary_uid)

            rewards = [outcomes[index][2] for index in indices]
            visit = PromptVisit.from_rewards(
                step=step,
                rewards=rewards,
                group_size=self.config.group_size,
                response_entropy=None,
            )
            pending_visits.append(PendingPromptVisit(prompt_id=prompt_id, visit=visit))
            if not visit.zero_variance:
                effective_group_indices.append(indices)

        generated_groups = len(prompt_order)
        easy_groups = sum(
            pending.visit.zero_variance_type is ZeroVarianceType.EASY for pending in pending_visits
        )
        hard_groups = sum(
            pending.visit.zero_variance_type is ZeroVarianceType.HARD for pending in pending_visits
        )
        other_groups = sum(
            pending.visit.zero_variance_type is ZeroVarianceType.OTHER for pending in pending_visits
        )
        zero_var_groups = easy_groups + hard_groups + other_groups
        effective_groups = len(effective_group_indices)
        training_group_indices = effective_group_indices[: self.config.effective_batch_size]
        training_indices = [index for indices in training_group_indices for index in indices]
        effective_indices = [index for indices in effective_group_indices for index in indices]
        response_width = batch.batch["responses"].shape[-1]
        generated_prompt_tokens = int(
            batch.batch["attention_mask"][:, :-response_width].sum().detach().cpu().item()
        )
        generated_response_tokens = _response_tokens(batch, range(len(batch)))
        effective_response_tokens = _response_tokens(batch, effective_indices)

        exploration_update = compute_exploration_update(
            self.selector_snapshot,
            observed_easy_ratio=easy_groups / generated_groups,
            observed_hard_ratio=hard_groups / generated_groups,
            config=self.config.controller,
        )
        pending_commit = HiveStepPendingCommit(
            selector_snapshot=self.selector_snapshot,
            step=step,
            visits=pending_visits,
            exploration_update=exploration_update,
        )

        can_train = effective_groups >= self.config.effective_batch_size
        training_batch = None
        training_reward_tensor = None
        training_reward_infos: dict[str, Any] = {}
        if can_train:
            training_batch = batch.select_idxs(training_indices)
            training_batch.meta_info = _filtered_meta_info(batch.meta_info, training_indices, len(batch))
            training_reward_tensor = reward_tensor[training_indices]
            training_reward_infos = _filter_aligned_mapping(reward_extra_infos, training_indices, len(batch))

        diagnostics = HivePostRolloutDiagnostics(
            generated_prompt_groups=generated_groups,
            generated_responses=len(batch),
            generated_prompt_tokens=generated_prompt_tokens,
            generated_response_tokens=generated_response_tokens,
            easy_zero_var_groups=easy_groups,
            hard_zero_var_groups=hard_groups,
            other_zero_var_groups=other_groups,
            total_zero_var_groups=zero_var_groups,
            effective_prompt_groups=effective_groups,
            effective_responses=len(effective_indices),
            effective_response_tokens=effective_response_tokens,
            training_prompt_groups=len(training_group_indices) if can_train else 0,
            training_responses=len(training_indices) if can_train else 0,
            discarded_zero_var_groups=zero_var_groups,
            effective_but_not_trained_groups=max(effective_groups - self.config.effective_batch_size, 0),
        )
        metrics = _metrics(diagnostics, exploration_update, self.config.effective_batch_size)
        return HivePostRolloutResult(
            training_batch=training_batch,
            training_reward_tensor=training_reward_tensor,
            training_reward_extra_infos=training_reward_infos,
            pending_commit=pending_commit,
            diagnostics=diagnostics,
            metrics=metrics,
        )


def _structured_reward_outcomes(
    reward_extra_infos: Mapping[str, Any], batch_size: int
) -> tuple[tuple[bool, bool, float], ...]:
    required = ("extracted", "correct", "reward")
    values = {}
    for key in required:
        if key not in reward_extra_infos:
            raise ValueError(f"HIVE reward output is missing structured field {key!r}")
        value = reward_extra_infos[key]
        value = value.tolist() if hasattr(value, "tolist") else list(value)
        if len(value) != batch_size:
            raise ValueError(f"HIVE reward field {key!r} must align with the rollout batch")
        values[key] = value

    outcomes = []
    for index, (extracted, correct, reward) in enumerate(
        zip(values["extracted"], values["correct"], values["reward"], strict=True)
    ):
        if not isinstance(extracted, (bool, np.bool_)) or not isinstance(correct, (bool, np.bool_)):
            raise ValueError(f"structured reward booleans are invalid at response index {index}")
        if correct and not extracted:
            raise ValueError(f"correct response at index {index} cannot have extracted=False")
        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise ValueError(f"structured reward is invalid at response index {index}")
        normalized_reward = float(reward)
        expected_reward = 1.0 if correct else 0.1 if extracted else 0.0
        if not math.isfinite(normalized_reward) or normalized_reward != expected_reward:
            raise ValueError(
                f"structured reward at response index {index} violates frozen semantics: "
                f"got {normalized_reward}, expected {expected_reward}"
            )
        outcomes.append((bool(extracted), bool(correct), normalized_reward))
    return tuple(outcomes)


def _validate_reward_tensor(
    reward_tensor: torch.Tensor,
    outcomes: Sequence[tuple[bool, bool, float]],
    batch_size: int,
) -> None:
    if not isinstance(reward_tensor, torch.Tensor) or reward_tensor.ndim != 2:
        raise ValueError("reward_tensor must be a rank-2 token-level tensor")
    if reward_tensor.shape[0] != batch_size:
        raise ValueError("reward_tensor must align with the rollout batch")
    scalar_rewards = reward_tensor.sum(dim=-1).detach().cpu().float()
    structured = torch.tensor([outcome[2] for outcome in outcomes], dtype=torch.float32)
    if not torch.allclose(scalar_rewards, structured, rtol=0.0, atol=1e-6):
        raise ValueError("reward tensor does not match structured HIVE reward values")


def _aligned_object_values(batch: DataProto, key: str) -> tuple[Any, ...]:
    if key not in batch.non_tensor_batch:
        raise ValueError(f"rollout batch is missing {key!r}")
    values = tuple(np.asarray(batch.non_tensor_batch[key], dtype=object).tolist())
    if len(values) != len(batch):
        raise ValueError(f"rollout field {key!r} must align with the batch")
    return values


def _validate_prompt_order(candidate_prompt_ids: Sequence[str]) -> tuple[str, ...]:
    prompt_ids = tuple(candidate_prompt_ids)
    if not prompt_ids:
        raise ValueError("candidate_prompt_ids must not be empty")
    if any(not isinstance(prompt_id, str) or not prompt_id.strip() for prompt_id in prompt_ids):
        raise ValueError("candidate_prompt_ids must contain non-empty strings")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("candidate_prompt_ids must be unique")
    return prompt_ids


def _response_tokens(batch: DataProto, indices: Sequence[int] | range) -> int:
    if "response_mask" not in batch.batch:
        raise ValueError("rollout batch is missing response_mask")
    normalized = list(indices)
    if not normalized:
        return 0
    return int(batch.batch["response_mask"][normalized].sum().detach().cpu().item())


def _filter_aligned_mapping(mapping: Mapping[str, Any], indices: Sequence[int], batch_size: int) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=np.int64)
    output = {}
    for key, value in mapping.items():
        normalized = value.tolist() if hasattr(value, "tolist") else value
        if isinstance(normalized, (list, tuple)) and len(normalized) == batch_size:
            output[key] = np.asarray(normalized, dtype=object)[selected].tolist()
        else:
            output[key] = value
    return output


def _filtered_meta_info(meta_info: Mapping[str, Any], indices: Sequence[int], batch_size: int) -> dict[str, Any]:
    output = copy.deepcopy(dict(meta_info))
    for key, value in list(output.items()):
        if isinstance(value, list) and len(value) == batch_size:
            output[key] = [value[index] for index in indices]
    return output


def _validate_live_state_matches_snapshot(state: HiveSelectorState, snapshot: HiveSelectorSnapshot) -> None:
    current = state.snapshot()
    if current.group_size != snapshot.group_size or current.global_step != snapshot.global_step:
        raise RuntimeError("live HIVE selector state changed after the step snapshot")
    if (
        current.p_easy != snapshot.p_easy
        or current.p_hard != snapshot.p_hard
        or current.p_default != snapshot.p_default
    ):
        raise RuntimeError("live HIVE exploration probabilities changed after the step snapshot")
    if dict(current.prompt_history) != dict(snapshot.prompt_history):
        raise RuntimeError("live HIVE prompt history changed after the step snapshot")
    if current.selector_rng_state != snapshot.selector_rng_state:
        raise RuntimeError("live HIVE selector RNG changed after the step snapshot")


def _metrics(
    diagnostics: HivePostRolloutDiagnostics,
    update: ExplorationUpdate,
    required_prompt_groups: int,
) -> dict[str, float]:
    generated = diagnostics.generated_prompt_groups
    return {
        "hive/generated_prompt_groups": float(generated),
        "hive/generated_responses": float(diagnostics.generated_responses),
        "hive/generated_prompt_tokens": float(diagnostics.generated_prompt_tokens),
        "hive/generated_response_tokens": float(diagnostics.generated_response_tokens),
        "hive/easy_zero_var_groups": float(diagnostics.easy_zero_var_groups),
        "hive/hard_zero_var_groups": float(diagnostics.hard_zero_var_groups),
        "hive/other_zero_var_groups": float(diagnostics.other_zero_var_groups),
        "hive/total_zero_var_groups": float(diagnostics.total_zero_var_groups),
        "hive/easy_zero_var_ratio": diagnostics.easy_zero_var_groups / generated,
        "hive/hard_zero_var_ratio": diagnostics.hard_zero_var_groups / generated,
        "hive/total_zero_var_ratio": diagnostics.total_zero_var_groups / generated,
        "hive/effective_prompt_groups": float(diagnostics.effective_prompt_groups),
        "hive/effective_responses": float(diagnostics.effective_responses),
        "hive/effective_response_tokens": float(diagnostics.effective_response_tokens),
        "hive/training_prompt_groups": float(diagnostics.training_prompt_groups),
        "hive/discarded_zero_var_groups": float(diagnostics.discarded_zero_var_groups),
        "hive/effective_but_not_trained_groups": float(diagnostics.effective_but_not_trained_groups),
        "hive/p_easy_before": update.p_easy_before,
        "hive/p_easy_after": update.p_easy_after,
        "hive/p_hard_before": update.p_hard_before,
        "hive/p_hard_after": update.p_hard_after,
        "hive/history_visits_committed": 0.0,
        "hive/required_prompt_groups": float(required_prompt_groups),
    }


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
