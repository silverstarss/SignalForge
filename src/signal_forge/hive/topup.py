"""Pure adaptive HIVE top-up sizing and effective-group accumulation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Mapping

import numpy as np
import torch

from signal_forge.hive.post_rollout import (
    HivePostRolloutDiagnostics,
    HivePostRolloutResult,
    HiveStepPendingCommit,
)
from signal_forge.hive.stage1 import ExplorationControllerConfig, compute_exploration_update
from signal_forge.hive.state import HiveSelectorSnapshot, ZeroVarianceType
from verl import DataProto


@dataclass(frozen=True)
class HiveTopupRoundDiagnostics:
    round_index: int
    generated_prompt_groups: int
    easy_zero_var_groups: int
    hard_zero_var_groups: int
    other_zero_var_groups: int
    effective_groups: int
    total_zero_var_ratio: float
    candidate_target: int
    candidate_actual: int
    candidate_overshoot: int
    cumulative_rho_zv: float


@dataclass(frozen=True)
class HiveTopupFailureDiagnostics:
    required_effective_groups: int
    effective_groups: int
    topup_rounds: int
    rounds: tuple[HiveTopupRoundDiagnostics, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HiveTopupError(RuntimeError):
    """Base class for explicit adaptive top-up failures."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: HiveTopupFailureDiagnostics | None = None,
    ) -> None:
        self.diagnostics = diagnostics
        if diagnostics is not None:
            message = (
                f"{message}\nHIVE top-up diagnostic snapshot: "
                f"{json.dumps(diagnostics.to_dict(), sort_keys=True)}"
            )
        super().__init__(message)


class HiveTopupSurvivalError(HiveTopupError):
    """The observed survival rate cannot support a finite estimate."""


class HiveTopupRoundLimitError(HiveTopupError):
    """The configured top-up round limit was reached before filling B_t."""


class HiveTopupDataExhaustedError(HiveTopupError):
    """The dataloader ended before the effective buffer reached B_t."""


@dataclass(frozen=True)
class HiveAdaptiveTopupConfig:
    effective_batch_size: int
    group_size: int = 8
    eta: float = 1.25
    b_min: int = 64
    max_topup_rounds: int = 8
    survival_epsilon: float = 1e-6
    controller: ExplorationControllerConfig = ExplorationControllerConfig()

    def __post_init__(self) -> None:
        _positive_integer("effective_batch_size", self.effective_batch_size)
        _positive_integer("group_size", self.group_size)
        _positive_integer("b_min", self.b_min)
        _positive_integer("max_topup_rounds", self.max_topup_rounds)
        _positive_finite("eta", self.eta)
        epsilon = _positive_finite("survival_epsilon", self.survival_epsilon)
        if epsilon >= 1.0:
            raise ValueError("survival_epsilon must be less than 1")
        if (3 * self.effective_batch_size) % 2:
            raise ValueError("faithful HIVE requires 3 * B_t to be divisible by 2")
        if self.b_min > self.candidate_cap:
            raise ValueError(
                "faithful HIVE requires b_min <= B_cand: "
                f"b_min={self.b_min}, B_cand={self.candidate_cap}"
            )

    @property
    def candidate_cap(self) -> int:
        return 3 * self.effective_batch_size // 2


@dataclass(frozen=True)
class HiveAdaptiveTopupEstimate:
    remaining_groups: int
    remaining_responses: int
    rho_zv: float
    survival_rate: float
    estimated_candidates: int
    candidate_target: int
    b_min_binding: bool
    candidate_cap_binding: bool


@dataclass(frozen=True)
class HiveTopupAcquisitionDiagnostics:
    candidate_target: int
    candidate_actual: int
    raw_prompts_seen: int

    def __post_init__(self) -> None:
        _positive_integer("candidate_target", self.candidate_target)
        _nonnegative_integer("candidate_actual", self.candidate_actual)
        _nonnegative_integer("raw_prompts_seen", self.raw_prompts_seen)
        if self.candidate_actual < self.candidate_target:
            raise ValueError("top-up candidate_actual must reach the candidate_target lower bound")


@dataclass(frozen=True)
class HiveAdaptiveTopupResult:
    training_batch: DataProto
    training_reward_tensor: torch.Tensor
    training_reward_extra_infos: dict[str, Any]
    pending_commit: HiveStepPendingCommit
    diagnostics: HivePostRolloutDiagnostics
    metrics: dict[str, float]


def observed_zero_variance_ratio(*, zero_variance_groups: int, generated_groups: int) -> float:
    """Return current-step cumulative zero-var groups / complete rolled-out groups."""
    zero = _nonnegative_integer("zero_variance_groups", zero_variance_groups)
    total = _positive_integer("generated_groups", generated_groups)
    if zero > total:
        raise ValueError("zero_variance_groups cannot exceed generated_groups")
    return zero / total


def compute_adaptive_candidate_target(
    config: HiveAdaptiveTopupConfig,
    *,
    remaining_groups: int,
    rho_zv: Real,
) -> HiveAdaptiveTopupEstimate:
    """Apply Appendix B.3 using explicit ceil, cap, then b_min semantics."""
    if not isinstance(config, HiveAdaptiveTopupConfig):
        raise TypeError("config must be a HiveAdaptiveTopupConfig")
    remaining = _nonnegative_integer("remaining_groups", remaining_groups)
    rho = _finite_ratio("rho_zv", rho_zv)
    survival = 1.0 - rho
    if remaining == 0:
        return HiveAdaptiveTopupEstimate(
            remaining_groups=0,
            remaining_responses=0,
            rho_zv=rho,
            survival_rate=survival,
            estimated_candidates=0,
            candidate_target=0,
            b_min_binding=False,
            candidate_cap_binding=False,
        )
    if survival <= config.survival_epsilon:
        return HiveAdaptiveTopupEstimate(
            remaining_groups=remaining,
            remaining_responses=remaining * config.group_size,
            rho_zv=rho,
            survival_rate=survival,
            estimated_candidates=config.candidate_cap,
            candidate_target=config.candidate_cap,
            b_min_binding=config.b_min == config.candidate_cap,
            candidate_cap_binding=True,
        )

    estimated = math.ceil(float(config.eta) * remaining / survival)
    capped = min(config.candidate_cap, estimated)
    target = max(config.b_min, capped)
    return HiveAdaptiveTopupEstimate(
        remaining_groups=remaining,
        remaining_responses=remaining * config.group_size,
        rho_zv=rho,
        survival_rate=survival,
        estimated_candidates=estimated,
        candidate_target=target,
        b_min_binding=config.b_min >= capped,
        candidate_cap_binding=estimated >= config.candidate_cap,
    )


def compute_adaptive_candidate_target_from_responses(
    config: HiveAdaptiveTopupConfig,
    *,
    effective_response_count: int,
    rho_zv: Real,
) -> HiveAdaptiveTopupEstimate:
    """Response-level form of the paper equation, with complete-group validation."""
    effective = _nonnegative_integer("effective_response_count", effective_response_count)
    required = config.effective_batch_size * config.group_size
    if effective > required:
        effective = required
    remaining_responses = required - effective
    if remaining_responses % config.group_size:
        raise ValueError("effective_response_count must preserve complete rollout groups")
    result = compute_adaptive_candidate_target(
        config,
        remaining_groups=remaining_responses // config.group_size,
        rho_zv=rho_zv,
    )
    if result.remaining_responses != remaining_responses:
        raise RuntimeError("response-level and group-level top-up equations disagree")
    return result


class HiveAdaptiveTopupAccumulator:
    """Accumulate all current-step rollout rounds against one frozen selector snapshot."""

    def __init__(self, *, selector_snapshot: HiveSelectorSnapshot, config: HiveAdaptiveTopupConfig) -> None:
        if not isinstance(selector_snapshot, HiveSelectorSnapshot):
            raise TypeError("selector_snapshot must be a HiveSelectorSnapshot")
        if selector_snapshot.group_size != config.group_size:
            raise ValueError("top-up group_size must match the selector snapshot")
        self.selector_snapshot = selector_snapshot
        self.config = config
        self._rounds: list[HivePostRolloutResult] = []
        self._initial_acquisition: HiveTopupAcquisitionDiagnostics | None = None
        self._topup_acquisitions: list[HiveTopupAcquisitionDiagnostics] = []
        self._topup_estimates: list[HiveAdaptiveTopupEstimate] = []
        self._pending_estimate: HiveAdaptiveTopupEstimate | None = None
        self._prompt_ids: set[str] = set()

    @property
    def generated_group_count(self) -> int:
        return sum(result.diagnostics.generated_prompt_groups for result in self._rounds)

    @property
    def zero_variance_group_count(self) -> int:
        return sum(result.diagnostics.total_zero_var_groups for result in self._rounds)

    @property
    def effective_group_count(self) -> int:
        return sum(result.diagnostics.effective_prompt_groups for result in self._rounds)

    @property
    def topup_rounds(self) -> int:
        return len(self._topup_acquisitions)

    @property
    def prompt_ids(self) -> frozenset[str]:
        return frozenset(self._prompt_ids)

    @property
    def is_complete(self) -> bool:
        return self.effective_group_count >= self.config.effective_batch_size

    @property
    def rho_zv(self) -> float:
        if not self._rounds:
            raise RuntimeError("rho_zv is unavailable before the initial rollout round")
        return observed_zero_variance_ratio(
            zero_variance_groups=self.zero_variance_group_count,
            generated_groups=self.generated_group_count,
        )

    def observe_initial(
        self,
        result: HivePostRolloutResult,
        acquisition: HiveTopupAcquisitionDiagnostics | None = None,
    ) -> None:
        if self._rounds:
            raise RuntimeError("the initial rollout round has already been observed")
        generated = result.diagnostics.generated_prompt_groups
        if acquisition is None:
            acquisition = HiveTopupAcquisitionDiagnostics(
                candidate_target=generated,
                candidate_actual=generated,
                raw_prompts_seen=generated,
            )
        if acquisition.candidate_actual != generated:
            raise ValueError("every initial candidate must receive one complete rollout group")
        self._initial_acquisition = acquisition
        self._append_result(result)

    def failure_diagnostics(self) -> HiveTopupFailureDiagnostics:
        """Capture completed rollout rounds even when normal finalization is skipped."""
        if not self._rounds or self._initial_acquisition is None:
            return HiveTopupFailureDiagnostics(
                required_effective_groups=self.config.effective_batch_size,
                effective_groups=0,
                topup_rounds=0,
                rounds=(),
            )
        acquisitions = [self._initial_acquisition, *self._topup_acquisitions]
        cumulative_generated = 0
        cumulative_zero = 0
        rounds: list[HiveTopupRoundDiagnostics] = []
        for index, (result, acquisition) in enumerate(
            zip(self._rounds, acquisitions, strict=True)
        ):
            item = result.diagnostics
            cumulative_generated += item.generated_prompt_groups
            cumulative_zero += item.total_zero_var_groups
            rounds.append(
                HiveTopupRoundDiagnostics(
                    round_index=index,
                    generated_prompt_groups=item.generated_prompt_groups,
                    easy_zero_var_groups=item.easy_zero_var_groups,
                    hard_zero_var_groups=item.hard_zero_var_groups,
                    other_zero_var_groups=item.other_zero_var_groups,
                    effective_groups=item.effective_prompt_groups,
                    total_zero_var_ratio=(
                        item.total_zero_var_groups / item.generated_prompt_groups
                    ),
                    candidate_target=acquisition.candidate_target,
                    candidate_actual=acquisition.candidate_actual,
                    candidate_overshoot=(
                        acquisition.candidate_actual - acquisition.candidate_target
                    ),
                    cumulative_rho_zv=cumulative_zero / cumulative_generated,
                )
            )
        return HiveTopupFailureDiagnostics(
            required_effective_groups=self.config.effective_batch_size,
            effective_groups=self.effective_group_count,
            topup_rounds=self.topup_rounds,
            rounds=tuple(rounds),
        )

    def plan_next_topup(self) -> HiveAdaptiveTopupEstimate | None:
        if not self._rounds:
            raise RuntimeError("initial rollout observations are required before top-up sizing")
        if self.is_complete:
            return None
        if self._pending_estimate is not None:
            raise RuntimeError("the previous top-up plan has not been observed")
        if self.topup_rounds >= self.config.max_topup_rounds:
            raise HiveTopupRoundLimitError(
                "HIVE adaptive top-up exceeded max_topup_rounds before filling B_t: "
                f"rounds={self.topup_rounds}, effective={self.effective_group_count}, "
                f"required={self.config.effective_batch_size}, rho_zv={self.rho_zv}",
                diagnostics=self.failure_diagnostics(),
            )
        estimate = compute_adaptive_candidate_target(
            self.config,
            remaining_groups=self.config.effective_batch_size - self.effective_group_count,
            rho_zv=self.rho_zv,
        )
        self._pending_estimate = estimate
        return estimate

    def observe_topup(
        self,
        result: HivePostRolloutResult,
        acquisition: HiveTopupAcquisitionDiagnostics,
    ) -> None:
        if self._pending_estimate is None:
            raise RuntimeError("plan_next_topup must be called before observing a top-up round")
        if acquisition.candidate_target != self._pending_estimate.candidate_target:
            raise ValueError("top-up acquisition target does not match the adaptive estimate")
        if acquisition.candidate_actual != result.diagnostics.generated_prompt_groups:
            raise ValueError("every acquired top-up candidate must receive one complete rollout group")
        self._append_result(result)
        self._topup_estimates.append(self._pending_estimate)
        self._topup_acquisitions.append(acquisition)
        self._pending_estimate = None

    def finalize(self, *, step: int) -> HiveAdaptiveTopupResult:
        if self._pending_estimate is not None:
            raise RuntimeError("cannot finalize while a top-up acquisition is pending")
        if not self.is_complete:
            raise HiveTopupError(
                "HIVE effective buffer is incomplete; acquire another adaptive top-up round first",
                diagnostics=self.failure_diagnostics(),
            )
        effective_batches = [result.effective_batch for result in self._rounds if result.effective_batch is not None]
        effective_rewards = [
            result.effective_reward_tensor
            for result in self._rounds
            if result.effective_reward_tensor is not None
        ]
        if not effective_batches or len(effective_batches) != len(effective_rewards):
            raise RuntimeError("complete HIVE effective buffer is missing aligned tensors")
        for part in effective_batches:
            part.meta_info.pop("global_token_num", None)
            part.meta_info.pop("images_seqlens", None)
        effective_batch = DataProto.concat(effective_batches)
        effective_reward_tensor = torch.cat(effective_rewards, dim=0)
        effective_reward_infos = _concat_aligned_mappings(
            [result.effective_reward_extra_infos for result in self._rounds if result.effective_batch is not None]
        )
        _validate_complete_groups(effective_batch, self.config.group_size)
        training_response_count = self.config.effective_batch_size * self.config.group_size
        indices = list(range(training_response_count))
        training_batch = effective_batch.select_idxs(indices)
        training_reward_tensor = effective_reward_tensor[indices]
        training_reward_infos = _filter_aligned_mapping(
            effective_reward_infos,
            indices,
            len(effective_batch),
        )

        visits = tuple(pending for result in self._rounds for pending in result.pending_commit.visits)
        easy = sum(item.visit.zero_variance_type is ZeroVarianceType.EASY for item in visits)
        hard = sum(item.visit.zero_variance_type is ZeroVarianceType.HARD for item in visits)
        generated = len(visits)
        update = compute_exploration_update(
            self.selector_snapshot,
            observed_easy_ratio=easy / generated,
            observed_hard_ratio=hard / generated,
            config=self.config.controller,
        )
        pending_commit = HiveStepPendingCommit(
            selector_snapshot=self.selector_snapshot,
            step=step,
            visits=visits,
            exploration_update=update,
        )
        diagnostics = _aggregate_diagnostics(self._rounds, self.config)
        metrics = _aggregate_metrics(
            diagnostics=diagnostics,
            update=update,
            config=self.config,
            topup_estimates=self._topup_estimates,
            acquisitions=self._topup_acquisitions,
            initial_effective=self._rounds[0].diagnostics.effective_prompt_groups,
            topup_results=self._rounds[1:],
        )
        for item in self.failure_diagnostics().rounds:
            prefix = f"hive/rollout_round_{item.round_index}"
            metrics.update(
                {
                    f"{prefix}/generated_prompt_groups": float(item.generated_prompt_groups),
                    f"{prefix}/easy_zero_var_groups": float(item.easy_zero_var_groups),
                    f"{prefix}/hard_zero_var_groups": float(item.hard_zero_var_groups),
                    f"{prefix}/other_zero_var_groups": float(item.other_zero_var_groups),
                    f"{prefix}/effective_groups": float(item.effective_groups),
                    f"{prefix}/total_zero_var_ratio": item.total_zero_var_ratio,
                    f"{prefix}/candidate_target": float(item.candidate_target),
                    f"{prefix}/candidate_actual": float(item.candidate_actual),
                    f"{prefix}/candidate_overshoot": float(item.candidate_overshoot),
                    f"{prefix}/cumulative_rho_zv": item.cumulative_rho_zv,
                }
            )
        return HiveAdaptiveTopupResult(
            training_batch=training_batch,
            training_reward_tensor=training_reward_tensor,
            training_reward_extra_infos=training_reward_infos,
            pending_commit=pending_commit,
            diagnostics=diagnostics,
            metrics=metrics,
        )

    def _append_result(self, result: HivePostRolloutResult) -> None:
        if result.pending_commit.selector_snapshot != self.selector_snapshot:
            raise ValueError("all top-up rounds must use the same frozen selector snapshot")
        prompt_ids = {pending.prompt_id for pending in result.pending_commit.visits}
        duplicates = self._prompt_ids.intersection(prompt_ids)
        if duplicates:
            raise ValueError(f"top-up rollout rounds contain duplicate stable prompt_ids: {sorted(duplicates)[:5]}")
        self._prompt_ids.update(prompt_ids)
        self._rounds.append(result)


def _aggregate_diagnostics(
    results: list[HivePostRolloutResult], config: HiveAdaptiveTopupConfig
) -> HivePostRolloutDiagnostics:
    items = [result.diagnostics for result in results]
    effective_groups = sum(item.effective_prompt_groups for item in items)
    return HivePostRolloutDiagnostics(
        generated_prompt_groups=sum(item.generated_prompt_groups for item in items),
        generated_responses=sum(item.generated_responses for item in items),
        generated_prompt_tokens=sum(item.generated_prompt_tokens for item in items),
        generated_response_tokens=sum(item.generated_response_tokens for item in items),
        easy_zero_var_groups=sum(item.easy_zero_var_groups for item in items),
        hard_zero_var_groups=sum(item.hard_zero_var_groups for item in items),
        other_zero_var_groups=sum(item.other_zero_var_groups for item in items),
        total_zero_var_groups=sum(item.total_zero_var_groups for item in items),
        effective_prompt_groups=effective_groups,
        effective_responses=sum(item.effective_responses for item in items),
        effective_response_tokens=sum(item.effective_response_tokens for item in items),
        training_prompt_groups=config.effective_batch_size,
        training_responses=config.effective_batch_size * config.group_size,
        discarded_zero_var_groups=sum(item.discarded_zero_var_groups for item in items),
        effective_but_not_trained_groups=effective_groups - config.effective_batch_size,
    )


def _aggregate_metrics(
    *,
    diagnostics: HivePostRolloutDiagnostics,
    update,
    config: HiveAdaptiveTopupConfig,
    topup_estimates: list[HiveAdaptiveTopupEstimate],
    acquisitions: list[HiveTopupAcquisitionDiagnostics],
    initial_effective: int,
    topup_results: list[HivePostRolloutResult],
) -> dict[str, float]:
    generated = diagnostics.generated_prompt_groups
    topup_generated_groups = sum(item.candidate_actual for item in acquisitions)
    topup_generated_responses = sum(item.candidate_actual * config.group_size for item in acquisitions)
    latest = topup_estimates[-1] if topup_estimates else None
    metrics = {
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
        "hive/p_easy": update.p_easy_after,
        "hive/p_hard": update.p_hard_after,
        "hive/history_visits_committed": 0.0,
        "hive/required_prompt_groups": float(config.effective_batch_size),
        "hive/topup_rounds": float(len(acquisitions)),
        "hive/topup_triggered": float(bool(acquisitions)),
        "hive/topup_remaining_groups": float(latest.remaining_groups if latest else 0),
        "hive/topup_rho_zv": float(latest.rho_zv if latest else diagnostics.total_zero_var_groups / generated),
        "hive/estimated_zero_var_ratio": float(
            latest.rho_zv if latest else diagnostics.total_zero_var_groups / generated
        ),
        "hive/topup_survival_rate": float(
            latest.survival_rate
            if latest
            else 1.0 - diagnostics.total_zero_var_groups / generated
        ),
        "hive/topup_estimated_candidates": float(sum(item.estimated_candidates for item in topup_estimates)),
        "hive/topup_candidate_target": float(sum(item.candidate_target for item in topup_estimates)),
        "hive/topup_candidate_actual": float(topup_generated_groups),
        "hive/topup_candidate_overshoot": float(
            sum(item.candidate_actual - item.candidate_target for item in acquisitions)
        ),
        "hive/effective_groups_before_topup": float(initial_effective),
        "hive/effective_groups_after_topup": float(diagnostics.effective_prompt_groups),
        "hive/raw_prompts_seen_topup": float(sum(item.raw_prompts_seen for item in acquisitions)),
        "hive/generated_groups_topup": float(topup_generated_groups),
        "hive/generated_responses_topup": float(topup_generated_responses),
        "hive/generated_tokens_topup": float(
            sum(
                result.diagnostics.generated_prompt_tokens
                + result.diagnostics.generated_response_tokens
                for result in topup_results
            )
        ),
        "hive/topup_b_min": float(config.b_min),
        "hive/topup_b_min_binding": float(any(item.b_min_binding for item in topup_estimates)),
    }
    for index, (estimate, acquisition) in enumerate(zip(topup_estimates, acquisitions, strict=True), start=1):
        prefix = f"hive/topup_round_{index}"
        metrics.update(
            {
                f"{prefix}/rho_zv": estimate.rho_zv,
                f"{prefix}/candidate_target": float(estimate.candidate_target),
                f"{prefix}/candidate_actual": float(acquisition.candidate_actual),
                f"{prefix}/candidate_overshoot": float(
                    acquisition.candidate_actual - acquisition.candidate_target
                ),
            }
        )
    return metrics


def _validate_complete_groups(batch: DataProto, group_size: int) -> None:
    prompt_ids = np.asarray(batch.non_tensor_batch.get("prompt_id", []), dtype=object).tolist()
    if len(prompt_ids) != len(batch):
        raise ValueError("effective buffer is missing aligned stable prompt_ids")
    if len(prompt_ids) % group_size:
        raise ValueError("effective buffer response count must be divisible by group_size")
    grouped_ids = []
    for start in range(0, len(prompt_ids), group_size):
        block = prompt_ids[start : start + group_size]
        if len(set(block)) != 1:
            raise ValueError("effective buffer prompt groups must remain contiguous and complete")
        grouped_ids.append(block[0])
    if len(grouped_ids) != len(set(grouped_ids)):
        raise ValueError("effective buffer contains duplicate stable prompt groups")


def _concat_aligned_mappings(mappings: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not mappings:
        return {}
    keys = set(mappings[0])
    if any(set(mapping) != keys for mapping in mappings[1:]):
        raise ValueError("effective reward info keys differ across rollout rounds")
    output: dict[str, Any] = {}
    for key in keys:
        values: list[Any] = []
        for mapping in mappings:
            value = mapping[key]
            values.extend(value.tolist() if hasattr(value, "tolist") else list(value))
        output[key] = values
    return output


def _filter_aligned_mapping(
    mapping: Mapping[str, Any], indices: list[int], batch_size: int
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=np.int64)
    output = {}
    for key, value in mapping.items():
        normalized = value.tolist() if hasattr(value, "tolist") else value
        if isinstance(normalized, (list, tuple)) and len(normalized) == batch_size:
            output[key] = np.asarray(normalized, dtype=object)[selected].tolist()
        else:
            output[key] = value
    return output


def _finite_ratio(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return normalized


def _positive_finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return normalized


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
