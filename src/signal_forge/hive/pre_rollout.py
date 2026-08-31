"""HIVE pre-rollout selection over original prompt-level ``DataProto`` rows."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np
from tensordict import TensorDict

from signal_forge.hive.identity import PromptIdentityError, attach_stable_prompt_ids
from signal_forge.hive.prompt_entropy import PromptEntropyRecord
from signal_forge.hive.prompt_preprocessing import CanonicalHivePrompt, HivePromptPreprocessor
from signal_forge.hive.stage1 import Stage1BatchResult, Stage1StepSelector
from signal_forge.hive.stage2 import Stage2BatchResult, Stage2Selector
from signal_forge.hive.state import ZeroVarianceType
from verl import DataProto
from verl.utils import tensordict_utils as tu


@dataclass(frozen=True)
class HivePreRolloutConfig:
    effective_batch_size: int
    prompt_entropy_micro_batch_size: int = 1

    def __post_init__(self) -> None:
        _positive_integer("effective_batch_size", self.effective_batch_size)
        _positive_integer("prompt_entropy_micro_batch_size", self.prompt_entropy_micro_batch_size)
        if (3 * self.effective_batch_size) % 2:
            raise ValueError("faithful HIVE requires 3 * B_t to be divisible by 2")

    @property
    def candidate_target(self) -> int:
        return 3 * self.effective_batch_size // 2


@dataclass(frozen=True)
class HivePreRolloutPreparedRound:
    raw_batch: DataProto
    prompt_ids: tuple[str, ...]
    stage1: Stage1BatchResult
    accepted_indices: tuple[int, ...]
    canonical_prompts: tuple[CanonicalHivePrompt, ...]
    entropy_rpc_batch: TensorDict | None
    stage1_latency_seconds: float


@dataclass(frozen=True)
class HivePreRolloutRoundResult:
    selected_batch: DataProto
    stage1: Stage1BatchResult
    stage2: Stage2BatchResult[PromptEntropyRecord]
    entropy_records: tuple[PromptEntropyRecord, ...]
    entropy_latency_seconds: float
    entropy_peak_allocated_bytes: int
    entropy_peak_reserved_bytes: int
    stage1_latency_seconds: float

    @property
    def selected_prompt_ids(self) -> tuple[str, ...]:
        return tuple(record.prompt_id for record in self.stage2.kept)


@dataclass(frozen=True)
class HivePreRolloutStepResult:
    selected_batch: DataProto
    rounds: tuple[HivePreRolloutRoundResult, ...]
    candidate_target: int
    metrics: dict[str, float]


class HivePreRolloutStep:
    """Run all pre-rollout selection rounds against one frozen Stage-1 selector."""

    def __init__(
        self,
        *,
        stage1_selector: Stage1StepSelector,
        prompt_preprocessor: HivePromptPreprocessor,
        stage2_selector: Stage2Selector,
        config: HivePreRolloutConfig,
        candidate_target: int | None = None,
        excluded_prompt_ids: Sequence[str] = (),
    ) -> None:
        self.stage1_selector = stage1_selector
        self.prompt_preprocessor = prompt_preprocessor
        self.stage2_selector = stage2_selector
        self.config = config
        self._candidate_target = (
            config.candidate_target
            if candidate_target is None
            else _positive_integer("candidate_target", candidate_target)
        )
        self._rounds: list[HivePreRolloutRoundResult] = []
        self._selected_batches: list[DataProto] = []
        self._selected_prompt_ids: set[str] = set(excluded_prompt_ids)
        self._actual_count = 0

    @property
    def candidate_target(self) -> int:
        return self._candidate_target

    @property
    def candidate_actual(self) -> int:
        return self._actual_count

    @property
    def is_complete(self) -> bool:
        return self.candidate_actual >= self.candidate_target

    def prepare_round(self, raw_batch: DataProto) -> HivePreRolloutPreparedRound:
        if not isinstance(raw_batch, DataProto):
            raise TypeError("raw_batch must be a prompt-level DataProto")
        if "uid" in raw_batch.non_tensor_batch:
            raise ValueError("temporary rollout uid must be assigned after HIVE pre-rollout selection")

        prompt_ids = attach_stable_prompt_ids(raw_batch.non_tensor_batch)
        stage1_started_at = time.perf_counter()
        stage1 = self.stage1_selector.select(prompt_ids, historical_entropy_scores=None)
        stage1_latency_seconds = time.perf_counter() - stage1_started_at
        accepted_set = set(stage1.accepted_prompt_ids)
        accepted_indices = tuple(index for index, prompt_id in enumerate(prompt_ids) if prompt_id in accepted_set)

        raw_prompts = raw_batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None or len(raw_prompts) != len(prompt_ids):
            raise ValueError("raw prompt batch must contain one raw_prompt per stable prompt_id")
        accepted_prompt_ids = tuple(prompt_ids[index] for index in accepted_indices)
        accepted_raw_prompts = tuple(raw_prompts[index] for index in accepted_indices)
        canonical_prompts = self.prompt_preprocessor.preprocess_batch(
            accepted_prompt_ids,
            accepted_raw_prompts,
        )
        entropy_rpc_batch = self.prompt_preprocessor.build_entropy_rpc_batch(canonical_prompts)
        if entropy_rpc_batch is not None:
            tu.assign_non_tensor_data(
                entropy_rpc_batch,
                "prompt_entropy_micro_batch_size",
                self.config.prompt_entropy_micro_batch_size,
            )

        return HivePreRolloutPreparedRound(
            raw_batch=raw_batch,
            prompt_ids=prompt_ids,
            stage1=stage1,
            accepted_indices=accepted_indices,
            canonical_prompts=canonical_prompts,
            entropy_rpc_batch=entropy_rpc_batch,
            stage1_latency_seconds=stage1_latency_seconds,
        )

    def finish_round(
        self,
        prepared: HivePreRolloutPreparedRound,
        entropy_rpc_result: TensorDict | None,
    ) -> HivePreRolloutRoundResult:
        if prepared.entropy_rpc_batch is None:
            if entropy_rpc_result is not None:
                raise ValueError("entropy RPC output is invalid when Stage 1 accepted no prompts")
            records: tuple[PromptEntropyRecord, ...] = ()
            latency = 0.0
            peak_allocated = 0
            peak_reserved = 0
        else:
            if entropy_rpc_result is None:
                raise ValueError("entropy RPC output is required for Stage-1 survivors")
            records, latency, peak_allocated, peak_reserved = _parse_entropy_rpc_result(
                entropy_rpc_result,
                expected_prompt_ids=tuple(prompt.prompt_id for prompt in prepared.canonical_prompts),
            )

        stage2 = self.stage2_selector.select(records)
        kept_ids = tuple(record.prompt_id for record in stage2.kept)
        if len(kept_ids) != stage2.diagnostics.post_round_keep_count:
            raise RuntimeError("Stage-2 kept prompt count does not match post-round diagnostics")
        if len(set(kept_ids)) != len(kept_ids):
            raise PromptIdentityError("Stage-2 kept prompts contain duplicate stable prompt_id values")

        original_index = {prompt_id: index for index, prompt_id in enumerate(prepared.prompt_ids)}
        try:
            kept_indices = [original_index[prompt_id] for prompt_id in kept_ids]
        except KeyError as exc:
            raise PromptIdentityError("Stage-2 output cannot be mapped to the original raw prompt batch") from exc
        selected_batch = prepared.raw_batch.select_idxs(kept_indices)
        selected_ids = tuple(np.asarray(selected_batch.non_tensor_batch.get("prompt_id", []), dtype=object).tolist())
        if selected_ids != kept_ids:
            raise PromptIdentityError("selected original rows do not match Stage-2 stable prompt ordering")

        result = HivePreRolloutRoundResult(
            selected_batch=selected_batch,
            stage1=prepared.stage1,
            stage2=stage2,
            entropy_records=records,
            entropy_latency_seconds=latency,
            entropy_peak_allocated_bytes=peak_allocated,
            entropy_peak_reserved_bytes=peak_reserved,
            stage1_latency_seconds=prepared.stage1_latency_seconds,
        )
        self.append_round(result)
        return result

    def append_round(self, result: HivePreRolloutRoundResult) -> None:
        if self.is_complete:
            raise RuntimeError("candidate accumulator already reached its lower-bound target")
        duplicates = self._selected_prompt_ids.intersection(result.selected_prompt_ids)
        if duplicates:
            sample = sorted(duplicates)[:5]
            raise PromptIdentityError(f"candidate accumulator contains duplicate prompt_id values: {sample}")
        self._rounds.append(result)
        if result.selected_prompt_ids:
            self._selected_batches.append(result.selected_batch)
            self._selected_prompt_ids.update(result.selected_prompt_ids)
            self._actual_count += len(result.selected_prompt_ids)

    def finalize(self) -> HivePreRolloutStepResult:
        if not self.is_complete:
            raise RuntimeError(
                "raw prompt source exhausted before the HIVE candidate lower-bound target was reached: "
                f"actual={self.candidate_actual}, target={self.candidate_target}, rounds={len(self._rounds)}"
            )
        selected_batch = DataProto.concat(self._selected_batches)
        selected_ids = tuple(np.asarray(selected_batch.non_tensor_batch["prompt_id"], dtype=object).tolist())
        if len(selected_ids) != self.candidate_actual or len(set(selected_ids)) != len(selected_ids):
            raise PromptIdentityError("final HIVE candidate buffer has invalid stable prompt identities")
        metrics = _aggregate_metrics(
            self._rounds,
            effective_batch_size=self.config.effective_batch_size,
            candidate_target=self.candidate_target,
            candidate_actual=self.candidate_actual,
        )
        return HivePreRolloutStepResult(
            selected_batch=selected_batch,
            rounds=tuple(self._rounds),
            candidate_target=self.candidate_target,
            metrics=metrics,
        )


def _parse_entropy_rpc_result(
    result: TensorDict,
    *,
    expected_prompt_ids: Sequence[str],
) -> tuple[tuple[PromptEntropyRecord, ...], float, int, int]:
    if not isinstance(result, TensorDict):
        raise TypeError("actor prompt-entropy RPC must return a TensorDict")
    required = (
        "prompt_id",
        "entropy",
        "valid_token_count",
        "predictive_position_count",
        "entropy_eval_latency_seconds",
        "entropy_eval_peak_allocated_bytes",
        "entropy_eval_peak_reserved_bytes",
    )
    for key in required:
        if key not in result:
            raise ValueError(f"actor prompt-entropy RPC output is missing {key!r}")

    prompt_ids = tuple(tu.get(result, "prompt_id"))
    if len(prompt_ids) != len(set(prompt_ids)):
        raise PromptIdentityError("actor prompt-entropy RPC returned duplicate prompt_id values")
    if set(prompt_ids) != set(expected_prompt_ids) or len(prompt_ids) != len(expected_prompt_ids):
        raise PromptIdentityError("actor prompt-entropy RPC prompt_ids do not match Stage-1 survivors")

    entropy = tu.get(result, "entropy")
    valid_counts = tu.get(result, "valid_token_count")
    predictive_counts = tu.get(result, "predictive_position_count")
    if any(len(value) != len(prompt_ids) for value in (entropy, valid_counts, predictive_counts)):
        raise ValueError("actor prompt-entropy RPC output fields have inconsistent batch dimensions")
    records = tuple(
        PromptEntropyRecord(
            prompt_id=prompt_id,
            entropy=float(entropy[index].item()),
            valid_token_count=int(valid_counts[index].item()),
            predictive_position_count=int(predictive_counts[index].item()),
        )
        for index, prompt_id in enumerate(prompt_ids)
    )
    latency = _finite_nonnegative_max(result, "entropy_eval_latency_seconds", integer=False)
    peak_allocated = _finite_nonnegative_max(result, "entropy_eval_peak_allocated_bytes", integer=True)
    peak_reserved = _finite_nonnegative_max(result, "entropy_eval_peak_reserved_bytes", integer=True)
    return records, float(latency), int(peak_allocated), int(peak_reserved)


def _finite_nonnegative_max(result: TensorDict, key: str, *, integer: bool) -> int | float:
    values = tu.get(result, key)
    if len(values) == 0:
        return 0
    value = float(values.max().item())
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"actor prompt-entropy RPC diagnostic {key!r} must be finite and non-negative")
    return int(value) if integer else value


def aggregate_pre_rollout_selection_metrics(
    results: Sequence[HivePreRolloutStepResult],
) -> dict[str, float]:
    """Aggregate Section-18 selection metrics across initial and top-up acquisitions."""
    rounds = tuple(round_result for result in results for round_result in result.rounds)
    return _aggregate_selection_metrics(rounds)


def _aggregate_metrics(
    rounds: Sequence[HivePreRolloutRoundResult],
    *,
    effective_batch_size: int,
    candidate_target: int,
    candidate_actual: int,
) -> dict[str, float]:
    metrics = _aggregate_selection_metrics(rounds)
    metrics.update(
        {
            "hive/pre_rollout_candidate_count": float(candidate_actual),
            "hive/candidate_target": float(candidate_target),
            "hive/candidate_actual": float(candidate_actual),
            "hive/candidate_overshoot": float(candidate_actual - candidate_target),
            "hive/candidate_actual_ratio_to_Bt": float(candidate_actual / effective_batch_size),
            "hive/candidate_accumulation_rounds": float(len(rounds)),
        }
    )
    return metrics


def _aggregate_selection_metrics(
    rounds: Sequence[HivePreRolloutRoundResult],
) -> dict[str, float]:
    stage1_diagnostics = [round_result.stage1.diagnostics for round_result in rounds]
    stage1_decisions = [
        decision for round_result in rounds for decision in round_result.stage1.decisions
    ]
    stage2_diagnostics = [round_result.stage2.diagnostics for round_result in rounds]
    raw = sum(item.raw_prompts_seen for item in stage1_diagnostics)
    unseen = sum(item.unseen_prompts_seen for item in stage1_diagnostics)
    accepted = sum(item.accepted for item in stage1_diagnostics)
    rejected = sum(item.rejected for item in stage1_diagnostics)
    entropy_values = np.asarray(
        [record.entropy for round_result in rounds for record in round_result.entropy_records],
        dtype=np.float64,
    )
    selected_entropy = np.asarray(
        [record.entropy for round_result in rounds for record in round_result.stage2.kept],
        dtype=np.float64,
    )
    rejected_entropy = np.asarray(
        [
            record.entropy
            for round_result in rounds
            for record in (
                *round_result.stage2.upper_trimmed,
                *round_result.stage2.low_entropy_rejected,
                *round_result.stage2.rounding_dropped,
            )
        ],
        dtype=np.float64,
    )
    if entropy_values.size:
        entropy_mean = float(entropy_values.mean())
        entropy_std = float(entropy_values.std())
        entropy_min = float(entropy_values.min())
        entropy_max = float(entropy_values.max())
        q25, q50, q75 = (float(value) for value in np.quantile(entropy_values, [0.25, 0.5, 0.75]))
    else:
        entropy_mean = entropy_std = entropy_min = entropy_max = q25 = q50 = q75 = 0.0
    stage2_input = sum(item.input_count for item in stage2_diagnostics)
    stage2_kept = sum(item.post_round_keep_count for item in stage2_diagnostics)

    return {
        "hive/raw_prompts_seen": float(raw),
        "hive/seen_prompts_seen": float(raw - unseen),
        "hive/unseen_prompts_seen": float(unseen),
        "hive/stage1_accepted": float(accepted),
        "hive/stage1_rejected": float(rejected),
        "hive/stage1_easy_history_seen": float(
            sum(decision.zero_variance_type is ZeroVarianceType.EASY for decision in stage1_decisions)
        ),
        "hive/stage1_hard_history_seen": float(
            sum(decision.zero_variance_type is ZeroVarianceType.HARD for decision in stage1_decisions)
        ),
        "hive/stage1_other_history_seen": float(
            sum(decision.zero_variance_type is ZeroVarianceType.OTHER for decision in stage1_decisions)
        ),
        "hive/stage1_rejected_easy_history": float(
            sum(
                not decision.accepted and decision.zero_variance_type is ZeroVarianceType.EASY
                for decision in stage1_decisions
            )
        ),
        "hive/stage1_rejected_hard_history": float(
            sum(
                not decision.accepted and decision.zero_variance_type is ZeroVarianceType.HARD
                for decision in stage1_decisions
            )
        ),
        "hive/stage1_rejected_other_history": float(
            sum(
                not decision.accepted and decision.zero_variance_type is ZeroVarianceType.OTHER
                for decision in stage1_decisions
            )
        ),
        "hive/stage1_accept_ratio": float(accepted / raw) if raw else 0.0,
        "hive/stage2_input": float(stage2_input),
        "hive/stage2_upper_trimmed": float(
            sum(item.actual_upper_trim_count for item in stage2_diagnostics)
        ),
        "hive/stage2_pre_round_keep": float(
            sum(item.pre_round_keep_count for item in stage2_diagnostics)
        ),
        "hive/stage2_rounding_dropped": float(
            sum(item.rounding_dropped_count for item in stage2_diagnostics)
        ),
        "hive/stage2_kept": float(stage2_kept),
        "hive/stage2_keep_ratio": float(stage2_kept / stage2_input) if stage2_input else 0.0,
        "hive/stage2_low_entropy_rejected": float(
            sum(item.low_entropy_reject_count for item in stage2_diagnostics)
        ),
        "hive/prompt_entropy_mean": entropy_mean,
        "hive/prompt_entropy_std": entropy_std,
        "hive/prompt_entropy_min": entropy_min,
        "hive/prompt_entropy_max": entropy_max,
        "hive/prompt_entropy_q25": q25,
        "hive/prompt_entropy_q50": q50,
        "hive/prompt_entropy_q75": q75,
        "hive/selected_entropy_mean": (
            float(selected_entropy.mean()) if selected_entropy.size else 0.0
        ),
        "hive/rejected_entropy_mean": (
            float(rejected_entropy.mean()) if rejected_entropy.size else 0.0
        ),
        "hive/stage1_latency_seconds": float(
            sum(item.stage1_latency_seconds for item in rounds)
        ),
        "hive/stage2_entropy_latency_seconds": float(
            sum(item.entropy_latency_seconds for item in rounds)
        ),
        "hive/stage2_entropy_peak_allocated_bytes": float(
            max((item.entropy_peak_allocated_bytes for item in rounds), default=0)
        ),
        "hive/stage2_entropy_peak_reserved_bytes": float(
            max((item.entropy_peak_reserved_bytes for item in rounds), default=0)
        ),
    }


def _positive_integer(name: str, value: Real) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
