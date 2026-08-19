"""Pure Stage-1 selection and exploration control for HIVE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence

from signal_forge.hive.state import (
    HiveSelectorSnapshot,
    HiveSelectorState,
    ZeroVarianceType,
    restore_selector_rng,
)


@dataclass(frozen=True)
class Stage1Config:
    """Configuration for the Stage-1 combined probability."""

    lambda_weight: float = 1.0
    epsilon_p: float = 0.01

    def __post_init__(self) -> None:
        _validate_unit_interval("lambda_weight", self.lambda_weight)
        _validate_unit_interval("epsilon_p", self.epsilon_p)


@dataclass(frozen=True)
class ExplorationControllerConfig:
    """Paper-deployed adaptive easy/hard controller constants."""

    alpha_total: float = 0.25
    delta_p: float = 0.01
    p_min: float = 0.05
    p_max: float = 0.95

    def __post_init__(self) -> None:
        _validate_unit_interval("alpha_total", self.alpha_total)
        _validate_unit_interval("delta_p", self.delta_p)
        _validate_unit_interval("p_min", self.p_min)
        _validate_unit_interval("p_max", self.p_max)
        if self.p_min > self.p_max:
            raise ValueError("p_min must be less than or equal to p_max")

    @property
    def alpha_easy(self) -> float:
        return self.alpha_total / 3.0

    @property
    def alpha_hard(self) -> float:
        return 2.0 * self.alpha_total / 3.0


@dataclass(frozen=True)
class RewardHistorySignal:
    prompt_id: str
    unseen: bool
    trailing_zero_variance_streak: int
    zero_variance_type: ZeroVarianceType | None
    s_reward: float


@dataclass(frozen=True)
class Stage1PromptDecision:
    prompt_id: str
    unseen: bool
    trailing_zero_variance_streak: int
    zero_variance_type: ZeroVarianceType | None
    s_reward: float
    s_entropy: float | None
    selection_probability: float
    accepted: bool


@dataclass(frozen=True)
class Stage1Diagnostics:
    raw_prompts_seen: int
    unseen_prompts_seen: int
    easy_history_count: int
    hard_history_count: int
    other_history_count: int
    accepted: int
    rejected: int
    acceptance_ratio: float
    s_reward_values: tuple[float, ...]
    selection_probabilities: tuple[float, ...]
    p_easy: float
    p_hard: float
    p_default: float


@dataclass(frozen=True)
class Stage1BatchResult:
    decisions: tuple[Stage1PromptDecision, ...]
    diagnostics: Stage1Diagnostics

    @property
    def accepted_prompt_ids(self) -> tuple[str, ...]:
        return tuple(decision.prompt_id for decision in self.decisions if decision.accepted)

    @property
    def rejected_prompt_ids(self) -> tuple[str, ...]:
        return tuple(decision.prompt_id for decision in self.decisions if not decision.accepted)


@dataclass(frozen=True)
class ExplorationUpdate:
    source_global_step: int
    observed_easy_ratio: float
    observed_hard_ratio: float
    alpha_easy: float
    alpha_hard: float
    p_easy_before: float
    p_easy_after: float
    p_hard_before: float
    p_hard_after: float
    p_default: float


def _validate_unit_interval(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return normalized


def _validate_prompt_ids(prompt_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(prompt_ids)
    for index, prompt_id in enumerate(normalized):
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompt_id at index {index} must be a non-empty string")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Stage-1 selection round contains duplicate prompt_id values")
    return normalized


def _trailing_signal(snapshot: HiveSelectorSnapshot, prompt_id: str) -> tuple[bool, int, ZeroVarianceType | None]:
    visits = snapshot.prompt_history.get(prompt_id)
    if not visits:
        return True, 0, None

    streak = 0
    for visit in reversed(visits):
        if not visit.zero_variance:
            break
        streak += 1
    zero_variance_type = visits[-1].zero_variance_type if streak else None
    if streak and zero_variance_type is None:
        raise ValueError(f"prompt {prompt_id!r} has an invalid zero-variance history type")
    return False, streak, zero_variance_type


def compute_reward_history_signal(
    snapshot: HiveSelectorSnapshot,
    prompt_id: str,
    *,
    epsilon_p: float,
) -> RewardHistorySignal:
    """Compute max(p_tau**z, epsilon_p) from a frozen selector snapshot."""
    epsilon = _validate_unit_interval("epsilon_p", epsilon_p)
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError("prompt_id must be a non-empty string")

    unseen, streak, zero_variance_type = _trailing_signal(snapshot, prompt_id)
    if streak == 0:
        score = 1.0
    elif zero_variance_type is ZeroVarianceType.EASY:
        score = max(snapshot.p_easy**streak, epsilon)
    elif zero_variance_type is ZeroVarianceType.HARD:
        score = max(snapshot.p_hard**streak, epsilon)
    elif zero_variance_type is ZeroVarianceType.OTHER:
        score = max(snapshot.p_default**streak, epsilon)
    else:
        raise ValueError(f"prompt {prompt_id!r} has an unsupported zero-variance history type")

    return RewardHistorySignal(
        prompt_id=prompt_id,
        unseen=unseen,
        trailing_zero_variance_streak=streak,
        zero_variance_type=zero_variance_type,
        s_reward=score,
    )


class Stage1StepSelector:
    """Stateful Bernoulli selector bound to one frozen optimizer-step snapshot."""

    def __init__(self, snapshot: HiveSelectorSnapshot, config: Stage1Config | None = None):
        self.snapshot = snapshot
        self.config = config or Stage1Config()
        self._rng = restore_selector_rng(snapshot.selector_rng_state)
        self._finalized = False

    def select(
        self,
        prompt_ids: Sequence[str],
        *,
        historical_entropy_scores: Mapping[str, float] | None = None,
    ) -> Stage1BatchResult:
        """Select one raw-prompt round while retaining RNG position for later rounds."""
        if self._finalized:
            raise RuntimeError("Stage-1 selector has already committed its RNG state")
        normalized_prompt_ids = _validate_prompt_ids(prompt_ids)
        if self.config.lambda_weight < 1.0 and historical_entropy_scores is None:
            raise ValueError("historical entropy scores are required when lambda_weight < 1")

        decisions: list[Stage1PromptDecision] = []
        for prompt_id in normalized_prompt_ids:
            signal = compute_reward_history_signal(
                self.snapshot,
                prompt_id,
                epsilon_p=self.config.epsilon_p,
            )
            entropy_score: float | None = None
            if self.config.lambda_weight < 1.0:
                if prompt_id not in historical_entropy_scores:
                    raise ValueError(f"historical entropy score is missing for prompt {prompt_id!r}")
                entropy_score = _validate_unit_interval(
                    f"historical entropy score for {prompt_id!r}",
                    historical_entropy_scores[prompt_id],
                )

            selection_probability = self.config.lambda_weight * signal.s_reward
            if entropy_score is not None:
                selection_probability += (1.0 - self.config.lambda_weight) * entropy_score
            selection_probability = min(max(selection_probability, 0.0), 1.0)
            accepted = bool(self._rng.random() < selection_probability)
            decisions.append(
                Stage1PromptDecision(
                    prompt_id=prompt_id,
                    unseen=signal.unseen,
                    trailing_zero_variance_streak=signal.trailing_zero_variance_streak,
                    zero_variance_type=signal.zero_variance_type,
                    s_reward=signal.s_reward,
                    s_entropy=entropy_score,
                    selection_probability=selection_probability,
                    accepted=accepted,
                )
            )

        return Stage1BatchResult(
            decisions=tuple(decisions),
            diagnostics=_build_diagnostics(decisions, self.snapshot),
        )

    def commit_rng_state(self, state: HiveSelectorState) -> None:
        """Publish RNG progress after the optimizer step; history/probabilities remain caller-owned."""
        if self._finalized:
            raise RuntimeError("Stage-1 selector RNG state has already been committed")
        if state.group_size != self.snapshot.group_size:
            raise ValueError("selector state group_size changed after the step snapshot")
        if state.selector_rng_state != self.snapshot.selector_rng_state:
            raise RuntimeError("live selector RNG changed after the step snapshot")
        state.capture_rng(self._rng)
        self._finalized = True


def _build_diagnostics(
    decisions: Sequence[Stage1PromptDecision],
    snapshot: HiveSelectorSnapshot,
) -> Stage1Diagnostics:
    accepted = sum(decision.accepted for decision in decisions)
    raw_prompts_seen = len(decisions)
    return Stage1Diagnostics(
        raw_prompts_seen=raw_prompts_seen,
        unseen_prompts_seen=sum(decision.unseen for decision in decisions),
        easy_history_count=sum(decision.zero_variance_type is ZeroVarianceType.EASY for decision in decisions),
        hard_history_count=sum(decision.zero_variance_type is ZeroVarianceType.HARD for decision in decisions),
        other_history_count=sum(decision.zero_variance_type is ZeroVarianceType.OTHER for decision in decisions),
        accepted=accepted,
        rejected=raw_prompts_seen - accepted,
        acceptance_ratio=accepted / raw_prompts_seen if raw_prompts_seen else 0.0,
        s_reward_values=tuple(decision.s_reward for decision in decisions),
        selection_probabilities=tuple(decision.selection_probability for decision in decisions),
        p_easy=snapshot.p_easy,
        p_hard=snapshot.p_hard,
        p_default=snapshot.p_default,
    )


def _next_probability(current: float, observed: float, target: float, config: ExplorationControllerConfig) -> float:
    if observed < target:
        candidate = current + config.delta_p
    elif observed > target:
        candidate = current - config.delta_p
    else:
        candidate = current
    return min(max(candidate, config.p_min), config.p_max)


def compute_exploration_update(
    snapshot: HiveSelectorSnapshot,
    *,
    observed_easy_ratio: float,
    observed_hard_ratio: float,
    config: ExplorationControllerConfig | None = None,
) -> ExplorationUpdate:
    """Compute, but do not publish, the paper's easy/hard probability update."""
    controller_config = config or ExplorationControllerConfig()
    easy_ratio = _validate_unit_interval("observed_easy_ratio", observed_easy_ratio)
    hard_ratio = _validate_unit_interval("observed_hard_ratio", observed_hard_ratio)
    return ExplorationUpdate(
        source_global_step=snapshot.global_step,
        observed_easy_ratio=easy_ratio,
        observed_hard_ratio=hard_ratio,
        alpha_easy=controller_config.alpha_easy,
        alpha_hard=controller_config.alpha_hard,
        p_easy_before=snapshot.p_easy,
        p_easy_after=_next_probability(
            snapshot.p_easy,
            easy_ratio,
            controller_config.alpha_easy,
            controller_config,
        ),
        p_hard_before=snapshot.p_hard,
        p_hard_after=_next_probability(
            snapshot.p_hard,
            hard_ratio,
            controller_config.alpha_hard,
            controller_config,
        ),
        p_default=snapshot.p_default,
    )


def apply_exploration_update(state: HiveSelectorState, update: ExplorationUpdate) -> None:
    """Explicitly publish a computed controller update after an optimizer step."""
    if state.global_step != update.source_global_step:
        raise RuntimeError("selector global_step changed after exploration update was computed")
    if state.p_easy != update.p_easy_before or state.p_hard != update.p_hard_before:
        raise RuntimeError("adaptive exploration probabilities changed after the step snapshot")
    if state.p_default != update.p_default:
        raise RuntimeError("p_default changed after the step snapshot")
    state.p_easy = update.p_easy_after
    state.p_hard = update.p_hard_after
