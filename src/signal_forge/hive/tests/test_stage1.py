from __future__ import annotations

import copy

import pytest

from signal_forge.hive.stage1 import (
    ExplorationControllerConfig,
    Stage1Config,
    Stage1StepSelector,
    apply_exploration_update,
    compute_exploration_update,
    compute_reward_history_signal,
)
from signal_forge.hive.state import HiveSelectorState, PromptVisit, ZeroVarianceType


GROUP_SIZE = 8


def _state(
    *,
    seed: int = 42,
    p_easy: float = 0.5,
    p_hard: float = 0.5,
    p_default: float = 0.5,
) -> HiveSelectorState:
    return HiveSelectorState.create(
        group_size=GROUP_SIZE,
        seed=seed,
        p_easy=p_easy,
        p_hard=p_hard,
        p_default=p_default,
    )


def _append_visits(
    state: HiveSelectorState,
    prompt_id: str,
    rewards: list[float],
    count: int,
    *,
    start_step: int = 1,
) -> None:
    for step in range(start_step, start_step + count):
        state.append_visit(
            prompt_id,
            PromptVisit.from_rewards(
                step=step,
                rewards=rewards,
                group_size=GROUP_SIZE,
            ),
        )


def test_unseen_prompt_has_probability_one_and_is_accepted():
    selector = Stage1StepSelector(_state(seed=7).snapshot(), Stage1Config(lambda_weight=1.0))

    result = selector.select(["unseen:1"])
    decision = result.decisions[0]

    assert decision.unseen is True
    assert decision.trailing_zero_variance_streak == 0
    assert decision.s_reward == 1.0
    assert decision.s_entropy is None
    assert decision.selection_probability == 1.0
    assert decision.accepted is True


@pytest.mark.parametrize(
    "rewards, probability_name",
    [
        ([1.0] * GROUP_SIZE, "p_easy"),
        ([0.1] * GROUP_SIZE, "p_hard"),
        ([0.0] * GROUP_SIZE, "p_default"),
    ],
)
@pytest.mark.parametrize("streak, expected", [(1, 0.5), (2, 0.25), (3, 0.125)])
def test_reward_history_score_decays_by_current_zero_variance_type(
    rewards, probability_name, streak, expected
):
    state = _state()
    _append_visits(state, f"prompt:{probability_name}", rewards, streak)

    signal = compute_reward_history_signal(
        state.snapshot(),
        f"prompt:{probability_name}",
        epsilon_p=0.01,
    )

    assert signal.trailing_zero_variance_streak == streak
    assert signal.s_reward == expected
    assert signal.zero_variance_type is {
        "p_easy": ZeroVarianceType.EASY,
        "p_hard": ZeroVarianceType.HARD,
        "p_default": ZeroVarianceType.OTHER,
    }[probability_name]


def test_reward_history_score_respects_epsilon_floor():
    state = _state(p_easy=0.5)
    _append_visits(state, "easy:floor", [1.0] * GROUP_SIZE, 12)

    signal = compute_reward_history_signal(state.snapshot(), "easy:floor", epsilon_p=0.01)

    assert signal.s_reward == 0.01


def test_lambda_one_ignores_missing_historical_entropy():
    state = _state()
    _append_visits(state, "easy:1", [1.0] * GROUP_SIZE, 1)
    selector = Stage1StepSelector(state.snapshot(), Stage1Config(lambda_weight=1.0))

    result = selector.select(["easy:1"], historical_entropy_scores=None)

    assert result.decisions[0].s_entropy is None
    assert result.decisions[0].selection_probability == 0.5


def test_general_lambda_combines_reward_and_historical_entropy_scores():
    state = _state()
    _append_visits(state, "easy:1", [1.0] * GROUP_SIZE, 1)
    selector = Stage1StepSelector(
        state.snapshot(),
        Stage1Config(lambda_weight=0.25),
    )

    decision = selector.select(["easy:1"], historical_entropy_scores={"easy:1": 0.2}).decisions[0]

    assert decision.s_entropy == 0.2
    assert decision.selection_probability == pytest.approx(0.25 * 0.5 + 0.75 * 0.2)


def test_entropy_weight_requires_real_normalized_historical_entropy():
    selector = Stage1StepSelector(_state().snapshot(), Stage1Config(lambda_weight=0.5))

    with pytest.raises(ValueError, match="historical entropy"):
        selector.select(["prompt:1"], historical_entropy_scores=None)


def test_fixed_rng_state_produces_identical_bernoulli_sequence():
    first_state = _state(seed=123)
    second_state = _state(seed=123)
    prompt_ids = [f"easy:{index}" for index in range(24)]
    for state in (first_state, second_state):
        for prompt_id in prompt_ids:
            _append_visits(state, prompt_id, [1.0] * GROUP_SIZE, 1)

    first = Stage1StepSelector(first_state.snapshot()).select(prompt_ids)
    second = Stage1StepSelector(second_state.snapshot()).select(prompt_ids)

    assert [decision.accepted for decision in first.decisions] == [
        decision.accepted for decision in second.decisions
    ]


def test_checkpoint_resume_reproduces_subsequent_bernoulli_decisions(tmp_path):
    continuous_state = _state(seed=314)
    split_state = _state(seed=314)
    first_prompt_ids = [f"easy:first:{index}" for index in range(10)]
    second_prompt_ids = [f"easy:second:{index}" for index in range(16)]
    for state in (continuous_state, split_state):
        for prompt_id in [*first_prompt_ids, *second_prompt_ids]:
            _append_visits(state, prompt_id, [1.0] * GROUP_SIZE, 1)

    continuous = Stage1StepSelector(continuous_state.snapshot())
    continuous.select(first_prompt_ids)
    expected = continuous.select(second_prompt_ids)

    before_checkpoint = Stage1StepSelector(split_state.snapshot())
    before_checkpoint.select(first_prompt_ids)
    before_checkpoint.commit_rng_state(split_state)
    split_state.save_checkpoint(tmp_path)
    restored_state = HiveSelectorState.load_checkpoint(tmp_path)
    resumed = Stage1StepSelector(restored_state.snapshot()).select(second_prompt_ids)

    assert [decision.accepted for decision in resumed.decisions] == [
        decision.accepted for decision in expected.decisions
    ]


@pytest.mark.parametrize(
    "kind, observed_offset, expected_delta",
    [
        ("easy", -0.01, 0.01),
        ("easy", 0.01, -0.01),
        ("hard", -0.01, 0.01),
        ("hard", 0.01, -0.01),
    ],
)
def test_exploration_probability_moves_toward_target_ratio(kind, observed_offset, expected_delta):
    state = _state()
    config = ExplorationControllerConfig()
    observed_easy = config.alpha_easy
    observed_hard = config.alpha_hard
    if kind == "easy":
        observed_easy += observed_offset
    else:
        observed_hard += observed_offset

    update = compute_exploration_update(
        state.snapshot(),
        observed_easy_ratio=observed_easy,
        observed_hard_ratio=observed_hard,
        config=config,
    )

    expected_easy = 0.5 + expected_delta if kind == "easy" else 0.5
    expected_hard = 0.5 + expected_delta if kind == "hard" else 0.5
    assert update.p_easy_after == pytest.approx(expected_easy)
    assert update.p_hard_after == pytest.approx(expected_hard)


def test_exploration_probability_is_unchanged_at_exact_targets():
    state = _state()
    config = ExplorationControllerConfig()

    update = compute_exploration_update(
        state.snapshot(),
        observed_easy_ratio=config.alpha_easy,
        observed_hard_ratio=config.alpha_hard,
        config=config,
    )

    assert update.p_easy_after == 0.5
    assert update.p_hard_after == 0.5


def test_exploration_probability_is_clipped_at_bounds():
    state = _state(p_easy=0.05, p_hard=0.95)
    config = ExplorationControllerConfig()

    update = compute_exploration_update(
        state.snapshot(),
        observed_easy_ratio=config.alpha_easy + 0.01,
        observed_hard_ratio=config.alpha_hard - 0.01,
        config=config,
    )

    assert update.p_easy_after == 0.05
    assert update.p_hard_after == 0.95


def test_p_default_never_adapts():
    state = _state(p_default=0.37)
    config = ExplorationControllerConfig()
    update = compute_exploration_update(
        state.snapshot(),
        observed_easy_ratio=1.0,
        observed_hard_ratio=1.0,
        config=config,
    )

    apply_exploration_update(state, update)

    assert state.p_easy == 0.49
    assert state.p_hard == 0.49
    assert state.p_default == 0.37


def test_snapshot_does_not_observe_live_state_mutations_until_refreshed():
    state = _state(p_easy=0.5)
    _append_visits(state, "easy:1", [1.0] * GROUP_SIZE, 1)
    frozen_snapshot = state.snapshot()

    state.p_easy = 0.9
    _append_visits(state, "easy:1", [1.0] * GROUP_SIZE, 1, start_step=2)

    frozen_signal = compute_reward_history_signal(frozen_snapshot, "easy:1", epsilon_p=0.01)
    refreshed_signal = compute_reward_history_signal(state.snapshot(), "easy:1", epsilon_p=0.01)

    assert frozen_signal.trailing_zero_variance_streak == 1
    assert frozen_signal.s_reward == 0.5
    assert refreshed_signal.trailing_zero_variance_streak == 2
    assert refreshed_signal.s_reward == pytest.approx(0.81)


def test_stage1_diagnostics_are_sufficient_for_later_metric_logging():
    state = _state(seed=99)
    _append_visits(state, "easy:1", [1.0] * GROUP_SIZE, 1)
    _append_visits(state, "hard:1", [0.1] * GROUP_SIZE, 1)
    _append_visits(state, "other:1", [0.0] * GROUP_SIZE, 1)
    _append_visits(state, "mixed:1", [1.0, 0.1] * 4, 1)
    original_rng_state = copy.deepcopy(state.selector_rng_state)

    result = Stage1StepSelector(state.snapshot()).select(
        ["easy:1", "hard:1", "other:1", "mixed:1", "unseen:1"]
    )
    diagnostics = result.diagnostics

    assert diagnostics.raw_prompts_seen == 5
    assert diagnostics.unseen_prompts_seen == 1
    assert diagnostics.easy_history_count == 1
    assert diagnostics.hard_history_count == 1
    assert diagnostics.other_history_count == 1
    assert diagnostics.accepted + diagnostics.rejected == 5
    assert diagnostics.acceptance_ratio == diagnostics.accepted / 5
