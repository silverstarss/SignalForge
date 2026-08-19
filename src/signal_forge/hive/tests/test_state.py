from __future__ import annotations

import gzip
import json
import math

import pytest

from signal_forge.hive.state import (
    HIVE_STATE_FILENAME,
    HiveSelectorState,
    PromptHistory,
    PromptVisit,
    ZeroVarianceType,
    classify_zero_variance,
)


GROUP_SIZE = 8


@pytest.mark.parametrize(
    "rewards, expected_zero, expected_type",
    [
        ([1.0] * GROUP_SIZE, True, ZeroVarianceType.EASY),
        ([0.1] * GROUP_SIZE, True, ZeroVarianceType.HARD),
        ([0.0] * GROUP_SIZE, True, ZeroVarianceType.OTHER),
        ([0.5] * GROUP_SIZE, True, ZeroVarianceType.OTHER),
        ([1.0, 0.1] * 4, False, None),
    ],
)
def test_exact_zero_variance_classification(rewards, expected_zero, expected_type):
    classification = classify_zero_variance(rewards, group_size=GROUP_SIZE)

    assert classification.zero_variance is expected_zero
    assert classification.zero_variance_type is expected_type
    assert (classification.reward_variance == 0.0) is expected_zero


@pytest.mark.parametrize(
    "rewards, message",
    [
        ([1.0] * 7, "exactly 8"),
        ([1.0] * 8 + [0.1], "exactly 8"),
        ([1.0] * 7 + [math.nan], "finite"),
        ([1.0] * 7 + [True], "real number"),
    ],
)
def test_reward_group_validation(rewards, message):
    with pytest.raises(ValueError, match=message):
        classify_zero_variance(rewards, group_size=GROUP_SIZE)


def _visit(step: int, rewards: list[float], entropy: float | None = None) -> PromptVisit:
    return PromptVisit.from_rewards(
        step=step,
        rewards=rewards,
        group_size=GROUP_SIZE,
        response_entropy=entropy,
    )


@pytest.mark.parametrize(
    "visits, expected",
    [
        ([_visit(1, [1.0, 0.1] * 4)], 0),
        ([_visit(1, [1.0] * 8)], 1),
        ([_visit(1, [1.0] * 8), _visit(2, [0.1] * 8)], 2),
        ([_visit(1, [1.0] * 8), _visit(2, [0.1] * 8), _visit(3, [1.0, 0.1] * 4)], 0),
        (
            [
                _visit(1, [1.0] * 8),
                _visit(2, [0.1] * 8),
                _visit(3, [1.0, 0.1] * 4),
                _visit(4, [0.0] * 8),
            ],
            1,
        ),
    ],
)
def test_trailing_zero_variance_streak(visits, expected):
    history = PromptHistory(prompt_id="prompt:1", visits=list(visits))

    assert history.trailing_zero_variance_streak() == expected


def test_unseen_prompt_has_zero_trailing_streak():
    state = HiveSelectorState.create(group_size=GROUP_SIZE, seed=7)

    assert state.trailing_zero_variance_streak("unseen") == 0


def test_prompt_history_rejects_out_of_order_steps():
    history = PromptHistory(prompt_id="prompt:1")
    history.append(_visit(3, [1.0] * 8))

    with pytest.raises(ValueError, match="nondecreasing"):
        history.append(_visit(2, [0.1] * 8))


def test_selector_state_checkpoint_round_trip_preserves_full_state_and_rng(tmp_path):
    state = HiveSelectorState.create(
        group_size=GROUP_SIZE,
        seed=123,
        p_easy=0.4,
        p_hard=0.6,
        p_default=0.5,
        configuration={"enable": True, "group_size": GROUP_SIZE, "p_default": 0.5},
    )
    state.global_step = 11
    state.append_visit("prompt:1", _visit(2, [1.0] * 8, entropy=1.25))
    state.append_visit("prompt:1", _visit(7, [1.0, 0.1] * 4, entropy=None))
    state.append_visit("prompt:2", _visit(9, [0.0] * 8, entropy=0.75))

    rng = state.restore_rng()
    rng.random(13)
    state.capture_rng(rng)
    expected_next = state.restore_rng().random(5).tolist()

    checkpoint_path = state.save_checkpoint(tmp_path)
    restored = HiveSelectorState.load_checkpoint(tmp_path)

    assert checkpoint_path == tmp_path / HIVE_STATE_FILENAME
    assert restored.to_dict() == state.to_dict()
    assert restored.restore_rng().random(5).tolist() == expected_next
    assert restored.prompt_history["prompt:1"].visits[0].response_entropy == 1.25
    assert restored.prompt_history["prompt:1"].visits[1].zero_variance is False
    assert restored.prompt_history["prompt:2"].visits[0].zero_variance_type is ZeroVarianceType.OTHER


def test_selector_checkpoint_rejects_inconsistent_derived_visit_fields(tmp_path):
    state = HiveSelectorState.create(group_size=GROUP_SIZE, seed=123)
    state.append_visit("prompt:1", _visit(2, [1.0] * 8))
    state.save_checkpoint(tmp_path)

    checkpoint_path = tmp_path / HIVE_STATE_FILENAME
    with gzip.open(checkpoint_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["state"]["prompt_history"]["prompt:1"]["visits"][0]["zero_variance_type"] = "hard"
    with gzip.open(checkpoint_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(ValueError, match="zero_variance_type"):
        HiveSelectorState.load_checkpoint(tmp_path)
