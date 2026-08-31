from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from signal_forge.hive.post_rollout import (
    HiveComputeCounters,
    HiveInsufficientEffectiveGroupsError,
    HivePostRolloutConfig,
    HivePostRolloutInterpreter,
)
from signal_forge.hive.signal_metrics import (
    HiveGroupSignalCounts,
    HiveSignalCounters,
    HiveSignalStepCounts,
)
from signal_forge.hive.stage1 import Stage1StepSelector, compute_reward_history_signal
from signal_forge.hive.state import HiveSelectorState, ZeroVarianceType
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


G = 8


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _rollout_fixture(group_rewards: list[list[float]]):
    prompt_ids = [f"prompt:{index}" for index in range(len(group_rewards))]
    repeated_prompt_ids = [prompt_id for prompt_id in prompt_ids for _ in range(G)]
    repeated_uids = [f"uid:{index}" for index in range(len(group_rewards)) for _ in range(G)]
    rewards = [reward for group in group_rewards for reward in group]
    response_mask = torch.ones((len(rewards), 3), dtype=torch.long)
    reward_tensor = torch.zeros((len(rewards), 3), dtype=torch.float32)
    reward_tensor[:, -1] = torch.tensor(rewards, dtype=torch.float32)
    batch = DataProto(
        batch=TensorDict(
            {
                "responses": torch.ones((len(rewards), 3), dtype=torch.long),
                "response_mask": response_mask,
                "attention_mask": torch.ones((len(rewards), 5), dtype=torch.long),
            },
            batch_size=[len(rewards)],
        ),
        non_tensor_batch={
            "prompt_id": np.asarray(repeated_prompt_ids, dtype=object),
            "uid": np.asarray(repeated_uids, dtype=object),
            "extra_info": _object_array([{"prompt_id": value} for value in repeated_prompt_ids]),
        },
        meta_info={"global_token_num": [5] * len(rewards)},
    )
    reward_infos = {
        "reward": rewards,
        "extracted": [reward != 0.0 for reward in rewards],
        "correct": [reward == 1.0 for reward in rewards],
        "raw_correctness": [float(reward == 1.0) for reward in rewards],
    }
    return prompt_ids, batch, reward_tensor, reward_infos


def _interpret(group_rewards, *, effective_batch_size=2, state=None, step=1):
    prompt_ids, batch, reward_tensor, reward_infos = _rollout_fixture(group_rewards)
    state = state or HiveSelectorState.create(group_size=G, seed=7)
    selector = Stage1StepSelector(state.snapshot())
    result = HivePostRolloutInterpreter(
        selector_snapshot=selector.snapshot,
        config=HivePostRolloutConfig(
            effective_batch_size=effective_batch_size,
            group_size=G,
        ),
    ).interpret(
        batch=batch,
        reward_tensor=reward_tensor,
        reward_extra_infos=reward_infos,
        candidate_prompt_ids=prompt_ids,
        step=step,
    )
    return state, selector, result


def test_exact_classification_filtering_and_generated_accounting():
    groups = [[1.0] * G, [0.1] * G, [0.0] * G, [1.0, 0.1] * 4, [0.1, 0.0] * 4]
    _, _, result = _interpret(groups, effective_batch_size=2)

    assert [pending.visit.zero_variance_type for pending in result.pending_commit.visits] == [
        ZeroVarianceType.EASY,
        ZeroVarianceType.HARD,
        ZeroVarianceType.OTHER,
        None,
        None,
    ]
    assert result.diagnostics.generated_prompt_groups == 5
    assert result.diagnostics.generated_responses == 5 * G
    assert result.diagnostics.generated_response_tokens == 5 * G * 3
    assert result.diagnostics.effective_prompt_groups == 2
    assert result.diagnostics.discarded_zero_var_groups == 3
    assert len(result.training_batch) == 2 * G
    assert tuple(dict.fromkeys(result.training_batch.non_tensor_batch["prompt_id"])) == (
        "prompt:3",
        "prompt:4",
    )


def test_every_rollout_uses_stable_prompt_id_for_pending_history_not_uid():
    _, _, result = _interpret([[1.0, 0.1] * 4, [0.1, 0.0] * 4])

    assert [pending.prompt_id for pending in result.pending_commit.visits] == ["prompt:0", "prompt:1"]
    assert all(not pending.prompt_id.startswith("uid:") for pending in result.pending_commit.visits)


def test_pending_history_and_probabilities_publish_only_on_explicit_commit():
    state, selector, result = _interpret(
        [[1.0] * G, [0.1] * G, [1.0, 0.1] * 4, [0.1, 0.0] * 4],
        effective_batch_size=2,
    )
    frozen_snapshot = selector.snapshot

    assert state.prompt_history == {}
    assert state.p_easy == 0.5
    assert state.p_hard == 0.5
    assert compute_reward_history_signal(frozen_snapshot, "prompt:0", epsilon_p=0.01).unseen is True

    commit_metrics = result.pending_commit.commit(state, selector)

    assert len(state.prompt_history) == 4
    assert state.prompt_history["prompt:0"].visits[0].zero_variance_type is ZeroVarianceType.EASY
    assert compute_reward_history_signal(frozen_snapshot, "prompt:0", epsilon_p=0.01).unseen is True
    assert compute_reward_history_signal(state.snapshot(), "prompt:0", epsilon_p=0.01).unseen is False
    assert state.p_easy == pytest.approx(0.49)
    assert state.p_hard == pytest.approx(0.49)
    assert state.p_default == 0.5
    assert commit_metrics["hive/history_visits_committed"] == 4.0


def test_trainer_commit_advances_selector_and_compute_counter_steps_together():
    state, selector, result = _interpret(
        [[1.0] * G, [0.1] * G, [1.0, 0.1] * 4],
        effective_batch_size=1,
    )
    trainer = object.__new__(RayPPOTrainer)
    trainer.hive_selector_state = state
    trainer._hive_compute_counters = HiveComputeCounters()
    trainer._hive_compute_counters.update(result.diagnostics)
    trainer._hive_signal_counters = HiveSignalCounters()
    trainer._hive_signal_counters.update(
        HiveSignalStepCounts(
            candidate=HiveGroupSignalCounts(3, 1, 1, 0),
            training=HiveGroupSignalCounts(1, 1, 1, 0),
            generated_response_tokens=result.diagnostics.generated_response_tokens,
            topup_groups=0,
        )
    )
    trainer.global_steps = 1

    trainer._commit_hive_step(selector, result.pending_commit)

    assert trainer.hive_selector_state.global_step == 1
    assert trainer._hive_compute_counters.global_step == 1
    assert trainer._hive_signal_counters.global_step == 1


def test_effective_overshoot_trains_first_bt_but_keeps_all_visits_and_accounting():
    _, _, result = _interpret(
        [[1.0, 0.1] * 4, [0.1, 0.0] * 4, [1.0, 0.0] * 4],
        effective_batch_size=2,
    )

    assert result.diagnostics.effective_prompt_groups == 3
    assert result.diagnostics.training_prompt_groups == 2
    assert result.diagnostics.effective_but_not_trained_groups == 1
    assert tuple(dict.fromkeys(result.training_batch.non_tensor_batch["prompt_id"])) == (
        "prompt:0",
        "prompt:1",
    )
    assert len(result.pending_commit.visits) == 3


def test_insufficient_effective_groups_has_phase6_diagnostic_without_fallback():
    _, _, result = _interpret(
        [[1.0] * G, [1.0, 0.1] * 4, [0.1] * G],
        effective_batch_size=2,
    )

    assert result.training_batch is None
    with pytest.raises(HiveInsufficientEffectiveGroupsError, match="adaptive top-up is not implemented"):
        result.require_training_batch()


def test_incomplete_or_malformed_groups_are_rejected():
    prompt_ids, batch, reward_tensor, reward_infos = _rollout_fixture([[1.0, 0.1] * 4, [0.1, 0.0] * 4])
    batch = batch.select_idxs(list(range(len(batch) - 1)))
    reward_tensor = reward_tensor[:-1]
    reward_infos = {key: values[:-1] for key, values in reward_infos.items()}
    state = HiveSelectorState.create(group_size=G, seed=7)

    with pytest.raises(ValueError, match="exactly 8"):
        HivePostRolloutInterpreter(
            selector_snapshot=state.snapshot(),
            config=HivePostRolloutConfig(effective_batch_size=1),
        ).interpret(
            batch=batch,
            reward_tensor=reward_tensor,
            reward_extra_infos=reward_infos,
            candidate_prompt_ids=prompt_ids,
            step=1,
        )


def test_completed_step_checkpoint_preserves_history_and_controller(tmp_path):
    state, selector, result = _interpret(
        [[1.0] * G, [0.1] * G, [1.0, 0.1] * 4],
        effective_batch_size=1,
        step=4,
    )
    result.pending_commit.commit(state, selector)
    counters = HiveComputeCounters()
    counters.update(result.diagnostics)
    counters.mark_step_complete(4)
    state.save_checkpoint(tmp_path)
    counters.save_checkpoint(tmp_path)

    restored = HiveSelectorState.load_checkpoint(tmp_path)
    restored_counters = HiveComputeCounters.load_checkpoint(
        tmp_path, expected_global_step=restored.global_step
    )

    assert restored.to_dict() == state.to_dict()
    assert restored.global_step == 4
    assert sum(len(history.visits) for history in restored.prompt_history.values()) == 3
    assert restored_counters.global_step == restored.global_step == 4
    with pytest.raises(ValueError, match="does not match"):
        HiveComputeCounters.load_checkpoint(tmp_path, expected_global_step=5)


def test_cumulative_compute_counters_include_discarded_groups():
    counters = HiveComputeCounters()
    _, _, first = _interpret([[1.0] * G, [1.0, 0.1] * 4], effective_batch_size=1)
    _, _, second = _interpret([[0.1] * G, [0.1, 0.0] * 4], effective_batch_size=1)

    counters.update(first.diagnostics)
    metrics = counters.update(second.diagnostics)

    assert metrics["compute/generated_prompt_groups"] == 4.0
    assert metrics["compute/generated_responses"] == 4 * G
    assert metrics["compute/generated_response_tokens"] == 4 * G * 3
    assert metrics["compute/effective_prompt_groups"] == 2.0
    assert metrics["compute/effective_responses"] == 2 * G
    assert metrics["compute/effective_prompt_group_ratio"] == 0.5
    assert metrics["compute/effective_response_ratio"] == 0.5
    assert metrics["compute/effective_training_token_ratio"] == 0.5
    assert metrics["compute/effective_prompt_groups_per_1m_generated_response_tokens"] == pytest.approx(
        2 * 1_000_000 / (4 * G * 3)
    )


def test_cumulative_compute_counters_checkpoint_round_trip(tmp_path):
    counters = HiveComputeCounters()
    _, _, result = _interpret([[1.0] * G, [1.0, 0.1] * 4], effective_batch_size=1)
    counters.update(result.diagnostics)

    counters.save_checkpoint(tmp_path)
    restored = HiveComputeCounters.load_checkpoint(tmp_path)

    assert restored.to_dict() == counters.to_dict()


def test_hive_disabled_trainer_bypasses_post_rollout_interpreter():
    trainer = object.__new__(RayPPOTrainer)
    trainer.hive_selector_state = None

    result = trainer._interpret_hive_post_rollout(
        selector=None,
        candidate_prompt_ids=None,
        batch=None,
        reward_tensor=None,
        reward_extra_infos_dict=None,
    )

    assert result is None
