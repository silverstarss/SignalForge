from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from signal_forge.hive.post_rollout import HivePostRolloutConfig, HivePostRolloutInterpreter
from signal_forge.hive.stage1 import Stage1StepSelector
from signal_forge.hive.state import HiveSelectorState
from signal_forge.hive.topup import (
    HiveAdaptiveTopupAccumulator,
    HiveAdaptiveTopupConfig,
    HiveTopupAcquisitionDiagnostics,
    HiveTopupRoundLimitError,
    HiveTopupSurvivalError,
    compute_adaptive_candidate_target,
    compute_adaptive_candidate_target_from_responses,
)
from verl import DataProto


G = 8
MIXED = [1.0, 0.1] * 4
ZERO = [0.0] * G


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _result(snapshot, *, start: int, group_rewards: list[list[float]], effective_batch_size: int):
    prompt_ids = [f"prompt:{index:04d}" for index in range(start, start + len(group_rewards))]
    repeated_ids = [prompt_id for prompt_id in prompt_ids for _ in range(G)]
    repeated_uids = [f"uid:{index:04d}" for index in range(start, start + len(group_rewards)) for _ in range(G)]
    rewards = [reward for group in group_rewards for reward in group]
    reward_tensor = torch.zeros((len(rewards), 2), dtype=torch.float32)
    reward_tensor[:, -1] = torch.tensor(rewards)
    batch = DataProto(
        batch=TensorDict(
            {
                "responses": torch.ones((len(rewards), 2), dtype=torch.long),
                "response_mask": torch.ones((len(rewards), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(rewards), 5), dtype=torch.long),
            },
            batch_size=[len(rewards)],
        ),
        non_tensor_batch={
            "prompt_id": np.asarray(repeated_ids, dtype=object),
            "uid": np.asarray(repeated_uids, dtype=object),
            "extra_info": _object_array([{"prompt_id": prompt_id} for prompt_id in repeated_ids]),
        },
        meta_info={"global_token_num": [5] * len(rewards)},
    )
    reward_infos = {
        "reward": rewards,
        "extracted": [reward != 0.0 for reward in rewards],
        "correct": [reward == 1.0 for reward in rewards],
        "raw_correctness": [float(reward == 1.0) for reward in rewards],
    }
    return HivePostRolloutInterpreter(
        selector_snapshot=snapshot,
        config=HivePostRolloutConfig(effective_batch_size=effective_batch_size),
    ).interpret(
        batch=batch,
        reward_tensor=reward_tensor,
        reward_extra_infos=reward_infos,
        candidate_prompt_ids=prompt_ids,
        step=1,
    )


def _config(**overrides):
    values = {
        "effective_batch_size": 4,
        "b_min": 1,
        "eta": 1.25,
        "max_topup_rounds": 4,
    }
    values.update(overrides)
    return HiveAdaptiveTopupConfig(**values)


def _acquisition(plan, actual):
    return HiveTopupAcquisitionDiagnostics(
        candidate_target=plan.candidate_target,
        candidate_actual=actual,
        raw_prompts_seen=actual * 2,
    )


def test_exact_paper_formula_hand_computed():
    config = _config(effective_batch_size=128, b_min=64)
    result = compute_adaptive_candidate_target(config, remaining_groups=64, rho_zv=0.5)

    assert result.estimated_candidates == 160
    assert result.candidate_target == 160
    assert result.remaining_responses == 64 * G


def test_response_and_group_formula_are_exactly_equivalent():
    config = _config(effective_batch_size=128, b_min=64)
    group_result = compute_adaptive_candidate_target(config, remaining_groups=37, rho_zv=0.375)
    response_result = compute_adaptive_candidate_target_from_responses(
        config,
        effective_response_count=(128 - 37) * G,
        rho_zv=0.375,
    )

    assert response_result == group_result
    with pytest.raises(ValueError, match="complete rollout groups"):
        compute_adaptive_candidate_target_from_responses(
            config,
            effective_response_count=17,
            rho_zv=0.0,
        )


def test_formula_zero_survival_guards_and_bounds():
    config = _config(effective_batch_size=8, b_min=1)
    assert compute_adaptive_candidate_target(config, remaining_groups=2, rho_zv=0.0).estimated_candidates == 3
    close = compute_adaptive_candidate_target(config, remaining_groups=2, rho_zv=0.999)
    assert close.candidate_target == config.candidate_cap
    assert close.candidate_cap_binding is True
    with pytest.raises(HiveTopupSurvivalError):
        compute_adaptive_candidate_target(config, remaining_groups=2, rho_zv=1.0)
    for invalid in (math.nan, math.inf, -0.1, 1.1):
        with pytest.raises(ValueError):
            compute_adaptive_candidate_target(config, remaining_groups=2, rho_zv=invalid)


def test_formula_zero_remaining_and_invalid_counts():
    config = _config(effective_batch_size=8, b_min=1)
    no_topup = compute_adaptive_candidate_target(config, remaining_groups=0, rho_zv=1.0)
    assert no_topup.candidate_target == 0
    assert no_topup.estimated_candidates == 0
    with pytest.raises(ValueError, match="non-negative"):
        compute_adaptive_candidate_target(config, remaining_groups=-1, rho_zv=0.0)


def test_b_min_cap_and_eta_binding():
    b_min = compute_adaptive_candidate_target(
        _config(effective_batch_size=8, b_min=12), remaining_groups=1, rho_zv=0.0
    )
    assert b_min.candidate_target == 12
    assert b_min.b_min_binding is True

    capped = compute_adaptive_candidate_target(
        _config(effective_batch_size=128, b_min=1), remaining_groups=128, rho_zv=0.9
    )
    assert capped.candidate_target == 192
    assert capped.candidate_cap_binding is True

    low_eta = compute_adaptive_candidate_target(
        _config(effective_batch_size=128, b_min=1, eta=1.0), remaining_groups=10, rho_zv=0.5
    )
    high_eta = compute_adaptive_candidate_target(
        _config(effective_batch_size=128, b_min=1, eta=1.5), remaining_groups=10, rho_zv=0.5
    )
    assert (low_eta.estimated_candidates, high_eta.estimated_candidates) == (20, 30)


def test_b_min_configuration_domain_accepts_equal_and_less_than_candidate_cap():
    equal = _config(effective_batch_size=8, b_min=12)
    less = _config(effective_batch_size=8, b_min=11)

    assert equal.b_min == equal.candidate_cap
    assert less.b_min < less.candidate_cap


def test_b_min_configuration_domain_rejects_above_candidate_cap():
    with pytest.raises(ValueError, match="b_min <= B_cand"):
        _config(effective_batch_size=8, b_min=13)


def test_no_topup_when_initial_round_already_fills_bt():
    state = HiveSelectorState.create(group_size=G, seed=4)
    accumulator = HiveAdaptiveTopupAccumulator(selector_snapshot=state.snapshot(), config=_config())
    accumulator.observe_initial(_result(state.snapshot(), start=0, group_rewards=[MIXED] * 4, effective_batch_size=4))

    assert accumulator.plan_next_topup() is None
    final = accumulator.finalize(step=1)
    assert final.metrics["hive/topup_triggered"] == 0.0
    assert final.diagnostics.training_prompt_groups == 4


def test_one_topup_round_fills_batch_and_keeps_partition_overshoot():
    state = HiveSelectorState.create(group_size=G, seed=4)
    snapshot = state.snapshot()
    accumulator = HiveAdaptiveTopupAccumulator(selector_snapshot=snapshot, config=_config())
    accumulator.observe_initial(_result(snapshot, start=0, group_rewards=[MIXED, ZERO], effective_batch_size=4))
    plan = accumulator.plan_next_topup()
    assert plan is not None
    actual = plan.candidate_target + 2
    topup_rewards = [MIXED] * 4 + [ZERO] * (actual - 4)
    accumulator.observe_topup(
        _result(snapshot, start=100, group_rewards=topup_rewards, effective_batch_size=4),
        _acquisition(plan, actual),
    )
    final = accumulator.finalize(step=1)

    assert final.metrics["hive/topup_rounds"] == 1.0
    assert final.metrics["hive/topup_candidate_overshoot"] == 2.0
    assert final.metrics["hive/topup_candidate_actual"] == float(actual)
    assert final.metrics["hive/topup_candidate_target"] == float(plan.candidate_target)
    assert final.metrics["hive/topup_b_min"] == 1.0
    assert final.metrics["hive/topup_b_min_binding"] == 0.0
    assert final.metrics["hive/generated_groups_topup"] == float(actual)
    assert final.metrics["hive/generated_responses_topup"] == float(actual * G)
    assert final.metrics["hive/generated_tokens_topup"] > 0.0
    assert final.diagnostics.generated_prompt_groups == 2 + actual
    assert len(final.pending_commit.visits) == 2 + actual
    assert final.diagnostics.effective_but_not_trained_groups == 1


def test_b_min_binding_is_exposed_in_step_metrics():
    state = HiveSelectorState.create(group_size=G, seed=4)
    snapshot = state.snapshot()
    accumulator = HiveAdaptiveTopupAccumulator(
        selector_snapshot=snapshot,
        config=_config(b_min=6),
    )
    accumulator.observe_initial(
        _result(snapshot, start=0, group_rewards=[MIXED, ZERO], effective_batch_size=4)
    )
    plan = accumulator.plan_next_topup()
    assert plan.candidate_target == 6
    assert plan.b_min_binding is True
    rewards = [MIXED] * 3 + [ZERO] * 3
    accumulator.observe_topup(
        _result(snapshot, start=100, group_rewards=rewards, effective_batch_size=4),
        _acquisition(plan, len(rewards)),
    )

    final = accumulator.finalize(step=1)

    assert final.metrics["hive/topup_b_min"] == 6.0
    assert final.metrics["hive/topup_b_min_binding"] == 1.0


def test_multiple_rounds_use_cumulative_rho_and_deterministic_arrival_order():
    state = HiveSelectorState.create(group_size=G, seed=4)
    snapshot = state.snapshot()
    accumulator = HiveAdaptiveTopupAccumulator(selector_snapshot=snapshot, config=_config())
    accumulator.observe_initial(_result(snapshot, start=0, group_rewards=[MIXED, ZERO], effective_batch_size=4))
    first_rho = accumulator.rho_zv
    first_plan = accumulator.plan_next_topup()
    first_rewards = [MIXED] + [ZERO] * (first_plan.candidate_target - 1)
    accumulator.observe_topup(
        _result(snapshot, start=100, group_rewards=first_rewards, effective_batch_size=4),
        _acquisition(first_plan, len(first_rewards)),
    )

    assert accumulator.rho_zv != first_rho
    second_plan = accumulator.plan_next_topup()
    second_rewards = [MIXED] * 3 + [ZERO] * (second_plan.candidate_target - 3)
    accumulator.observe_topup(
        _result(snapshot, start=200, group_rewards=second_rewards, effective_batch_size=4),
        _acquisition(second_plan, len(second_rewards)),
    )
    final = accumulator.finalize(step=1)
    trained_ids = tuple(dict.fromkeys(final.training_batch.non_tensor_batch["prompt_id"]))

    assert trained_ids == ("prompt:0000", "prompt:0100", "prompt:0200", "prompt:0201")
    assert len(final.training_batch) == 4 * G
    assert final.metrics["hive/topup_rounds"] == 2.0
    assert len(final.pending_commit.visits) == accumulator.generated_group_count
    assert final.metrics["hive/topup_round_1/rho_zv"] == first_rho
    assert final.metrics["hive/topup_round_2/rho_zv"] != first_rho
    assert state.prompt_history == {}


def test_max_topup_rounds_fails_explicitly():
    state = HiveSelectorState.create(group_size=G, seed=4)
    snapshot = state.snapshot()
    accumulator = HiveAdaptiveTopupAccumulator(
        selector_snapshot=snapshot,
        config=_config(max_topup_rounds=1),
    )
    accumulator.observe_initial(_result(snapshot, start=0, group_rewards=[MIXED, ZERO], effective_batch_size=4))
    plan = accumulator.plan_next_topup()
    rewards = [MIXED] + [ZERO] * (plan.candidate_target - 1)
    accumulator.observe_topup(
        _result(snapshot, start=100, group_rewards=rewards, effective_batch_size=4),
        _acquisition(plan, len(rewards)),
    )

    with pytest.raises(HiveTopupRoundLimitError, match="max_topup_rounds"):
        accumulator.plan_next_topup()


def test_snapshot_mismatch_and_duplicate_prompt_ids_are_rejected():
    state = HiveSelectorState.create(group_size=G, seed=4)
    snapshot = state.snapshot()
    accumulator = HiveAdaptiveTopupAccumulator(selector_snapshot=snapshot, config=_config())
    initial = _result(snapshot, start=0, group_rewards=[MIXED, ZERO], effective_batch_size=4)
    accumulator.observe_initial(initial)
    plan = accumulator.plan_next_topup()

    with pytest.raises(ValueError, match="duplicate stable prompt_ids"):
        duplicate = _result(
            snapshot,
            start=0,
            group_rewards=[MIXED] * plan.candidate_target,
            effective_batch_size=4,
        )
        accumulator.observe_topup(duplicate, _acquisition(plan, plan.candidate_target))
