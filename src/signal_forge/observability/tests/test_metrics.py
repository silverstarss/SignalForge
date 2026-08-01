from signal_forge.observability import (
    RolloutBudgetTracker,
    compute_group_metrics,
    compute_length_metrics,
    compute_reward_extra_metrics,
    compute_validation_alias_metrics,
)


def test_group_metrics_all_correct_all_wrong_mixed_and_incomplete():
    metrics = compute_group_metrics(
        uids=["a", "a", "b", "b", "c", "c", "d"],
        raw_correctness=[1, 1, 0, 0, 1, 0, 1],
        expected_group_size=2,
    )

    assert metrics["group/all_correct_count"] == 1.0
    assert metrics["group/all_wrong_count"] == 1.0
    assert metrics["group/mixed_count"] == 1.0
    assert metrics["group/incomplete_group_count"] == 1.0
    assert metrics["group/num_groups"] == 4.0
    assert round(metrics["group/raw_reward_variance_nonzero_ratio"], 6) == round(1 / 3, 6)


def test_group_metrics_missing_and_oversized_uid():
    metrics = compute_group_metrics(
        uids=["a", "a", "a", None, ""],
        raw_correctness=[1, 0, 1, 1, 0],
        expected_group_size=2,
    )

    assert metrics["group/oversized_group_count"] == 1.0
    assert metrics["group/missing_uid_count"] == 2.0


def test_reward_extra_metrics_keep_zero_component_schema():
    metrics = compute_reward_extra_metrics(
        {
            "score": [1.0, 0.0],
            "raw_correctness": [1.0, 0.0],
            "extraction_ok": [1, 0],
            "format_ok": [1, 1],
        }
    )

    assert metrics["reward/raw_correctness_mean"] == 0.5
    assert metrics["reward/extraction_failure_ratio"] == 0.5
    assert metrics["reward/format_reward_mean"] == 0.0
    assert metrics["reward/length_penalty_mean"] == 0.0


def test_length_metrics():
    metrics = compute_length_metrics(
        response_lengths=[10, 20, 30, 40],
        raw_correctness=[1, 0, 1, 0],
        max_response_length=40,
    )

    assert metrics["length/response_p50"] == 25.0
    assert metrics["length/truncated_ratio"] == 0.25
    assert metrics["length/correct_response_mean"] == 20.0
    assert metrics["length/incorrect_response_mean"] == 30.0


def test_budget_tracker_step_and_cumulative():
    tracker = RolloutBudgetTracker(start_time=0.0)
    metrics = tracker.update(
        candidate_prompt_groups_step=5,
        accepted_prompt_groups_step=5,
        responses_generated_step=40,
        prompt_tokens_generated_step=400,
        response_tokens_generated_step=1200,
        optimizer_steps_step=1,
        n_gpus=1,
    )

    assert metrics["budget/rejected_prompt_groups_step"] == 0.0
    assert metrics["budget/rollout_tokens_generated_step"] == 1600.0
    assert metrics["budget/responses_per_optimizer_step"] == 40.0


def test_validation_alias_metrics_by_source():
    metrics = compute_validation_alias_metrics(
        ["gsm8k", "math_level_3"],
        {"acc": [1.0, 0.0], "raw_correctness": [1.0, 0.0], "extraction_ok": [1, 1], "format_ok": [1, 0]},
    )

    assert metrics["val/pass_at_1"] == 0.5
    assert metrics["val/gsm8k/pass_at_1"] == 1.0
    assert metrics["val/math_l3/pass_at_1"] == 0.0
