from signal_forge.observability import (
    RolloutBudgetTracker,
    compute_group_metrics,
    compute_length_metrics,
    compute_reward_extra_metrics,
    compute_section18_timing_metrics,
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


def test_budget_tracker_exposes_shared_compute_metrics_and_checkpoint(tmp_path):
    tracker = RolloutBudgetTracker(start_time=10.0)
    metrics = tracker.update(
        candidate_prompt_groups_step=5,
        accepted_prompt_groups_step=4,
        responses_generated_step=40,
        prompt_tokens_generated_step=400,
        response_tokens_generated_step=1200,
        effective_prompt_groups_step=3,
        effective_responses_step=24,
        effective_response_tokens_step=600,
        rollout_time_seconds_step=7.5,
        optimizer_steps_step=1,
        n_gpus=1,
    )
    assert metrics["compute/generated_prompt_groups"] == 5.0
    assert metrics["compute/effective_prompt_groups"] == 3.0
    assert metrics["compute/effective_prompt_group_ratio"] == 0.6
    assert metrics["compute/effective_response_ratio"] == 0.6
    assert metrics["compute/effective_training_token_ratio"] == 0.5
    assert metrics["compute/effective_prompt_groups_per_1m_generated_response_tokens"] == 2500.0
    assert metrics["time/rollout_wall_clock_cumulative"] == 7.5

    tracker.save_checkpoint(tmp_path)
    restored = RolloutBudgetTracker.load_checkpoint(tmp_path, expected_optimizer_steps=1)
    assert restored.candidate_prompt_groups == 5
    assert restored.effective_prompt_groups == 3
    assert restored.response_tokens_generated == 1200
    assert restored.rollout_wall_time_seconds == 7.5


def test_budget_tracker_checkpoint_rejects_step_mismatch(tmp_path):
    tracker = RolloutBudgetTracker()
    tracker.save_checkpoint(tmp_path)
    import pytest
    with pytest.raises(ValueError, match="does not match"):
        RolloutBudgetTracker.load_checkpoint(tmp_path, expected_optimizer_steps=1)


def test_section18_timing_metrics_include_topup_rollout_and_reward():
    metrics = compute_section18_timing_metrics(
        timing_raw={
            "gen": 10.0,
            "reward": 2.0,
            "topup": 8.0,
            "topup/rollout_wall_seconds": 5.0,
            "topup/reward_wall_seconds": 1.0,
            "update_actor": 4.0,
            "testing": 6.0,
            "save_checkpoint": 3.0,
        },
        stage1_seconds=0.25,
        stage2_entropy_seconds=1.5,
        iteration_total_seconds=40.0,
    )
    assert metrics == {
        "time/stage1": 0.25,
        "time/stage2_entropy": 1.5,
        "time/rollout": 15.0,
        "time/reward": 3.0,
        "time/topup": 8.0,
        "time/grpo_update": 4.0,
        "time/validation": 6.0,
        "time/checkpoint": 3.0,
        "time/iteration_total": 40.0,
    }


def test_validation_alias_metrics_by_source():
    metrics = compute_validation_alias_metrics(
        ["gsm8k", "math_level_3"],
        {"acc": [1.0, 0.0], "raw_correctness": [1.0, 0.0], "extraction_ok": [1, 1], "format_ok": [1, 0]},
    )

    assert metrics["val/pass_at_1"] == 0.5
    assert metrics["val/gsm8k/pass_at_1"] == 1.0
    assert metrics["val/math_l3/pass_at_1"] == 0.0


def test_budget_tracker_baseline_and_hive_share_compute_schema():
    baseline = RolloutBudgetTracker()
    baseline_metrics = baseline.update(
        candidate_prompt_groups_step=4,
        accepted_prompt_groups_step=4,
        responses_generated_step=32,
        prompt_tokens_generated_step=320,
        response_tokens_generated_step=960,
        optimizer_steps_step=1,
        n_gpus=1,
    )
    hive = RolloutBudgetTracker()
    hive_metrics = hive.update(
        candidate_prompt_groups_step=6,
        accepted_prompt_groups_step=4,
        responses_generated_step=48,
        prompt_tokens_generated_step=480,
        response_tokens_generated_step=1440,
        effective_prompt_groups_step=5,
        effective_responses_step=40,
        effective_response_tokens_step=1100,
        optimizer_steps_step=1,
        n_gpus=1,
    )

    compute_keys = {key for key in baseline_metrics if key.startswith("compute/")}
    assert compute_keys == {key for key in hive_metrics if key.startswith("compute/")}
    assert baseline_metrics["compute/generated_prompt_groups"] == 4.0
    assert baseline_metrics["compute/effective_prompt_groups"] == 4.0
    assert baseline_metrics["compute/generated_responses"] == 32.0
    assert baseline_metrics["compute/effective_responses"] == 32.0
    assert hive_metrics["compute/generated_prompt_groups"] == 6.0
    assert hive_metrics["compute/effective_prompt_groups"] == 5.0
    assert hive_metrics["compute/generated_responses"] == 48.0
    assert hive_metrics["compute/effective_responses"] == 40.0


def test_validation_alias_metrics_exposes_unweighted_six_benchmark_mean():
    sources = [
        "math500",
        "math500",
        "aime24",
        "amc23",
        "minerva_math",
        "gaokao2023en",
        "olympiadbench",
    ]
    metrics = compute_validation_alias_metrics(
        sources,
        {
            "acc": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "raw_correctness": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )

    assert metrics["val/pass_at_1"] == 2 / 7
    assert metrics["val/six_benchmark_mean_accuracy"] == 1 / 6


def test_validation_alias_metrics_omits_formal_mean_for_incomplete_suite():
    metrics = compute_validation_alias_metrics(
        ["math500", "aime24"],
        {"acc": [1.0, 0.0], "raw_correctness": [1.0, 0.0]},
    )
    assert "val/six_benchmark_mean_accuracy" not in metrics
