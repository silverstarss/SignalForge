from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from signal_forge.hive.signal_metrics import (
    HiveGroupSignalCounts,
    HiveSignalCounters,
    HiveSignalStepCounts,
    compute_hive_group_signal_counts,
)


def _group(uid: str, rewards: list[float]):
    return [uid] * 8, rewards, [value == 1.0 for value in rewards]


def test_signal_classification_distinguishes_correctness_and_extraction_variance():
    mixed_uid, mixed_rewards, mixed_correct = _group("mixed", [1.0, 0.1] * 4)
    extraction_uid, extraction_rewards, extraction_correct = _group(
        "extraction", [0.1, 0.0] * 4
    )
    easy_uid, easy_rewards, easy_correct = _group("easy", [1.0] * 8)

    counts = compute_hive_group_signal_counts(
        uids=mixed_uid + extraction_uid + easy_uid,
        scalar_rewards=mixed_rewards + extraction_rewards + easy_rewards,
        raw_correctness=mixed_correct + extraction_correct + easy_correct,
        group_size=8,
    )

    assert counts == HiveGroupSignalCounts(
        group_count=3,
        optimization_effective=2,
        raw_correctness_mixed=1,
        extraction_only_effective=1,
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"uids": ["x"] * 7, "scalar_rewards": [1.0] * 7, "raw_correctness": [True] * 7}, "exactly 8"),
        ({"uids": ["x"] * 8, "scalar_rewards": [0.2] * 8, "raw_correctness": [False] * 8}, "frozen"),
        ({"uids": ["x"] * 8, "scalar_rewards": [0.1] * 8, "raw_correctness": [True] * 8}, "does not match"),
    ],
)
def test_signal_classification_rejects_malformed_groups(kwargs, message):
    with pytest.raises(ValueError, match=message):
        compute_hive_group_signal_counts(group_size=8, **kwargs)


def test_signal_counters_emit_step_cumulative_and_per_token_metrics(tmp_path):
    counters = HiveSignalCounters()
    step = HiveSignalStepCounts(
        candidate=HiveGroupSignalCounts(4, 3, 2, 1),
        training=HiveGroupSignalCounts(2, 2, 1, 1),
        generated_response_tokens=2_000_000,
        topup_groups=1,
    )

    metrics = counters.update(step)

    assert metrics["candidate/raw_correctness_mixed"] == 2.0
    assert metrics["candidate/extraction_only_effective"] == 1.0
    assert metrics["training/raw_correctness_mixed"] == 1.0
    assert metrics["training/extraction_only_effective"] == 1.0
    assert metrics["candidate/raw_correctness_mixed_per_1m_generated_response_tokens"] == 1.0
    assert metrics["training/raw_correctness_mixed_per_1m_generated_response_tokens"] == 0.5
    assert metrics["efficiency/scalar_zero_var_ratio"] == 0.25
    assert metrics["efficiency/raw_correctness_zero_var_ratio"] == 0.5
    with pytest.raises(RuntimeError, match="before step commit"):
        counters.save_checkpoint(tmp_path)

    counters.mark_step_complete(1)
    counters.save_checkpoint(tmp_path)
    restored = HiveSignalCounters.load_checkpoint(tmp_path, expected_global_step=1)
    assert restored == counters
    with pytest.raises(ValueError, match="does not match"):
        HiveSignalCounters.load_checkpoint(tmp_path, expected_global_step=2)


def test_step50_backfill_preserves_candidate_history_and_marks_training_boundary(tmp_path):
    script_path = Path(__file__).parents[3] / "scripts_formal" / "backfill_formal_b_signal_counters.py"
    spec = importlib.util.spec_from_file_location("backfill_signal", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    log_path = tmp_path / "train.log"
    lines = []
    for step in range(1, 3):
        lines.append(
            f"step:{step} - hive/generated_prompt_groups:4.0 - hive/effective_prompt_groups:3.0 "
            "- group/mixed_count:2.0 - hive/generated_groups_topup:1.0 "
            f"- compute/generated_prompt_groups:{step * 4}.0 "
            f"- compute/effective_prompt_groups:{step * 3}.0 "
            f"- compute/generated_response_tokens:{step * 100}.0\n"
        )
    log_path.write_text("".join(lines), encoding="utf-8")

    counters = module.build_step50_counters(log_path, expected_step=2)

    assert counters.global_step == 2
    assert counters.candidate_raw_correctness_mixed == 4
    assert counters.candidate_extraction_only_effective == 2
    assert counters.candidate_generated_response_tokens == 200
    assert counters.training_observation_start_step == 2
    assert counters.training_observed_updates == 0

