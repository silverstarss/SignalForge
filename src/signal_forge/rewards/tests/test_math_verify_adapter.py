from __future__ import annotations

from types import SimpleNamespace

import pytest

from signal_forge.rewards import math_verify_adapter


def _verification_result(*, extracted: bool, correct: bool):
    extraction = SimpleNamespace(
        extraction_ok=extracted,
        format_ok=extracted,
        extraction_status=SimpleNamespace(value="success" if extracted else "missing"),
        normalized_answer="2" if extracted else None,
        raw_answer="\\boxed{2}" if extracted else None,
    )
    return SimpleNamespace(extraction=extraction, is_correct=correct, error_type=None)


@pytest.mark.parametrize(
    "extracted,correct,expected_reward",
    [
        (True, True, 1.0),
        (True, False, 0.1),
        (False, False, 0.0),
    ],
)
def test_frozen_hive_reward_mapping(monkeypatch, extracted, correct, expected_reward):
    monkeypatch.setattr(
        math_verify_adapter,
        "_verify",
        lambda **_: _verification_result(extracted=extracted, correct=correct),
    )

    result = math_verify_adapter.compute_score(
        data_source="gsm8k",
        solution_str="response",
        ground_truth="2",
        verify_timeout_mode="inline",
    )

    assert result["score"] == expected_reward
    assert result["reward"] == expected_reward
    assert result["extracted"] is extracted
    assert result["correct"] is correct
    assert result["acc"] == float(correct)
    assert result["raw_correctness"] == float(correct)


def test_nonzero_timeout_fallback_reward_is_rejected():
    with pytest.raises(ValueError, match="fallback score must be 0.0"):
        math_verify_adapter.compute_score(
            data_source="gsm8k",
            solution_str="response",
            ground_truth="2",
            verify_timeout_fallback_score=0.1,
        )


def test_olympiad_multiple_answer_reward_uses_unordered_source_semantics():
    extra_info = {
        "source_dataset": "math",
        "is_multiple_answer": True,
        "multiple_answers_json": '["\\\\frac{1}{2}", "2"]',
    }
    correct = math_verify_adapter.compute_score(
        data_source="olympiadbench",
        solution_str=r"Therefore \\boxed{2, \\frac{2}{4}}.",
        ground_truth=r"\\boxed{\\frac{1}{2}, 2}",
        extra_info=extra_info,
        verify_timeout_mode="inline",
    )
    incorrect = math_verify_adapter.compute_score(
        data_source="olympiadbench",
        solution_str=r"Therefore \\boxed{\\frac{1}{2}, 3}.",
        ground_truth=r"\\boxed{\\frac{1}{2}, 2}",
        extra_info=extra_info,
        verify_timeout_mode="inline",
    )
    missing = math_verify_adapter.compute_score(
        data_source="olympiadbench",
        solution_str="No final boxed answer.",
        ground_truth=r"\\boxed{\\frac{1}{2}, 2}",
        extra_info=extra_info,
        verify_timeout_mode="inline",
    )

    assert (correct["score"], correct["extracted"], correct["correct"]) == (1.0, True, True)
    assert (incorrect["score"], incorrect["extracted"], incorrect["correct"]) == (0.1, True, False)
    assert (missing["score"], missing["extracted"], missing["correct"]) == (0.0, False, False)


def test_multiple_answer_metadata_is_required_when_flagged():
    with pytest.raises(ValueError, match="multiple_answers_json"):
        math_verify_adapter.compute_score(
            data_source="olympiadbench",
            solution_str=r"\\boxed{1}",
            ground_truth=r"\\boxed{1}",
            extra_info={"source_dataset": "math", "is_multiple_answer": True},
            verify_timeout_mode="inline",
        )
