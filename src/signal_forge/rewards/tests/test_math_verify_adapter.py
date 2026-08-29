from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from signal_forge.rewards import math_verify_adapter
from signal_forge.rewards.process_timeout import IsolatedCallResult


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


def test_only_hard_timeout_falls_back_and_writes_diagnostic(monkeypatch, tmp_path):
    monkeypatch.setattr(
        math_verify_adapter,
        "call_with_hard_timeout",
        lambda *_, **__: IsolatedCallResult(ok=False, timed_out=True, elapsed_ms=120_001.5),
    )
    diagnostics_path = tmp_path / "verifier_timeouts.jsonl"

    result = math_verify_adapter.compute_score(
        data_source="gsm8k",
        solution_str=r"work without a final answer",
        ground_truth="2",
        extra_info={"prompt_id": "math:17"},
        verify_timeout_mode="process",
        verify_timeout_seconds=120,
        verify_timeout_fallback=True,
        verify_timeout_diagnostics_path=str(diagnostics_path),
    )

    assert (result["score"], result["extracted"], result["correct"]) == (0.0, False, False)
    assert result["failure_reason"] == "parse_timeout"
    assert result["parser_timeout"] is True
    assert result["parser_exception"] is False
    assert result["fallback_used"] is True

    records = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["event"] == "math_verify_timeout"
    assert records[0]["prompt_id"] == "math:17"
    assert records[0]["data_source"] == "gsm8k"
    assert records[0]["timeout_seconds"] == 120.0
    assert records[0]["elapsed_ms"] == 120_001.5
    assert records[0]["solution_str"] == r"work without a final answer"
    assert records[0]["ground_truth"] == "2"


def test_timeout_without_fallback_still_records_then_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        math_verify_adapter,
        "call_with_hard_timeout",
        lambda *_, **__: IsolatedCallResult(ok=False, timed_out=True, elapsed_ms=120_000.0),
    )
    diagnostics_path = tmp_path / "verifier_timeouts.jsonl"

    with pytest.raises(TimeoutError, match="timed out after 120s"):
        math_verify_adapter.compute_score(
            data_source="gsm8k",
            solution_str="response",
            ground_truth="2",
            verify_timeout_mode="process",
            verify_timeout_seconds=120,
            verify_timeout_fallback=False,
            verify_timeout_diagnostics_path=str(diagnostics_path),
        )

    assert len(diagnostics_path.read_text().splitlines()) == 1


def test_child_exception_remains_fail_fast_when_timeout_fallback_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(
        math_verify_adapter,
        "call_with_hard_timeout",
        lambda *_, **__: IsolatedCallResult(
            ok=False,
            timed_out=False,
            exception_type="ValueError",
            exception_message="bad latex",
            traceback_text="child traceback",
        ),
    )
    diagnostics_path = tmp_path / "verifier_timeouts.jsonl"

    with pytest.raises(RuntimeError, match="ValueError: bad latex"):
        math_verify_adapter.compute_score(
            data_source="gsm8k",
            solution_str="response",
            ground_truth="2",
            verify_timeout_mode="process",
            verify_timeout_seconds=120,
            verify_timeout_fallback=True,
            verify_timeout_diagnostics_path=str(diagnostics_path),
        )

    assert not diagnostics_path.exists()


def test_input_limit_remains_fail_fast_when_timeout_fallback_enabled(tmp_path):
    diagnostics_path = tmp_path / "verifier_timeouts.jsonl"

    with pytest.raises(ValueError, match="Verifier input too long"):
        math_verify_adapter.compute_score(
            data_source="gsm8k",
            solution_str="response",
            ground_truth="2",
            verify_timeout_mode="process",
            verify_timeout_seconds=120,
            verify_timeout_fallback=True,
            verify_timeout_diagnostics_path=str(diagnostics_path),
            verifier_max_input_chars=1,
        )

    assert not diagnostics_path.exists()
