"""Thin veRL reward adapter over the frozen RewardScope Math-Verify verifier.

The adapter intentionally owns no grading logic. It only maps veRL's reward
function signature to RewardScope's frozen training verifier and returns the
flat dict format consumed by this repository's NaiveRewardManager.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from functools import lru_cache
from typing import Any

from signal_forge.rewards.process_timeout import call_with_hard_timeout


_GSM8K_SOURCES = frozenset({"gsm8k", "openai/gsm8k"})
_MATH_LEVEL_3_SOURCES = frozenset({"math_level_3", "math", "hendrycks_math", "competition_math"})


@lru_cache(maxsize=1)
def _numeric_verifier():
    from rewardscope.verification.math_verify import MathVerifyNumericVerifier

    return MathVerifyNumericVerifier(mode="training")


@lru_cache(maxsize=1)
def _latex_verifier():
    from rewardscope.verification.math_verify import MathVerifyLatexVerifier

    return MathVerifyLatexVerifier(mode="training")


def _canonical_source(data_source: str, extra_info: dict[str, Any] | None) -> str:
    source = str(data_source)
    if source in _GSM8K_SOURCES:
        return "gsm8k"
    if source in _MATH_LEVEL_3_SOURCES:
        return "math_level_3"

    source_dataset = ""
    if extra_info:
        source_dataset = str(extra_info.get("source_dataset") or extra_info.get("dataset_name") or "")
    if source_dataset == "gsm8k":
        return "gsm8k"
    if source_dataset == "math":
        return "math_level_3"

    raise ValueError(f"Unsupported data_source for Math-Verify reward: {data_source!r}")


def _verify(source: str, solution_str: str, ground_truth: str):
    if source == "gsm8k":
        return _numeric_verifier().verify(response=solution_str, ground_truth=ground_truth)
    if source == "math_level_3":
        return _latex_verifier().verify(response=solution_str, ground_truth=ground_truth)
    raise AssertionError(f"Unexpected canonical source: {source}")


def _result_payload(
    source: str,
    solution_str: str,
    ground_truth: str,
    expected_multiple_answers: tuple[str, ...] = (),
) -> dict[str, Any]:
    result = _verify(source=source, solution_str=solution_str, ground_truth=ground_truth)
    extraction = result.extraction
    extracted = bool(extraction.extraction_ok)
    correct = bool(result.is_correct)
    if expected_multiple_answers:
        from signal_forge.data.validation_decontamination import olympiad_multiple_answers_equal

        correct = olympiad_multiple_answers_equal(
            solution_str,
            expected_multiple_answers,
            verifier=_latex_answer_equivalent,
        )
    reward = 1.0 if correct else 0.1 if extracted else 0.0
    raw_correctness = float(correct)
    return {
        "score": reward,
        "reward": reward,
        "acc": raw_correctness,
        "raw_correctness": raw_correctness,
        "extracted": extracted,
        "correct": correct,
        "extraction_ok": extracted,
        "format_ok": bool(extraction.format_ok),
        "verification_status": extraction.extraction_status.value,
        "verification_error_type": result.error_type or "",
        "predicted_answer": extraction.normalized_answer or "",
        "raw_answer": extraction.raw_answer or "",
        "reward_source": source,
    }


def _latex_answer_equivalent(prediction: str, gold: str) -> bool:
    result = _latex_verifier().verify(
        response=f"\\boxed{{{prediction}}}",
        ground_truth=f"\\boxed{{{gold}}}",
    )
    return bool(result.is_correct)


def _expected_multiple_answers(extra_info: dict[str, Any] | None) -> tuple[str, ...]:
    if not extra_info or not bool(extra_info.get("is_multiple_answer", False)):
        return ()
    values = extra_info.get("multiple_answers")
    if values is None:
        encoded = extra_info.get("multiple_answers_json")
        if not isinstance(encoded, str):
            raise ValueError("multiple-answer validation row is missing multiple_answers_json")
        values = json.loads(encoded)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("multiple_answers must be a non-empty list")
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError("multiple_answers contains an empty value")
    return normalized


def _verify_serializable(
    source: str,
    solution_str: str,
    ground_truth: str,
    expected_multiple_answers: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Child-process entrypoint. RewardScope itself keeps parsing_timeout=None."""
    return _result_payload(
        source=source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        expected_multiple_answers=tuple(expected_multiple_answers),
    )


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _fallback_payload(*, source: str, reason: str, fallback_score: float, detail: str = "") -> dict[str, Any]:
    score = float(fallback_score)
    if score != 0.0:
        raise ValueError("frozen HIVE fallback score must be 0.0")
    return {
        "score": score,
        "reward": score,
        "acc": 0.0,
        "raw_correctness": 0.0,
        "extracted": False,
        "correct": False,
        "extraction_ok": False,
        "format_ok": False,
        "verification_status": reason,
        "verification_error_type": reason,
        "predicted_answer": "",
        "raw_answer": "",
        "reward_source": source,
        "failure_reason": reason,
        "verifier_error_detail": detail,
        "fallback_used": True,
    }


def _write_timeout_diagnostic(
    path: str | None,
    *,
    source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None,
    timeout_seconds: float,
    elapsed_ms: float,
) -> None:
    if not path:
        return
    destination = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    payload = {
        "event": "math_verify_timeout",
        "recorded_at_unix": time.time(),
        "prompt_id": str((extra_info or {}).get("prompt_id", "")),
        "data_source": source,
        "timeout_seconds": float(timeout_seconds),
        "elapsed_ms": float(elapsed_ms),
        "solution_chars": len(solution_str),
        "solution_sha256": hashlib.sha256(solution_str.encode("utf-8")).hexdigest(),
        "solution_str": solution_str,
        "ground_truth": ground_truth,
    }
    encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded)
    finally:
        os.close(fd)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    verify_timeout_mode: str | None = None,
    verify_timeout_seconds: float | None = None,
    verify_timeout_fallback: bool | None = None,
    verify_timeout_fallback_score: float = 0.0,
    verify_timeout_diagnostics_path: str | None = None,
    verifier_max_input_chars: int | None = None,
) -> dict[str, Any]:
    """Return the frozen three-outcome Math-Verify reward and diagnostics for veRL.

    Parser and verifier exceptions intentionally propagate. A verifier/library
    failure is a training correctness bug, not an ordinary wrong answer unless
    the explicit process-timeout fallback path is enabled by config.
    """
    if float(verify_timeout_fallback_score) != 0.0:
        raise ValueError("frozen HIVE fallback score must be 0.0")
    started = time.perf_counter()
    source = _canonical_source(data_source, extra_info)
    expected_multiple_answers = _expected_multiple_answers(extra_info)
    solution_text = str(solution_str)
    ground_truth_text = str(ground_truth)
    input_chars = len(solution_text) + len(ground_truth_text)
    timeout_mode = (verify_timeout_mode or os.environ.get("SIGNAL_FORGE_VERIFY_TIMEOUT_MODE") or "inline").lower()
    timeout_seconds = (
        verify_timeout_seconds
        if verify_timeout_seconds is not None
        else _env_float("SIGNAL_FORGE_VERIFY_TIMEOUT_SECONDS")
    )
    fallback_enabled = (
        bool(verify_timeout_fallback)
        if verify_timeout_fallback is not None
        else _env_bool("SIGNAL_FORGE_VERIFY_TIMEOUT_FALLBACK", False)
    )

    if verifier_max_input_chars is not None and input_chars > int(verifier_max_input_chars):
        raise ValueError(f"Verifier input too long: {input_chars} > {verifier_max_input_chars} chars")
    elif timeout_mode == "process":
        if timeout_seconds is None or timeout_seconds <= 0:
            raise ValueError("verify_timeout_mode='process' requires verify_timeout_seconds > 0")
        isolated = call_with_hard_timeout(
            "signal_forge.rewards.math_verify_adapter:_verify_serializable",
            {
                "source": source,
                "solution_str": solution_text,
                "ground_truth": ground_truth_text,
                "expected_multiple_answers": expected_multiple_answers,
            },
            timeout_seconds=float(timeout_seconds),
        )
        if isolated.ok:
            payload = dict(isolated.value)
        elif isolated.timed_out:
            _write_timeout_diagnostic(
                verify_timeout_diagnostics_path,
                source=source,
                solution_str=solution_text,
                ground_truth=ground_truth_text,
                extra_info=extra_info,
                timeout_seconds=float(timeout_seconds),
                elapsed_ms=isolated.elapsed_ms,
            )
            if not fallback_enabled:
                raise TimeoutError(f"Math-Verify process timed out after {timeout_seconds}s")
            payload = _fallback_payload(
                source=source,
                reason="parse_timeout",
                fallback_score=verify_timeout_fallback_score,
                detail=f"deadline_seconds={timeout_seconds}",
            )
        else:
            raise RuntimeError(
                "Math-Verify process failed: "
                f"{isolated.exception_type}: {isolated.exception_message}\n{isolated.traceback_text}"
            )
    elif timeout_mode == "inline":
        payload = _result_payload(
            source=source,
            solution_str=solution_text,
            ground_truth=ground_truth_text,
            expected_multiple_answers=expected_multiple_answers,
        )
    else:
        raise ValueError(f"Unsupported verify_timeout_mode: {verify_timeout_mode!r}")

    latency_ms = (time.perf_counter() - started) * 1000.0
    payload.setdefault("failure_reason", "extraction_failure" if not payload.get("extraction_ok") else "")
    payload.setdefault("verifier_error_detail", "")
    payload.setdefault("fallback_used", False)
    payload["parser_latency_ms"] = float(latency_ms if math.isfinite(latency_ms) else 0.0)
    payload["parser_timeout"] = bool(payload.get("failure_reason") == "parse_timeout")
    payload["parser_exception"] = bool(payload.get("failure_reason") in {"parse_exception", "verifier_internal_error"})
    payload["verifier_input_chars"] = int(input_chars)
    return payload
