"""Thin veRL reward adapter over the frozen RewardScope Math-Verify verifier.

The adapter intentionally owns no grading logic. It only maps veRL's reward
function signature to RewardScope's frozen training verifier and returns the
flat dict format consumed by this repository's NaiveRewardManager.
"""

from __future__ import annotations

from functools import lru_cache
import math
import os
import time
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


def _result_payload(source: str, solution_str: str, ground_truth: str) -> dict[str, Any]:
    result = _verify(source=source, solution_str=solution_str, ground_truth=ground_truth)
    extraction = result.extraction
    raw_correctness = float(result.is_correct)
    return {
        "score": raw_correctness,
        "acc": raw_correctness,
        "raw_correctness": raw_correctness,
        "extraction_ok": bool(extraction.extraction_ok),
        "format_ok": bool(extraction.format_ok),
        "verification_status": extraction.extraction_status.value,
        "verification_error_type": result.error_type or "",
        "predicted_answer": extraction.normalized_answer or "",
        "raw_answer": extraction.raw_answer or "",
        "reward_source": source,
    }


def _verify_serializable(source: str, solution_str: str, ground_truth: str) -> dict[str, Any]:
    """Child-process entrypoint. RewardScope itself keeps parsing_timeout=None."""
    return _result_payload(source=source, solution_str=solution_str, ground_truth=ground_truth)


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
    return {
        "score": score,
        "acc": score,
        "raw_correctness": score,
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


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    verify_timeout_mode: str | None = None,
    verify_timeout_seconds: float | None = None,
    verify_timeout_fallback: bool | None = None,
    verify_timeout_fallback_score: float = 0.0,
    verifier_max_input_chars: int | None = None,
) -> dict[str, Any]:
    """Return a binary Math-Verify reward and diagnostic fields for veRL.

    Parser and verifier exceptions intentionally propagate. A verifier/library
    failure is a training correctness bug, not an ordinary wrong answer unless
    the explicit process-timeout fallback path is enabled by config.
    """
    started = time.perf_counter()
    source = _canonical_source(data_source, extra_info)
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
        if not fallback_enabled:
            raise ValueError(f"Verifier input too long: {input_chars} > {verifier_max_input_chars} chars")
        payload = _fallback_payload(
            source=source,
            reason="input_too_long",
            fallback_score=verify_timeout_fallback_score,
            detail=f"{input_chars} > {verifier_max_input_chars}",
        )
    elif timeout_mode == "process":
        if timeout_seconds is None or timeout_seconds <= 0:
            raise ValueError("verify_timeout_mode='process' requires verify_timeout_seconds > 0")
        isolated = call_with_hard_timeout(
            "signal_forge.rewards.math_verify_adapter:_verify_serializable",
            {"source": source, "solution_str": solution_text, "ground_truth": ground_truth_text},
            timeout_seconds=float(timeout_seconds),
        )
        if isolated.ok:
            payload = dict(isolated.value)
        elif isolated.timed_out:
            if not fallback_enabled:
                raise TimeoutError(f"Math-Verify process timed out after {timeout_seconds}s")
            payload = _fallback_payload(
                source=source,
                reason="parse_timeout",
                fallback_score=verify_timeout_fallback_score,
                detail=f"deadline_seconds={timeout_seconds}",
            )
        else:
            reason = "parse_exception" if isolated.exception_type else "verifier_internal_error"
            if not fallback_enabled:
                raise RuntimeError(
                    "Math-Verify process failed: "
                    f"{isolated.exception_type}: {isolated.exception_message}\n{isolated.traceback_text}"
                )
            payload = _fallback_payload(
                source=source,
                reason=reason,
                fallback_score=verify_timeout_fallback_score,
                detail=f"{isolated.exception_type}: {isolated.exception_message}",
            )
    elif timeout_mode == "inline":
        payload = _result_payload(source=source, solution_str=solution_text, ground_truth=ground_truth_text)
    else:
        raise ValueError(f"Unsupported verify_timeout_mode: {verify_timeout_mode!r}")

    latency_ms = (time.perf_counter() - started) * 1000.0
    payload.setdefault("failure_reason", "extraction_failure" if not payload.get("extraction_ok") else "")
    payload.setdefault("fallback_used", False)
    payload["parser_latency_ms"] = float(latency_ms if math.isfinite(latency_ms) else 0.0)
    payload["parser_timeout"] = bool(payload.get("failure_reason") == "parse_timeout")
    payload["parser_exception"] = bool(payload.get("failure_reason") in {"parse_exception", "verifier_internal_error"})
    payload["verifier_input_chars"] = int(input_chars)
    return payload
