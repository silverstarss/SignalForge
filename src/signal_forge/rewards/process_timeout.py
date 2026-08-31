"""Spawn-based hard timeout helpers for serializable verifier calls."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IsolatedCallResult:
    ok: bool
    timed_out: bool
    value: Any = None
    exception_type: str = ""
    exception_message: str = ""
    traceback_text: str = ""
    elapsed_ms: float = 0.0
    exitcode: int | None = None


def _join_until_deadline(process: mp.Process, deadline: float) -> bool:
    """Wait through spurious early ``join`` returns until exit or deadline."""
    while process.is_alive():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return False
        process.join(remaining)
    return True


def _load_function(function_ref: str):
    module_name, sep, func_name = function_ref.partition(":")
    if not sep:
        raise ValueError(f"function_ref must be 'module:function', got {function_ref!r}")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _child_entry(result_queue, function_ref: str, kwargs: dict[str, Any]) -> None:
    # Keep verifier children CPU/string-only. This prevents accidental CUDA init
    # through inherited environment in Ray workers.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.perf_counter()
    try:
        fn = _load_function(function_ref)
        value = fn(**kwargs)
        result_queue.put(
            {
                "ok": True,
                "value": value,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
    except BaseException as exc:  # noqa: BLE001 - child must report every failure.
        result_queue.put(
            {
                "ok": False,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback_text": traceback.format_exc(),
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }
        )


def call_with_hard_timeout(
    function_ref: str,
    kwargs: dict[str, Any],
    timeout_seconds: float,
) -> IsolatedCallResult:
    """Run a serializable call in a spawned child and kill it on deadline."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_child_entry, args=(result_queue, function_ref, kwargs))
    started = time.perf_counter()
    process.start()
    completed = _join_until_deadline(process, started + timeout_seconds)

    if not completed:
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join(1.0)
        return IsolatedCallResult(
            ok=False,
            timed_out=True,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            exitcode=process.exitcode,
        )

    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        return IsolatedCallResult(
            ok=False,
            timed_out=False,
            exception_type="verifier_internal_error",
            exception_message="verifier child exited without returning a result",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            exitcode=process.exitcode,
        )

    return IsolatedCallResult(
        ok=bool(payload.get("ok")),
        timed_out=False,
        value=payload.get("value"),
        exception_type=str(payload.get("exception_type", "")),
        exception_message=str(payload.get("exception_message", "")),
        traceback_text=str(payload.get("traceback_text", "")),
        elapsed_ms=float(payload.get("elapsed_ms", (time.perf_counter() - started) * 1000.0)),
        exitcode=process.exitcode,
    )


def _deliberately_hanging_verifier_for_tests(delay_seconds: float = 60.0) -> dict[str, Any]:
    time.sleep(delay_seconds)
    return {"score": 1.0}
