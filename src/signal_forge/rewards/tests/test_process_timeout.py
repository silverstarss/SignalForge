from __future__ import annotations

import time

from signal_forge.rewards.process_timeout import call_with_hard_timeout


def test_hanging_verifier_deadline_terminates_child_and_later_request_succeeds():
    started = time.perf_counter()
    result = call_with_hard_timeout(
        "signal_forge.rewards.process_timeout:_deliberately_hanging_verifier_for_tests",
        {"delay_seconds": 5.0},
        timeout_seconds=0.25,
    )
    elapsed = time.perf_counter() - started
    assert result.timed_out
    assert not result.ok
    assert elapsed < 2.0
    assert result.exitcode is not None

    next_result = call_with_hard_timeout(
        "signal_forge.rewards.process_timeout:_deliberately_hanging_verifier_for_tests",
        {"delay_seconds": 0.01},
        timeout_seconds=2.0,
    )
    assert next_result.ok
    assert not next_result.timed_out
    assert next_result.value == {"score": 1.0}
