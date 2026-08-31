from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

from signal_forge.rewards.process_timeout import _join_until_deadline, call_with_hard_timeout


def test_spurious_early_join_return_does_not_create_subdeadline_timeout():
    class EarlyReturningProcess:
        def __init__(self):
            self.join_calls = 0

        def is_alive(self):
            return self.join_calls < 2

        def join(self, timeout):
            assert timeout > 0
            self.join_calls += 1

    process = EarlyReturningProcess()

    assert _join_until_deadline(process, time.perf_counter() + 1.0)
    assert process.join_calls == 2


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


def test_concurrent_isolated_calls_do_not_report_subdeadline_timeouts():
    def invoke(_):
        return call_with_hard_timeout(
            "signal_forge.rewards.process_timeout:_deliberately_hanging_verifier_for_tests",
            {"delay_seconds": 0.02},
            timeout_seconds=10.0,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(invoke, range(24)))

    assert all(result.ok for result in results)
    assert all(not result.timed_out for result in results)
    assert all(result.value == {"score": 1.0} for result in results)
    assert all(result.elapsed_ms < 10_000 for result in results)
