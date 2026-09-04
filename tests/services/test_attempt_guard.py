from __future__ import annotations

import pytest

from vidaio.services.attempt_guard import (
    ConsecutiveFailureGuard,
    CursorStallGuard,
    configured_stall_limit,
    configured_failure_limit,
)


@pytest.mark.parametrize("value", (None, True, False, 0, -1, "x"))
def test_failure_limit_must_be_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        configured_failure_limit(value, setting="driver.failure_limit")


def test_consecutive_failures_flip_health_and_reach_the_exact_threshold() -> None:
    guard = ConsecutiveFailureGuard(max_failures=3)

    assert guard.healthy
    assert guard.record_failure(RuntimeError("one")) is False
    assert guard.healthy is False
    assert guard.consecutive_failures == 1
    assert guard.last_error == "RuntimeError: one"
    assert guard.record_failure(OSError("two")) is False
    assert guard.record_failure(ValueError("three")) is True
    assert guard.consecutive_failures == 3


def test_success_or_expected_no_work_resets_the_streak_and_health() -> None:
    guard = ConsecutiveFailureGuard(max_failures=2)
    assert guard.record_failure(RuntimeError("transient")) is False

    guard.record_success()

    assert guard.healthy
    assert guard.consecutive_failures == 0
    assert guard.last_error is None
    assert guard.record_failure(RuntimeError("new streak")) is False


@pytest.mark.parametrize("value", (None, True, False, 0, 1, -1, "x"))
def test_stall_limit_preserves_one_transient_hold(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer|at least 2"):
        configured_stall_limit(value, setting="auditor.max_stalls")


def test_cursor_stall_degrades_then_fails_at_bounded_threshold() -> None:
    guard = CursorStallGuard(max_stalls=4)

    assert guard.record_stall(42) is False
    assert guard.healthy  # one expected HOLD never degrades health
    assert guard.record_stall(42) is False
    assert guard.healthy is False
    assert guard.record_stall(42) is False
    assert guard.record_stall(42) is True
    assert guard.consecutive_stalls == 4


def test_cursor_progress_or_a_new_epoch_resets_the_stall_budget() -> None:
    guard = CursorStallGuard(max_stalls=4)
    guard.record_stall(42)
    guard.record_stall(42)
    assert guard.healthy is False

    guard.record_progress()
    assert guard.healthy
    assert guard.epoch_id is None
    assert guard.consecutive_stalls == 0

    guard.record_stall(42)
    guard.record_stall(43)
    assert guard.epoch_id == 43
    assert guard.consecutive_stalls == 1
    assert guard.healthy
