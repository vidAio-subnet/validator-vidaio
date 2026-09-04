"""Bound repeated unexpected driver-loop failures.

Long-running authority/auditor drivers legitimately have no-work passes, but an
invalid credential, corrupt database, or permanently incompatible peer must not
leave a green process retrying forever.  ``ConsecutiveFailureGuard`` keeps that
policy deterministic and independently testable; the service decides how to log
and calls ``fail_fatal`` when ``record_failure`` returns true.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
DEFAULT_MAX_CONSECUTIVE_STALLS = 30


def configured_failure_limit(value: object, *, setting: str) -> int:
    """Parse a strictly positive retry limit without accepting booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{setting} must be a positive integer")
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{setting} must be a positive integer") from exc
    if limit < 1:
        raise ValueError(f"{setting} must be a positive integer")
    return limit


def configured_stall_limit(value: object, *, setting: str) -> int:
    """Parse a stall budget that preserves one genuinely transient HOLD."""
    limit = configured_failure_limit(value, setting=setting)
    if limit < 2:
        raise ValueError(f"{setting} must be an integer of at least 2")
    return limit


@dataclass(slots=True)
class ConsecutiveFailureGuard:
    """Track attempts and request a fatal restart at a bounded threshold."""

    max_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    consecutive_failures: int = 0
    last_error: str | None = None

    def __post_init__(self) -> None:
        self.max_failures = configured_failure_limit(
            self.max_failures, setting="max_failures"
        )

    @property
    def healthy(self) -> bool:
        return self.consecutive_failures == 0

    def record_success(self) -> None:
        """A successful or expected no-work pass breaks the failure streak."""
        self.consecutive_failures = 0
        self.last_error = None

    def record_failure(self, error: BaseException) -> bool:
        """Record an unexpected error; return true once restart is required."""
        self.consecutive_failures += 1
        self.last_error = f"{type(error).__name__}: {error}"
        return self.consecutive_failures >= self.max_failures


@dataclass(slots=True)
class CursorStallGuard:
    """Bound repeated HOLD/REFUSE passes on one required cursor epoch.

    A single HOLD is an expected distributed-system transient. Repeated passes
    against the same required epoch are different: the no-skip cursor is doing
    its job, but a green process must not conceal that it has stopped making
    audit progress. Health degrades halfway through the bounded retry budget
    (never on the first pass), and the caller must terminate/restart when
    ``record_stall`` returns true. Any completed/no-work pass resets the streak;
    changing epochs starts a fresh streak rather than carrying an old epoch's
    budget forward.
    """

    max_stalls: int = DEFAULT_MAX_CONSECUTIVE_STALLS
    epoch_id: int | None = None
    consecutive_stalls: int = 0

    def __post_init__(self) -> None:
        self.max_stalls = configured_stall_limit(
            self.max_stalls, setting="max_stalls"
        )

    @property
    def degrade_after(self) -> int:
        """First count reported unhealthy, leaving a pre-fatal alert window."""
        return max(2, (self.max_stalls + 1) // 2)

    @property
    def healthy(self) -> bool:
        return self.consecutive_stalls < self.degrade_after

    def record_progress(self) -> None:
        """A completed walk or genuine no-work pass clears the stall streak."""
        self.epoch_id = None
        self.consecutive_stalls = 0

    def record_stall(self, epoch_id: int) -> bool:
        """Record one blocked pass; return true at the exact fatal threshold."""
        epoch = int(epoch_id)
        if epoch < 0:
            raise ValueError("stalled epoch_id must be non-negative")
        if self.epoch_id != epoch:
            self.epoch_id = epoch
            self.consecutive_stalls = 1
        else:
            self.consecutive_stalls += 1
        return self.consecutive_stalls >= self.max_stalls
