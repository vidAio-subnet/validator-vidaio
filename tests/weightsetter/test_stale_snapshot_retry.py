"""A stale-pointer HOLD retries soon; other HOLDs and REFUSEs keep the full cadence.

Mainnet 2026-09-06: the thin validator attempted every 72 min, phase-locked ~1 min
after each epoch close, and saw ``pointer=(24967, …), chain=(24968, …)`` before the
finalizer published — two publications in sixteen attempts.
"""
from __future__ import annotations

import asyncio

from vidaio.weightsetter.shared_snapshot import SnapshotDigestMismatch, SnapshotUnavailable


class _Provider:
    def __init__(self, exc: Exception) -> None:
        self.exc, self.calls = exc, 0

    def miner_snapshots(self):
        self.calls += 1
        raise self.exc


def _lagging() -> _Provider:
    return _Provider(SnapshotUnavailable(
        "authority latest pointer is not the archive chain's latest finalized epoch: "
        "pointer=(24967, 9011542), chain=(24968, 9011902)"))


async def test_stale_pointer_hold_retries_on_the_short_cadence(make_setter, mk_miner):
    provider = _lagging()
    setter = make_setter([mk_miner(1)], snapshots_override=provider,
                         attempt_interval_seconds=3600.0, stale_snapshot_retry_seconds=0.01)
    attempts = 0
    original = setter.attempt_once

    async def attempt():
        nonlocal attempts
        attempts += 1
        if attempts >= 3:
            setter.request_stop()
        return await original()

    setter.attempt_once = attempt
    await asyncio.wait_for(setter.run(), timeout=5)  # far below one 3600 s cadence
    assert attempts == 3 and provider.calls == 3
    assert setter._next_attempt_delay(0.0) == 0.01


async def test_stale_hold_shortens_only_the_next_wait(make_setter, mk_miner):
    setter = make_setter([mk_miner(1)], snapshots_override=_lagging(),
                         attempt_interval_seconds=3600.0, stale_snapshot_retry_seconds=120.0)
    assert await setter.attempt_once() is False
    assert setter._next_attempt_delay(20.0) == 100.0
    setter._hold_retry_soon = False  # what the loop does before every attempt
    assert setter._next_attempt_delay(20.0) == 3580.0
    assert setter._next_attempt_delay(0.0) == 3600.0


async def test_tamper_refuse_keeps_the_full_cadence(make_setter, mk_miner):
    provider = _Provider(SnapshotDigestMismatch("someone tampered"))
    setter = make_setter([mk_miner(1)], snapshots_override=provider,
                         attempt_interval_seconds=3600.0, stale_snapshot_retry_seconds=0.01)
    setter._hold_retry_soon = False
    assert await setter.attempt_once() is False
    assert setter._next_attempt_delay(0.0) == 3600.0
