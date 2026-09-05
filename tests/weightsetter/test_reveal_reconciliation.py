from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import pytest

from weightsetter_support import CommitRevealChain

from vidaio.chain import ChainStateUnavailable, PendingWeightReveal, SubmittedWeights
from vidaio.weightsetter import intents
from vidaio.weightsetter.config import WeightSetterConfig


async def test_confirmed_reveal_refreshes_health_and_counts_success_exactly_once(
    make_setter, chain, conn, mk_miner, clock, caplog,
):
    remote = CommitRevealChain(chain)
    setter = make_setter(
        [mk_miner(1)], chain_override=remote, validator_hotkey="validator",
        max_last_success_age_seconds=100, attempt_interval_seconds=50,
        reveal_grace_seconds=50, publication_enabled=False,
    )
    assert not await setter.attempt_once()
    assert intents.intents(conn)[0]["resolution"] == "commit_reveal_pending"
    assert setter.metric_successes._value.get() == 0
    assert setter.metric_unresolved_intents._value.get() == 0
    clock.advance(101)
    assert not setter.health.health_payload()[0]
    await setter.reconcile()
    assert not setter.health.health_payload()[0]
    assert intents.intents(conn)[0]["state"] == intents.STATE_PENDING
    remote.reveal()
    assert await setter.reconcile() == 1
    assert setter.health.health_payload()[0]
    assert setter.metric_successes._value.get() == 1
    assert intents.intents(conn)[0]["resolution"] == "commit_reveal_confirmed"
    await setter.reconcile()
    assert setter.metric_successes._value.get() == 1
    clock.advance(101)
    assert not setter.health.health_payload()[0]
    assert "crashed" not in caplog.text
    assert "NOT settled" not in caplog.text


async def test_loop_reconciles_reveal_and_publishes_before_next_submission(
    make_setter, chain, conn, mk_miner, clock,
):
    remote = CommitRevealChain(chain)
    setter = make_setter(
        [mk_miner(1)], chain_override=remote, validator_hotkey="validator",
        attempt_interval_seconds=30, reconciliation_interval_seconds=0.01,
        publication_attempt_timeout_seconds=0.01, publication_enabled=False,
    )
    loop = asyncio.create_task(setter.run())
    try:
        async with asyncio.timeout(2):
            while not remote.pending:
                await asyncio.sleep(0.005)
            clock.advance(10)
            remote.reveal()
            while intents.intents(conn)[0]["state"] != intents.STATE_PUBLISHED:
                await asyncio.sleep(0.005)
        assert remote.calls == 1
        assert len(intents.intents(conn)) == 1
        assert setter.metric_successes._value.get() == 1
        assert setter._last_success_at == clock()
    finally:
        setter.request_stop()
        await asyncio.wait_for(loop, 2)


async def test_reveal_mismatch_is_not_success_or_publication(make_setter, chain, conn, mk_miner):
    remote = CommitRevealChain(chain)
    setter = make_setter(
        [mk_miner(1)], chain_override=remote, validator_hotkey="validator",
        publication_enabled=False,
    )
    await setter.attempt_once()
    remote.reveal()
    remote.staged = {77: 65535.0}
    await setter.reconcile()
    assert intents.intents(conn)[0]["state"] == intents.STATE_PENDING
    assert setter.metric_successes._value.get() == 0
    assert setter._last_success_at is None


async def test_durable_acceptance_failure_does_not_refresh_health(
    make_setter, chain, conn, mk_miner, monkeypatch,
):
    remote = CommitRevealChain(chain)
    setter = make_setter(
        [mk_miner(1)], chain_override=remote, validator_hotkey="validator",
        publication_enabled=False,
    )
    await setter.attempt_once()
    remote.reveal()

    def broken(*args, **kwargs):
        raise RuntimeError("durable write unavailable")

    monkeypatch.setattr(intents, "accept_with_vector", broken)
    with pytest.raises(RuntimeError, match="durable write"):
        await setter.reconcile()
    assert intents.intents(conn)[0]["state"] == intents.STATE_PENDING
    assert setter.metric_successes._value.get() == 0
    assert setter._last_success_at is None


async def test_poll_failure_does_not_kill_scheduled_attempts(make_setter, mk_miner):
    setter = make_setter(
        [mk_miner(1)], attempt_interval_seconds=0.04, reconciliation_interval_seconds=0.01,
    )
    attempts = 0
    original_attempt = setter.attempt_once

    async def attempt():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await original_attempt()
            intent_id = intents.record_intent(
                setter._conn, created_at=setter._iso_now(), attempt_block=2,
                version_key=16, weights={1: 1.0}, packet_digests=[],
            )
            intents.note_commit_reveal_pending(setter._conn, intent_id)
        else:
            setter.request_stop()

    async def broken_reconcile(**kwargs):
        raise RuntimeError("read unavailable")

    setter.attempt_once = attempt
    setter.reconcile = broken_reconcile
    await asyncio.wait_for(setter.run(), timeout=2)
    assert attempts == 2
    assert setter.metric_loop_errors._value.get() >= 1


def test_health_threshold_covers_cadence_plus_reveal_allowance():
    config = WeightSetterConfig(
        max_last_success_age_seconds=60, attempt_interval_seconds=4320,
        reveal_grace_seconds=4320,
    )
    assert config.effective_max_last_success_age_seconds == 8640
    assert WeightSetterConfig().effective_max_last_success_age_seconds == 8640


@pytest.mark.parametrize("values", [
    {"reconciliation_interval_seconds": 0}, {"reveal_grace_seconds": -1},
])
def test_invalid_reveal_timing_rejected(values):
    with pytest.raises(ValueError):
        WeightSetterConfig(**values)


@pytest.fixture
def poll_clock(monkeypatch):
    def install(setter, on_wake=None):
        now = [0.0]
        waits = []
        real_wait_for = asyncio.wait_for
        real_clock = setter._clock
        setter._monotonic_clock = lambda: now[0]
        setter._clock = lambda: real_clock() + timedelta(seconds=now[0])

        async def wait_for(awaitable, timeout):
            if (
                asyncio.iscoroutine(awaitable)
                and awaitable.cr_code is asyncio.Event.wait.__code__
            ):
                awaitable.close()
                waits.append(timeout)
                now[0] += timeout
                if on_wake is not None:
                    on_wake(now[0])
                raise TimeoutError
            return await real_wait_for(awaitable, timeout)

        monkeypatch.setattr(asyncio, "wait_for", wait_for)
        return now, waits

    return install


def record_pending(setter, *, reveal=True):
    intent_id = intents.record_intent(
        setter._conn,
        created_at=setter._iso_now(),
        attempt_block=1,
        version_key=16,
        weights={1: 1.0},
        packet_digests=[],
    )
    if reveal:
        intents.note_commit_reveal_pending(setter._conn, intent_id)
    return intent_id


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("gap", ["none", "empty", "old_match", "old_other", "undated"])
async def test_no_commit_old_or_empty_finalized_weights_never_denies_known_cr(
    make_setter, chain, conn, mk_miner, clock, caplog, gap, restart,
):
    class FinalityGapChain(CommitRevealChain):
        finality_gap = False

        def submitted_weights(self, hotkey):
            if not self.finality_gap:
                return super().submitted_weights(hotkey)
            if gap == "none":
                return None
            if gap == "empty":
                return SubmittedWeights(weights={}, block=0)
            if gap == "old_other":
                return SubmittedWeights(weights={77: 65535.0}, block=0)
            return SubmittedWeights(
                weights=dict(self.staged), block=None if gap == "undated" else 0
            )

    remote = FinalityGapChain(chain)
    options = dict(chain_override=remote, validator_hotkey="validator")
    setter = make_setter([mk_miner(1)], **options)
    assert not await setter.attempt_once()
    remote.pending = False
    remote.finality_gap = True
    clock.advance(100_000)
    if restart:
        setter = make_setter([mk_miner(1)], **options)
    with caplog.at_level(logging.INFO, logger="weight-setter"):
        for _ in range(5):
            assert await setter.reconcile() == 0
            assert not await setter.attempt_once()
            row = intents.intents(conn)[0]
            assert row["state"] == intents.STATE_PENDING
            assert row["last_check"] == "unknown"
            assert row["resolution"] == "commit_reveal_pending"
            assert row["commitment_id"] is None
    assert len(intents.intents(conn)) == 1
    assert remote.calls == 1
    assert chain.anchored == []
    assert setter.metric_successes._value.get() == 0
    assert setter._last_success_at is None
    assert "ABANDONING" not in caplog.text
    remote.finality_gap = False
    remote.reveal()
    assert await setter.reconcile() == 1
    assert await setter.reconcile() == 0
    assert intents.intents(conn)[0]["state"] == intents.STATE_PUBLISHED
    assert setter.metric_successes._value.get() == 1
    assert len(chain.anchored) == 1
    assert remote.calls == 1


async def test_pending_reveal_inner_exception_logs_info_once_per_intent(
    make_setter, chain, caplog,
):
    remote = CommitRevealChain(chain)
    remote.pending = True
    setter = make_setter([], chain_override=remote, validator_hotkey="validator")
    intent_ids = [record_pending(setter), record_pending(setter)]
    with caplog.at_level(logging.INFO, logger="weight-setter"):
        for _ in range(10):
            assert await setter.reconcile() == 0
            assert not await setter.attempt_once()
    records = [record for record in caplog.records if record.name == "weight-setter"]
    assert [record.fields["intent_id"] for record in records] == intent_ids
    assert all(record.levelno == logging.INFO for record in records)
    assert all("CRv4 reveal remains UNKNOWN" in record.message for record in records)
    assert remote.calls == 0


async def test_unknown_non_cr_read_still_warns(make_setter, chain, caplog):
    remote = CommitRevealChain(chain)
    remote.pending = True
    setter = make_setter([], chain_override=remote, validator_hotkey="validator")
    record_pending(setter, reveal=False)
    with caplog.at_level(logging.WARNING, logger="weight-setter"):
        assert await setter.reconcile() == 0
    assert "reading our on-chain weight vector failed" in caplog.text


@pytest.mark.parametrize("message", ["RPC down", "malformed Weights storage"])
async def test_known_cr_generic_unavailability_warns_on_every_poll(
    make_setter, chain, conn, caplog, message,
):
    class UnavailableChain(CommitRevealChain):
        def submitted_weights(self, hotkey):
            raise ChainStateUnavailable(message)

    remote = UnavailableChain(chain)
    setter = make_setter([], chain_override=remote, validator_hotkey="validator")
    intent_id = record_pending(setter)
    with caplog.at_level(logging.INFO, logger="weight-setter"):
        for _ in range(5):
            assert await setter.reconcile(pending_reveals_only=True) == 0
    warnings = [
        record for record in caplog.records
        if record.name == "weight-setter" and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 5
    assert all("reading our on-chain weight vector failed" in record.message for record in warnings)
    assert all(message in record.fields["error"] for record in warnings)
    assert not any(
        record.name == "weight-setter" and record.levelno == logging.INFO
        for record in caplog.records
    )
    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_PENDING
    assert row["last_check"] == "unknown"
    assert row["reveal_wait_logged_at"] is None
    assert remote.calls == 0
    assert chain.anchored == []

    def pending_reveal(hotkey):
        raise PendingWeightReveal("weight commit pending reveal")

    remote.submitted_weights = pending_reveal
    with caplog.at_level(logging.INFO, logger="weight-setter"):
        for _ in range(3):
            assert await setter.reconcile(pending_reveals_only=True) == 0
    records = [record for record in caplog.records if record.name == "weight-setter"]
    assert sum(record.levelno == logging.WARNING for record in records) == 5
    assert sum(record.levelno == logging.INFO for record in records) == 1


@pytest.mark.parametrize("state", [None, "pending", "published"])
async def test_no_between_attempt_poll_without_pending_cr_row(
    make_setter, poll_clock, state,
):
    setter = make_setter([])
    if state is not None:
        intent_id = record_pending(setter, reveal=state == "published")
        if state == "published":
            intents.mark_published(setter._conn, intent_id, at=setter._iso_now())

    async def unexpected(**kwargs):
        pytest.fail("no CR row: neither RPC reconciliation nor publication is allowed")

    setter.reconcile = unexpected
    setter._drain_one_accepted_publication = unexpected
    now, waits = poll_clock(setter)
    await setter._wait_for_next_attempt(4320)
    assert now[0] == 4320
    assert waits == [4320]


async def test_default_poll_can_write_anchor_but_never_weights_or_advance_cadence(
    make_setter, chain, conn, mk_miner, poll_clock,
):
    remote = CommitRevealChain(chain)
    setter = make_setter(
        [mk_miner(1)], chain_override=remote, validator_hotkey="validator"
    )
    assert not await setter.attempt_once()
    now, waits = poll_clock(setter, on_wake=lambda timestamp: remote.reveal())
    await setter._wait_for_next_attempt(4320)
    assert waits == [300, 4020]
    assert now[0] == 4320
    assert remote.calls == 1
    assert len(chain.anchored) == 1
    assert len(intents.intents(conn)) == 1
    assert intents.intents(conn)[0]["state"] == intents.STATE_PUBLISHED
    assert setter.metric_successes._value.get() == 1


async def test_poll_publication_budget_never_overruns_submission_deadline(
    make_setter, chain, conn, mk_miner, poll_clock,
):
    remote = CommitRevealChain(chain)
    setter = make_setter(
        [mk_miner(1)], chain_override=remote, validator_hotkey="validator"
    )
    assert not await setter.attempt_once()
    now, waits = poll_clock(setter, on_wake=lambda timestamp: remote.reveal())
    await setter._wait_for_next_attempt(599)
    assert waits == [300, 299]
    assert now[0] == 599
    assert chain.anchored == []
    assert remote.calls == 1
    assert intents.intents(conn)[0]["state"] == intents.STATE_ACCEPTED
    assert setter.metric_pending_intents._value.get() == 1


@pytest.mark.parametrize("pending_cr", [False, True])
async def test_poll_retries_exponentially_three_per_tempo_without_drain_bypass(
    make_setter, chain, conn, poll_clock, pending_cr,
):
    class FailingAnchorChain(CommitRevealChain):
        async def anchor_commitment(self, payload):
            writes.append(now[0])
            raise OSError("anchor temporarily unavailable")

    writes = []
    remote = FailingAnchorChain(chain)
    remote.pending = True
    setter = make_setter([], chain_override=remote, validator_hotkey="validator")
    now, waits = poll_clock(setter)
    accepted_ids = [record_pending(setter, reveal=False) for _ in range(2)]
    for intent_id in accepted_ids:
        intents.mark_accepted(
            conn, intent_id, accepted_block=1, resolution="chain_accepted"
        )
    if pending_cr:
        record_pending(setter)
    else:
        def unexpected_refresh():
            pytest.fail("accepted-only publication must not refresh the metagraph")

        remote.refresh = unexpected_refresh
    await setter._wait_for_next_attempt(4320)
    assert writes == [300, 600, 1200]
    if pending_cr:
        assert all(delay <= 300 for delay in waits)
    else:
        assert waits == [300, 300, 600, 3120]
    assert len(writes) == len(set(writes))
    assert not await setter._drain_one_accepted_publication()
    assert not await setter._publish_intent_bounded(accepted_ids[1])
    if pending_cr:
        assert await setter.reconcile() == 0
    assert writes == [300, 600, 1200]
    if pending_cr:
        assert setter.metric_pending_intents._value.get() == 3
    now[0] = 4620
    assert not await setter._drain_one_accepted_publication()
    assert writes == [300, 600, 1200, 4620]
    now[0] = 7019
    assert not await setter._drain_one_accepted_publication()
    assert writes == [300, 600, 1200, 4620]
    now[0] = 7020
    assert not await setter._drain_one_accepted_publication()
    assert writes == [300, 600, 1200, 4620, 7020]
    assert remote.calls == 0
    assert chain.anchored == []
    assert len(intents.intents(conn)) == 2 + int(pending_cr)


async def test_post_submit_drain_cannot_immediately_repeat_failed_publication(
    make_setter, chain, conn, mk_miner, poll_clock, monkeypatch,
):
    setter = make_setter([mk_miner(1)])
    now, _ = poll_clock(setter)
    writes = []

    async def fail_anchor(payload):
        writes.append(now[0])
        raise OSError("anchor unavailable")

    monkeypatch.setattr(chain, "anchor_commitment", fail_anchor)
    assert await setter.attempt_once()
    for _ in range(10):
        assert not await setter._drain_one_accepted_publication()
        assert await setter.reconcile() == 0
    assert writes == [0]
    assert len(chain.weight_calls) == 1
    assert intents.intents(conn)[0]["state"] == intents.STATE_ACCEPTED
    now[0] = 300
    assert not await setter._drain_one_accepted_publication()
    assert writes == [0, 300]


async def test_detached_publication_remains_single_flight_across_retry_windows(
    make_setter, chain, mk_miner, poll_clock, monkeypatch,
):
    setter = make_setter([mk_miner(1)], publication_attempt_timeout_seconds=0.01)
    now, waits = poll_clock(setter)
    release = asyncio.Event()
    writes = []
    original_anchor = chain.anchor_commitment

    async def slow_anchor(payload):
        writes.append(now[0])
        await release.wait()
        return await original_anchor(payload)

    monkeypatch.setattr(chain, "anchor_commitment", slow_anchor)
    try:
        assert await setter.attempt_once()
        now[0] = 100_000
        for _ in range(5):
            assert not await setter._drain_one_accepted_publication()
            assert await setter.reconcile() == 0
        await setter._wait_for_next_attempt(4320)
        assert waits == [4320]
        assert writes == [0]
        assert len(chain.weight_calls) == 1
    finally:
        release.set()
        await setter._finish_publication_task()
    assert len(chain.anchored) == 1
    assert not await setter._drain_one_accepted_publication()


async def test_last_reveal_publication_failure_retries_without_more_chain_reads(
    make_setter, chain, conn, mk_miner, poll_clock,
):
    class RecoveringAnchorChain(CommitRevealChain):
        reads = 0

        def refresh(self):
            self.reads += 1
            return self.inner.refresh()

        async def anchor_commitment(self, payload):
            writes.append(now[0])
            if len(writes) == 1:
                raise OSError("first anchor failed")
            return await self.inner.anchor_commitment(payload)

    writes = []
    remote = RecoveringAnchorChain(chain)
    setter = make_setter(
        [mk_miner(1)], chain_override=remote, validator_hotkey="validator"
    )
    assert not await setter.attempt_once()
    reads_before = remote.reads
    now, waits = poll_clock(setter, on_wake=lambda timestamp: remote.reveal())
    await setter._wait_for_next_attempt(4320)
    assert writes == [300, 600]
    assert waits == [300, 300, 3720]
    assert remote.reads == reads_before + 1
    assert remote.calls == 1
    assert len(chain.anchored) == 1
    assert len(intents.intents(conn)) == 1
    assert intents.intents(conn)[0]["state"] == intents.STATE_PUBLISHED
    assert setter.metric_successes._value.get() == 1


async def test_publication_budget_has_no_boundary_burst_or_success_reset(
    make_setter, poll_clock, monkeypatch,
):
    setter = make_setter([])
    now, _ = poll_clock(setter)
    launches = []
    for _ in range(5):
        record_pending(setter, reveal=False)

    async def publish(intent_id):
        launches.append((intent_id, now[0]))
        return True

    monkeypatch.setattr(setter, "_publish_intent", publish)
    for intent_id, timestamp in enumerate([0, 300, 600], start=1):
        now[0] = timestamp
        assert await setter._publish_intent_bounded(intent_id)
    now[0] = 4319
    assert not await setter._publish_intent_bounded(4)
    now[0] = 4320
    assert await setter._publish_intent_bounded(4)
    assert not await setter._publish_intent_bounded(5)
    now[0] = 4619
    assert not await setter._publish_intent_bounded(5)
    now[0] = 4620
    assert await setter._publish_intent_bounded(5)
    assert launches == [(1, 0), (2, 300), (3, 600), (4, 4320), (5, 4620)]


@pytest.mark.parametrize("interval", [1.0, 60.0, 4320.0, 10_000.0])
@pytest.mark.parametrize("grace", [0.0, 30.0, 4320.0])
@pytest.mark.parametrize("maximum", [0.1, 8640.0, 1_000_000.0])
def test_health_limit_is_between_reveal_window_and_one_missed_tempo(
    interval, grace, maximum,
):
    config = WeightSetterConfig(
        attempt_interval_seconds=interval,
        reveal_grace_seconds=grace,
        max_last_success_age_seconds=maximum,
    )
    assert interval + grace <= config.effective_max_last_success_age_seconds
    assert config.effective_max_last_success_age_seconds <= 2 * interval + grace


async def test_health_flags_one_missed_tempo_despite_oversized_override(
    make_setter, clock,
):
    setter = make_setter(
        [], attempt_interval_seconds=50, reveal_grace_seconds=25,
        max_last_success_age_seconds=1_000_000,
    )
    clock.advance(125)
    assert setter.health.health_payload()[0]
    clock.advance(0.01)
    assert not setter.health.health_payload()[0]


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("gap", ["none", "empty", "old_match", "old_other"])
async def test_legacy_null_intent_nine_holds_gap_on_scheduled_path(
    make_setter, chain, conn, clock, gap, restart,
):
    class LegacyCRChain(CommitRevealChain):
        visible = False
        mode_calls = 0

        def commit_reveal_enabled(self):
            self.mode_calls += 1
            return True

        def submitted_weights(self, hotkey):
            if self.visible:
                return SubmittedWeights(weights={1: 65535.0}, block=11)
            if gap == "none":
                return None
            weights = {} if gap == "empty" else {77 if gap == "old_other" else 1: 65535.0}
            return SubmittedWeights(weights=weights, block=1)

    remote = LegacyCRChain(chain)
    setter = make_setter([], chain_override=remote, validator_hotkey="validator")
    for _ in range(8):
        old_id = intents.record_intent(
            conn, created_at=setter._iso_now(), attempt_block=1, version_key=16,
            weights={77: 1.0}, packet_digests=[],
        )
        intents.mark_abandoned(conn, old_id, at=setter._iso_now(), resolution="rejected")
    intent_id = intents.record_intent(
        conn, created_at=setter._iso_now(), attempt_block=10, version_key=16,
        weights={1: 1.0}, packet_digests=[],
    )
    assert intent_id == 9
    assert intents.get_intent(conn, 9)["resolution"] is None
    clock.advance(100_000)
    if restart:
        setter = make_setter([], chain_override=remote, validator_hotkey="validator")
    for _ in range(4):
        assert not await setter.attempt_once()
        row = intents.get_intent(conn, 9)
        assert row["state"] == intents.STATE_PENDING
        assert row["resolution"] == "commit_reveal_pending"
        assert row["last_check"] == "unknown"
    assert remote.mode_calls == 1
    assert remote.calls == 0
    assert len(intents.intents(conn)) == 9
    assert chain.anchored == []
    assert setter._last_success_at is None
    remote.visible = True
    assert await setter.reconcile() == 1
    assert await setter.reconcile() == 0
    assert remote.calls == 0
    assert intents.get_intent(conn, 9)["state"] == intents.STATE_PUBLISHED
    assert len(chain.anchored) == 1


@pytest.mark.parametrize("mode", [None, 1, "true", "error"])
async def test_legacy_unknown_mode_is_not_classified_denied_or_resubmitted(
    make_setter, chain, conn, clock, mode,
):
    class UnknownModeChain(CommitRevealChain):
        def commit_reveal_enabled(self):
            if mode == "error":
                raise ChainStateUnavailable("mode unavailable")
            return mode

        def submitted_weights(self, hotkey):
            pytest.fail("unreadable legacy mode must HOLD before interpreting old weights")

    remote = UnknownModeChain(chain)
    setter = make_setter([], chain_override=remote, validator_hotkey="validator")
    intent_id = record_pending(setter, reveal=False)
    clock.advance(100_000)
    for _ in range(3):
        assert not await setter.attempt_once()
        row = intents.get_intent(conn, intent_id)
        assert row["state"] == intents.STATE_PENDING
        assert row["resolution"] is None
        assert row["last_check"] == "unknown"
    assert remote.calls == 0
    assert len(intents.intents(conn)) == 1
    assert chain.anchored == []


async def test_positive_non_cr_mode_preserves_legacy_denial(make_setter, chain, conn, clock):
    class NonCRChain(CommitRevealChain):
        def commit_reveal_enabled(self):
            return False

    remote = NonCRChain(chain)
    setter = make_setter([], chain_override=remote, validator_hotkey="validator")
    intent_id = record_pending(setter, reveal=False)
    clock.advance(100_000)
    assert await setter.reconcile() == 1
    assert intents.get_intent(conn, intent_id)["state"] == intents.STATE_ABANDONED
    assert remote.calls == 0


async def test_known_cr_stays_protected_when_mode_turns_off(make_setter, chain, conn, clock):
    class DisabledCRChain(CommitRevealChain):
        def commit_reveal_enabled(self):
            return False

    remote = DisabledCRChain(chain)
    setter = make_setter([], chain_override=remote, validator_hotkey="validator")
    intent_id = record_pending(setter)
    clock.advance(100_000)
    assert not await setter.attempt_once()
    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_PENDING
    assert row["last_check"] == "unknown"
    assert remote.calls == 0


async def test_publication_backoff_and_three_start_budget_survive_every_restart(
    make_setter, chain, conn, clock, mk_miner, monkeypatch,
):
    writes = []
    started = clock().timestamp()

    async def fail_anchor(payload):
        writes.append(clock().timestamp() - started)
        raise OSError("anchor unavailable")

    monkeypatch.setattr(chain, "anchor_commitment", fail_anchor)
    setter = make_setter([mk_miner(1)])
    assert await setter.attempt_once()
    for delay in (300, 600):
        setter = make_setter([mk_miner(1)])
        assert not await setter._drain_one_accepted_publication()
        clock.advance(delay)
        assert not await setter._drain_one_accepted_publication()
    assert writes == [0, 300, 900]
    for _ in range(3):
        setter = make_setter([mk_miner(1)])
        assert not await setter._drain_one_accepted_publication()
        assert await setter.reconcile() == 0
    clock.advance(3419)
    setter = make_setter([mk_miner(1)])
    assert not await setter._drain_one_accepted_publication()
    assert writes == [0, 300, 900]
    clock.advance(1)
    assert not await setter._drain_one_accepted_publication()
    assert writes == [0, 300, 900, 4320]
    assert len(chain.weight_calls) == 1
    rows = conn.execute("SELECT * FROM weight_publication_attempts ORDER BY id").fetchall()
    assert [row["failure_count"] for row in rows] == [1, 2, 3, 4]
    assert all(row["intent_id"] == 1 and row["succeeded"] == 0 for row in rows)


async def test_crash_after_retry_reservation_still_spends_budget_and_delays_restart(
    make_setter, chain, conn, clock,
):
    setter = make_setter([])
    intent_id = record_pending(setter, reveal=False)
    intents.mark_accepted(conn, intent_id, accepted_block=1, resolution="accepted")
    reserved = intents.reserve_publication_attempt(
        conn, intent_id, now=clock().timestamp(), interval=4320,
        base_delay=300, timeout=300,
    )
    assert reserved is not None
    setter = make_setter([])
    assert not await setter._drain_one_accepted_publication()
    clock.advance(599)
    assert not await setter._drain_one_accepted_publication()
    assert chain.anchored == []
    clock.advance(1)
    assert await setter._drain_one_accepted_publication()
    rows = conn.execute("SELECT * FROM weight_publication_attempts ORDER BY id").fetchall()
    assert len(rows) == 2
    assert rows[0]["finished_at"] is None
    assert rows[1]["succeeded"] == 1
    assert len(chain.anchored) == 1


async def test_reveal_wait_log_claim_survives_restart(make_setter, chain, caplog):
    remote = CommitRevealChain(chain)
    remote.pending = True
    setter = make_setter([], chain_override=remote, validator_hotkey="validator")
    record_pending(setter)
    with caplog.at_level(logging.INFO, logger="weight-setter"):
        for _ in range(3):
            setter = make_setter([], chain_override=remote, validator_hotkey="validator")
            await setter.reconcile()
    records = [record for record in caplog.records if record.name == "weight-setter"]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
