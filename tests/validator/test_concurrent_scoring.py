"""Bounded scoring preserves ordered evidence and isolates worker failures."""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
import time
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from vidaio.core import apply_migrations, connect
from vidaio.core.config import load_raw_config, section
from vidaio.validator import MIGRATIONS_DIR, ScorePacketEvidence, ValidatorConfig, miner_manager
from vidaio.validator import inference

from validator_support import FakeChallengeClient, FakeMinerClient, FakeScoringClient, mk_neuron


def test_scoring_concurrency_defaults_and_env_override(monkeypatch):
    config = ValidatorConfig()
    assert config.scoring_concurrency == 1
    assert config.scoring_shed_wait_seconds == 120.0
    monkeypatch.setenv("VIDAIO__VALIDATOR__SCORING_CONCURRENCY", "3")
    config = section(load_raw_config(), "validator", ValidatorConfig)
    assert config.scoring_concurrency == 3
    assert isinstance(config.scoring_concurrency, int)


@pytest.mark.parametrize("concurrency", [
    0, 17, -1, 1.0, 1.5, float("nan"), float("inf"), True, False, "invalid",
])
def test_scoring_concurrency_rejects_out_of_range_or_noninteger_values(concurrency):
    with pytest.raises(ValidationError, match="scoring_concurrency"):
        ValidatorConfig(scoring_concurrency=concurrency)


@pytest.mark.parametrize("concurrency", [1, 16])
def test_scoring_concurrency_accepts_both_bounds(concurrency):
    assert ValidatorConfig(scoring_concurrency=concurrency).scoring_concurrency == concurrency


def test_shed_wait_budget_is_nonnegative():
    assert ValidatorConfig(scoring_shed_wait_seconds=0).scoring_shed_wait_seconds == 0
    with pytest.raises(ValidationError, match="scoring_shed_wait_seconds"):
        ValidatorConfig(scoring_shed_wait_seconds=-0.1)


@pytest.fixture
def scoring_round(make_validator, chain, tmp_path):
    """Independent rounds share identical request paths, but no persisted state."""
    connections = []

    def build(scorer, *, concurrency=3, count=5, **config):
        conn = connect(":memory:")
        apply_migrations(conn, MIGRATIONS_DIR)
        connections.append(conn)
        challenge = FakeChallengeClient(tmp_path)
        miner = FakeMinerClient(tmp_path / "outputs")
        miner.tracks = {uid: "compression" for uid in range(1, count + 1)}
        chain.set_neurons([mk_neuron(uid) for uid in range(1, count + 1)])
        validator = make_validator(
            conn=conn, challenge_client=challenge, miner_client=miner, scoring_client=scorer,
            config={"scoring_concurrency": concurrency, "scoring_request_timeout_seconds": 10,
                **config},
        )
        case = SimpleNamespace(validator=validator, conn=conn, challenge=challenge,
            scorer=scorer, evidence=[], scores=None)
        run_track = validator._run_track

        async def capture_track(track, neurons, report, **kwargs):
            scores = await run_track(track, neurons, report, **kwargs)
            case.scores = scores
            case.evidence = list(kwargs["evidence"])
            return scores

        validator._run_track = capture_track
        return case

    yield build
    for conn in connections:
        conn.close()


class EventScorer(FakeScoringClient):
    def __init__(self, *, block=True, count=5):
        super().__init__()
        self.active = 0
        self.peak = 0
        self.started = []
        self.completed = []
        self.cancelled = []
        self.cancel_cleanup_started = asyncio.Event()
        self.cancel_cleanup_release = None
        self.saturated = asyncio.Event()
        self.entered = {uid: asyncio.Event() for uid in range(1, count + 1)}
        self.release = {uid: asyncio.Event() for uid in range(1, count + 1)}
        self.finished = {uid: asyncio.Event() for uid in range(1, count + 1)}
        self.scores = {f"hk{uid}": uid / 10 for uid in range(1, count + 1)}
        if not block:
            self.release_all()

    def release_all(self):
        for release in self.release.values():
            release.set()

    async def score(self, request):
        uid = int(request.item_id.rsplit(":", 1)[1])
        self.started.append(uid)
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.entered[uid].set()
        if self.active == 3:
            self.saturated.set()
        try:
            await self.release[uid].wait()
            result = await super().score(request)
            self.completed.append(uid)
            return result
        except asyncio.CancelledError:
            self.cancelled.append(uid)
            if uid == 1 and self.cancel_cleanup_release is not None:
                self.cancel_cleanup_started.set()
                await self.cancel_cleanup_release.wait()
            raise
        finally:
            self.active -= 1
            self.finished[uid].set()


async def test_three_concurrent_scores_keep_serial_results_and_packet_order(scoring_round, caplog):
    caplog.set_level("INFO")
    serial_scorer = EventScorer(block=False)
    serial = scoring_round(serial_scorer, concurrency=1)
    serial_report = await serial.validator.run_round()
    assert serial_scorer.peak == 1
    assert serial_scorer.started == [1, 2, 3, 4, 5]
    assert [request.miner_hotkey for request in serial_scorer.requests] == [
        "hk1", "hk2", "hk3", "hk4", "hk5"]

    scorer = EventScorer()
    parallel = scoring_round(scorer)
    task = asyncio.create_task(parallel.validator.run_round())
    try:
        await asyncio.wait_for(scorer.saturated.wait(), timeout=2)
        assert scorer.active == scorer.peak == 3
        assert scorer.started == [1, 2, 3]
        # Force completion order to differ from dispatch and evidence order.
        for uid in (3, 4, 5, 2, 1):
            await asyncio.wait_for(scorer.entered[uid].wait(), timeout=2)
            scorer.release[uid].set()
            await asyncio.wait_for(scorer.finished[uid].wait(), timeout=2)
        parallel_report = await asyncio.wait_for(task, timeout=2)
    finally:
        scorer.release_all()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert scorer.peak == 3 and scorer.active == 0
    assert scorer.completed == [3, 4, 5, 2, 1]
    assert replace(parallel_report, round_id=None) == replace(serial_report, round_id=None)
    assert parallel.scores == serial.scores == serial_report.scored
    assert list(parallel.scores) == list(serial.scores) == [1, 2, 3, 4, 5]
    assert parallel.evidence == serial.evidence
    assert [packet.uid for packet in parallel.evidence] == [1, 2, 3, 4, 5]
    assert Counter(request.model_dump_json() for request in scorer.requests) == Counter(
        request.model_dump_json() for request in serial_scorer.requests)
    assert parallel.challenge.resolves == serial.challenge.resolves == [("ch-compression", "resolved")]
    assert [row["packet_digest"] for row in ScorePacketEvidence(parallel.conn).packets()] == [
        row["packet_digest"] for row in ScorePacketEvidence(serial.conn).packets()]
    assert sum(record.getMessage() == "track scored" for record in caplog.records) == 2


async def test_worker_failure_isolated_to_one_miner(scoring_round):
    scorer = FakeScoringClient()
    scorer.fail_hotkeys = {"hk2"}
    case = scoring_round(scorer)
    report = await case.validator.run_round()
    assert report.scoring_failed == [2]
    assert report.scored == {1: 0.8, 3: 0.8, 4: 0.8, 5: 0.8}
    assert report.zeroed == {}
    assert [packet.uid for packet in case.evidence] == [1, 3, 4, 5]
    assert Counter(request.miner_hotkey for request in scorer.requests) == {
        "hk1": 1, "hk2": 2, "hk3": 1, "hk4": 1, "hk5": 1}
    assert miner_manager.get_miner(case.conn, 2)["accumulate_score"] == 0
    assert case.challenge.resolves == [("ch-compression", "resolved")]


@pytest.mark.parametrize("cancel_during_drain", [False, True])
async def test_unexpected_scoring_task_crash_drains_peers_and_rolls_back_round(
    scoring_round, cancel_during_drain,
):
    case = scoring_round(FakeScoringClient())
    await case.validator.run_round()
    before_miners = [dict(row) for row in case.conn.execute("SELECT * FROM miners ORDER BY uid")]
    before_packets = ScorePacketEvidence(case.conn).recent_packet_digests()
    assert before_miners and before_packets
    case.validator.chain.set_neurons([mk_neuron(1, hotkey="hk1-rotated"),
        *(mk_neuron(uid) for uid in range(2, 6))])
    scorer = EventScorer()
    scorer.cancel_cleanup_release = asyncio.Event()
    case.validator.scoring_client = scorer
    score_one = case.validator._score_one
    allow_crash = asyncio.Event()
    discard = case.validator._discard_miner_artifact
    cleanup_active_counts = []

    async def crash_one(item, neuron, response, report, **kwargs):
        if neuron.uid == 2:
            await allow_crash.wait()
            raise RuntimeError("unexpected packet processing crash")
        return await score_one(item, neuron, response, report, **kwargs)

    def check_cleanup(response):
        cleanup_active_counts.append(scorer.active)
        discard(response)

    case.validator._score_one = crash_one
    case.validator._discard_miner_artifact = check_cleanup
    task = asyncio.create_task(case.validator.run_round())
    try:
        await asyncio.wait_for(scorer.entered[1].wait(), timeout=2)
        await asyncio.wait_for(scorer.entered[3].wait(), timeout=2)
        assert scorer.active == 2
        allow_crash.set()
        await asyncio.wait_for(scorer.cancel_cleanup_started.wait(), timeout=2)
        assert not task.done() and cleanup_active_counts == []
        if cancel_during_drain:
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not task.done() and cleanup_active_counts == []
            assert scorer.active == 1
        scorer.cancel_cleanup_release.set()
        with pytest.raises(RuntimeError, match="unexpected packet processing crash"):
            await asyncio.wait_for(task, timeout=2)
    finally:
        allow_crash.set()
        scorer.release_all()
        scorer.cancel_cleanup_release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert scorer.active == 0 and scorer.completed == []
    assert sorted(scorer.cancelled) == sorted(scorer.started)
    assert cleanup_active_counts == [0] * 5
    assert [dict(row) for row in case.conn.execute("SELECT * FROM miners ORDER BY uid")] == before_miners
    assert ScorePacketEvidence(case.conn).recent_packet_digests() == before_packets
    assert len(miner_manager.uncommitted_rounds(case.conn)) == 1


def shed(retry_after="1", *, status=503):
    request = httpx.Request("POST", "http://scoring.test/score")
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    response = httpx.Response(status, headers=headers, request=request)
    return httpx.HTTPStatusError("worker unavailable", request=request, response=response)


class SequenceScorer(FakeScoringClient):
    def __init__(self, outcomes):
        super().__init__()
        self.outcomes = list(outcomes)
        self.attempts = []

    async def score(self, request):
        self.attempts.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        return await super().score(request)


@pytest.fixture
def recorded_sleeps(monkeypatch):
    sleeps = []
    real_sleep = asyncio.sleep

    async def record_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(inference.asyncio, "sleep", record_sleep)
    return sleeps


async def test_worker_shed_once_waits_and_scores(scoring_round):
    scorer = SequenceScorer([shed(), None])
    case = scoring_round(scorer, count=1, scoring_shed_wait_seconds=1.5)
    report = await case.validator.run_round()
    assert report.scored == {1: 0.8} and report.scoring_failed == []
    assert len(scorer.attempts) == 2
    assert scorer.attempts[0] == scorer.attempts[1]


async def test_persistent_sheds_exhaust_wait_budget_without_zeroing(scoring_round):
    scorer = SequenceScorer([shed(), shed(), shed()])
    case = scoring_round(scorer, count=1, scoring_shed_wait_seconds=1.5)
    started = time.monotonic()
    report = await asyncio.wait_for(case.validator.run_round(), timeout=5)
    elapsed = time.monotonic() - started
    assert elapsed < 5
    assert report.scoring_failed == [1] and report.scored == {} and report.zeroed == {}
    assert len(scorer.attempts) == 2
    assert case.evidence == []
    assert case.challenge.resolves == [("ch-compression", "resolved")]


async def test_zero_shed_budget_does_not_wait_or_retry(scoring_round, recorded_sleeps):
    scorer = SequenceScorer([shed(), None])
    case = scoring_round(scorer, count=1, scoring_shed_wait_seconds=0)
    report = await case.validator.run_round()
    assert report.scoring_failed == [1] and report.scored == {}
    assert len(scorer.attempts) == 1 and recorded_sleeps == []


@pytest.mark.parametrize("retry_after, expected", [
    (None, 5), ("invalid", 5), ("0", 5), ("-1", 5), ("1.5", 5),
    ("Wed, 21 Oct 2030 07:28:00 GMT", 5), ("90", 30),
])
async def test_shed_header_defaults_and_cap(scoring_round, recorded_sleeps, retry_after, expected):
    scorer = SequenceScorer([shed(retry_after), None])
    case = scoring_round(scorer, count=1)
    report = await case.validator.run_round()
    assert report.scored == {1: 0.8} and report.scoring_failed == []
    assert len(scorer.attempts) == 2 and recorded_sleeps == [expected]


async def test_each_shed_retry_logs_uid_delay_and_prior_wait(scoring_round, recorded_sleeps, caplog):
    scorer = SequenceScorer([shed(), shed("2"), shed(), None])
    case = scoring_round(scorer, count=1, scoring_shed_wait_seconds=3)
    report = await case.validator.run_round()
    warnings = [record for record in caplog.records
        if record.getMessage() == "scoring worker shed the request; retrying"]
    assert [record.levelname for record in warnings] == ["WARNING", "WARNING"]
    assert [record.fields for record in warnings] == [
        {"uid": 1, "delay": 1, "waited_so_far": 0},
        {"uid": 1, "delay": 2, "waited_so_far": 1},
    ]
    assert recorded_sleeps == [1, 2]
    assert len(scorer.attempts) == 3 and report.scoring_failed == [1]
    assert "scoring worker failed; miner not accumulated this round" in caplog.text


async def test_sheds_do_not_consume_generic_attempts(scoring_round, recorded_sleeps):
    scorer = SequenceScorer([shed(), RuntimeError("transient"), shed(), None])
    case = scoring_round(scorer, count=1, scoring_shed_wait_seconds=2)
    report = await case.validator.run_round()
    assert report.scored == {1: 0.8} and report.scoring_failed == []
    assert len(scorer.attempts) == 4
    assert sum(delay == 1 for delay in recorded_sleeps) == 2


async def test_shed_wait_budget_is_not_reset_by_generic_retry(scoring_round, recorded_sleeps):
    scorer = SequenceScorer([shed(), RuntimeError("transient"), shed(), None])
    case = scoring_round(scorer, count=1, scoring_shed_wait_seconds=1.5)
    report = await case.validator.run_round()
    assert report.scoring_failed == [1] and report.scored == {} and report.zeroed == {}
    assert len(scorer.attempts) == 3
    assert sum(delay == 1 for delay in recorded_sleeps) == 1
    assert case.evidence == []


async def test_second_generic_error_still_fails_with_interleaved_shed(scoring_round, recorded_sleeps):
    scorer = SequenceScorer([RuntimeError("first"), shed(), RuntimeError("second"), None])
    case = scoring_round(scorer, count=1)
    report = await case.validator.run_round()
    assert report.scoring_failed == [1] and report.scored == {} and report.zeroed == {}
    assert len(scorer.attempts) == 3


async def test_other_http_error_keeps_two_attempt_policy(scoring_round, recorded_sleeps):
    scorer = SequenceScorer([shed(status=500), shed(status=500), None])
    case = scoring_round(scorer, count=1)
    report = await case.validator.run_round()
    assert report.scoring_failed == [1] and report.scored == {}
    assert len(scorer.attempts) == 2
    assert all(delay < 1 for delay in recorded_sleeps)


@pytest.mark.parametrize("cancel_during_drain", [False, True])
async def test_cancellation_stops_inflight_and_queued_scoring_before_artifact_cleanup(
    scoring_round, cancel_during_drain,
):
    scorer = EventScorer()
    scorer.cancel_cleanup_release = asyncio.Event()
    case = scoring_round(scorer)
    discard = case.validator._discard_miner_artifact
    cleanup_active_counts = []

    def check_cleanup(response):
        cleanup_active_counts.append(scorer.active)
        discard(response)

    case.validator._discard_miner_artifact = check_cleanup
    task = asyncio.create_task(case.validator.run_round())
    try:
        await asyncio.wait_for(scorer.saturated.wait(), timeout=2)
        task.cancel()
        await asyncio.wait_for(scorer.cancel_cleanup_started.wait(), timeout=2)
        assert not task.done()
        assert scorer.active == 1
        assert scorer.started == [1, 2, 3]
        assert cleanup_active_counts == []
        if cancel_during_drain:
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not task.done() and cleanup_active_counts == []
            assert scorer.active == 1
        scorer.cancel_cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        scorer.release_all()
        scorer.cancel_cleanup_release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert scorer.active == 0 and sorted(scorer.cancelled) == [1, 2, 3]
    assert scorer.started == [1, 2, 3] and scorer.completed == []
    assert cleanup_active_counts == [0] * 5
    assert ScorePacketEvidence(case.conn).packets() == []
    assert all(miner_manager.get_miner(case.conn, uid) is None for uid in range(1, 6))
