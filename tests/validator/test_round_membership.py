"""Completion membership, real-head fencing, and atomic capture under contention."""
from datetime import datetime, timezone
from pathlib import Path
import threading

import pytest

from vidaio.core import apply_migrations, connect
from vidaio.validator import MIGRATIONS_DIR, miner_manager
from vidaio.validator.evidence import AvailabilityFoldEvidence, ScorePacketEvidence
from vidaio.validator.round_membership import EpochCaptureUnavailable, RoundCommitDeferred, capture_epoch_inputs
from validator_support import mk_neuron

NOW = "2026-09-07T01:00:00+00:00"


def packet(uid=1):
    return dict(uid=uid, item_id="c:1", challenge_id="c", track="compression", miner_hotkey=f"hk{uid}",
                content_digest="a" * 64, packet_digest="b" * 64, packet_json="{}", scorer_version="test", score=.8)


def complete(conn, round_id="r", assigned=10, committed=21, **kwargs):
    miner_manager.begin_round(conn, round_id, assigned, NOW)
    miner_manager.commit_round(conn, round_id, scores={1: .8}, decay=.75, committed_at=NOW,
        registry=miner_manager.RegistryUpdate((mk_neuron(1),), assigned, {1: "compression"}),
        commit_block=committed, **kwargs)


def capture(conn, head=20, close=20, prior=0, **kwargs):
    return capture_epoch_inputs(conn, prior_close_block=prior, close_block=close,
        chain_neurons=[mk_neuron(1)], read_best_head=lambda: head, **kwargs)


def test_straddle_lands_in_next_window_and_snapshot_uses_completion(conn):
    complete(conn, packets=[packet()], challenges=[dict(challenge_id="c", track="compression", anchor_block=10, ordering_key=1)])
    evidence = ScorePacketEvidence(conn)
    assert evidence.packets(through_block=20) == []
    assert evidence.packets(through_block=20, membership="assignment")[0]["round_id"] == "r"
    assert miner_manager.snapshot_at(conn, [mk_neuron(1)], 20, datetime.now(timezone.utc)) == []
    assert len(miner_manager.snapshot_at(conn, [mk_neuron(1)], 20, datetime.now(timezone.utc), membership="assignment")) == 1
    before = capture(conn)
    after = capture(conn, head=30, close=30, prior=20)
    assert not before.packets and not before.snapshots and not before.round_commits
    assert after.packets[0]["commit_block"] == 21
    assert after.round_commits[0]["commit_block"] == 21
    assert after.snapshots[0].accumulate_score == .2
    assert evidence.packets(after_block=21, through_block=30) == []


def test_empty_open_round_does_not_block_close_and_head_equal_close_seals(conn):
    miner_manager.begin_round(conn, "open", 1, NOW)
    result = capture(conn)
    assert not result.round_commits
    assert conn.execute("SELECT sealed_close FROM round_seal").fetchone()[0] == 20
    with pytest.raises(RoundCommitDeferred):
        miner_manager.commit_round(conn, "open", scores={}, decay=.75, committed_at=NOW, read_best_head=lambda: 20)
    assert not conn.in_transaction
    miner_manager.commit_round(conn, "open", scores={}, decay=.75, committed_at=NOW, read_best_head=lambda: 21)
    assert conn.execute("SELECT commit_block FROM rounds").fetchone()[0] == 21


@pytest.mark.parametrize("head,limit", [(19, None), (25, 25), (26, 25)])
def test_stale_head_or_closed_anchor_window_never_seals(conn, head, limit):
    with pytest.raises(EpochCaptureUnavailable):
        capture(conn, head=head, max_head=limit)
    assert conn.execute("SELECT sealed_close FROM round_seal").fetchone()[0] == -1
    assert not conn.in_transaction


def test_capture_failure_and_missing_watermark_are_atomic(conn):
    def fail():
        assert conn.in_transaction
        raise OSError("offline test rpc failure")
    with pytest.raises(OSError):
        capture_epoch_inputs(conn, prior_close_block=None, close_block=20, chain_neurons=[], read_best_head=fail)
    assert conn.execute("SELECT sealed_close FROM round_seal").fetchone()[0] == -1
    conn.execute("DELETE FROM round_seal")
    with pytest.raises(EpochCaptureUnavailable, match="watermark"):
        capture(conn)


def test_watermark_survives_reopen_and_rejects_regressed_capture(tmp_path):
    path = tmp_path / "ledger.db"
    conn = connect(path); apply_migrations(conn, MIGRATIONS_DIR)
    capture(conn)
    conn.close()
    conn = connect(path)
    with pytest.raises(EpochCaptureUnavailable, match="regresses"):
        capture(conn, close=19, head=30)
    assert conn.execute("SELECT sealed_close FROM round_seal").fetchone()[0] == 20
    conn.close()


def test_old_writer_cannot_publish_unstamped_completion_and_stamp_is_immutable(conn):
    import sqlite3
    miner_manager.begin_round(conn, "open", 10, NOW)
    with pytest.raises(sqlite3.IntegrityError, match="requires commit_block"):
        conn.execute("UPDATE rounds SET committed_at=? WHERE round_id='open'", (NOW,))
    miner_manager.commit_round(conn, "open", scores={}, decay=.75, committed_at=NOW, commit_block=21)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE rounds SET commit_block=22 WHERE round_id='open'")


@pytest.mark.parametrize("head", [True, -1, 10.5, "21"])
def test_writer_refuses_non_height_and_never_mutates(conn, head):
    miner_manager.begin_round(conn, "r", 10, NOW)
    with pytest.raises(ValueError):
        miner_manager.commit_round(conn, "r", scores={}, decay=.75, committed_at=NOW, read_best_head=lambda: head)
    assert conn.execute("SELECT committed_at FROM rounds").fetchone()[0] is None
    assert not conn.in_transaction


def test_writer_requires_explicit_completion_and_rejects_head_before_anchor(conn):
    miner_manager.begin_round(conn, "r", 10, NOW)
    with pytest.raises(ValueError, match="exactly one"):
        miner_manager.commit_round(conn, "r", scores={}, decay=.75, committed_at=NOW)
    with pytest.raises(RoundCommitDeferred):
        miner_manager.commit_round(conn, "r", scores={}, decay=.75, committed_at=NOW, commit_block=15,
            challenges=[dict(challenge_id="c", track="compression", anchor_block=16, ordering_key=1)])
    assert conn.execute("SELECT count(*) FROM round_challenges").fetchone()[0] == 0


def test_all_skip_context_is_atomic_and_reader_returns_it(conn):
    complete(conn, challenges=[dict(challenge_id="c", track="compression", anchor_block=10, ordering_key=1)])
    result = capture(conn, close=30, head=30)
    assert result.packets == result.availability == result.content_rounds == ()
    assert result.round_commits == ({"round_id": "r", "challenge_id": "c", "track": "compression",
                                    "anchor_block": 10, "ordering_key": 1, "commit_block": 21},)


def test_failed_evidence_rolls_back_height_context_and_private_state(conn):
    miner_manager.begin_round(conn, "r", 10, NOW)
    bad = packet(); bad.pop("score")
    with pytest.raises(KeyError):
        miner_manager.commit_round(conn, "r", scores={1: .8}, decay=.75, committed_at=NOW, commit_block=21,
            registry=miner_manager.RegistryUpdate((mk_neuron(1),), 10, {1: "compression"}), packets=[bad],
            challenges=[dict(challenge_id="c", track="compression", anchor_block=10, ordering_key=1)])
    row = conn.execute("SELECT committed_at,commit_block FROM rounds").fetchone()
    assert tuple(row) == (None, None)
    assert miner_manager.get_miner(conn, 1) is None
    assert conn.execute("SELECT count(*) FROM round_challenges").fetchone()[0] == 0


def test_regressed_head_cannot_move_later_dispatch_before_prior_completion(conn):
    complete(conn, challenges=[dict(challenge_id="c2", track="compression", anchor_block=10, ordering_key=2)])
    conn.execute("UPDATE round_seal SET sealed_close=19")
    miner_manager.begin_round(conn, "later", 10, NOW)
    kwargs = dict(scores={}, decay=.75, committed_at=NOW,
                  challenges=[dict(challenge_id="c3", track="compression", anchor_block=10, ordering_key=3)])
    with pytest.raises(RoundCommitDeferred):
        miner_manager.commit_round(conn, "later", read_best_head=lambda: 20, **kwargs)
    assert conn.execute("SELECT commit_block FROM rounds WHERE round_id='later'").fetchone()[0] is None
    assert conn.execute("SELECT count(*) FROM round_challenges").fetchone()[0] == 1
    miner_manager.commit_round(conn, "later", read_best_head=lambda: 21, **kwargs)
    assert [(r["ordering_key"], r["commit_block"]) for r in ScorePacketEvidence(conn).round_commits()] == [(2, 21), (3, 21)]


def test_availability_uses_identical_strict_completion_window(conn):
    observation = dict(uid=1, item_id="c:1", challenge_id="c", track="compression", miner_hotkey="hk1",
                       endpoint="https://example.invalid", reason="transport_error", observation_digest="c" * 64,
                       observation_json="{}")
    complete(conn, availability_observations=[observation])
    reader = AvailabilityFoldEvidence(conn)
    assert reader.observations(through_block=20) == []
    assert reader.observations(after_block=20, through_block=21)[0]["commit_block"] == 21
    assert reader.observations(after_block=21, through_block=30) == []


def test_fence_blocks_waiting_writer_then_requires_new_real_height(tmp_path):
    path = tmp_path / "ledger.db"
    primary = connect(path); apply_migrations(primary, MIGRATIONS_DIR)
    miner_manager.begin_round(primary, "waiting", 10, NOW)
    waiting, observed = threading.Event(), threading.Event()
    outcome = []

    def writer():
        secondary = connect(path)
        waiting.set()
        def head():
            observed.set()
            assert secondary.in_transaction
            return 20
        try:
            miner_manager.commit_round(secondary, "waiting", scores={}, decay=.75, committed_at=NOW, read_best_head=head)
        except RoundCommitDeferred:
            outcome.append("deferred")
        finally:
            secondary.close()

    thread = threading.Thread(target=writer)
    def sealing_head():
        assert primary.in_transaction
        thread.start()
        assert waiting.wait(2)
        assert not observed.wait(.05)
        return 20
    capture_epoch_inputs(primary, prior_close_block=0, close_block=20, chain_neurons=[], read_best_head=sealing_head)
    thread.join(3)
    assert not thread.is_alive() and outcome == ["deferred"]
    miner_manager.commit_round(primary, "waiting", scores={}, decay=.75, committed_at=NOW, read_best_head=lambda: 21)
    assert primary.execute("SELECT commit_block FROM rounds").fetchone()[0] == 21
    primary.close()


def test_writer_lock_timeout_defers_only_before_any_transaction_or_head_read(tmp_path):
    path = tmp_path / "ledger.db"
    primary = connect(path); apply_migrations(primary, MIGRATIONS_DIR)
    miner_manager.begin_round(primary, "waiting", 10, NOW)
    secondary = connect(path)
    secondary.execute("PRAGMA busy_timeout=1")
    reads = []
    def head():
        reads.append(True)
        return 21
    with miner_manager.transaction(primary):
        with pytest.raises(RoundCommitDeferred, match="exclusion"):
            miner_manager.commit_round(secondary, "waiting", scores={}, decay=.75,
                committed_at=NOW, read_best_head=head)
        assert not secondary.in_transaction and reads == []
    miner_manager.commit_round(secondary, "waiting", scores={}, decay=.75, committed_at=NOW, read_best_head=head)
    assert reads == [True]
    primary.close(); secondary.close()


def test_migration_backfills_only_preexisting_completed_assignment(tmp_path):
    historical = tmp_path / "migrations"
    historical.mkdir()
    for path in sorted(Path(MIGRATIONS_DIR).glob("*.sql")):
        if path.name < "0012":
            (historical / path.name).write_bytes(path.read_bytes())
    conn = connect(":memory:")
    apply_migrations(conn, historical)
    conn.execute("INSERT INTO rounds VALUES ('old',?,10,?)", (NOW, NOW))
    conn.execute("INSERT INTO rounds VALUES ('open',?,20,NULL)", (NOW,))
    apply_migrations(conn, MIGRATIONS_DIR)
    assert dict(conn.execute("SELECT round_id,commit_block FROM rounds")) == {"old": 10, "open": None}
    assert conn.execute("SELECT sealed_close FROM round_seal").fetchone()[0] == -1
    conn.close()


async def test_real_round_samples_completion_after_work_inside_sql(validator, chain, miner_client, conn, monkeypatch):
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    reads = []
    def fresh():
        reads.append(conn.in_transaction)
        return 2
    monkeypatch.setattr(chain, "best_head_block", fresh)
    report = await validator.run_round()
    row = conn.execute("SELECT block,commit_block FROM rounds WHERE round_id=?", (report.round_id,)).fetchone()
    assert tuple(row) == (1, 2) and reads == [True]
    assert ScorePacketEvidence(conn).packets(through_block=1) == []
    assert len(ScorePacketEvidence(conn).packets(after_block=1, through_block=2)) == 1


async def test_real_round_defers_outside_sql_without_redispatch(validator, chain, miner_client, challenge_client,
                                                              conn, monkeypatch):
    import asyncio
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    conn.execute("UPDATE round_seal SET sealed_close=1")
    real_sleep = asyncio.sleep
    waits = []
    async def next_head(delay):
        waits.append(delay)
        assert not conn.in_transaction
        chain.advance_blocks(1)
        await real_sleep(0)
    monkeypatch.setattr(asyncio, "sleep", next_head)
    report = await validator.run_round()
    assert waits == [1.0]
    assert challenge_client.fetches == ["compression"]
    assert conn.execute("SELECT commit_block FROM rounds WHERE round_id=?", (report.round_id,)).fetchone()[0] == 2
    assert len(ScorePacketEvidence(conn).packets()) == 1
