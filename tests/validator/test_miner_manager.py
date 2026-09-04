"""Registry sync, hotkey purge, EWMA parity, track integrity."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from vidaio.tokenomics import EXCLUDED_SCORE, accumulate
from vidaio.validator import miner_manager

from validator_support import mk_neuron

NOW = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
DECAY = 0.75


def test_sync_inserts_and_updates(conn):
    n = mk_neuron(1)
    assert miner_manager.sync_neurons(conn, [n], block=10) == []
    row = miner_manager.get_miner(conn, 1)
    assert row["hotkey"] == "hk1" and row["first_seen_block"] == 10
    assert row["track"] is None and row["accumulate_score"] == 0.0
    # ip/coldkey churn without a hotkey change is an update, not a purge
    moved = mk_neuron(1, ip="10.9.9.9")
    assert miner_manager.sync_neurons(conn, [moved], block=20) == []
    row = miner_manager.get_miner(conn, 1)
    assert row["ip"] == "10.9.9.9" and row["first_seen_block"] == 10


def test_hotkey_change_purges_score_and_track(conn):
    n = mk_neuron(1)
    miner_manager.sync_neurons(conn, [n], block=10)
    miner_manager.record_track(conn, 1, "compression")
    miner_manager.apply_scores(conn, {1: 0.8}, DECAY)

    rotated = mk_neuron(1, hotkey="hk1-new")
    assert miner_manager.sync_neurons(conn, [rotated], block=50) == [1]
    row = miner_manager.get_miner(conn, 1)
    assert row["hotkey"] == "hk1-new"
    assert row["accumulate_score"] == 0.0  # no phantom history carries over
    assert row["track"] is None  # must be re-probed — never assumed
    assert row["first_seen_block"] == 50


# (test_retention_accumulates_and_full_window_flips / test_retention_skipped_window_is_not_complete
# were REMOVED with the retention-window bookkeeping — retention removed — owner decision;
# an internal review.)


def test_apply_scores_matches_tokenomics_accumulate(conn):
    miner_manager.sync_neurons(conn, [mk_neuron(1)], block=1)
    expected = 0.0
    for score in (0.8, 0.4, 0.0, 0.9):
        miner_manager.apply_scores(conn, {1: score}, DECAY)
        expected = accumulate(expected, score, DECAY)
        assert miner_manager.get_miner(conn, 1)["accumulate_score"] == expected
    # unknown uid is ignored, not crashed
    miner_manager.apply_scores(conn, {999: 0.5}, DECAY)


def test_exclusion_sentinel_latches_and_restarts(conn):
    miner_manager.sync_neurons(conn, [mk_neuron(1)], block=1)
    miner_manager.apply_scores(conn, {1: 0.8}, DECAY)
    miner_manager.apply_scores(conn, {1: EXCLUDED_SCORE}, DECAY)
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == EXCLUDED_SCORE
    # snapshot passes the sentinel through untouched for rank_curve to gate on
    miner_manager.record_track(conn, 1, "compression")
    snap = miner_manager.snapshot(conn, [mk_neuron(1)], NOW)[0]
    assert snap.accumulate_score == EXCLUDED_SCORE
    # a genuine score after exclusion restarts from 0.0, not from phantom history
    miner_manager.apply_scores(conn, {1: 0.4}, DECAY)
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == accumulate(
        EXCLUDED_SCORE, 0.4, DECAY
    )


def test_snapshot_excludes_unknown_track_and_stale_hotkeys(conn):
    miner_manager.sync_neurons(conn, [mk_neuron(1), mk_neuron(2)], block=1)
    miner_manager.record_track(conn, 1, "upscaling")
    # uid 2 has no warrant record -> absent from the tokenomics snapshot
    neurons = [mk_neuron(1), mk_neuron(2)]
    snaps = miner_manager.snapshot(conn, neurons, NOW)
    assert [s.uid for s in snaps] == [1]
    assert snaps[0].track == "upscaling"
    # a chain-side hotkey change not yet synced is excluded too (identity mismatch)
    snaps = miner_manager.snapshot(conn, [mk_neuron(1, hotkey="rotated")], NOW)
    assert snaps == []


def test_snapshot_at_uses_close_block_state_not_live_head(conn):
    neuron = mk_neuron(1)
    t1 = "2026-08-20T12:00:00+00:00"
    t2 = "2026-08-20T12:05:00+00:00"

    miner_manager.begin_round(conn, "r1", 10, t1)
    miner_manager.commit_round(
        conn,
        "r1",
        scores={1: 0.8},
        decay=DECAY,
        committed_at=t1,
        registry=miner_manager.RegistryUpdate(
            neurons=(neuron,), block=10, tracks={1: "compression"}
        ),
    )
    score_at_t1 = accumulate(0.0, 0.8, DECAY)

    miner_manager.begin_round(conn, "r2", 20, t2)
    miner_manager.commit_round(
        conn,
        "r2",
        scores={1: 0.4},
        decay=DECAY,
        committed_at=t2,
        registry=miner_manager.RegistryUpdate(neurons=(neuron,), block=20),
    )

    assert miner_manager.snapshot(conn, [neuron], NOW)[0].accumulate_score != pytest.approx(
        score_at_t1
    )
    pinned = miner_manager.snapshot_at(conn, [neuron], 15, NOW)
    assert len(pinned) == 1
    assert pinned[0].accumulate_score == pytest.approx(score_at_t1)


def test_snapshot_at_rejects_a_recycled_hotkey(conn):
    original = mk_neuron(1)
    miner_manager.begin_round(conn, "r1", 10, NOW.isoformat())
    miner_manager.commit_round(
        conn,
        "r1",
        scores={1: 0.8},
        decay=DECAY,
        committed_at=NOW.isoformat(),
        registry=miner_manager.RegistryUpdate(
            neurons=(original,), block=10, tracks={1: "compression"}
        ),
    )

    assert miner_manager.snapshot_at(
        conn, [mk_neuron(1, hotkey="replacement")], 10, NOW
    ) == []


def test_track_integrity(conn):
    miner_manager.sync_neurons(conn, [mk_neuron(1)], block=1)
    with pytest.raises(ValueError):
        miner_manager.record_track(conn, 1, "definitely-not-a-track")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE miners SET track = 'garbage' WHERE uid = 1")
    assert miner_manager.normalize_track("garbage") is None
    assert miner_manager.normalize_track(None) is None
    assert miner_manager.normalize_track("compression") == "compression"
    assert miner_manager.track_of(conn, 1) is None
    assert miner_manager.track_of(conn, 404) is None
