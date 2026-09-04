"""record_item_score is packet-bound: score/valid/digest derive from the
scorer's ItemScore packet bytes only — no caller-supplied score can reach the DB."""

import hashlib
import inspect
import json
import sqlite3
from datetime import timedelta

import pytest

from vidaio.competition import ScorePacketError
from vidaio.competition import repository as repo

from support import BASELINE, FINALIZATION, Driver, build_manifest, packet_bytes

RECORD_AT = FINALIZATION + timedelta(minutes=30)


def _to_scoring(driver: Driver, hotkeys: list[str] = ["hk-1", "hk-2"]):
    cid, ids, item_ids = driver.run_to_evaluating(build_manifest(baseline=BASELINE), hotkeys)
    driver.engine.mark_evaluation_complete(driver.conn, cid, RECORD_AT)
    return cid, ids, item_ids


def _row(conn, contender_id: int, item_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM performance_history WHERE contender_id = ? AND item_id = ?",
        (contender_id, item_id),
    ).fetchone()
    assert row is not None
    return row


def test_score_valid_and_digest_derive_from_packet_bytes(driver: Driver) -> None:
    cid, ids, item_ids = _to_scoring(driver)
    packet = driver.make_packet(ids["hk-1"], item_ids[0], 0.8, vmaf=93.5)
    repo.record_item_score(
        driver.conn,
        cid,
        contender_id=ids["hk-1"],
        item_id=item_ids[0],
        packet_bytes=packet,
        now=RECORD_AT,
    )
    row = _row(driver.conn, ids["hk-1"], item_ids[0])
    assert row["item_score"] == 0.8
    assert row["valid"] == 1
    assert row["score_packet_digest"] == hashlib.sha256(packet).hexdigest()
    assert row["vmaf"] == 93.5  # metrics come from the packet too


def test_gate_failed_packet_persists_zero(driver: Driver) -> None:
    cid, ids, item_ids = _to_scoring(driver)
    packet = driver.make_packet(ids["hk-1"], item_ids[0], 0.0, gate_passed=False)
    repo.record_item_score(
        driver.conn,
        cid,
        contender_id=ids["hk-1"],
        item_id=item_ids[0],
        packet_bytes=packet,
        now=RECORD_AT,
    )
    row = _row(driver.conn, ids["hk-1"], item_ids[0])
    assert row["valid"] == 0
    assert row["item_score"] == 0.0


def test_codex_bypass_gate_failed_with_nonzero_score_is_impossible(driver: Driver) -> None:
    # The integration "golden path" bypass: a gate-failed packet accompanied by a
    # freely chosen 0.35. There is no parameter to supply the score — and a packet
    # that itself violates gates-first (gate_passed=False, score=0.35) is rejected.
    cid, ids, item_ids = _to_scoring(driver)
    tampered = driver.make_packet(ids["hk-1"], item_ids[0], 0.35, gate_passed=False)
    with pytest.raises(ScorePacketError, match="gates-first"):
        repo.record_item_score(
            driver.conn,
            cid,
            contender_id=ids["hk-1"],
            item_id=item_ids[0],
            packet_bytes=tampered,
            now=RECORD_AT,
        )
    assert (
        driver.conn.execute(
            "SELECT COUNT(*) AS n FROM performance_history WHERE competition_id = ?", (cid,)
        ).fetchone()["n"]
        == 0
    )


def test_no_free_supply_parameters_exist() -> None:
    params = set(inspect.signature(repo.record_item_score).parameters)
    assert {"item_score", "valid", "score_packet_digest", "vmaf"}.isdisjoint(params)
    assert "packet_bytes" in params


def test_mismatched_hotkey_raises(driver: Driver) -> None:
    cid, ids, item_ids = _to_scoring(driver)
    # Packet minted for hk-2, recorded against hk-1's contender row.
    packet = driver.make_packet(ids["hk-2"], item_ids[0], 0.9)
    with pytest.raises(ScorePacketError, match="miner_hotkey"):
        repo.record_item_score(
            driver.conn,
            cid,
            contender_id=ids["hk-1"],
            item_id=item_ids[0],
            packet_bytes=packet,
            now=RECORD_AT,
        )


def test_calibration_contender_requires_hotkeyless_packet(driver: Driver) -> None:
    cid, ids, item_ids = _to_scoring(driver)
    baseline = next(c for c in repo.list_contenders(driver.conn, cid) if c.is_calibration)
    with pytest.raises(ScorePacketError, match="miner_hotkey"):
        repo.record_item_score(
            driver.conn,
            cid,
            contender_id=baseline.contender_id,
            item_id=item_ids[0],
            packet_bytes=driver.make_packet(ids["hk-1"], item_ids[0], 0.9),
            now=RECORD_AT,
        )
    # A hotkey-less packet records fine for the baseline.
    repo.record_item_score(
        driver.conn,
        cid,
        contender_id=baseline.contender_id,
        item_id=item_ids[0],
        packet_bytes=driver.make_packet(baseline.contender_id, item_ids[0], 0.9),
        now=RECORD_AT,
    )


def test_mismatched_item_identity_raises(driver: Driver) -> None:
    cid, ids, item_ids = _to_scoring(driver)
    # Packet minted for item 0, recorded against item 1.
    packet = driver.make_packet(ids["hk-1"], item_ids[0], 0.9)
    with pytest.raises(ScorePacketError, match="identity"):
        repo.record_item_score(
            driver.conn,
            cid,
            contender_id=ids["hk-1"],
            item_id=item_ids[1],
            packet_bytes=packet,
            now=RECORD_AT,
        )
    # Wrong challenge id.
    challenge_id, scoring_item_id = driver.item_identity(item_ids[0])
    packet = packet_bytes(
        challenge_id="chal-other",
        scoring_item_id=scoring_item_id,
        miner_hotkey="hk-1",
        score=0.9,
    )
    with pytest.raises(ScorePacketError, match="identity"):
        repo.record_item_score(
            driver.conn,
            cid,
            contender_id=ids["hk-1"],
            item_id=item_ids[0],
            packet_bytes=packet,
            now=RECORD_AT,
        )


def test_unknown_contender_or_item_raises(driver: Driver) -> None:
    cid, ids, item_ids = _to_scoring(driver)
    packet = driver.make_packet(ids["hk-1"], item_ids[0], 0.9)
    with pytest.raises(ScorePacketError, match="contender"):
        repo.record_item_score(
            driver.conn,
            cid,
            contender_id=999_999,
            item_id=item_ids[0],
            packet_bytes=packet,
            now=RECORD_AT,
        )
    with pytest.raises(ScorePacketError, match="item"):
        repo.record_item_score(
            driver.conn,
            cid,
            contender_id=ids["hk-1"],
            item_id=999_999,
            packet_bytes=packet,
            now=RECORD_AT,
        )


def test_non_finite_or_out_of_range_score_is_rejected(driver: Driver) -> None:
    """review round-2: a gate-passing Infinity packet persisted in a probe. The
    packet score must be a FINITE float in [0, 1] — Infinity/NaN and out-of-range
    values are rejected at the persistence boundary and can never reach ranking."""
    cid, ids, item_ids = _to_scoring(driver)
    for bad in (float("inf"), float("-inf"), float("nan"), 1.5, -0.1):
        packet = driver.make_packet(ids["hk-1"], item_ids[0], bad)
        with pytest.raises(ScorePacketError):
            repo.record_item_score(
                driver.conn,
                cid,
                contender_id=ids["hk-1"],
                item_id=item_ids[0],
                packet_bytes=packet,
                now=RECORD_AT,
            )
    # NaN slipped with gate_passed=False is equally unrepresentable.
    with pytest.raises(ScorePacketError):
        repo.record_item_score(
            driver.conn,
            cid,
            contender_id=ids["hk-1"],
            item_id=item_ids[0],
            packet_bytes=driver.make_packet(
                ids["hk-1"], item_ids[0], float("nan"), gate_passed=False
            ),
            now=RECORD_AT,
        )
    assert (
        driver.conn.execute(
            "SELECT COUNT(*) AS n FROM performance_history WHERE competition_id = ?", (cid,)
        ).fetchone()["n"]
        == 0
    )


def test_unparseable_packet_raises(driver: Driver) -> None:
    cid, ids, item_ids = _to_scoring(driver)
    for bad in (b"not json", b"{}", json.dumps({"item_id": "x"}).encode()):
        with pytest.raises(ScorePacketError, match="unparseable"):
            repo.record_item_score(
                driver.conn,
                cid,
                contender_id=ids["hk-1"],
                item_id=item_ids[0],
                packet_bytes=bad,
                now=RECORD_AT,
            )
