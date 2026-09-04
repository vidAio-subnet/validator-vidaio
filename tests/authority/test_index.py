"""EpochIndex — append-only, immutable pointer index (record/anchor/latest/get)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vidaio.audit.canonical import sha256_hex
from vidaio.authority import EpochIndex, EpochIndexConflict
from vidaio.authority.finalizer import FinalizedEpoch, epoch_prefix
from vidaio.audit.store import set_member_key
from vidaio.epoch import AuditManifest, EpochLog, weight_vector_digest


def _finalized(epoch_id: int, *, digest_seed: str = "a") -> FinalizedEpoch:
    """A minimal, VALID FinalizedEpoch pointer (burn-vector log keeps it self-contained)."""
    weight_u16 = {0: 65535}
    log = EpochLog(
        epoch_id=epoch_id,
        close_block=epoch_id * 360 + 359,
        scorer_version="scoring-1+abc",
        created_at=__import__("datetime").datetime(2026, 8, 20, tzinfo=__import__("datetime").timezone.utc),
        burn_uid=0,
        miners=(),
        # This index fixture represents a genuine empty/burn epoch.
        weight_shares={0: 1.0},
        weight_u16=weight_u16,
        weight_vector_digest=weight_vector_digest(weight_u16),
        audit_manifest=AuditManifest(),
    )
    data = log.to_json()
    prefix = epoch_prefix(epoch_id)
    return FinalizedEpoch(
        epoch_id=epoch_id,
        close_block=log.close_block,
        snapshot_key=set_member_key(prefix, "log.json"),
        log_digest=sha256_hex(data),
        weight_vector_digest=log.weight_vector_digest,
        log=log,
    )


@pytest.fixture
def index(tmp_path: Path) -> EpochIndex:
    return EpochIndex.open(tmp_path / "authority.db")


def test_record_and_get_roundtrip(index: EpochIndex) -> None:
    fin = _finalized(41822)
    rec = index.record_finalized(fin, finalized_at="2026-08-20T12:00:00+00:00")
    assert rec.epoch_id == 41822
    assert rec.log_digest == fin.log_digest
    assert not rec.anchored
    got = index.get(41822)
    assert got == rec


def test_get_unknown_is_none(index: EpochIndex) -> None:
    assert index.get(123) is None
    assert index.latest() is None


def test_latest_is_highest_epoch_id(index: EpochIndex) -> None:
    for eid in (10, 15, 12):
        index.record_finalized(_finalized(eid), finalized_at="2026-08-20T12:00:00+00:00")
    assert index.latest().epoch_id == 15


def test_record_is_idempotent_same_digest(index: EpochIndex) -> None:
    fin = _finalized(7)
    a = index.record_finalized(fin, finalized_at="2026-08-20T12:00:00+00:00")
    b = index.record_finalized(fin, finalized_at="2026-08-20T13:00:00+00:00")  # later ts ignored
    assert a == b
    assert a.finalized_at == "2026-08-20T12:00:00+00:00"


def test_record_conflict_on_divergent_digest(index: EpochIndex) -> None:
    index.record_finalized(_finalized(7), finalized_at="2026-08-20T12:00:00+00:00")
    other = _finalized(8)  # different log_digest
    conflicting = other.model_copy(update={"epoch_id": 7})
    with pytest.raises(EpochIndexConflict, match="immutable"):
        index.record_finalized(conflicting, finalized_at="2026-08-20T12:00:00+00:00")


def test_set_anchor_then_read(index: EpochIndex) -> None:
    index.record_finalized(_finalized(7), finalized_at="2026-08-20T12:00:00+00:00")
    rec = index.set_anchor(7, txid="0xdeadbeef", block=999)
    assert rec.anchored and rec.anchor_txid == "0xdeadbeef" and rec.anchor_block == 999
    assert index.get(7).anchor_txid == "0xdeadbeef"


def test_set_anchor_idempotent_same_txid(index: EpochIndex) -> None:
    index.record_finalized(_finalized(7), finalized_at="2026-08-20T12:00:00+00:00")
    index.set_anchor(7, txid="0xabc", block=1)
    again = index.set_anchor(7, txid="0xabc", block=1)
    assert again.anchor_txid == "0xabc"


def test_set_anchor_conflict_on_different_txid(index: EpochIndex) -> None:
    index.record_finalized(_finalized(7), finalized_at="2026-08-20T12:00:00+00:00")
    index.set_anchor(7, txid="0xabc", block=1)
    with pytest.raises(EpochIndexConflict, match="already anchored"):
        index.set_anchor(7, txid="0xdifferent", block=2)


def test_set_anchor_requires_finalized_epoch(index: EpochIndex) -> None:
    with pytest.raises(EpochIndexConflict, match="not finalized"):
        index.set_anchor(404, txid="0xabc", block=1)


# --------------------------------------------------------------------------------------
# P1.5/v16 gap tombstones — the named remediation for an unanchorable orphaned epoch.
# --------------------------------------------------------------------------------------


def test_tombstoned_epoch_disappears_from_get_and_latest(index: EpochIndex) -> None:
    index.record_finalized(_finalized(10), finalized_at="t10")
    index.record_finalized(_finalized(11), finalized_at="t11")
    index.set_anchor(10, txid="0x" + "aa" * 16)
    # 11 crashed before anchoring and its window elapsed: acknowledge the gap.
    index.mark_gap_tombstone(11, acknowledged_at="t12", reason="outage")
    assert index.get(11) is None
    latest = index.latest()
    assert latest is not None and latest.epoch_id == 10
    # idempotent
    index.mark_gap_tombstone(11, acknowledged_at="t13", reason="outage")


def test_tombstoning_an_anchored_epoch_is_refused(index: EpochIndex) -> None:
    import sqlite3

    index.record_finalized(_finalized(10), finalized_at="t10")
    index.set_anchor(10, txid="0x" + "aa" * 16)
    with pytest.raises(sqlite3.IntegrityError, match="ANCHORED"):
        index.mark_gap_tombstone(10, acknowledged_at="t11", reason="nope")


def test_tombstoning_an_unknown_epoch_is_refused(index: EpochIndex) -> None:
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError, match="not indexed"):
        index.mark_gap_tombstone(99, acknowledged_at="t", reason="nope")


def test_a_gap_epoch_can_never_be_refinalized(index: EpochIndex) -> None:
    index.record_finalized(_finalized(11), finalized_at="t11")
    index.mark_gap_tombstone(11, acknowledged_at="t12", reason="outage")
    with pytest.raises(EpochIndexConflict, match="tombstoned"):
        index.record_finalized(_finalized(11), finalized_at="t13")
