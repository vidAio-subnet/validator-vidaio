"""The Scoring Authority's epoch INDEX — which epochs are finalized (thin pointers).

The authority tracks, per finalized epoch, the POINTER a validator needs: the
object-store `snapshot_key`, the `log_digest` (== sha256 of the mirrored bytes ==
the on-chain anchored digest), the `weight_vector_digest`, and — once anchored —
the anchor txid/block. It never holds the epoch-log bytes; those live in the object
store (the project design record §3.1, build-wave 4).

Append-only + immutable: `record_finalized` writes a row once; a finalized epoch's
pointer fields can never change (idempotent re-finalize is a NO-OP that returns the
same row; a re-finalize with a DIFFERENT digest is a conflict and raises). `set_anchor`
fills the anchor columns once, after the digest is anchored on chain (idempotent for
the same txid). The in-database triggers (migration 0001) enforce this even against
direct SQL.

The separate submission journal retains signed authority extrinsics before
dispatch. Its evidence is immutable; one terminal outcome may be added later.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vidaio.audit.canonical import SHA256_HEX_PATTERN
from vidaio.authority.finalizer import FinalizedEpoch
from vidaio.core import apply_migrations, connect

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class EpochIndexConflict(ValueError):
    """A record contradicts an already-finalized (immutable) epoch.

    Raised when `record_finalized` is called for an epoch that is already indexed
    with a DIFFERENT log_digest / snapshot_key (a finalized set is immutable, so a
    second, divergent finalization is a bug, never a silent overwrite), or when
    `set_anchor` tries to re-anchor an epoch with a different txid.
    """


class EpochRecord(BaseModel):
    """One indexed epoch: the thin pointer + (optional) on-chain anchor."""

    model_config = ConfigDict(frozen=True)

    epoch_id: int
    close_block: int
    snapshot_key: str
    log_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    weight_vector_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    anchor_txid: str | None = None
    anchor_block: int | None = None
    finalized_at: str

    @property
    def anchored(self) -> bool:
        return self.anchor_txid is not None


class AnchorSubmissionRecord(BaseModel):
    """Immutable signed dispatch evidence and its optional terminal disposition."""

    model_config = ConfigDict(frozen=True)

    extrinsic_hash: str
    epoch_id: int
    netuid: int
    log_digest: str
    submission: dict[str, Any]
    outcome: Literal["confirmed", "expired_absent", "operator_acknowledged"] | None = None
    outcome_at: str | None = None
    outcome_evidence: dict[str, Any] | None = None


def _row_to_submission(row: sqlite3.Row) -> AnchorSubmissionRecord:
    return AnchorSubmissionRecord(
        extrinsic_hash=row["extrinsic_hash"],
        epoch_id=row["epoch_id"],
        netuid=row["netuid"],
        log_digest=row["log_digest"],
        submission=json.loads(row["submission_json"]),
        outcome=row["outcome"],
        outcome_at=row["outcome_at"],
        outcome_evidence=(json.loads(row["outcome_evidence"])
                          if row["outcome_evidence"] is not None else None),
    )


def _row_to_record(row: sqlite3.Row) -> EpochRecord:
    return EpochRecord(
        epoch_id=row["epoch_id"],
        close_block=row["close_block"],
        snapshot_key=row["snapshot_key"],
        log_digest=row["log_digest"],
        weight_vector_digest=row["weight_vector_digest"],
        anchor_txid=row["anchor_txid"],
        anchor_block=row["anchor_block"],
        finalized_at=row["finalized_at"],
    )


class EpochIndex:
    """SQLite index of finalized epochs (own migrations, append-only).

    Constructed over a connection this class owns (`open`) or one injected for
    tests. Every finalized epoch becomes one immutable row; the anchor columns are
    the only fields that transition (NULL -> value, once).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        apply_migrations(conn, MIGRATIONS_DIR)

    @classmethod
    def open(cls, db_path: str | Path) -> "EpochIndex":
        return cls(connect(db_path))

    def close(self) -> None:
        self._conn.close()

    def _durable_execute(self, sql: str, parameters: tuple[Any, ...]) -> None:
        """Fsync the WAL before dispatch or releasing a durable writer fence."""
        if self._conn.in_transaction:
            raise EpochIndexConflict("anchor evidence requires its own durable commit")
        synchronous = int(self._conn.execute("PRAGMA synchronous").fetchone()[0])
        self._conn.execute("PRAGMA synchronous=FULL")
        try:
            self._conn.execute(sql, parameters)
        finally:
            self._conn.execute(f"PRAGMA synchronous={synchronous}")

    def record_anchor_submission(
        self, epoch_id: int, *, netuid: int, log_digest: str,
        submission: dict[str, Any],
    ) -> AnchorSubmissionRecord:
        """Durably reserve one signed hash before it may be sent, never replace it."""
        txid = submission.get("extrinsic_hash")
        if not isinstance(txid, str) or re.fullmatch(r"0x[0-9a-f]{64}", txid) is None:
            raise ValueError("anchor submission hash must be canonical 32-byte hex")
        encoded = json.dumps(submission, sort_keys=True, separators=(",", ":"), allow_nan=False)
        existing = self.anchor_submission_by_hash(txid)
        if existing is not None:
            if (existing.epoch_id != epoch_id or existing.netuid != netuid
                    or existing.log_digest != log_digest or existing.submission != submission):
                raise EpochIndexConflict("anchor hash already binds different durable evidence")
            return existing
        previous = self.get_anchor_submission(epoch_id)
        if previous is not None and previous.outcome != "expired_absent":
            raise EpochIndexConflict("refusing a second authority submission without proven expiry")
        self._durable_execute(
            "INSERT INTO authority_anchor_submissions"
            " (extrinsic_hash,epoch_id,netuid,log_digest,submission_json) VALUES (?,?,?,?,?)",
            (txid, epoch_id, netuid, log_digest, encoded),
        )
        recorded = self.anchor_submission_by_hash(txid)
        assert recorded is not None
        return recorded

    def finish_anchor_submission(
        self, extrinsic_hash: str, *,
        outcome: Literal["confirmed", "expired_absent", "operator_acknowledged"],
        outcome_at: str, evidence: dict[str, Any],
    ) -> AnchorSubmissionRecord:
        """Append a terminal disposition once, retaining the original signed bytes."""
        record = self.anchor_submission_by_hash(extrinsic_hash)
        if record is None:
            raise EpochIndexConflict("cannot finish an unknown anchor submission")
        if record.outcome is not None:
            if record.outcome != outcome or record.outcome_evidence != evidence:
                raise EpochIndexConflict("anchor submission already has a different terminal outcome")
            return record
        if outcome == "confirmed":
            epoch = self.get(record.epoch_id)
            if (epoch is None or epoch.anchor_txid != extrinsic_hash
                    or epoch.anchor_block != evidence.get("block") or epoch.anchor_block is None):
                raise EpochIndexConflict("confirmed outcome requires the exact durably indexed anchor")
        elif outcome == "expired_absent":
            era = record.submission.get("mortal_era")
            depth = evidence.get("confirmation_depth")
            start = record.submission.get("submit_block")
            finalized_head = evidence.get("finalized_block")
            if (isinstance(era, bool) or not isinstance(era, int) or era <= 0
                    or isinstance(depth, bool) or not isinstance(depth, int) or depth < 0
                    or not isinstance(start, int) or isinstance(start, bool)
                    or not isinstance(finalized_head, int) or isinstance(finalized_head, bool)
                    or evidence.get("extrinsic_hash") != extrinsic_hash
                    or evidence.get("mortal_era") != era
                    or evidence.get("search_start") != start
                    or evidence.get("search_end") != start + era + depth
                    or finalized_head <= start + era + depth):
                raise EpochIndexConflict("expired-absent outcome requires full finalized mortal-era evidence")
        elif outcome == "operator_acknowledged" and evidence.get("operator_ack") != extrinsic_hash:
            raise EpochIndexConflict("operator acknowledgement must bind the pending hash")
        self._durable_execute(
            "UPDATE authority_anchor_submissions SET outcome=?,outcome_at=?,outcome_evidence=?"
            " WHERE extrinsic_hash=? AND outcome IS NULL",
            (outcome, outcome_at, json.dumps(evidence, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False), extrinsic_hash),
        )
        updated = self.anchor_submission_by_hash(extrinsic_hash)
        assert updated is not None
        return updated

    def anchor_submission_by_hash(self, extrinsic_hash: str) -> AnchorSubmissionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM authority_anchor_submissions WHERE extrinsic_hash=?",
            (extrinsic_hash,),
        ).fetchone()
        return _row_to_submission(row) if row is not None else None

    def get_anchor_submission(self, epoch_id: int) -> AnchorSubmissionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM authority_anchor_submissions WHERE epoch_id=? ORDER BY rowid DESC LIMIT 1",
            (epoch_id,),
        ).fetchone()
        return _row_to_submission(row) if row is not None else None

    def pending_anchor_submissions(self) -> list[AnchorSubmissionRecord]:
        return [_row_to_submission(row) for row in self._conn.execute(
            "SELECT * FROM authority_anchor_submissions WHERE outcome IS NULL ORDER BY rowid"
        )]

    # -- writes ----------------------------------------------------------------

    def record_finalized(self, finalized: FinalizedEpoch, *, finalized_at: str) -> EpochRecord:
        """Index a finalized epoch's pointer. Idempotent; immutable once written.

        A re-record of the SAME epoch with the SAME log_digest returns the existing
        row unchanged (finalize is idempotent, so this composes cleanly). A record
        with a DIFFERENT digest/key for an already-indexed epoch raises
        `EpochIndexConflict` — a finalized epoch is immutable.
        """
        tombstoned = self._conn.execute(
            "SELECT 1 FROM authority_epoch_tombstones WHERE epoch_id = ?",
            (finalized.epoch_id,),
        ).fetchone()
        if tombstoned is not None:
            raise EpochIndexConflict(
                f"epoch {finalized.epoch_id} is tombstoned as an acknowledged outage "
                "gap — a gap epoch can never be (re-)finalized"
            )
        existing = self.get(finalized.epoch_id)
        if existing is not None:
            if (
                existing.log_digest != finalized.log_digest
                or existing.snapshot_key != finalized.snapshot_key
                or existing.weight_vector_digest != finalized.weight_vector_digest
                or existing.close_block != finalized.close_block
            ):
                raise EpochIndexConflict(
                    f"epoch {finalized.epoch_id} is already finalized with a different "
                    f"pointer (indexed log_digest {existing.log_digest}, new "
                    f"{finalized.log_digest}) — a finalized epoch is immutable"
                )
            return existing
        self._conn.execute(
            "INSERT INTO authority_epochs"
            " (epoch_id, close_block, snapshot_key, log_digest, weight_vector_digest,"
            "  anchor_txid, anchor_block, finalized_at)"
            " VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)",
            (
                finalized.epoch_id,
                finalized.close_block,
                finalized.snapshot_key,
                finalized.log_digest,
                finalized.weight_vector_digest,
                finalized_at,
            ),
        )
        record = self.get(finalized.epoch_id)
        assert record is not None  # just inserted
        return record

    def set_anchor(
        self, epoch_id: int, *, txid: str, block: int | None = None
    ) -> EpochRecord:
        """Record the on-chain anchor for a finalized epoch. Idempotent per txid.

        The epoch must already be indexed (finalize precedes anchor). Setting the
        anchor a second time with the SAME txid is a NO-OP; a different txid raises
        `EpochIndexConflict` (an epoch is anchored exactly once).
        """
        record = self.get(epoch_id)
        if record is None:
            raise EpochIndexConflict(
                f"cannot anchor epoch {epoch_id}: it is not finalized/indexed yet"
            )
        if record.anchored:
            if record.anchor_txid != txid:
                raise EpochIndexConflict(
                    f"epoch {epoch_id} is already anchored as {record.anchor_txid}; "
                    f"refusing to re-anchor as {txid}"
                )
            return record
        self._conn.execute(
            "UPDATE authority_epochs SET anchor_txid = ?, anchor_block = ? WHERE epoch_id = ?",
            (txid, block, epoch_id),
        )
        updated = self.get(epoch_id)
        assert updated is not None
        return updated

    def mark_gap_tombstone(
        self, epoch_id: int, *, acknowledged_at: str, reason: str
    ) -> None:
        """Acknowledge an indexed-but-UNANCHORED epoch as an outage gap (P1.5/v16).

        The row itself is never deleted (audit trail); it simply stops being served
        (`get`/`latest` exclude tombstoned epochs, so the API 404s it and the spine
        resumes from the previous anchored epoch, declaring this one in the next
        log's ``gap_epochs``). The in-database triggers refuse tombstoning an
        anchored epoch and make tombstones append-only, immutable and permanent.
        Idempotent for an already-tombstoned epoch.
        """
        already = self._conn.execute(
            "SELECT 1 FROM authority_epoch_tombstones WHERE epoch_id = ?",
            (epoch_id,),
        ).fetchone()
        if already is not None:
            return
        self._conn.execute(
            "INSERT INTO authority_epoch_tombstones"
            " (epoch_id, acknowledged_at, reason) VALUES (?, ?, ?)",
            (epoch_id, acknowledged_at, reason),
        )

    # -- reads -----------------------------------------------------------------

    def get(self, epoch_id: int) -> EpochRecord | None:
        row = self._conn.execute(
            "SELECT * FROM authority_epochs WHERE epoch_id = ?"
            " AND epoch_id NOT IN (SELECT epoch_id FROM authority_epoch_tombstones)",
            (epoch_id,),
        ).fetchone()
        return _row_to_record(row) if row is not None else None

    def latest(self) -> EpochRecord | None:
        """The newest non-tombstoned finalized epoch (highest epoch_id), or None."""
        row = self._conn.execute(
            "SELECT * FROM authority_epochs"
            " WHERE epoch_id NOT IN (SELECT epoch_id FROM authority_epoch_tombstones)"
            " ORDER BY epoch_id DESC LIMIT 1"
        ).fetchone()
        return _row_to_record(row) if row is not None else None
