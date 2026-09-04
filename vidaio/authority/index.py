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
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

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
