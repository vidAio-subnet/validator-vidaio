"""OwnAuditCursor — the legacy contiguous own-audit classifier cursor.

Finding #2 (HIGH): the round-12 own-audit gate resolved only the authority's LATEST pointer,
so a SKIPPED epoch (weightsetting ~30s vs ~20s epochs) permanently stranded the CLEAN ledger
chain — the next positive-carry audit stayed INCONCLUSIVE because its predecessor was never recorded,
and later passes only fetched still-newer epochs. A restart with an empty ledger mid-chain
wedged the same way.

This cursor mirrors the public auditor's ``AuditCursor`` (vidaio.auditor.cursor): a single-row
SQLite record of the HIGHEST epoch the gate has own-audited CLEAN CONTIGUOUSLY. The gate walks
``cursor + 1 .. latest`` each attempt, BACKFILLING every missed epoch (fetched by
``pointer_for(epoch_id)``), own-auditing it, recording it in the ``OwnAuditLedger``, and
advancing this cursor ONLY after a CLEAN clear — so the ledger never gaps and a positive-carry
epoch's predecessor is always present when the chain is honest. Advance is monotonic and
enforced in-database, so the cursor can never rewind (which would re-open a skip window).
Durable, so a restart resumes exactly where it left off.

Production weight-setting does not construct ``OwnAuditGate`` or this cursor. The live
own-auditor instead owns ``vidaio.auditor.cursor.AuditCursor`` and a pending-report
outbox in its separate process/container. This class remains test/legacy policy state.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from vidaio.core.db import apply_migrations, connect

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

#: The singleton row id (the cursor is one row; see 0005_own_audit_cursor.sql).
_ROW_ID = 0


class OwnAuditCursor:
    """SQLite-backed durable cursor of the highest CONTIGUOUSLY own-audited-CLEAN epoch.

    Constructed over a connection this class owns (`open`) or one injected for tests.
    ``last_clean()`` is None until the first epoch is recorded (fresh gate / first run);
    ``advance_to(epoch_id)`` records a contiguously-cleared epoch and MUST be called with
    strictly increasing epoch ids (the backfill walk drives it contiguously), which the
    in-database monotonic trigger also enforces.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        apply_migrations(conn, MIGRATIONS_DIR)

    @classmethod
    def open(cls, db_path: str | Path) -> "OwnAuditCursor":
        return cls(connect(db_path))

    def close(self) -> None:
        self._conn.close()

    def last_clean(self) -> int | None:
        """The highest contiguously own-audited-CLEAN epoch, or None on a fresh cursor."""
        row = self._conn.execute(
            "SELECT last_clean_epoch FROM own_audit_cursor WHERE id = ?", (_ROW_ID,)
        ).fetchone()
        return None if row is None else int(row["last_clean_epoch"])

    def advance_to(self, epoch_id: int) -> None:
        """Record ``epoch_id`` as contiguously own-audited CLEAN (monotonic; inserts once).

        The first call inserts the singleton row; later calls UPDATE it, which the
        ``own_audit_cursor_monotonic`` trigger permits only when ``epoch_id`` strictly
        exceeds the stored value — so a bug that tried to rewind (and re-open a skip window)
        is aborted in-database rather than silently accepted.
        """
        current = self.last_clean()
        if current is None:
            self._conn.execute(
                "INSERT INTO own_audit_cursor (id, last_clean_epoch) VALUES (?, ?)",
                (_ROW_ID, int(epoch_id)),
            )
            return
        if epoch_id <= current:
            raise ValueError(
                f"own-audit cursor cannot advance to {epoch_id}: it is already at {current}"
                ""
            )
        self._conn.execute(
            "UPDATE own_audit_cursor SET last_clean_epoch = ? WHERE id = ?",
            (int(epoch_id), _ROW_ID),
        )


__all__ = ["OwnAuditCursor"]
