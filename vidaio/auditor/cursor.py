"""AuditCursor — the auditor's DURABLE contiguous-coverage cursor.

Finding #3: the auditor used to audit only ``latest_pointer()`` with an in-memory
``_last_audited``. A HELD epoch, or a pointer jump E -> E+2, then permanently SKIPPED
epochs — an untrusted authority could race past an invalid epoch before auditors got
its beacon. This module gives the auditor a small SQLite cursor (mirroring the
per-service DB pattern: a ``migrations/`` dir + ``apply_migrations`` + ``connect`` from
``vidaio.core.db``), recording the highest epoch it has AUDITED-AND-SUBMITTED.

The auditor audits epochs CONTIGUOUSLY in ascending order from ``cursor + 1`` up to the
authority's latest finalized epoch. The cursor advances past an epoch ONLY once that
epoch has been audited-and-submitted; a HOLD/REFUSE leaves it in place, so that epoch is
retried next pass and no later epoch is audited ahead of it. Advance is monotonic and
enforced in-database (see the migration), so the cursor can never rewind (which would
re-open a skip window). Durable, so a restart resumes exactly where it left off.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from vidaio.core import apply_migrations, connect

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

#: The singleton row id (the cursor is one row; see 0001_audit_cursor.sql).
_ROW_ID = 0


class AuditCursor:
    """SQLite-backed durable cursor of the highest audited-and-submitted epoch.

    Constructed over a connection this class owns (`open`) or one injected for tests.
    ``last_audited()`` is None until the first epoch is recorded (genesis/first run);
    ``advance_to(epoch_id)`` records an audited-and-submitted epoch and MUST be called
    with strictly increasing epoch ids (the loop drives it contiguously), which the
    in-database monotonic trigger also enforces.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        apply_migrations(conn, MIGRATIONS_DIR)

    @classmethod
    def open(cls, db_path: str | Path) -> "AuditCursor":
        return cls(connect(db_path))

    def close(self) -> None:
        self._conn.close()

    def last_audited(self) -> int | None:
        """The highest AUDITED-AND-SUBMITTED epoch, or None on first run (no row yet)."""
        row = self._conn.execute(
            "SELECT last_audited_epoch FROM audit_cursor WHERE id = ?", (_ROW_ID,)
        ).fetchone()
        return None if row is None else int(row["last_audited_epoch"])

    def advance_to(self, epoch_id: int) -> None:
        """Record ``epoch_id`` as audited-and-submitted (monotonic; inserts the row once).

        The first call inserts the singleton row; later calls UPDATE it, which the
        ``audit_cursor_monotonic`` trigger permits only when ``epoch_id`` strictly
        exceeds the stored value — so a bug that tried to rewind (and re-open a skip
        window) is aborted in-database rather than silently accepted.
        """
        current = self.last_audited()
        if current is None:
            self._conn.execute(
                "INSERT INTO audit_cursor (id, last_audited_epoch) VALUES (?, ?)",
                (_ROW_ID, epoch_id),
            )
            return
        if epoch_id <= current:
            # The loop should never call this non-monotonically; guard here too so a
            # caller bug is a loud error, not a silent cursor rewind.
            raise ValueError(
                f"audit cursor cannot advance to {epoch_id}: it is already at {current}"
                ""
            )
        self._conn.execute(
            "UPDATE audit_cursor SET last_audited_epoch = ? WHERE id = ?",
            (epoch_id, _ROW_ID),
        )
