"""OwnAuditLedger — the legacy own-audited-CLEAN classifier chain.

Finding #1 (CRITICAL): the legacy own-audit classifier re-folds the CURRENT epoch's earning
state and verifies its nonzero carry-in against the PREDECESSOR's stated
``accumulate_score`` (``_carry_in_check``), but it never verified that the predecessor was
ITSELF own-audited CLEAN. So an untrusted authority could publish a structurally-invalid
predecessor carrying an INJECTED accumulator, chain it into a self-consistent current
epoch, and pass the current own-audit — the public auditor loop may later dispute the
predecessor. Either verdict is report-only for weight-setting.

This module gives the gate a small SQLite ledger (mirroring the auditor's durable-cursor
pattern — a ``migrations/`` dir + ``apply_migrations`` + ``connect`` from ``vidaio.core.db``)
that RECORDS every (epoch_id, log_digest) the gate has cleared CLEAN. Before clearing an
epoch whose carry-in is NONZERO the gate REQUIRES the predecessor (epoch_id-1,
prior_log_digest) to be a recorded CLEAN entry here — otherwise the carry-in is classified
INCONCLUSIVE. This is the "durable, contiguous chain of previously
own-audited-CLEAN digests" review specifies: it extends by one entry per CLEAN clear and, by
the in-database monotonic trigger, never rewinds. Production ``WeightSetter`` does not
construct this reviewer or ledger. The isolated own-auditor uses the current auditor
cursor/outbox path; this classifier state remains for tests and legacy callers only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from vidaio.core.db import apply_migrations, connect

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class OwnAuditCleanConflict(ValueError):
    """An epoch was re-recorded with a DIFFERENT log_digest — a CLEAN entry is immutable."""


class OwnAuditLedger:
    """SQLite-backed durable ledger of (epoch_id, log_digest) own-audited-CLEAN entries.

    Constructed over a connection this class owns (`open`) or one injected for tests.
    ``is_clean(epoch_id, log_digest)`` answers whether that exact epoch+digest was recorded
    CLEAN; ``record_clean(epoch_id, log_digest)`` extends the chain (idempotent for a repeat
    of the SAME (epoch_id, log_digest); a conflicting digest for an already-recorded epoch
    raises; a lower/equal epoch_id is aborted in-database by the monotonic trigger).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        apply_migrations(conn, MIGRATIONS_DIR)

    @classmethod
    def open(cls, db_path: str | Path) -> "OwnAuditLedger":
        return cls(connect(db_path))

    def close(self) -> None:
        self._conn.close()

    def is_clean(self, epoch_id: int, log_digest: str) -> bool:
        """True iff (epoch_id, log_digest) was previously recorded own-audited CLEAN."""
        row = self._conn.execute(
            "SELECT 1 FROM own_audit_clean WHERE epoch_id = ? AND log_digest = ?",
            (int(epoch_id), str(log_digest)),
        ).fetchone()
        return row is not None

    def record_clean(self, epoch_id: int, log_digest: str) -> None:
        """Record ``(epoch_id, log_digest)`` as own-audited CLEAN (idempotent, forward-only).

        Re-recording the SAME (epoch_id, log_digest) is a no-op (a re-review of an already
        cleared epoch). Re-recording an already-recorded epoch with a DIFFERENT digest is a
        conflict (a CLEAN entry is immutable) and raises. A brand-new epoch is INSERTed; the
        ``own_audit_clean_forward`` trigger aborts an epoch_id at/below the highest recorded
        one, so the chain can never be rewound.
        """
        existing = self._conn.execute(
            "SELECT log_digest FROM own_audit_clean WHERE epoch_id = ?", (int(epoch_id),)
        ).fetchone()
        if existing is not None:
            if str(existing["log_digest"]) != str(log_digest):
                raise OwnAuditCleanConflict(
                    f"epoch {epoch_id} already recorded own-audited CLEAN with a different "
                    f"log_digest ({existing['log_digest']!r} != {log_digest!r}) — a CLEAN "
                    "entry is immutable"
                )
            return
        self._conn.execute(
            "INSERT INTO own_audit_clean (epoch_id, log_digest) VALUES (?, ?)",
            (int(epoch_id), str(log_digest)),
        )


__all__ = ["OwnAuditLedger", "OwnAuditCleanConflict"]
