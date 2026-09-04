"""PendingReportStore — the auditor loop's BYTE-IDEMPOTENT pending report.

Finding #3 (HIGH): the public auditor loop recreated the report with a FRESH ``sampled_at`` on
every retry, changing its signed digest. If the Audit Results API COMMITTED the first POST but
its response was LOST, the retry's DIFFERENT digest CONFLICTs with the stored report
(vidaio.audit_api.store) and the cursor never advances → every later epoch is blocked.

This tiny SQLite store (mirroring the auditor cursor's durable-DB pattern) persists the EXACT
signed report the loop first built for an epoch. On a retry the loop RESENDS those identical
bytes, so a lost-response retry is reconciled as a DUPLICATE (idempotent accept) rather than a
CONFLICT — the cursor advances instead of wedging. ``put`` keeps the FIRST report for an epoch
(a re-audit that happened to differ must not change the committed bytes); ``clear`` drops it
once the report is durably accepted.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from vidaio.auditor.report import AuditReport
from vidaio.core.db import apply_migrations, connect

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class PendingReportStore:
    """SQLite store of the one in-flight signed AuditReport per epoch (own migrations)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        apply_migrations(conn, MIGRATIONS_DIR)

    @classmethod
    def open(cls, db_path: str | Path) -> "PendingReportStore":
        return cls(connect(db_path))

    def close(self) -> None:
        self._conn.close()

    def get(self, epoch_id: int) -> AuditReport | None:
        """The exact report first built for ``epoch_id`` (byte-for-byte), or None."""
        row = self._conn.execute(
            "SELECT report_json FROM pending_report WHERE epoch_id = ?", (int(epoch_id),)
        ).fetchone()
        if row is None:
            return None
        return AuditReport.model_validate(json.loads(row["report_json"]))

    def put(self, epoch_id: int, report: AuditReport) -> None:
        """Persist ``report`` as the epoch's canonical in-flight bytes (keeps the FIRST).

        A repeat call for an epoch that already has a pending report is a NO-OP, so the exact
        bytes the API may already have committed are the ones we resend — a re-audit whose
        ``sampled_at`` (or any field) differs cannot clobber the committed digest.
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO pending_report (epoch_id, report_json) VALUES (?, ?)",
            (int(epoch_id), json.dumps(report.model_dump(mode="json"), sort_keys=True)),
        )

    def clear(self, epoch_id: int) -> None:
        """Drop the pending report once it is durably accepted (or superseded)."""
        self._conn.execute(
            "DELETE FROM pending_report WHERE epoch_id = ?", (int(epoch_id),)
        )


__all__ = ["PendingReportStore"]
