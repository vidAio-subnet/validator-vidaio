"""The Audit Results store — append-only persistence of the auditors' verdicts.

One SQLite database (own migrations) holding, per
``(auditor_hotkey, epoch_id, audit_mode)``, the one signed ``AuditReport`` that
auditor committed to for that epoch and mode, plus a conflict ledger. It is the
substrate the aggregate ``/audit/status`` investigation/alerting surface
and the dashboard feed read from (the project design record §3.2, build-wave 7).

Persistence rules (mirrors ``vidaio.authority.index``):

- ``record`` writes one immutable row. A resubmission of the SAME report (same
  ``report_id`` = report digest) is IDEMPOTENT — the existing row is returned. A
  resubmission with a DIFFERENT digest for an already-reported
  (auditor, epoch, mode) tuple
  is a CONFLICT: the first report is KEPT and the divergence is logged in the conflict
  ledger (the conflict is itself a signal); the store never overwrites a verdict.
- reports and conflicts are append-only; the in-database triggers enforce it even
  against direct SQL.

The full report is stored as its serialized JSON, so any read can reconstruct the
exact ``AuditReport`` (and re-verify its signature / re-derive the aggregate).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from vidaio.auditor.report import AuditMode, AuditReport, overall_status
from vidaio.core import apply_migrations, connect

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _recomputed_overall(report: AuditReport) -> str:
    """The report's EFFECTIVE verdict, recomputed from its item + weight verdicts.

    Never trusts the report's self-reported ``overall``: a report claiming CLEAN while
    carrying a FAIL item is stored (and aggregated) as DISPUTED. This is the same rule
    the aggregate enforces — applied at write time so the persisted ``overall`` column,
    the disputed-epochs gauge, and the feed can't be fooled either.

    ``earning_verdicts`` is passed as the third channel so the persisted verdict matches
    the report's own DERIVED ``overall`` for the earning-SKIP (INCONCLUSIVE) case — an
    unverifiable earning state must not wash to CLEAN in the store while the report says
    INCONCLUSIVE.
    """
    return overall_status(
        report.item_verdicts, report.weight_verdict, report.earning_verdicts
    ).value


class RecordOutcome(StrEnum):
    """What ``record`` did with a submitted report."""

    NEW = "new"          # first report for this (auditor, epoch, mode) — persisted
    DUPLICATE = "duplicate"  # identical report already stored (idempotent)
    CONFLICT = "conflict"    # divergent report for this (auditor, epoch, mode) — first kept


@dataclass(frozen=True)
class StoredReport:
    """A persisted report row + the reconstructed ``AuditReport``."""

    report_id: str
    auditor_hotkey: str
    epoch_id: int
    audit_mode: AuditMode
    snapshot_digest: str
    pipeline_version: str
    overall: str
    competition_n: int
    inference_n: int
    sampled_at: str
    received_at: str
    report: AuditReport


@dataclass(frozen=True)
class RecordResult:
    """The outcome of ``record`` — what happened, and the report now of record.

    ``kept`` is always the report that is (or was already) persisted for the
    auditor+epoch+mode tuple:
    the just-inserted one on NEW, the identical existing one on DUPLICATE, the FIRST
    (retained) one on CONFLICT. ``report_id`` is that kept report's id — the value
    the auditor's ``SubmitAck`` carries back.
    """

    outcome: RecordOutcome
    kept: StoredReport
    report_id: str


def _row_to_stored(row: sqlite3.Row) -> StoredReport:
    return StoredReport(
        report_id=row["report_id"],
        auditor_hotkey=row["auditor_hotkey"],
        epoch_id=row["epoch_id"],
        audit_mode=AuditMode(row["audit_mode"]),
        snapshot_digest=row["snapshot_digest"],
        pipeline_version=row["pipeline_version"],
        overall=row["overall"],
        competition_n=row["competition_n"],
        inference_n=row["inference_n"],
        sampled_at=row["sampled_at"],
        received_at=row["received_at"],
        report=AuditReport.model_validate(json.loads(row["report_json"])),
    )


class AuditResultsStore:
    """SQLite store of received AuditReports (own migrations, append-only)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        apply_migrations(conn, MIGRATIONS_DIR)

    @classmethod
    def open(cls, db_path: str | Path) -> "AuditResultsStore":
        return cls(connect(db_path))

    def close(self) -> None:
        self._conn.close()

    # -- writes ----------------------------------------------------------------

    def record(self, report: AuditReport, *, received_at: str) -> RecordResult:
        """Persist a report. Idempotent per digest; conflicts keep the first.

        A re-post of the identical report (same ``report_digest``) returns the
        stored row unchanged. A DIFFERENT report for an already-reported
        (auditor_hotkey, epoch_id, audit_mode) is a conflict: the first is kept and the
        divergence is recorded in the conflict ledger.
        """
        report_id = report.report_digest()
        existing = self.get(report_id)
        if existing is not None:
            return RecordResult(RecordOutcome.DUPLICATE, existing, report_id)

        prior = self.get_for_pair(
            report.auditor_hotkey, report.epoch_id, report.audit_mode
        )
        if prior is not None:
            self._conn.execute(
                "INSERT INTO audit_report_conflicts"
                " (auditor_hotkey, epoch_id, audit_mode, kept_report_id, kept_snapshot_digest,"
                "  rejected_report_id, rejected_snapshot_digest, rejected_overall,"
                "  recorded_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report.auditor_hotkey,
                    report.epoch_id,
                    report.audit_mode.value,
                    prior.report_id,
                    prior.snapshot_digest,
                    report_id,
                    report.snapshot_digest,
                    # the REJECTED report's recomputed verdict — a divergent report that
                    # is DISPUTED must still surface as a dispute even though it is not
                    # persisted (a CLEAN first report cannot hide a later DISPUTED one).
                    _recomputed_overall(report),
                    received_at,
                ),
            )
            return RecordResult(RecordOutcome.CONFLICT, prior, prior.report_id)

        self._conn.execute(
            "INSERT INTO audit_reports"
            " (report_id, auditor_hotkey, epoch_id, audit_mode, snapshot_digest, pipeline_version,"
            "  overall, competition_n, inference_n, sampled_at, report_json, received_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report_id,
                report.auditor_hotkey,
                report.epoch_id,
                report.audit_mode.value,
                report.snapshot_digest,
                report.pipeline_version,
                # recomputed from item/weight verdicts — never the self-reported field.
                _recomputed_overall(report),
                report.competition_n,
                report.inference_n,
                report.sampled_at.isoformat(),
                json.dumps(report.model_dump(mode="json"), sort_keys=True),
                received_at,
            ),
        )
        stored = self.get(report_id)
        assert stored is not None  # just inserted
        return RecordResult(RecordOutcome.NEW, stored, report_id)

    # -- reads -----------------------------------------------------------------

    def get(self, report_id: str) -> StoredReport | None:
        row = self._conn.execute(
            "SELECT * FROM audit_reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        return _row_to_stored(row) if row is not None else None

    def get_for_pair(
        self,
        auditor_hotkey: str,
        epoch_id: int,
        audit_mode: AuditMode | str = AuditMode.BEACON,
    ) -> StoredReport | None:
        """Return the report for one auditor+epoch+mode tuple.

        The default remains ``beacon`` so existing callers retain their historical
        meaning while own-audit callers opt into their independent namespace.
        """
        row = self._conn.execute(
            "SELECT * FROM audit_reports"
            " WHERE auditor_hotkey = ? AND epoch_id = ? AND audit_mode = ?",
            (auditor_hotkey, epoch_id, AuditMode(audit_mode).value),
        ).fetchone()
        return _row_to_stored(row) if row is not None else None

    def for_epoch(self, epoch_id: int) -> list[StoredReport]:
        """All reports for one epoch, ordered by auditor (deterministic)."""
        rows = self._conn.execute(
            "SELECT * FROM audit_reports WHERE epoch_id = ?"
            " ORDER BY auditor_hotkey, audit_mode, report_id",
            (epoch_id,),
        ).fetchall()
        return [_row_to_stored(r) for r in rows]

    def recent(self, limit: int) -> list[StoredReport]:
        """The newest reports first (dashboard feed).

        Ties on ``received_at`` break by INSERTION order (rowid) — the true receive
        order — never by the content-derived ``report_id``, which would let two
        same-instant reports surface in an order unrelated to when they arrived.
        """
        rows = self._conn.execute(
            "SELECT * FROM audit_reports ORDER BY received_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_stored(r) for r in rows]

    def conflicts_for_epoch(
        self, epoch_id: int, audit_mode: AuditMode | str | None = None
    ) -> int:
        """Count conflicts for an epoch, optionally scoped to one report mode."""
        sql = "SELECT COUNT(*) AS n FROM audit_report_conflicts WHERE epoch_id = ?"
        params: tuple[object, ...] = (epoch_id,)
        if audit_mode is not None:
            sql += " AND audit_mode = ?"
            params += (AuditMode(audit_mode).value,)
        row = self._conn.execute(sql, params).fetchone()
        return int(row["n"])

    def disputed_conflicts_for_epoch(
        self, epoch_id: int, audit_mode: AuditMode | str | None = None
    ) -> int:
        """Conflicts for an epoch whose REJECTED (divergent) report was DISPUTED.

        A non-zero count means a signed, divergent report found a provable fault but
        was not persisted (its pair already had a first report). The aggregate must
        still mark the epoch DISPUTED so a CLEAN first report cannot bury it.
        """
        sql = (
            "SELECT COUNT(*) AS n FROM audit_report_conflicts"
            " WHERE epoch_id = ? AND rejected_overall = 'DISPUTED'"
        )
        params: tuple[object, ...] = (epoch_id,)
        if audit_mode is not None:
            sql += " AND audit_mode = ?"
            params += (AuditMode(audit_mode).value,)
        row = self._conn.execute(sql, params).fetchone()
        return int(row["n"])

    def total_conflicts(self, audit_mode: AuditMode | str | None = None) -> int:
        """Count all conflicts, optionally scoped to one report mode."""
        sql = "SELECT COUNT(*) AS n FROM audit_report_conflicts"
        params: tuple[object, ...] = ()
        if audit_mode is not None:
            sql += " WHERE audit_mode = ?"
            params = (AuditMode(audit_mode).value,)
        row = self._conn.execute(sql, params).fetchone()
        return int(row["n"])

    def epoch_ids(self) -> list[int]:
        """Every epoch that has at least one report, newest first."""
        rows = self._conn.execute(
            "SELECT DISTINCT epoch_id FROM audit_reports ORDER BY epoch_id DESC"
        ).fetchall()
        return [int(r["epoch_id"]) for r in rows]

    def disputed_epoch_count(self) -> int:
        """Epochs the aggregate marks DISPUTED — the disputed-epochs gauge.

        An epoch is disputed if a PERSISTED report is DISPUTED (recomputed ``overall``)
        OR a divergent (rejected) report was DISPUTED — the same rule ``/audit/status``
        applies, so the gauge cannot disagree with the surface.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM ("
            "  SELECT epoch_id FROM audit_reports WHERE overall = 'DISPUTED'"
            "  UNION"
            "  SELECT epoch_id FROM audit_report_conflicts WHERE rejected_overall = 'DISPUTED'"
            ")"
        ).fetchone()
        return int(row["n"])
