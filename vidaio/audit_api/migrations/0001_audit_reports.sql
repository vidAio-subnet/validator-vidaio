-- The Audit Results API's report store: the auditors' signed verdicts on each
-- epoch, and the conflict ledger. This is the honesty surface a dashboard reads
-- (the project design record §3.2, build-wave 7) — DISPUTED epochs are how a
-- dishonest Scoring Authority becomes publicly visible.
--
-- Append-only + immutable, and ONE report per (auditor_hotkey, epoch_id): an
-- auditor commits to exactly one verdict per epoch. A resubmission with the SAME
-- report_id (== report digest) is idempotent; a resubmission with a DIFFERENT
-- digest for that same pair is a CONFLICT — the first report is kept and the
-- divergence is recorded in audit_report_conflicts as itself a signal (the
-- service never silently overwrites a persisted verdict). The in-database triggers
-- enforce append-only + immutability even against direct SQL.

CREATE TABLE audit_reports (
    report_id        TEXT PRIMARY KEY CHECK (length(report_id) = 64),  -- the report digest
    auditor_hotkey   TEXT NOT NULL,
    epoch_id         INTEGER NOT NULL,
    snapshot_digest  TEXT NOT NULL CHECK (length(snapshot_digest) = 64),
    pipeline_version TEXT NOT NULL,
    overall          TEXT NOT NULL CHECK (overall IN ('CLEAN', 'DISPUTED')),
    competition_n    INTEGER NOT NULL,
    inference_n      INTEGER NOT NULL,
    sampled_at       TEXT NOT NULL,
    report_json      TEXT NOT NULL,   -- the full serialized AuditReport (re-verifiable)
    received_at      TEXT NOT NULL,
    UNIQUE (auditor_hotkey, epoch_id)
);

CREATE INDEX audit_reports_by_epoch ON audit_reports (epoch_id);
CREATE INDEX audit_reports_by_received ON audit_reports (received_at DESC, report_id);

-- A persisted verdict is frozen: no field of an audit_reports row may change.
CREATE TRIGGER audit_reports_immutable
BEFORE UPDATE ON audit_reports
BEGIN
    SELECT RAISE(ABORT, 'a persisted audit report is immutable (append-only)');
END;

CREATE TRIGGER audit_reports_no_delete
BEFORE DELETE ON audit_reports
BEGIN
    SELECT RAISE(ABORT, 'the audit report store is append-only');
END;

-- The conflict ledger: a divergent resubmission for an already-reported
-- (auditor_hotkey, epoch_id). The first report is KEPT; this row records the
-- divergence so the aggregate/dashboard can surface it as a signal.
CREATE TABLE audit_report_conflicts (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    auditor_hotkey           TEXT NOT NULL,
    epoch_id                 INTEGER NOT NULL,
    kept_report_id           TEXT NOT NULL,
    kept_snapshot_digest     TEXT NOT NULL,
    rejected_report_id       TEXT NOT NULL,
    rejected_snapshot_digest TEXT NOT NULL,
    recorded_at              TEXT NOT NULL
);

CREATE INDEX audit_report_conflicts_by_epoch ON audit_report_conflicts (epoch_id);

CREATE TRIGGER audit_report_conflicts_no_update
BEFORE UPDATE ON audit_report_conflicts
BEGIN
    SELECT RAISE(ABORT, 'the audit conflict ledger is append-only');
END;

CREATE TRIGGER audit_report_conflicts_no_delete
BEFORE DELETE ON audit_report_conflicts
BEGIN
    SELECT RAISE(ABORT, 'the audit conflict ledger is append-only');
END;
