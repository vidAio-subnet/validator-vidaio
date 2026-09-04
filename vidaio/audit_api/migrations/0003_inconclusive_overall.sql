-- Widen audit_reports.overall to admit INCONCLUSIVE. The auditor now emits an
-- INCONCLUSIVE verdict for an all-SKIP media sample (nothing could be recomputed —
-- not clean, a needs-attention state, #8, the project design record §3.2). The
-- shipped 0001 CHECK only allowed ('CLEAN','DISPUTED'), so a valid INCONCLUSIVE
-- report raised sqlite3.IntegrityError at POST /audit/report instead of persisting.
--
-- SQLite cannot ALTER a CHECK constraint, so the append-only table is RECREATED with
-- the widened CHECK and its rows copied across data-preservingly (the standard
-- table-rebuild: new table + copy + drop + rename + re-create indexes/triggers).
-- No foreign key references audit_reports, so no FK-off dance is needed. DROP TABLE
-- does not fire the BEFORE DELETE immutability trigger (it deletes no rows), and the
-- copy is an INSERT (not an UPDATE), so the append-only triggers do not block it.
-- The old triggers/indexes are dropped with the old table and recreated verbatim.

CREATE TABLE audit_reports_v3 (
    report_id        TEXT PRIMARY KEY CHECK (length(report_id) = 64),  -- the report digest
    auditor_hotkey   TEXT NOT NULL,
    epoch_id         INTEGER NOT NULL,
    snapshot_digest  TEXT NOT NULL CHECK (length(snapshot_digest) = 64),
    pipeline_version TEXT NOT NULL,
    overall          TEXT NOT NULL CHECK (overall IN ('CLEAN', 'DISPUTED', 'INCONCLUSIVE')),
    competition_n    INTEGER NOT NULL,
    inference_n      INTEGER NOT NULL,
    sampled_at       TEXT NOT NULL,
    report_json      TEXT NOT NULL,   -- the full serialized AuditReport (re-verifiable)
    received_at      TEXT NOT NULL,
    UNIQUE (auditor_hotkey, epoch_id)
);

-- Preserve every persisted report (append-only: nothing is dropped or rewritten).
INSERT INTO audit_reports_v3
    (report_id, auditor_hotkey, epoch_id, snapshot_digest, pipeline_version,
     overall, competition_n, inference_n, sampled_at, report_json, received_at)
SELECT
    report_id, auditor_hotkey, epoch_id, snapshot_digest, pipeline_version,
    overall, competition_n, inference_n, sampled_at, report_json, received_at
FROM audit_reports;

DROP TABLE audit_reports;

ALTER TABLE audit_reports_v3 RENAME TO audit_reports;

-- Recreate the indexes verbatim (dropped with the old table).
CREATE INDEX audit_reports_by_epoch ON audit_reports (epoch_id);
CREATE INDEX audit_reports_by_received ON audit_reports (received_at DESC, report_id);

-- Recreate the append-only immutability triggers verbatim (dropped with the old table).
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
