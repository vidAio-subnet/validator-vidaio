-- Let the validator's independent beacon and own-audit paths report the same
-- hotkey+epoch without being treated as divergent submissions. Historical rows
-- are beacon reports; the DEFAULT backfills that mode data-preservingly.
--
-- SQLite cannot replace the existing UNIQUE(auditor_hotkey, epoch_id)
-- constraint in place, so rebuild audit_reports with the mode in its uniqueness
-- scope, preserving every immutable row and its original report_json bytes.

CREATE TABLE audit_reports_v5 (
    report_id        TEXT PRIMARY KEY CHECK (length(report_id) = 64),
    auditor_hotkey   TEXT NOT NULL,
    epoch_id         INTEGER NOT NULL,
    audit_mode       TEXT NOT NULL DEFAULT 'beacon'
                         CHECK (audit_mode IN ('beacon', 'own_audit')),
    snapshot_digest  TEXT NOT NULL CHECK (length(snapshot_digest) = 64),
    pipeline_version TEXT NOT NULL,
    overall          TEXT NOT NULL CHECK (overall IN ('CLEAN', 'DISPUTED', 'INCONCLUSIVE')),
    competition_n    INTEGER NOT NULL,
    inference_n      INTEGER NOT NULL,
    sampled_at       TEXT NOT NULL,
    report_json      TEXT NOT NULL,
    received_at      TEXT NOT NULL,
    UNIQUE (auditor_hotkey, epoch_id, audit_mode)
);

INSERT INTO audit_reports_v5
    (report_id, auditor_hotkey, epoch_id, audit_mode, snapshot_digest,
     pipeline_version, overall, competition_n, inference_n, sampled_at,
     report_json, received_at)
SELECT
    report_id, auditor_hotkey, epoch_id, 'beacon', snapshot_digest,
    pipeline_version, overall, competition_n, inference_n, sampled_at,
    report_json, received_at
FROM audit_reports;

DROP TABLE audit_reports;

ALTER TABLE audit_reports_v5 RENAME TO audit_reports;

CREATE INDEX audit_reports_by_epoch ON audit_reports (epoch_id);
CREATE INDEX audit_reports_by_received ON audit_reports (received_at DESC, report_id);

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

-- Conflict rows are already keyed by an opaque id, so an additive column is
-- sufficient. Existing conflicts arose from beacon reports and backfill as such.
ALTER TABLE audit_report_conflicts
    ADD COLUMN audit_mode TEXT NOT NULL DEFAULT 'beacon'
        CHECK (audit_mode IN ('beacon', 'own_audit'));

CREATE INDEX audit_report_conflicts_by_epoch_mode
    ON audit_report_conflicts (epoch_id, audit_mode);
