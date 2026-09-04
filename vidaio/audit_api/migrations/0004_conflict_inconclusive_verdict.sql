-- Widen audit_report_conflicts.rejected_overall to admit INCONCLUSIVE. 0002 added
-- the column with CHECK (rejected_overall IN ('CLEAN','DISPUTED')), but the auditor
-- now emits INCONCLUSIVE for an all-SKIP media sample (0003 widened audit_reports
-- for exactly this). A DIVERGENT resubmission whose recomputed verdict is
-- INCONCLUSIVE reaches the conflict-ledger INSERT (store.py record()) and raised
-- sqlite3.IntegrityError -> an UNRECORDED 500 at POST /audit/report, so the conflict
-- was neither stored nor surfaced (#5, #8, the project design record §3.2, §5).
--
-- SQLite cannot ALTER a CHECK constraint, so the append-only ledger is RECREATED
-- with the widened CHECK and its rows copied across data-preservingly (the standard
-- table-rebuild, exactly as 0003 did for audit_reports): new table + copy + drop +
-- rename + re-create index/triggers. No foreign key references it. DROP TABLE fires
-- no BEFORE DELETE trigger (it deletes no rows) and the copy is an INSERT (not an
-- UPDATE), so the append-only triggers do not block the rebuild. The id column stays
-- INTEGER PRIMARY KEY AUTOINCREMENT and every id is copied verbatim, so no surrogate
-- key is renumbered and the next insert still allocates above the current maximum.

CREATE TABLE audit_report_conflicts_v4 (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    auditor_hotkey           TEXT NOT NULL,
    epoch_id                 INTEGER NOT NULL,
    kept_report_id           TEXT NOT NULL,
    kept_snapshot_digest     TEXT NOT NULL,
    rejected_report_id       TEXT NOT NULL,
    rejected_snapshot_digest TEXT NOT NULL,
    rejected_overall         TEXT NOT NULL DEFAULT 'CLEAN'
        CHECK (rejected_overall IN ('CLEAN', 'DISPUTED', 'INCONCLUSIVE')),
    recorded_at              TEXT NOT NULL
);

-- Preserve every recorded conflict verbatim (append-only: nothing is dropped or
-- rewritten), ids included, so AUTOINCREMENT continues above the current maximum.
INSERT INTO audit_report_conflicts_v4
    (id, auditor_hotkey, epoch_id, kept_report_id, kept_snapshot_digest,
     rejected_report_id, rejected_snapshot_digest, rejected_overall, recorded_at)
SELECT
    id, auditor_hotkey, epoch_id, kept_report_id, kept_snapshot_digest,
    rejected_report_id, rejected_snapshot_digest, rejected_overall, recorded_at
FROM audit_report_conflicts;

DROP TABLE audit_report_conflicts;

ALTER TABLE audit_report_conflicts_v4 RENAME TO audit_report_conflicts;

-- Recreate the index verbatim (dropped with the old table).
CREATE INDEX audit_report_conflicts_by_epoch ON audit_report_conflicts (epoch_id);

-- Recreate the append-only immutability triggers verbatim (dropped with the old table).
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
