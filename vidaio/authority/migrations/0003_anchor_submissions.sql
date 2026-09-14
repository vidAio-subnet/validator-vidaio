-- Signed authority anchors survive process restarts before any wire dispatch.
-- Each submission is immutable; its outcome may be filled exactly once.
CREATE TABLE authority_anchor_submissions (
    extrinsic_hash TEXT PRIMARY KEY,
    epoch_id INTEGER NOT NULL REFERENCES authority_epochs(epoch_id),
    netuid INTEGER NOT NULL,
    log_digest TEXT NOT NULL CHECK (length(log_digest) = 64),
    submission_json TEXT NOT NULL,
    outcome TEXT CHECK (outcome IN ('confirmed', 'expired_absent', 'operator_acknowledged')),
    outcome_at TEXT,
    outcome_evidence TEXT,
    CHECK ((outcome IS NULL AND outcome_at IS NULL AND outcome_evidence IS NULL)
        OR (outcome IS NOT NULL AND outcome_at IS NOT NULL AND outcome_evidence IS NOT NULL))
);

CREATE UNIQUE INDEX authority_anchor_one_pending_per_epoch
ON authority_anchor_submissions(epoch_id) WHERE outcome IS NULL;

CREATE TRIGGER authority_anchor_submissions_bind_epoch
BEFORE INSERT ON authority_anchor_submissions
BEGIN
    SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1 FROM authority_epochs WHERE epoch_id = NEW.epoch_id
            AND log_digest = NEW.log_digest AND anchor_txid IS NULL
            AND epoch_id NOT IN (SELECT epoch_id FROM authority_epoch_tombstones)
        )
        THEN RAISE(ABORT, 'anchor submission must bind an indexed unanchored epoch')
        WHEN EXISTS (
            SELECT 1 FROM authority_anchor_submissions WHERE epoch_id = NEW.epoch_id
            AND (outcome IS NULL OR outcome <> 'expired_absent')
        )
        THEN RAISE(ABORT, 'only proven expired-absent anchors may have a fresh submission')
    END;
END;

CREATE TRIGGER authority_anchor_submissions_immutable
BEFORE UPDATE ON authority_anchor_submissions
BEGIN
    SELECT CASE
        WHEN OLD.extrinsic_hash IS NOT NEW.extrinsic_hash
          OR OLD.rowid IS NOT NEW.rowid
          OR OLD.epoch_id IS NOT NEW.epoch_id
          OR OLD.netuid IS NOT NEW.netuid
          OR OLD.log_digest IS NOT NEW.log_digest
          OR OLD.submission_json IS NOT NEW.submission_json
          OR OLD.outcome IS NOT NULL
        THEN RAISE(ABORT, 'anchor submission evidence and terminal outcomes are immutable')
    END;
END;

CREATE TRIGGER authority_anchor_submissions_no_delete
BEFORE DELETE ON authority_anchor_submissions
BEGIN
    SELECT RAISE(ABORT, 'anchor submission evidence is append-only');
END;
