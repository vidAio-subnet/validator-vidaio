-- P1.5 outage-gap recovery (epoch schema v16): a crash between `record_finalized`
-- and `anchor_epoch`, followed by an outage past the epoch's un-grindable anchor
-- window (`close + K`), leaves an indexed-but-UNANCHORED row that can never be
-- anchored (a late anchor is grindable and is refused forever). Before v16 this
-- wedged the spine permanently ("explicit operator remediation"). The remediation
-- now has a name: the operator ACKNOWLEDGES the epoch as an outage gap; the row is
-- tombstoned (never deleted — the audit trail keeps the exact orphaned pointer),
-- the spine resumes from the previous ANCHORED epoch, and the next finalized log
-- declares the tombstoned epoch in its `gap_epochs`.
--
-- A tombstone is only ever legal for an UNANCHORED epoch: an anchored epoch is
-- consensus history and can never become a gap. Tombstones are themselves
-- append-only and immutable.

CREATE TABLE authority_epoch_tombstones (
    epoch_id        INTEGER PRIMARY KEY REFERENCES authority_epochs(epoch_id),
    acknowledged_at TEXT NOT NULL,
    reason          TEXT NOT NULL
);

CREATE TRIGGER authority_epoch_tombstones_unanchored_only
BEFORE INSERT ON authority_epoch_tombstones
BEGIN
    SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1 FROM authority_epochs WHERE epoch_id = NEW.epoch_id
        )
        THEN RAISE(ABORT, 'cannot tombstone an epoch that is not indexed')
        WHEN EXISTS (
            SELECT 1 FROM authority_epochs
            WHERE epoch_id = NEW.epoch_id AND anchor_txid IS NOT NULL
        )
        THEN RAISE(ABORT, 'cannot tombstone an ANCHORED epoch — anchored history is immutable')
    END;
END;

CREATE TRIGGER authority_epoch_tombstones_immutable
BEFORE UPDATE ON authority_epoch_tombstones
BEGIN
    SELECT RAISE(ABORT, 'epoch tombstones are immutable');
END;

CREATE TRIGGER authority_epoch_tombstones_permanent
BEFORE DELETE ON authority_epoch_tombstones
BEGIN
    SELECT RAISE(ABORT, 'epoch tombstones are permanent (audit trail)');
END;
