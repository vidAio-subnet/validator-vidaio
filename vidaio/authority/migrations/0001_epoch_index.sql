-- The Scoring Authority's epoch INDEX: which epochs are finalized, and the
-- pointer (object-store key + digests) + on-chain anchor for each. This is a
-- thin index, NOT the content plane — the epoch-log bytes live in the object
-- store, addressed by `snapshot_key` (the project design record §3.1, wave 4).
--
-- Append-only / immutable: a finalized epoch's identity fields (close_block,
-- snapshot_key, log_digest, weight_vector_digest, finalized_at) can never change
-- once written — a finalized set is immutable (the finalizer's wave-2 invariant),
-- so a row that tried to rewrite them is aborted in-database. The anchor columns
-- (anchor_txid, anchor_block) are the ONE exception: they start NULL at finalize
-- and are filled in ONCE when the log_digest is anchored on chain, then frozen
-- (a second, different anchor is rejected). Direct SQL cannot rewrite a pointer
-- or re-anchor an epoch.

CREATE TABLE authority_epochs (
    epoch_id             INTEGER PRIMARY KEY,
    close_block          INTEGER NOT NULL,
    snapshot_key         TEXT NOT NULL,
    log_digest           TEXT NOT NULL CHECK (length(log_digest) = 64),
    weight_vector_digest TEXT NOT NULL CHECK (length(weight_vector_digest) = 64),
    anchor_txid          TEXT,
    anchor_block         INTEGER,
    finalized_at         TEXT NOT NULL
);

-- Immutability of the finalized pointer fields (anchor columns may transition
-- NULL -> value exactly once; everything else is frozen at insert).
CREATE TRIGGER authority_epochs_immutable
BEFORE UPDATE ON authority_epochs
BEGIN
    SELECT CASE
        WHEN OLD.epoch_id             <> NEW.epoch_id
          OR OLD.close_block          <> NEW.close_block
          OR OLD.snapshot_key         <> NEW.snapshot_key
          OR OLD.log_digest           <> NEW.log_digest
          OR OLD.weight_vector_digest <> NEW.weight_vector_digest
          OR OLD.finalized_at         <> NEW.finalized_at
        THEN RAISE(ABORT, 'authority epoch pointer is immutable once finalized')
        WHEN OLD.anchor_txid IS NOT NULL AND NEW.anchor_txid IS NOT OLD.anchor_txid
        THEN RAISE(ABORT, 'authority epoch is already anchored; the anchor cannot change')
        WHEN OLD.anchor_block IS NOT NULL AND NEW.anchor_block IS NOT OLD.anchor_block
        THEN RAISE(ABORT, 'authority epoch anchor block is already set')
    END;
END;

CREATE TRIGGER authority_epochs_no_delete
BEFORE DELETE ON authority_epochs
BEGIN
    SELECT RAISE(ABORT, 'authority epoch index is append-only');
END;
