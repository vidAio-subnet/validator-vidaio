-- Finalized-chain receipt for every challenge commitment that may be dispatched.
--
-- Bittensor's Commitments pallet is a single mutable slot per (netuid, account),
-- so the inclusion block is part of the durable receipt: later verification pins
-- archive state to that exact block after newer challenge/epoch anchors overwrite
-- the head slot.  Rows are append-only and the trigger binds the receipt's lookup
-- identifier to the ordering key already hashed into the commitment preimage.

CREATE TABLE challenge_commitment_anchors (
  commit_hash           TEXT PRIMARY KEY
                        REFERENCES challenge_commitments(commit_hash),
  netuid                INTEGER NOT NULL CHECK (netuid >= 0),
  dispatch_ordering_key INTEGER NOT NULL CHECK (dispatch_ordering_key >= 1),
  anchor_block          INTEGER NOT NULL CHECK (anchor_block >= 0),
  anchor_block_hash     TEXT CHECK (
                        anchor_block_hash IS NULL OR
                        (length(anchor_block_hash) = 64 AND
                         anchor_block_hash NOT GLOB '*[^0-9a-f]*')
                        ),
  anchor_txid           TEXT,
  anchored_at           TEXT NOT NULL
);

CREATE TRIGGER challenge_commitment_anchor_matches
BEFORE INSERT ON challenge_commitment_anchors
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM challenge_commitments c
    WHERE c.commit_hash = NEW.commit_hash
      AND c.dispatch_ordering_key = NEW.dispatch_ordering_key
  )
  THEN RAISE(ABORT, 'challenge anchor does not match commitment ordering key')
  END;
END;

CREATE TRIGGER challenge_commitment_anchors_no_update
BEFORE UPDATE ON challenge_commitment_anchors
BEGIN
  SELECT RAISE(ABORT, 'challenge commitment anchors are append-only');
END;

CREATE TRIGGER challenge_commitment_anchors_no_delete
BEFORE DELETE ON challenge_commitment_anchors
BEGIN
  SELECT RAISE(ABORT, 'challenge commitment anchors are append-only');
END;
