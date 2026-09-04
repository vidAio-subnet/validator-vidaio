-- Immutable post-round miner-state snapshots for close-block-pinned finalization.
--
-- The `miners` table is the live head. Using it while catching up an older epoch
-- relabels later EWMA folds and hotkey/track changes as state at that epoch's close
-- block. Every committed round now records the complete staged registry view in
-- the same transaction as its scores and evidence. The authority can therefore
-- select the newest state whose round block is <= the requested close block.

CREATE TABLE miner_state_history (
  round_id          TEXT NOT NULL REFERENCES rounds(round_id),
  block             INTEGER NOT NULL,
  uid               INTEGER NOT NULL,
  hotkey            TEXT NOT NULL,
  coldkey           TEXT NOT NULL,
  ip                TEXT NOT NULL,
  track             TEXT,
  accumulate_score  REAL NOT NULL,
  committed_at      TEXT NOT NULL,
  PRIMARY KEY (round_id, uid)
);

CREATE INDEX idx_miner_state_history_block_uid
  ON miner_state_history(block, uid, committed_at);
