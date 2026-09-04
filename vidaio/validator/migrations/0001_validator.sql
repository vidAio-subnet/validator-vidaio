-- Validator state: the miner registry (EWMA accumulator + warrant-probed track).
-- (The block-driven retention_windows table was REMOVED with the retention multiplier
-- for v1 — retention removed — owner decision; an internal review.
-- Pre-release, so 0001 is adjusted cleanly rather than shipping a drop-table follow-up,
.)

CREATE TABLE miners (
  uid              INTEGER PRIMARY KEY,
  hotkey           TEXT NOT NULL,
  coldkey          TEXT NOT NULL,
  ip               TEXT NOT NULL,
  -- Track comes ONLY from a recorded warrant probe result. NULL = unknown; an
  -- unknown-track miner is SKIPPED for the round, never bucketed by default
  -- (the deliberate fix of the old validator.py:844 default-to-upscaling bug).
  track            TEXT CHECK (track IN ('compression', 'upscaling') OR track IS NULL),
  accumulate_score REAL NOT NULL DEFAULT 0.0,
  first_seen_block INTEGER NOT NULL
);

CREATE INDEX idx_miners_track ON miners(track);
