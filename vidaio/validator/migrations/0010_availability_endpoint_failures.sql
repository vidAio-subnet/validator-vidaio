-- Extend the closed availability taxonomy for two miner-controlled refusal modes.
-- SQLite cannot alter a CHECK constraint in place, so rebuild the evidence table
-- transactionally while preserving every existing signed observation.

ALTER TABLE availability_folds RENAME TO availability_folds_v1;

CREATE TABLE availability_folds (
  round_id           TEXT NOT NULL REFERENCES rounds(round_id),
  uid                INTEGER NOT NULL,
  item_id            TEXT NOT NULL,
  challenge_id       TEXT NOT NULL,
  track               TEXT NOT NULL,
  miner_hotkey       TEXT NOT NULL,
  endpoint            TEXT NOT NULL,
  reason              TEXT NOT NULL CHECK (reason IN (
                        'timeout',
                        'transport_error',
                        'restart_fence_exhausted',
                        'unreachable_endpoint',
                        'protocol_error',
                        'task_id_mismatch',
                        'output_digest_mismatch',
                        'receipt_invalid'
                      )),
  score               REAL NOT NULL DEFAULT 0.0 CHECK (score = 0.0),
  observation_digest  TEXT NOT NULL CHECK (
                        length(observation_digest) = 64
                        AND observation_digest NOT GLOB '*[^0-9a-f]*'
                      ),
  observation_json    TEXT NOT NULL,
  created_at          TEXT NOT NULL,
  PRIMARY KEY (round_id, uid, item_id)
);

INSERT INTO availability_folds (
  round_id, uid, item_id, challenge_id, track, miner_hotkey, endpoint, reason,
  score, observation_digest, observation_json, created_at
)
SELECT
  round_id, uid, item_id, challenge_id, track, miner_hotkey, endpoint, reason,
  score, observation_digest, observation_json, created_at
FROM availability_folds_v1;

DROP TABLE availability_folds_v1;

CREATE INDEX idx_availability_folds_created
  ON availability_folds(created_at);
CREATE INDEX idx_availability_folds_digest
  ON availability_folds(observation_digest);
