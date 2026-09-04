-- Evidence-backed miner availability zeros.
--
-- These rows are committed in the same transaction as the score=0 EWMA fold and
-- post-round miner-state history. They are not media score packets: observation_json
-- contains the exact validator-signed artifact-v2 request, finalized challenge anchor,
-- target hotkey/endpoint, bounded deadline and protocol-enumerated failure reason.

CREATE TABLE availability_folds (
  round_id           TEXT NOT NULL REFERENCES rounds(round_id),
  uid                INTEGER NOT NULL,
  item_id            TEXT NOT NULL,
  challenge_id       TEXT NOT NULL,
  track              TEXT NOT NULL,
  miner_hotkey       TEXT NOT NULL,
  endpoint            TEXT NOT NULL,
  reason              TEXT NOT NULL CHECK (reason IN (
                        'timeout',
                        'transport_error',
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

CREATE INDEX idx_availability_folds_created
  ON availability_folds(created_at);
CREATE INDEX idx_availability_folds_digest
  ON availability_folds(observation_digest);
