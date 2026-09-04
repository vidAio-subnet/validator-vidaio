-- Challenge module schema: content pool, append-only provenance log,
-- commit-reveal commitments, and dispatched challenges.

CREATE TABLE assets (
  id                     TEXT PRIMARY KEY,
  content_digest         TEXT NOT NULL UNIQUE,
  perceptual_fingerprint TEXT NOT NULL,
  source_url             TEXT NOT NULL,
  license_basis          TEXT NOT NULL,
  ingest_date            TEXT NOT NULL,
  creator                TEXT NOT NULL,
  source                 TEXT NOT NULL,
  subject                TEXT NOT NULL DEFAULT '',
  scene                  TEXT NOT NULL DEFAULT '',
  resolution_tag         TEXT NOT NULL,
  motion_tag             TEXT NOT NULL,
  content_type_tag       TEXT NOT NULL,
  metadata_stripped      INTEGER NOT NULL DEFAULT 0,
  split                  TEXT NOT NULL CHECK (split IN ('challenge', 'holdout')),
  -- Lifecycle: ingesting -> fresh -> in_use -> (fresh ... ->) retired.
  -- Newly registered assets are 'ingesting' until every planned ingest step
  -- (fetch, transcode, segment) is confirmed; only 'fresh' assets are issuable.
  status                 TEXT NOT NULL DEFAULT 'ingesting'
                         CHECK (status IN ('ingesting', 'fresh', 'in_use', 'retired')),
  use_count              INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_assets_status_split ON assets(status, split);

-- Rights/provenance log: append-only per asset. Updates and deletes are rejected
-- at the database layer, not just by convention.
CREATE TABLE provenance_log (
  seq         INTEGER PRIMARY KEY AUTOINCREMENT,
  asset_id    TEXT NOT NULL REFERENCES assets(id),
  event       TEXT NOT NULL,
  detail      TEXT NOT NULL DEFAULT '{}',
  recorded_at TEXT NOT NULL
);

CREATE TRIGGER provenance_log_no_update
BEFORE UPDATE ON provenance_log
BEGIN
  SELECT RAISE(ABORT, 'provenance_log is append-only');
END;

CREATE TRIGGER provenance_log_no_delete
BEFORE DELETE ON provenance_log
BEGIN
  SELECT RAISE(ABORT, 'provenance_log is append-only');
END;

-- Ingest-confirmation idempotence at the DB layer: each completion fact may be
-- recorded at most once per asset. Partial UNIQUE index scoped to exactly the
-- confirmation event names — ordinary provenance events (checked_out, released,
-- retired, ...) repeat freely. A duplicate confirm therefore fails inside
-- SQLite itself, not merely at the Python read that precedes the insert.
CREATE UNIQUE INDEX idx_provenance_ingest_confirm_once
ON provenance_log(asset_id, event)
WHERE event IN ('fetch_completed', 'transcode_completed', 'segment_completed',
                'metadata_stripped', 'ingest_confirmed');

-- Commit-reveal: committed before dispatch, revealed only after asset retirement.
-- seed is TEXT so arbitrary-precision Python ints round-trip exactly.
-- Named challenge_commitments (not commitments): the audit module's ledger shares
-- the core database, and both modules must co-apply on one connection.
CREATE TABLE challenge_commitments (
  commit_hash    TEXT PRIMARY KEY,
  clean_asset_id TEXT NOT NULL REFERENCES assets(id),
  dag_digest     TEXT NOT NULL,
  seed           TEXT NOT NULL,
  scorer_version TEXT NOT NULL,
  committed_at   TEXT NOT NULL,
  revealed_at    TEXT
);

-- A challenge row cannot exist without its commitment row (commit-before-dispatch).
CREATE TABLE challenges (
  challenge_id TEXT PRIMARY KEY,
  track        TEXT NOT NULL,
  asset_id     TEXT NOT NULL REFERENCES assets(id),
  commit_hash  TEXT NOT NULL REFERENCES challenge_commitments(commit_hash),
  dag_json     TEXT NOT NULL,
  created_at   TEXT NOT NULL
);
