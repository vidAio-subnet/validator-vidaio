-- Bind dispatched challenges to their commitment and give them a resolution
-- lifecycle (dispatched -> resolved | expired).
--
--   * dag_digest is stored on the challenge row and a BEFORE INSERT trigger
--     requires (asset_id, dag_digest) to equal the referenced commitment's
--     (clean_asset_id, dag_digest) — a challenge can never claim a commitment
--     that binds different private material.
--   * commit_hash is UNIQUE: one challenge per commitment, commitments cannot
--     be reused across dispatches.
--   * status drives reveal safety: reveal_commitment refuses while any
--     challenge on the asset is still 'dispatched'.
--
-- The 0001 challenges table has no deployed data anywhere, so it is rebuilt
-- in place rather than ALTERed.

DROP TABLE challenges;

CREATE TABLE challenges (
  challenge_id TEXT PRIMARY KEY,
  track        TEXT NOT NULL,
  asset_id     TEXT NOT NULL REFERENCES assets(id),
  commit_hash  TEXT NOT NULL UNIQUE REFERENCES challenge_commitments(commit_hash),
  dag_digest   TEXT NOT NULL,
  dag_json     TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'dispatched'
               CHECK (status IN ('dispatched', 'resolved', 'expired')),
  created_at   TEXT NOT NULL,
  resolved_at  TEXT
);

CREATE INDEX idx_challenges_asset_status ON challenges(asset_id, status);

CREATE TRIGGER challenges_match_commitment
BEFORE INSERT ON challenges
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM challenge_commitments c
    WHERE c.commit_hash = NEW.commit_hash
      AND c.clean_asset_id = NEW.asset_id
      AND c.dag_digest = NEW.dag_digest
  )
  THEN RAISE(ABORT, 'challenge does not match its commitment (asset_id/dag_digest)')
  END;
END;

-- Identity immutability: the insert-time binding above would be worthless if a
-- later UPDATE could rewrite what was bound. Once inserted, everything except
-- the resolution lifecycle (status, resolved_at) is frozen at the DB layer.
CREATE TRIGGER challenges_identity_immutable
BEFORE UPDATE ON challenges
WHEN OLD.challenge_id IS NOT NEW.challenge_id
  OR OLD.track        IS NOT NEW.track
  OR OLD.asset_id     IS NOT NEW.asset_id
  OR OLD.commit_hash  IS NOT NEW.commit_hash
  OR OLD.dag_digest   IS NOT NEW.dag_digest
  OR OLD.dag_json     IS NOT NEW.dag_json
  OR OLD.created_at   IS NOT NEW.created_at
BEGIN
  SELECT RAISE(ABORT, 'challenge identity columns are immutable (only status/resolved_at may change)');
END;
