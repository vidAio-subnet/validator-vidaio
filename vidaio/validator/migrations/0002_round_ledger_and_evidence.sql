-- Round atomicity, score-packet evidence, and in-flight challenge recovery.

--
-- #9  round persistence is non-atomic: `rounds` is the round ledger. A round row
--     is inserted (committed_at NULL) when the round starts and its committed_at
--     is stamped in the SAME BEGIN IMMEDIATE transaction that folds the round's
--     EWMA updates and packet evidence. A partial round is therefore both
--     impossible to observe (the transaction rolls back) and DETECTABLE
--     afterwards (committed_at stays NULL) — readers ignore uncommitted rounds.
--
-- #7  score evidence is transient: `score_packets` keeps the exact packet BYTES
--     and their digest per (round, uid, item), plus the request bindings the
--     packet was accepted against. `audit_ref` is the audit-store backend key
--     when the SCORE_PACKET artifact was archived; NULL means DB-only (no audit
--     store configured — the validator logs that at startup).
--
-- #5  challenges are never resolved: `inflight_challenges` records every
--     challenge the validator FETCHED but has not yet resolved with the
--     challenge service, so a crashed round's checked-out assets are resolved by
--     the next startup's recovery pass instead of stranding the pool in_use.

CREATE TABLE rounds (
  round_id     TEXT PRIMARY KEY,
  started_at   TEXT NOT NULL,
  block        INTEGER NOT NULL,
  -- NULL until the round's whole score/evidence write commits. Readers
  -- (weight-setter evidence queries) consider only non-NULL rounds.
  committed_at TEXT
);

CREATE INDEX idx_rounds_committed ON rounds(committed_at);

CREATE TABLE score_packets (
  round_id      TEXT NOT NULL REFERENCES rounds(round_id),
  uid           INTEGER NOT NULL,
  item_id       TEXT NOT NULL,
  challenge_id  TEXT NOT NULL,
  track         TEXT NOT NULL,
  miner_hotkey  TEXT NOT NULL,
  content_digest TEXT NOT NULL,
  -- sha256 of packet_json; this is the merkle leaf a publication commits to
  packet_digest TEXT NOT NULL,
  packet_json   TEXT NOT NULL,
  scorer_version TEXT NOT NULL,
  score         REAL NOT NULL,
  audit_ref     TEXT,
  created_at    TEXT NOT NULL,
  PRIMARY KEY (round_id, uid, item_id)
);

CREATE INDEX idx_score_packets_created ON score_packets(created_at);
CREATE INDEX idx_score_packets_digest ON score_packets(packet_digest);

CREATE TABLE inflight_challenges (
  challenge_id TEXT PRIMARY KEY,
  round_id     TEXT NOT NULL,
  track        TEXT NOT NULL,
  -- What the challenge service must be told when this row is drained. Starts as
  -- 'expired' (a round that never finished scoring this item did not resolve it)
  -- and flips to 'resolved' only once the track's scoring completed.
  outcome      TEXT NOT NULL DEFAULT 'expired'
                 CHECK (outcome IN ('resolved', 'expired')),
  fetched_at   TEXT NOT NULL
);
