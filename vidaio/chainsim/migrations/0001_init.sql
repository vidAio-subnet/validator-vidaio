-- Chain simulator state. Restart-safe: everything the sim knows lives here.

CREATE TABLE meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE neurons (
  uid INTEGER PRIMARY KEY,
  hotkey TEXT NOT NULL UNIQUE,
  coldkey TEXT NOT NULL,
  ip TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('miner', 'validator')),
  alpha_stake REAL NOT NULL DEFAULT 0.0,
  -- cumulative emission credited by the sim's lazy settlement
  emission_credited REAL NOT NULL DEFAULT 0.0,
  registered_block INTEGER NOT NULL
);

-- Accepted set_weights calls only (tempo-rejected calls are not recorded,
-- mirroring InMemoryChain.weight_calls).
CREATE TABLE weight_calls (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  hotkey TEXT NOT NULL,
  block INTEGER NOT NULL,
  version_key INTEGER NOT NULL,
  vector_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_weight_calls_hotkey_block ON weight_calls (hotkey, block);

CREATE TABLE anchors (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  txid TEXT NOT NULL,
  payload_hex TEXT NOT NULL,
  hotkey TEXT,
  block INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
