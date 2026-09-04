-- Per-identity bearer credentials (the sim's stand-in for hotkey signatures —
-- see the authorization section of vidaio/chainsim/service.py's module docstring).
--
-- Only the SHA-256 of the token is stored: the sim can verify a presented token
-- but can never hand one back out. It is returned exactly once, in the response
-- to the registration that CLAIMS the hotkey.
--
-- Rows migrated from a pre-auth sim database land here with token_sha256 = ''
-- ("unclaimed"): they cannot authenticate anything, and the next /register for
-- that hotkey claims them. If that grandfathering matters to you, delete the old
-- chainsim.db instead — it is simulator state, not chain history.
ALTER TABLE neurons ADD COLUMN token_sha256 TEXT NOT NULL DEFAULT '';

-- Lookup path for /report/write ("is this any registered identity's token?").
CREATE INDEX idx_neurons_token ON neurons (token_sha256);
