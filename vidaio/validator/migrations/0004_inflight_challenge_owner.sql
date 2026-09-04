-- The in-flight challenge's OWNER.
--
-- `inflight_challenges` recorded the obligation to resolve a fetched challenge
-- but not WHO fetched it, so the startup recovery pass resolved every stranded
-- row with the validator's CURRENTLY CONFIGURED `validator.identity`.
--
-- The challenge service enforces ownership on resolve (403 `not_owner`,
-- round-2 new-4). So if identity A fetched a challenge, the process crashed, and
-- it came back as identity B — a key rotation, a copy-pasted config, two
-- validators sharing a data dir — recovery resolved as B against a challenge
-- owned by A and was refused. Forever: the row survives a failed resolve by
-- design, so it retried the same impossible call every round while the service's
-- asset stayed `in_use` with its commitment unrevealed.
--
-- The owner is therefore recorded WITH the obligation and recovery resolves with
-- the RECORDED owner, not the current config. Rows written before this migration
-- have owner '' and keep the old behaviour (resolve as whoever we are now),
-- which is exactly what they meant when they were written.

ALTER TABLE inflight_challenges ADD COLUMN owner TEXT NOT NULL DEFAULT '';
