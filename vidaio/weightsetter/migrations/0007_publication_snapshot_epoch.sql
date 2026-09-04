-- Preserve the exact authority epoch needed to recover post-submit publication
-- leaves when best-effort pre-submit capture failed. Weight submission must not be
-- gated by audit/publication evidence, but an accepted intent must also never publish
-- the empty-set sentinel merely because the provider later advanced to a newer epoch.

ALTER TABLE weight_intents ADD COLUMN snapshot_epoch_id INTEGER;
