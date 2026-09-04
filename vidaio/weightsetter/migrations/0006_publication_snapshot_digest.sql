-- Bind each durable weight intent/publication to the exact verified EpochLog.
-- Nullable preserves legacy/local report-mode rows, which predate shared snapshots.
-- The shared provider path requires a non-null digest before any chain write.

ALTER TABLE weight_intents ADD COLUMN snapshot_digest TEXT;
