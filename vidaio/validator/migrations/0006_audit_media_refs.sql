-- Persist the exact media refs behind every accepted packet. The authority
-- finalizer consumes these to build genuinely recomputable POST_RETIREMENT
-- bundles; absolute filesystem paths never cross the service boundary.
ALTER TABLE score_packets ADD COLUMN challenge_input_ref TEXT;
ALTER TABLE score_packets ADD COLUMN miner_output_ref TEXT;
ALTER TABLE score_packets ADD COLUMN reference_original_ref TEXT;
