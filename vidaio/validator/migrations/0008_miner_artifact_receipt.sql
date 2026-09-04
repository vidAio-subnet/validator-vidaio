-- Miner-signed chronology evidence for each accepted score packet. Artifact-v2
-- binds the finalized challenge commitment receipt into its signed request, then
-- binds that request digest into the signed response. The authority copies this
-- JSON into the audit bundle; third-party auditors verify the signature and the
-- chain receipt independently.
ALTER TABLE score_packets ADD COLUMN miner_receipt_json TEXT;
