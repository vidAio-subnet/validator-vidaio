-- The weight-setter OWN-AUDIT gate's DURABLE own-audited-CLEAN ledger.
--
-- Finding #1 (CRITICAL): the own-audit SUBMIT gate re-folds the CURRENT epoch's earning
-- state and verifies its nonzero carry-in against the PREDECESSOR's stated
-- accumulate_score, but it never verified that the predecessor was ITSELF own-audited
-- CLEAN. An untrusted authority could publish a structurally-invalid predecessor carrying
-- an INJECTED accumulator, chain it into a self-consistent current epoch, and slip the
-- current own-audit (the separate public auditor loop may HOLD/dispute the predecessor
-- later, but that never blocks THIS submission).
--
-- This table is the durable, contiguous chain of (epoch_id, log_digest) entries the gate
-- has PREVIOUSLY cleared CLEAN. Before clearing an epoch whose earning carry-in is NONZERO
-- for any uid, the gate REQUIRES the predecessor (epoch_id-1, prior_log_digest) to be a
-- recorded CLEAN entry here; otherwise the carry-in is UNVERIFIED and the gate HOLDS (fail
-- closed) — it never vouches for a carry-in it did not itself audit. On a CLEAN clear the
-- gate RECORDS this epoch, so the chain extends contiguously (genesis has a zero carry-in
-- and needs no predecessor).
--
-- One row per own-audited-CLEAN epoch, keyed by epoch_id (an epoch has one canonical
-- log_digest). Advance is monotonic-forward like the auditor cursor: the gate layer makes a
-- re-record of the SAME (epoch_id, log_digest) idempotent, and this trigger aborts any
-- INSERT of an epoch_id at/below the highest recorded one in-database, so the ledger can
-- never be rewound (which would re-open a window to launder an unaudited carry-in).

CREATE TABLE own_audit_clean (
    epoch_id    INTEGER PRIMARY KEY,
    log_digest  TEXT NOT NULL
);

CREATE TRIGGER own_audit_clean_forward
BEFORE INSERT ON own_audit_clean
WHEN (SELECT MAX(epoch_id) FROM own_audit_clean) >= NEW.epoch_id
BEGIN
    SELECT RAISE(ABORT, 'own-audit-clean ledger only extends forward');
END;
