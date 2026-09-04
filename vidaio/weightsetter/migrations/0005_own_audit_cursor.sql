-- The weight-setter OWN-AUDIT gate's DURABLE contiguous cursor.
--
-- Finding #2 (HIGH): the round-12 own-audit gate resolved only the authority's LATEST
-- pointer. If the weight-setter SKIPPED an epoch (weightsetting runs ~30s while epochs are
-- ~20s), the next positive-carry epoch HOLDs FOREVER — its predecessor is absent from the
-- own-audited-CLEAN ledger and later passes only ever fetch still-newer epochs, so the gap
-- can never be filled (a restart with an empty ledger mid-chain wedges the same way).
--
-- This single-row cursor mirrors the public auditor's `audit_cursor` (vidaio/auditor/
-- migrations/0001_audit_cursor.sql): it records the HIGHEST epoch the gate has own-audited
-- CLEAN CONTIGUOUSLY. Each attempt walks cursor+1 .. latest, BACKFILLING every missed epoch
-- (fetched by `pointer_for(epoch_id)`), own-auditing it, recording it in the ledger, and
-- advancing this cursor ONLY after a CLEAN clear. So the ledger stays gap-free and a
-- positive-carry epoch's predecessor is always present when the chain is honest — a genuinely
-- withheld/unavailable predecessor still HOLDs (fail closed), but an honest contiguous chain
-- never wedges. Durable, so a restart resumes exactly where it left off.
--
-- One row, pinned to id = 0 (CHECK enforces the singleton). Monotonic advance is enforced
-- in-database so even a direct SQL write cannot rewind the cursor (which would silently
-- re-open a skip window).

CREATE TABLE own_audit_cursor (
    id                 INTEGER PRIMARY KEY CHECK (id = 0),
    last_clean_epoch   INTEGER NOT NULL
);

-- The cursor only moves FORWARD: an update that would lower (or hold) it is aborted.
CREATE TRIGGER own_audit_cursor_monotonic
BEFORE UPDATE ON own_audit_cursor
BEGIN
    SELECT CASE
        WHEN NEW.last_clean_epoch <= OLD.last_clean_epoch
        THEN RAISE(ABORT, 'own-audit cursor only advances (monotonic; never rewinds)')
    END;
END;
