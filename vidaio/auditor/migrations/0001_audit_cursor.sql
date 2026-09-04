-- The auditor's DURABLE audit cursor.
--
-- The auditor audits epochs CONTIGUOUSLY in ascending order (never just the
-- authority's latest pointer). A single-row table records the highest epoch this
-- auditor has AUDITED-AND-SUBMITTED; the next pass resumes at cursor + 1. It only
-- ever advances (monotonic, once per audited-and-submitted epoch), so:
--   - a restart resumes from the persisted cursor (survives process death);
--   - a HELD epoch (beacon not finalized yet / not anchored yet) does NOT advance
--     the cursor and is retried next pass — later epochs are never audited ahead of
--     it, so a malicious authority cannot race a fraudulent epoch past the auditor.
--
-- One row, pinned to id = 0 (CHECK enforces the singleton). Monotonic advance is
-- enforced in-database so even a direct SQL write cannot rewind the cursor (which
-- would silently re-open a window to skip an epoch).

CREATE TABLE audit_cursor (
    id                  INTEGER PRIMARY KEY CHECK (id = 0),
    last_audited_epoch  INTEGER NOT NULL
);

-- The cursor only moves FORWARD: an update that would lower (or hold) it is aborted.
CREATE TRIGGER audit_cursor_monotonic
BEFORE UPDATE ON audit_cursor
BEGIN
    SELECT CASE
        WHEN NEW.last_audited_epoch <= OLD.last_audited_epoch
        THEN RAISE(ABORT, 'audit cursor only advances (monotonic; never rewinds)')
    END;
END;
