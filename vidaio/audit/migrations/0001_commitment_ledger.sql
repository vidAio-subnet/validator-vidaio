-- Append-only commitment ledger. Immutability is enforced in-database:
-- UPDATE/DELETE on either table aborts via trigger. Status changes are
-- modeled as appended event rows in commitment_ledger_status, never as
-- edits, and their ordering is enforced in-database too: the first status
-- must be pending_chain, every later event must advance strictly forward
-- ONE step at a time (pending_chain -> anchored -> published; skipping
-- anchored is rejected), and every status timestamp must be >= the ledger
-- row's created_at and >= every earlier status timestamp — direct SQL
-- cannot backdate, regress, skip, or restart a commitment's status history.
-- Timestamp comparisons go through julianday(), which parses ISO-8601
-- '+HH:MM' offsets and normalizes to UTC, so they compare INSTANTS: an
-- offset cannot disguise a backdate (e.g. '09:00+05:00' is 04:00Z and is
-- rejected after a '08:00+00:00' event even though it sorts later as a
-- string). Unparseable timestamps (julianday() -> NULL) are rejected.
-- (Tables are namespaced commitment_ledger* so this migration can share a
-- database with the challenge module's own commitments table.)

CREATE TABLE commitment_ledger (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    kind           TEXT NOT NULL CHECK (kind IN ('competition', 'publication')),
    root_digest    TEXT NOT NULL CHECK (length(root_digest) = 64),
    payload        BLOB NOT NULL,
    canonical_json TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE commitment_ledger_status (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    commitment_id INTEGER NOT NULL REFERENCES commitment_ledger(id),
    status        TEXT NOT NULL CHECK (status IN ('pending_chain', 'anchored', 'published')),
    recorded_at   TEXT NOT NULL
);

CREATE INDEX idx_commitment_ledger_status_commitment
    ON commitment_ledger_status(commitment_id, id);

CREATE TRIGGER commitment_ledger_no_update
BEFORE UPDATE ON commitment_ledger
BEGIN
    SELECT RAISE(ABORT, 'commitment ledger is append-only');
END;

CREATE TRIGGER commitment_ledger_no_delete
BEFORE DELETE ON commitment_ledger
BEGIN
    SELECT RAISE(ABORT, 'commitment ledger is append-only');
END;

CREATE TRIGGER commitment_ledger_status_no_update
BEFORE UPDATE ON commitment_ledger_status
BEGIN
    SELECT RAISE(ABORT, 'commitment status history is append-only');
END;

CREATE TRIGGER commitment_ledger_status_no_delete
BEFORE DELETE ON commitment_ledger_status
BEGIN
    SELECT RAISE(ABORT, 'commitment status history is append-only');
END;

CREATE TRIGGER commitment_ledger_created_at_parseable
BEFORE INSERT ON commitment_ledger
BEGIN
    SELECT RAISE(ABORT, 'commitment created_at is not a parseable ISO-8601 timestamp')
    WHERE julianday(NEW.created_at) IS NULL;
END;

CREATE TRIGGER commitment_ledger_status_forward_only
BEFORE INSERT ON commitment_ledger_status
BEGIN
    SELECT RAISE(ABORT, 'status timestamp is not a parseable ISO-8601 timestamp')
    WHERE julianday(NEW.recorded_at) IS NULL;
    SELECT RAISE(ABORT, 'first commitment status must be pending_chain')
    WHERE NEW.status <> 'pending_chain'
      AND NOT EXISTS (
          SELECT 1 FROM commitment_ledger_status
          WHERE commitment_id = NEW.commitment_id
      );
    SELECT RAISE(ABORT, 'commitment status may only advance forward')
    WHERE EXISTS (
        SELECT 1 FROM commitment_ledger_status s
        WHERE s.commitment_id = NEW.commitment_id
          AND (CASE s.status
                   WHEN 'pending_chain' THEN 0
                   WHEN 'anchored' THEN 1
                   ELSE 2
               END)
              >= (CASE NEW.status
                      WHEN 'pending_chain' THEN 0
                      WHEN 'anchored' THEN 1
                      ELSE 2
                  END)
    );
    SELECT RAISE(ABORT, 'commitment status may not skip anchored')
    WHERE NEW.status = 'published'
      AND NOT EXISTS (
          SELECT 1 FROM commitment_ledger_status
          WHERE commitment_id = NEW.commitment_id
            AND status = 'anchored'
      );
    SELECT RAISE(ABORT, 'status timestamp must not precede commitment creation')
    WHERE julianday(NEW.recorded_at) < (
        SELECT julianday(created_at) FROM commitment_ledger
        WHERE id = NEW.commitment_id
    );
    SELECT RAISE(ABORT, 'status timestamps must be monotonically non-decreasing')
    WHERE EXISTS (
        SELECT 1 FROM commitment_ledger_status s
        WHERE s.commitment_id = NEW.commitment_id
          AND julianday(s.recorded_at) > julianday(NEW.recorded_at)
    );
END;
