-- Durable, database-serialized allocation for pre-committed dispatch order.
--
-- A process-local `time.time_ns()` counter can regress after a restart when the
-- host clock moves backwards. Schema-v11 auditors persist fold watermarks, so a
-- regressed key makes genuinely new work look replayed. Seed this one-row
-- allocator from every existing commitment during upgrade, then advance it under
-- BEGIN IMMEDIATE for each new challenge. Failed media work may leave harmless
-- gaps; a key is never reused.

CREATE TABLE dispatch_order_allocator (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  last_key  INTEGER NOT NULL CHECK (last_key >= 0)
);

INSERT INTO dispatch_order_allocator(singleton, last_key)
SELECT 1, COALESCE(MAX(dispatch_ordering_key), 0)
FROM challenge_commitments;
