-- The packet-evidence WATERMARK and the unresolved-attempt bookkeeping
--.
--
-- new-6. Consecutive publications are supposed to PARTITION the score-packet
-- evidence: each one commits to the packets produced since the previous one. The
-- lower bound used to be the previous intent's `settled_at` — the moment its
-- anchor finally succeeded. But its packet list was FROZEN much earlier, when the
-- intent was recorded. Every packet created in between (an anchor that hung for
-- minutes, or failed and was re-driven a whole cycle later) belonged to NEITHER
-- publication: not the first, whose list was already closed, and not the second,
-- which starts after the first one settled. That evidence gap is permanent and
-- silent. `packets_frozen_at` records when the list was actually captured, and
-- THAT is the next publication's lower bound.
--
-- #10. `last_checked_at` / `last_check` record what the most recent chain
-- confirmation said about a still-pending intent. A pending intent whose fate the
-- chain cannot decide (UNKNOWN) must never be abandoned — a vector that may be
-- live on chain has to stay publishable — so it is re-checked on later passes and
-- these columns are how an operator (and the bounded-age abandon rule) can see
-- how long that has been going on and on what evidence.

ALTER TABLE weight_intents ADD COLUMN packets_frozen_at TEXT;
ALTER TABLE weight_intents ADD COLUMN last_checked_at TEXT;
ALTER TABLE weight_intents ADD COLUMN last_check TEXT;

CREATE INDEX idx_weight_intents_watermark
  ON weight_intents(state, packets_frozen_at);
