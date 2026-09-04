-- The DURABLE scorer pin.
--
-- The scorer identity a validator binds to was held in memory only, so a restart
-- onto a differently-configured worker silently mixed two scorers' packets into
-- one EWMA accumulator — the exact drift the pin exists to prevent, invisible
-- because nothing outlived the process.
--
-- The pin is now a single durable row. It is written on FIRST successful
-- discovery and never rewritten: a later discovery that disagrees is a refusal
-- (the validator stops scoring and logs CRITICAL), not an update. An operator
-- who genuinely re-points the validator at another scorer must acknowledge it
-- explicitly (`validator.reset_scorer_pin`), which clears this row and says so
-- loudly — accumulators built under the previous scorer are then knowingly mixed.

CREATE TABLE scorer_pin (
  -- Exactly one row can ever exist: a validator has ONE scorer identity.
  id             INTEGER PRIMARY KEY CHECK (id = 1),
  scorer_version TEXT NOT NULL,
  pinned_at      TEXT NOT NULL,
  -- Which half of the contract took the pin: 'discovered' (pin on first contact)
  -- or 'operator' (validator.scorer_version was set and discovery agreed).
  source         TEXT NOT NULL DEFAULT 'discovered'
);
