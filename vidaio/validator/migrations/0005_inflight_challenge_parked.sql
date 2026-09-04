-- PARKED in-flight challenges.
--
-- A genuine ownership refusal (403 `not_owner`) on /challenge/{id}/resolve is
-- PERMANENT: no number of retries can move an ownership boundary. Round-3 #5
-- stopped filing it under the flaky-resolve failures, but the row itself stayed
-- in the ordinary drain selection — so every round, and every restart, re-picked
-- it, attempted the same impossible resolve, bumped the metric and logged the
-- same WARNING. Forever.
--
-- `parked_at` is the durable answer: set (with `park_reason`) when the service
-- positively refuses ownership. Parked rows are EXCLUDED from the drain/recovery
-- selection but never deleted — the row is the only record that a service-side
-- asset is stranded, so it stays visible through `parked_challenges()`, the
-- `vidaio_validator_parked_challenges` gauge, and the startup recovery log line.
--
-- An operator clears them, after fixing (or accepting) the service-side
-- ownership state, via `validator.unpark_challenges = true` at startup or the
-- `InferenceValidator.unpark_challenges()` admin method: unparking returns the
-- rows to the normal drain, which either resolves them or — if the boundary
-- still stands — parks them again on the next 403.
--
-- NULL = live (the shape every pre-migration row already has, so the backfill
-- is implicit and correct: nothing was parked before parking existed).

ALTER TABLE inflight_challenges ADD COLUMN parked_at TEXT;
ALTER TABLE inflight_challenges ADD COLUMN park_reason TEXT NOT NULL DEFAULT '';
