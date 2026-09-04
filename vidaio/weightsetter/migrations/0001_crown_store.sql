-- Weight-setter persistence: breakthrough crown state + ingested competition results.
-- CrownState is a singleton row (id = 1) mirroring vidaio.tokenomics.state.CrownState;
-- competition_results holds one immutable row per completed cycle (the weight loop
-- only ever composes against the latest one). All timestamps are ISO-8601 strings
-- supplied by callers — crown resolution uses the result's own completed_at, never
-- wall-clock (the project design record bright lines; deterministic across replays).

CREATE TABLE crown_state (
    id                 INTEGER PRIMARY KEY CHECK (id = 1),
    champion_hotkey    TEXT,
    champion_uid       INTEGER,
    crowned_at         TEXT,
    margin             REAL,
    watermark          REAL NOT NULL DEFAULT 0.0,
    reign_tranche      REAL NOT NULL DEFAULT 0.0,
    podium_hotkeys     TEXT NOT NULL DEFAULT '[]',  -- JSON array, champion first
    last_applied_cycle INTEGER
);

CREATE TABLE competition_results (
    cycle                     INTEGER PRIMARY KEY CHECK (cycle >= 0),
    completed_at              TEXT NOT NULL,
    contenders_json           TEXT NOT NULL,  -- JSON array [{hotkey, uid, margin}], ranked best-first
    baseline_score            REAL,
    prev_champion_rerun_score REAL
);

-- Competition results are audit inputs: one row per cycle, never rewritten in
-- place. Re-ingesting a cycle with different content is rejected in code
-- (ResultConflictError); these triggers make silent rewrites a schema error too.
CREATE TRIGGER competition_results_no_update
BEFORE UPDATE ON competition_results
BEGIN
    SELECT RAISE(ABORT, 'competition_results rows are immutable');
END;

CREATE TRIGGER competition_results_no_delete
BEFORE DELETE ON competition_results
BEGIN
    SELECT RAISE(ABORT, 'competition_results rows are immutable');
END;
