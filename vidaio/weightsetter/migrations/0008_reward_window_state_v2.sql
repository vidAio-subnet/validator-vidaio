-- Schema-v14 Tokenomics V2 persistence. Legacy v1 state/result tables remain inert and
-- readable for rollback/diagnostics; no economic state is derived from them.

CREATE TABLE IF NOT EXISTS reward_window_state (
    id                          INTEGER PRIMARY KEY CHECK (id = 1),
    kind                        TEXT NOT NULL CHECK (kind IN ('IDLE', 'PODIUM', 'CROWN')),
    starts_at                   TEXT,
    ends_at                     TEXT,
    podium_hotkeys_json         TEXT NOT NULL,
    winner_hotkey               TEXT,
    winner_uid                  INTEGER CHECK (winner_uid IS NULL OR winner_uid >= 0),
    winner_score                REAL,
    winner_margin               REAL,
    baseline_score              REAL,
    baseline_version            INTEGER CHECK (
                                      baseline_version IS NULL OR baseline_version >= 0
                                  ),
    baseline_artifact_digest    TEXT,
    source_competition_id       TEXT,
    source_track                TEXT,
    source_cycle                INTEGER CHECK (source_cycle IS NULL OR source_cycle >= 1),
    last_applied_cycle          INTEGER CHECK (
                                      last_applied_cycle IS NULL OR last_applied_cycle >= 1
                                  )
);

CREATE TABLE IF NOT EXISTS competition_results_v2 (
    cycle                       INTEGER PRIMARY KEY CHECK (cycle >= 1),
    competition_id              TEXT NOT NULL UNIQUE,
    track                       TEXT NOT NULL,
    applied_at                  TEXT NOT NULL,
    contenders_json             TEXT NOT NULL,
    baseline_score              REAL,
    baseline_version            INTEGER NOT NULL CHECK (baseline_version >= 0),
    baseline_artifact_digest    TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS competition_results_v2_no_update
BEFORE UPDATE ON competition_results_v2
BEGIN
    SELECT RAISE(ABORT, 'competition_results_v2 rows are immutable');
END;

CREATE TRIGGER IF NOT EXISTS competition_results_v2_no_delete
BEFORE DELETE ON competition_results_v2
BEGIN
    SELECT RAISE(ABORT, 'competition_results_v2 rows are immutable');
END;
