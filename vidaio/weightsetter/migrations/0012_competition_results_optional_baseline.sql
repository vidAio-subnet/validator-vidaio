-- A competition may be anchored without an executable baseline (its crown and podium
-- are decided by the anchored absolute score bars). Such a result carries no baseline
-- version or artifact, so the two columns become nullable. SQLite cannot relax NOT NULL
-- in place: the table is rebuilt, every row copied verbatim, and the immutability
-- triggers recreated. Nothing references this table.

CREATE TABLE competition_results_v2_v12 (
    cycle                       INTEGER PRIMARY KEY CHECK (cycle >= 1),
    competition_id              TEXT NOT NULL UNIQUE,
    track                       TEXT NOT NULL,
    applied_at                  TEXT NOT NULL,
    contenders_json             TEXT NOT NULL,
    baseline_score              REAL,
    baseline_version            INTEGER CHECK (
                                      baseline_version IS NULL OR baseline_version >= 0
                                  ),
    baseline_artifact_digest    TEXT,
    CHECK ((baseline_version IS NULL) = (baseline_artifact_digest IS NULL))
);
INSERT INTO competition_results_v2_v12
    (cycle, competition_id, track, applied_at, contenders_json, baseline_score,
     baseline_version, baseline_artifact_digest)
  SELECT cycle, competition_id, track, applied_at, contenders_json, baseline_score,
         baseline_version, baseline_artifact_digest
  FROM competition_results_v2;
DROP TABLE competition_results_v2;
ALTER TABLE competition_results_v2_v12 RENAME TO competition_results_v2;

CREATE TRIGGER competition_results_v2_no_update
BEFORE UPDATE ON competition_results_v2
BEGIN
    SELECT RAISE(ABORT, 'competition_results_v2 rows are immutable');
END;

CREATE TRIGGER competition_results_v2_no_delete
BEFORE DELETE ON competition_results_v2
BEGIN
    SELECT RAISE(ABORT, 'competition_results_v2 rows are immutable');
END;
