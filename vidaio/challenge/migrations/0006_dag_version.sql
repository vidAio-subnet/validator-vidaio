-- Persist the committed procedural-DAG version as an indexed scalar instead of
-- requiring recovery/operations code to parse private dag_json. Existing rows
-- are version 5 or older; json_extract recovers their actual committed version.
-- New rows explicitly provide DAG_VERSION from record_challenge.

ALTER TABLE challenges
  ADD COLUMN dag_version INTEGER NOT NULL DEFAULT 6 CHECK (dag_version >= 1);

UPDATE challenges
SET dag_version = COALESCE(
  CAST(json_extract(dag_json, '$.dag_version') AS INTEGER),
  5
);

CREATE INDEX idx_challenges_dag_version_status
ON challenges(dag_version, status);

CREATE TRIGGER challenges_dag_version_matches_json
BEFORE INSERT ON challenges
WHEN json_type(NEW.dag_json, '$.dag_version') = 'integer'
 AND NEW.dag_version != CAST(json_extract(NEW.dag_json, '$.dag_version') AS INTEGER)
BEGIN
  SELECT RAISE(ABORT, 'challenge dag_version does not match dag_json');
END;

CREATE TRIGGER challenges_dag_version_immutable
BEFORE UPDATE OF dag_version ON challenges
WHEN OLD.dag_version IS NOT NEW.dag_version
BEGIN
  SELECT RAISE(ABORT, 'challenge dag_version is immutable');
END;
