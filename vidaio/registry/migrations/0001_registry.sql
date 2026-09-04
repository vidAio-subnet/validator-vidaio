-- Model registry (spec: design spec §20): the archived champion executable per track
-- and its promotion history. Rows are NEVER deleted or rewritten: a promotion
-- appends a new version and flips the previous active row to 'superseded'; a
-- rollback flips the active row to 'rolled_back' and APPENDS a fresh version
-- reinstating an earlier artifact — versions are monotonic per track and the
-- full audit trail is the table itself (plus the append-only event log).

CREATE TABLE champions (
    champion_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    track                 TEXT NOT NULL,
    version               INTEGER NOT NULL CHECK (version >= 1),
    -- Audit-store reference to the archived champion executable (LocalFsStore /
    -- Hippius): digest + kind + byte size fully reconstruct the ArtifactRef.
    artifact_digest       TEXT NOT NULL CHECK (length(artifact_digest) = 64
                                               AND artifact_digest NOT GLOB '*[^0-9a-f]*'),
    artifact_kind         TEXT NOT NULL,
    artifact_bytes        INTEGER NOT NULL CHECK (artifact_bytes >= 0),
    source_competition_id TEXT NOT NULL,
    contender_hotkey      TEXT NOT NULL,
    -- The HIDDEN-HOLDOUT score that won promotion (design spec §20: the holdout
    -- winner is promoted, never a public-board winner). Scores are [0, 1].
    holdout_score         REAL NOT NULL CHECK (holdout_score >= 0.0 AND holdout_score <= 1.0),
    -- Verified audit-bundle linkage: promotion without it is impossible
    -- (NOT NULL here, typed error in the promotion pipeline).
    audit_bundle_digest   TEXT NOT NULL CHECK (length(audit_bundle_digest) = 64
                                               AND audit_bundle_digest NOT GLOB '*[^0-9a-f]*'),
    status                TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'rolled_back')),
    -- Set only on rollback-appended rows: which earlier version this reinstates.
    reinstated_version    INTEGER CHECK (reinstated_version IS NULL
                                         OR reinstated_version >= 1),
    rollback_reason       TEXT,
    promoted_at           TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    UNIQUE (track, version),
    -- A rollback row carries both provenance fields; a promotion row neither.
    CHECK ((reinstated_version IS NULL) = (rollback_reason IS NULL))
);

-- Exactly one serving champion per track at any time.
CREATE UNIQUE INDEX ux_champions_one_active ON champions (track) WHERE status = 'active';
CREATE INDEX ix_champions_track_version ON champions (track, version);

-- Append-only registry event log: every promote/supersede/rollback lands here.
CREATE TABLE registry_events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    track        TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    version      INTEGER,
    payload_json TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX ix_registry_events_track ON registry_events (track, event_id);

CREATE TRIGGER registry_events_append_only_update BEFORE UPDATE ON registry_events
BEGIN
    SELECT RAISE(ABORT, 'registry_events is append-only');
END;

CREATE TRIGGER registry_events_append_only_delete BEFORE DELETE ON registry_events
BEGIN
    SELECT RAISE(ABORT, 'registry_events is append-only');
END;

-- History rows are immutable except for the status lifecycle (active ->
-- superseded | rolled_back) — artifact identity, provenance, and score can
-- never be edited after promotion.
CREATE TRIGGER champions_immutable_columns
BEFORE UPDATE ON champions
WHEN NEW.track != OLD.track
     OR NEW.version != OLD.version
     OR NEW.artifact_digest != OLD.artifact_digest
     OR NEW.artifact_kind != OLD.artifact_kind
     OR NEW.artifact_bytes != OLD.artifact_bytes
     OR NEW.source_competition_id != OLD.source_competition_id
     OR NEW.contender_hotkey != OLD.contender_hotkey
     OR NEW.holdout_score != OLD.holdout_score
     OR NEW.audit_bundle_digest != OLD.audit_bundle_digest
     OR NEW.promoted_at != OLD.promoted_at
     OR NEW.reinstated_version IS NOT OLD.reinstated_version
     OR NEW.rollback_reason IS NOT OLD.rollback_reason
BEGIN
    SELECT RAISE(ABORT, 'champion rows are immutable except status/updated_at');
END;

-- Terminal statuses stay terminal: once superseded/rolled_back, a row can
-- never become active again (reinstatement APPENDS a new version instead).
CREATE TRIGGER champions_no_reactivation
BEFORE UPDATE OF status ON champions
WHEN OLD.status != 'active' AND NEW.status != OLD.status
BEGIN
    SELECT RAISE(ABORT, 'a superseded/rolled_back champion row can never change status');
END;
