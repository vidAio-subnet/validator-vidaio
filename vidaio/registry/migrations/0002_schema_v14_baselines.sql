-- Schema-v14 executable baseline registry.
--
-- The historical `champions` tables remain readable during the rollout, but every
-- schema-v14 write uses this ledger.  Version zero is the public reference
-- implementation installed by `seed_genesis_baselines`; later versions can only be
-- activated from verified CROWN epoch evidence (or by an explicit, append-only
-- rollback).

CREATE TABLE baselines (
    baseline_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    track                    TEXT NOT NULL CHECK (track IN ('compression', 'upscaling')),
    version                  INTEGER NOT NULL CHECK (version >= 0),
    artifact_digest          TEXT NOT NULL CHECK (
        length(artifact_digest) = 64
        AND artifact_digest NOT GLOB '*[^0-9a-f]*'
    ),
    artifact_kind            TEXT NOT NULL,
    artifact_bytes           INTEGER NOT NULL CHECK (artifact_bytes > 0),
    image_digest             TEXT NOT NULL CHECK (
        length(image_digest) = 64
        AND image_digest NOT GLOB '*[^0-9a-f]*'
    ),
    provenance_digest        TEXT NOT NULL CHECK (
        length(provenance_digest) = 64
        AND provenance_digest NOT GLOB '*[^0-9a-f]*'
    ),
    provenance_kind          TEXT NOT NULL,
    provenance_bytes         INTEGER NOT NULL CHECK (provenance_bytes > 0),
    repo_url                 TEXT NOT NULL CHECK (length(repo_url) > 0),
    commit_sha               TEXT NOT NULL CHECK (
        length(commit_sha) = 40
        AND commit_sha NOT GLOB '*[^0-9a-f]*'
    ),
    tree_sha                 TEXT NOT NULL CHECK (
        length(tree_sha) = 40
        AND tree_sha NOT GLOB '*[^0-9a-f]*'
    ),
    source_kind              TEXT NOT NULL CHECK (source_kind IN ('genesis', 'crown', 'rollback')),
    source_epoch_id          TEXT,
    source_snapshot_digest   TEXT CHECK (
        source_snapshot_digest IS NULL
        OR (length(source_snapshot_digest) = 64
            AND source_snapshot_digest NOT GLOB '*[^0-9a-f]*')
    ),
    source_anchor_block      INTEGER CHECK (source_anchor_block IS NULL OR source_anchor_block >= 0),
    source_anchor_digest     TEXT CHECK (
        source_anchor_digest IS NULL
        OR (length(source_anchor_digest) = 64
            AND source_anchor_digest NOT GLOB '*[^0-9a-f]*')
    ),
    source_competition_id    TEXT,
    source_cycle             INTEGER CHECK (source_cycle IS NULL OR source_cycle >= 1),
    winner_uid               INTEGER CHECK (winner_uid IS NULL OR winner_uid >= 0),
    winner_hotkey            TEXT,
    winner_score             REAL CHECK (winner_score IS NULL OR (winner_score >= 0.0 AND winner_score <= 1.0)),
    winner_margin            REAL CHECK (winner_margin IS NULL OR winner_margin >= 0.0),
    compared_baseline_version INTEGER CHECK (
        compared_baseline_version IS NULL OR compared_baseline_version >= 0
    ),
    compared_baseline_score   REAL CHECK (
        compared_baseline_score IS NULL
        OR (compared_baseline_score > 0.0 AND compared_baseline_score <= 1.0)
    ),
    compared_baseline_digest TEXT CHECK (
        compared_baseline_digest IS NULL
        OR (length(compared_baseline_digest) = 64
            AND compared_baseline_digest NOT GLOB '*[^0-9a-f]*')
    ),
    status                   TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'rolled_back')),
    reinstated_version       INTEGER CHECK (reinstated_version IS NULL OR reinstated_version >= 0),
    rollback_reason          TEXT,
    activated_at             TEXT NOT NULL,
    updated_at               TEXT NOT NULL,
    UNIQUE (track, version),
    CHECK ((reinstated_version IS NULL) = (rollback_reason IS NULL)),
    CHECK (
        (source_kind = 'genesis' AND version = 0
         AND source_epoch_id IS NULL
         AND source_snapshot_digest IS NULL
         AND source_anchor_block IS NULL
         AND source_anchor_digest IS NULL
         AND source_competition_id IS NULL
         AND source_cycle IS NULL
         AND winner_uid IS NULL
         AND winner_hotkey IS NULL
         AND winner_score IS NULL
         AND winner_margin IS NULL
         AND compared_baseline_version IS NULL
         AND compared_baseline_score IS NULL
         AND compared_baseline_digest IS NULL
         AND reinstated_version IS NULL)
        OR
        (source_kind = 'crown' AND version >= 1
         AND source_epoch_id IS NOT NULL
         AND source_snapshot_digest IS NOT NULL
         AND source_anchor_block IS NOT NULL
         AND source_anchor_digest IS NOT NULL
         AND source_competition_id IS NOT NULL
         AND source_cycle IS NOT NULL
         AND winner_uid IS NOT NULL
         AND winner_hotkey IS NOT NULL
         AND winner_score IS NOT NULL
         AND winner_margin IS NOT NULL
         AND compared_baseline_version IS NOT NULL
         AND compared_baseline_score IS NOT NULL
         AND compared_baseline_digest IS NOT NULL
         AND reinstated_version IS NULL)
        OR
        (source_kind = 'rollback' AND version >= 1
         AND reinstated_version IS NOT NULL)
    )
);

CREATE UNIQUE INDEX ux_baselines_one_active
    ON baselines(track) WHERE status = 'active';
CREATE INDEX ix_baselines_track_version ON baselines(track, version);
CREATE UNIQUE INDEX ux_baselines_crown_idempotence
    ON baselines(source_snapshot_digest, source_competition_id, track)
    WHERE source_kind = 'crown';

-- A CROWN is latched before its executable is built/rerun.  The partial unique
-- index is the orchestrator interlock: one unresolved CROWN per track, and a next
-- competition is refused while this row exists in `pending` state.
CREATE TABLE baseline_promotion_latches (
    latch_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    track                    TEXT NOT NULL CHECK (track IN ('compression', 'upscaling')),
    snapshot_digest          TEXT NOT NULL CHECK (
        length(snapshot_digest) = 64
        AND snapshot_digest NOT GLOB '*[^0-9a-f]*'
    ),
    competition_id           TEXT NOT NULL,
    epoch_id                 TEXT NOT NULL,
    cycle                    INTEGER NOT NULL CHECK (cycle >= 1),
    anchor_block             INTEGER NOT NULL CHECK (anchor_block >= 0),
    anchor_digest            TEXT NOT NULL CHECK (
        length(anchor_digest) = 64
        AND anchor_digest NOT GLOB '*[^0-9a-f]*'
    ),
    winner_uid               INTEGER NOT NULL CHECK (winner_uid >= 0),
    winner_hotkey            TEXT NOT NULL,
    compared_baseline_version INTEGER NOT NULL CHECK (compared_baseline_version >= 0),
    compared_baseline_digest TEXT NOT NULL CHECK (
        length(compared_baseline_digest) = 64
        AND compared_baseline_digest NOT GLOB '*[^0-9a-f]*'
    ),
    status                   TEXT NOT NULL CHECK (status IN ('pending', 'promoted')),
    promoted_baseline_id     INTEGER REFERENCES baselines(baseline_id),
    latched_at               TEXT NOT NULL,
    resolved_at              TEXT,
    UNIQUE (snapshot_digest, competition_id, track),
    CHECK (
        (status = 'pending' AND promoted_baseline_id IS NULL AND resolved_at IS NULL)
        OR
        (status = 'promoted' AND promoted_baseline_id IS NOT NULL AND resolved_at IS NOT NULL)
    )
);

CREATE UNIQUE INDEX ux_baseline_latches_one_pending
    ON baseline_promotion_latches(track) WHERE status = 'pending';

CREATE TABLE baseline_events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    track          TEXT NOT NULL CHECK (track IN ('compression', 'upscaling')),
    event_type     TEXT NOT NULL,
    version        INTEGER,
    snapshot_digest TEXT,
    payload_json   TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE INDEX ix_baseline_events_track ON baseline_events(track, event_id);

CREATE TRIGGER baseline_events_append_only_update
BEFORE UPDATE ON baseline_events
BEGIN
    SELECT RAISE(ABORT, 'baseline_events is append-only');
END;

CREATE TRIGGER baseline_events_append_only_delete
BEFORE DELETE ON baseline_events
BEGIN
    SELECT RAISE(ABORT, 'baseline_events is append-only');
END;

CREATE TRIGGER baselines_immutable_columns
BEFORE UPDATE ON baselines
WHEN NEW.track != OLD.track
     OR NEW.version != OLD.version
     OR NEW.artifact_digest != OLD.artifact_digest
     OR NEW.artifact_kind != OLD.artifact_kind
     OR NEW.artifact_bytes != OLD.artifact_bytes
     OR NEW.image_digest != OLD.image_digest
     OR NEW.provenance_digest != OLD.provenance_digest
     OR NEW.provenance_kind != OLD.provenance_kind
     OR NEW.provenance_bytes != OLD.provenance_bytes
     OR NEW.repo_url != OLD.repo_url
     OR NEW.commit_sha != OLD.commit_sha
     OR NEW.tree_sha != OLD.tree_sha
     OR NEW.source_kind != OLD.source_kind
     OR NEW.source_epoch_id IS NOT OLD.source_epoch_id
     OR NEW.source_snapshot_digest IS NOT OLD.source_snapshot_digest
     OR NEW.source_anchor_block IS NOT OLD.source_anchor_block
     OR NEW.source_anchor_digest IS NOT OLD.source_anchor_digest
     OR NEW.source_competition_id IS NOT OLD.source_competition_id
     OR NEW.source_cycle IS NOT OLD.source_cycle
     OR NEW.winner_uid IS NOT OLD.winner_uid
     OR NEW.winner_hotkey IS NOT OLD.winner_hotkey
     OR NEW.winner_score IS NOT OLD.winner_score
     OR NEW.winner_margin IS NOT OLD.winner_margin
     OR NEW.compared_baseline_version IS NOT OLD.compared_baseline_version
     OR NEW.compared_baseline_score IS NOT OLD.compared_baseline_score
     OR NEW.compared_baseline_digest IS NOT OLD.compared_baseline_digest
     OR NEW.reinstated_version IS NOT OLD.reinstated_version
     OR NEW.rollback_reason IS NOT OLD.rollback_reason
     OR NEW.activated_at != OLD.activated_at
BEGIN
    SELECT RAISE(ABORT, 'baseline rows are immutable except status/updated_at');
END;

CREATE TRIGGER baselines_no_reactivation
BEFORE UPDATE OF status ON baselines
WHEN OLD.status != 'active' AND NEW.status != OLD.status
BEGIN
    SELECT RAISE(ABORT, 'a terminal baseline row cannot be reactivated');
END;

CREATE TRIGGER baseline_latches_immutable_proof
BEFORE UPDATE ON baseline_promotion_latches
WHEN NEW.track != OLD.track
     OR NEW.snapshot_digest != OLD.snapshot_digest
     OR NEW.competition_id != OLD.competition_id
     OR NEW.epoch_id != OLD.epoch_id
     OR NEW.cycle != OLD.cycle
     OR NEW.anchor_block != OLD.anchor_block
     OR NEW.anchor_digest != OLD.anchor_digest
     OR NEW.winner_uid != OLD.winner_uid
     OR NEW.winner_hotkey != OLD.winner_hotkey
     OR NEW.compared_baseline_version != OLD.compared_baseline_version
     OR NEW.compared_baseline_digest != OLD.compared_baseline_digest
     OR NEW.latched_at != OLD.latched_at
BEGIN
    SELECT RAISE(ABORT, 'promotion latch proof is immutable');
END;

CREATE TRIGGER baseline_latches_no_reopen
BEFORE UPDATE OF status ON baseline_promotion_latches
WHEN OLD.status = 'promoted' AND NEW.status != OLD.status
BEGIN
    SELECT RAISE(ABORT, 'a promoted latch cannot be reopened');
END;
