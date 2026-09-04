-- Competition schema — spec: the design spec §06, redesigned for auditability.
-- The old schema declared only 2 FKs (competitions->contender_metadata and the
-- human-review links); here every logically-required relation is a real FK, all
-- ON DELETE RESTRICT: competition data is audit data and is never deleted in place.
-- Child tables that reference a contender/item/batch/sandbox do so through COMPOSITE
-- (competition_id, id) foreign keys so a row can never point at an entity belonging
-- to a DIFFERENT competition (cross-competition forgery is a schema error, not a
-- code-review hope). All timestamps are ISO-8601 UTC strings written by the engine
-- (no wall-clock defaults in logic paths — `now` is always passed in explicitly).

CREATE TABLE competitions (
    competition_id        TEXT PRIMARY KEY,
    track                 TEXT NOT NULL,
    status                TEXT NOT NULL CHECK (status IN (
                              'SCHEDULED','ENROLLING','FINALIZING_SUBMISSIONS','VALIDATING',
                              'BUILDING','EVALUATING','SCORING','AWAITING_END_TIME',
                              'COMPLETED','FAILED','CANCELLED')),
    manifest_json         TEXT NOT NULL,
    manifest_digest       TEXT NOT NULL,             -- sha256(canonical manifest JSON); pre-committed on chain
    -- sha256 root of the anchored pre-commitment (manifest digest is part of that
    -- commitment upstream). NULL at creation; enrollment can NEVER open while NULL —
    -- the SCHEDULED->ENROLLING guard requires it (engine.mark_commitment_anchored).
    commitment_root       TEXT CHECK (commitment_root IS NULL
                                      OR (length(commitment_root) = 64
                                          AND commitment_root NOT GLOB '*[^0-9a-f]*')),
    start_time            TEXT NOT NULL,
    enrollment_deadline   TEXT NOT NULL,
    finalization_time     TEXT NOT NULL,
    end_time              TEXT NOT NULL,
    human_review_deadline TEXT,                      -- set when scores are persisted (now + review window)
    failure_reason        TEXT,
    -- Constant column backing the single-running-competition invariant below.
    running_guard         INTEGER NOT NULL DEFAULT 1 CHECK (running_guard = 1),
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);

-- Single-running-competition invariant (spec §04): at most one row may hold a
-- "running" status at any time. SCHEDULED does not occupy the slot; terminal
-- statuses free it.
CREATE UNIQUE INDEX ux_competitions_single_running ON competitions (running_guard)
    WHERE status IN ('ENROLLING','FINALIZING_SUBMISSIONS','VALIDATING','BUILDING',
                     'EVALUATING','SCORING','AWAITING_END_TIME');

CREATE INDEX ix_competitions_status ON competitions (status);

CREATE TABLE contenders (
    contender_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id      TEXT NOT NULL REFERENCES competitions (competition_id) ON DELETE RESTRICT,
    hotkey              TEXT,                        -- NULL iff calibration (baseline has no payout identity)
    is_calibration      INTEGER NOT NULL DEFAULT 0 CHECK (is_calibration IN (0, 1)),
    repo_url            TEXT NOT NULL,
    commit_sha          TEXT NOT NULL,
    tree_sha            TEXT NOT NULL,
    image_digest        TEXT,                        -- stable logical build identity once built (spec §05)
    status              TEXT NOT NULL DEFAULT 'ENROLLED' CHECK (status IN (
                            'ENROLLED','ACCEPTED','REJECTED','BUILT','BUILD_FAILED')),
    enrollment_stake    REAL NOT NULL DEFAULT 0 CHECK (enrollment_stake >= 0),
    eligible            INTEGER NOT NULL DEFAULT 1 CHECK (eligible IN (0, 1)),
    manual_disqualified INTEGER NOT NULL DEFAULT 0 CHECK (manual_disqualified IN (0, 1)),
    -- final_score contributors (spec §06 contender_metadata):
    final_score              REAL,
    final_rank               INTEGER CHECK (final_rank IS NULL OR final_rank >= 1),
    media_score_aggregate    REAL,                   -- length-weighted mean media score (item lengths from evaluation_items)
    worst_decile_aggregate   REAL,                   -- worst-decile mean of per-item scores (spec §18 bottleneck aggregation)
    cost_efficiency_aggregate REAL,                  -- mean over items of min(1, item-cheapest-valid / own cost)
    length_coverage          REAL,                   -- length-weighted completion, clamped to [0, 1]
    average_vmaf             REAL,
    average_compression_rate REAL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    -- Composite parent for same-competition child FKs (sandboxes/batches/
    -- performance_history/human_reviews reference contenders through this pair).
    UNIQUE (competition_id, contender_id),
    -- Calibration rows have no hotkey; real contenders must have one.
    CHECK ((is_calibration = 1 AND hotkey IS NULL) OR (is_calibration = 0 AND hotkey IS NOT NULL)),
    -- Baseline is a NON-EARNING calibration baseline (the project design record #1): it is
    -- evaluated but excluded from ranking/podium/payout BY CONSTRUCTION — a
    -- calibration row can never carry a final_rank.
    CHECK (is_calibration = 0 OR final_rank IS NULL)
);

CREATE UNIQUE INDEX ux_contenders_hotkey ON contenders (competition_id, hotkey)
    WHERE hotkey IS NOT NULL;
-- At most one calibration (baseline) contender per competition.
CREATE UNIQUE INDEX ux_contenders_one_calibration ON contenders (competition_id)
    WHERE is_calibration = 1;
CREATE INDEX ix_contenders_competition_status ON contenders (competition_id, status);
CREATE INDEX ix_contenders_ranking ON contenders (competition_id, final_rank)
    WHERE final_rank IS NOT NULL;

CREATE TABLE evaluation_items (
    item_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id       TEXT NOT NULL REFERENCES competitions (competition_id) ON DELETE RESTRICT,
    item_index           INTEGER NOT NULL CHECK (item_index >= 0),
    input_sha256         TEXT NOT NULL,              -- raw bytes are archived to the audit store, not the DB
    input_bytes          INTEGER NOT NULL CHECK (input_bytes >= 0),
    length_seconds       REAL CHECK (length_seconds IS NULL OR length_seconds > 0),
    threshold_commitment TEXT NOT NULL,              -- sha256 commitment to this item's sealed vmaf variant
    sealed_vmaf_threshold REAL,                      -- NULL until revealed post-competition
    -- Packet-binding identity (spec §14 auditability): a persisted score packet's
    -- (challenge_id, item_id) MUST equal this pair — record_item_score refuses
    -- packets minted for another challenge or item.
    challenge_id         TEXT NOT NULL,
    scoring_item_id      TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    UNIQUE (competition_id, item_index),
    -- Composite parent for the performance_history same-competition FK.
    UNIQUE (competition_id, item_id)
);

CREATE TABLE sandboxes (
    sandbox_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id       TEXT NOT NULL REFERENCES competitions (competition_id) ON DELETE RESTRICT,
    contender_id         INTEGER,
    provider_ref         TEXT,                       -- opaque runner handle (e.g. Modal sandbox id)
    image_digest         TEXT,
    status               TEXT NOT NULL DEFAULT 'CREATED' CHECK (status IN (
                             'CREATED','RUNNING','ROLLED_OVER','TERMINATED','FAILED')),
    isolation_probe_json TEXT,                       -- serialized IsolationProbeReport (spec §05)
    created_at           TEXT NOT NULL,
    expires_at           TEXT,                       -- rollover before the ~23h30m lifetime cap
    terminated_at        TEXT,
    -- Composite parent for the batches same-competition FK.
    UNIQUE (competition_id, sandbox_id),
    -- Same-competition integrity: the sandbox's contender must belong to the
    -- sandbox's competition (NULL contender_id = shared sandbox, FK not applied).
    FOREIGN KEY (competition_id, contender_id)
        REFERENCES contenders (competition_id, contender_id) ON DELETE RESTRICT
);

CREATE INDEX ix_sandboxes_competition ON sandboxes (competition_id, status);

CREATE TABLE batches (
    batch_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id TEXT NOT NULL REFERENCES competitions (competition_id) ON DELETE RESTRICT,
    contender_id   INTEGER NOT NULL,
    sandbox_id     INTEGER,
    batch_index    INTEGER NOT NULL CHECK (batch_index >= 0),
    status         TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN (
                       'PENDING','CLAIMED','RUNNING','COMPLETED','FAILED','REQUEUED')),
    failure_code   TEXT,
    started_at     TEXT,
    finished_at    TEXT,
    created_at     TEXT NOT NULL,
    UNIQUE (contender_id, batch_index),
    -- Composite parent for the performance_history same-competition FK.
    UNIQUE (competition_id, batch_id),
    -- Same-competition integrity: contender and sandbox must belong to this
    -- batch's competition.
    FOREIGN KEY (competition_id, contender_id)
        REFERENCES contenders (competition_id, contender_id) ON DELETE RESTRICT,
    FOREIGN KEY (competition_id, sandbox_id)
        REFERENCES sandboxes (competition_id, sandbox_id) ON DELETE RESTRICT
);

CREATE INDEX ix_batches_competition_status ON batches (competition_id, status);

CREATE TABLE performance_history (
    performance_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id      TEXT NOT NULL REFERENCES competitions (competition_id) ON DELETE RESTRICT,
    contender_id        INTEGER NOT NULL,
    item_id             INTEGER NOT NULL,
    batch_id            INTEGER,
    vmaf                REAL,
    compression_rate    REAL,
    cost                REAL CHECK (cost IS NULL OR cost >= 0),
    length_seconds      REAL CHECK (length_seconds IS NULL OR length_seconds > 0),
    valid               INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0, 1)),
    item_score          REAL NOT NULL,               -- the packet's top-level score (record_item_score derives it)
    output_sha256       TEXT,
    output_bytes        INTEGER CHECK (output_bytes IS NULL OR output_bytes >= 0),
    -- sha256 of the EXACT ItemScore packet bytes the score was derived from
    -- (recompute audit). NOT NULL: a score without its packet digest cannot exist.
    -- typeof guard: a 64-byte BLOB containing a NUL would pass length()+GLOB
    -- (GLOB stops at NUL), so non-text values must be rejected explicitly.
    score_packet_digest TEXT NOT NULL CHECK (typeof(score_packet_digest) = 'text'
                                             AND length(score_packet_digest) = 64
                                             AND score_packet_digest NOT GLOB '*[^0-9a-f]*'),
    -- Per-(contender, item) audit-bundle linkage (§08): each contender×item output
    -- has its own bundle. NULL until the audit run links it (engine check hook).
    -- When set it must be a real sha256 hex digest — an empty/malformed value can
    -- never satisfy the completion gate — and it is WRITE-ONCE (trigger below).
    audit_bundle_digest TEXT CHECK (audit_bundle_digest IS NULL
                                    OR (typeof(audit_bundle_digest) = 'text'
                                        AND length(audit_bundle_digest) = 64
                                        AND audit_bundle_digest NOT GLOB '*[^0-9a-f]*')),
    created_at          TEXT NOT NULL,
    UNIQUE (contender_id, item_id),
    -- Gates-first invariant (spec §18): a gate-failed row persists score 0, always.
    CHECK (valid = 1 OR item_score = 0),
    -- Same-competition integrity: contender, item and batch must all belong to
    -- this row's competition (cross-competition score forgery is impossible).
    FOREIGN KEY (competition_id, contender_id)
        REFERENCES contenders (competition_id, contender_id) ON DELETE RESTRICT,
    FOREIGN KEY (competition_id, item_id)
        REFERENCES evaluation_items (competition_id, item_id) ON DELETE RESTRICT,
    FOREIGN KEY (competition_id, batch_id)
        REFERENCES batches (competition_id, batch_id) ON DELETE RESTRICT
);

CREATE INDEX ix_performance_competition_contender
    ON performance_history (competition_id, contender_id);
CREATE INDEX ix_performance_item ON performance_history (item_id);

-- Audit linkage is WRITE-ONCE: once a row carries its audit_bundle_digest, the
-- digest can never be changed or cleared — completion evidence is immutable
-- (re-writing the SAME value is allowed: idempotent re-link).
CREATE TRIGGER performance_audit_digest_write_once
BEFORE UPDATE OF audit_bundle_digest ON performance_history
WHEN OLD.audit_bundle_digest IS NOT NULL
     AND NEW.audit_bundle_digest IS NOT OLD.audit_bundle_digest
BEGIN
    SELECT RAISE(ABORT, 'audit_bundle_digest is write-once');
END;

-- Append-only event log: every lifecycle transition and notable action lands here.
CREATE TABLE events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id TEXT NOT NULL REFERENCES competitions (competition_id) ON DELETE RESTRICT,
    event_type     TEXT NOT NULL,
    from_phase     TEXT,
    to_phase       TEXT,
    guard          TEXT,
    payload_json   TEXT,
    created_at     TEXT NOT NULL
);

CREATE INDEX ix_events_competition ON events (competition_id, event_id);

CREATE TRIGGER events_append_only_update BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only');
END;

CREATE TRIGGER events_append_only_delete BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only');
END;

-- Append-only, hash-chained human reviews (spec §06): per row,
-- integrity_hash = sha256(prev_row_hash || canonical_row_json). The chain is
-- per-competition; verification recomputes it end to end (repository.verify_review_chain).
CREATE TABLE human_reviews (
    review_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id       TEXT NOT NULL REFERENCES competitions (competition_id) ON DELETE RESTRICT,
    contender_id         INTEGER NOT NULL,
    action               TEXT NOT NULL CHECK (action IN ('DISQUALIFY','REINSTATE','TIE_BREAK')),
    reviewer             TEXT NOT NULL,
    reason               TEXT NOT NULL,
    detail_json          TEXT,
    supersedes_review_id INTEGER,
    prev_hash            TEXT NOT NULL,
    integrity_hash       TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    -- Composite parent for the same-competition supersedes FK below.
    UNIQUE (competition_id, review_id),
    -- Same-competition integrity: the reviewed contender and any superseded review
    -- must belong to this review's competition (a review in competition B can never
    -- supersede — i.e. silence — a review in competition A).
    FOREIGN KEY (competition_id, contender_id)
        REFERENCES contenders (competition_id, contender_id) ON DELETE RESTRICT,
    FOREIGN KEY (competition_id, supersedes_review_id)
        REFERENCES human_reviews (competition_id, review_id) ON DELETE RESTRICT
);

CREATE INDEX ix_reviews_competition ON human_reviews (competition_id, review_id);
CREATE INDEX ix_reviews_supersedes ON human_reviews (supersedes_review_id)
    WHERE supersedes_review_id IS NOT NULL;

CREATE TRIGGER human_reviews_append_only_update BEFORE UPDATE ON human_reviews
BEGIN
    SELECT RAISE(ABORT, 'human_reviews is append-only');
END;

CREATE TRIGGER human_reviews_append_only_delete BEFORE DELETE ON human_reviews
BEGIN
    SELECT RAISE(ABORT, 'human_reviews is append-only');
END;
