-- Modal provider Image ids are opaque per-build handles, not stable content
-- identities.  Keep the protocol-facing logical build digest separate from the
-- exact provider object restored after a process restart.  This typed,
-- append-only ledger is the authoritative ownership binding; the matching event
-- remains the human/operator chronology.

CREATE TABLE modal_image_bindings (
    binding_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id         TEXT NOT NULL
                           REFERENCES competitions (competition_id) ON DELETE RESTRICT,
    contender_id           INTEGER CHECK (
                               contender_id IS NULL OR contender_id >= 0
                           ),
    is_calibration         INTEGER NOT NULL CHECK (is_calibration IN (0, 1)),
    repo_url               TEXT NOT NULL CHECK (length(repo_url) > 0),
    commit_sha             TEXT NOT NULL CHECK (
                               length(commit_sha) = 40
                               AND commit_sha NOT GLOB '*[^0-9a-f]*'
                           ),
    tree_sha               TEXT NOT NULL CHECK (
                               length(tree_sha) = 40
                               AND tree_sha NOT GLOB '*[^0-9a-f]*'
                           ),
    build_identity_scheme  TEXT NOT NULL CHECK (
                               build_identity_scheme =
                               'vidaio.competition.logical-build.v1'
                           ),
    image_digest           TEXT NOT NULL CHECK (
                               length(image_digest) = 64
                               AND image_digest NOT GLOB '*[^0-9a-f]*'
                           ),
    provider               TEXT NOT NULL CHECK (provider = 'modal'),
    image_object_id        TEXT NOT NULL CHECK (
                               length(image_object_id) BETWEEN 4 AND 131
                               AND substr(image_object_id, 1, 3) = 'im-'
                               AND substr(image_object_id, 4)
                                   NOT GLOB '*[^A-Za-z0-9_-]*'
                           ),
    runtime_session_id     TEXT NOT NULL CHECK (
                               length(runtime_session_id) = 64
                               AND runtime_session_id NOT GLOB '*[^0-9a-f]*'
                           ),
    runtime_label          TEXT NOT NULL CHECK (
                               runtime_label GLOB 'vidaio-next-*'
                           ),
    created_at             TEXT NOT NULL,
    UNIQUE (
        competition_id,
        contender_id,
        is_calibration,
        runtime_session_id,
        image_object_id
    )
);

CREATE INDEX ix_modal_image_bindings_lookup
    ON modal_image_bindings (
        competition_id, image_digest, is_calibration, binding_id DESC
    );

CREATE TRIGGER modal_image_bindings_append_only_update
BEFORE UPDATE ON modal_image_bindings
BEGIN
    SELECT RAISE(ABORT, 'modal_image_bindings is append-only');
END;

CREATE TRIGGER modal_image_bindings_append_only_delete
BEFORE DELETE ON modal_image_bindings
BEGIN
    SELECT RAISE(ABORT, 'modal_image_bindings is append-only');
END;
