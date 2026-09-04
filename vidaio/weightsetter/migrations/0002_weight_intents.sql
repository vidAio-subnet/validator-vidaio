-- Weight-submission intent ledger.
--
-- `set_weights` is a NON-IDEMPOTENT chain write with a retry envelope around it.
-- If the chain accepts the first request but the response is lost, the retry is
-- tempo-rejected and the old code recorded the whole attempt as FAILED even
-- though the weights had changed — and publication only started afterwards, so a
-- crash between acceptance and publication left an accepted vector permanently
-- unaudited with nothing to re-drive it.
--
-- An intent row is written BEFORE the first set_weights call, carrying the exact
-- vector, its digest, the score-packet digests backing it, and the block the
-- attempt was made at. Every later step is driven FROM this row, so a restart can
-- always tell what was attempted and finish it:
--
--   pending    the vector was written but the chain outcome is not yet known
--   accepted   the chain accepted it (directly, or reconciled after an ambiguous
--              attempt); publication is owed
--   published  artifacts stored, PublicationRecord ledgered and anchored
--   abandoned  the attempt provably did not change the chain (tempo gate, an
--              explicit rejection, or an unconfirmable crash)
--
-- `resolution` records HOW the state was reached so a reconciled-by-inference
-- acceptance is never mistaken for a directly-observed one.

CREATE TABLE weight_intents (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    attempt_block       INTEGER NOT NULL,
    version_key         INTEGER NOT NULL,
    vector_json         TEXT NOT NULL,   -- canonical JSON {uid(str): weight}
    vector_digest       TEXT NOT NULL,   -- sha256 of vector_json
    packet_digests_json TEXT NOT NULL,   -- JSON array of score-packet digests
    state               TEXT NOT NULL
        CHECK (state IN ('pending', 'accepted', 'published', 'abandoned')),
    resolution          TEXT,
    accepted_block      INTEGER,
    commitment_id       INTEGER,         -- CommitmentLedger id (separate database)
    settled_at          TEXT
);

CREATE INDEX idx_weight_intents_state ON weight_intents(state, id);
CREATE INDEX idx_weight_intents_created ON weight_intents(created_at);
