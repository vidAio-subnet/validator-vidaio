-- Historical committed rows retain their assignment-era membership. No SQL
-- wall-clock timestamp is converted into a fabricated historical block height.
ALTER TABLE rounds ADD COLUMN commit_block INTEGER CHECK (commit_block >= 0);
UPDATE rounds SET commit_block = block WHERE committed_at IS NOT NULL;
CREATE INDEX rounds_commit_block ON rounds(commit_block) WHERE committed_at IS NOT NULL;
CREATE TRIGGER rounds_require_commit_block_update BEFORE UPDATE ON rounds
WHEN NEW.committed_at IS NOT NULL AND NEW.commit_block IS NULL
BEGIN SELECT RAISE(ABORT, 'completed round requires commit_block'); END;
CREATE TRIGGER rounds_require_commit_block_insert BEFORE INSERT ON rounds
WHEN NEW.committed_at IS NOT NULL AND NEW.commit_block IS NULL
BEGIN SELECT RAISE(ABORT, 'completed round requires commit_block'); END;
CREATE TRIGGER rounds_immutable_completion BEFORE UPDATE ON rounds
WHEN OLD.committed_at IS NOT NULL AND
     (NEW.committed_at IS NOT OLD.committed_at OR NEW.commit_block IS NOT OLD.commit_block)
BEGIN SELECT RAISE(ABORT, 'round completion is immutable'); END;

CREATE TABLE round_seal (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    sealed_close INTEGER NOT NULL CHECK (sealed_close >= -1)
);
INSERT INTO round_seal(singleton, sealed_close) VALUES (1, -1);

-- Persist fetched challenge identities even when every candidate was skipped.
-- The context becomes visible atomically with the entire completed round.
CREATE TABLE round_challenges (
    round_id TEXT NOT NULL REFERENCES rounds(round_id),
    challenge_id TEXT NOT NULL UNIQUE,
    track TEXT NOT NULL CHECK (track IN ('compression', 'upscaling')),
    anchor_block INTEGER NOT NULL CHECK (anchor_block >= 0),
    ordering_key INTEGER NOT NULL UNIQUE CHECK (ordering_key >= 1),
    PRIMARY KEY (round_id, challenge_id)
);
