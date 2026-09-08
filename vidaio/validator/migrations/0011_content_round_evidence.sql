-- D-024: original measured roster and full component decisions, including all-skip rounds.
CREATE TABLE content_round_evidence (
    round_id TEXT NOT NULL REFERENCES rounds(round_id),
    challenge_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    track TEXT NOT NULL,
    round_json TEXT NOT NULL,
    round_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (round_id, challenge_id, item_id, track)
);
