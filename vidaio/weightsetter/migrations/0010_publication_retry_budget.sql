CREATE TABLE weight_publication_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intent_id INTEGER NOT NULL REFERENCES weight_intents(id),
    started_at REAL NOT NULL,
    retry_after REAL NOT NULL,
    failure_count INTEGER NOT NULL,
    finished_at REAL,
    succeeded INTEGER CHECK (succeeded IN (0, 1))
);

ALTER TABLE weight_intents ADD COLUMN reveal_wait_logged_at TEXT;
