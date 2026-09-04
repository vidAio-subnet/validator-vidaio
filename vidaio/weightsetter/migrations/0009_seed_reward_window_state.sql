-- Persist the canonical pristine reward window at schema initialization.  The owner
-- already interprets an absent row as RewardWindowState(), but cross-service readers
-- must not have to infer durable economic state from absence.  This additive migration
-- also repairs databases that applied 0008 before the singleton was seeded.

INSERT OR IGNORE INTO reward_window_state (
    id,
    kind,
    podium_hotkeys_json
) VALUES (
    1,
    'IDLE',
    '[]'
);
