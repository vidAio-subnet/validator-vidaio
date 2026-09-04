-- Schema-v14 global result ordering is the immutable order of actual terminal
-- AWAITING_END_TIME -> COMPLETED transitions, never competition creation/start order.
-- The event_id primary key is the persisted order key and is allocated atomically
-- with the status transition by LifecycleEngine._apply.  Enforce the lifecycle's
-- one-terminal-event invariant for direct-SQL/concurrency safety and index the
-- ordered scan used by epoch evidence.

CREATE UNIQUE INDEX ux_events_one_terminal_completion
    ON events (competition_id)
    WHERE from_phase = 'AWAITING_END_TIME'
      AND to_phase = 'COMPLETED';

CREATE INDEX ix_events_terminal_completion_order
    ON events (event_id)
    WHERE from_phase = 'AWAITING_END_TIME'
      AND to_phase = 'COMPLETED';
