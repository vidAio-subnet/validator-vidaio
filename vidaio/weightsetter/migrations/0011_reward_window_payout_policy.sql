-- Tokenomics v3 (2026-09-23): a reward window carries its own payout policy so the
-- weight setter pays it without consulting mutable configuration (epoch log schema v18).
ALTER TABLE reward_window_state ADD COLUMN competition_share REAL;
ALTER TABLE reward_window_state ADD COLUMN place_shares_json TEXT NOT NULL DEFAULT '[]';
