-- Bind the scoring TRACK and a monotonic DISPATCH ORDERING KEY into the challenge
-- commitment, PRE-DISPATCH.
--
-- The commitment already fixes {asset_id, dag_digest, seed, scorer_version} before a
-- challenge is dispatched and is anchored independently of the epoch log. Extending it
-- with the track + the fold-order key means the EWMA earning fold's ORDER and the
-- item's TRACK are fixed BEFORE any score exists, so a dishonest authority can no longer
-- reorder scores and stamp matching sequences/tracks at finalization: the auditor
-- re-reads these committed values from the anchored commitment (the DAG_REVEAL preimage)
-- and rejects a reordered fold or a fabricated track.
--
-- Both are folded into the commit-hash preimage, so a row's commit_hash binds them; the
-- columns persist the same values for reveal. No commitment rows are deployed anywhere,
-- so the defaults only satisfy the NOT NULL contract for the rebuilt-in-place table.

ALTER TABLE challenge_commitments ADD COLUMN track TEXT NOT NULL DEFAULT '';
ALTER TABLE challenge_commitments
  ADD COLUMN dispatch_ordering_key INTEGER NOT NULL DEFAULT 0;
