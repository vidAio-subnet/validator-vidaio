-- The auditor loop's PENDING (in-flight) signed report, for BYTE-IDEMPOTENT retries
--.
--
-- Finding #3 (HIGH): the loop rebuilt the report with a FRESH `sampled_at` on every retry,
-- changing its signed digest. If the Audit Results API COMMITTED the first POST but its
-- response was LOST, the retry's different digest CONFLICTS with the stored report and the
-- cursor never advances → all later epochs are blocked (a permanent wedge on an honest epoch).
--
-- This single-row-per-epoch table persists the EXACT signed report bytes the loop first built
-- for an epoch. On a retry the loop RESENDS those identical bytes, so a lost-response retry is
-- reconciled by the store as a DUPLICATE (idempotent accept) instead of a CONFLICT — the
-- cursor advances. The row is written once (the first build is canonical; a re-audit that
-- happened to differ must NOT overwrite it) and deleted once the report is durably accepted.

CREATE TABLE pending_report (
    epoch_id     INTEGER PRIMARY KEY,
    report_json  TEXT NOT NULL
);
