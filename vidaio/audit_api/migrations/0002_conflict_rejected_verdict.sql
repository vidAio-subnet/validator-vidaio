-- The conflict ledger records a divergent resubmission for an already-reported
-- (auditor_hotkey, epoch_id). Carry the REJECTED report's recomputed verdict so a
-- signed, divergent DISPUTED report still surfaces as a dispute even though it is not
-- persisted — a CLEAN first report must not be able to bury a later DISPUTED one
-- (the project design record §3.2, §5). Recomputed from the report's item/weight
-- verdicts (never the self-reported `overall`). Default 'CLEAN' for any pre-existing
-- row (there are none before this feature ships).
ALTER TABLE audit_report_conflicts
    ADD COLUMN rejected_overall TEXT NOT NULL DEFAULT 'CLEAN'
        CHECK (rejected_overall IN ('CLEAN', 'DISPUTED'));
