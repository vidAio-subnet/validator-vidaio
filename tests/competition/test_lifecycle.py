from datetime import timedelta

from vidaio.competition import Phase
from vidaio.competition import repository as repo

from support import (
    BACKUP_REF,
    BASELINE,
    END,
    FINALIZATION,
    SCORES_AT,
    START,
    T0,
    Driver,
    build_manifest,
)


def test_happy_path_full_walk_with_event_log(driver: Driver) -> None:
    manifest = build_manifest(baseline=BASELINE)
    cid = manifest.competition_id

    driver.engine.create_competition(driver.conn, manifest, T0)
    assert driver.phase(cid) is Phase.SCHEDULED
    driver.anchor(cid)

    # Not yet due: ticking before start_time changes nothing.
    assert driver.engine.tick(driver.conn, T0 + timedelta(minutes=1)) == []
    assert driver.phase(cid) is Phase.SCHEDULED

    driver.engine.tick(driver.conn, START)
    assert driver.phase(cid) is Phase.ENROLLING

    driver.enroll(cid, "hk-1")
    driver.enroll(cid, "hk-2")

    driver.engine.tick(driver.conn, FINALIZATION)
    assert driver.phase(cid) is Phase.FINALIZING_SUBMISSIONS

    # Baseline injected during FINALIZING as a calibration contender: no hotkey, flagged.
    baseline_rows = [c for c in repo.list_contenders(driver.conn, cid) if c.is_calibration]
    assert len(baseline_rows) == 1
    assert baseline_rows[0].hotkey is None
    assert baseline_rows[0].tree_sha == BASELINE["tree_sha"]

    driver.accept_all(cid)
    t = FINALIZATION + timedelta(minutes=1)
    assert driver.engine.mark_submissions_backed_up(driver.conn, cid, BACKUP_REF, t) is True
    assert driver.phase(cid) is Phase.VALIDATING

    assert driver.engine.mark_validation_complete(driver.conn, cid, t) is Phase.BUILDING
    assert driver.engine.mark_builds_complete(driver.conn, cid, 3, t) is Phase.EVALUATING
    item_ids = driver.seed_items(cid)
    assert driver.engine.mark_evaluation_complete(driver.conn, cid, t) is True
    for c in repo.list_contenders(driver.conn, cid):
        driver.score_contender(cid, c.contender_id, item_ids, 0.8)
    assert driver.engine.mark_scores_persisted(driver.conn, cid, SCORES_AT) is True
    assert driver.phase(cid) is Phase.AWAITING_END_TIME

    comp = repo.get_competition(driver.conn, cid)
    assert comp is not None and comp.human_review_deadline is not None

    # The audit runner links every score row (baseline included) to its bundle —
    # completion is gated on this (require_audit_linkage).
    assert driver.link_audit_bundles(cid) > 0

    # Before end_time nothing completes.
    driver.engine.tick(driver.conn, END - timedelta(minutes=1))
    assert driver.phase(cid) is Phase.AWAITING_END_TIME
    driver.engine.tick(driver.conn, END)
    assert driver.phase(cid) is Phase.COMPLETED

    # Event log completeness: creation + every transition, in order.
    transitions = [
        (e["from_phase"], e["to_phase"])
        for e in driver.events(cid)
        if e["event_type"] == "transition"
    ]
    assert transitions == [
        ("SCHEDULED", "ENROLLING"),
        ("ENROLLING", "FINALIZING_SUBMISSIONS"),
        ("FINALIZING_SUBMISSIONS", "VALIDATING"),
        ("VALIDATING", "BUILDING"),
        ("BUILDING", "EVALUATING"),
        ("EVALUATING", "SCORING"),
        ("SCORING", "AWAITING_END_TIME"),
        ("AWAITING_END_TIME", "COMPLETED"),
    ]
    types = [e["event_type"] for e in driver.events(cid)]
    assert types[0] == "created"
    assert "calibration_injected" in types
    assert "ranks_recalculated" in types
    # The anchored pre-commitment event strictly precedes the enrolling transition.
    anchored_at = types.index("commitment_anchored")
    enrolling_at = next(
        i
        for i, e in enumerate(driver.events(cid))
        if e["event_type"] == "transition" and e["to_phase"] == "ENROLLING"
    )
    assert anchored_at < enrolling_at
    # Every transition event carries its guard name from the table.
    guards = [
        e["guard"] for e in driver.events(cid) if e["event_type"] == "transition"
    ]
    assert all(g for g in guards)


def test_tick_is_idempotent(driver: Driver) -> None:
    manifest = build_manifest()
    driver.engine.create_competition(driver.conn, manifest, T0)
    cid = manifest.competition_id
    driver.anchor(cid)

    applied = driver.engine.tick(driver.conn, START)
    assert applied == [(cid, Phase.SCHEDULED, Phase.ENROLLING)]
    n_events = len(driver.events(cid))

    # Same tick again: nothing applied, no new events.
    assert driver.engine.tick(driver.conn, START) == []
    assert len(driver.events(cid)) == n_events


def test_pipeline_marks_are_idempotent(driver: Driver) -> None:
    manifest = build_manifest()
    cid, _ = driver.run_to_awaiting(manifest, {"hk-1": 0.9})
    n_events = len(driver.events(cid))

    # Re-applying an already-applied transition is a no-op, not an error.
    assert driver.engine.mark_scores_persisted(driver.conn, cid, SCORES_AT) is False
    assert len(driver.events(cid)) == n_events
    assert driver.phase(cid) is Phase.AWAITING_END_TIME

    assert driver.engine.mark_evaluation_complete(driver.conn, cid, SCORES_AT) is False


def test_baseline_not_injected_without_manifest_baseline(driver: Driver) -> None:
    manifest = build_manifest()
    driver.engine.create_competition(driver.conn, manifest, T0)
    cid = manifest.competition_id
    driver.anchor(cid)
    driver.engine.tick(driver.conn, START)
    driver.enroll(cid, "hk-1")
    driver.engine.tick(driver.conn, FINALIZATION)
    assert all(not c.is_calibration for c in repo.list_contenders(driver.conn, cid))
