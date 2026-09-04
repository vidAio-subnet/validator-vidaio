"""Every edge NOT in the transition table must raise IllegalTransition."""

import pytest

from vidaio.competition import IllegalTransition, Phase, TRANSITIONS, is_allowed

from support import BACKUP_REF, FINALIZATION, START, T0, Driver, build_manifest

ALL_PHASES = list(Phase)
ILLEGAL_EDGES = [
    (a, b)
    for a in ALL_PHASES
    for b in ALL_PHASES
    if a is not b and (a, b) not in TRANSITIONS
]


def test_transition_table_matches_spec_diagram() -> None:
    assert len(TRANSITIONS) == 12
    assert is_allowed(Phase.SCHEDULED, Phase.ENROLLING)
    assert is_allowed(Phase.SCHEDULED, Phase.FAILED)
    assert is_allowed(Phase.VALIDATING, Phase.FAILED)
    assert is_allowed(Phase.BUILDING, Phase.FAILED)
    assert is_allowed(Phase.ENROLLING, Phase.CANCELLED)
    assert not is_allowed(Phase.COMPLETED, Phase.SCHEDULED)
    # No edge ever leaves a terminal phase.
    for a, _ in TRANSITIONS:
        assert a not in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED)


@pytest.mark.parametrize("from_phase,to_phase", ILLEGAL_EDGES)
def test_every_illegal_edge_raises(
    driver: Driver, from_phase: Phase, to_phase: Phase
) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    # Force the starting phase directly (bypassing the engine) to probe the edge.
    driver.conn.execute(
        "UPDATE competitions SET status = ? WHERE competition_id = ?",
        (from_phase.value, cid),
    )
    with pytest.raises(IllegalTransition) as exc_info:
        driver.engine._apply(driver.conn, cid, to_phase, START)
    err = exc_info.value
    assert err.from_phase is from_phase
    assert err.to_phase is to_phase
    # The illegal edge must not have moved the phase or logged an event.
    assert driver.phase(cid) is from_phase
    assert all(e["event_type"] != "transition" for e in driver.events(cid))


def test_reapplying_same_phase_is_noop_not_error(driver: Driver) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    driver.anchor(cid)
    driver.engine.tick(driver.conn, START)
    assert driver.engine._apply(driver.conn, cid, Phase.ENROLLING, START) is False


def test_pipeline_marks_at_wrong_phase_raise(driver: Driver) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    driver.anchor(cid)

    # SCHEDULED: none of the pipeline-completion calls are legal.
    with pytest.raises(IllegalTransition):
        driver.engine.mark_submissions_backed_up(driver.conn, cid, BACKUP_REF, T0)
    with pytest.raises(IllegalTransition):
        driver.engine.mark_validation_complete(driver.conn, cid, T0)
    with pytest.raises(IllegalTransition):
        driver.engine.mark_builds_complete(driver.conn, cid, 1, T0)
    with pytest.raises(IllegalTransition):
        driver.engine.mark_evaluation_complete(driver.conn, cid, T0)
    with pytest.raises(IllegalTransition):
        driver.engine.mark_scores_persisted(driver.conn, cid, T0)

    driver.engine.tick(driver.conn, START)  # -> ENROLLING
    with pytest.raises(IllegalTransition):
        driver.engine.mark_builds_complete(driver.conn, cid, 1, START)
    # cancel is legal only from ENROLLING; fail is not.
    with pytest.raises(IllegalTransition):
        driver.engine.fail(driver.conn, cid, START, "nope")


def test_validation_blocked_while_reviews_pending(driver: Driver) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    driver.anchor(cid)
    driver.engine.tick(driver.conn, START)
    driver.enroll(cid, "hk-1")
    driver.engine.tick(driver.conn, FINALIZATION)
    driver.engine.mark_submissions_backed_up(driver.conn, cid, BACKUP_REF, FINALIZATION)
    # Contender still ENROLLED (pending validation review) -> guard blocks BUILDING.
    with pytest.raises(IllegalTransition) as exc_info:
        driver.engine.mark_validation_complete(driver.conn, cid, FINALIZATION)
    assert exc_info.value.guard == "accepted_contender_and_no_pending_review"
    assert driver.phase(cid) is Phase.VALIDATING


def test_unknown_competition_raises(driver: Driver) -> None:
    with pytest.raises(IllegalTransition):
        driver.engine.mark_evaluation_complete(driver.conn, "nope-01", T0)
