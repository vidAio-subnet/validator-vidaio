"""Failure and cancellation edges (spec §04 diagram)."""

from datetime import timedelta

import pytest

from vidaio.competition import EnrollmentError, Phase
from vidaio.competition import repository as repo

from support import BACKUP_REF, BASELINE, FINALIZATION, START, T0, Driver, build_manifest


def _to_validating(driver: Driver, manifest, hotkeys: list[str]) -> str:
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    driver.anchor(cid)
    driver.engine.tick(driver.conn, START)
    for hk in hotkeys:
        driver.enroll(cid, hk)
    driver.engine.tick(driver.conn, FINALIZATION)
    driver.engine.mark_submissions_backed_up(
        driver.conn, cid, BACKUP_REF, FINALIZATION + timedelta(minutes=1)
    )
    return cid


def test_no_accepted_contender_fails_validating(driver: Driver) -> None:
    cid = _to_validating(driver, build_manifest(), ["hk-1", "hk-2"])
    for c in repo.list_contenders(driver.conn, cid):
        repo.set_contender_status(driver.conn, c.contender_id, "REJECTED", FINALIZATION)

    result = driver.engine.mark_validation_complete(
        driver.conn, cid, FINALIZATION + timedelta(minutes=2)
    )
    assert result is Phase.FAILED
    comp = repo.get_competition(driver.conn, cid)
    assert comp is not None and comp.status is Phase.FAILED
    assert comp.failure_reason == "no accepted contender after validation"
    event = [e for e in driver.events(cid) if e["to_phase"] == "FAILED"][0]
    assert event["guard"] == "no_accepted_contender"


def test_baseline_alone_cannot_carry_a_competition(driver: Driver) -> None:
    # Only the calibration baseline passes validation: still FAILED — the baseline is a
    # non-earning calibration baseline, not a contender (the project design record #1).
    cid = _to_validating(driver, build_manifest(baseline=BASELINE), ["hk-1"])
    for c in repo.list_contenders(driver.conn, cid):
        status = "ACCEPTED" if c.is_calibration else "REJECTED"
        repo.set_contender_status(driver.conn, c.contender_id, status, FINALIZATION)
    result = driver.engine.mark_validation_complete(
        driver.conn, cid, FINALIZATION + timedelta(minutes=2)
    )
    assert result is Phase.FAILED


def test_all_builds_failed_fails_building(driver: Driver) -> None:
    cid = _to_validating(driver, build_manifest(), ["hk-1", "hk-2"])
    driver.accept_all(cid)
    t = FINALIZATION + timedelta(minutes=2)
    driver.engine.mark_validation_complete(driver.conn, cid, t)
    assert driver.phase(cid) is Phase.BUILDING

    result = driver.engine.mark_builds_complete(driver.conn, cid, 0, t)
    assert result is Phase.FAILED
    comp = repo.get_competition(driver.conn, cid)
    assert comp is not None and comp.failure_reason == "all builds failed"
    event = [e for e in driver.events(cid) if e["to_phase"] == "FAILED"][0]
    assert event["guard"] == "all_builds_failed"


def test_scheduled_can_fail(driver: Driver) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    assert driver.engine.fail(driver.conn, cid, T0, "manifest commitment mismatch") is True
    comp = repo.get_competition(driver.conn, cid)
    assert comp is not None and comp.status is Phase.FAILED
    assert comp.failure_reason == "manifest commitment mismatch"


def test_enrolling_can_cancel_and_cancellation_is_idempotent(driver: Driver) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    driver.anchor(cid)
    driver.engine.tick(driver.conn, START)
    assert driver.engine.cancel(driver.conn, cid, START, "operator abort") is True
    assert driver.phase(cid) is Phase.CANCELLED
    # Idempotent re-apply.
    assert driver.engine.cancel(driver.conn, cid, START, "operator abort") is False


def test_enrollment_gates(driver: Driver) -> None:
    manifest = build_manifest()  # minimum_alpha_stake=500
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    driver.anchor(cid)

    # Not ENROLLING yet.
    with pytest.raises(EnrollmentError):
        driver.enroll(cid, "hk-early", now=T0)

    driver.engine.tick(driver.conn, START)
    # Alpha-stake gate.
    with pytest.raises(EnrollmentError):
        driver.enroll(cid, "hk-poor", stake=499.0)
    # Past enrollment_deadline (still ENROLLING until finalization_time).
    with pytest.raises(EnrollmentError):
        driver.enroll(cid, "hk-late", now=FINALIZATION - timedelta(minutes=1))
    # In-window, sufficient stake.
    driver.enroll(cid, "hk-ok")
    assert len(repo.list_contenders(driver.conn, cid)) == 1
