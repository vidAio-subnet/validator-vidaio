"""Single-running-competition invariant (spec §04): engine gate + SQL partial unique index."""

import sqlite3
from datetime import timedelta

import pytest

from vidaio.competition import Phase
from vidaio.competition import repository as repo

from support import BACKUP_REF, END, FINALIZATION, SCORES_AT, START, T0, Driver, build_manifest


def test_second_competition_cannot_start_while_one_running(driver: Driver) -> None:
    first = build_manifest("comp-01")
    second = build_manifest(
        "comp-02",
        start_time=START + timedelta(minutes=10),
        enrollment_deadline=END + timedelta(hours=1),
        finalization_time=END + timedelta(hours=2),
        end_time=END + timedelta(hours=6),
    )
    driver.engine.create_competition(driver.conn, first, T0)
    driver.engine.create_competition(driver.conn, second, T0)
    driver.anchor("comp-01")
    driver.anchor("comp-02")

    driver.engine.tick(driver.conn, START)
    assert driver.phase("comp-01") is Phase.ENROLLING

    # comp-02's start_time passes while comp-01 runs: it stays SCHEDULED.
    driver.engine.tick(driver.conn, START + timedelta(minutes=30))
    assert driver.phase("comp-02") is Phase.SCHEDULED

    # Walk comp-01 all the way to COMPLETED.
    driver.enroll("comp-01", "hk-1")
    driver.engine.tick(driver.conn, FINALIZATION)
    driver.accept_all("comp-01")
    t = FINALIZATION + timedelta(minutes=1)
    driver.engine.mark_submissions_backed_up(driver.conn, "comp-01", BACKUP_REF, t)
    driver.engine.mark_validation_complete(driver.conn, "comp-01", t)
    driver.engine.mark_builds_complete(driver.conn, "comp-01", 1, t)
    driver.engine.mark_evaluation_complete(driver.conn, "comp-01", t)
    driver.engine.mark_scores_persisted(driver.conn, "comp-01", SCORES_AT)
    driver.link_audit_bundles("comp-01")
    driver.engine.tick(driver.conn, END)
    assert driver.phase("comp-01") is Phase.COMPLETED

    # The same tick already freed the slot and started comp-02.
    assert driver.phase("comp-02") is Phase.ENROLLING


def test_partial_unique_index_is_the_backstop(driver: Driver) -> None:
    a = build_manifest("comp-01")
    b = build_manifest("comp-02")
    driver.engine.create_competition(driver.conn, a, T0)
    driver.engine.create_competition(driver.conn, b, T0)
    driver.anchor("comp-01")
    driver.anchor("comp-02")
    driver.engine.tick(driver.conn, START)
    assert driver.phase("comp-01") is Phase.ENROLLING

    # Bypassing the engine entirely, SQL refuses a second running competition.
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute(
            "UPDATE competitions SET status = 'ENROLLING' WHERE competition_id = 'comp-02'"
        )
    assert driver.phase("comp-02") is Phase.SCHEDULED


def test_terminal_phase_frees_the_slot(driver: Driver) -> None:
    a = build_manifest("comp-01")
    b = build_manifest("comp-02")
    driver.engine.create_competition(driver.conn, a, T0)
    driver.engine.create_competition(driver.conn, b, T0)
    driver.anchor("comp-01")
    driver.anchor("comp-02")
    driver.engine.tick(driver.conn, START)
    driver.engine.cancel(driver.conn, "comp-01", START + timedelta(minutes=1), "operator abort")
    assert driver.phase("comp-01") is Phase.CANCELLED

    driver.engine.tick(driver.conn, START + timedelta(minutes=2))
    assert driver.phase("comp-02") is Phase.ENROLLING

    assert repo.running_competition_id(driver.conn) == "comp-02"
