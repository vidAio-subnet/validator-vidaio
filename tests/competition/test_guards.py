"""Lifecycle guards are verified conditions, not labels: evidence-carrying
backup, DB-checked batches and score rows, atomic scoring transition, untruncated
review window."""

from datetime import timedelta

import pytest

from vidaio.competition import (
    CompetitionConfig,
    IllegalTransition,
    LifecycleEngine,
    Phase,
)
from vidaio.competition import engine as engine_mod
from vidaio.competition import repository as repo

from support import (
    BACKUP_REF,
    END,
    FINALIZATION,
    SCORES_AT,
    START,
    T0,
    Driver,
    build_manifest,
)

T_EVAL = FINALIZATION + timedelta(minutes=10)


def _to_finalizing(driver: Driver) -> str:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    driver.anchor(cid)
    driver.engine.tick(driver.conn, START)
    driver.enroll(cid, "hk-1")
    driver.engine.tick(driver.conn, FINALIZATION)
    return cid


def test_backup_ref_required_and_persisted(driver: Driver) -> None:
    cid = _to_finalizing(driver)
    for empty in ("", "   "):
        with pytest.raises(ValueError, match="backup_ref"):
            driver.engine.mark_submissions_backed_up(driver.conn, cid, empty, FINALIZATION)
    assert driver.phase(cid) is Phase.FINALIZING_SUBMISSIONS

    assert (
        driver.engine.mark_submissions_backed_up(driver.conn, cid, BACKUP_REF, FINALIZATION)
        is True
    )
    event = next(
        e
        for e in driver.events(cid)
        if e["event_type"] == "transition" and e["to_phase"] == "VALIDATING"
    )
    assert BACKUP_REF in (event["payload_json"] or "")


def test_evaluation_complete_requires_terminal_batches(driver: Driver) -> None:
    cid, ids, item_ids = driver.run_to_evaluating(build_manifest(), ["hk-1"])
    driver.conn.execute(
        "INSERT INTO batches (competition_id, contender_id, batch_index, status, created_at)"
        " VALUES (?, ?, 0, 'RUNNING', ?)",
        (cid, ids["hk-1"], repo.iso(T_EVAL)),
    )
    with pytest.raises(IllegalTransition, match="not yet terminal"):
        driver.engine.mark_evaluation_complete(driver.conn, cid, T_EVAL)
    assert driver.phase(cid) is Phase.EVALUATING

    # COMPLETED and FAILED both count as terminal.
    driver.conn.execute("UPDATE batches SET status = 'COMPLETED' WHERE competition_id = ?", (cid,))
    driver.conn.execute(
        "INSERT INTO batches (competition_id, contender_id, batch_index, status, created_at)"
        " VALUES (?, ?, 1, 'FAILED', ?)",
        (cid, ids["hk-1"], repo.iso(T_EVAL)),
    )
    assert driver.engine.mark_evaluation_complete(driver.conn, cid, T_EVAL) is True
    assert driver.phase(cid) is Phase.SCORING


def test_scores_persisted_requires_a_row_per_contender_and_item(driver: Driver) -> None:
    cid, ids, item_ids = driver.run_to_evaluating(build_manifest(), ["hk-1", "hk-2"])
    driver.engine.mark_evaluation_complete(driver.conn, cid, T_EVAL)
    driver.score_contender(cid, ids["hk-1"], item_ids, 0.9)
    driver.score_contender(cid, ids["hk-2"], item_ids[:1], 0.5)  # hk-2 missing item 2
    with pytest.raises(IllegalTransition, match="no persisted score row"):
        driver.engine.mark_scores_persisted(driver.conn, cid, SCORES_AT)
    assert driver.phase(cid) is Phase.SCORING

    driver.score_contender(cid, ids["hk-2"], item_ids[1:], 0.5)
    assert driver.engine.mark_scores_persisted(driver.conn, cid, SCORES_AT) is True
    assert driver.phase(cid) is Phase.AWAITING_END_TIME


def test_scores_persisted_is_one_transaction(
    driver: Driver, monkeypatch: pytest.MonkeyPatch
) -> None:
    cid, ids, item_ids = driver.run_to_evaluating(build_manifest(), ["hk-1"])
    driver.engine.mark_evaluation_complete(driver.conn, cid, T_EVAL)
    driver.score_contender(cid, ids["hk-1"], item_ids, 0.9)

    def boom(*args, **kwargs):
        raise RuntimeError("ranking crashed mid-transition")

    monkeypatch.setattr(engine_mod, "recalculate_ranks", boom)
    with pytest.raises(RuntimeError, match="ranking crashed"):
        driver.engine.mark_scores_persisted(driver.conn, cid, SCORES_AT)
    # The crash rolled EVERYTHING back: phase, deadline, event log, ranks.
    comp = repo.get_competition(driver.conn, cid)
    assert comp is not None
    assert comp.status is Phase.SCORING
    assert comp.human_review_deadline is None
    assert all(e["to_phase"] != "AWAITING_END_TIME" for e in driver.events(cid))

    monkeypatch.undo()
    assert driver.engine.mark_scores_persisted(driver.conn, cid, SCORES_AT) is True
    comp = repo.get_competition(driver.conn, cid)
    assert comp is not None and comp.status is Phase.AWAITING_END_TIME
    assert comp.human_review_deadline is not None


def test_completion_never_truncates_the_review_window(conn) -> None:
    # A review window stretching past end_time: 60h from SCORES_AT > END.
    engine = LifecycleEngine(CompetitionConfig(human_review_window_hours=60.0))
    driver = Driver(conn, engine)
    cid, _ = driver.run_to_awaiting(build_manifest(), {"hk-1": 0.9})
    comp = repo.get_competition(conn, cid)
    assert comp is not None and comp.human_review_deadline is not None
    assert comp.human_review_deadline > END

    engine.tick(conn, END)  # end_time reached, review window still open
    assert driver.phase(cid) is Phase.AWAITING_END_TIME
    engine.tick(conn, comp.human_review_deadline)
    assert driver.phase(cid) is Phase.COMPLETED


# ---- TOCTOU probes: guards re-checked INSIDE the transaction ----


def test_toctou_item_inserted_after_matrix_precheck_is_caught_in_txn(
    driver: Driver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An evaluation item landing between the complete-matrix pre-check and the
    BEGIN IMMEDIATE transaction must NOT slip the competition into
    AWAITING_END_TIME: the in-transaction re-check catches it and rolls back."""
    cid, ids, item_ids = driver.run_to_evaluating(build_manifest(), ["hk-1"])
    driver.engine.mark_evaluation_complete(driver.conn, cid, T_EVAL)
    driver.score_contender(cid, ids["hk-1"], item_ids, 0.9)

    real = repo.count_missing_item_scores
    calls = {"n": 0}

    def racy(conn, competition_id):
        calls["n"] += 1
        if calls["n"] == 1:
            # The pre-check genuinely sees a complete matrix ...
            assert real(conn, competition_id) == 0
            # ... and a concurrent writer lands a NEW item right after it.
            repo.add_evaluation_item(
                conn,
                competition_id,
                item_index=99,
                input_sha256="9" * 64,
                input_bytes=1,
                threshold_commitment="f" * 64,
                challenge_id=f"chal-{competition_id}",
                now=SCORES_AT,
            )
            return 0
        return real(conn, competition_id)

    monkeypatch.setattr(engine_mod.repo, "count_missing_item_scores", racy)
    with pytest.raises(IllegalTransition, match="no persisted score row"):
        driver.engine.mark_scores_persisted(driver.conn, cid, SCORES_AT)
    assert calls["n"] >= 2  # the in-txn re-check ran and caught it
    comp = repo.get_competition(driver.conn, cid)
    assert comp is not None
    assert comp.status is Phase.SCORING  # transition fully rolled back
    assert comp.human_review_deadline is None
    assert all(e["to_phase"] != "AWAITING_END_TIME" for e in driver.events(cid))


def test_toctou_batch_inserted_after_terminal_precheck_is_caught_in_txn(
    driver: Driver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-terminal batch landing between the terminal-batch pre-check and the
    BEGIN IMMEDIATE transaction must NOT slip the competition into SCORING."""
    cid, ids, item_ids = driver.run_to_evaluating(build_manifest(), ["hk-1"])

    real = repo.count_non_terminal_batches
    calls = {"n": 0}

    def racy(conn, competition_id):
        calls["n"] += 1
        if calls["n"] == 1:
            assert real(conn, competition_id) == 0
            conn.execute(
                "INSERT INTO batches (competition_id, contender_id, batch_index,"
                " status, created_at) VALUES (?, ?, 0, 'RUNNING', ?)",
                (competition_id, ids["hk-1"], repo.iso(T_EVAL)),
            )
            return 0
        return real(conn, competition_id)

    monkeypatch.setattr(engine_mod.repo, "count_non_terminal_batches", racy)
    with pytest.raises(IllegalTransition, match="not yet terminal"):
        driver.engine.mark_evaluation_complete(driver.conn, cid, T_EVAL)
    assert calls["n"] >= 2
    assert driver.phase(cid) is Phase.EVALUATING  # transition fully rolled back
    assert all(e["to_phase"] != "SCORING" for e in driver.events(cid))
