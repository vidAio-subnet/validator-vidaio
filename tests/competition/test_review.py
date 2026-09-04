"""Human review: hash chain, deadline, supersedes + recalculation, calibration exclusion."""

from datetime import timedelta

import pytest

from vidaio.competition import (
    CompetitionConfig,
    LifecycleEngine,
    Phase,
    ReviewError,
    ReviewWindowClosed,
    submit_review,
    verify_review_chain,
)
from vidaio.competition import repository as repo

from support import BASELINE, END, SCORES_AT, Driver, build_manifest

REVIEW_AT = SCORES_AT + timedelta(hours=1)


def test_calibration_contender_never_in_ranking_or_podium(driver: Driver) -> None:
    cid, ids = driver.run_to_awaiting(
        build_manifest(baseline=BASELINE),
        {"hk-1": 0.9, "hk-2": 0.5},
        baseline_score=0.99,  # baseline beats everyone — and still earns nothing
    )
    ranking = repo.ranking(driver.conn, cid)
    assert [c.hotkey for c in ranking] == ["hk-1", "hk-2"]
    assert all(not c.is_calibration for c in ranking)
    assert all(not c.is_calibration for c in repo.podium(driver.conn, cid))

    baseline_row = next(c for c in repo.list_contenders(driver.conn, cid) if c.is_calibration)
    # The baseline WAS evaluated (its aggregates exist — that's the calibration signal)...
    assert baseline_row.final_score is not None
    assert baseline_row.final_score > max(c.final_score for c in ranking)
    # ...but can never hold a rank.
    assert baseline_row.final_rank is None


def test_calibration_rank_is_impossible_at_schema_level(driver: Driver) -> None:
    import sqlite3

    cid, _ = driver.run_to_awaiting(build_manifest(baseline=BASELINE), {"hk-1": 0.9}, baseline_score=0.5)
    baseline_row = next(c for c in repo.list_contenders(driver.conn, cid) if c.is_calibration)
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute(
            "UPDATE contenders SET final_rank = 1 WHERE contender_id = ?",
            (baseline_row.contender_id,),
        )


def test_disqualify_rerank_and_reinstate_via_supersedes(driver: Driver) -> None:
    cid, ids = driver.run_to_awaiting(build_manifest(), {"hk-1": 0.9, "hk-2": 0.5})
    assert [c.hotkey for c in repo.ranking(driver.conn, cid)] == ["hk-1", "hk-2"]

    dq_id = submit_review(
        driver.conn,
        cid,
        contender_id=ids["hk-1"],
        action="disqualify",
        reviewer="owner",
        reason="output hash mismatch on item 2",
        now=REVIEW_AT,
    )
    ranking = repo.ranking(driver.conn, cid)
    assert [c.hotkey for c in ranking] == ["hk-2"]
    assert ranking[0].final_rank == 1
    dq_row = repo.get_contender(driver.conn, ids["hk-1"])
    assert dq_row is not None and dq_row.manual_disqualified and dq_row.final_rank is None

    # Reinstate must supersede the disqualification (append-only correction).
    with pytest.raises(ReviewError):
        submit_review(
            driver.conn,
            cid,
            contender_id=ids["hk-1"],
            action="reinstate",
            reviewer="owner",
            reason="appeal accepted",
            now=REVIEW_AT + timedelta(minutes=5),
        )
    submit_review(
        driver.conn,
        cid,
        contender_id=ids["hk-1"],
        action="reinstate",
        reviewer="owner",
        reason="appeal accepted",
        now=REVIEW_AT + timedelta(minutes=5),
        supersedes_review_id=dq_id,
    )
    assert [c.hotkey for c in repo.ranking(driver.conn, cid)] == ["hk-1", "hk-2"]
    # Recalculation used only persisted per-item scores — same final_score as before.
    r1 = repo.get_contender(driver.conn, ids["hk-1"])
    assert r1 is not None and r1.final_rank == 1

    # Both review rows remain (append-only) and the chain still verifies.
    assert len(repo.list_reviews(driver.conn, cid)) == 2
    assert verify_review_chain(driver.conn, cid) is True


def test_tie_break_review_orders_equal_scores(driver: Driver) -> None:
    cid, ids = driver.run_to_awaiting(build_manifest(), {"hk-1": 0.8, "hk-2": 0.8})
    # Deterministic default: equal scores order by contender_id.
    assert [c.hotkey for c in repo.ranking(driver.conn, cid)] == ["hk-1", "hk-2"]

    submit_review(
        driver.conn,
        cid,
        contender_id=ids["hk-2"],
        action="tie_break",
        reviewer="owner",
        reason="earlier submission timestamp",
        now=REVIEW_AT,
        detail={"wins_over_contender_id": ids["hk-1"]},
    )
    assert [c.hotkey for c in repo.ranking(driver.conn, cid)] == ["hk-2", "hk-1"]

    with pytest.raises(ReviewError):
        submit_review(
            driver.conn,
            cid,
            contender_id=ids["hk-2"],
            action="tie_break",
            reviewer="owner",
            reason="missing detail",
            now=REVIEW_AT,
        )


def test_review_chain_verifies_and_detects_tampering(driver: Driver) -> None:
    cid, ids = driver.run_to_awaiting(build_manifest(), {"hk-1": 0.9, "hk-2": 0.5})
    for i, hk in enumerate(["hk-1", "hk-2"]):
        submit_review(
            driver.conn,
            cid,
            contender_id=ids[hk],
            action="disqualify",
            reviewer="owner",
            reason=f"reason {i}",
            now=REVIEW_AT + timedelta(minutes=i),
        )
    assert verify_review_chain(driver.conn, cid) is True

    # The DB itself refuses mutation (append-only triggers)...
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute("UPDATE human_reviews SET reason = 'edited' WHERE review_id = 1")
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute("DELETE FROM human_reviews WHERE review_id = 1")

    # ...and out-of-band tampering (e.g. editing the file directly) breaks the chain.
    driver.conn.execute("DROP TRIGGER human_reviews_append_only_update")
    driver.conn.execute("UPDATE human_reviews SET reason = 'edited' WHERE review_id = 1")
    assert verify_review_chain(driver.conn, cid) is False


def test_review_rejected_after_deadline(conn) -> None:
    engine = LifecycleEngine(CompetitionConfig(human_review_window_hours=2.0))
    driver = Driver(conn, engine)
    cid, ids = driver.run_to_awaiting(build_manifest(), {"hk-1": 0.9})
    # Window: SCORES_AT .. SCORES_AT+2h; competition still AWAITING_END_TIME after it.
    late = SCORES_AT + timedelta(hours=3)
    assert late < END
    with pytest.raises(ReviewWindowClosed):
        submit_review(
            conn,
            cid,
            contender_id=ids["hk-1"],
            action="disqualify",
            reviewer="owner",
            reason="too late",
            now=late,
        )
    # In-window works.
    submit_review(
        conn,
        cid,
        contender_id=ids["hk-1"],
        action="disqualify",
        reviewer="owner",
        reason="in window",
        now=SCORES_AT + timedelta(hours=1),
    )


def test_review_rejected_after_completion(driver: Driver) -> None:
    cid, ids = driver.run_to_awaiting(build_manifest(), {"hk-1": 0.9})
    driver.engine.tick(driver.conn, END)
    assert driver.phase(cid) is Phase.COMPLETED
    with pytest.raises(ReviewWindowClosed):
        submit_review(
            driver.conn,
            cid,
            contender_id=ids["hk-1"],
            action="disqualify",
            reviewer="owner",
            reason="post-completion",
            now=END,
        )


def test_review_of_calibration_contender_rejected(driver: Driver) -> None:
    cid, _ = driver.run_to_awaiting(build_manifest(baseline=BASELINE), {"hk-1": 0.9}, baseline_score=0.5)
    baseline_row = next(c for c in repo.list_contenders(driver.conn, cid) if c.is_calibration)
    with pytest.raises(ReviewError, match="DECISIONS"):
        submit_review(
            driver.conn,
            cid,
            contender_id=baseline_row.contender_id,
            action="disqualify",
            reviewer="owner",
            reason="pointless",
            now=REVIEW_AT,
        )
