"""recalculate_ranks aggregation rules: failures count as zero, item
length comes from evaluation_items, coverage is clamped, worst-decile is computed."""

from datetime import timedelta

import pytest

from vidaio.competition import recalculate_ranks
from vidaio.competition import repository as repo

from support import FINALIZATION, SCORES_AT, Driver, build_manifest

RECORD_AT = FINALIZATION + timedelta(minutes=30)


def _score_items(
    driver: Driver,
    cid: str,
    contender_id: int,
    item_scores: dict[int, float],
    *,
    gate_passed: dict[int, bool] | None = None,
    length_seconds: dict[int, float] | None = None,
) -> None:
    for item_id, score in item_scores.items():
        repo.record_item_score(
            driver.conn,
            cid,
            contender_id=contender_id,
            item_id=item_id,
            packet_bytes=driver.make_packet(
                contender_id,
                item_id,
                score,
                gate_passed=(gate_passed or {}).get(item_id, True),
                length_seconds=(length_seconds or {}).get(item_id, 10.0),
            ),
            now=RECORD_AT,
        )


def test_gate_failed_items_drag_the_aggregate_to_zero_not_out(driver: Driver) -> None:
    cid, ids, item_ids = driver.run_to_evaluating(build_manifest(), ["hk-1", "hk-2"])
    driver.engine.mark_evaluation_complete(driver.conn, cid, RECORD_AT)
    # hk-1: one brilliant item, one gate failure. hk-2: solid 0.5 on both.
    _score_items(
        driver,
        cid,
        ids["hk-1"],
        {item_ids[0]: 0.9, item_ids[1]: 0.0},
        gate_passed={item_ids[1]: False},
    )
    _score_items(driver, cid, ids["hk-2"], {item_ids[0]: 0.5, item_ids[1]: 0.5})
    driver.engine.mark_scores_persisted(driver.conn, cid, SCORES_AT)

    r1 = repo.get_contender(driver.conn, ids["hk-1"])
    r2 = repo.get_contender(driver.conn, ids["hk-2"])
    assert r1 is not None and r2 is not None
    # The failed item stays IN the aggregate as zero: (0.9*10 + 0*10) / 20.
    assert r1.media_score_aggregate == pytest.approx(0.45)
    assert r1.worst_decile_aggregate == pytest.approx(0.0)
    assert r1.length_coverage == pytest.approx(0.5)  # only the valid item covers
    assert r2.media_score_aggregate == pytest.approx(0.5)
    assert r2.worst_decile_aggregate == pytest.approx(0.5)
    assert r2.length_coverage == pytest.approx(1.0)
    # A contender whose failures drag it down loses to a steady one.
    assert [c.hotkey for c in repo.ranking(driver.conn, cid)] == ["hk-2", "hk-1"]


def test_item_length_comes_from_evaluation_items_and_coverage_is_clamped(
    driver: Driver,
) -> None:
    cid, ids, item_ids = driver.run_to_evaluating(build_manifest(), ["hk-1"])
    driver.engine.mark_evaluation_complete(driver.conn, cid, RECORD_AT)
    # The packets claim absurd per-response lengths (1000s vs the items' 10s each).
    # Aggregation must ignore them: quality weighting and coverage use ONLY
    # evaluation_items.length_seconds.
    _score_items(
        driver,
        cid,
        ids["hk-1"],
        {item_ids[0]: 1.0, item_ids[1]: 0.0},
        length_seconds={item_ids[0]: 1000.0, item_ids[1]: 1000.0},
    )
    driver.engine.mark_scores_persisted(driver.conn, cid, SCORES_AT)
    r1 = repo.get_contender(driver.conn, ids["hk-1"])
    assert r1 is not None
    # Inflated response lengths would have given (1.0*1000 + 0*1000)/2000 ≠ 0.5 with
    # a >1 coverage; the item-length weighting gives exactly (1*10 + 0*10)/20.
    assert r1.media_score_aggregate == pytest.approx(0.5)
    assert r1.length_coverage == pytest.approx(1.0)
    assert 0.0 <= r1.length_coverage <= 1.0


def test_missing_item_rows_count_as_zero_after_recalc(driver: Driver) -> None:
    cid, ids = driver.run_to_awaiting(build_manifest(), {"hk-1": 0.9, "hk-2": 0.5})
    # Simulate a lost/never-written row for hk-1's second item, then re-rank: the
    # missing item enters the aggregate as zero (never silently filtered).
    item_ids = [
        r["item_id"]
        for r in driver.conn.execute(
            "SELECT item_id FROM evaluation_items WHERE competition_id = ? ORDER BY item_index",
            (cid,),
        )
    ]
    driver.conn.execute(
        "DELETE FROM performance_history WHERE contender_id = ? AND item_id = ?",
        (ids["hk-1"], item_ids[1]),
    )
    recalculate_ranks(driver.conn, cid, SCORES_AT + timedelta(minutes=5))
    r1 = repo.get_contender(driver.conn, ids["hk-1"])
    assert r1 is not None
    assert r1.media_score_aggregate == pytest.approx(0.45)  # (0.9*10 + 0)/20
    assert r1.length_coverage == pytest.approx(0.5)
    assert [c.hotkey for c in repo.ranking(driver.conn, cid)] == ["hk-2", "hk-1"]


def test_use_worst_decile_flag_switches_the_quality_term(driver: Driver) -> None:
    def run(flag: bool, comp_id: str) -> list[str | None]:
        cid, ids, item_ids = driver.run_to_evaluating(
            build_manifest(comp_id, use_worst_decile=flag), ["hk-1", "hk-2"]
        )
        driver.engine.mark_evaluation_complete(driver.conn, cid, RECORD_AT)
        # hk-1: high mean (0.5), terrible worst item (0.1). hk-2: steady 0.4.
        _score_items(driver, cid, ids["hk-1"], {item_ids[0]: 0.9, item_ids[1]: 0.1})
        _score_items(driver, cid, ids["hk-2"], {item_ids[0]: 0.4, item_ids[1]: 0.4})
        driver.engine.mark_scores_persisted(driver.conn, cid, SCORES_AT)
        # Both aggregates are stored regardless of the flag.
        r1 = repo.get_contender(driver.conn, ids["hk-1"])
        assert r1 is not None
        assert r1.media_score_aggregate == pytest.approx(0.5)
        assert r1.worst_decile_aggregate == pytest.approx(0.1)
        ranking = [c.hotkey for c in repo.ranking(driver.conn, cid)]
        # Free the running slot so the second competition can run.
        driver.conn.execute(
            "UPDATE competitions SET status = 'COMPLETED' WHERE competition_id = ?", (cid,)
        )
        return ranking

    # Default (False): quality term is the length-weighted mean -> hk-1 wins.
    assert run(False, "comp-mean") == ["hk-1", "hk-2"]
    # Flag on: quality term is the worst-decile aggregate -> hk-2 wins.
    assert run(True, "comp-decile") == ["hk-2", "hk-1"]


def test_contender_with_no_rows_gets_zero_aggregates_and_ranks_last(driver: Driver) -> None:
    """review #8 residual (a): a contender with NO score rows is zero-scored and
    ranked LAST — never dropped from the ranking with None aggregates."""
    cid, ids = driver.run_to_awaiting(
        build_manifest(), {"hk-1": 0.9, "hk-2": 0.5, "hk-3": 0.1}
    )
    # Simulate a contender whose rows were all lost/never written, then re-rank.
    driver.conn.execute(
        "DELETE FROM performance_history WHERE contender_id = ?", (ids["hk-3"],)
    )
    recalculate_ranks(driver.conn, cid, SCORES_AT + timedelta(minutes=5))

    r3 = repo.get_contender(driver.conn, ids["hk-3"])
    assert r3 is not None
    assert r3.media_score_aggregate == 0.0
    assert r3.worst_decile_aggregate == 0.0
    assert r3.cost_efficiency_aggregate == 0.0
    assert r3.length_coverage == 0.0
    assert r3.final_score == 0.0
    assert r3.final_rank == 3  # still ranked — last
    assert [c.hotkey for c in repo.ranking(driver.conn, cid)] == ["hk-1", "hk-2", "hk-3"]


def test_cost_efficiency_zeroes_gate_failed_and_missing_items(driver: Driver) -> None:
    """review #8 residual (b): the cost-efficiency aggregate spans ALL evaluation
    items — gate-failed and missing items contribute ZERO instead of shrinking the
    denominator, so completing only your cheap items cannot inflate the aggregate."""
    cid, ids, item_ids = driver.run_to_evaluating(build_manifest(), ["hk-1", "hk-2"])
    driver.engine.mark_evaluation_complete(driver.conn, cid, RECORD_AT)
    # hk-1: valid on both items at cost 2.0. hk-2: valid on item 0 at cost 1.0 (the
    # cheapest valid per-item cost anywhere -> the anchor), gate-FAILED on item 1.
    for item_id in item_ids:
        repo.record_item_score(
            driver.conn, cid, contender_id=ids["hk-1"], item_id=item_id,
            packet_bytes=driver.make_packet(ids["hk-1"], item_id, 0.8, cost=2.0),
            now=RECORD_AT,
        )
    repo.record_item_score(
        driver.conn, cid, contender_id=ids["hk-2"], item_id=item_ids[0],
        packet_bytes=driver.make_packet(ids["hk-2"], item_ids[0], 0.8, cost=1.0),
        now=RECORD_AT,
    )
    repo.record_item_score(
        driver.conn, cid, contender_id=ids["hk-2"], item_id=item_ids[1],
        packet_bytes=driver.make_packet(
            ids["hk-2"], item_ids[1], 0.0, gate_passed=False, cost=1.0
        ),
        now=RECORD_AT,
    )
    driver.engine.mark_scores_persisted(driver.conn, cid, SCORES_AT)

    r1 = repo.get_contender(driver.conn, ids["hk-1"])
    r2 = repo.get_contender(driver.conn, ids["hk-2"])
    assert r1 is not None and r2 is not None
    # Anchors are PER ITEM: item 0's cheapest valid cost is hk-2's 1.0; on item 1
    # hk-1 holds the only valid row, so it anchors item 1 itself at 2.0.
    # hk-1: (1.0/2.0 + 2.0/2.0) / 2 items = 0.75.
    assert r1.cost_efficiency_aggregate == pytest.approx(0.75)
    # hk-2: (1.0/1.0 + ZERO for the gate-failed item) / 2 items = 0.5 — the old
    # valid-rows-only mean would have given a perfect 1.0.
    assert r2.cost_efficiency_aggregate == pytest.approx(0.5)

    # A MISSING row zeroes its item's contribution exactly like a gate-failed one.
    driver.conn.execute(
        "DELETE FROM performance_history WHERE contender_id = ? AND item_id = ?",
        (ids["hk-2"], item_ids[1]),
    )
    recalculate_ranks(driver.conn, cid, SCORES_AT + timedelta(minutes=5))
    r2 = repo.get_contender(driver.conn, ids["hk-2"])
    assert r2 is not None
    assert r2.cost_efficiency_aggregate == pytest.approx(0.5)


def test_cost_efficiency_anchor_is_per_item(driver: Driver) -> None:
    """review #8 residual (round 3): the cost anchor is computed PER evaluation item
    — a cheap item's anchor must not suppress the efficiency earned on an expensive
    item whose costs live on a different scale."""
    cid, ids, item_ids = driver.run_to_evaluating(build_manifest(), ["hk-1", "hk-2"])
    driver.engine.mark_evaluation_complete(driver.conn, cid, RECORD_AT)
    # Item 0 is cheap (costs ~1), item 1 is expensive (costs ~100). hk-1 is the
    # cheapest on BOTH items; hk-2 pays exactly double on both.
    costs = {
        ids["hk-1"]: {item_ids[0]: 1.0, item_ids[1]: 100.0},
        ids["hk-2"]: {item_ids[0]: 2.0, item_ids[1]: 200.0},
    }
    for contender_id, per_item in costs.items():
        for item_id, cost in per_item.items():
            repo.record_item_score(
                driver.conn, cid, contender_id=contender_id, item_id=item_id,
                packet_bytes=driver.make_packet(contender_id, item_id, 0.8, cost=cost),
                now=RECORD_AT,
            )
    driver.engine.mark_scores_persisted(driver.conn, cid, SCORES_AT)

    r1 = repo.get_contender(driver.conn, ids["hk-1"])
    r2 = repo.get_contender(driver.conn, ids["hk-2"])
    assert r1 is not None and r2 is not None
    # hk-1 is the per-item anchor on both items: (1/1 + 100/100) / 2 = 1.0. The old
    # GLOBAL anchor (1.0) would have crushed its expensive item to 1/100 -> 0.505.
    assert r1.cost_efficiency_aggregate == pytest.approx(1.0)
    # hk-2 pays double on each item: (1/2 + 100/200) / 2 = 0.5 — not the old
    # globally-anchored (1/2 + 1/200) / 2 = 0.2525.
    assert r2.cost_efficiency_aggregate == pytest.approx(0.5)
