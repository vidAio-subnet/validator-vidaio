"""Human review & rank recalculation — spec: design spec §04 (review window) and §06 (schema).

Reviews are append-only and hash-chained; corrections happen via a superseding row,
never by mutation. Disqualify / reinstate / tie-break are allowed only while the
competition sits in AWAITING_END_TIME and `now` is at or before human_review_deadline.

recalculate_ranks() re-ranks strictly from the persisted per-item scores — media is
NEVER re-run (spec §04: "recalculation in AWAITING_END_TIME re-ranks without
re-running media"). Calibration (baseline) and disqualified contenders are excluded from
ranking; the baseline additionally can never hold a final_rank at the schema level
(the project design record #1).
"""

from __future__ import annotations

import math
import sqlite3
from contextlib import contextmanager, nullcontext
from datetime import datetime
from typing import Any, Iterator

from vidaio.core.logging import get_logger, log_fields
from vidaio.competition import repository as repo
from vidaio.competition.states import Phase

logger = get_logger("vidaio.competition.review")

ACTION_DISQUALIFY = "DISQUALIFY"
ACTION_REINSTATE = "REINSTATE"
ACTION_TIE_BREAK = "TIE_BREAK"
_ACTIONS = {ACTION_DISQUALIFY, ACTION_REINSTATE, ACTION_TIE_BREAK}


class ReviewError(Exception):
    pass


class ReviewWindowClosed(ReviewError):
    pass


@contextmanager
def _txn(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def submit_review(
    conn: sqlite3.Connection,
    competition_id: str,
    *,
    contender_id: int,
    action: str,
    reviewer: str,
    reason: str,
    now: datetime,
    detail: dict[str, Any] | None = None,
    supersedes_review_id: int | None = None,
) -> int:
    """Append a human review, apply its effect, and re-rank. Returns the review_id."""
    if now.tzinfo is None:
        raise ValueError("submit_review requires a timezone-aware `now`")
    action = action.upper()
    if action not in _ACTIONS:
        raise ReviewError(f"unknown review action {action!r}")

    comp = repo.get_competition(conn, competition_id)
    if comp is None:
        raise ReviewError(f"unknown competition {competition_id}")
    if comp.status is not Phase.AWAITING_END_TIME:
        raise ReviewWindowClosed(
            f"reviews only allowed in AWAITING_END_TIME (competition is {comp.status})"
        )
    if comp.human_review_deadline is None or now > comp.human_review_deadline:
        raise ReviewWindowClosed(
            f"human review window closed at {comp.human_review_deadline}"
        )

    contender = repo.get_contender(conn, contender_id)
    if contender is None or contender.competition_id != competition_id:
        raise ReviewError(f"contender {contender_id} not part of {competition_id}")
    if contender.is_calibration:
        raise ReviewError(
            "the baseline calibration contender is outside ranking/payout by construction; "
            "it cannot be the subject of a review (the project design record #1)"
        )

    if action == ACTION_REINSTATE:
        _check_reinstate(conn, competition_id, contender_id, supersedes_review_id)
    if action == ACTION_TIE_BREAK:
        _check_tie_break(conn, competition_id, contender_id, detail)

    with _txn(conn):
        review_id = repo.insert_human_review(
            conn,
            competition_id,
            contender_id=contender_id,
            action=action,
            reviewer=reviewer,
            reason=reason,
            now=now,
            detail=detail,
            supersedes_review_id=supersedes_review_id,
        )
        if action == ACTION_DISQUALIFY:
            conn.execute(
                "UPDATE contenders SET manual_disqualified = 1, eligible = 0, updated_at = ?"
                " WHERE contender_id = ?",
                (repo.iso(now), contender_id),
            )
        elif action == ACTION_REINSTATE:
            conn.execute(
                "UPDATE contenders SET manual_disqualified = 0, eligible = 1, updated_at = ?"
                " WHERE contender_id = ?",
                (repo.iso(now), contender_id),
            )
        repo.record_event(
            conn,
            competition_id,
            "human_review",
            now,
            payload={
                "review_id": review_id,
                "contender_id": contender_id,
                "action": action,
                "supersedes_review_id": supersedes_review_id,
            },
        )
    logger.info(
        "human review recorded",
        extra=log_fields(
            competition_id=competition_id,
            phase=comp.status.value,
            review_id=review_id,
            contender_id=contender_id,
            action=action,
        ),
    )
    recalculate_ranks(conn, competition_id, now)
    return review_id


def _check_reinstate(
    conn: sqlite3.Connection,
    competition_id: str,
    contender_id: int,
    supersedes_review_id: int | None,
) -> None:
    if supersedes_review_id is None:
        raise ReviewError("REINSTATE must supersede a prior DISQUALIFY review")
    prior = conn.execute(
        "SELECT * FROM human_reviews WHERE review_id = ?", (supersedes_review_id,)
    ).fetchone()
    if (
        prior is None
        or prior["competition_id"] != competition_id
        or prior["contender_id"] != contender_id
        or prior["action"] != ACTION_DISQUALIFY
    ):
        raise ReviewError(
            "REINSTATE must supersede a DISQUALIFY review of the same contender "
            f"in the same competition (got review {supersedes_review_id})"
        )


def _check_tie_break(
    conn: sqlite3.Connection,
    competition_id: str,
    contender_id: int,
    detail: dict[str, Any] | None,
) -> None:
    if not detail or "wins_over_contender_id" not in detail:
        raise ReviewError('TIE_BREAK requires detail {"wins_over_contender_id": <id>}')
    other_id = detail["wins_over_contender_id"]
    if other_id == contender_id:
        raise ReviewError("TIE_BREAK cannot reference the same contender")
    other = repo.get_contender(conn, other_id)
    if other is None or other.competition_id != competition_id:
        raise ReviewError(f"tie-break counterpart {other_id} not part of {competition_id}")


# ---- rank recalculation -------------------------------------------------------

def _worst_decile(scores: list[float], fraction: float = 0.1) -> float:
    """Mean of the worst ceil(n * fraction) scores (at least one). Empty -> 0.0.
    Mirrors vidaio/scoring/aggregate.py:worst_decile_score — reimplemented locally so
    the competition module stays independent of scoring internals."""
    if not scores:
        return 0.0
    ordered = sorted(scores)
    k = max(1, math.ceil(len(ordered) * fraction))
    worst = ordered[:k]
    return sum(worst) / len(worst)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def recalculate_ranks(
    conn: sqlite3.Connection,
    competition_id: str,
    now: datetime,
    *,
    manage_txn: bool = True,
) -> list[tuple[int, int]]:
    """Recompute aggregates + final_score from persisted per-item scores, then re-rank.

    Never touches media or sandboxes. Aggregation rules:

    - EVERY persisted item participates: gate-failed/invalid rows enter every
      aggregate as ZERO — a contender's failures drag its aggregate down, never
      vanish. Items the contender has no row for count as zero too.
    - A contender with NO score rows at all gets ZERO aggregates and final_score
      0.0 — it is still RANKED (last, behind anyone with a positive score), never
      silently dropped from the ranking.
    - Cost efficiency is aggregated over ALL evaluation items and anchored PER
      ITEM: for each item the anchor is the cheapest valid positive cost recorded
      on THAT item across all contenders (calibration included: the baseline
      legitimately anchors the cost bar; it still earns nothing). A valid row with
      a positive cost contributes min(1, item_anchor / own_cost); a gate-failed,
      cost-less or missing row contributes ZERO — failures never shrink the
      denominator, and a cheap item can never suppress the efficiency earned on
      an expensive one (per-item cost scales stay independent).
    - Item length comes ONLY from evaluation_items.length_seconds; the per-response
      length_seconds column is never trusted for aggregation.
    - length_coverage sums the item lengths of VALID rows and is clamped to [0, 1].
    - Both quality aggregates are computed and stored: the length-weighted mean
      (media_score_aggregate) and the worst-decile mean (worst_decile_aggregate,
      spec §18). The quality term of final_score is the length-weighted mean unless
      the manifest sets use_worst_decile.

    Included in scoring: every contender — including the calibration baseline (its
    aggregates are the calibration signal) and contenders with no rows (all-zero).
    Included in RANKING: only eligible, non-disqualified, non-calibration
    contenders — zero-scored ones included, ranked last. Returns the new
    [(contender_id, final_rank), ...].

    manage_txn=False is for callers already inside a transaction (the engine's
    single-transaction SCORING -> AWAITING_END_TIME commit).
    """
    if now.tzinfo is None:
        raise ValueError("recalculate_ranks requires a timezone-aware `now`")
    comp = repo.get_competition(conn, competition_id)
    if comp is None:
        raise ReviewError(f"unknown competition {competition_id}")
    if comp.status is not Phase.AWAITING_END_TIME:
        raise ReviewError(
            f"ranks are recalculated in AWAITING_END_TIME (competition is {comp.status})"
        )
    manifest = repo.get_manifest(conn, competition_id)
    factors = manifest.scoring_factors

    item_lengths: dict[int, float] = {
        row["item_id"]: row["length"]
        for row in conn.execute(
            "SELECT item_id, COALESCE(length_seconds, 1.0) AS length"
            " FROM evaluation_items WHERE competition_id = ?",
            (competition_id,),
        )
    }
    total_length = sum(item_lengths.values())

    rows_by_contender: dict[int, list[sqlite3.Row]] = {}
    for row in conn.execute(
        """SELECT contender_id, item_id, valid, item_score, vmaf, compression_rate, cost
           FROM performance_history WHERE competition_id = ?""",
        (competition_id,),
    ):
        rows_by_contender.setdefault(row["contender_id"], []).append(row)

    # Per-item cost anchors: for EACH evaluation item, the
    # cheapest valid positive cost recorded on THAT item across all contenders
    # (calibration included: the baseline legitimately anchors the cost bar; it still
    # earns nothing). Anchoring per item keeps cost scales independent — a cheap
    # item can never suppress the efficiency earned on an expensive one.
    cheapest_cost_by_item: dict[int, float] = {}
    for rows in rows_by_contender.values():
        for r in rows:
            if r["valid"] and r["cost"] is not None and r["cost"] > 0:
                iid = r["item_id"]
                if iid not in cheapest_cost_by_item or r["cost"] < cheapest_cost_by_item[iid]:
                    cheapest_cost_by_item[iid] = r["cost"]

    contenders = repo.list_contenders(conn, competition_id)
    scored: dict[int, dict[str, float | None]] = {}
    for contender in contenders:
        rows = rows_by_contender.get(contender.contender_id)
        if not rows or not total_length:
            # No score rows (or no items at all): ZERO aggregates, final_score 0.0.
            # The contender still gets RANKED — last — never dropped.
            scored[contender.contender_id] = {
                "media_score_aggregate": 0.0,
                "worst_decile_aggregate": 0.0,
                "cost_efficiency_aggregate": 0.0,
                "length_coverage": 0.0,
                "average_vmaf": None,
                "average_compression_rate": None,
                "final_score": 0.0,
            }
            continue
        # Effective per-item score: gate-failed rows are zero (the schema also forces
        # item_score = 0 when valid = 0); items with no row at all are zero.
        by_item = {r["item_id"]: (r["item_score"] if r["valid"] else 0.0) for r in rows}
        media = (
            sum(score * item_lengths[iid] for iid, score in by_item.items()) / total_length
        )
        worst_decile = _worst_decile([by_item.get(iid, 0.0) for iid in item_lengths])
        covered = sum(item_lengths[r["item_id"]] for r in rows if r["valid"])
        coverage = min(1.0, max(0.0, covered / total_length))
        valid_rows = [r for r in rows if r["valid"]]
        # Cost efficiency over ALL items: a valid row with a positive cost
        # contributes min(1, item_anchor / own_cost) against ITS OWN item's anchor;
        # gate-failed, cost-less and missing items contribute ZERO — they never
        # leave the denominator. (A valid positive-cost row always has an anchor:
        # the row itself participated in its item's minimum.)
        eff_by_item = {
            r["item_id"]: min(1.0, cheapest_cost_by_item[r["item_id"]] / r["cost"])
            for r in valid_rows
            if r["cost"] is not None and r["cost"] > 0
        }
        cost_eff = sum(eff_by_item.get(iid, 0.0) for iid in item_lengths) / len(
            item_lengths
        )
        quality = worst_decile if manifest.use_worst_decile else media
        final = (
            factors.quality * quality
            + factors.cost_efficiency * cost_eff
            + factors.length_coverage * coverage
        )
        scored[contender.contender_id] = {
            "media_score_aggregate": media,
            "worst_decile_aggregate": worst_decile,
            "cost_efficiency_aggregate": cost_eff,
            "length_coverage": coverage,
            "average_vmaf": _mean([r["vmaf"] for r in valid_rows if r["vmaf"] is not None]),
            "average_compression_rate": _mean(
                [r["compression_rate"] for r in valid_rows if r["compression_rate"] is not None]
            ),
            "final_score": final,
        }

    # Every eligible, non-disqualified, non-calibration contender is ranked —
    # including all-zero ones (final_score is always a float now; zero ranks last,
    # ties resolved deterministically).
    rankable = [
        c
        for c in contenders
        if not c.is_calibration and not c.manual_disqualified and c.eligible
    ]
    prefers = _tie_break_preferences(conn, competition_id)
    ordered = _order_with_tie_breaks(
        [(c.contender_id, scored[c.contender_id]["final_score"]) for c in rankable],  # type: ignore[misc]
        prefers,
    )
    ranks = {contender_id: idx + 1 for idx, contender_id in enumerate(ordered)}

    ts = repo.iso(now)
    with _txn(conn) if manage_txn else nullcontext():
        for contender in contenders:
            s = scored[contender.contender_id]
            conn.execute(
                """UPDATE contenders SET
                       media_score_aggregate = ?, worst_decile_aggregate = ?,
                       cost_efficiency_aggregate = ?,
                       length_coverage = ?, average_vmaf = ?, average_compression_rate = ?,
                       final_score = ?, final_rank = ?, updated_at = ?
                   WHERE contender_id = ?""",
                (
                    s["media_score_aggregate"],
                    s["worst_decile_aggregate"],
                    s["cost_efficiency_aggregate"],
                    s["length_coverage"],
                    s["average_vmaf"],
                    s["average_compression_rate"],
                    s["final_score"],
                    ranks.get(contender.contender_id),  # NULL for calibration/DQ/unscored
                    ts,
                    contender.contender_id,
                ),
            )
        repo.record_event(
            conn,
            competition_id,
            "ranks_recalculated",
            now,
            payload={"ranking": [[cid, ranks[cid]] for cid in ordered]},
        )
    logger.info(
        "ranks recalculated",
        extra=log_fields(
            competition_id=competition_id,
            phase=comp.status.value,
            ranked=len(ordered),
            contenders=len(contenders),
        ),
    )
    return [(cid, ranks[cid]) for cid in ordered]


def _tie_break_preferences(
    conn: sqlite3.Connection, competition_id: str
) -> set[tuple[int, int]]:
    """Effective (winner, loser) pairs from non-superseded TIE_BREAK reviews."""
    import json

    prefers: set[tuple[int, int]] = set()
    for row in repo.effective_reviews(conn, competition_id):
        if row["action"] != ACTION_TIE_BREAK or not row["detail_json"]:
            continue
        detail = json.loads(row["detail_json"])
        loser = detail.get("wins_over_contender_id")
        if loser is not None:
            prefers.add((row["contender_id"], int(loser)))
    return prefers


def _order_with_tie_breaks(
    scores: list[tuple[int, float]], prefers: set[tuple[int, int]]
) -> list[int]:
    """Order by final_score desc; exact ties resolved by tie-break preference, then by
    contender_id (deterministic)."""
    by_score: dict[float, list[int]] = {}
    for contender_id, score in scores:
        by_score.setdefault(score, []).append(contender_id)
    ordered: list[int] = []
    for score in sorted(by_score, reverse=True):
        group = sorted(by_score[score])
        placed: list[int] = []
        remaining = list(group)
        while remaining:
            # pick the lowest-id contender that no other remaining contender beats
            pick = None
            for candidate in remaining:
                if not any((other, candidate) in prefers for other in remaining if other != candidate):
                    pick = candidate
                    break
            if pick is None:  # preference cycle — fall back to id order
                pick = remaining[0]
            placed.append(pick)
            remaining.remove(pick)
        ordered.extend(placed)
    return ordered
