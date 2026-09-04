"""Multi-item aggregation — worst-decile bottleneck + length-weighted competition score.

Spec §18: "use worst-decile / bottleneck aggregation over a simple average" — one
excellent item must never offset a failed one, so the round score is the mean of the
*worst* fraction of item scores. Gate-failed items enter as 0.0, giving min-of-gates
semantics: any gated failure that lands inside the worst decile drags the aggregate
toward zero.

Competition aggregate (spec §02)::

    final_score = 0.6*quality + 0.25*cost_efficiency + 0.15*length_coverage

length-weighted, with the weights injectable from the competition manifest (the live
competition-01 manifest, for example, overrides to 0.6 / 0.00 / 0.4).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from vidaio.scoring.config import AggregateWeights, ScoringConfig
from vidaio.scoring.finite import require_finite


def worst_decile_score(
    scores: Sequence[float], *, fraction: float = 0.1
) -> float:
    """Mean of the worst ``ceil(n * fraction)`` scores (at least one). Empty -> 0.0.

    Deterministic: ties broken by sorted order of the values themselves; input order
    never matters. Fail closed: any non-finite score raises (NaN sorts arbitrarily
    and would poison — or silently miss — the worst bucket).
    """
    if not scores:
        return 0.0
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    for score in scores:
        require_finite("score", score)
    ordered = sorted(scores)
    k = max(1, math.ceil(len(ordered) * fraction))
    worst = ordered[:k]
    return sum(worst) / len(worst)


def worst_decile_from_config(scores: Sequence[float], config: ScoringConfig) -> float:
    return worst_decile_score(scores, fraction=config.worst_decile_fraction)


def length_weighted_mean(pairs: Iterable[tuple[float, float]]) -> float:
    """Mean of (value, weight) pairs weighted by content length. Zero weight -> 0.0.
    Non-finite values or weights raise (fail closed)."""
    total_weight = 0.0
    total = 0.0
    for value, weight in pairs:
        require_finite("value", value)
        require_finite("weight", weight)
        if weight < 0:
            raise ValueError("weights must be >= 0")
        total += value * weight
        total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return total / total_weight


def competition_final_score(
    *,
    quality: float,
    cost_efficiency: float,
    length_coverage: float,
    weights: AggregateWeights | None = None,
) -> float:
    """final_score = w_q*quality + w_c*cost_efficiency + w_l*length_coverage.

    Weights default to the spec 0.6/0.25/0.15 and are injectable from the manifest.
    Non-finite term values raise (fail closed).
    """
    require_finite("quality", quality)
    require_finite("cost_efficiency", cost_efficiency)
    require_finite("length_coverage", length_coverage)
    w = weights if weights is not None else AggregateWeights()
    return (
        w.quality * quality
        + w.cost_efficiency * cost_efficiency
        + w.length_coverage * length_coverage
    )
