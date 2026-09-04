"""Worst-decile bottleneck aggregation vs mean; competition aggregate weights."""

import pytest

from vidaio.scoring import (
    AggregateWeights,
    ScoringConfig,
    competition_final_score,
    length_weighted_mean,
    worst_decile_from_config,
    worst_decile_score,
)


def test_worst_decile_punishes_one_failure_where_mean_hides_it() -> None:
    scores = [1.0] * 9 + [0.0]
    assert sum(scores) / len(scores) == pytest.approx(0.9)  # mean hides the failure
    assert worst_decile_score(scores) == 0.0  # bottleneck does not


def test_worst_decile_sizing_uses_ceil() -> None:
    # 25 items, fraction 0.1 -> ceil(2.5) = 3 worst items
    scores = [0.0, 0.1, 0.2] + [1.0] * 22
    assert worst_decile_score(scores) == pytest.approx(0.1)


def test_worst_decile_small_n_takes_at_least_one() -> None:
    assert worst_decile_score([0.4, 0.9]) == 0.4
    assert worst_decile_score([0.7]) == 0.7


def test_worst_decile_edge_cases() -> None:
    assert worst_decile_score([]) == 0.0
    assert worst_decile_score([0.5] * 10, fraction=1.0) == 0.5
    with pytest.raises(ValueError):
        worst_decile_score([0.5], fraction=0.0)


def test_worst_decile_order_independent() -> None:
    a = [0.9, 0.1, 0.5, 1.0, 0.3, 0.8, 0.2, 0.7, 0.6, 0.4]
    assert worst_decile_score(a) == worst_decile_score(sorted(a, reverse=True))


def test_worst_decile_from_config() -> None:
    cfg = ScoringConfig(worst_decile_fraction=0.5)
    assert worst_decile_from_config([0.0, 0.2, 1.0, 1.0], cfg) == pytest.approx(0.1)


def test_length_weighted_mean() -> None:
    assert length_weighted_mean([(1.0, 10.0), (0.0, 30.0)]) == pytest.approx(0.25)
    assert length_weighted_mean([]) == 0.0
    assert length_weighted_mean([(0.9, 0.0)]) == 0.0
    with pytest.raises(ValueError):
        length_weighted_mean([(0.5, -1.0)])


def test_competition_final_score_default_weights() -> None:
    # 0.6*0.8 + 0.25*0.5 + 0.15*1.0 = 0.755
    assert competition_final_score(
        quality=0.8, cost_efficiency=0.5, length_coverage=1.0
    ) == pytest.approx(0.755)


def test_worst_decile_raises_on_non_finite_scores() -> None:
    # Fail closed: NaN sorts arbitrarily and would poison or dodge the worst bucket.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="score"):
            worst_decile_score([0.5, bad, 0.9])
    assert worst_decile_score([]) == 0.0  # empty stays the documented 0.0


def test_length_weighted_mean_raises_on_non_finite() -> None:
    with pytest.raises(ValueError, match="value"):
        length_weighted_mean([(float("nan"), 10.0)])
    with pytest.raises(ValueError, match="weight"):
        length_weighted_mean([(0.5, float("inf"))])
    assert length_weighted_mean([]) == 0.0  # empty stays the documented 0.0


def test_competition_final_score_raises_on_non_finite() -> None:
    with pytest.raises(ValueError, match="quality"):
        competition_final_score(
            quality=float("nan"), cost_efficiency=0.5, length_coverage=1.0
        )
    with pytest.raises(ValueError, match="cost_efficiency"):
        competition_final_score(
            quality=0.5, cost_efficiency=float("inf"), length_coverage=1.0
        )
    with pytest.raises(ValueError, match="length_coverage"):
        competition_final_score(
            quality=0.5, cost_efficiency=0.5, length_coverage=float("nan")
        )


def test_competition_final_score_injected_manifest_weights() -> None:
    # live competition-01 override: 0.6 / 0.00 / 0.4 — cost efficiency zeroed
    weights = AggregateWeights(quality=0.6, cost_efficiency=0.0, length_coverage=0.4)
    assert competition_final_score(
        quality=0.5, cost_efficiency=1.0, length_coverage=0.5, weights=weights
    ) == pytest.approx(0.5)
