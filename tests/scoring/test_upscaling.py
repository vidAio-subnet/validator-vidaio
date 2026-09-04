"""Golden-value tests for the upscaling formula, incl. the spec anchor points."""

import math

import pytest

from vidaio.scoring import ScoringConfig, score_upscaling
from vidaio.scoring.upscaling import final_from_pre, length_score, quality_from_pieapp

CFG = ScoringConfig()


def test_anchor_spre_half_gives_exactly_coefficient() -> None:
    # s_pre = 0.5 -> final = 0.1 * exp(0) = 0.1 exactly.
    assert final_from_pre(0.5, CFG) == pytest.approx(0.1)
    # Full path: pieapp 1.0 -> s_q 0.5; content_length sqrt(321)-1 -> s_l 0.5.
    b = score_upscaling(
        pieapp=1.0, content_length=math.sqrt(321) - 1, config=CFG
    )
    assert b.s_q == pytest.approx(0.5)
    assert b.s_l == pytest.approx(0.5)
    assert b.s_pre == pytest.approx(0.5)
    assert b.final == pytest.approx(0.1)


def test_anchor_spre_one_clamps_to_one() -> None:
    # raw = 0.1 * e^(6.979*0.5) ~ 3.277 -> clamped to 1.
    assert final_from_pre(1.0, CFG) == 1.0
    b = score_upscaling(pieapp=0.0, content_length=320.0, config=CFG)
    assert b.s_q == 1.0
    assert b.s_l == pytest.approx(1.0)
    assert b.final == 1.0


def test_golden_mid_case() -> None:
    # pieapp 0.5 -> s_q 2/3; content_length 100 -> s_l = ln(101)/ln(321)
    b = score_upscaling(pieapp=0.5, content_length=100.0, config=CFG)
    assert b.s_q == pytest.approx(2 / 3)
    assert b.s_l == pytest.approx(0.7996478554282382)
    assert b.s_pre == pytest.approx(0.7331572610474524)
    assert b.final == pytest.approx(0.5089626887600393)


def test_quality_mapping_monotone_decreasing_and_clamped() -> None:
    assert quality_from_pieapp(0.0) == 1.0
    # negative PieAPP (better than reference) clamps at distance 0 -> 1.0
    assert quality_from_pieapp(-0.7) == 1.0
    assert quality_from_pieapp(1.0) == pytest.approx(0.5)
    values = [quality_from_pieapp(d) for d in (0.0, 0.5, 1.0, 2.0, 5.0, 50.0)]
    assert values == sorted(values, reverse=True)
    assert 0.0 < values[-1] < 0.02


def test_length_score_clamps_at_base_minus_one() -> None:
    assert length_score(0.0, CFG) == 0.0
    assert length_score(320.0, CFG) == pytest.approx(1.0)
    assert length_score(10_000.0, CFG) == 1.0
    with pytest.raises(ValueError):
        length_score(-1.0, CFG)


def test_final_never_leaves_unit_interval() -> None:
    for s_pre in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        assert 0.0 <= final_from_pre(s_pre, CFG) <= 1.0


def test_non_finite_pieapp_raises_fail_closed() -> None:
    # NaN PieAPP previously clamped through max(0, nan) to a PERFECT s_q of 1.0.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="pieapp"):
            quality_from_pieapp(bad)
        with pytest.raises(ValueError, match="pieapp"):
            score_upscaling(pieapp=bad, content_length=100.0, config=CFG)


def test_non_finite_content_length_raises() -> None:
    # NaN < 0 is False, so NaN previously slipped past the negative-length check.
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="content_length"):
            length_score(bad, CFG)
        with pytest.raises(ValueError, match="content_length"):
            score_upscaling(pieapp=0.5, content_length=bad, config=CFG)


def test_non_finite_s_pre_raises() -> None:
    with pytest.raises(ValueError, match="s_pre"):
        final_from_pre(float("nan"), CFG)
