"""Golden-value tests for the compression formula, incl. every boundary case."""

import pytest

from vidaio.scoring import ReasonCode, ScoringConfig, score_compression
from vidaio.scoring.compression import compression_rate, compression_score_from_rate

CFG = ScoringConfig()  # threshold 90, band 5, weights 0.7/0.3, norm 1.12, rate max 0.80


def test_golden_mid_case() -> None:
    # rate 0.5 -> comp 0.5; vmaf 92 -> quality 0.92
    # (0.7*0.5 + 0.3*0.92) / 1.12 = 0.626/1.12
    b = score_compression(
        candidate_bytes=500, reference_bytes=1000, vmaf=92.0, config=CFG
    )
    assert b.compression_rate == 0.5
    assert b.compression_score == 0.5
    assert b.quality_score == pytest.approx(0.92)
    assert b.final == pytest.approx(0.5589285714285713)
    assert b.zero_reason is None


def test_rate_exactly_at_max_scores_zero() -> None:
    b = score_compression(
        candidate_bytes=800, reference_bytes=1000, vmaf=99.0, config=CFG
    )
    assert b.compression_rate == pytest.approx(0.80)
    assert b.final == 0.0
    assert b.zero_reason == ReasonCode.COMPRESSION_RATE_TOO_HIGH


def test_rate_just_below_max_scores() -> None:
    b = score_compression(
        candidate_bytes=799, reference_bytes=1000, vmaf=99.0, config=CFG
    )
    assert b.zero_reason is None
    assert b.final > 0.0


def test_vmaf_exactly_at_threshold_scores_full_formula() -> None:
    # rate 0.4 -> comp 0.6; vmaf 90 -> quality 0.9; (0.42+0.27)/1.12
    b = score_compression(
        candidate_bytes=400, reference_bytes=1000, vmaf=90.0, config=CFG
    )
    assert b.zero_reason is None
    assert b.final == pytest.approx(0.6160714285714285)


def test_vmaf_exactly_at_threshold_minus_band_is_zero_near_miss() -> None:
    # Documented reading: the spec defines no formula for [threshold-5, threshold),
    # so the band scores 0 — with the distinct near-miss reason code.
    b = score_compression(
        candidate_bytes=400, reference_bytes=1000, vmaf=85.0, config=CFG
    )
    assert b.final == 0.0
    assert b.zero_reason == ReasonCode.VMAF_BELOW_THRESHOLD


def test_vmaf_below_floor_is_zero_with_floor_reason() -> None:
    b = score_compression(
        candidate_bytes=400, reference_bytes=1000, vmaf=84.999, config=CFG
    )
    assert b.final == 0.0
    assert b.zero_reason == ReasonCode.VMAF_BELOW_FLOOR


def test_best_case_stays_below_one_with_default_weights() -> None:
    # comp 0.999, quality 1.0 -> 0.9993/1.12 — the min(1, .) clamp never binds
    # with the spec weights; theoretical max ~0.893.
    b = score_compression(candidate_bytes=1, reference_bytes=1000, vmaf=100.0, config=CFG)
    assert b.final == pytest.approx(0.8922321428571427)


def test_min_clamp_binds_with_nondefault_weights() -> None:
    cfg = ScoringConfig(compression_weights={"comp": 1.0, "vmaf": 1.0})
    b = score_compression(candidate_bytes=1, reference_bytes=1000, vmaf=100.0, config=cfg)
    assert b.final == 1.0


def test_manifest_threshold_override() -> None:
    b = score_compression(
        candidate_bytes=400,
        reference_bytes=1000,
        vmaf=80.0,
        config=CFG,
        vmaf_threshold=80.0,
    )
    assert b.zero_reason is None
    assert b.vmaf_threshold == 80.0


def test_rate_helpers() -> None:
    assert compression_rate(250, 1000) == 0.25
    with pytest.raises(ValueError):
        compression_rate(1, 0)
    assert compression_score_from_rate(0.25) == 0.75
    assert compression_score_from_rate(1.5) == 0.0
    assert compression_score_from_rate(-0.1) == 1.0


def test_non_finite_vmaf_raises_fail_closed() -> None:
    # A NaN VMAF must never compose into a score (it previously sailed past every
    # case-check and produced a formula final).
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="vmaf"):
            score_compression(
                candidate_bytes=500, reference_bytes=1000, vmaf=bad, config=CFG
            )


def test_non_finite_threshold_override_raises() -> None:
    with pytest.raises(ValueError, match="vmaf_threshold"):
        score_compression(
            candidate_bytes=500,
            reference_bytes=1000,
            vmaf=92.0,
            config=CFG,
            vmaf_threshold=float("nan"),
        )


def test_non_finite_rate_raises() -> None:
    with pytest.raises(ValueError, match="compression_rate"):
        compression_score_from_rate(float("nan"))
    with pytest.raises(ValueError, match="compression_rate"):
        compression_score_from_rate(float("inf"))


def test_breakdown_records_every_term() -> None:
    b = score_compression(
        candidate_bytes=500, reference_bytes=1000, vmaf=92.0, config=CFG
    )
    assert b.candidate_bytes == 500
    assert b.reference_bytes == 1000
    assert b.weight_comp == 0.7
    assert b.weight_vmaf == 0.3
    assert b.normalizer == 1.12
    assert b.vmaf_threshold == 90.0
