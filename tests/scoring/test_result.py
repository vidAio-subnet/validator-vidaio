"""ItemScore JSON round-trip carries everything the audit recompute needs."""

import pytest
from pydantic import ValidationError

from vidaio.scoring import (
    GateSkip,
    ItemScore,
    ReasonCode,
    ScoringConfig,
    ValidityViolation,
    compose_item_score,
    config_digest,
    score_compression,
    score_upscaling,
)
from vidaio.scoring.compression import CompressionBreakdown
from vidaio.scoring.upscaling import UpscalingBreakdown

CFG = ScoringConfig()


def test_round_trip_compression_item() -> None:
    breakdown = score_compression(
        candidate_bytes=500, reference_bytes=1000, vmaf=92.0, config=CFG
    )
    item = compose_item_score(
        item_id="item-1",
        challenge_id="chal-1",
        track="compression",
        gate_passed=True,
        violations=[],
        breakdown=breakdown,
        config=CFG,
        miner_hotkey="5Fexample",
        content_digest="abc123",
        metrics={
            "vmaf_primary": 92.0,
            "vmaf_secondary": 91.2,
            "candidate_bytes": 500,
            "reference_bytes": 1000,
        },
        backend_versions={"vmaf": "libvmaf/3.0 NEG"},
        canonicalization_plan_digest="c" * 64,
    )
    restored = ItemScore.from_json(item.to_json())
    assert restored == item
    # everything the recompute needs survived
    assert isinstance(restored.breakdown, CompressionBreakdown)
    assert restored.breakdown.compression_rate == 0.5
    assert restored.breakdown.normalizer == 1.12
    assert restored.metrics["vmaf_secondary"] == 91.2
    assert restored.backend_versions["vmaf"] == "libvmaf/3.0 NEG"
    assert restored.canonicalization_plan_digest == "c" * 64
    assert restored.scoring_config_digest == config_digest(CFG)
    assert restored.score == breakdown.final


def test_round_trip_upscaling_item_with_violations() -> None:
    breakdown = score_upscaling(pieapp=0.5, content_length=100.0, config=CFG)
    violations = [
        ValidityViolation(
            code=ReasonCode.VMAF_BELOW_FLOOR, detail="x", measured=42.0, limit=50.0
        )
    ]
    item = compose_item_score(
        item_id="item-2",
        challenge_id="chal-2",
        track="upscaling",
        gate_passed=False,
        violations=violations,
        breakdown=breakdown,
        config=CFG,
        pieapp_start_frame=335,
    )
    assert item.score == 0.0  # gates-first invariant, even with a computed breakdown
    restored = ItemScore.from_json(item.to_json())
    assert restored == item
    assert isinstance(restored.breakdown, UpscalingBreakdown)  # discriminator held
    assert restored.breakdown.s_pre == breakdown.s_pre
    assert restored.pieapp_start_frame == 335
    assert restored.violations[0].code == ReasonCode.VMAF_BELOW_FLOOR


def test_config_digest_tracks_config_changes() -> None:
    assert config_digest(CFG) == config_digest(ScoringConfig())
    assert config_digest(CFG) != config_digest(ScoringConfig(compression_norm=1.13))


def test_compose_without_breakdown_scores_zero() -> None:
    item = compose_item_score(
        item_id="item-3",
        challenge_id="chal-3",
        track="compression",
        gate_passed=True,
        violations=[],
        breakdown=None,
        config=CFG,
    )
    assert item.score == 0.0
    assert item.skips == []


def test_gate_skips_round_trip_through_item_json() -> None:
    # A consciously-disabled check is part of the persisted audit packet, not
    # transient GateContext state (compose_item_score threads GateContext.skips).
    skip = GateSkip(
        gate="vmaf_model_delta",
        detail="secondary VMAF run absent; check disabled by require_secondary_vmaf=False",
    )
    breakdown = score_compression(
        candidate_bytes=500, reference_bytes=1000, vmaf=92.0, config=CFG
    )
    item = compose_item_score(
        item_id="item-4",
        challenge_id="chal-4",
        track="compression",
        gate_passed=True,
        violations=[],
        breakdown=breakdown,
        config=CFG,
        skips=[skip],
    )
    assert item.skips == [skip]
    restored = ItemScore.from_json(item.to_json())
    assert restored == item
    assert restored.skips == [skip]
    assert restored.skips[0].gate == "vmaf_model_delta"
    assert "require_secondary_vmaf" in restored.skips[0].detail
    # and the field is visibly present in the serialized packet
    assert '"skips"' in item.to_json()


def _valid_item_kwargs() -> dict:
    return dict(
        item_id="item-5",
        challenge_id="chal-5",
        track="compression",
        score=0.5,
        gate_passed=True,
    )


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf"), 1.0000001, -0.0001]
)
def test_non_finite_or_out_of_range_score_is_unconstructible(bad: float) -> None:
    kwargs = _valid_item_kwargs()
    kwargs["score"] = bad
    with pytest.raises(ValidationError):
        ItemScore(**kwargs)


@pytest.mark.parametrize("bad_json", ["Infinity", "-Infinity", "NaN", "1.5", "-0.1"])
def test_bad_score_packet_is_unparseable(bad_json: str) -> None:
    # An Infinity packet must be rejected at model_validate_json too — a packet that
    # cannot exist in memory must not be slipped in through the JSON door.
    item = ItemScore(**_valid_item_kwargs())
    payload = item.to_json().replace('"score":0.5', f'"score":{bad_json}')
    assert f'"score":{bad_json}' in payload  # the tamper landed
    with pytest.raises(ValidationError):
        ItemScore.from_json(payload)
    with pytest.raises(ValidationError):
        ItemScore.model_validate_json(payload)


def test_score_boundaries_zero_and_one_are_valid() -> None:
    for ok in (0.0, 1.0):
        kwargs = _valid_item_kwargs()
        kwargs["score"] = ok
        assert ItemScore(**kwargs).score == ok


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5, -0.5])
def test_breakdown_final_is_bounded_too(bad: float) -> None:
    good = score_compression(
        candidate_bytes=500, reference_bytes=1000, vmaf=92.0, config=CFG
    )
    with pytest.raises(ValidationError):
        CompressionBreakdown(**{**good.model_dump(), "final": bad})
    good_up = score_upscaling(pieapp=0.5, content_length=100.0, config=CFG)
    with pytest.raises(ValidationError):
        UpscalingBreakdown(**{**good_up.model_dump(), "final": bad})


def test_breakdown_final_infinity_unparseable_inside_item_json() -> None:
    breakdown = score_upscaling(pieapp=0.5, content_length=100.0, config=CFG)
    item = compose_item_score(
        item_id="item-6",
        challenge_id="chal-6",
        track="upscaling",
        gate_passed=True,
        violations=[],
        breakdown=breakdown,
        config=CFG,
    )
    payload = item.to_json()
    needle = f'"final":{breakdown.final}'
    assert needle in payload
    tampered = payload.replace(needle, '"final":Infinity')
    with pytest.raises(ValidationError):
        ItemScore.from_json(tampered)
