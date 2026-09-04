import pytest
from pydantic import ValidationError

from vidaio.challenge import (
    DAG_VERSION,
    LAUNCH_MAX_ELIGIBILITY_SCAN_ASSETS,
    ChallengeConfig,
)
from vidaio.core.config import section


def test_defaults() -> None:
    cfg = ChallengeConfig()
    assert cfg.dag_version == DAG_VERSION
    assert cfg.min_seed_bits == 128
    assert cfg.retire_after_uses == 1
    assert cfg.tracks == ["compression", "upscaling"]
    assert 0.0 <= cfg.holdout_fraction <= 1.0
    assert cfg.min_clip_seconds == 4.0
    assert cfg.upscaling_min_clip_seconds == 10.0
    assert cfg.max_clip_seconds == 12.0
    assert cfg.max_eligibility_scan_assets == LAUNCH_MAX_ELIGIBILITY_SCAN_ASSETS
    assert cfg.split_key_fields == ["creator", "source"]


def test_section_loading() -> None:
    raw = {"challenge": {"retire_after_uses": 3, "holdout_fraction": 0.25}}
    cfg = section(raw, "challenge", ChallengeConfig)
    assert cfg.retire_after_uses == 3
    assert cfg.holdout_fraction == 0.25


def test_empty_section_uses_defaults() -> None:
    assert section({}, "challenge", ChallengeConfig) == ChallengeConfig()


def test_invalid_clip_bounds_rejected() -> None:
    with pytest.raises(ValidationError):
        ChallengeConfig(min_clip_seconds=10, max_clip_seconds=5)
    with pytest.raises(ValidationError):
        ChallengeConfig(upscaling_min_clip_seconds=3)
    with pytest.raises(ValidationError):
        ChallengeConfig(upscaling_min_clip_seconds=13)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("min_clip_seconds", float("nan")),
        ("upscaling_min_clip_seconds", float("inf")),
        ("max_clip_seconds", float("inf")),
    ),
)
def test_clip_bounds_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        ChallengeConfig(**{field: value})


def test_invalid_split_key_field_rejected() -> None:
    with pytest.raises(ValidationError):
        ChallengeConfig(split_key_fields=["clip"])  # per-clip splits are forbidden


def test_min_seed_bits_cannot_be_weakened_below_floor() -> None:
    with pytest.raises(ValidationError):
        ChallengeConfig(min_seed_bits=64)  # 128 is a hard security floor
    assert ChallengeConfig(min_seed_bits=256).min_seed_bits == 256


def test_eligibility_scan_bound_is_positive_and_cannot_exceed_qualified_ceiling() -> (
    None
):
    assert (
        ChallengeConfig(max_eligibility_scan_assets=1).max_eligibility_scan_assets == 1
    )
    with pytest.raises(ValidationError):
        ChallengeConfig(max_eligibility_scan_assets=0)
    with pytest.raises(ValidationError):
        ChallengeConfig(
            max_eligibility_scan_assets=LAUNCH_MAX_ELIGIBILITY_SCAN_ASSETS + 1
        )
