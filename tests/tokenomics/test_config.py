from __future__ import annotations

import pytest
from pydantic import ValidationError

from vidaio.tokenomics import TokenomicsConfig


def test_v3_defaults_are_exact_protocol_allocations() -> None:
    cfg = TokenomicsConfig()
    assert (cfg.idle_inference_share, cfg.idle_burn_share) == (0.0, 1.0)
    assert (cfg.podium_inference_share, cfg.podium_competition_share) == (0.0, 1.0)
    assert (cfg.crown_inference_share, cfg.crown_competition_share) == (0.0, 1.0)
    assert cfg.crown_split == (0.90, 0.04, 0.03, 0.02, 0.01)
    assert cfg.podium_split == (0.50, 0.24, 0.13, 0.08, 0.05)
    assert cfg.redistribute_empty_places and cfg.no_qualifier_closes_window
    assert cfg.result_window_hours == 168
    assert cfg.breakthrough_margin_floor == 0.05
    assert cfg.track_weights == {"compression": 0.8, "upscaling": 0.2}
    assert cfg.burn_proportion == cfg.alpha_stake_weigh_factor == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"idle_inference_share": 0.81},
        {"idle_burn_share": 0.19, "idle_inference_share": 0.80},
        {"podium_inference_share": 0.61},
        {"crown_competition_share": 0.89},
        {"idle_inference_share": float("nan")},
        {"podium_competition_share": float("inf")},
        {"crown_inference_share": -0.1, "crown_competition_share": 1.1},
        {"breakthrough_margin_floor": 0.0},
        {"crown_split": (0.9, 0.2)},
        {"podium_split": ()},
        {"podium_split": (0.5, 0.2, 0.1, 0.1, 0.05, 0.05)},
        {"crown_inference_share": 0.5, "crown_competition_share": 0.5, "podium_inference_share": 0.2, "podium_competition_share": 0.8},
        {"breakthrough_margin_floor": 1.0},
        {"breakthrough_margin_floor": float("nan")},
        {"result_window_hours": 0.0},
        {"result_window_hours": float("inf")},
        {"alpha_stake_weigh_factor": -0.1},
        {"emission_liquidation_weigh_factor": -1.0},
        {"burn_proportion": -0.1},
        {"burn_proportion": 1.5},
        {"ewma_decay": 0.0},
        {"ewma_decay": 1.0},
        {"top_n_per_track": 0},
        {"minimum_payout_score": 0.0},
        {"minimum_payout_score": float("nan")},
        {"empty_pool_policy": "renormalize"},
        {"track_weights": {}},
        {"track_weights": {"compression": 1.0, "upscaling": 0.0}},
        {"track_weights": {"compression": 0.5, "upscaling": 0.2}},
    ],
)
def test_invalid_configs_rejected(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        TokenomicsConfig(**overrides)


def test_testnet_duration_override_is_allowed() -> None:
    assert TokenomicsConfig(result_window_hours=0.25).result_window_hours == 0.25


def test_unknown_and_retired_elastic_keys_are_forbidden() -> None:
    for key in (
        "surprise_lever",
        "inference_pool",
        "competition_pool_standing",
        "breakthrough_pool_max",
        "breakthrough_margin_full",
        "reign_duration_hours",
        "staleness_ttl_cycles",
        "legacy_calibration_earning",
    ):
        with pytest.raises(ValidationError):
            TokenomicsConfig(**{key: 1})


def test_range_valid_but_locked_levers_parse_for_explicit_guard_error() -> None:
    assert (
        TokenomicsConfig(alpha_stake_weigh_factor=0.5).alpha_stake_weigh_factor == 0.5
    )
    assert TokenomicsConfig(burn_proportion=0.05).burn_proportion == 0.05
