"""Tokenomics v4: the licensing treasury takes half of a gap and 0.5 % of a window."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.commitments import REWARD_POLICY_DOMAIN, reward_parameter_digest
from vidaio.competition.podium_snapshots import CARRIER_FALLBACK_TRACK, with_treasury_carrier
from vidaio.tokenomics import EMISSION_PROFILES, TokenomicsConfig
from vidaio.tokenomics.state import EmissionState, MinerSnapshot, RewardWindowState
from vidaio.tokenomics.weights import build_weight_vector, emission_profile, ensure_locked_levers

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)
TREASURY = "hk-treasury"
BURN = 0


def _v4(**overrides) -> TokenomicsConfig:
    return TokenomicsConfig(
        **{
            **EMISSION_PROFILES["v4"],
            "treasury_hotkey": TREASURY,
            "result_window_hours": 336.0,
            "competition_emissions_enabled": True,
            **overrides,
        }
    )


def _row(uid: int, hotkey: str) -> MinerSnapshot:
    return MinerSnapshot(
        uid=uid, hotkey=hotkey, coldkey=f"ck-{hotkey}", ip=f"10.0.0.{uid}",
        track=CARRIER_FALLBACK_TRACK, accumulate_score=0.0,
    )


def _crown_window() -> RewardWindowState:
    return RewardWindowState(
        kind=EmissionState.CROWN, starts_at=NOW - timedelta(hours=1),
        ends_at=NOW + timedelta(days=13), podium_hotkeys=("winner",),
        competition_share=1.0, place_shares=(1.0,), winner_hotkey="winner", winner_uid=5,
        winner_score=0.8, winner_margin=0.8, baseline_score=0.0, baseline_version=0,
        baseline_artifact_digest="ab" * 32, source_competition_id="interp",
        source_track="interpolation", source_cycle=1, last_applied_cycle=1,
    )


def test_v4_is_a_supported_profile_and_v3_stays_the_default() -> None:
    assert emission_profile(TokenomicsConfig()) == "v3"
    assert emission_profile(_v4()) == "v4"
    ensure_locked_levers(_v4())
    with pytest.raises(ValueError, match="supported profile"):
        ensure_locked_levers(_v4(idle_burn_share=0.7, idle_treasury_share=0.3))
    with pytest.raises(ValueError, match="supported profile"):
        ensure_locked_levers(_v4(window_treasury_share=0.05))


def test_a_gap_pays_half_to_the_registered_treasury_and_burns_half() -> None:
    weights = build_weight_vector(
        _v4(), [_row(7, TREASURY)], burn_uid=BURN, reward_state=RewardWindowState(), now=NOW
    )
    assert weights == {7: 0.5, BURN: 0.5}


def test_an_unregistered_or_unset_treasury_share_burns() -> None:
    assert build_weight_vector(_v4(), [], burn_uid=BURN, reward_state=RewardWindowState(), now=NOW) == {
        BURN: 1.0
    }
    unset = _v4(treasury_hotkey=None)
    assert build_weight_vector(
        unset, [_row(7, TREASURY)], burn_uid=BURN, reward_state=RewardWindowState(), now=NOW
    ) == {7: 0.0, BURN: 1.0}


def test_an_active_window_keeps_half_a_percent_for_the_treasury() -> None:
    weights = build_weight_vector(
        _v4(), [_row(5, "winner"), _row(7, TREASURY)], burn_uid=BURN,
        reward_state=_crown_window(), now=NOW,
    )
    assert weights[5] == pytest.approx(0.995) and weights[7] == pytest.approx(0.005)
    assert BURN not in weights or weights[BURN] == pytest.approx(0.0, abs=1e-12)
    assert sum(weights.values()) == pytest.approx(1.0)
    # v3 (no treasury) pays the whole pot to the window
    v3 = build_weight_vector(
        TokenomicsConfig(competition_emissions_enabled=True), [_row(5, "winner")],
        burn_uid=BURN, reward_state=_crown_window(), now=NOW,
    )
    assert v3 == {5: 1.0}


def test_v3_reward_policy_digest_is_unchanged_by_the_treasury_fields() -> None:
    config = TokenomicsConfig()
    legacy = sha256_hex(
        canonical_json_bytes(
            {
                "domain": REWARD_POLICY_DOMAIN,
                "tokenomics": config.model_dump(
                    mode="json",
                    exclude={
                        "payout_min_alpha_stake", "treasury_hotkey",
                        "idle_treasury_share", "window_treasury_share",
                    },
                ),
                "competition_podium_split": list(config.podium_split),
                "competition_crown_split": list(config.crown_split),
            }
        )
    )
    assert reward_parameter_digest(config) == legacy
    assert reward_parameter_digest(_v4()) != legacy
    assert reward_parameter_digest(_v4()) != reward_parameter_digest(_v4(treasury_hotkey="other"))


def test_treasury_carrier_row_only_for_a_registered_hotkey() -> None:
    neurons = [
        SimpleNamespace(uid=3, hotkey="miner", coldkey="ck-m", ip="10.0.0.3", alpha_stake=1.0),
        SimpleNamespace(uid=7, hotkey=TREASURY, coldkey="ck-t", ip="0.0.0.0", alpha_stake=0.0),
    ]
    rows = with_treasury_carrier([], chain_neurons=neurons, treasury_hotkey=TREASURY)
    assert [(r.uid, r.hotkey, r.accumulate_score, r.track) for r in rows] == [
        (7, TREASURY, 0.0, CARRIER_FALLBACK_TRACK)
    ]
    assert with_treasury_carrier([], chain_neurons=neurons, treasury_hotkey=None) == []
    assert with_treasury_carrier([], chain_neurons=neurons[:1], treasury_hotkey=TREASURY) == []


def test_treasury_config_is_validated() -> None:
    with pytest.raises(ValueError, match="IDLE allocation shares must sum"):
        TokenomicsConfig(idle_burn_share=0.5, idle_treasury_share=0.6)
    with pytest.raises(ValueError, match="below 1.0"):
        TokenomicsConfig(window_treasury_share=1.0)
    with pytest.raises(ValueError, match="non-empty"):
        TokenomicsConfig(treasury_hotkey="  ")
