"""Deterministic v2 weight composition with explicit canonical-sink routing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Iterable

from vidaio.core import get_logger, log_fields
from vidaio.tokenomics.breakthrough import emission_shares, podium_hotkey_shares
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.rank_curve import inference_shares
from vidaio.tokenomics.state import EmissionState, MinerSnapshot, RewardWindowState

logger = get_logger("vidaio.tokenomics.weights")

_LAUNCH_TOP_N_PER_TRACK = 5
_LAUNCH_MINIMUM_PAYOUT_SCORE = 0.05  # 0.10 -> 0.05 on 2026-09-15: the upscaling score curve tops out near 0.10, the pool burned
_LAUNCH_TRACK_WEIGHTS = {"compression": 0.8, "upscaling": 0.2}
# Tokenomics v3 (2026-09-23): inference earns nothing, a result pays 100 % of miner
# emissions, an idle subnet burns. Per-competition rules may override the pot and
# the split for THEIR window; these pins keep the protocol defaults from drifting
# through configuration.
# Exactly two emission profiles are valid; anything in between is configuration drift.
#   v3 (mainnet from 2026-09-27): inference earns nothing, a result pays 100 %.
#   v2 (launch, 2026-09-04 .. 09-27): 80/20 idle, 60/40 podium, 10/90 crown, 70/20/10.
_V3_PROFILE = {
    "idle_inference_share": 0.0,
    "idle_burn_share": 1.0,
    "podium_inference_share": 0.0,
    "podium_competition_share": 1.0,
    "crown_inference_share": 0.0,
    "crown_competition_share": 1.0,
    "crown_split": (0.90, 0.04, 0.03, 0.02, 0.01),
    "podium_split": (0.50, 0.24, 0.13, 0.08, 0.05),
    "redistribute_empty_places": True,
    "no_qualifier_closes_window": True,
}
_V2_PROFILE = {
    "idle_inference_share": 0.80,
    "idle_burn_share": 0.20,
    "podium_inference_share": 0.60,
    "podium_competition_share": 0.40,
    "crown_inference_share": 0.10,
    "crown_competition_share": 0.90,
    "crown_split": (0.70, 0.20, 0.10),
    "podium_split": (0.70, 0.20, 0.10),
    "redistribute_empty_places": False,
    "no_qualifier_closes_window": False,
}
EMISSION_PROFILES = {"v3": _V3_PROFILE, "v2": _V2_PROFILE}
_LAUNCH_BREAKTHROUGH_MARGIN_FLOOR = 0.05


def emission_profile(config: TokenomicsConfig) -> str | None:
    """Name of the supported emission profile the config matches exactly, or None."""
    for name, profile in EMISSION_PROFILES.items():
        if all(
            (tuple(getattr(config, key)) if isinstance(expected, tuple) else getattr(config, key))
            == expected
            for key, expected in profile.items()
        ):
            return name
    return None


def ensure_alpha_stake_factor_disabled(config: TokenomicsConfig) -> None:
    if config.alpha_stake_weigh_factor != 0.0:
        raise ValueError(
            "alpha_stake_weigh_factor must stay 0.0 — buyable stake must not become weight"
        )


def ensure_locked_levers(config: TokenomicsConfig) -> None:
    """Keep discretionary redistribution disabled; state-derived sink shares are live."""
    if config.burn_proportion != 0.0:
        raise ValueError(
            "burn_proportion must stay 0.0 — v2 sink allocation is derived only from "
            "the committed emission state and unavailable fixed shares"
        )
    if not config.retention_full_window_required:
        raise ValueError(
            "retention_full_window_required must stay true as a validated compatibility lever"
        )
    if config.top_n_per_track != _LAUNCH_TOP_N_PER_TRACK:
        raise ValueError(
            f"top_n_per_track must stay {_LAUNCH_TOP_N_PER_TRACK} for launch"
        )
    if config.minimum_payout_score != _LAUNCH_MINIMUM_PAYOUT_SCORE:
        raise ValueError("minimum_payout_score must stay 0.05 (launch pin, lowered 2026-09-15)")
    if config.track_weights != _LAUNCH_TRACK_WEIGHTS:
        raise ValueError(
            "track_weights must stay compression=0.8/upscaling=0.2 for launch"
        )
    if emission_profile(config) is None:
        raise ValueError(
            "emission shares/splits must match exactly one supported profile "
            "(v3: inference 0, result pays 100 %, 90/4/3/2/1 and 50/24/13/8/5, "
            "redistribute, close-on-no-qualifier; or the v2 launch profile)"
        )
    if config.breakthrough_margin_floor != _LAUNCH_BREAKTHROUGH_MARGIN_FLOOR:
        raise ValueError("breakthrough_margin_floor must stay 0.05 for launch")
    if config.empty_pool_policy != "withhold":
        raise ValueError(
            "empty_pool_policy must stay 'withhold'; unavailable shares go to the canonical sink"
        )
    ensure_alpha_stake_factor_disabled(config)


def build_weight_vector(
    config: TokenomicsConfig,
    miners: Iterable[MinerSnapshot],
    *,
    burn_uid: int | None = None,
    reward_state: RewardWindowState | None = None,
    now: datetime | None = None,
) -> dict[int, float]:
    """Compose uid shares from inference standing plus the active global window.

    Missing inference tracks, ineligible miners, deregistered podium hotkeys and the
    idle burn are never redistributed here; they must be explicitly assigned to the
    canonical ``burn_uid`` (omitting a required sink fails closed before chain
    normalisation can cross-subsidise earners). Unfilled podium PLACES are handled
    by the window's own payout policy (``place_shares``), which may already have
    redistributed them to the filled places when the result was folded.
    """
    ensure_locked_levers(config)
    miners = sorted(miners, key=lambda miner: miner.uid)
    by_uid = {miner.uid: miner for miner in miners}
    by_hotkey = {miner.hotkey: miner.uid for miner in miners}
    if len(by_uid) != len(miners):
        raise ValueError("miner snapshot contains duplicate uids")
    if len(by_hotkey) != len(miners):
        raise ValueError("miner snapshot contains duplicate hotkeys")
    if burn_uid is not None and (
        isinstance(burn_uid, bool) or not isinstance(burn_uid, int) or burn_uid < 0
    ):
        raise ValueError("burn_uid must be a non-negative integer")
    state = reward_state or RewardWindowState()
    if config.competition_emissions_enabled and now is None:
        raise ValueError("now is required while competition emissions are enabled")
    effective_now = now or datetime.min.replace(tzinfo=UTC)
    pools = emission_shares(config, state, effective_now)

    vector: dict[int, float] = {miner.uid: 0.0 for miner in miners}
    inf_shares = inference_shares(config, miners)
    for uid, share in inf_shares.items():
        vector[uid] += pools.inference * share

    for hotkey, share in podium_hotkey_shares(state, config).items():
        uid = by_hotkey.get(hotkey)
        if uid is not None:
            vector[uid] += pools.competition * share

    total = sum(vector.values())
    withheld = max(0.0, 1.0 - total)
    if withheld > 1e-12 and burn_uid is None:
        raise ValueError(
            "canonical burn_uid is required whenever fixed allocation remains "
            f"unassigned (withheld={withheld:.17g})"
        )
    if withheld > 1e-12:
        assert burn_uid is not None
        if burn_uid in by_uid:
            raise ValueError(f"burn_uid {burn_uid} overlaps an economic miner snapshot")
        vector[burn_uid] = vector.get(burn_uid, 0.0) + withheld
        total = sum(vector.values())

    logger.info(
        "composed tokenomics-v2 weight pools",
        extra=log_fields(
            emission_state=(
                state.kind.value if pools.competition else EmissionState.IDLE.value
            ),
            inference_pool=pools.inference,
            competition_pool=pools.competition,
            protocol_burn_pool=pools.burn,
            miners=len(miners),
            inference_eligible=len(inf_shares),
            allocated_total=total,
            withheld=withheld,
        ),
    )
    return vector
