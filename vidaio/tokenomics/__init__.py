"""Pure schema-v15 scoring accumulation and emission composition."""

from vidaio.tokenomics.breakthrough import (
    PODIUM_SPLIT,
    active_emission_state,
    contender_margin,
    emission_shares,
    podium_hotkey_shares,
    qualifies_for_crown,
    resolve_reward_window,
    window_active,
    winner,
)
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.ewma import EXCLUDED_SCORE, accumulate, is_excluded
from vidaio.tokenomics.quantize import U16_MAX, max_normalize_u16, quantize_u16
from vidaio.tokenomics.rank_curve import (
    dedup_ip_key,
    eligible_for_ranking,
    inference_shares,
    track_shares,
)
from vidaio.tokenomics.state import (
    CompetitionResult,
    ContenderResult,
    EmissionShares,
    EmissionState,
    MinerSnapshot,
    RewardWindowState,
)
from vidaio.tokenomics.weights import (
    build_weight_vector,
    ensure_alpha_stake_factor_disabled,
    ensure_locked_levers,
)

__all__ = [
    "TokenomicsConfig",
    "MinerSnapshot",
    "ContenderResult",
    "CompetitionResult",
    "EmissionState",
    "EmissionShares",
    "RewardWindowState",
    "EXCLUDED_SCORE",
    "accumulate",
    "is_excluded",
    "U16_MAX",
    "max_normalize_u16",
    "quantize_u16",
    "ensure_alpha_stake_factor_disabled",
    "eligible_for_ranking",
    "dedup_ip_key",
    "track_shares",
    "inference_shares",
    "PODIUM_SPLIT",
    "contender_margin",
    "qualifies_for_crown",
    "winner",
    "resolve_reward_window",
    "window_active",
    "active_emission_state",
    "emission_shares",
    "podium_hotkey_shares",
    "build_weight_vector",
    "ensure_locked_levers",
]
