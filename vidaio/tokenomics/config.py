"""Configuration for the schema-v15 emission state machine.

The three allocations are protocol values, not normalisation hints: unavailable
inference or podium shares go to the caller-supplied canonical sink.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class TokenomicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    burn_proportion: float = 0.0
    alpha_stake_weigh_factor: float = 0.0
    emission_liquidation_weigh_factor: float = 5.0
    ewma_decay: float = 0.75
    top_n_per_track: int = 5
    minimum_payout_score: float = 0.05
    # Inference eligibility only; never scales a score or a competition award.
    payout_min_alpha_stake: float = 0.0

    # Tokenomics v3 (2026-09-23): inference no longer earns; a competition result
    # pays 100 % of miner emissions to its top places until the next result, and
    # an idle subnet burns. Each competition may override the payout policy in its
    # anchored result rules; these are the protocol defaults.
    idle_inference_share: float = 0.0
    idle_burn_share: float = 1.0
    podium_inference_share: float = 0.0
    podium_competition_share: float = 1.0
    crown_inference_share: float = 0.0
    crown_competition_share: float = 1.0
    #: Per-rank fractions of the competition pot (1 to 5 places, each summing to 1).
    crown_split: tuple[float, ...] = (0.90, 0.04, 0.03, 0.02, 0.01)
    podium_split: tuple[float, ...] = (0.50, 0.24, 0.13, 0.08, 0.05)
    #: Shares of unfilled places go to the filled ones in proportion (else: sink).
    redistribute_empty_places: bool = True
    #: A result with no qualifying contender closes the running window (burn until
    #: the next result) instead of leaving the previous window untouched.
    no_qualifier_closes_window: bool = True
    breakthrough_margin_floor: float = 0.05
    result_window_hours: float = 168.0

    # False forces IDLE; it never redirects IDLE's 20% sink share to inference.
    competition_emissions_enabled: bool = False
    empty_pool_policy: Literal["withhold", "redistribute"] = "withhold"
    retention_full_window_required: bool = True
    track_weights: dict[str, float] = {"compression": 0.8, "upscaling": 0.2}

    @model_validator(mode="after")
    def _validate(self) -> "TokenomicsConfig":
        if (
            not math.isfinite(self.payout_min_alpha_stake)
            or self.payout_min_alpha_stake < 0.0
        ):
            raise ValueError("payout_min_alpha_stake must be finite and >= 0")
        if not 0.0 <= self.burn_proportion <= 1.0:
            raise ValueError("burn_proportion must be in [0, 1]")
        if self.alpha_stake_weigh_factor < 0.0:
            raise ValueError("alpha_stake_weigh_factor must be >= 0")
        if self.emission_liquidation_weigh_factor < 0.0:
            raise ValueError("emission_liquidation_weigh_factor must be >= 0")
        if not 0.0 < self.ewma_decay < 1.0:
            raise ValueError("ewma_decay must be in (0, 1)")
        if self.top_n_per_track < 1:
            raise ValueError("top_n_per_track must be >= 1")
        if (
            not math.isfinite(self.minimum_payout_score)
            or not 0.0 < self.minimum_payout_score <= 1.0
        ):
            raise ValueError("minimum_payout_score must be finite and in (0, 1]")

        for name in (
            "idle_inference_share",
            "idle_burn_share",
            "podium_inference_share",
            "podium_competition_share",
            "crown_inference_share",
            "crown_competition_share",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        # Exact checks prevent implementation-specific renormalisation.
        for state, left, right in (
            ("IDLE", self.idle_inference_share, self.idle_burn_share),
            ("PODIUM", self.podium_inference_share, self.podium_competition_share),
            ("CROWN", self.crown_inference_share, self.crown_competition_share),
        ):
            if left + right != 1.0:
                raise ValueError(f"{state} allocation shares must sum exactly to 1.0")
        if not (
            self.crown_inference_share
            <= self.podium_inference_share
            <= self.idle_inference_share
        ):
            raise ValueError(
                "inference shares must satisfy CROWN <= PODIUM <= IDLE"
            )
        if not 0.0 < self.podium_competition_share <= self.crown_competition_share <= 1.0:
            raise ValueError("competition shares must satisfy 0 < PODIUM <= CROWN <= 1")
        from vidaio.tokenomics.state import _validate_split

        object.__setattr__(self, "crown_split", _validate_split("crown_split", tuple(self.crown_split)))
        object.__setattr__(self, "podium_split", _validate_split("podium_split", tuple(self.podium_split)))
        if (
            not math.isfinite(self.breakthrough_margin_floor)
            or not 0.0 < self.breakthrough_margin_floor < 1.0
        ):
            raise ValueError("breakthrough_margin_floor must be finite and in (0, 1)")
        if not math.isfinite(self.result_window_hours) or self.result_window_hours <= 0:
            raise ValueError("result_window_hours must be finite and > 0")
        if not self.track_weights:
            raise ValueError("track_weights must declare at least one track")
        if any(not math.isfinite(w) or w <= 0 for w in self.track_weights.values()):
            raise ValueError("every track weight must be finite and > 0")
        if sum(self.track_weights.values()) != 1.0:
            raise ValueError("track_weights must sum exactly to 1.0")
        return self
