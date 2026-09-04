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
    minimum_payout_score: float = 0.10

    idle_inference_share: float = 0.80
    idle_burn_share: float = 0.20
    podium_inference_share: float = 0.60
    podium_competition_share: float = 0.40
    crown_inference_share: float = 0.10
    crown_competition_share: float = 0.90
    breakthrough_margin_floor: float = 0.05
    result_window_hours: float = 168.0

    # False forces IDLE; it never redirects IDLE's 20% sink share to inference.
    competition_emissions_enabled: bool = False
    empty_pool_policy: Literal["withhold", "redistribute"] = "withhold"
    retention_full_window_required: bool = True
    track_weights: dict[str, float] = {"compression": 0.8, "upscaling": 0.2}

    @model_validator(mode="after")
    def _validate(self) -> "TokenomicsConfig":
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
        if (
            not 0.0
            < self.crown_inference_share
            < self.podium_inference_share
            < self.idle_inference_share
            < 1.0
        ):
            raise ValueError(
                "inference shares must satisfy 0 < CROWN < PODIUM < IDLE < 1"
            )
        if not 0.0 < self.podium_competition_share < self.crown_competition_share < 1.0:
            raise ValueError("competition shares must satisfy 0 < PODIUM < CROWN < 1")
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
