"""Challenge module configuration (config section: `challenge`).

Levers for the degrade->restore challenge factory: which DAG version to build,
how aggressively assets are retired, clip length bounds for the ingest segmenter,
and the leakage-control split parameters (salt + grouping fields + holdout size).

Note: the *structure* of the degradation DAG (operator ranges, stage ordering) is
versioned inside vidaio.challenge.dag and keyed by `dag_version` — it is not
config-tunable, so a config edit can never silently change what a digest means.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

from vidaio.challenge.dag import DAG_VERSION

SPLIT_KEY_CHOICES = ("creator", "source", "subject", "scene")

# Launch calibration requires enough temporal content for an upscaling miner to
# beat the perceptual gates.  Production/release guards pin this floor; keeping
# the constant beside the schema gives every caller one definition of the rule.
LAUNCH_UPSCALING_MIN_CLIP_SECONDS = 10.0
# Challenge production validates every fresh candidate's immutable segment bytes
# before checkout.  Keep that work explicitly bounded: the certified launch corpus
# contains at most 96 simultaneously-fresh challenge assets, and increasing this
# ceiling requires a new release qualification rather than an operator-only edit.
LAUNCH_MAX_ELIGIBILITY_SCAN_ASSETS = 96
# ``segment_time`` is a target: muxer/frame timestamp quantization may put an
# otherwise correct final timestamp just above it.  This allowance is upper-only;
# minimum floors never receive epsilon and the measured duration is never clamped.
MAX_CLIP_DURATION_OVERSHOOT_SECONDS = 0.100


class ChallengeConfig(BaseModel):
    # Which procedural-DAG version build_dag() must produce.
    dag_version: int = Field(DAG_VERSION, ge=1)
    # Minimum private-seed entropy accepted by make_challenge. Seeds MUST come from
    # a CSPRNG at the call site (secrets.randbits(256) or better); date-sized or
    # counter seeds are brute-forceable from public dispatch material. The floor of
    # 128 is a hard security bound — config may only raise it.
    min_seed_bits: int = Field(128, ge=128)
    # An asset is retired after this many issued uses (spec default: single-use).
    retire_after_uses: int = Field(1, ge=1)
    # Clip length bounds used by the ingest segmenter/checkout plan. Minimums are
    # exact. max_clip_seconds is the segment_time target; produced timestamps may
    # use only the upper-only MAX_CLIP_DURATION_OVERSHOOT_SECONDS allowance.
    min_clip_seconds: float = Field(4.0, gt=0)
    upscaling_min_clip_seconds: float = Field(LAUNCH_UPSCALING_MIN_CLIP_SECONDS, gt=0)
    max_clip_seconds: float = Field(12.0, gt=0)
    # Maximum number of fresh challenge-split assets one challenge-production pass
    # may hash + ffprobe.  The query reads one sentinel row beyond this value and
    # fails closed instead of silently ignoring a candidate (which would bias draws).
    max_eligibility_scan_assets: int = Field(
        LAUNCH_MAX_ELIGIBILITY_SCAN_ASSETS,
        ge=1,
        le=LAUNCH_MAX_ELIGIBILITY_SCAN_ASSETS,
    )
    # Competition tracks this deployment produces challenges for.
    tracks: list[str] = Field(default_factory=lambda: ["compression", "upscaling"])
    # Fraction of source groups routed to the sealed holdout split.
    holdout_fraction: float = Field(0.1, ge=0.0, le=1.0)
    # Private salt for the deterministic split hash. Rotate only with a full re-split.
    split_salt: str = "vidaio-split-v1"
    # Which asset identity fields form the split grouping key. Splits are computed
    # per source group, NEVER per clip; the default groups by (creator, source) so
    # every clip from one source lands in the same split.
    split_key_fields: list[str] = Field(default_factory=lambda: ["creator", "source"])

    @model_validator(mode="after")
    def _check(self) -> "ChallengeConfig":
        duration_bounds = {
            "min_clip_seconds": self.min_clip_seconds,
            "upscaling_min_clip_seconds": self.upscaling_min_clip_seconds,
            "max_clip_seconds": self.max_clip_seconds,
        }
        non_finite = [
            name for name, value in duration_bounds.items() if not math.isfinite(value)
        ]
        if non_finite:
            raise ValueError(
                "clip duration bounds must be finite: " + ", ".join(non_finite)
            )
        if not (
            self.min_clip_seconds
            <= self.upscaling_min_clip_seconds
            <= self.max_clip_seconds
        ):
            raise ValueError(
                "clip duration bounds must satisfy min_clip_seconds <= "
                "upscaling_min_clip_seconds <= max_clip_seconds"
            )
        if not self.tracks:
            raise ValueError("tracks must not be empty")
        if not self.split_key_fields:
            raise ValueError("split_key_fields must not be empty")
        bad = [f for f in self.split_key_fields if f not in SPLIT_KEY_CHOICES]
        if bad:
            raise ValueError(
                f"invalid split_key_fields {bad}; allowed: {SPLIT_KEY_CHOICES}"
            )
        return self
