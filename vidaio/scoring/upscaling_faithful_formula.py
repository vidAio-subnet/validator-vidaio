"""The faithful video-upscaling score formula (dependency-light: no NumPy, no OpenCV).

The media measurements live in :mod:`vidaio.scoring.upscaling_faithful`; this module only
turns the measured metrics of the upscaled frames and the validator's Lanczos floor into the
score, so packet models and auditors can import it without the media stack.

"Faithful, not generative": the candidate is compared with the sealed original that the
served input was degraded from, so invented detail costs PSNR/SSIM/LPIPS instead of earning
it. Each term credits only the improvement over the free Lanczos upscale of the served input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, field_validator

from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.finite import require_finite

#: Upscale factors the faithful track scores (served input -> reference geometry).
UPSCALING_FAITHFUL_FACTORS = (2, 4)


@dataclass(frozen=True)
class UpscaledFrameMetrics:
    """Quality of upscaled frames against the sealed original frames."""

    psnr_db: float
    ssim: float
    lpips_vgg: float
    frames: int


class UpscalingFaithfulBreakdown(BaseModel):
    """Every term of the faithful upscaling formula — the audit-recompute record."""

    model_config = {"frozen": True}

    kind: Literal["upscaling_faithful"] = "upscaling_faithful"
    upscale_factor: int
    psnr_db: float
    ssim: float
    lpips_vgg: float
    frames: int
    #: the Lanczos upscale of the served input, measured by the validator with the same metrics
    floor_psnr_db: float
    floor_ssim: float
    floor_lpips_vgg: float
    s_psnr: float
    s_ssim: float
    s_lpips: float
    weight_psnr: float
    weight_ssim: float
    weight_lpips: float
    final: float

    @field_validator("final")
    @classmethod
    def _final_bounded(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"final must be finite, got {value!r}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"final must be in [0, 1], got {value!r}")
        return value


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def score_upscaling_faithful(
    *,
    metrics: UpscaledFrameMetrics,
    floor: UpscaledFrameMetrics,
    upscale_factor: int,
    config: ScoringConfig,
) -> UpscalingFaithfulBreakdown:
    """Compose the faithful upscaling score from the candidate metrics and the Lanczos floor.

    Each term credits only the improvement over the floor, as a share of what is left to
    gain: PSNR over a fixed span, SSIM and LPIPS over the distance to a perfect frame.
    Returning the Lanczos upscale itself (or anything no closer to the original) scores 0."""
    for name, value in (
        ("psnr_db", metrics.psnr_db),
        ("ssim", metrics.ssim),
        ("lpips_vgg", metrics.lpips_vgg),
        ("floor.psnr_db", floor.psnr_db),
        ("floor.ssim", floor.ssim),
        ("floor.lpips_vgg", floor.lpips_vgg),
    ):
        require_finite(name, value)
    if upscale_factor not in UPSCALING_FAITHFUL_FACTORS:
        raise ValueError(f"upscale_factor {upscale_factor} not in {UPSCALING_FAITHFUL_FACTORS}")
    weights = config.upscaling_faithful_weights
    s_psnr = _clamp01((metrics.psnr_db - floor.psnr_db) / config.upscaling_faithful_psnr_span_db)
    ssim_headroom = 1.0 - floor.ssim
    s_ssim = _clamp01((metrics.ssim - floor.ssim) / ssim_headroom) if ssim_headroom > 0 else 0.0
    s_lpips = (
        _clamp01((floor.lpips_vgg - metrics.lpips_vgg) / floor.lpips_vgg)
        if floor.lpips_vgg > 0
        else 0.0
    )
    final = _clamp01(weights.psnr * s_psnr + weights.ssim * s_ssim + weights.lpips * s_lpips)
    return UpscalingFaithfulBreakdown(
        upscale_factor=upscale_factor,
        psnr_db=metrics.psnr_db,
        ssim=metrics.ssim,
        lpips_vgg=metrics.lpips_vgg,
        frames=metrics.frames,
        floor_psnr_db=floor.psnr_db,
        floor_ssim=floor.ssim,
        floor_lpips_vgg=floor.lpips_vgg,
        s_psnr=s_psnr,
        s_ssim=s_ssim,
        s_lpips=s_lpips,
        weight_psnr=weights.psnr,
        weight_ssim=weights.ssim,
        weight_lpips=weights.lpips,
        final=final,
    )
