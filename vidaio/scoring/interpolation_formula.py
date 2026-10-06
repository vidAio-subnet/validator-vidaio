"""The frame-interpolation score formula (dependency-light: no NumPy, no OpenCV).

The media measurements live in :mod:`vidaio.scoring.interpolation`; this module only turns
the measured metrics of the inserted frames and the validator's hold/blend floor into the
score, so packet models and auditors can import it without the media stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, field_validator

from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.finite import require_finite


#: Output frames per input frame interval: N input frames -> (N - 1) * factor + 1 output
#: frames. The track scores 2x only (one new frame between each pair of input frames).
INTERPOLATION_FRAME_FACTOR = 2


@dataclass(frozen=True)
class InsertedFrameMetrics:
    """Quality of the inserted (odd-position) frames against the real frames."""

    psnr_db: float
    ssim: float
    lpips_vgg: float
    inserted_frames: int


def best_baseline(*baselines: InsertedFrameMetrics) -> InsertedFrameMetrics:
    """The floor an interpolation must beat: the best of the given baselines on every term.

    The validator measures two answers anyone gets for free: repeating the previous frame
    (hold) and averaging the two neighbours (blend). A candidate is credited only for what it
    adds over the better of them per term (highest PSNR/SSIM, lowest LPIPS)."""
    if not baselines:
        raise ValueError("best_baseline needs at least one baseline")
    return InsertedFrameMetrics(
        psnr_db=max(b.psnr_db for b in baselines),
        ssim=max(b.ssim for b in baselines),
        lpips_vgg=min(b.lpips_vgg for b in baselines),
        inserted_frames=baselines[0].inserted_frames,
    )


class InterpolationBreakdown(BaseModel):
    """Every term of the frame-interpolation formula — the audit-recompute record."""

    model_config = {"frozen": True}

    kind: Literal["interpolation"] = "interpolation"
    psnr_db: float
    ssim: float
    lpips_vgg: float
    inserted_frames: int
    #: the hold/blend floor measured by the validator with the same metrics
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


def score_interpolation(
    *,
    metrics: InsertedFrameMetrics,
    floor: InsertedFrameMetrics,
    config: ScoringConfig,
) -> InterpolationBreakdown:
    """Compose the interpolation score from the inserted-frame metrics and the floor.

    Each term credits only the improvement over the floor, as a share of what is left to
    gain: PSNR over a fixed span, SSIM and LPIPS over the distance to a perfect frame.
    Returning the hold or the blend scores exactly zero."""
    for name, value in (
        ("psnr_db", metrics.psnr_db),
        ("ssim", metrics.ssim),
        ("lpips_vgg", metrics.lpips_vgg),
        ("floor.psnr_db", floor.psnr_db),
        ("floor.ssim", floor.ssim),
        ("floor.lpips_vgg", floor.lpips_vgg),
    ):
        require_finite(name, value)
    weights = config.interpolation_weights
    s_psnr = _clamp01((metrics.psnr_db - floor.psnr_db) / config.interpolation_psnr_span_db)
    ssim_headroom = 1.0 - floor.ssim
    s_ssim = _clamp01((metrics.ssim - floor.ssim) / ssim_headroom) if ssim_headroom > 0 else 0.0
    s_lpips = (
        _clamp01((floor.lpips_vgg - metrics.lpips_vgg) / floor.lpips_vgg)
        if floor.lpips_vgg > 0
        else 0.0
    )
    final = _clamp01(weights.psnr * s_psnr + weights.ssim * s_ssim + weights.lpips * s_lpips)
    return InterpolationBreakdown(
        psnr_db=metrics.psnr_db,
        ssim=metrics.ssim,
        lpips_vgg=metrics.lpips_vgg,
        inserted_frames=metrics.inserted_frames,
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
