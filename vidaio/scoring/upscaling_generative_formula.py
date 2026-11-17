"""The generative video-upscaling score formula (dependency-light: no NumPy, no OpenCV).

The media measurements live in :mod:`vidaio.scoring.upscaling_generative`; this module turns
the measured metrics into the score, so packet models and auditors can import it without the
media stack.

"Generative, not faithful": a generative upscaler invents plausible detail the served input no
longer holds. Pixel metrics (PSNR/SSIM) punish exactly that, so the score is built from
perceptual and temporal measurements instead, and two gates (low-resolution consistency and
colour, :func:`vidaio.scoring.gates.upscaling_generative_violations`) keep the invented detail
anchored to the served input::

    s_dists    = band((1 - dists), anchors.dists_similarity)
    s_lpips    = band((1 - lpips), anchors.lpips_similarity)
    s_fr       = (s_dists + s_lpips) / 2
    temporal   = max(e_ref, warp_floor) / max(e_cand, e_ref, warp_floor)       (<= 1)
    s_temporal = band(temporal, anchors.temporal)
    nr_ratio   = min(nr / nr_reference, 1 + nr_cap)
    s_nr       = band(nr_ratio, anchors.nr_ratio)
    final      = w_fr * s_fr + w_temporal * s_temporal + w_nr * s_nr          (0.55 / 0.15 / 0.30)

where ``band(v, a) = clamp((v - a.floor) / (a.anchor - a.floor), 0, 1)``: the floor and the
anchor are the medians of the weakest and the strongest public baseline upscaler (Lanczos,
Real-ESRGAN, SeedVR2-3B) on that metric, per upscale factor
(``ScoringConfig.upscaling_generative_anchors``, calibrated 2026-11-16). Temporal consistency is
measured against the reference's own motion: a candidate as consistent as the original (or
smoother) gets the full ratio, never more, so smoothing below the reference earns nothing. The
no-reference quality is capped at the reference clip's own value + 5 %: "sharper than reality"
earns nothing either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, field_validator

from vidaio.scoring.config import GenerativeAnchor, ScoringConfig
from vidaio.scoring.finite import require_finite

#: Upscale factors the generative track scores (served input -> reference geometry).
UPSCALING_GENERATIVE_FACTORS = (2, 4)


@dataclass(frozen=True)
class GenerativeFrameMetrics:
    """Everything the generative score and its gates are computed from."""

    #: mean DISTS distance (candidate vs sealed original) on the sampled full frames
    dists: float
    #: mean LPIPS-VGG distance on the same frames
    lpips_vgg: float
    #: mean absolute warp error of the candidate under the reference's optical flow
    warp_error: float
    #: the reference's own warp error under the same flow
    warp_error_reference: float
    #: mean no-reference quality (CLIP-IQA) of the candidate on the sampled frames
    nr: float
    #: the reference's own no-reference quality on the same frames
    nr_reference: float
    #: low-resolution consistency: luma PSNR of the blurred, Lanczos-downscaled output
    #: against the blurred served input
    lowres_psnr_db: float
    #: mean CIEDE2000 against the sealed original on area-downscaled frames
    color_delta_e: float
    frames: int
    fr_frames: int


class UpscalingGenerativeBreakdown(BaseModel):
    """Every term of the generative upscaling formula — the audit-recompute record."""

    model_config = {"frozen": True}

    kind: Literal["upscaling_generative"] = "upscaling_generative"
    upscale_factor: int
    dists: float
    lpips_vgg: float
    warp_error: float
    warp_error_reference: float
    nr: float
    nr_reference: float
    lowres_psnr_db: float
    color_delta_e: float
    frames: int
    fr_frames: int
    dists_similarity: float
    lpips_similarity: float
    temporal: float
    nr_ratio: float
    s_dists: float
    s_lpips: float
    s_fr: float
    s_temporal: float
    s_nr: float
    weight_fr: float
    weight_temporal: float
    weight_nr: float
    #: the normalisation bands in force for this factor: (floor, anchor) per metric
    band_dists: tuple[float, float]
    band_lpips: tuple[float, float]
    band_temporal: tuple[float, float]
    band_nr: tuple[float, float]
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


def band(value: float, anchor: GenerativeAnchor) -> float:
    """``value`` placed in its normalisation band: 0 at the floor, 1 at the anchor."""
    return _clamp01((value - anchor.floor) / (anchor.anchor - anchor.floor))


def temporal_ratio(warp_error: float, warp_error_reference: float, warp_floor: float) -> float:
    """Reference warp error over the candidate's, capped at 1 (no reward below the reference)."""
    ref = max(warp_error_reference, warp_floor)
    return ref / max(warp_error, ref)


def nr_ratio(nr: float, nr_reference: float, cap: float) -> float:
    """Candidate no-reference quality relative to the reference's own, capped at 1 + cap."""
    if not nr_reference > 0.0:
        raise ValueError(f"reference no-reference quality must be > 0, got {nr_reference!r}")
    return min(max(nr, 0.0) / nr_reference, 1.0 + cap)


def lowres_psnr_threshold(config: ScoringConfig) -> float:
    """The low-resolution consistency gate threshold: max(minimum, calibrated floor)."""
    return max(
        config.upscaling_generative_lowres_psnr_min_db,
        config.upscaling_generative_lowres_psnr_floor_db,
    )


def score_upscaling_generative(
    *,
    metrics: GenerativeFrameMetrics,
    upscale_factor: int,
    config: ScoringConfig,
) -> UpscalingGenerativeBreakdown:
    """Compose the generative upscaling score from the measured metrics.

    The gates are separate (:func:`vidaio.scoring.gates.upscaling_generative_violations`):
    a gated item scores 0 whatever this breakdown says (gates-first zeroing in
    :func:`vidaio.scoring.result.compose_item_score`)."""
    for name, value in (
        ("dists", metrics.dists),
        ("lpips_vgg", metrics.lpips_vgg),
        ("warp_error", metrics.warp_error),
        ("warp_error_reference", metrics.warp_error_reference),
        ("nr", metrics.nr),
        ("nr_reference", metrics.nr_reference),
        ("lowres_psnr_db", metrics.lowres_psnr_db),
        ("color_delta_e", metrics.color_delta_e),
    ):
        require_finite(name, value)
    if upscale_factor not in UPSCALING_GENERATIVE_FACTORS:
        raise ValueError(f"upscale_factor {upscale_factor} not in {UPSCALING_GENERATIVE_FACTORS}")
    bands = config.upscaling_generative_anchors[upscale_factor]
    weights = config.upscaling_generative_weights
    dists_similarity = 1.0 - metrics.dists
    lpips_similarity = 1.0 - metrics.lpips_vgg
    s_dists = band(dists_similarity, bands.dists_similarity)
    s_lpips = band(lpips_similarity, bands.lpips_similarity)
    s_fr = (s_dists + s_lpips) / 2.0
    temporal = temporal_ratio(
        metrics.warp_error, metrics.warp_error_reference, config.upscaling_generative_warp_floor
    )
    s_temporal = band(temporal, bands.temporal)
    ratio = nr_ratio(metrics.nr, metrics.nr_reference, config.upscaling_generative_nr_cap)
    s_nr = band(ratio, bands.nr_ratio)
    final = _clamp01(weights.fr * s_fr + weights.temporal * s_temporal + weights.nr * s_nr)
    return UpscalingGenerativeBreakdown(
        upscale_factor=upscale_factor,
        dists=metrics.dists,
        lpips_vgg=metrics.lpips_vgg,
        warp_error=metrics.warp_error,
        warp_error_reference=metrics.warp_error_reference,
        nr=metrics.nr,
        nr_reference=metrics.nr_reference,
        lowres_psnr_db=metrics.lowres_psnr_db,
        color_delta_e=metrics.color_delta_e,
        frames=metrics.frames,
        fr_frames=metrics.fr_frames,
        dists_similarity=dists_similarity,
        lpips_similarity=lpips_similarity,
        temporal=temporal,
        nr_ratio=ratio,
        s_dists=s_dists,
        s_lpips=s_lpips,
        s_fr=s_fr,
        s_temporal=s_temporal,
        s_nr=s_nr,
        weight_fr=weights.fr,
        weight_temporal=weights.temporal,
        weight_nr=weights.nr,
        band_dists=(bands.dists_similarity.floor, bands.dists_similarity.anchor),
        band_lpips=(bands.lpips_similarity.floor, bands.lpips_similarity.anchor),
        band_temporal=(bands.temporal.floor, bands.temporal.anchor),
        band_nr=(bands.nr_ratio.floor, bands.nr_ratio.anchor),
        final=final,
    )
