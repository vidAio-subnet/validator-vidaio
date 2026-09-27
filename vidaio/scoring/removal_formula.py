"""The object-removal score formula (dependency-light: no NumPy, no OpenCV).

The media measurements live in :mod:`vidaio.scoring.removal`; this module only turns
measured region metrics and the free-fill floor into the score, so packet models and
auditors can import it without the media stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, field_validator

from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.finite import require_finite


@dataclass(frozen=True)
class RegionMetrics:
    psnr_db: float
    ssim: float
    lpips_vgg: float
    warp_error: float
    masked_pixels: int


class RemovalBreakdown(BaseModel):
    """Every term of the removal formula — the audit-recompute record."""

    model_config = {"frozen": True}

    kind: Literal["removal"] = "removal"
    psnr_db: float
    ssim: float
    lpips_vgg: float
    warp_error: float
    warp_error_reference: float
    masked_pixels: int
    #: the free-fill floor measured by the validator with the same metrics
    floor_psnr_db: float
    floor_lpips_vgg: float
    #: the two zero-rule limits the score was decided against (audit hysteresis reads
    #: measured-vs-limit from here; see vidaio/audit/recompute.py::_formula_boundary)
    psnr_margin_db: float
    warp_cap: float
    s_psnr: float
    s_lpips: float
    weight_psnr: float
    weight_lpips: float
    final: float
    zero_reason: str | None = None

    @field_validator("final")
    @classmethod
    def _final_bounded(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"final must be finite, got {value!r}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"final must be in [0, 1], got {value!r}")
        return value


def score_removal(
    *,
    metrics: RegionMetrics,
    floor: RegionMetrics,
    warp_error_reference: float,
    config: ScoringConfig,
) -> RemovalBreakdown:
    """Compose the removal score from the region metrics and the free-fill floor."""
    for name, value in (
        ("psnr_db", metrics.psnr_db),
        ("ssim", metrics.ssim),
        ("lpips_vgg", metrics.lpips_vgg),
        ("warp_error", metrics.warp_error),
        ("floor.psnr_db", floor.psnr_db),
        ("floor.lpips_vgg", floor.lpips_vgg),
        ("warp_error_reference", warp_error_reference),
    ):
        require_finite(name, value)
    w_psnr, w_lpips = config.removal_weights.psnr, config.removal_weights.lpips
    zero_reason: str | None = None
    warp_cap = config.removal_warp_cap_factor * max(warp_error_reference, config.removal_warp_floor)
    if metrics.psnr_db < floor.psnr_db + config.removal_psnr_margin_db:
        zero_reason = "REGION_BASELINE_NOT_BEATEN"
    elif metrics.warp_error > warp_cap:
        zero_reason = "WARP_ERROR_EXCEEDED"
    s_psnr = min(1.0, max(0.0, (metrics.psnr_db - floor.psnr_db) / config.removal_psnr_span_db))
    s_lpips = (
        min(1.0, max(0.0, (floor.lpips_vgg - metrics.lpips_vgg) / floor.lpips_vgg))
        if floor.lpips_vgg > 0
        else 0.0
    )
    final = 0.0 if zero_reason else min(1.0, max(0.0, w_psnr * s_psnr + w_lpips * s_lpips))
    return RemovalBreakdown(
        psnr_db=metrics.psnr_db,
        ssim=metrics.ssim,
        lpips_vgg=metrics.lpips_vgg,
        warp_error=metrics.warp_error,
        warp_error_reference=warp_error_reference,
        masked_pixels=metrics.masked_pixels,
        floor_psnr_db=floor.psnr_db,
        floor_lpips_vgg=floor.lpips_vgg,
        psnr_margin_db=float(config.removal_psnr_margin_db),
        warp_cap=float(warp_cap),
        s_psnr=s_psnr,
        s_lpips=s_lpips,
        weight_psnr=w_psnr,
        weight_lpips=w_lpips,
        final=final,
        zero_reason=zero_reason,
    )
