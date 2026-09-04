"""Deterministic CPU perceptual-manipulation checks.

The real backend decodes a fixed, evenly-spaced frame/pixel sample and reduces it
to these statistics using integer arithmetic.  This module owns the calibrated
decision surface so scorers and CPU-only auditors use exactly the same thresholds.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidaio.scoring.backends import PerceptualCheckResult

CPU_PERCEPTUAL_ALGORITHM_VERSION = 1


class CpuPerceptualConfig(BaseModel):
    """Pinned sampling and conservative manipulation thresholds.

    Distances are normalized to ``[0, 1]``.  The defaults intentionally leave
    normal codec/upscaler drift ample headroom while rejecting global edits that
    materially alter brightness/contrast, remove colour, or replace chroma.
    """

    model_config = ConfigDict(frozen=True)

    sample_frames: int = Field(8, ge=1, le=64)
    sample_edge: int = Field(128, ge=16, le=512)
    tone_mean_delta_max: float = Field(0.06, gt=0.0, le=1.0)
    tone_std_delta_max: float = Field(0.08, gt=0.0, le=1.0)
    grayscale_reference_chroma_min: float = Field(0.03, ge=0.0, le=1.0)
    grayscale_chroma_ratio_min: float = Field(0.35, gt=0.0, le=1.0)
    chroma_mae_max: float = Field(0.10, gt=0.0, le=1.0)
    chroma_energy_ratio_min: float = Field(0.40, gt=0.0)
    chroma_energy_ratio_max: float = Field(1.80, gt=0.0)

    @model_validator(mode="after")
    def _ratio_bounds(self) -> CpuPerceptualConfig:
        if self.chroma_energy_ratio_max <= self.chroma_energy_ratio_min:
            raise ValueError(
                "chroma_energy_ratio_max must exceed chroma_energy_ratio_min"
            )
        return self

    def digest(self) -> str:
        """Stable identity for the exact CPU decision configuration."""
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


class PerceptualStatistics(BaseModel):
    """Normalized sufficient statistics from one reference/candidate sample."""

    model_config = ConfigDict(frozen=True)

    sampled_pixels: int = Field(gt=0)
    reference_luma_mean: float = Field(ge=0.0, le=1.0)
    candidate_luma_mean: float = Field(ge=0.0, le=1.0)
    reference_luma_std: float = Field(ge=0.0, le=1.0)
    candidate_luma_std: float = Field(ge=0.0, le=1.0)
    reference_chroma_energy: float = Field(ge=0.0, le=1.0)
    candidate_chroma_energy: float = Field(ge=0.0, le=1.0)
    chroma_mae: float = Field(ge=0.0, le=1.0)


def tone_manipulation_result(
    stats: PerceptualStatistics, config: CpuPerceptualConfig
) -> PerceptualCheckResult:
    mean_delta = abs(stats.candidate_luma_mean - stats.reference_luma_mean)
    std_delta = abs(stats.candidate_luma_std - stats.reference_luma_std)
    normalized = max(
        mean_delta / config.tone_mean_delta_max,
        std_delta / config.tone_std_delta_max,
    )
    return PerceptualCheckResult(
        passed=normalized <= 1.0,
        measure=normalized,
        limit=1.0,
        comparison="maximum",
        detail=(
            "CPU tone check: normalized deviation "
            f"{normalized:.6f}; mean delta {mean_delta:.6f}/"
            f"{config.tone_mean_delta_max:.6f}, luma-std delta "
            f"{std_delta:.6f}/{config.tone_std_delta_max:.6f}"
        ),
    )


def grayscale_result(
    stats: PerceptualStatistics, config: CpuPerceptualConfig
) -> PerceptualCheckResult:
    reference = stats.reference_chroma_energy
    if reference < config.grayscale_reference_chroma_min:
        return PerceptualCheckResult(
            passed=True,
            measure=1.0,
            detail=(
                "CPU grayscale check: reference is naturally low-chroma "
                f"({reference:.6f} < {config.grayscale_reference_chroma_min:.6f})"
            ),
        )
    ratio = stats.candidate_chroma_energy / reference
    return PerceptualCheckResult(
        passed=ratio >= config.grayscale_chroma_ratio_min,
        measure=ratio,
        limit=config.grayscale_chroma_ratio_min,
        comparison="minimum",
        detail=(
            "CPU grayscale check: candidate/reference chroma-energy ratio "
            f"{ratio:.6f}, minimum {config.grayscale_chroma_ratio_min:.6f}"
        ),
    )


def chroma_uv_result(
    stats: PerceptualStatistics, config: CpuPerceptualConfig
) -> PerceptualCheckResult:
    mae_factor = stats.chroma_mae / config.chroma_mae_max
    reference = stats.reference_chroma_energy
    ratio = (
        stats.candidate_chroma_energy / reference
        if reference >= config.grayscale_reference_chroma_min
        else 1.0
    )
    if ratio < config.chroma_energy_ratio_min:
        ratio_factor = config.chroma_energy_ratio_min / max(ratio, 1e-12)
    elif ratio > config.chroma_energy_ratio_max:
        ratio_factor = ratio / config.chroma_energy_ratio_max
    else:
        ratio_factor = 1.0
    normalized = max(mae_factor, ratio_factor)
    return PerceptualCheckResult(
        passed=normalized <= 1.0,
        measure=normalized,
        limit=1.0,
        comparison="maximum",
        detail=(
            "CPU chroma check: normalized deviation "
            f"{normalized:.6f}; chroma MAE {stats.chroma_mae:.6f}/"
            f"{config.chroma_mae_max:.6f}, energy ratio {ratio:.6f} in "
            f"[{config.chroma_energy_ratio_min:.6f}, "
            f"{config.chroma_energy_ratio_max:.6f}]"
        ),
    )
