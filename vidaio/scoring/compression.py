"""Compression-track score composition — exact spec formula (the design spec §02).

Spec case structure::

    CASE compression_rate >= 0.80        -> 0   (needs >= 1.25x shrink)
    CASE vmaf < threshold - 5            -> 0
    CASE vmaf >= threshold               -> final = min(1, (0.7*comp + 0.3*quality) / 1.12)

Documented reading of the band ``threshold-5 <= vmaf < threshold`` (spec ambiguity):
the spec section defines *no* formula for that band — only the three cases above — so
this implementation treats the case list as exhaustive-with-fallthrough-to-zero: any
``vmaf < threshold`` scores 0, i.e. there is NO linear ramp. The distinct
``threshold - 5`` case is preserved as reason-code granularity: below the band the
reason is ``VMAF_BELOW_FLOOR`` (hopelessly bad quality), inside the band it is
``VMAF_BELOW_THRESHOLD`` (near miss) — auditable, but both zero.

Term definitions (each documented, spec names in parentheses):

* ``compression_rate`` — candidate_bytes / reference_bytes; smaller is better.
* ``compression_score`` (comp) — ``1 - compression_rate`` clamped to [0, 1]. The byte
  ratio mapped so that a stronger shrink scores higher: rate 0 -> 1.0, rate 0.8 -> 0.2.
  This is the documented derivation of "compression_score derives from the byte ratio".
* ``quality_score`` — ``vmaf / 100`` (VMAF is a scored term on this track).
* ``final`` — ``min(1, (w_comp*comp + w_vmaf*quality) / 1.12)``. With the default
  weights the theoretical maximum is (0.7 + 0.3)/1.12 ~ 0.893; the ``min(1, .)`` is the
  spec's safety clamp for non-default weights.

Bitrate is never a score term — only an encoding constraint enforced by gates.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, field_validator

from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.finite import require_finite
from vidaio.scoring.gates import ReasonCode


class CompressionBreakdown(BaseModel):
    """Every term of the compression formula — the audit-recompute record."""

    model_config = {"frozen": True}

    kind: Literal["compression"] = "compression"
    candidate_bytes: int
    reference_bytes: int
    compression_rate: float
    compression_score: float
    vmaf: float
    quality_score: float
    vmaf_threshold: float
    weight_comp: float
    weight_vmaf: float
    normalizer: float
    #: Bounded by construction (finite, [0, 1]) — an Infinity/NaN final must be
    #: unconstructible and unparseable, matching ItemScore.score.
    final: float
    zero_reason: ReasonCode | None = None

    @field_validator("final")
    @classmethod
    def _final_bounded(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"final must be finite, got {value!r}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"final must be in [0, 1], got {value!r}")
        return value


def compression_rate(candidate_bytes: int, reference_bytes: int) -> float:
    """Byte ratio candidate/reference. Raises on a zero-byte reference (undefined)
    or a non-finite result (fail closed — NaN must never enter a comparison)."""
    if reference_bytes <= 0:
        raise ValueError("reference_bytes must be > 0")
    return require_finite("compression_rate", candidate_bytes / reference_bytes)


def compression_score_from_rate(rate: float) -> float:
    """comp term: 1 - rate, clamped to [0, 1] (smaller byte ratio -> higher score)."""
    require_finite("compression_rate", rate)
    return min(1.0, max(0.0, 1.0 - rate))


def score_compression(
    *,
    candidate_bytes: int,
    reference_bytes: int,
    vmaf: float,
    config: ScoringConfig,
    vmaf_threshold: float | None = None,
) -> CompressionBreakdown:
    """Compose the compression-track item score. Pure and deterministic.

    ``vmaf_threshold`` defaults to the configured track threshold; the competition
    manifest may inject a per-item override.

    Fail closed (defense in depth — gates run first, but a NaN must never compose):
    non-finite ``vmaf`` or threshold raises ``ValueError``.
    """
    threshold = (
        vmaf_threshold
        if vmaf_threshold is not None
        else config.vmaf_threshold("compression")
    )
    require_finite("vmaf", vmaf)
    require_finite("vmaf_threshold", threshold)
    rate = compression_rate(candidate_bytes, reference_bytes)
    comp = compression_score_from_rate(rate)
    quality = vmaf / 100.0
    weights = config.compression_weights

    zero_reason: ReasonCode | None = None
    if rate >= config.compression_rate_max:
        zero_reason = ReasonCode.COMPRESSION_RATE_TOO_HIGH
    elif vmaf < threshold - config.compression_vmaf_band:
        zero_reason = ReasonCode.VMAF_BELOW_FLOOR
    elif vmaf < threshold:
        # The undefined spec band: zero, but with a distinct near-miss reason code.
        zero_reason = ReasonCode.VMAF_BELOW_THRESHOLD

    if zero_reason is not None:
        final = 0.0
    else:
        final = min(
            1.0, (weights.comp * comp + weights.vmaf * quality) / config.compression_norm
        )

    return CompressionBreakdown(
        candidate_bytes=candidate_bytes,
        reference_bytes=reference_bytes,
        compression_rate=rate,
        compression_score=comp,
        vmaf=vmaf,
        quality_score=quality,
        vmaf_threshold=threshold,
        weight_comp=weights.comp,
        weight_vmaf=weights.vmaf,
        normalizer=config.compression_norm,
        final=final,
        zero_reason=zero_reason,
    )
