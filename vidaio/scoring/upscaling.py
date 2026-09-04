"""Upscaling-track score composition — exact spec formula (the design spec §02).

Spec::

    s_q   = quality(pieapp)
    s_l   = log(1 + content_length) / log(321)      (clamped to 1)
    s_pre = 0.5*s_q + 0.5*s_l
    s_f   = 0.1 * exp(6.979 * (s_pre - 0.5))        (clamped to [0, 1])

VMAF is only a pass/fail *gate* on this track (vmaf/100 < 0.5 -> 0, enforced by
``VmafFloorGate``); the numeric score is PieAPP + length.

Documented PieAPP -> [0, 1] mapping (spec ambiguity — the spec names ``quality(pieapp)``
without a formula): PieAPP is a perceptual *distance*, lower = closer to the reference,
and can go slightly negative for outputs the model prefers to the reference. This
implementation uses::

    s_q = 1 / (1 + max(0, pieapp))

Properties: monotone **decreasing** in the PieAPP distance; a perfect (or
better-than-reference, clamped) output maps to exactly 1.0; distance 1 -> 0.5;
asymptotically 0 for garbage. No tunable shape constants -> nothing hidden to tune per
miner, and the mapping is trivially recomputable by an auditor.

Reference anchor points: s_pre = 0.5 -> final = 0.1 exactly; s_pre = 1 -> 0.1*e^3.4895
~ 3.27, clamped to 1.0.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, field_validator

from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.finite import require_finite


class UpscalingBreakdown(BaseModel):
    """Every term of the upscaling formula — the audit-recompute record."""

    model_config = {"frozen": True}

    kind: Literal["upscaling"] = "upscaling"
    pieapp: float
    s_q: float
    content_length: float
    s_l: float
    s_pre: float
    coefficient: float
    exponent: float
    length_log_base: float
    #: Bounded by construction (finite, [0, 1]) — an Infinity/NaN final must be
    #: unconstructible and unparseable, matching ItemScore.score.
    final: float

    @field_validator("final")
    @classmethod
    def _final_bounded(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"final must be finite, got {value!r}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"final must be in [0, 1], got {value!r}")
        return value


def quality_from_pieapp(pieapp: float) -> float:
    """s_q = 1 / (1 + max(0, pieapp)) — see module docstring for the rationale.

    Fail closed: a non-finite PieAPP raises (NaN would otherwise clamp to a perfect
    1.0 via ``max(0, nan)``)."""
    require_finite("pieapp", pieapp)
    return 1.0 / (1.0 + max(0.0, pieapp))


def length_score(content_length: float, config: ScoringConfig) -> float:
    """s_l = log(1 + content_length) / log(base), clamped to [0, 1]. Base 321 ->
    saturates at content_length = 320. Non-finite content_length raises."""
    require_finite("content_length", content_length)
    if content_length < 0:
        raise ValueError("content_length must be >= 0")
    raw = math.log1p(content_length) / math.log(config.upscale_length_log_base)
    return min(1.0, raw)


def final_from_pre(s_pre: float, config: ScoringConfig) -> float:
    """s_f = coefficient * exp(exponent * (s_pre - 0.5)), clamped to [0, 1]."""
    require_finite("s_pre", s_pre)
    raw = config.upscale_coefficient * math.exp(config.upscale_exponent * (s_pre - 0.5))
    return min(1.0, max(0.0, raw))


def score_upscaling(
    *, pieapp: float, content_length: float, config: ScoringConfig
) -> UpscalingBreakdown:
    """Compose the upscaling-track item score. Pure and deterministic."""
    s_q = quality_from_pieapp(pieapp)
    s_l = length_score(content_length, config)
    s_pre = 0.5 * s_q + 0.5 * s_l
    final = final_from_pre(s_pre, config)
    return UpscalingBreakdown(
        pieapp=pieapp,
        s_q=s_q,
        content_length=content_length,
        s_l=s_l,
        s_pre=s_pre,
        coefficient=config.upscale_coefficient,
        exponent=config.upscale_exponent,
        length_log_base=config.upscale_length_log_base,
        final=final,
    )
