"""Scoring configuration — every lever of the score composition, gates and aggregation.

Defaults mirror the authoritative spec (the design spec §02/§18):

* upscaling final  = 0.1 * exp(6.979 * (0.5*s_q + 0.5*s_l - 0.5)), s_l log base 321
* compression final = min(1, (0.7*comp + 0.3*vmaf/100) / 1.12), rate >= 0.80 -> 0
* VMAF gate for upscaling at vmaf/100 < 0.5; VMAF model-delta gate <= 3.0
* file-size caps: 8x for 2x upscale, 20x for 4x upscale
* worst-decile aggregation fraction 0.1

Load via ``vidaio.core.config.section(raw, "scoring", ScoringConfig)``; every field is
overridable from ``config/default.yaml`` or ``VIDAIO__SCORING__<KEY>`` env vars.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

TRACK_COMPRESSION = "compression"
TRACK_UPSCALING = "upscaling"

#: Tolerance for weight-sum checks — weights that must partition 1.0 may carry float
#: representation error (0.3 + 0.3 + 0.4 == 0.9999999999999999), nothing more.
_WEIGHT_SUM_TOLERANCE = 1e-9


def _require_finite_config(name: str, value: float) -> None:
    """Fail CLOSED at construction: a NaN/inf config lever must never reach a formula
    (NaN compares False against every threshold and would silently disable it)."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_unit_interval(name: str, value: float) -> None:
    _require_finite_config(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


class CompressionWeights(BaseModel):
    """Weights of the two scored terms of the compression formula (spec: 0.7 / 0.3).

    Each weight must be finite and in [0, 1]. The pair deliberately need NOT sum to 1:
    the spec's ``min(1, .)`` safety clamp exists precisely for non-default weights.
    """

    comp: float = 0.7
    vmaf: float = 0.3

    @model_validator(mode="after")
    def _finite(self) -> "CompressionWeights":
        _require_unit_interval("compression_weights.comp", self.comp)
        _require_unit_interval("compression_weights.vmaf", self.vmaf)
        return self


class AggregateWeights(BaseModel):
    """Competition aggregate final_score weights (spec §02, manifest-injectable).

    Each weight is finite in [0, 1] and the three must partition 1.0 (within float
    representation tolerance) — the aggregate is a convex combination, so a manifest
    that leaks or mints weight mass is a config error, not a scoring outcome.
    """

    quality: float = 0.6
    cost_efficiency: float = 0.25
    length_coverage: float = 0.15

    @model_validator(mode="after")
    def _finite_and_convex(self) -> "AggregateWeights":
        _require_unit_interval("aggregate_weights.quality", self.quality)
        _require_unit_interval("aggregate_weights.cost_efficiency", self.cost_efficiency)
        _require_unit_interval("aggregate_weights.length_coverage", self.length_coverage)
        total = self.quality + self.cost_efficiency + self.length_coverage
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"aggregate_weights must sum to 1.0, got {total!r}")
        return self


class ScoringConfig(BaseModel):
    # --- gates -------------------------------------------------------------
    #: Upscaling VMAF pass/fail gate on the vmaf/100 scale (< 0.5 -> score 0).
    vmaf_gate_upscaling: float = 0.5
    #: Compression rate at/above which the item scores 0 (needs >= 1.25x shrink).
    compression_rate_max: float = 0.80
    #: Max allowed |delta| between the two VMAF model runs (anti model-gaming gate).
    vmaf_model_delta_max: float = 3.0
    #: Fail-closed default: the model-delta gate REQUIRES a secondary VMAF run and
    #: treats its absence as a METRIC_MISSING violation. A track whose pipeline
    #: genuinely has no secondary model run must say so by setting this to False,
    #: which records an informational GateSkip on the context — never a silent pass.
    require_secondary_vmaf: bool = True
    #: Candidate byte-size caps relative to the miner *input* payload, per upscale factor.
    file_size_caps: dict[int, float] = Field(default_factory=lambda: {2: 8.0, 4: 20.0})
    #: Codecs a candidate stream may use (EncodingGate).
    codec_allowlist: tuple[str, ...] = ("h264", "hevc", "vp9", "av1")
    #: Per-track VMAF thresholds on the 0-100 scale. The competition manifest overrides
    #: the compression threshold per item; this is the standing default. The upscaling
    #: floor is derived from ``vmaf_gate_upscaling`` (see :meth:`vmaf_floor`).
    vmaf_thresholds: dict[str, float] = Field(
        default_factory=lambda: {TRACK_COMPRESSION: 90.0}
    )
    #: Width (in VMAF points) of the sub-threshold band the compression spec calls out
    #: (``vmaf < threshold - 5 -> 0``). See vidaio/scoring/compression.py for the
    #: documented reading of scores inside the band.
    compression_vmaf_band: float = 5.0

    # --- compression formula ----------------------------------------------
    compression_weights: CompressionWeights = Field(default_factory=CompressionWeights)
    #: Normalization constant of the compression formula (spec: divide by 1.12).
    compression_norm: float = 1.12

    # --- upscaling formula -------------------------------------------------
    #: Length score log base: s_l = log(1 + content_length) / log(321), clamped to 1.
    upscale_length_log_base: float = 321.0
    #: Exponent of the upscaling final: 0.1 * exp(6.979 * (s_pre - 0.5)).
    upscale_exponent: float = 6.979
    upscale_coefficient: float = 0.1
    #: PieAPP samples this many consecutive frames from the derived start frame.
    pieapp_sample_window: int = 4

    # --- aggregation ------------------------------------------------------
    #: Fraction of worst items averaged by worst-decile aggregation (spec §18).
    worst_decile_fraction: float = 0.1
    aggregate_weights: AggregateWeights = Field(default_factory=AggregateWeights)

    @model_validator(mode="after")
    def _sane(self) -> "ScoringConfig":
        # Finiteness FIRST, on every float lever (fail closed): NaN compares False
        # against any bound, so e.g. a NaN compression_norm previously sailed past
        # the `<= 0` check and composed to score 1.0. Nested weight models validate
        # themselves at construction.
        scalars = {
            "vmaf_gate_upscaling": self.vmaf_gate_upscaling,
            "compression_rate_max": self.compression_rate_max,
            "vmaf_model_delta_max": self.vmaf_model_delta_max,
            "compression_vmaf_band": self.compression_vmaf_band,
            "compression_norm": self.compression_norm,
            "upscale_length_log_base": self.upscale_length_log_base,
            "upscale_exponent": self.upscale_exponent,
            "upscale_coefficient": self.upscale_coefficient,
            "worst_decile_fraction": self.worst_decile_fraction,
        }
        for name, value in scalars.items():
            _require_finite_config(name, value)
        for factor, cap in self.file_size_caps.items():
            _require_finite_config(f"file_size_caps[{factor}]", cap)
            if cap <= 0.0:
                raise ValueError(f"file_size_caps[{factor}] must be > 0, got {cap!r}")
        for track, threshold in self.vmaf_thresholds.items():
            _require_finite_config(f"vmaf_thresholds[{track!r}]", threshold)
            if not 0.0 <= threshold <= 100.0:
                raise ValueError(
                    f"vmaf_thresholds[{track!r}] must be in [0, 100], got {threshold!r}"
                )
        # Range sanity on the finite values.
        if not 0.0 < self.vmaf_gate_upscaling <= 1.0:
            raise ValueError("vmaf_gate_upscaling must be in (0, 1]")
        if not 0.0 < self.compression_rate_max <= 1.0:
            raise ValueError("compression_rate_max must be in (0, 1]")
        if self.vmaf_model_delta_max < 0.0:
            raise ValueError("vmaf_model_delta_max must be >= 0")
        if self.compression_vmaf_band < 0.0:
            raise ValueError("compression_vmaf_band must be >= 0")
        if not 0.0 < self.worst_decile_fraction <= 1.0:
            raise ValueError("worst_decile_fraction must be in (0, 1]")
        if self.upscale_length_log_base <= 1.0:
            raise ValueError("upscale_length_log_base must be > 1")
        if self.upscale_exponent <= 0.0:
            raise ValueError("upscale_exponent must be > 0")
        if self.upscale_coefficient <= 0.0:
            raise ValueError("upscale_coefficient must be > 0")
        if self.compression_norm <= 0.0:
            raise ValueError("compression_norm must be > 0")
        if self.pieapp_sample_window < 1:
            raise ValueError("pieapp_sample_window must be >= 1")
        return self

    def vmaf_floor(self, track: str) -> float:
        """Hard VMAF floor (0-100 scale) below which the item scores 0 for `track`.

        * upscaling: the spec gate ``vmaf/100 < vmaf_gate_upscaling`` -> floor = gate*100.
        * compression: ``threshold - compression_vmaf_band`` (spec: threshold - 5).
        """
        if track == TRACK_UPSCALING:
            return self.vmaf_gate_upscaling * 100.0
        threshold = self.vmaf_threshold(track)
        return threshold - self.compression_vmaf_band

    def vmaf_threshold(self, track: str) -> float:
        """Per-track scoring threshold on the 0-100 scale."""
        if track == TRACK_UPSCALING:
            return self.vmaf_gate_upscaling * 100.0
        if track not in self.vmaf_thresholds:
            raise KeyError(f"no vmaf threshold configured for track {track!r}")
        return self.vmaf_thresholds[track]
