"""Streaming compression — quality at a fixed bitrate under streaming delivery rules.

A competition manifest with ``"compression_scoring": "streaming"`` scores each compression
item as a streaming rendition instead of "smallest file over a VMAF bar":

* **Budget.** Every clip has a bitrate budget derived from its own geometry and frame rate
  by the anchored rule ``target_kbps = bits_per_pixel * width * height * fps / 1000``
  (``StreamingRules``), so the budget is public, the same for every contender and needs no
  per-item storage.
* **Delivery rules** (measured on the encoded video packets, container overhead excluded):
  the average bitrate over the clip may exceed the budget by at most
  ``streaming_rate_tolerance``; the bitrate over any ``peak_window_seconds`` of consecutive
  frames (decode order) may not exceed ``peak_ratio`` x the budget; the first frame is a
  keyframe and no presented frame is more than ``max_keyframe_interval_seconds`` after the
  last keyframe (every segment of that length starts on a keyframe). A broken rule zeroes
  the item.
* **Score.** ``final = clamp((VMAF - streaming_vmaf_floor) / (100 - streaming_vmaf_floor))``
  with VMAF measured exactly as for compression (same pinned model, same anti-gaming model
  delta and perceptual checks). There is no VMAF threshold: quality is the score.

This module is dependency-light (no media tools): the scoring worker measures the packets
(:mod:`vidaio.scoring_worker.streaming_backend`) and auditors recompute from the same data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.finite import require_finite
from vidaio.scoring.gates import ReasonCode, ValidityViolation

__all__ = [
    "StreamingBreakdown",
    "StreamingMeasurement",
    "StreamingPacket",
    "StreamingRules",
    "measure_streaming",
    "score_streaming",
    "streaming_target_kbps",
    "streaming_violations",
]


class StreamingRules(BaseModel):
    """The delivery rules a streaming compression competition anchors in its manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: bitrate budget per pixel per frame: target_kbps = bpp * width * height * fps / 1000
    bits_per_pixel: float = Field(gt=0, le=2, allow_inf_nan=False)
    #: longest allowed time from a keyframe to the next one (the segment length)
    max_keyframe_interval_seconds: float = Field(default=2.0, gt=0, le=10, allow_inf_nan=False)
    #: peak bitrate allowed over any window, as a multiple of the budget
    peak_ratio: float = Field(default=1.5, ge=1, le=4, allow_inf_nan=False)
    #: length of the sliding peak-bitrate window
    peak_window_seconds: float = Field(default=2.0, gt=0, le=10, allow_inf_nan=False)

    def as_params(self) -> dict[str, float | str]:
        """The scoring-request params (orchestrator and auditor build the same)."""
        return {
            "scoring": "streaming",
            "bits_per_pixel": self.bits_per_pixel,
            "max_keyframe_interval_s": self.max_keyframe_interval_seconds,
            "peak_ratio": self.peak_ratio,
            "peak_window_s": self.peak_window_seconds,
        }


@dataclass(frozen=True)
class StreamingPacket:
    """One encoded video packet of the candidate, in decode (file) order."""

    size_bytes: int
    pts_seconds: float | None
    keyframe: bool


@dataclass(frozen=True)
class StreamingMeasurement:
    avg_kbps: float
    peak_kbps: float
    #: longest keyframe-to-next-keyframe span, including the tail after the last one
    max_keyframe_interval_seconds: float
    first_frame_is_keyframe: bool
    keyframes: int
    packets: int


def streaming_target_kbps(*, width: int, height: int, fps: float, bits_per_pixel: float) -> float:
    """The clip's bitrate budget from its own geometry and frame rate."""
    if width <= 0 or height <= 0 or not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"cannot derive a budget for {width}x{height} at {fps} fps")
    return bits_per_pixel * width * height * fps / 1000.0


def measure_streaming(
    packets: Sequence[StreamingPacket], *, fps: float, frames: int, peak_window_seconds: float
) -> StreamingMeasurement:
    """Average and peak bitrate and keyframe spacing of the candidate's video packets.

    ``fps`` and ``frames`` are the reference's (the candidate must match them anyway):
    the clip lasts ``frames / fps`` seconds and a window of ``peak_window_seconds`` holds
    ``round(peak_window_seconds * fps)`` consecutive packets."""
    if not packets:
        raise ValueError("the candidate has no video packets")
    if frames < 1 or not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"invalid reference timing: {frames} frames at {fps} fps")
    if any(p.pts_seconds is None or not math.isfinite(p.pts_seconds) for p in packets):
        raise ValueError("a video packet has no presentation timestamp")
    sizes = [int(p.size_bytes) for p in packets]
    if any(s < 0 for s in sizes):
        raise ValueError("negative packet size")
    duration = frames / fps
    avg_kbps = 8.0 * sum(sizes) / duration / 1000.0
    window = max(1, round(peak_window_seconds * fps))
    if len(sizes) <= window:
        peak_kbps = 8.0 * sum(sizes) / (len(sizes) / fps) / 1000.0
    else:
        running = sum(sizes[:window])
        best = running
        for i in range(window, len(sizes)):
            running += sizes[i] - sizes[i - window]
            best = max(best, running)
        peak_kbps = 8.0 * best / (window / fps) / 1000.0
    pts = [float(p.pts_seconds) for p in packets]  # type: ignore[arg-type]
    start = min(pts)
    end = max(pts) + 1.0 / fps
    keys = sorted(float(p.pts_seconds) for p in packets if p.keyframe)  # type: ignore[arg-type]
    first_is_key = bool(keys) and abs(keys[0] - start) < 0.5 / fps
    if keys:
        spans = [b - a for a, b in zip(keys, keys[1:])] + [end - keys[-1], keys[0] - start]
        max_interval = max(spans)
    else:
        max_interval = end - start
    return StreamingMeasurement(
        avg_kbps=avg_kbps,
        peak_kbps=peak_kbps,
        max_keyframe_interval_seconds=max_interval,
        first_frame_is_keyframe=first_is_key,
        keyframes=len(keys),
        packets=len(packets),
    )


def streaming_violations(
    measurement: StreamingMeasurement,
    *,
    target_kbps: float,
    rules: StreamingRules,
    fps: float,
    config: ScoringConfig,
) -> list[ValidityViolation]:
    """The delivery rules the candidate breaks (each one zeroes the item)."""
    violations: list[ValidityViolation] = []
    limit = target_kbps * (1.0 + config.streaming_rate_tolerance)
    if measurement.avg_kbps > limit:
        violations.append(
            ValidityViolation(
                code=ReasonCode.STREAMING_BITRATE_EXCEEDED,
                detail=f"average {measurement.avg_kbps:.1f} kbps over the {target_kbps:.1f} kbps budget",
                measured=measurement.avg_kbps,
                limit=limit,
            )
        )
    peak_limit = target_kbps * rules.peak_ratio
    if measurement.peak_kbps > peak_limit:
        violations.append(
            ValidityViolation(
                code=ReasonCode.STREAMING_PEAK_EXCEEDED,
                detail=(
                    f"{measurement.peak_kbps:.1f} kbps over a {rules.peak_window_seconds:g} s window, "
                    f"limit {rules.peak_ratio:g}x the budget"
                ),
                measured=measurement.peak_kbps,
                limit=peak_limit,
            )
        )
    # half a frame of slack for timestamp rounding
    interval_limit = rules.max_keyframe_interval_seconds + 0.5 / fps
    if not measurement.first_frame_is_keyframe or measurement.max_keyframe_interval_seconds > interval_limit:
        violations.append(
            ValidityViolation(
                code=ReasonCode.STREAMING_KEYFRAME_INTERVAL,
                detail=(
                    "the first frame is not a keyframe"
                    if not measurement.first_frame_is_keyframe
                    else f"{measurement.max_keyframe_interval_seconds:.3f} s between keyframes"
                ),
                measured=measurement.max_keyframe_interval_seconds,
                limit=interval_limit,
            )
        )
    return violations


class StreamingBreakdown(BaseModel):
    """Every term of the streaming compression score — the audit-recompute record."""

    model_config = {"frozen": True}

    kind: Literal["streaming"] = "streaming"
    vmaf: float
    vmaf_floor: float
    target_kbps: float
    avg_kbps: float
    peak_kbps: float
    max_keyframe_interval_seconds: float
    final: float

    @field_validator("final")
    @classmethod
    def _final_bounded(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"final must be finite, got {value!r}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"final must be in [0, 1], got {value!r}")
        return value


def score_streaming(
    *, vmaf: float, measurement: StreamingMeasurement, target_kbps: float, config: ScoringConfig
) -> StreamingBreakdown:
    """Quality within the budget: VMAF mapped from the configured floor to 100 onto [0, 1].
    The delivery rules are gates (``streaming_violations``), not part of the formula."""
    require_finite("vmaf", vmaf)
    floor = config.streaming_vmaf_floor
    final = min(1.0, max(0.0, (vmaf - floor) / (100.0 - floor)))
    return StreamingBreakdown(
        vmaf=vmaf,
        vmaf_floor=floor,
        target_kbps=target_kbps,
        avg_kbps=measurement.avg_kbps,
        peak_kbps=measurement.peak_kbps,
        max_keyframe_interval_seconds=measurement.max_keyframe_interval_seconds,
        final=final,
    )
