"""Faithful upscaling metrics backend for the scoring worker (CPU, deterministic).

Measures a candidate's frames against the sealed original (vidaio/scoring/upscaling_faithful.py)
and the validator's free answer — the served input Lanczos-upscaled to the original's geometry —
with the same metrics (the floor a miner must beat).

All inputs are the worker's private canonical y4m copies, so an auditor recomputes exactly
this from the archived artifacts.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Hashable, Protocol

from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.removal import LpipsVgg, Y4MError, Y4MReader
from vidaio.scoring.upscaling_faithful import UpscalingFaithfulCancelled, upscaled_frame_metrics
from vidaio.scoring.upscaling_faithful_formula import UpscaledFrameMetrics

__all__ = [
    "CpuUpscalingFaithfulBackend",
    "UpscalingFaithfulCancelled",
    "UpscalingFaithfulInputError",
    "UpscalingFaithfulMeasurement",
    "UpscalingFaithfulMeasurementError",
    "UpscalingFaithfulMetricsBackend",
]


@dataclass(frozen=True)
class UpscalingFaithfulMeasurement:
    metrics: UpscaledFrameMetrics
    floor: UpscaledFrameMetrics
    #: phase of the every-k-th-frame LPIPS sampling (derived from the held-out reference digest)
    lpips_offset: int = 0


class UpscalingFaithfulMetricsBackend(Protocol):
    def measure(
        self,
        *,
        canonical_reference: str,
        canonical_candidate: str,
        canonical_upscaled_input: str,
        config: ScoringConfig,
        lpips_offset: int = 0,
        cancelled: Callable[[], bool] | None = None,
        floor_key: Hashable | None = None,
    ) -> UpscalingFaithfulMeasurement: ...

    def version(self) -> str: ...


class UpscalingFaithfulMeasurementError(RuntimeError):
    """CANDIDATE-side failure: the miner's output cannot be measured (unreadable canonical
    candidate, geometry or frame count unlike the original). Scored as a violation."""


class UpscalingFaithfulInputError(RuntimeError):
    """VALIDATOR-side failure: the original or the upscaled served input cannot be read or do
    not match. Never the miner's fault — the worker refuses the request (422) so the item is
    skipped instead of zeroed."""


class CpuUpscalingFaithfulBackend:
    """The shipped backend: numpy/OpenCV PSNR and SSIM + LPIPS-VGG on CPU."""

    VERSION = "upscaling-faithful-metrics/1"  # keep equal to runtime_identity.UPSCALING_FAITHFUL_METRICS_VERSION

    #: floors kept per item (a competition scores every contender on the same items)
    FLOOR_CACHE_ITEMS = 128

    def __init__(
        self,
        *,
        lpips_enabled: bool = True,
        lpips_loader: Callable[[], LpipsVgg | None] | None = None,
    ) -> None:
        """``lpips_loader`` shares an already-loaded LPIPS-VGG network instead of loading a
        second copy of the same pinned weights."""
        self._lpips_enabled = lpips_enabled
        self._lpips_loader = lpips_loader
        self._lpips: LpipsVgg | None = None
        self._floors: OrderedDict[Hashable, UpscaledFrameMetrics] = OrderedDict()
        self._floors_lock = threading.Lock()

    def version(self) -> str:
        return self.VERSION

    def preload(self) -> None:
        if self._lpips_enabled and self._lpips is None:
            self._lpips = self._lpips_loader() if self._lpips_loader is not None else LpipsVgg()

    def measure(
        self,
        *,
        canonical_reference: str,
        canonical_candidate: str,
        canonical_upscaled_input: str,
        config: ScoringConfig,
        lpips_offset: int = 0,
        cancelled: Callable[[], bool] | None = None,
        floor_key: Hashable | None = None,
    ) -> UpscalingFaithfulMeasurement:
        """``floor_key`` names the verified original and served input (their digests); with it
        the Lanczos floor of an item is measured once and reused for every candidate."""
        try:
            ref = Y4MReader.open(canonical_reference)
            upscaled = Y4MReader.open(canonical_upscaled_input)
        except (Y4MError, OSError) as exc:
            raise UpscalingFaithfulInputError(str(exc)) from exc
        if (upscaled.width, upscaled.height) != (ref.width, ref.height):
            raise UpscalingFaithfulInputError(
                f"upscaled served input {upscaled.width}x{upscaled.height} != original {ref.width}x{ref.height}"
            )
        if ref.frame_count < 1 or upscaled.frame_count != ref.frame_count:
            raise UpscalingFaithfulInputError(
                f"original has {ref.frame_count} frames and the served input {upscaled.frame_count}"
            )
        try:
            cand = Y4MReader.open(canonical_candidate)
        except (Y4MError, OSError) as exc:
            raise UpscalingFaithfulMeasurementError(f"candidate unreadable: {exc}") from exc
        if (cand.width, cand.height) != (ref.width, ref.height):
            raise UpscalingFaithfulMeasurementError(
                f"candidate geometry {cand.width}x{cand.height} != original {ref.width}x{ref.height}"
            )
        if cand.frame_count != ref.frame_count:
            raise UpscalingFaithfulMeasurementError(
                f"candidate has {cand.frame_count} frames, expected {ref.frame_count}"
            )
        self.preload()
        stride = max(1, int(config.upscaling_faithful_lpips_stride))
        offset = int(lpips_offset) % stride
        max_side = max(0, int(config.upscaling_faithful_lpips_max_side))
        cache_key = (
            None if floor_key is None else (floor_key, stride, offset, max_side, self._lpips is not None)
        )
        floor = None
        if cache_key is not None:
            with self._floors_lock:
                floor = self._floors.get(cache_key)
                if floor is not None:
                    self._floors.move_to_end(cache_key)
        try:
            metrics, floor_measured = upscaled_frame_metrics(
                cand,
                ref,
                upscaled,
                lpips=self._lpips,
                lpips_stride=stride,
                lpips_max_side=max_side,
                lpips_offset=offset,
                cancelled=cancelled,
                floor=floor,
            )
        except Y4MError as exc:
            raise UpscalingFaithfulMeasurementError(str(exc)) from exc
        if cache_key is not None and floor is None:
            with self._floors_lock:
                self._floors[cache_key] = floor_measured
                self._floors.move_to_end(cache_key)
                while len(self._floors) > self.FLOOR_CACHE_ITEMS:
                    self._floors.popitem(last=False)
        for name, value in (("lpips_vgg", metrics.lpips_vgg), ("floor.lpips_vgg", floor_measured.lpips_vgg)):
            if self._lpips is not None and not math.isfinite(value):
                raise UpscalingFaithfulInputError(f"{name} is not finite (no LPIPS sample)")
        return UpscalingFaithfulMeasurement(metrics=metrics, floor=floor_measured, lpips_offset=offset)
