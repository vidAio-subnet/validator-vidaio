"""Frame-interpolation metrics backend for the scoring worker (CPU, deterministic).

Measures a candidate's inserted frames against the sealed reference
(vidaio/scoring/interpolation.py), measures the validator's two free answers (hold and
blend) with the same metrics (the floor a miner must beat), and checks that the input
frames came back unchanged at the even output positions.

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
from vidaio.scoring.interpolation import (
    InterpolationCancelled,
    inserted_frame_metrics,
    kept_frame_change,
)
from vidaio.scoring.interpolation_formula import InsertedFrameMetrics, best_baseline
from vidaio.scoring.removal import LpipsVgg, Y4MError, Y4MReader

__all__ = [
    "CpuInterpolationBackend",
    "InterpolationCancelled",
    "InterpolationInputError",
    "InterpolationMeasurement",
    "InterpolationMeasurementError",
    "InterpolationMetricsBackend",
]


@dataclass(frozen=True)
class InterpolationMeasurement:
    metrics: InsertedFrameMetrics
    floor: InsertedFrameMetrics
    hold: InsertedFrameMetrics
    blend: InsertedFrameMetrics
    kept_change: float
    input_frames: int
    #: phase of the every-k-th-inserted-frame LPIPS sampling (derived from the held-out
    #: reference digest by the caller; published so the packet states what was sampled)
    lpips_offset: int = 0


class InterpolationMetricsBackend(Protocol):
    def measure(
        self,
        *,
        canonical_reference: str,
        canonical_candidate: str,
        canonical_input: str,
        config: ScoringConfig,
        lpips_offset: int = 0,
        cancelled: Callable[[], bool] | None = None,
        floor_key: Hashable | None = None,
    ) -> InterpolationMeasurement: ...

    def version(self) -> str: ...


class InterpolationMeasurementError(RuntimeError):
    """CANDIDATE-side failure: the miner's output cannot be measured (unreadable canonical
    candidate, geometry or frame count unlike the reference). Scored as a violation."""


class InterpolationInputError(RuntimeError):
    """VALIDATOR-side failure: the reference or the served input cannot be read, or they do
    not form a 2x frame grid. Never the miner's fault — the worker must refuse the request
    (422) so the authority skips the item instead of zeroing it."""


class CpuInterpolationBackend:
    """The shipped backend: numpy/OpenCV PSNR and SSIM + LPIPS-VGG on CPU."""

    VERSION = "interpolation-metrics/1"  # keep equal to runtime_identity.INTERPOLATION_METRICS_VERSION

    #: hold/blend measurements kept per item (a competition scores every contender on
    #: the same few items).
    FLOOR_CACHE_ITEMS = 64

    def __init__(
        self,
        *,
        lpips_enabled: bool = True,
        lpips_loader: Callable[[], LpipsVgg | None] | None = None,
    ) -> None:
        """``lpips_loader`` shares an already-loaded LPIPS-VGG network (the removal
        backend's) instead of loading a second copy of the same pinned weights."""
        self._lpips_enabled = lpips_enabled
        self._lpips_loader = lpips_loader
        self._lpips: LpipsVgg | None = None
        self._floors: OrderedDict[
            Hashable, tuple[InsertedFrameMetrics, InsertedFrameMetrics]
        ] = OrderedDict()
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
        canonical_input: str,
        config: ScoringConfig,
        lpips_offset: int = 0,
        cancelled: Callable[[], bool] | None = None,
        floor_key: Hashable | None = None,
    ) -> InterpolationMeasurement:
        """``floor_key`` names the verified reference and served input (their digests).
        With it, the hold/blend floor of an item is measured once and reused for every
        candidate of that item: it depends on no candidate, and the same deterministic
        code measures it either way."""
        # validator-side inputs first: if these fail nothing about the miner is known
        try:
            ref = Y4MReader.open(canonical_reference)
            served = Y4MReader.open(canonical_input)
        except (Y4MError, OSError) as exc:
            raise InterpolationInputError(str(exc)) from exc
        if (served.width, served.height) != (ref.width, ref.height):
            raise InterpolationInputError(
                f"served input geometry {served.width}x{served.height} != reference {ref.width}x{ref.height}"
            )
        if served.frame_count < 2 or ref.frame_count != 2 * served.frame_count - 1:
            raise InterpolationInputError(
                f"reference has {ref.frame_count} frames and the served input "
                f"{served.frame_count}: not a 2x frame grid"
            )
        # the miner's output
        try:
            cand = Y4MReader.open(canonical_candidate)
        except (Y4MError, OSError) as exc:
            raise InterpolationMeasurementError(f"candidate unreadable: {exc}") from exc
        if (cand.width, cand.height) != (ref.width, ref.height):
            raise InterpolationMeasurementError(
                f"candidate geometry {cand.width}x{cand.height} != reference {ref.width}x{ref.height}"
            )
        if cand.frame_count != ref.frame_count:
            raise InterpolationMeasurementError(
                f"candidate has {cand.frame_count} frames, expected {ref.frame_count}"
            )
        self.preload()
        stride = max(1, int(config.interpolation_lpips_stride))
        offset = int(lpips_offset) % stride
        max_side = max(0, int(config.interpolation_lpips_max_side))
        cache_key = (
            None
            if floor_key is None
            else (floor_key, stride, offset, max_side, self._lpips is not None)
        )
        baselines = None
        if cache_key is not None:
            with self._floors_lock:
                baselines = self._floors.get(cache_key)
                if baselines is not None:
                    self._floors.move_to_end(cache_key)
        try:
            metrics, hold, blend = inserted_frame_metrics(
                cand,
                ref,
                served,
                lpips=self._lpips,
                lpips_stride=stride,
                lpips_max_side=max_side,
                lpips_offset=offset,
                cancelled=cancelled,
                baselines=baselines,
            )
            kept = kept_frame_change(cand, served, cancelled=cancelled)
        except Y4MError as exc:
            raise InterpolationMeasurementError(str(exc)) from exc
        if cache_key is not None and baselines is None:
            with self._floors_lock:
                self._floors[cache_key] = (hold, blend)
                self._floors.move_to_end(cache_key)
                while len(self._floors) > self.FLOOR_CACHE_ITEMS:
                    self._floors.popitem(last=False)
        floor = best_baseline(hold, blend)
        for name, value in (("lpips_vgg", metrics.lpips_vgg), ("floor.lpips_vgg", floor.lpips_vgg)):
            if self._lpips is not None and not math.isfinite(value):
                raise InterpolationInputError(f"{name} is not finite (no LPIPS sample)")
        return InterpolationMeasurement(
            metrics=metrics,
            floor=floor,
            hold=hold,
            blend=blend,
            kept_change=kept,
            input_frames=served.frame_count,
            lpips_offset=offset,
        )
