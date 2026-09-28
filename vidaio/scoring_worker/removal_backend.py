"""Object-removal metrics backend for the scoring worker (CPU, deterministic).

Measures a candidate's masked region against the sealed reference (vidaio/scoring/removal.py),
computes the validator's free temporal-median fill and scores it with the same metrics
(the floor a miner must beat), and checks that nothing outside the mask changed.

The mask is video stream 1 of the served input (CompositeOccluder in the challenge DAG);
``mask_stream_index`` in the task params names it. All inputs are the worker's private
canonical y4m copies, so an auditor recomputes exactly this from the archived artifacts.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Protocol

from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.removal import (
    LpipsVgg,
    RegionMetrics,
    RemovalCancelled,
    Y4MError,
    Y4MReader,
    decode_mask_stream,
    outside_region_change,
    region_metrics,
    temporal_median_fill,
)
from vidaio.scoring.removal_formula import best_baseline


@dataclass(frozen=True)
class RemovalMeasurement:
    metrics: RegionMetrics
    floor: RegionMetrics
    warp_error_reference: float
    outside_change: float
    mask_frames: int
    mask_fraction: float
    #: phase of the every-k-th-masked-frame LPIPS sampling (derived from the held-out
    #: reference digest by the caller; published so the packet states what was sampled)
    lpips_offset: int = 0


class RemovalMetricsBackend(Protocol):
    def measure(
        self,
        *,
        canonical_reference: str,
        canonical_candidate: str,
        canonical_input: str,
        served_input_path: str,
        mask_stream_index: int,
        config: ScoringConfig,
        lpips_offset: int = 0,
        cancelled: Callable[[], bool] | None = None,
    ) -> RemovalMeasurement: ...

    def version(self) -> str: ...


class RemovalMeasurementError(RuntimeError):
    """CANDIDATE-side failure: the miner's output cannot be measured (unreadable canonical
    candidate, geometry unlike the reference). Scored as a violation — the miner's fault."""


class RemovalInputError(RuntimeError):
    """VALIDATOR-side failure: the reference, the served input or its mask stream cannot be
    read, or the mask is empty on every frame. Never the miner's fault — the worker must
    refuse the request (422) so the authority skips the item instead of zeroing it."""


class CpuRemovalBackend:
    """The shipped backend: numpy/OpenCV region metrics + LPIPS-VGG on CPU."""

    VERSION = "removal-metrics/2"  # keep equal to runtime_identity.REMOVAL_METRICS_VERSION

    def __init__(self, ffmpeg_path: str = "ffmpeg", *, lpips_enabled: bool = True) -> None:
        self.ffmpeg_path = ffmpeg_path
        self._lpips_enabled = lpips_enabled
        self._lpips: LpipsVgg | None = None

    def version(self) -> str:
        return self.VERSION

    def preload(self) -> None:
        if self._lpips_enabled and self._lpips is None:
            self._lpips = LpipsVgg()

    def measure(
        self,
        *,
        canonical_reference: str,
        canonical_candidate: str,
        canonical_input: str,
        served_input_path: str,
        mask_stream_index: int,
        config: ScoringConfig,
        lpips_offset: int = 0,
        cancelled: Callable[[], bool] | None = None,
    ) -> RemovalMeasurement:
        # validator-side inputs first: if these fail nothing about the miner is known
        try:
            ref = Y4MReader.open(canonical_reference)
            served = Y4MReader.open(canonical_input)
            masks = decode_mask_stream(
                self.ffmpeg_path, served_input_path, stream_index=mask_stream_index, cancelled=cancelled
            )
        except (Y4MError, OSError, subprocess.SubprocessError) as exc:
            raise RemovalInputError(str(exc)) from exc
        if masks.shape[1:] != (ref.height, ref.width):
            raise RemovalInputError(
                f"mask geometry {masks.shape[2]}x{masks.shape[1]} != reference {ref.width}x{ref.height}"
            )
        if (served.width, served.height) != (ref.width, ref.height):
            raise RemovalInputError(
                f"served input geometry {served.width}x{served.height} != reference {ref.width}x{ref.height}"
            )
        n_input = min(ref.frame_count, served.frame_count, len(masks))
        if n_input == 0:
            raise RemovalInputError("reference / served input / mask stream have no frames in common")
        if not masks[:n_input].any():
            raise RemovalInputError("the mask is empty on every frame")
        # the miner's output
        try:
            cand = Y4MReader.open(canonical_candidate)
        except (Y4MError, OSError) as exc:
            raise RemovalMeasurementError(f"candidate unreadable: {exc}") from exc
        if (cand.width, cand.height) != (ref.width, ref.height):
            raise RemovalMeasurementError(
                f"candidate geometry {cand.width}x{cand.height} != reference {ref.width}x{ref.height}"
            )
        n = min(n_input, cand.frame_count)
        if n == 0:
            raise RemovalMeasurementError("candidate has no frames")
        masks = masks[:n]
        if not masks.any():
            raise RemovalMeasurementError("candidate is shorter than every masked frame")
        self.preload()
        stride = max(1, int(config.removal_lpips_stride))
        max_side = max(0, int(config.removal_lpips_max_side))
        offset = int(lpips_offset) % stride
        try:
            metrics, warp_ref = region_metrics(
                cand, ref, masks, lpips=self._lpips, lpips_stride=stride, lpips_max_side=max_side,
                lpips_offset=offset, cancelled=cancelled,
            )
            outside = outside_region_change(cand, served, masks, cancelled=cancelled)
        except Y4MError as exc:
            raise RemovalMeasurementError(str(exc)) from exc
        # the free fill, written next to the canonical files and scored the same way
        fd, floor_path = tempfile.mkstemp(prefix="median-fill-", suffix=".y4m", dir=os.path.dirname(canonical_candidate))
        os.close(fd)
        try:
            temporal_median_fill(served, masks, floor_path, cancelled=cancelled)
            floor_reader = Y4MReader.open(floor_path)
            floor, _ = region_metrics(
                floor_reader, ref, masks, lpips=self._lpips, lpips_stride=stride, lpips_max_side=max_side,
                lpips_offset=offset, cancelled=cancelled,
            )
        except (Y4MError, OSError) as exc:
            raise RemovalInputError(f"free-fill floor unavailable: {exc}") from exc
        finally:
            try:
                os.unlink(floor_path)
            except OSError:
                pass
        # the unchanged served input is a baseline too: returning the clip as received must never
        # beat the floor. On moving-camera clips the temporal median is often worse than leaving a
        # well-lit object in place, so the floor is the better of the two per term.
        try:
            unchanged, _ = region_metrics(
                Y4MReader.open(canonical_input), ref, masks, lpips=self._lpips, lpips_stride=stride,
                lpips_max_side=max_side, lpips_offset=offset, cancelled=cancelled,
            )
        except (Y4MError, OSError) as exc:
            raise RemovalInputError(f"unchanged-input baseline unavailable: {exc}") from exc
        floor = best_baseline(floor, unchanged)
        for name, value in (("lpips_vgg", metrics.lpips_vgg), ("floor.lpips_vgg", floor.lpips_vgg)):
            if self._lpips is not None and not math.isfinite(value):
                raise RemovalInputError(f"{name} is not finite (no LPIPS sample)")
        return RemovalMeasurement(
            metrics=metrics,
            floor=floor,
            warp_error_reference=warp_ref,
            outside_change=outside,
            mask_frames=n,
            mask_fraction=float(masks.mean()),
            lpips_offset=offset,
        )
