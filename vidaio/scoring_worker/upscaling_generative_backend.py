"""Generative upscaling metrics backend for the scoring worker (CPU, deterministic).

Measures a candidate against the sealed original and the served input
(vidaio/scoring/upscaling_generative.py): DISTS/LPIPS and CLIP-IQA on sampled full frames,
warp error under the reference's motion, low-resolution consistency and CIEDE2000. The
reference's own no-reference quality depends on no candidate and is measured once per item.

All inputs are the worker's private canonical y4m copies, so an auditor recomputes exactly
this from the archived artifacts.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Protocol

from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.removal import LpipsVgg, Y4MError, Y4MReader
from vidaio.scoring.upscaling_generative import (
    ClipIqa,
    DistsFull,
    UpscalingGenerativeCancelled,
    fr_frame_indices,
    generative_frame_metrics,
)
from vidaio.scoring.upscaling_generative_formula import GenerativeFrameMetrics

__all__ = [
    "CpuUpscalingGenerativeBackend",
    "UpscalingGenerativeCancelled",
    "UpscalingGenerativeInputError",
    "UpscalingGenerativeMeasurement",
    "UpscalingGenerativeMeasurementError",
    "UpscalingGenerativeMetricsBackend",
]


@dataclass(frozen=True)
class UpscalingGenerativeMeasurement:
    metrics: GenerativeFrameMetrics
    #: phase of the sampled full frames (derived from the held-out reference digest)
    fr_offset: int = 0


class UpscalingGenerativeMetricsBackend(Protocol):
    def measure(
        self,
        *,
        canonical_reference: str,
        canonical_candidate: str,
        canonical_served_input: str,
        config: ScoringConfig,
        fr_offset: int = 0,
        cancelled: Callable[[], bool] | None = None,
        reference_key: Hashable | None = None,
    ) -> UpscalingGenerativeMeasurement: ...

    def version(self) -> str: ...


class UpscalingGenerativeMeasurementError(RuntimeError):
    """CANDIDATE-side failure: the miner's output cannot be measured (unreadable canonical
    candidate, geometry or frame count unlike the original). Scored as a violation."""


class UpscalingGenerativeInputError(RuntimeError):
    """VALIDATOR-side failure: the original or the served input cannot be read or do not
    match. Never the miner's fault — the worker refuses the request (422) so the item is
    skipped instead of zeroed."""


class CpuUpscalingGenerativeBackend:
    """The shipped backend: DISTS + LPIPS-VGG + CLIP-IQA on CPU, OpenCV for the rest."""

    VERSION = "upscaling-generative-metrics/1"  # keep equal to runtime_identity.UPSCALING_GENERATIVE_METRICS_VERSION

    #: reference no-reference values kept per item (every contender scores the same items)
    REFERENCE_CACHE_ITEMS = 128

    def __init__(
        self,
        *,
        lpips_loader: Callable[[], LpipsVgg | None] | None = None,
        dists_loader: Callable[[], Any] | None = None,
        nr_loader: Callable[[], Any] | None = None,
    ) -> None:
        """``lpips_loader`` shares an already-loaded LPIPS-VGG network; the DISTS and
        CLIP-IQA loaders exist for tests (fake networks with the same interface)."""
        self._lpips_loader = lpips_loader
        self._dists_loader = dists_loader
        self._nr_loader = nr_loader
        self._lpips: Any = None
        self._dists: Any = None
        self._nr: Any = None
        self._load_lock = threading.Lock()
        self._references: OrderedDict[Hashable, float] = OrderedDict()
        self._references_lock = threading.Lock()

    def version(self) -> str:
        return self.VERSION

    def preload(self) -> None:
        with self._load_lock:
            if self._lpips is None:
                self._lpips = self._lpips_loader() if self._lpips_loader is not None else LpipsVgg()
            if self._dists is None:
                self._dists = self._dists_loader() if self._dists_loader is not None else DistsFull()
            if self._nr is None:
                self._nr = self._nr_loader() if self._nr_loader is not None else ClipIqa()

    def measure(
        self,
        *,
        canonical_reference: str,
        canonical_candidate: str,
        canonical_served_input: str,
        config: ScoringConfig,
        fr_offset: int = 0,
        cancelled: Callable[[], bool] | None = None,
        reference_key: Hashable | None = None,
    ) -> UpscalingGenerativeMeasurement:
        """``reference_key`` names the verified original and served input (their digests);
        with it the reference's no-reference quality is measured once per item."""
        try:
            ref = Y4MReader.open(canonical_reference)
            served = Y4MReader.open(canonical_served_input)
        except (Y4MError, OSError) as exc:
            raise UpscalingGenerativeInputError(str(exc)) from exc
        if ref.frame_count < 1 or served.frame_count != ref.frame_count:
            raise UpscalingGenerativeInputError(
                f"original has {ref.frame_count} frames and the served input {served.frame_count}"
            )
        if served.width > ref.width or served.height > ref.height:
            raise UpscalingGenerativeInputError(
                f"served input {served.width}x{served.height} is larger than the original "
                f"{ref.width}x{ref.height}"
            )
        try:
            cand = Y4MReader.open(canonical_candidate)
        except (Y4MError, OSError) as exc:
            raise UpscalingGenerativeMeasurementError(f"candidate unreadable: {exc}") from exc
        if (cand.width, cand.height) != (ref.width, ref.height):
            raise UpscalingGenerativeMeasurementError(
                f"candidate geometry {cand.width}x{cand.height} != original {ref.width}x{ref.height}"
            )
        if cand.frame_count != ref.frame_count:
            raise UpscalingGenerativeMeasurementError(
                f"candidate has {cand.frame_count} frames, expected {ref.frame_count}"
            )
        self.preload()
        indices = tuple(
            fr_frame_indices(ref.frame_count, config.upscaling_generative_fr_frames, fr_offset)
        )
        cache_key = (
            None
            if reference_key is None
            else (reference_key, indices, int(config.upscaling_generative_fr_max_side))
        )
        nr_reference = None
        if cache_key is not None:
            with self._references_lock:
                nr_reference = self._references.get(cache_key)
                if nr_reference is not None:
                    self._references.move_to_end(cache_key)
        try:
            metrics = generative_frame_metrics(
                cand,
                ref,
                served,
                dists=self._dists,
                lpips=self._lpips,
                nr=self._nr,
                config=config,
                fr_offset=fr_offset,
                nr_reference=nr_reference,
                cancelled=cancelled,
            )
        except Y4MError as exc:
            raise UpscalingGenerativeMeasurementError(str(exc)) from exc
        except ValueError as exc:
            raise UpscalingGenerativeInputError(str(exc)) from exc
        for name in ("dists", "lpips_vgg", "nr", "nr_reference"):
            if not math.isfinite(getattr(metrics, name)):
                raise UpscalingGenerativeInputError(f"{name} is not finite")
        if not metrics.nr_reference > 0.0:
            raise UpscalingGenerativeInputError("the original's no-reference quality is not positive")
        if cache_key is not None and nr_reference is None:
            with self._references_lock:
                self._references[cache_key] = metrics.nr_reference
                self._references.move_to_end(cache_key)
                while len(self._references) > self.REFERENCE_CACHE_ITEMS:
                    self._references.popitem(last=False)
        return UpscalingGenerativeMeasurement(metrics=metrics, fr_offset=fr_offset)
