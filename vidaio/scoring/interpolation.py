"""Frame-interpolation track — the inserted frames against the real ones.

Task: the served input is the sealed reference with every other frame removed (frames
0, 2, 4, ... of a genuine high-frame-rate clip, at half the rate). The miner returns the
clip at the reference rate: the input frames unchanged at the even positions and one new
frame between each pair (N input frames -> 2N - 1 output frames). Scoring compares ONLY
the inserted frames (odd positions) with the reference frames that were removed, and
demands that the even positions still carry the input frames.

Metrics over the inserted frames (all recomputable by an auditor from the archived input,
output and reference):

* ``psnr_db``   — global PSNR over every pixel of every inserted frame (RGB, 8-bit).
* ``ssim``      — SSIM map (Gaussian 11/1.5 on luma) averaged per frame, then over frames.
* ``lpips_vgg`` — LPIPS (VGG16 backbone, v0.1 linear heads) on every k-th inserted frame,
                  phase-shifted by an offset derived from the held-out reference digest,
                  frames bounded to a long side of ``interpolation_lpips_max_side``.

Floor: the validator measures, with the same metrics, the two answers anyone gets for
free — repeating the previous input frame (hold) and averaging its two neighbours
(blend, per I420 plane with rounding) — and the floor is the better of the two on every
term (``best_baseline``).

Score (the audit-recompute record is :class:`InterpolationBreakdown`)::

    s_psnr  = clamp((psnr - psnr_floor) / interpolation_psnr_span, 0, 1)
    s_ssim  = clamp((ssim - ssim_floor) / (1 - ssim_floor), 0, 1)
    s_lpips = clamp((lpips_floor - lpips) / lpips_floor, 0, 1)
    final   = w_psnr * s_psnr + w_ssim * s_ssim + w_lpips * s_lpips   (0.3 / 0.3 / 0.4)

Gates: the exact frame count (shared stream validation, ``FRAME_COUNT_MISMATCH``) and the
kept-frame identity check (``KEPT_FRAMES_MODIFIED``): the max over frames of the mean abs
RGB difference between output frame 2k and input frame k may not exceed
``interpolation_kept_tolerance``. It also catches an output shifted off the frame grid.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from vidaio.scoring.interpolation_formula import InsertedFrameMetrics
from vidaio.scoring.removal import (
    LpipsVgg,
    Y4MError,
    Y4MReader,
    _bound_pair,
    _ssim_map,
)

__all__ = [
    "InterpolationCancelled",
    "inserted_frame_metrics",
    "kept_frame_change",
]


class InterpolationCancelled(RuntimeError):
    """The worker's deadline/cancellation fired while measuring."""


def _rgb(i420: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)


def _blend_i420(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rounded per-plane average of two I420 frames (deterministic integer math)."""
    return ((a.astype(np.uint16) + b.astype(np.uint16) + 1) >> 1).astype(np.uint8)


def kept_frame_change(
    candidate: Y4MReader,
    served_input: Y4MReader,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> float:
    """Max over input frames k of the mean abs RGB difference (8-bit units) between
    output frame 2k and input frame k. An honest output keeps this near zero."""
    import cv2

    if (candidate.width, candidate.height) != (served_input.width, served_input.height):
        raise Y4MError(
            f"candidate geometry {candidate.width}x{candidate.height} != served input "
            f"{served_input.width}x{served_input.height}"
        )
    worst = 0.0
    for k in range(served_input.frame_count):
        if cancelled is not None and cancelled():
            raise InterpolationCancelled("kept-frame check cancelled")
        if 2 * k >= candidate.frame_count:
            break
        diff = cv2.absdiff(candidate.frame_rgb(2 * k), served_input.frame_rgb(k))
        worst = max(worst, float(sum(cv2.mean(diff)[:3]) / 3.0))
    return worst


class _Accumulator:
    """PSNR (integer-exact squared-error sum), SSIM and LPIPS for one answer."""

    def __init__(self) -> None:
        self.sq_err = 0
        self.values = 0
        self.ssim: list[float] = []
        self.lpips: list[float] = []

    def add(
        self,
        rgb: np.ndarray,
        luma: np.ndarray,
        ref_rgb: np.ndarray,
        ref_luma: np.ndarray,
        *,
        lpips: LpipsVgg | None,
        lpips_max_side: int,
        sample: bool,
    ) -> None:
        d = rgb.astype(np.int32) - ref_rgb.astype(np.int32)
        self.sq_err += int((d * d).sum())
        self.values += int(d.size)
        self.ssim.append(float(_ssim_map(luma, ref_luma).mean()))
        if lpips is not None and sample:
            self.lpips.append(lpips.distance(*_bound_pair(rgb, ref_rgb, lpips_max_side)))

    def metrics(self, inserted: int) -> InsertedFrameMetrics:
        mse = self.sq_err / self.values
        psnr = 10.0 * math.log10(255.0**2 / mse) if mse > 0 else 99.0
        return InsertedFrameMetrics(
            psnr_db=float(min(psnr, 99.0)),
            ssim=float(np.mean(self.ssim)),
            lpips_vgg=float(np.mean(self.lpips)) if self.lpips else float("nan"),
            inserted_frames=inserted,
        )


def inserted_frame_metrics(
    candidate: Y4MReader,
    reference: Y4MReader,
    served_input: Y4MReader,
    *,
    lpips: LpipsVgg | None,
    lpips_stride: int,
    lpips_max_side: int = 0,
    lpips_offset: int = 0,
    cancelled: Callable[[], bool] | None = None,
    baselines: tuple[InsertedFrameMetrics, InsertedFrameMetrics] | None = None,
) -> tuple[InsertedFrameMetrics, InsertedFrameMetrics, InsertedFrameMetrics]:
    """Metrics of the candidate's inserted frames and of the two free answers.

    Returns ``(candidate, hold, blend)``, all measured on the same inserted positions
    2k + 1 (k = 0 .. N-2) against the reference, with the same LPIPS sample.
    ``baselines`` = ``(hold, blend)`` already measured on this exact reference, input
    and sample: they depend on no candidate, so only the candidate is measured."""
    if not (
        (candidate.width, candidate.height)
        == (reference.width, reference.height)
        == (served_input.width, served_input.height)
    ):
        raise Y4MError(
            f"geometry differs: candidate {candidate.width}x{candidate.height}, reference "
            f"{reference.width}x{reference.height}, input {served_input.width}x{served_input.height}"
        )
    inserted = served_input.frame_count - 1
    if inserted < 1:
        raise ValueError("the input has fewer than two frames: nothing to insert")
    if reference.frame_count != 2 * served_input.frame_count - 1:
        raise ValueError(
            f"reference has {reference.frame_count} frames, expected "
            f"{2 * served_input.frame_count - 1} for {served_input.frame_count} input frames"
        )
    if candidate.frame_count != reference.frame_count:
        raise Y4MError(
            f"candidate has {candidate.frame_count} frames, reference {reference.frame_count}"
        )
    stride = max(1, int(lpips_stride))
    offset = int(lpips_offset) % stride
    plane = reference.width * reference.height
    width = reference.width
    names = ("candidate",) if baselines is not None else ("candidate", "hold", "blend")
    answers = {name: _Accumulator() for name in names}
    sampled_any = False
    previous = served_input.frame_i420(0)
    for k in range(inserted):
        if cancelled is not None and cancelled():
            raise InterpolationCancelled("inserted-frame metrics cancelled")
        nxt = served_input.frame_i420(k + 1)
        position = 2 * k + 1
        ref_i420 = reference.frame_i420(position)
        ref_rgb, ref_luma = _rgb(ref_i420), ref_i420[: plane // width]
        sample = (k + offset) % stride == 0
        if k == inserted - 1 and not sampled_any:
            sample = True  # fewer inserted frames than the stride: sample the last one
        sampled_any = sampled_any or sample
        cand_i420 = candidate.frame_i420(position)
        free = () if baselines is not None else (
            ("hold", previous), ("blend", _blend_i420(previous, nxt))
        )
        for name, i420 in (("candidate", cand_i420), *free):
            answers[name].add(
                _rgb(i420),
                i420[: plane // width],
                ref_rgb,
                ref_luma,
                lpips=lpips,
                lpips_max_side=lpips_max_side,
                sample=sample,
            )
        previous = nxt
    if baselines is not None:
        return (answers["candidate"].metrics(inserted), *baselines)
    return (
        answers["candidate"].metrics(inserted),
        answers["hold"].metrics(inserted),
        answers["blend"].metrics(inserted),
    )
