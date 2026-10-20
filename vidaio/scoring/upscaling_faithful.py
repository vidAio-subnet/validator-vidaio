"""Faithful video-upscaling track — the upscaled frames against the sealed original.

Task: the served input is the sealed original degraded and downscaled by the item's
``upscale_factor`` (2 or 4); the miner returns the clip at the original's geometry and frame
count. Every output frame is compared with the original frame it must restore.

Metrics over all frames (recomputable by an auditor from the archived input, output and
reference):

* ``psnr_db``   — global PSNR over every pixel of every frame (RGB, 8-bit).
* ``ssim``      — SSIM map (Gaussian 11/1.5 on luma) averaged per frame, then over frames.
* ``lpips_vgg`` — LPIPS (VGG16 backbone, v0.1 linear heads) on every k-th frame, phase-shifted
                  by an offset derived from the held-out reference digest, frames bounded to a
                  long side of ``upscaling_faithful_lpips_max_side``.

Floor: the free answer — the served input Lanczos-upscaled to the original's geometry. The
scoring worker already canonicalizes the served input exactly so (ffmpeg ``scale=W:H:flags=
lanczos``, then the shared canonical pixel format), and the floor is measured from that file
with the same metrics.

Score (the audit-recompute record is :class:`UpscalingFaithfulBreakdown`)::

    s_psnr  = clamp((psnr - psnr_floor) / upscaling_faithful_psnr_span, 0, 1)
    s_ssim  = clamp((ssim - ssim_floor) / (1 - ssim_floor), 0, 1)
    s_lpips = clamp((lpips_floor - lpips) / lpips_floor, 0, 1)
    final   = w_psnr * s_psnr + w_ssim * s_ssim + w_lpips * s_lpips   (0.3 / 0.3 / 0.4)

Gates: the shared stream validation (geometry equal to the original, exact frame count). No
VMAF, PieAPP, file-size or perceptual gate applies: the comparison with the original rules
out what those guarded against.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from vidaio.scoring.removal import (
    LpipsVgg,
    Y4MError,
    Y4MReader,
    _bound_pair,
    _ssim_map,
)
from vidaio.scoring.upscaling_faithful_formula import UpscaledFrameMetrics

__all__ = ["UpscalingFaithfulCancelled", "upscaled_frame_metrics"]


class UpscalingFaithfulCancelled(RuntimeError):
    """The worker's deadline/cancellation fired while measuring."""


def _rgb(i420: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)


class _Accumulator:
    """PSNR (integer-exact squared-error sum), SSIM and LPIPS for one answer."""

    def __init__(self) -> None:
        self.sq_err = 0
        self.values = 0
        self.ssim: list[float] = []
        self.lpips: list[float] = []

    def add(self, rgb, luma, ref_rgb, ref_luma, *, lpips: LpipsVgg | None, lpips_max_side: int, sample: bool) -> None:
        d = rgb.astype(np.int32) - ref_rgb.astype(np.int32)
        self.sq_err += int((d * d).sum())
        self.values += int(d.size)
        self.ssim.append(float(_ssim_map(luma, ref_luma).mean()))
        if lpips is not None and sample:
            self.lpips.append(lpips.distance(*_bound_pair(rgb, ref_rgb, lpips_max_side)))

    def metrics(self, frames: int) -> UpscaledFrameMetrics:
        mse = self.sq_err / self.values
        psnr = 10.0 * math.log10(255.0**2 / mse) if mse > 0 else 99.0
        return UpscaledFrameMetrics(
            psnr_db=float(min(psnr, 99.0)),
            ssim=float(np.mean(self.ssim)),
            lpips_vgg=float(np.mean(self.lpips)) if self.lpips else float("nan"),
            frames=frames,
        )


def upscaled_frame_metrics(
    candidate: Y4MReader,
    reference: Y4MReader,
    floor_input: Y4MReader,
    *,
    lpips: LpipsVgg | None,
    lpips_stride: int,
    lpips_max_side: int = 0,
    lpips_offset: int = 0,
    cancelled: Callable[[], bool] | None = None,
    floor: UpscaledFrameMetrics | None = None,
) -> tuple[UpscaledFrameMetrics, UpscaledFrameMetrics]:
    """Metrics of the candidate and of the Lanczos floor against the original.

    ``floor_input`` is the served input already Lanczos-upscaled to the original's geometry
    (the worker's canonical miner input). Returns ``(candidate, floor)``, both measured on the
    same frames with the same LPIPS sample. ``floor`` = the floor already measured on this exact
    reference, input and sample: it depends on no candidate, so only the candidate is measured."""
    if not (
        (candidate.width, candidate.height)
        == (reference.width, reference.height)
        == (floor_input.width, floor_input.height)
    ):
        raise Y4MError(
            f"geometry differs: candidate {candidate.width}x{candidate.height}, reference "
            f"{reference.width}x{reference.height}, upscaled input {floor_input.width}x{floor_input.height}"
        )
    frames = reference.frame_count
    if frames < 1:
        raise ValueError("the reference has no frames")
    if floor_input.frame_count != frames:
        raise ValueError(f"served input has {floor_input.frame_count} frames, reference {frames}")
    if candidate.frame_count != frames:
        raise Y4MError(f"candidate has {candidate.frame_count} frames, reference {frames}")
    stride = max(1, int(lpips_stride))
    offset = int(lpips_offset) % stride
    plane = reference.width * reference.height
    width = reference.width
    names = ("candidate",) if floor is not None else ("candidate", "floor")
    answers = {name: _Accumulator() for name in names}
    sampled_any = False
    for k in range(frames):
        if cancelled is not None and cancelled():
            raise UpscalingFaithfulCancelled("upscaled-frame metrics cancelled")
        ref_i420 = reference.frame_i420(k)
        ref_rgb, ref_luma = _rgb(ref_i420), ref_i420[: plane // width]
        sample = (k + offset) % stride == 0
        if k == frames - 1 and not sampled_any:
            sample = True  # fewer frames than the stride: sample the last one
        sampled_any = sampled_any or sample
        todo = [("candidate", candidate.frame_i420(k))]
        if floor is None:
            todo.append(("floor", floor_input.frame_i420(k)))
        for name, i420 in todo:
            answers[name].add(_rgb(i420), i420[: plane // width], ref_rgb, ref_luma,
                              lpips=lpips, lpips_max_side=lpips_max_side, sample=sample)
    measured = answers["candidate"].metrics(frames)
    return measured, (floor if floor is not None else answers["floor"].metrics(frames))
