"""Generative video-upscaling track — perceptual, temporal and no-reference measurements.

Task: the served input is the sealed original degraded and downscaled by the item's
``upscale_factor`` (2 or 4); the miner returns the clip at the original's geometry and frame
count. Unlike the faithful track, invented detail is allowed: the output is judged on how
close it LOOKS to the original, how stable it is in time, and how good it looks on its own,
while two gates keep it anchored to what the served input actually shows.

Measurements (one pass over every frame; recomputable by an auditor from the archived input,
output and reference — CPU only, single-threaded, pinned weights):

* ``dists`` / ``lpips_vgg`` — DISTS (VGG16 + L2 pooling, piq 0.8.0 weights) and LPIPS-VGG
  v0.1 against the sealed original on ``upscaling_generative_fr_frames`` sampled FULL frames
  (native resolution unless ``upscaling_generative_fr_max_side`` bounds them; DISTS runs
  without piq's internal 256-px resize). The sample's phase is derived from the held-out
  reference digest.
* ``warp_error`` / ``warp_error_reference`` — temporal consistency: the REFERENCE's DIS optical
  flow (the removal track's warp code, :func:`vidaio.scoring.removal.reference_flow_maps`)
  aligns frame i+1 with frame i; the mean absolute luma difference that remains is measured
  for the candidate and for the reference itself, on frames area-downscaled to
  ``upscaling_generative_temporal_max_side``.
* ``nr`` / ``nr_reference`` — no-reference quality: CLIP-IQA (the original CLIP RN50 image encoder
  without positional embedding, prompts "Good photo." / "Bad photo.", piq 0.8.0's
  implementation) on the same sampled full frames, for the candidate and the reference.
* ``lowres_psnr_db`` — low-resolution consistency: every output frame's luma is
  Lanczos-downscaled to the served input's geometry, both sides Gaussian-blurred
  (``upscaling_generative_lowres_blur_sigma``), global PSNR over all frames.
* ``color_delta_e`` — mean CIEDE2000 (BT.709 limited range -> sRGB -> CIELAB D65) against the
  original on frames area-downscaled to ``upscaling_generative_color_max_side``.

The score formula is :mod:`vidaio.scoring.upscaling_generative_formula`; the gates are
:func:`vidaio.scoring.gates.upscaling_generative_violations`.

Why CLIP-IQA and not MUSIQ: see the project design record (comp 006 entry). In short, the only
maintained PyTorch MUSIQ port (pyiqa) is licensed PolyForm Noncommercial and is not in the
lock, a clean-room
port cannot be validated against the reference implementation in time, and MUSIQ's
multi-scale patch transformer is slow single-threaded at 1080p. CLIP-IQA ships in the already
pinned piq 0.8.0 (Apache-2.0) with the MIT-licensed original CLIP weights, is a plain convolutional
network (deterministic on CPU) and accepts native-resolution frames.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.generative_weights import (
    CLIP_RN50_WEIGHTS,
    DISTS_WEIGHTS,
    VGG16_WEIGHTS,
    GenerativeWeightsError,
    PinnedWeights,
    generative_weights_digest,
    pinned_weights_path,
)
from vidaio.scoring.removal import (
    LpipsVgg,
    Y4MError,
    Y4MReader,
    _bound_pair,
    reference_flow_maps,
    warp_with_maps,
)
from vidaio.scoring.upscaling_generative_formula import GenerativeFrameMetrics

__all__ = [
    "CLIP_RN50_WEIGHTS",
    "DISTS_WEIGHTS",
    "VGG16_WEIGHTS",
    "ClipIqa",
    "DistsFull",
    "GenerativeWeightsError",
    "PinnedWeights",
    "UpscalingGenerativeCancelled",
    "ciede2000",
    "fr_frame_indices",
    "fr_sampling_stride",
    "generative_frame_metrics",
    "generative_weights_digest",
    "pinned_weights_path",
]


class UpscalingGenerativeCancelled(RuntimeError):
    """The worker's deadline/cancellation fired while measuring."""


#: CLIP BPE tokens of the CLIP-IQA antonym prompts "Good photo." and "Bad photo." (start,
#: words, ".", end; zero padded to 77). Identical to piq 0.8.0's ``clipiqa_tokens.pt``
#: (sha256 d5580bd2bf254546b4b11ce82aea51cd0556203c7822bd237057ce933bf7e07d), inlined so
#: nothing is fetched or unpickled at scoring time.
CLIPIQA_PROMPT_TOKENS = ((49406, 886, 1125, 269, 49407), (49406, 2103, 1125, 269, 49407))
_CLIP_CONTEXT = 77
_OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _to_tensor(rgb: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255.0


class DistsFull:
    """DISTS on CPU at the frames' own resolution (no 256-px resize), loaded once per thread.

    piq's ``DISTS.forward`` downsizes inputs to a 256-px short side, which would erase the
    detail a generative upscaler adds; the distance is computed from piq's own feature
    extractor and weights instead (``1 - weighted structure/texture similarity``)."""

    def __init__(self, *, download: bool = False) -> None:
        pinned_weights_path(VGG16_WEIGHTS, download=download)
        pinned_weights_path(DISTS_WEIGHTS, download=download)
        import piq
        import torch
        from piq.perceptual import ContentLoss

        torch.set_grad_enabled(False)
        self._torch = torch
        self._similarity = ContentLoss.forward
        self._net = piq.DISTS(reduction="none").eval()

    def distance(self, a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
        with self._torch.no_grad():
            similarity = self._similarity(self._net, _to_tensor(a_rgb), _to_tensor(b_rgb))
        return float(1.0 - float(similarity.reshape(-1)[0]))


class ClipIqa:
    """CLIP-IQA on CPU at the frame's own resolution, loaded once per thread.

    The image encoder runs without its positional embedding (as in the CLIP-IQA paper and
    piq's implementation), so a 1080p frame is judged whole; the score is the softmax
    probability of "Good photo." against "Bad photo.", in (0, 1)."""

    def __init__(self, *, download: bool = False) -> None:
        path = pinned_weights_path(CLIP_RN50_WEIGHTS, download=download)
        import torch
        from piq.feature_extractors import clip

        torch.set_grad_enabled(False)
        with path.open("rb") as handle:
            scripted = torch.jit.load(handle, map_location="cpu").eval()
        model = clip.build_model(scripted.state_dict()).float().eval()
        tokens = torch.zeros((len(CLIPIQA_PROMPT_TOKENS), _CLIP_CONTEXT), dtype=torch.long)
        for row, prompt in enumerate(CLIPIQA_PROMPT_TOKENS):
            tokens[row, : len(prompt)] = torch.tensor(prompt, dtype=torch.long)
        with torch.no_grad():
            anchors = model.encode_text(tokens).float()
        self._torch = torch
        self._model = model
        self._anchors = anchors / anchors.norm(dim=-1, keepdim=True)
        self._logit_scale = model.logit_scale.exp().float()
        self._mean = torch.tensor(_OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(_OPENAI_CLIP_STD).view(1, 3, 1, 1)

    def quality(self, rgb: np.ndarray) -> float:
        torch = self._torch
        x = (_to_tensor(rgb) - self._mean) / self._std
        with torch.no_grad():
            features = self._model.encode_image(x, pos_embedding=False).float()
            features = features / features.norm(dim=-1, keepdim=True)
            logits = self._logit_scale * features @ self._anchors.t()
            probs = logits.reshape(logits.shape[0], -1, 2).softmax(dim=-1)
        return float(probs[0, 0, 0])


# --- colour ------------------------------------------------------------------------------

_KR, _KB = 0.2126, 0.0722
_KG = 1.0 - _KR - _KB
_XYZ_FROM_LINEAR_RGB = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_D65_WHITE = np.array([0.95047, 1.0, 1.08883])


def _i420_lab(i420: np.ndarray, width: int, height: int, size: tuple[int, int]) -> np.ndarray:
    """CIELAB (D65) of a BT.709 limited-range I420 frame, area-downscaled to ``size``."""
    import cv2

    plane = width * height
    flat = i420.reshape(-1)
    y = flat[:plane].reshape(height, width)
    u = flat[plane : plane + plane // 4].reshape(height // 2, width // 2)
    v = flat[plane + plane // 4 : plane + plane // 2].reshape(height // 2, width // 2)
    ys = cv2.resize(y.astype(np.float32), size, interpolation=cv2.INTER_AREA).astype(np.float64)
    us = cv2.resize(u.astype(np.float32), size, interpolation=cv2.INTER_AREA).astype(np.float64)
    vs = cv2.resize(v.astype(np.float32), size, interpolation=cv2.INTER_AREA).astype(np.float64)
    luma = (ys - 16.0) / 219.0
    cb = (us - 128.0) / 224.0
    cr = (vs - 128.0) / 224.0
    r = luma + 2.0 * (1.0 - _KR) * cr
    g = luma - (2.0 * _KB * (1.0 - _KB) / _KG) * cb - (2.0 * _KR * (1.0 - _KR) / _KG) * cr
    b = luma + 2.0 * (1.0 - _KB) * cb
    rgb = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _XYZ_FROM_LINEAR_RGB.T / _D65_WHITE
    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3.0 * delta * delta) + 4.0 / 29.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference per element (Sharma, Wu & Dalal 2005), kL = kC = kH = 1."""
    l1, a1, b1 = (lab1[..., i].astype(np.float64) for i in range(3))
    l2, a2, b2 = (lab2[..., i].astype(np.float64) for i in range(3))
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar7 = ((c1 + c2) / 2.0) ** 7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.where((a1p == 0) & (b1 == 0), 0.0, np.degrees(np.arctan2(b1, a1p)) % 360.0)
    h2p = np.where((a2p == 0) & (b2 == 0), 0.0, np.degrees(np.arctan2(b2, a2p)) % 360.0)
    chroma_zero = (c1p * c2p) == 0
    dh = h2p - h1p
    dh = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dh = np.where(chroma_zero, 0.0, dh)
    d_l = l2 - l1
    d_c = c2p - c1p
    d_h = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2.0)
    l_bar = (l1 + l2) / 2.0
    c_bar_p = (c1p + c2p) / 2.0
    h_sum = h1p + h2p
    h_bar = np.where(
        chroma_zero,
        h_sum,
        np.where(
            np.abs(h1p - h2p) <= 180.0,
            h_sum / 2.0,
            np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0),
        ),
    )
    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar))
        + 0.32 * np.cos(np.radians(3.0 * h_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((h_bar - 275.0) / 25.0) ** 2))
    c_bar_p7 = c_bar_p**7
    r_c = 2.0 * np.sqrt(c_bar_p7 / (c_bar_p7 + 25.0**7))
    s_l = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c
    tl, tc, th = d_l / s_l, d_c / s_c, d_h / s_h
    return np.sqrt(np.maximum(tl * tl + tc * tc + th * th + r_t * tc * th, 0.0))


# --- sampling ----------------------------------------------------------------------------


def fr_sampling_stride(frames: int, count: int) -> int:
    """Spacing of the evenly spaced full-frame sample (>= 1)."""
    if frames < 1:
        raise ValueError("no frames to sample")
    return max(1, frames // max(1, min(int(count), frames)))


def fr_frame_indices(frames: int, count: int, offset: int) -> list[int]:
    """``count`` evenly spaced frame indices, phase-shifted by ``offset`` (mod the stride)."""
    n = max(1, min(int(count), frames))
    stride = fr_sampling_stride(frames, n)
    phase = int(offset) % stride
    return [phase + k * stride for k in range(n)]


def _bounded_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """(w, h) area-downscaled so the long side is <= max_side (0 = unchanged)."""
    long = max(width, height)
    if max_side <= 0 or long <= max_side:
        return width, height
    scale = max_side / long
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _rgb(i420: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)


def _blur_kernel(sigma: float) -> tuple[int, int]:
    size = 2 * int(math.ceil(3.0 * sigma)) + 1
    return size, size


def generative_frame_metrics(
    candidate: Y4MReader,
    reference: Y4MReader,
    served_input: Y4MReader,
    *,
    dists: DistsFull,
    lpips: LpipsVgg,
    nr: ClipIqa,
    config: ScoringConfig,
    fr_offset: int = 0,
    nr_reference: float | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> GenerativeFrameMetrics:
    """Every generative measurement of ``candidate`` in one pass over the frames.

    ``served_input`` is the canonical served input at its OWN geometry (not upscaled).
    ``nr_reference`` = the reference's no-reference quality on this exact frame sample when
    already measured for another candidate of the item (it depends on no candidate)."""
    import cv2

    if (candidate.width, candidate.height) != (reference.width, reference.height):
        raise Y4MError(
            f"geometry differs: candidate {candidate.width}x{candidate.height}, reference "
            f"{reference.width}x{reference.height}"
        )
    frames = reference.frame_count
    if frames < 1:
        raise ValueError("the reference has no frames")
    if served_input.frame_count != frames:
        raise ValueError(f"served input has {served_input.frame_count} frames, reference {frames}")
    if candidate.frame_count != frames:
        raise Y4MError(f"candidate has {candidate.frame_count} frames, reference {frames}")
    if served_input.width > reference.width or served_input.height > reference.height:
        raise ValueError("served input is larger than the reference")
    width, height = reference.width, reference.height
    in_size = (served_input.width, served_input.height)
    sigma = float(config.upscaling_generative_lowres_blur_sigma)
    kernel = _blur_kernel(sigma)
    color_size = _bounded_size(width, height, config.upscaling_generative_color_max_side)
    temporal_size = _bounded_size(width, height, config.upscaling_generative_temporal_max_side)
    fr_max_side = max(0, int(config.upscaling_generative_fr_max_side))
    fr_indices = set(fr_frame_indices(frames, config.upscaling_generative_fr_frames, fr_offset))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    lowres_sq = 0.0
    lowres_count = 0
    delta_e: list[float] = []
    warp_c: list[float] = []
    warp_r: list[float] = []
    dists_values: list[float] = []
    lpips_values: list[float] = []
    nr_values: list[float] = []
    nr_ref_values: list[float] = []
    prev_ref_small: np.ndarray | None = None
    prev_cand_small: np.ndarray | None = None

    def _small(luma: np.ndarray) -> np.ndarray:
        if temporal_size == (width, height):
            return np.ascontiguousarray(luma)
        return cv2.resize(np.ascontiguousarray(luma), temporal_size, interpolation=cv2.INTER_AREA)

    for k in range(frames):
        if cancelled is not None and cancelled():
            raise UpscalingGenerativeCancelled("generative upscaling metrics cancelled")
        ref_i420 = reference.frame_i420(k)
        cand_i420 = candidate.frame_i420(k)
        cand_luma = cand_i420[:height]
        ref_luma = ref_i420[:height]

        # low-resolution consistency (luma, at the served input's geometry)
        down = cv2.resize(cand_luma.astype(np.float32), in_size, interpolation=cv2.INTER_LANCZOS4)
        in_luma = served_input.frame_i420(k)[: served_input.height].astype(np.float32)
        diff = cv2.GaussianBlur(down, kernel, sigma, borderType=cv2.BORDER_REFLECT_101) - cv2.GaussianBlur(
            in_luma, kernel, sigma, borderType=cv2.BORDER_REFLECT_101
        )
        diff64 = diff.astype(np.float64)
        lowres_sq += float((diff64 * diff64).sum())
        lowres_count += int(diff64.size)

        # colour (regional CIEDE2000 against the original)
        delta_e.append(
            float(
                ciede2000(
                    _i420_lab(cand_i420, width, height, color_size),
                    _i420_lab(ref_i420, width, height, color_size),
                ).mean()
            )
        )

        # temporal consistency under the reference's own motion
        ref_small = _small(ref_luma)
        cand_small = _small(cand_luma)
        if prev_ref_small is not None and prev_cand_small is not None:
            mx, my = reference_flow_maps(dis, prev_ref_small, ref_small)
            wc = warp_with_maps(cand_small, mx, my).astype(np.float32)
            wr = warp_with_maps(ref_small, mx, my).astype(np.float32)
            warp_c.append(float(np.abs(wc - prev_cand_small.astype(np.float32)).mean()))
            warp_r.append(float(np.abs(wr - prev_ref_small.astype(np.float32)).mean()))
        prev_ref_small, prev_cand_small = ref_small, cand_small

        # full-reference and no-reference terms on the sampled full frames
        if k in fr_indices:
            cand_rgb, ref_rgb = _bound_pair(_rgb(cand_i420), _rgb(ref_i420), fr_max_side)
            dists_values.append(dists.distance(cand_rgb, ref_rgb))
            lpips_values.append(lpips.distance(cand_rgb, ref_rgb))
            nr_values.append(nr.quality(cand_rgb))
            if nr_reference is None:
                nr_ref_values.append(nr.quality(ref_rgb))

    mse = lowres_sq / lowres_count
    lowres_psnr = 10.0 * math.log10(255.0**2 / mse) if mse > 0 else 99.0
    return GenerativeFrameMetrics(
        dists=float(np.mean(dists_values)),
        lpips_vgg=float(np.mean(lpips_values)),
        warp_error=float(np.mean(warp_c)) if warp_c else 0.0,
        warp_error_reference=float(np.mean(warp_r)) if warp_r else 0.0,
        nr=float(np.mean(nr_values)),
        nr_reference=float(nr_reference) if nr_reference is not None else float(np.mean(nr_ref_values)),
        lowres_psnr_db=float(min(lowres_psnr, 99.0)),
        color_delta_e=float(np.mean(delta_e)),
        frames=frames,
        fr_frames=len(fr_indices),
    )
