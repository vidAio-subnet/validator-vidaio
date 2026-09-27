"""Object-removal track — region-restricted quality against the sealed reference.

Task: the served input is the reference clip with a seed-drawn opaque occluder (and its
soft shadow) composited on top; the per-frame binary mask of that occluder travels with
the input as a second, lossless gray video stream. The miner returns the clip with the
masked region filled. Scoring compares ONLY the masked region with the sealed reference
(the pristine pixels under the occluder are the ground truth), and demands that nothing
outside the mask changed.

Metrics (all inside the mask, all recomputable by an auditor from the archived input,
output and reference):

* ``psnr_db``     — global PSNR over every masked pixel of every frame (RGB, 8-bit).
* ``ssim``        — SSIM map (Gaussian 11/1.5 on luma) averaged over masked pixels.
* ``lpips_vgg``   — LPIPS (VGG16 backbone, v0.1 linear heads) on the mask's bounding
                    box per sampled frame, with pixels outside the mask equalised so
                    only the fill is judged.
* ``warp_error``  — temporal consistency: the reference's own DIS optical flow warps
                    frame i+1 of the candidate onto frame i; mean abs error inside the
                    mask. The reference's own warp error is published as the floor.

Free-fill floor: the validator computes the per-pixel TEMPORAL MEDIAN of the unmasked
frames (a static-background fill any miner could run for free) and scores it with the
same metrics. A miner must beat that floor by ``removal_psnr_margin_db`` in PSNR, else
the item is zeroed (``REGION_BASELINE_NOT_BEATEN``): on footage where the background
never moves the median reconstructs the reference almost exactly and nothing is worth
paying for. LPIPS is a soft term only: a fill that does not beat the floor's LPIPS earns
``s_lpips = 0`` but keeps its PSNR share. (The sandbox showed why: on a slowly panning
1080p scene the median fill is a smeared but sharp-textured copy that LPIPS prefers by
0.05 while the inpaint is 10 dB better in PSNR — a hard AND zeroed the better output.)

Score (the audit-recompute record is :class:`RemovalBreakdown`)::

    s_psnr  = clamp((psnr - psnr_floor) / removal_psnr_span, 0, 1)
    s_lpips = clamp((lpips_floor - lpips) / lpips_floor, 0, 1)
    final   = w_psnr * s_psnr + w_lpips * s_lpips          (weights 0.6 / 0.4)

It reads as "how much better than the free fill", which is the only thing worth
rewarding. Warp error is a CAP (``removal_warp_cap_factor`` x the reference's own
warp error), never a term: a flat fill is perfectly consistent and must not be rewarded
for it.
"""

from __future__ import annotations

import math
import os
import time
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, field_validator

from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.finite import require_finite
from vidaio.scoring.media_inputs import UNTRUSTED_INPUT_ARGS
from vidaio.scoring.removal_formula import RegionMetrics, RemovalBreakdown, score_removal

# --- y4m access ------------------------------------------------------------------------


class Y4MError(ValueError):
    pass


@dataclass(frozen=True)
class Y4MReader:
    """Random frame access into a canonical 4:2:0 8-bit y4m file via memory map.

    The canonicalizer writes a fixed geometry, so every frame occupies the same number
    of bytes after a fixed ``FRAME\\n`` marker; frame ``i`` is one offset computation.
    """

    path: str
    width: int
    height: int
    frame_count: int
    header_bytes: int
    frame_bytes: int  # marker + planes

    @classmethod
    def open(cls, path: str) -> "Y4MReader":
        with open(path, "rb") as fh:
            header = fh.readline()
        if not header.startswith(b"YUV4MPEG2"):
            raise Y4MError(f"{path}: not a y4m file")
        fields = header.decode("ascii", "replace").split()
        w = h = None
        colour = "420"
        for f in fields[1:]:
            if f[0] == "W":
                w = int(f[1:])
            elif f[0] == "H":
                h = int(f[1:])
            elif f[0] == "C":
                colour = f[1:]
        if w is None or h is None:
            raise Y4MError(f"{path}: y4m header without W/H")
        if not colour.startswith("420") or "p10" in colour or "p16" in colour:
            raise Y4MError(f"{path}: unsupported y4m colourspace {colour!r} (need 8-bit 4:2:0)")
        if w % 2 or h % 2:
            raise Y4MError(f"{path}: odd geometry {w}x{h}")
        planes = w * h + 2 * (w // 2) * (h // 2)
        marker = len(b"FRAME\n")
        import os

        total = os.path.getsize(path) - len(header)
        if total % (marker + planes):
            raise Y4MError(f"{path}: file size is not a whole number of frames")
        return cls(
            path=path,
            width=w,
            height=h,
            frame_count=total // (marker + planes),
            header_bytes=len(header),
            frame_bytes=marker + planes,
        )

    def _mm(self) -> np.memmap:
        return np.memmap(self.path, dtype=np.uint8, mode="r")

    def frame_rgb(self, index: int) -> np.ndarray:
        """Frame ``index`` as (h, w, 3) uint8 RGB (BT.601 limited, as ffmpeg's I420->RGB)."""
        import cv2

        if not 0 <= index < self.frame_count:
            raise IndexError(index)
        mm = self._mm()
        start = self.header_bytes + index * self.frame_bytes + 6
        planes = mm[start : start + self.frame_bytes - 6]
        i420 = np.asarray(planes).reshape(self.height * 3 // 2, self.width)
        return cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)

    def frame_luma(self, index: int) -> np.ndarray:
        mm = self._mm()
        start = self.header_bytes + index * self.frame_bytes + 6
        return np.asarray(mm[start : start + self.width * self.height]).reshape(
            self.height, self.width
        )


class RemovalCancelled(RuntimeError):
    """The worker's deadline/cancellation fired while measuring."""


def ffprobe_for(ffmpeg_path: str) -> str:
    """The ffprobe next to a configured ffmpeg binary (never a string replace on the
    whole path: ``/opt/ffmpeg/bin/ffmpeg`` must not become ``/opt/ffprobe/bin/ffprobe``)."""
    directory, name = os.path.split(ffmpeg_path)
    if name == "ffmpeg" and directory:
        return os.path.join(directory, "ffprobe")
    if name.startswith("ffmpeg"):
        return os.path.join(directory, "ffprobe" + name[len("ffmpeg"):]) if directory else "ffprobe" + name[len("ffmpeg"):]
    return "ffprobe"


def decode_mask_stream(
    ffmpeg_path: str,
    container_path: str,
    *,
    stream_index: int = 1,
    timeout: float = 600.0,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    """The per-frame binary mask carried as video stream ``stream_index`` of the served
    input, as (n, h, w) bool. Any luma > 127 is "masked".

    The frames are streamed out of ffmpeg one at a time into a preallocated bool array,
    so the peak is one gray frame plus the bool array (not the whole raw stream twice).
    """
    probe = subprocess.run(
        [
            ffprobe_for(ffmpeg_path),
            "-v",
            "error",
            "-select_streams",
            f"v:{stream_index}",
            "-show_entries",
            "stream=width,height,nb_frames,nb_read_frames",
            "-of",
            "csv=p=0",
            *UNTRUSTED_INPUT_ARGS,
            container_path,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    m = re.match(r"\s*(\d+),(\d+)", probe.stdout or "")
    if probe.returncode != 0 or not m:
        raise Y4MError(f"{container_path}: no video stream {stream_index} (mask)")
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        raise Y4MError(f"{container_path}: mask stream {stream_index} has no geometry")
    frame_bytes = w * h
    # stderr goes to a file (a full pipe can never stall ffmpeg), the process gets its
    # own group so the worker's request scope can kill it, and a watchdog thread
    # enforces the deadline and cancellation even while a read is blocked.
    from vidaio.scoring.backends_real import current_process_scope, kill_process_group

    scope = current_process_scope()
    err_file = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        [
            ffmpeg_path,
            "-v",
            "error",
            "-nostdin",
            *UNTRUSTED_INPUT_ARGS,
            "-i",
            container_path,
            "-map",
            f"0:v:{stream_index}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=err_file,
        start_new_session=True,
    )
    assert proc.stdout is not None
    if scope is not None:
        scope.register(proc)
    frames: list[np.ndarray] = []
    deadline = time.monotonic() + timeout
    stop = threading.Event()
    stopped_for: list[str] = []

    def _watchdog() -> None:
        while not stop.wait(0.25):
            if cancelled is not None and cancelled():
                stopped_for.append("cancelled")
            elif time.monotonic() > deadline:
                stopped_for.append("timeout")
            else:
                continue
            kill_process_group(proc)
            return

    watchdog = threading.Thread(target=_watchdog, daemon=True)
    watchdog.start()
    try:
        while True:
            chunk = proc.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(h, w) > 127)
        proc.wait(timeout=30)
    finally:
        stop.set()
        if proc.poll() is None:
            kill_process_group(proc)
            proc.wait(timeout=10)
        if scope is not None:
            scope.unregister(proc)
    if stopped_for and stopped_for[0] == "cancelled":
        err_file.close()
        raise RemovalCancelled("mask decode cancelled")
    if stopped_for and stopped_for[0] == "timeout":
        err_file.close()
        raise Y4MError(f"{container_path}: mask decode exceeded {timeout:.0f}s")
    err_file.seek(0)
    err = err_file.read()[-2000:].decode("utf-8", "replace")
    err_file.close()
    if proc.returncode != 0:
        raise Y4MError(f"{container_path}: mask decode failed: {err[-300:]!r}")
    if not frames:
        return np.zeros((0, h, w), dtype=bool)
    return np.stack(frames, axis=0)


# --- metrics ---------------------------------------------------------------------------


def _bbox(mask: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    h, w = mask.shape
    return (
        max(0, int(ys.min()) - pad),
        min(h, int(ys.max()) + pad + 1),
        max(0, int(xs.min()) - pad),
        min(w, int(xs.max()) + pad + 1),
    )


def _bound_pair(a: np.ndarray, b: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Downscale both crops identically so the long side is <= max_side (0 = no bound).
    INTER_AREA is a fixed box filter, so the result is deterministic across hosts."""
    if max_side <= 0:
        return a, b
    h, w = a.shape[:2]
    long = max(h, w)
    if long <= max_side:
        return a, b
    import cv2

    scale = max_side / long
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(a, size, interpolation=cv2.INTER_AREA), cv2.resize(b, size, interpolation=cv2.INTER_AREA)


def _ssim_map(a_luma: np.ndarray, b_luma: np.ndarray) -> np.ndarray:
    """Standard SSIM (Gaussian 11x11, sigma 1.5, K1 .01, K2 .03) on 8-bit luma."""
    import cv2

    a = a_luma.astype(np.float64)
    b = b_luma.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    s_aa = blur(a * a) - mu_a * mu_a
    s_bb = blur(b * b) - mu_b * mu_b
    s_ab = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (s_aa + s_bb + c2)
    )


class LpipsVgg:
    """LPIPS v0.1 (VGG16 backbone) on CPU, deterministic, loaded once per process."""

    def __init__(self) -> None:
        import lpips  # type: ignore[import-not-found]
        import torch

        torch.set_grad_enabled(False)
        self._torch = torch
        self._net = lpips.LPIPS(net="vgg", verbose=False).eval()

    def distance(self, a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
        torch = self._torch
        ta = torch.from_numpy(np.ascontiguousarray(a_rgb)).permute(2, 0, 1)[None].float() / 127.5 - 1
        tb = torch.from_numpy(np.ascontiguousarray(b_rgb)).permute(2, 0, 1)[None].float() / 127.5 - 1
        with torch.no_grad():
            return float(self._net(ta, tb).item())


def region_metrics(
    candidate: Y4MReader,
    reference: Y4MReader,
    masks: np.ndarray,
    *,
    lpips: LpipsVgg | None,
    lpips_stride: int,
    flow_reference: Y4MReader | None = None,
    lpips_max_side: int = 0,
    lpips_offset: int = 0,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[RegionMetrics, float]:
    """Region-restricted metrics of `candidate` vs `reference` inside `masks`.

    Returns the metrics and the reference's own warp error (the temporal floor).
    ``flow_reference`` defaults to `reference`; the flow field is always computed on
    the reference so it is the same for every candidate of a round.
    """
    import cv2

    if (candidate.width, candidate.height) != (reference.width, reference.height):
        raise Y4MError(
            f"candidate geometry {candidate.width}x{candidate.height} != reference "
            f"{reference.width}x{reference.height}"
        )
    n = min(candidate.frame_count, reference.frame_count, len(masks))
    if n == 0:
        raise ValueError("no frames to score")
    if masks.shape[1:] != (reference.height, reference.width):
        raise ValueError(
            f"mask geometry {masks.shape[1:]} != reference {(reference.height, reference.width)}"
        )
    flow_src = flow_reference or reference
    stride = max(1, int(lpips_stride))
    offset = int(lpips_offset) % stride
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    # Everything expensive runs on the padded bounding box of the mask, never on the
    # full frame: PSNR/SSIM inside the mask are unchanged by the crop (the SSIM window
    # is 11 px, the pad 32), and the flow/warp terms are defined on the crop around
    # the region on both frames of a pair. At 1080p this is ~20x less work per frame.
    sq_err = 0.0
    count = 0
    ssim_vals: list[float] = []
    lpips_vals: list[float] = []
    warp_c: list[float] = []
    warp_r: list[float] = []
    prev_c = prev_r = None
    prev_luma = None
    masked_seen = 0
    first_pair: tuple[np.ndarray, np.ndarray] | None = None
    for i in range(n):
        if cancelled is not None and cancelled():
            raise RemovalCancelled("region metrics cancelled")
        m = masks[i]
        c = candidate.frame_rgb(i)
        r = reference.frame_rgb(i)
        if m.any():
            y0, y1, x0, x1 = _bbox(m, 32)
            cm = m[y0:y1, x0:x1]
            cc = c[y0:y1, x0:x1]
            cr = r[y0:y1, x0:x1]
            d = cc[cm].astype(np.float32) - cr[cm].astype(np.float32)
            sq_err += float((d * d).sum())
            count += int(d.size)
            smap = _ssim_map(cv2.cvtColor(cc, cv2.COLOR_RGB2GRAY), cv2.cvtColor(cr, cv2.COLOR_RGB2GRAY))
            ssim_vals.append(float(smap[cm].mean()))
            if lpips is not None:
                # every stride-th MASKED frame, phase-shifted by an offset the miner
                # cannot predict (derived from the held-out reference digest), so the
                # sampled frames cannot be worked harder than the rest
                if (masked_seen + offset) % stride == 0:
                    ca = cc.copy()
                    ca[~cm] = cr[~cm]
                    ca, cb = _bound_pair(ca, cr, lpips_max_side)
                    lpips_vals.append(lpips.distance(ca, cb))
                elif first_pair is None:
                    ca = cc.copy()
                    ca[~cm] = cr[~cm]
                    first_pair = _bound_pair(ca, cr, lpips_max_side)
            masked_seen += 1
        luma = flow_src.frame_luma(i)
        if prev_c is not None and prev_luma is not None:
            mm = masks[i - 1] & m
            if mm.any():
                y0, y1, x0, x1 = _bbox(masks[i - 1] | m, 48)
                # DIS needs contiguous inputs; a crop of a memory-mapped plane is not
                flow = dis.calc(
                    np.ascontiguousarray(prev_luma[y0:y1, x0:x1]),
                    np.ascontiguousarray(luma[y0:y1, x0:x1]),
                    None,
                )
                gy, gx = np.mgrid[0 : y1 - y0, 0 : x1 - x0].astype(np.float32)
                mx, my = gx + flow[..., 0], gy + flow[..., 1]
                wc = cv2.remap(c[y0:y1, x0:x1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                wr = cv2.remap(r[y0:y1, x0:x1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                sel = mm[y0:y1, x0:x1]
                warp_c.append(float(np.abs(wc.astype(np.float32) - prev_c[y0:y1, x0:x1].astype(np.float32))[sel].mean()))
                warp_r.append(float(np.abs(wr.astype(np.float32) - prev_r[y0:y1, x0:x1].astype(np.float32))[sel].mean()))
        prev_c, prev_r, prev_luma = c, r, luma
    if count == 0:
        raise ValueError("mask is empty on every frame")
    if lpips is not None and not lpips_vals and first_pair is not None:
        # fewer masked frames than the stride: the first masked frame is the sample
        lpips_vals.append(lpips.distance(*first_pair))
    mse = sq_err / count
    psnr = 10.0 * math.log10(255.0**2 / mse) if mse > 0 else 99.0
    return (
        RegionMetrics(
            psnr_db=float(min(psnr, 99.0)),
            ssim=float(np.mean(ssim_vals)),
            lpips_vgg=float(np.mean(lpips_vals)) if lpips_vals else float("nan"),
            warp_error=float(np.mean(warp_c)) if warp_c else 0.0,
            masked_pixels=count,
        ),
        float(np.mean(warp_r)) if warp_r else 0.0,
    )


def outside_region_change(
    candidate: Y4MReader,
    served_input: Y4MReader,
    masks: np.ndarray,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> float:
    """Max over frames of the mean abs RGB difference OUTSIDE the mask between the
    candidate and the served input (8-bit units). Honest fills leave this near zero."""
    import cv2

    if (candidate.width, candidate.height) != (served_input.width, served_input.height):
        raise Y4MError(
            f"candidate geometry {candidate.width}x{candidate.height} != served input "
            f"{served_input.width}x{served_input.height}"
        )
    n = min(candidate.frame_count, served_input.frame_count, len(masks))
    worst = 0.0
    for i in range(n):
        if cancelled is not None and cancelled():
            raise RemovalCancelled("outside-region check cancelled")
        om = (~masks[i]).astype(np.uint8)
        if not om.any():
            continue
        diff = cv2.absdiff(candidate.frame_rgb(i), served_input.frame_rgb(i))
        means = cv2.mean(diff, mask=om)  # per channel over the unmasked pixels
        worst = max(worst, float(sum(means[:3]) / 3.0))
    return worst


def temporal_median_fill(
    served_input: Y4MReader,
    masks: np.ndarray,
    out_path: str,
    *,
    band_rows: int = 48,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """The free fill: every masked pixel takes the per-pixel median of the frames where
    it is NOT masked; a pixel that is masked on every frame gets a deterministic spatial
    fill (Telea inpainting of the median image from its unmasked neighbours), never a
    constant. Written as a y4m of the same geometry so it scores through the same path
    as a miner output.

    The output is built in place through a memory map (the served frames are copied
    once, then only the rows inside the union of all masks are rewritten band by band),
    so memory stays at frames x band_rows x width regardless of clip length.
    """
    import cv2

    n = min(served_input.frame_count, len(masks))
    h, w = served_input.height, served_input.width
    mm = served_input._mm()
    plane = w * h
    # keep the source header verbatim (frame rate etc.) so probes agree
    with open(served_input.path, "rb") as fh:
        header = fh.readline()
    payload = served_input.frame_bytes - 6
    with open(out_path, "wb") as out:
        out.write(header)
        for i in range(n):
            if cancelled is not None and cancelled():
                raise RemovalCancelled("median fill cancelled")
            start = served_input.header_bytes + i * served_input.frame_bytes + 6
            out.write(b"FRAME\n")
            out.write(mm[start : start + payload])
    any_mask = masks[:n].any(axis=0)
    if n == 0 or not any_mask.any():
        return
    out_mm = np.memmap(out_path, dtype=np.uint8, mode="r+")

    def out_view(i: int, off: int, count: int) -> np.ndarray:
        base = len(header) + i * served_input.frame_bytes + 6 + off
        return out_mm[base : base + count]

    def spatial_fill(med: np.ndarray, hole: np.ndarray, fallback: int) -> np.ndarray:
        """Median image with the never-unmasked pixels inpainted from their neighbours."""
        if not hole.any():
            return med.astype(np.uint8)
        filled = np.where(hole, fallback, med).astype(np.uint8)
        if hole.all():
            return filled
        return cv2.inpaint(filled, hole.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    # Pass 1: per-pixel medians over the unmasked frames inside the union bbox (padded so
    # the spatial fill below has real neighbours), band by band to bound memory.
    ry0, ry1, rx0, rx1 = _bbox(any_mask, 16)
    ry0 -= ry0 % 2
    rx0 -= rx0 % 2
    ry1 += ry1 % 2
    rx1 += rx1 % 2
    ry1, rx1 = min(h, ry1), min(w, rx1)
    rh, cw = ry1 - ry0, rx1 - rx0
    cx0, cx1 = rx0 // 2, rx1 // 2
    med_full = np.zeros((rh, cw), np.float32)
    hole_full = np.zeros((rh, cw), bool)
    cmed_full = [np.zeros((rh // 2, cx1 - cx0), np.float32) for _ in range(2)]
    chole_full = [np.zeros((rh // 2, cx1 - cx0), bool) for _ in range(2)]
    for y0 in range(ry0, ry1, band_rows):
        if cancelled is not None and cancelled():
            raise RemovalCancelled("median fill cancelled")
        y1 = min(ry1, y0 + band_rows)
        y1 -= (y1 - y0) % 2
        if y1 <= y0:
            continue
        rows = y1 - y0
        band = np.empty((n, rows, cw), np.float32)
        for i in range(n):
            start = served_input.header_bytes + i * served_input.frame_bytes + 6 + y0 * w
            band[i] = np.asarray(mm[start : start + rows * w]).reshape(rows, w)[:, rx0:rx1]
        mband = masks[:n, y0:y1, rx0:rx1]
        band[mband] = np.nan
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(band, axis=0)
        del band
        med_full[y0 - ry0 : y1 - ry0] = np.nan_to_num(med, nan=0.0)
        hole_full[y0 - ry0 : y1 - ry0] = np.isnan(med)
        cy0, cy1 = y0 // 2, y1 // 2
        crow = cy1 - cy0
        cmask = mband[:, ::2, ::2][:, :crow, :]
        for plane_index in (0, 1):
            cband = np.empty((n, crow, cx1 - cx0), np.float32)
            off = plane + plane_index * (plane // 4) + cy0 * (w // 2)
            for i in range(n):
                start = served_input.header_bytes + i * served_input.frame_bytes + 6 + off
                cband[i] = np.asarray(mm[start : start + crow * (w // 2)]).reshape(crow, w // 2)[:, cx0:cx1]
            cband[cmask] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cmed = np.nanmedian(cband, axis=0)
            del cband
            cmed_full[plane_index][cy0 - ry0 // 2 : cy1 - ry0 // 2] = np.nan_to_num(cmed, nan=128.0)
            chole_full[plane_index][cy0 - ry0 // 2 : cy1 - ry0 // 2] = np.isnan(cmed)
    # Spatial fill of the pixels that were masked on every frame, from the whole padded
    # region so there is context on all sides (Telea inpainting: deterministic, CPU).
    med_u8 = spatial_fill(med_full, hole_full, 0)
    cmed_u8 = [spatial_fill(cmed_full[k], chole_full[k], 128) for k in range(2)]
    # Pass 2: write the fill into the masked pixels of every frame.
    try:
        for i in range(n):
            if cancelled is not None and cancelled():
                raise RemovalCancelled("median fill cancelled")
            sel = masks[i, ry0:ry1, rx0:rx1]
            if not sel.any():
                continue
            view = out_view(i, ry0 * w, rh * w).reshape(rh, w)[:, rx0:rx1]
            view[sel] = med_u8[sel]
            csel = sel[::2, ::2]
            for plane_index in (0, 1):
                off = plane + plane_index * (plane // 4) + (ry0 // 2) * (w // 2)
                cview = out_view(i, off, (rh // 2) * (w // 2)).reshape(rh // 2, w // 2)[:, cx0:cx1]
                cview[csel] = cmed_u8[plane_index][csel]
        out_mm.flush()
    finally:
        del out_mm
