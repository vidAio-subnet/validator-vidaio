"""Deterministic chroma-plane PSNR over canonical yuv420p Y4M streams.

Source-proximity evidence (spec §18 anti-gaming): a compression output derived
from the *sealed pristine reference* instead of the served (degraded) input can
be told apart from an honest encode without any secret data, because the
scored VMAF model is luma-only and a score-driven encoder therefore has no
reason to move the chroma planes toward the pristine. The signal is

    chroma_residual = PSNR_UV(candidate, reference) - PSNR_UV(candidate, input)

measured on the SAME canonical y4m files the scorer already produced. An
encoder that only ever saw the input lands below zero (its chroma noise is
correlated with the input, not the pristine); an encode of the pristine lands
above it. The decision is population-relative and happens at round
composition (``vidaio.scoring.source_proximity_evidence``); this module only
measures.

Determinism: every plane difference is accumulated as an exact int64 sum of
squares (no float accumulation), one PSNR per frame from that integer MSE, and
the per-frame values are averaged in stream order. The auditor recomputes the
identical number from the identical canonical bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import BinaryIO, Callable

import numpy as np

#: PSNR assigned to a frame whose chroma planes are byte-identical (MSE = 0).
#: Finite by construction so packets stay JSON-representable and bounded.
IDENTICAL_PLANES_PSNR_DB = 100.0
#: Peak sample value of the canonical 8-bit planes.
_PEAK = 255.0
_Y4M_MAGIC = b"YUV4MPEG2"
_FRAME_MAGIC = b"FRAME"
_SUPPORTED_CHROMA = ("420", "420jpeg", "420mpeg2", "420paldv")


class PlanePsnrError(ValueError):
    """The two streams cannot be compared plane-wise (shape/format/length)."""


@dataclass(frozen=True)
class Y4mHeader:
    width: int
    height: int
    chroma: str

    @property
    def luma_bytes(self) -> int:
        return self.width * self.height

    @property
    def chroma_plane_bytes(self) -> int:
        # yuv420p: each chroma plane is ceil(w/2) x ceil(h/2) samples.
        return ((self.width + 1) // 2) * ((self.height + 1) // 2)

    @property
    def frame_bytes(self) -> int:
        return self.luma_bytes + 2 * self.chroma_plane_bytes


def read_y4m_header(handle: BinaryIO) -> Y4mHeader:
    """Parse the stream header line; only 8-bit 4:2:0 layouts are supported."""
    line = handle.readline()
    if not line.startswith(_Y4M_MAGIC):
        raise PlanePsnrError("not a YUV4MPEG2 stream")
    width = height = None
    chroma = "420jpeg"  # the y4m default when C is absent
    for token in line.strip().split(b" ")[1:]:
        if not token:
            continue
        tag, value = token[:1], token[1:].decode("ascii", "replace")
        if tag == b"W":
            width = int(value)
        elif tag == b"H":
            height = int(value)
        elif tag == b"C":
            chroma = value
    if width is None or height is None or width < 1 or height < 1:
        raise PlanePsnrError("y4m header lacks a positive W/H")
    if chroma not in _SUPPORTED_CHROMA:
        raise PlanePsnrError(f"unsupported y4m chroma layout {chroma!r} (need 8-bit 4:2:0)")
    return Y4mHeader(width=width, height=height, chroma=chroma)


def _read_frame(handle: BinaryIO, header: Y4mHeader) -> bytes | None:
    line = handle.readline()
    if not line:
        return None
    if not line.startswith(_FRAME_MAGIC):
        raise PlanePsnrError("y4m frame marker missing")
    data = handle.read(header.frame_bytes)
    if len(data) != header.frame_bytes:
        raise PlanePsnrError("truncated y4m frame")
    return data


def _chroma_sq_error(a: bytes, b: bytes, header: Y4mHeader) -> int:
    """Exact integer sum of squared differences over the U and V planes."""
    start = header.luma_bytes
    ua = np.frombuffer(a, dtype=np.uint8, count=2 * header.chroma_plane_bytes, offset=start)
    ub = np.frombuffer(b, dtype=np.uint8, count=2 * header.chroma_plane_bytes, offset=start)
    diff = ua.astype(np.int32) - ub.astype(np.int32)
    diff64 = diff.astype(np.int64)
    return int(np.dot(diff64, diff64))


def _frame_psnr(sq_error: int, samples: int) -> float:
    if sq_error == 0:
        return IDENTICAL_PLANES_PSNR_DB
    mse = sq_error / samples
    return min(IDENTICAL_PLANES_PSNR_DB, 10.0 * math.log10((_PEAK * _PEAK) / mse))


def chroma_psnr_uv(
    path_a: str,
    path_b: str,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> float:
    """Mean per-frame chroma (U+V) PSNR in dB between two canonical y4m streams.

    Both streams must share geometry, 4:2:0 layout and frame count — they are
    the scorer's own canonical outputs, so a mismatch is a caller error and is
    raised, never silently truncated.
    """
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        ha, hb = read_y4m_header(fa), read_y4m_header(fb)
        if (ha.width, ha.height) != (hb.width, hb.height):
            raise PlanePsnrError(
                f"geometry differs: {ha.width}x{ha.height} vs {hb.width}x{hb.height}"
            )
        samples = 2 * ha.chroma_plane_bytes
        total = 0.0
        frames = 0
        while True:
            if cancelled is not None and cancelled():
                raise PlanePsnrError("chroma PSNR cancelled")
            fa_frame = _read_frame(fa, ha)
            fb_frame = _read_frame(fb, hb)
            if fa_frame is None and fb_frame is None:
                break
            if fa_frame is None or fb_frame is None:
                raise PlanePsnrError("frame count differs between the two streams")
            total += _frame_psnr(_chroma_sq_error(fa_frame, fb_frame, ha), samples)
            frames += 1
    if frames == 0:
        raise PlanePsnrError("streams contain no frames")
    return total / frames


@dataclass(frozen=True)
class ChromaResidual:
    """The two chroma PSNRs and their difference, all in dB."""

    psnr_uv_reference: float
    psnr_uv_input: float

    @property
    def residual(self) -> float:
        return self.psnr_uv_reference - self.psnr_uv_input


def chroma_residual(
    candidate: str,
    reference: str,
    miner_input: str,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> ChromaResidual:
    """``PSNR_UV(candidate, reference) - PSNR_UV(candidate, miner_input)``."""
    return ChromaResidual(
        psnr_uv_reference=chroma_psnr_uv(candidate, reference, cancelled=cancelled),
        psnr_uv_input=chroma_psnr_uv(candidate, miner_input, cancelled=cancelled),
    )
