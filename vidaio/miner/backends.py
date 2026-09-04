"""Track backends: plain-ffmpeg processing, subprocess-isolated and time-bounded.

Both backends write mp4 (`yuv420p`, `+faststart`) — universally decodable by the
scoring worker's backends, no exotic container games. Audio is dropped (`-an`)
to match the pipeline: challenge inputs never carry audio. The upscaling track
always encodes h264; the compression track honours the task's `target_codec` /
`codec_mode` / `target_bitrate` / `compression_type` through
`vidaio.miner.encoding` (h264 CRF at the configured default when absent).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from vidaio.miner.encoding import EncodeParamError, resolve_encode

#: Upscale factors the reference backend supports — must mirror the challenge
#: DAG's discrete UPSCALE_FACTORS (the miner repo can't import them after the split).
SUPPORTED_UPSCALE_FACTORS = (2, 4)


class BackendError(RuntimeError):
    """The backend could not produce an output (bad params, ffmpeg failure...)."""


def _target_dims(params: Mapping[str, float | int | str]) -> tuple[int, int] | None:
    """Optional output-dimension contract from the task params.

    The validator publishes target_width/target_height because scoring gates on
    exact reference dimensions and the challenge degradation is not exactly
    invertible from the input alone; an honest miner hits the stated contract.
    """
    w, h = params.get("target_width"), params.get("target_height")
    if w is None and h is None:
        return None
    try:
        return int(w), int(h)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise BackendError(f"invalid target dims: {w!r}x{h!r}") from None


class BackendTimeoutError(BackendError):
    """The ffmpeg subprocess exceeded its wall-clock bound."""


class MinerBackend(Protocol):
    """Media-transform seam shared by local ffmpeg and remote GPU workers.

    The public miner service owns authentication, replay protection, byte caps,
    deadlines and response signing. A backend only receives a verified local
    input and must atomically produce one output at the requested path.
    """

    def process(
        self,
        input_path: str,
        output_path: str,
        params: Mapping[str, float | int | str],
        *,
        timeout: float | None = None,
    ) -> None: ...


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _FfmpegBackend:
    def __init__(self, ffmpeg_path: str = "ffmpeg", timeout_seconds: float = 240.0) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.timeout_seconds = timeout_seconds

    def _run(self, args: list[str], timeout: float | None = None) -> None:
        ffmpeg = shutil.which(self.ffmpeg_path)
        if ffmpeg is None:
            raise BackendError(f"ffmpeg not found: {self.ffmpeg_path!r}")
        bound = self.timeout_seconds if timeout is None else min(timeout, self.timeout_seconds)
        cmd = [ffmpeg, "-y", "-hide_banner", *args]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=bound, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendTimeoutError(f"ffmpeg exceeded {bound}s") from exc
        if proc.returncode != 0:
            raise BackendError(
                f"ffmpeg failed (rc={proc.returncode}): {' '.join(cmd)}\n"
                f"stderr tail: {proc.stderr[-2000:]}"
            )

    def process(
        self,
        input_path: str,
        output_path: str,
        params: Mapping[str, float | int | str],
        *,
        timeout: float | None = None,
    ) -> None:
        raise NotImplementedError


class FfmpegCompressBackend(_FfmpegBackend):
    """Compression track: re-encode per the task's codec params (h264 CRF default)."""

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        timeout_seconds: float = 240.0,
        *,
        crf: int = 22,
        preset: str = "medium",
    ) -> None:
        super().__init__(ffmpeg_path, timeout_seconds)
        self.crf = crf
        self.preset = preset

    def process(
        self,
        input_path: str,
        output_path: str,
        params: Mapping[str, float | int | str],
        *,
        timeout: float | None = None,
    ) -> None:
        dims = _target_dims(params)
        scale = [] if dims is None else ["-vf", f"scale={dims[0]}:{dims[1]}:flags=lanczos"]
        try:
            plan = resolve_encode(params, default_crf=self.crf, preset=self.preset)
        except EncodeParamError as exc:
            raise BackendError(str(exc)) from None
        self._run(
            [
                "-i", input_path,
                *scale,
                *plan.ffmpeg_args,
                "-movflags", "+faststart",
                "-an",
                output_path,
            ],
            timeout,
        )


class FfmpegUpscaleBackend(_FfmpegBackend):
    """Upscaling track: lanczos scale by params['upscale_factor'] (2 or 4)."""

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        timeout_seconds: float = 240.0,
        *,
        crf: int = 16,
        preset: str = "medium",
    ) -> None:
        super().__init__(ffmpeg_path, timeout_seconds)
        self.crf = crf
        self.preset = preset

    def process(
        self,
        input_path: str,
        output_path: str,
        params: Mapping[str, float | int | str],
        *,
        timeout: float | None = None,
    ) -> None:
        raw = params.get("upscale_factor")
        try:
            factor = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise BackendError(f"missing/invalid upscale_factor param: {raw!r}") from None
        if factor not in SUPPORTED_UPSCALE_FACTORS:
            raise BackendError(
                f"unsupported upscale_factor {factor}; supported: {SUPPORTED_UPSCALE_FACTORS}"
            )
        dims = _target_dims(params)
        expr = (
            f"scale=iw*{factor}:ih*{factor}:flags=lanczos"
            if dims is None
            else f"scale={dims[0]}:{dims[1]}:flags=lanczos"
        )
        self._run(
            [
                "-i", input_path,
                "-vf", expr,
                "-c:v", "libx264",
                "-preset", self.preset,
                "-crf", str(self.crf),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                output_path,
            ],
            timeout,
        )
