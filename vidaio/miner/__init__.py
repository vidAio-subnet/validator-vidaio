"""Reference miner — the honest baseline every real miner is expected to beat.

Two ffmpeg-backed track backends (compression re-encode, lanczos upscale) behind
one small HTTP service speaking vidaio.services.protocol. No tricks: it verifies
the input digest it was handed, processes with plain ffmpeg, and returns the
output path + sha256 within the request deadline. The endpoint is public, so the
ingress is bounded on every axis an anonymous caller could push — task ids that
cannot express a path, a concurrency cap, an input-size cap, swept task dirs,
and an optional shared secret (see vidaio.miner.config). This code will be split into
its own repo — it must depend only on vidaio.services.protocol and vidaio.core.
"""

from vidaio.miner.backends import (
    BackendError,
    BackendTimeoutError,
    FfmpegCompressBackend,
    FfmpegUpscaleBackend,
    MinerBackend,
    sha256_file,
)
from vidaio.miner.config import MinerConfig
from vidaio.miner.remote_gpu import RemoteGpuBackend
from vidaio.miner.service import (
    TASK_ID_PATTERN,
    Miner,
    TaskDirEscape,
    create_app,
    resolve_task_dir,
    sweep_task_dirs,
)

__all__ = [
    "TASK_ID_PATTERN",
    "BackendError",
    "BackendTimeoutError",
    "FfmpegCompressBackend",
    "FfmpegUpscaleBackend",
    "Miner",
    "MinerBackend",
    "MinerConfig",
    "RemoteGpuBackend",
    "TaskDirEscape",
    "create_app",
    "resolve_task_dir",
    "sha256_file",
    "sweep_task_dirs",
]
