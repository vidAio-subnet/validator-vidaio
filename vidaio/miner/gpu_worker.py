"""GPU media transform used by the fresh Modal miner deployment.

This is intentionally a *miner solution*, not a scorer. It transforms frames on
CUDA, encodes an ordinary H.264 MP4, and returns those exact bytes to the signed
miner ingress. The existing CPU scoring worker and CPU auditor are the only
economic measurement path.

Imports of torch/numpy are lazy so the normal CPU validator/miner release can
load this module without acquiring a CUDA dependency. The Modal image supplies
the pinned GPU packages.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import logging
import math
import os
import queue
import re
import selectors
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from vidaio.miner.encoding import EncodeParamError, resolve_encode
from vidaio.miner.premium import (
    DEVICE_PREFIX,
    PremiumEncodeError,
    run_premium_compression,
)
from vidaio.miner.remote_gpu import (
    CPU_FALLBACK_DEVICE,
    GPU_PROTOCOL_VERSION,
    SUPPORTED_GPU_VARIANTS,
)
from vidaio.scoring.canonicalize import build_canonicalization_plan

LOGGER = logging.getLogger(__name__)

MAX_GPU_METADATA_BYTES = 16 * 1024
SUPPORTED_TRACKS = ("compression", "upscaling")
SUPPORTED_VARIANTS = SUPPORTED_GPU_VARIANTS
SUPPORTED_UPSCALE_FACTORS = (2, 4)
#: Frame bound per request unless the deployment raises it (product clips run longer
#: than inference clips did).
DEFAULT_MAXIMUM_FRAMES = 3600
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class GpuTaskMetadata(BaseModel):
    protocol: str
    track: str
    solution_variant: str
    input_digest: str
    input_size: int = Field(gt=0)
    # Product clips can need more than the inference-era 10 minutes; the Modal function
    # timeout (at most 3600 s) still bounds every request.
    deadline_seconds: float = Field(gt=0, le=3600)
    params: dict[str, float | int | str] = Field(default_factory=dict)

    @field_validator("protocol")
    @classmethod
    def _protocol(cls, value: str) -> str:
        if value != GPU_PROTOCOL_VERSION:
            raise ValueError(f"unsupported GPU protocol {value!r}")
        return value

    @field_validator("track")
    @classmethod
    def _track(cls, value: str) -> str:
        if value not in SUPPORTED_TRACKS:
            raise ValueError(f"unsupported track {value!r}")
        return value

    @field_validator("solution_variant")
    @classmethod
    def _variant(cls, value: str) -> str:
        if value not in SUPPORTED_VARIANTS:
            raise ValueError(f"unsupported solution variant {value!r}")
        return value

    @field_validator("input_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not SHA256_HEX.fullmatch(value):
            raise ValueError("input_digest must be lowercase sha256 hex")
        return value

    @model_validator(mode="after")
    def _params_are_bounded(self) -> GpuTaskMetadata:
        if len(self.params) > 32:
            raise ValueError("too many transform parameters")
        encoded = json.dumps(self.params, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 8 * 1024:
            raise ValueError("transform parameters are too large")
        return self


def decode_gpu_metadata(value: str | None) -> GpuTaskMetadata:
    if not value:
        raise ValueError("missing GPU task metadata")
    max_encoded = ((MAX_GPU_METADATA_BYTES + 2) // 3) * 4
    if len(value) > max_encoded:
        raise ValueError("encoded GPU task metadata is too large")
    try:
        encoded = value.encode("ascii")
        raw = base64.b64decode(
            encoded + b"=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ValueError("GPU task metadata is not valid base64url") from exc
    if len(raw) > MAX_GPU_METADATA_BYTES:
        raise ValueError("GPU task metadata is too large")
    try:
        return GpuTaskMetadata.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError(f"invalid GPU task metadata: {exc}") from exc


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class TransformResult:
    output_path: Path
    frames: int
    width: int
    height: int
    device: str
    gpu_seconds: float


@dataclass(frozen=True)
class _Variant:
    compression_crf: int
    upscaling_crf: int
    preset: str
    compression_detail: float
    upscale_detail: float
    upscale_mode: str


_VARIANTS = {
    # These are examples, not protocol constants. The deliberately different
    # rate/detail trade-offs make ranking and output-digest dedup visible when
    # three fresh miner hotkeys are run in a testnet soak.
    # CRF 22 is the calibrated launch-quality example; the lower-ranked examples
    # remain deliberately distinct so ranking is observable.
    "quality": _Variant(22, 14, "medium", 0.10, 0.12, "bicubic"),
    "balanced": _Variant(28, 17, "medium", 0.04, 0.06, "bicubic"),
    "compact": _Variant(31, 20, "fast", -0.04, 0.00, "bilinear"),
    # The premium serving profile (P1.1a): compression runs the ab-av1
    # VMAF-target search (vidaio/miner/premium.py) instead of this row's CRF;
    # the row still drives the (for now reference-grade) upscaling path.
    "premium": _Variant(22, 14, "medium", 0.10, 0.12, "bicubic"),
}


class GpuTransformError(RuntimeError):
    pass


class RecoverableGpuTransformError(GpuTransformError):
    """A bounded raw-frame pipeline limitation, not invalid media or a CUDA fault."""


def _remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise GpuTransformError("GPU transform deadline exceeded")
    return value


def _run_capture(command: list[str], *, deadline: float) -> bytes:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=_remaining(deadline),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GpuTransformError(f"media tool exceeded deadline: {command[0]}") from exc
    if completed.returncode != 0:
        detail = completed.stderr[-2048:].decode("utf-8", errors="replace")
        raise GpuTransformError(
            f"media tool failed rc={completed.returncode}: {detail.strip()}"
        )
    return completed.stdout


def _probe(
    path: Path, *, ffprobe_path: str, deadline: float,
    maximum_frames: int = DEFAULT_MAXIMUM_FRAMES,
) -> VideoInfo:
    raw = _run_capture(
        [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        deadline=deadline,
    )
    try:
        stream = json.loads(raw)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        numerator, denominator = stream["avg_frame_rate"].split("/", 1)
        fps = int(numerator) / int(denominator)
        frame_count = int(stream["nb_read_frames"])
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise GpuTransformError("could not derive bounded video geometry") from exc
    if not (1 <= width <= 8192 and 1 <= height <= 8192):
        raise GpuTransformError(f"input dimensions are outside bounds: {width}x{height}")
    if not (math.isfinite(fps) and 0 < fps <= 120):
        raise GpuTransformError(f"input frame rate is outside bounds: {fps}")
    if not (1 <= frame_count <= maximum_frames):
        raise GpuTransformError(f"input frame count is outside bounds: {frame_count}")
    return VideoInfo(width, height, fps, frame_count)


def _positive_int(params: Mapping[str, object], name: str) -> int | None:
    raw = params.get(name)
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise GpuTransformError(f"{name} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise GpuTransformError(f"{name} must be an integer") from exc
    if str(value) != str(raw) and not isinstance(raw, int):
        try:
            if float(raw) != value:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise GpuTransformError(f"{name} must be an integer") from exc
    if value <= 0:
        raise GpuTransformError(f"{name} must be positive")
    return value


def _cfr_info(
    input_path: Path, info: VideoInfo, *, ffmpeg_path: str, deadline: float,
    maximum_frames: int = DEFAULT_MAXIMUM_FRAMES,
) -> VideoInfo:
    """Count the scorer's CFR timeline without allocating a whole raw clip."""
    command = build_canonicalization_plan(str(input_path), "pipe:1")
    command[0] = ffmpeg_path
    command[-1:-1] = [
        "-frames:v", str(maximum_frames + 1), "-c:v", "wrapped_avframe", "-f", "framecrc",
    ]
    lines = _run_capture(command, deadline=deadline).decode("ascii").splitlines()
    timebases = [line.split(":", 1)[1].strip() for line in lines if line.startswith("#tb 0:")]
    frames = [line for line in lines if line and not line.startswith("#")]
    try:
        if len(timebases) != 1 or not 1 <= len(frames) <= maximum_frames:
            raise ValueError("unbounded CFR frame count or missing timebase")
        fps = float(1 / Fraction(timebases[0]))
        if not math.isfinite(fps) or not 0 < fps <= 120:
            raise ValueError("unbounded CFR frame rate")
        for index, line in enumerate(frames):
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 6 or [int(value) for value in fields[:4]] != [0, index, index, 1]:
                raise ValueError("nonuniform canonical frame timeline")
    except (ValueError, ZeroDivisionError) as exc:
        raise GpuTransformError(f"could not derive bounded CFR geometry: {exc}") from exc
    return VideoInfo(info.width, info.height, fps, len(frames))


def _target(info: VideoInfo, metadata: GpuTaskMetadata) -> tuple[int, int]:
    width = _positive_int(metadata.params, "target_width")
    height = _positive_int(metadata.params, "target_height")
    if (width is None) != (height is None):
        raise GpuTransformError("target_width and target_height must appear together")
    if width is not None and height is not None:
        target = (width, height)
    elif metadata.track == "upscaling":
        factor = _positive_int(metadata.params, "upscale_factor")
        if factor not in SUPPORTED_UPSCALE_FACTORS:
            raise GpuTransformError(
                f"upscale_factor must be one of {SUPPORTED_UPSCALE_FACTORS}"
            )
        target = (info.width * factor, info.height * factor)
    else:
        target = (info.width, info.height)
    width, height = target
    if width > 8192 or height > 8192 or width * height > 33_554_432:
        raise GpuTransformError(f"target dimensions are outside bounds: {width}x{height}")
    if width % 2 or height % 2:
        raise GpuTransformError("H.264 yuv420p target dimensions must be even")
    return target


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _decode_bounded(
    input_path: Path,
    raw_path: Path,
    *,
    ffmpeg_path: str,
    maximum_bytes: int,
    deadline: float,
) -> int:
    """Decode stdout while concurrently draining stderr, with a hard byte cap."""
    command = build_canonicalization_plan(str(input_path), "pipe:1")
    command[0] = ffmpeg_path
    command[-1:-1] = ["-f", "rawvideo", "-pix_fmt", "rgb24"]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    total = 0
    stderr_tail = bytearray()
    try:
        with raw_path.open("xb") as sink:
            while selector.get_map():
                wait = min(0.5, _remaining(deadline))
                for key, _mask in selector.select(wait):
                    chunk = os.read(key.fileobj.fileno(), 1 << 20)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stderr":
                        stderr_tail.extend(chunk)
                        del stderr_tail[:-8192]
                        continue
                    total += len(chunk)
                    if total > maximum_bytes:
                        raise GpuTransformError(
                            f"decoded video crossed {maximum_bytes} bytes"
                        )
                    sink.write(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        returncode = process.wait(timeout=_remaining(deadline))
        if returncode != 0:
            detail = bytes(stderr_tail).decode("utf-8", errors="replace").strip()
            raise GpuTransformError(f"ffmpeg decode failed rc={returncode}: {detail}")
        return total
    except BaseException:
        _kill_group(process)
        raw_path.unlink(missing_ok=True)
        raise
    finally:
        selector.close()


def _detail_filter(tensor, amount: float):  # type: ignore[no-untyped-def]
    import torch
    from torch.nn import functional

    # reflect-pad + average pool is deterministic and cheap. Positive amounts
    # unsharp-mask; the compact profile applies a tiny denoise blend instead.
    blurred = functional.avg_pool2d(
        functional.pad(tensor, (1, 1, 1, 1), mode="reflect"), 3, stride=1
    )
    if amount >= 0:
        return torch.clamp(tensor + amount * (tensor - blurred), 0.0, 1.0)
    return torch.clamp(tensor * (1.0 + amount) - amount * blurred, 0.0, 1.0)


def _cuda_device(*, recoverable: bool = False, context: str = ""):  # type: ignore[no-untyped-def]
    """The CUDA device and its attested label.

    The spool path refuses to run without CUDA (CPU fallback is forbidden there). The
    streaming path only serves clips the spool path could never take, which used to go
    straight to the CPU fallback, so for it a missing GPU stays recoverable.
    """
    import torch

    if not torch.cuda.is_available():
        if recoverable:
            prefix = f"{context}; " if context else ""
            raise RecoverableGpuTransformError(
                f"{prefix}CUDA is unavailable for the streaming GPU path"
            )
        raise GpuTransformError("CUDA is unavailable; CPU fallback is forbidden")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    device_name = str(torch.cuda.get_device_name(device)).replace("\r", " ").replace(
        "\n", " "
    )
    return device, f"cuda:0/{device_name}"


# Batches in flight between the decoder, the GPU and the encoder. Two per hop keeps
# all three busy while memory stays a few batches whatever the clip length.
_STREAM_QUEUE_DEPTH = 2
_STREAM_END = object()


def _batch_size(info: VideoInfo, target: tuple[int, int]) -> int:
    # Dynamic batching keeps an 8K request from allocating the whole clip on GPU.
    pixels = max(info.width * info.height, target[0] * target[1])
    return max(1, min(8, 8_388_608 // pixels))


def _transform_batch(host, *, device, metadata: GpuTaskMetadata, info: VideoInfo, target: tuple[int, int]):  # type: ignore[no-untyped-def]
    """One batch of RGB24 frames (n, h, w, 3) through the CUDA transform, as uint8."""
    import torch
    from torch.nn import functional

    variant = _VARIANTS[metadata.solution_variant]
    target_width, target_height = target
    tensor = (
        torch.from_numpy(host)
        .to(device=device, dtype=torch.float32, non_blocking=False)
        .permute(0, 3, 1, 2)
        / 255.0
    )
    if metadata.track == "upscaling":
        tensor = functional.interpolate(
            tensor,
            size=(target_height, target_width),
            mode=variant.upscale_mode,
            align_corners=False,
            antialias=variant.upscale_mode == "bicubic",
        )
        tensor = _detail_filter(tensor, variant.upscale_detail)
    else:
        if (info.width, info.height) != (target_width, target_height):
            tensor = functional.interpolate(
                tensor,
                size=(target_height, target_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        tensor = _detail_filter(tensor, variant.compression_detail)
    return (
        torch.round(tensor * 255.0)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def _gpu_frames(
    source_raw: Path,
    transformed_raw: Path,
    *,
    info: VideoInfo,
    target: tuple[int, int],
    metadata: GpuTaskMetadata,
    maximum_output_raw_bytes: int,
    deadline: float,
) -> tuple[int, str, float]:
    import numpy as np
    import torch

    device, device_label = _cuda_device()
    frame_bytes = info.width * info.height * 3
    raw_size = source_raw.stat().st_size
    if raw_size % frame_bytes:
        raise GpuTransformError("decoded frame stream ended mid-frame")
    frames = raw_size // frame_bytes
    if frames != info.frame_count:
        raise GpuTransformError(
            f"decoded {frames} frames but ffprobe committed {info.frame_count}"
        )
    target_width, target_height = target
    output_bytes = frames * target_width * target_height * 3
    if output_bytes > maximum_output_raw_bytes:
        raise RecoverableGpuTransformError(
            f"transformed raw video projects to {output_bytes} bytes, cap is "
            f"{maximum_output_raw_bytes}"
        )
    source = np.memmap(
        source_raw, dtype=np.uint8, mode="r", shape=(frames, info.height, info.width, 3)
    )
    output = np.memmap(
        transformed_raw,
        dtype=np.uint8,
        mode="w+",
        shape=(frames, target_height, target_width, 3),
    )
    gpu_started = time.perf_counter()
    batch_size = _batch_size(info, target)
    try:
        for start in range(0, frames, batch_size):
            _remaining(deadline)
            end = min(frames, start + batch_size)
            host = np.asarray(source[start:end]).copy()
            output[start:end] = _transform_batch(
                host, device=device, metadata=metadata, info=info, target=target
            )
        torch.cuda.synchronize(device)
        output.flush()
    finally:
        del output
        del source
    return frames, device_label, time.perf_counter() - gpu_started


def _encode(
    raw_path: Path,
    output_path: Path,
    *,
    info: VideoInfo,
    target: tuple[int, int],
    metadata: GpuTaskMetadata,
    ffmpeg_path: str,
    deadline: float,
) -> None:
    _run_capture(
        _encode_command(
            str(raw_path), output_path, info=info, target=target,
            metadata=metadata, ffmpeg_path=ffmpeg_path,
        ),
        deadline=deadline,
    )


def _encode_command(
    source: str,
    output_path: Path,
    *,
    info: VideoInfo,
    target: tuple[int, int],
    metadata: GpuTaskMetadata,
    ffmpeg_path: str,
) -> list[str]:
    """Encode raw RGB24 frames from `source` (a spool file or ``pipe:0``)."""
    variant = _VARIANTS[metadata.solution_variant]
    if metadata.track == "compression":
        try:
            plan = resolve_encode(
                metadata.params, default_crf=variant.compression_crf, preset=variant.preset
            )
        except EncodeParamError as exc:
            raise GpuTransformError(str(exc)) from None
        codec_args = plan.ffmpeg_args
    else:
        codec_args = [
            "-c:v", "libx264",
            "-preset", variant.preset,
            "-crf", str(variant.upscaling_crf),
            "-pix_fmt", "yuv420p",
        ]
    width, height = target
    return [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        format(info.fps, ".12g"),
        "-i",
        source,
        *codec_args,
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]


def _read_up_to(stream, size: int) -> bytes:  # type: ignore[no-untyped-def]
    """Read `size` bytes; fewer only at the end of the stream."""
    parts: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _drain_tail(stream, tail: bytearray) -> None:  # type: ignore[no-untyped-def]
    for chunk in iter(lambda: stream.read(65536), b""):
        tail.extend(chunk)
        del tail[:-8192]


def _gpu_stream(
    input_path: Path,
    output_path: Path,
    *,
    info: VideoInfo,
    target: tuple[int, int],
    metadata: GpuTaskMetadata,
    ffmpeg_path: str,
    deadline: float,
    reason: str = "",
) -> tuple[int, str, float]:
    """Decode -> CUDA transform -> encode as one pipeline, for clips too long to spool.

    Frames flow through in small batches, so memory and disk stay bounded whatever
    the clip length, and the encoder sees one continuous stream (one rate-control
    pass, no segment joins). Reading, the transform and writing run on separate
    threads with bounded queues between them, so decode, GPU and encode overlap
    instead of taking turns. A CUDA fault is recoverable (CPU fallback); a decode or
    encode failure means the media itself is the problem.
    """
    import numpy as np

    device, device_label = _cuda_device(recoverable=True, context=reason)
    decode = build_canonicalization_plan(str(input_path), "pipe:1")
    decode[0] = ffmpeg_path
    decode[-1:-1] = ["-f", "rawvideo", "-pix_fmt", "rgb24"]
    encode = _encode_command(
        "pipe:0", output_path, info=info, target=target,
        metadata=metadata, ffmpeg_path=ffmpeg_path,
    )
    frame_bytes = info.width * info.height * 3
    batch = _batch_size(info, target)
    budget = _remaining(deadline)
    decoder = subprocess.Popen(
        decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    encoder = subprocess.Popen(
        encode, stdin=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    assert decoder.stdout is not None and decoder.stderr is not None
    assert encoder.stdin is not None and encoder.stderr is not None
    decode_tail, encode_tail = bytearray(), bytearray()
    drains = [
        threading.Thread(target=_drain_tail, args=(decoder.stderr, decode_tail), daemon=True),
        threading.Thread(target=_drain_tail, args=(encoder.stderr, encode_tail), daemon=True),
    ]
    for drain in drains:
        drain.start()
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        _kill_group(decoder)
        _kill_group(encoder)

    watchdog = threading.Timer(budget, expire)
    watchdog.daemon = True
    watchdog.start()
    stop = threading.Event()
    decoded: queue.Queue = queue.Queue(maxsize=_STREAM_QUEUE_DEPTH)
    transformed_batches: queue.Queue = queue.Queue(maxsize=_STREAM_QUEUE_DEPTH)
    write_failures: list[BaseException] = []

    def put(target_queue: queue.Queue, item: object) -> bool:
        while not stop.is_set():
            try:
                target_queue.put(item, timeout=0.25)
                return True
            except queue.Full:
                continue
        return False

    def get(source_queue: queue.Queue) -> object:
        while True:
            try:
                return source_queue.get(timeout=0.25)
            except queue.Empty:
                if stop.is_set():
                    return _STREAM_END

    def read_batches() -> None:
        decoded_frames = 0
        try:
            while not stop.is_set():
                data = _read_up_to(decoder.stdout, frame_bytes * batch)
                if not data:
                    break
                if len(data) % frame_bytes:
                    raise GpuTransformError("decoded frame stream ended mid-frame")
                decoded_frames += len(data) // frame_bytes
                if decoded_frames > info.frame_count:
                    raise GpuTransformError(
                        f"decoded more than the {info.frame_count} committed frames"
                    )
                if not put(decoded, data):
                    return
            put(decoded, _STREAM_END)
        except BaseException as exc:  # handed to the transform loop, which raises it
            put(decoded, exc)

    def write_batches() -> None:
        try:
            while True:
                item = get(transformed_batches)
                if item is _STREAM_END:
                    break
                encoder.stdin.write(item)
            if not stop.is_set():
                encoder.stdin.close()
        except BaseException as exc:  # BrokenPipeError when the encoder exits early
            write_failures.append(exc)
            stop.set()

    reader = threading.Thread(target=read_batches, daemon=True)
    writer = threading.Thread(target=write_batches, daemon=True)
    reader.start()
    writer.start()
    frames = 0
    gpu_seconds = 0.0
    try:
        while True:
            item = get(decoded)
            if item is _STREAM_END:
                break
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, bytes)
            count = len(item) // frame_bytes
            frames += count
            host = np.frombuffer(item, dtype=np.uint8).reshape(
                count, info.height, info.width, 3
            ).copy()
            started = time.perf_counter()
            try:
                transformed = _transform_batch(
                    host, device=device, metadata=metadata, info=info, target=target
                )
            except Exception as exc:  # a CUDA/torch fault mid-stream
                raise RecoverableGpuTransformError(
                    f"{reason}; GPU stream transform failed: {type(exc).__name__}: {exc}"
                ) from exc
            gpu_seconds += time.perf_counter() - started
            if not put(transformed_batches, transformed.tobytes()):
                break
        put(transformed_batches, _STREAM_END)
        writer.join(timeout=_remaining(deadline))
        if write_failures or writer.is_alive():
            if expired.is_set() or writer.is_alive():
                raise GpuTransformError("GPU transform deadline exceeded")
            with contextlib.suppress(subprocess.TimeoutExpired):
                encoder.wait(timeout=5)
            drains[1].join(timeout=5)
            raise GpuTransformError("ffmpeg encode stopped early: " + bytes(encode_tail).decode(
                "utf-8", errors="replace").strip()) from write_failures[0]
        decode_rc = decoder.wait(timeout=_remaining(deadline))
        encode_rc = encoder.wait(timeout=_remaining(deadline))
        if expired.is_set():
            raise GpuTransformError("GPU transform deadline exceeded")
        if decode_rc != 0:
            detail = bytes(decode_tail).decode("utf-8", errors="replace").strip()
            raise GpuTransformError(f"ffmpeg decode failed rc={decode_rc}: {detail}")
        if encode_rc != 0:
            detail = bytes(encode_tail).decode("utf-8", errors="replace").strip()
            raise GpuTransformError(f"ffmpeg encode failed rc={encode_rc}: {detail}")
        if frames != info.frame_count:
            raise GpuTransformError(
                f"decoded {frames} frames but ffprobe committed {info.frame_count}"
            )
        return frames, device_label, gpu_seconds
    finally:
        stop.set()
        watchdog.cancel()
        _kill_group(decoder)
        _kill_group(encoder)
        for thread in (reader, writer, *drains):
            thread.join(timeout=5)
        for pipe in (decoder.stdout, decoder.stderr, encoder.stdin, encoder.stderr):
            try:
                pipe.close()
            except (BrokenPipeError, OSError):
                pass


def transform_media(
    input_path: Path,
    output_path: Path,
    metadata: GpuTaskMetadata,
    *,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
    maximum_raw_bytes: int = 4 * 1024 * 1024 * 1024,
    maximum_output_bytes: int = 4 * 1024 * 1024 * 1024,
    allow_cpu_fallback: bool = False,
    maximum_frames: int = DEFAULT_MAXIMUM_FRAMES,
) -> TransformResult:
    """Bounded decode -> CUDA transform -> software encode per the task codec params.

    The `premium` variant's compression track never enters the raw-frames
    pipeline: ab-av1 searches for the smallest target-clearing encode of the
    input file directly, and the result attests `abav1:<encoder>` (CPU work —
    claiming CUDA here would violate the no-false-GPU rule).
    """
    deadline = time.monotonic() + metadata.deadline_seconds
    if metadata.solution_variant == "premium" and metadata.track == "compression":
        try:
            encoder = run_premium_compression(
                input_path,
                output_path,
                params=metadata.params,
                deadline=deadline,
                maximum_output_bytes=maximum_output_bytes,
                ffmpeg_path=ffmpeg_path,
            )
        except PremiumEncodeError as exc:
            raise GpuTransformError(str(exc)) from None
        out_info = _probe(
            output_path, ffprobe_path=ffprobe_path, deadline=deadline,
            maximum_frames=maximum_frames,
        )
        return TransformResult(
            output_path=output_path,
            frames=out_info.frame_count,
            width=out_info.width,
            height=out_info.height,
            device=f"{DEVICE_PREFIX}{encoder}",
            gpu_seconds=0.0,
        )
    info = _probe(
        input_path, ffprobe_path=ffprobe_path, deadline=deadline,
        maximum_frames=maximum_frames,
    )
    target = _target(info, metadata)
    info = _cfr_info(
        input_path, info, ffmpeg_path=ffmpeg_path, deadline=deadline,
        maximum_frames=maximum_frames,
    )
    input_raw = output_path.with_suffix(".input.rgb")
    transformed_raw = output_path.with_suffix(".gpu.rgb")
    expected_input_raw = info.frame_count * info.width * info.height * 3
    expected_output_raw = info.frame_count * target[0] * target[1] * 3
    try:
        if max(expected_input_raw, expected_output_raw) > maximum_raw_bytes:
            # Too long to spool as raw frames: stream them through the GPU instead.
            reason = (
                f"decoded input projects to {expected_input_raw} bytes, cap is "
                f"{maximum_raw_bytes}"
                if expected_input_raw > maximum_raw_bytes
                else f"transformed raw video projects to {expected_output_raw} bytes, "
                f"cap is {maximum_raw_bytes}"
            )
            LOGGER.info(
                "GPU streaming path track=%s variant=%s input=%s frames=%s reason=%s",
                metadata.track, metadata.solution_variant, metadata.input_digest[:12],
                info.frame_count, reason,
            )
            frames, device, gpu_seconds = _gpu_stream(
                input_path,
                output_path,
                info=info,
                target=target,
                metadata=metadata,
                ffmpeg_path=ffmpeg_path,
                deadline=deadline,
                reason=reason,
            )
        else:
            decoded = _decode_bounded(
                input_path,
                input_raw,
                ffmpeg_path=ffmpeg_path,
                maximum_bytes=maximum_raw_bytes,
                deadline=deadline,
            )
            if decoded != expected_input_raw:
                raise RecoverableGpuTransformError(
                    f"decoded byte count {decoded} != projected {expected_input_raw}"
                )
            frames, device, gpu_seconds = _gpu_frames(
                input_raw,
                transformed_raw,
                info=info,
                target=target,
                metadata=metadata,
                maximum_output_raw_bytes=maximum_raw_bytes,
                deadline=deadline,
            )
            _encode(
                transformed_raw,
                output_path,
                info=info,
                target=target,
                metadata=metadata,
                ffmpeg_path=ffmpeg_path,
                deadline=deadline,
            )
        if not output_path.is_file() or output_path.stat().st_size < 1:
            raise GpuTransformError("encoder produced no output")
        if output_path.stat().st_size > maximum_output_bytes:
            raise GpuTransformError(
                f"encoded output crossed {maximum_output_bytes} bytes"
            )
        return TransformResult(
            output_path=output_path,
            frames=frames,
            width=target[0],
            height=target[1],
            device=device,
            gpu_seconds=gpu_seconds,
        )
    except RecoverableGpuTransformError as exc:
        output_path.unlink(missing_ok=True)
        if not allow_cpu_fallback:
            raise
        _remaining(deadline)
        input_raw.unlink(missing_ok=True)
        transformed_raw.unlink(missing_ok=True)
        LOGGER.warning(
            "GPU raw pipeline fallback track=%s variant=%s input=%s reason=%s",
            metadata.track, metadata.solution_variant, metadata.input_digest[:12], exc,
        )
        return _stream_cpu_fallback(
            input_path, output_path, metadata, info=info, target=target,
            ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path,
            maximum_output_bytes=maximum_output_bytes, deadline=deadline,
            maximum_frames=maximum_frames,
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        input_raw.unlink(missing_ok=True)
        transformed_raw.unlink(missing_ok=True)


def _stream_cpu_fallback(
    input_path: Path,
    output_path: Path,
    metadata: GpuTaskMetadata,
    *,
    info: VideoInfo,
    target: tuple[int, int],
    ffmpeg_path: str,
    ffprobe_path: str,
    maximum_output_bytes: int,
    deadline: float,
    maximum_frames: int = DEFAULT_MAXIMUM_FRAMES,
) -> TransformResult:
    """Stream validated media without raw spools, under the original deadline."""
    variant = _VARIANTS[metadata.solution_variant]
    width, height = target
    scale_mode = variant.upscale_mode if metadata.track == "upscaling" else "bicubic"
    canonical = build_canonicalization_plan(str(input_path), str(output_path))
    filters = [canonical[canonical.index("-vf") + 1], f"scale={width}:{height}:flags={scale_mode}"]
    detail = (
        variant.upscale_detail if metadata.track == "upscaling"
        else variant.compression_detail
    )
    if detail:
        filters.append(f"unsharp=3:3:{detail}:3:3:0")
    frame_rate = Fraction(info.fps).limit_denominator(1_000_000)
    fps = str(frame_rate)
    if metadata.track == "compression":
        try:
            codec_args = resolve_encode(
                metadata.params, default_crf=variant.compression_crf, preset=variant.preset
            ).ffmpeg_args
        except EncodeParamError as exc:
            raise GpuTransformError(str(exc)) from None
    else:
        codec_args = [
            "-c:v", "libx264", "-preset", variant.preset,
            "-crf", str(variant.upscaling_crf), "-pix_fmt", "yuv420p",
        ]
    try:
        _run_capture(
            [
                ffmpeg_path, "-y", "-v", "error", "-threads", "4",
                "-filter_threads", "1", "-i", str(input_path), "-map", "0:v:0",
                "-vf", ",".join(filters), "-enc_time_base",
                f"{frame_rate.denominator}:{frame_rate.numerator}",
                "-r", fps, "-fps_mode", "cfr",
                *codec_args, "-threads", "4", "-fs", str(maximum_output_bytes),
                "-movflags", "+faststart", "-an", str(output_path),
            ],
            deadline=deadline,
        )
        if not output_path.is_file() or not 0 < output_path.stat().st_size <= maximum_output_bytes:
            raise GpuTransformError("CPU fallback output is empty or exceeded its byte cap")
        result = _probe(
            output_path, ffprobe_path=ffprobe_path, deadline=deadline,
            maximum_frames=maximum_frames,
        )
        if (result.width, result.height) != target or result.frame_count != info.frame_count:
            raise GpuTransformError("CPU fallback changed the required geometry or frame count")
        if not math.isclose(result.fps, info.fps, rel_tol=0.001):
            raise GpuTransformError("CPU fallback changed the required frame rate")
        return TransformResult(
            output_path=output_path, frames=result.frame_count,
            width=result.width, height=result.height,
            device=CPU_FALLBACK_DEVICE, gpu_seconds=0.0,
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
