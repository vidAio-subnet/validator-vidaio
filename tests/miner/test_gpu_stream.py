"""Streaming GPU path: clips too long for raw spools stay on the GPU.

CUDA is not available in CI, so a CPU torch device stands in for it; the decode ->
transform -> encode plumbing, frame accounting and fallbacks are what is tested.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from vidaio.miner import gpu_worker as worker
from vidaio.miner.remote_gpu import CPU_FALLBACK_DEVICE, GPU_PROTOCOL_VERSION
from vidaio.scoring.canonicalize import build_canonicalization_plan

torch = pytest.importorskip("torch")


@pytest.fixture
def media(tmp_path: Path) -> Path:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    source = tmp_path / "clip.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=24",
         "-frames:v", "30", "-c:v", "ffv1", str(source)],
        check=True, capture_output=True, timeout=20,
    )
    return source


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(
        worker, "_cuda_device",
        lambda *, recoverable=False, context="": (torch.device("cpu"), "cuda:0/test-double"),
    )


def metadata(source: Path, **updates) -> worker.GpuTaskMetadata:
    values = dict(
        protocol=GPU_PROTOCOL_VERSION, track="compression", solution_variant="balanced",
        input_digest=hashlib.sha256(source.read_bytes()).hexdigest(),
        input_size=source.stat().st_size, deadline_seconds=60, params={},
    )
    values.update(updates)
    return worker.GpuTaskMetadata(**values)


def canonical_frames(path: Path) -> int:
    output = path.with_suffix(".check.y4m")
    subprocess.run(build_canonicalization_plan(str(path), str(output)), check=True, capture_output=True, timeout=20)
    return worker._probe(output, ffprobe_path="ffprobe", deadline=time.monotonic() + 10).frame_count


@pytest.mark.parametrize(
    ("track", "params", "size"),
    [
        ("compression", {}, (64, 48)),
        ("upscaling", {"upscale_factor": 2}, (128, 96)),
    ],
)
def test_clips_over_the_raw_cap_stream_through_the_gpu(media, fake_cuda, track, params, size):
    output = media.with_suffix(".mp4")
    result = worker.transform_media(
        media, output, metadata(media, track=track, params=params),
        maximum_raw_bytes=10_000,  # one 64x48 frame is 9 216 bytes: the spool path is out
    )
    assert result.device == "cuda:0/test-double"
    assert (result.width, result.height) == size
    assert result.frames == canonical_frames(media) == canonical_frames(output) == 30
    assert output.stat().st_size > 0
    assert list(media.parent.glob("*.rgb")) == []


def test_decode_runs_ahead_of_the_gpu_transform(media, fake_cuda, monkeypatch):
    # Overlap, not turn-taking: while the first batch is still in the transform, the
    # reader must already have fetched the next one. A serial loop would time out here.
    real_read, real_transform = worker._read_up_to, worker._transform_batch
    reads: list[int] = []
    second_read = threading.Event()

    def read(stream, size):
        data = real_read(stream, size)
        reads.append(len(data))
        if len(reads) >= 2:
            second_read.set()
        return data

    overlapped: list[bool] = []

    def transform(host, **kwargs):
        overlapped.append(second_read.wait(timeout=5))
        return real_transform(host, **kwargs)

    monkeypatch.setattr(worker, "_read_up_to", read)
    monkeypatch.setattr(worker, "_transform_batch", transform)
    output = media.with_suffix(".mp4")
    result = worker.transform_media(media, output, metadata(media), maximum_raw_bytes=10_000)
    assert overlapped[0] is True
    assert result.frames == canonical_frames(output) == 30


def test_without_cuda_a_long_clip_still_falls_back_to_cpu(media):
    output = media.with_suffix(".mp4")
    result = worker.transform_media(
        media, output, metadata(media), maximum_raw_bytes=10_000, allow_cpu_fallback=True,
    )
    assert result.device == CPU_FALLBACK_DEVICE
    assert canonical_frames(output) == 30


def test_a_gpu_fault_mid_stream_falls_back_to_cpu(media, fake_cuda, monkeypatch):
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(worker, "_transform_batch", broken)
    output = media.with_suffix(".mp4")
    result = worker.transform_media(
        media, output, metadata(media), maximum_raw_bytes=10_000, allow_cpu_fallback=True,
    )
    assert calls and result.device == CPU_FALLBACK_DEVICE
    assert canonical_frames(output) == 30


def test_a_gpu_fault_without_fallback_is_reported_with_the_cap(media, fake_cuda, monkeypatch):
    monkeypatch.setattr(worker, "_transform_batch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(worker.RecoverableGpuTransformError, match=r"cap is 10000; GPU stream transform failed"):
        worker.transform_media(media, media.with_suffix(".mp4"), metadata(media), maximum_raw_bytes=10_000)
    assert not media.with_suffix(".mp4").exists()


def test_an_encoder_failure_is_not_hidden_by_the_fallback(media, fake_cuda, monkeypatch):
    original = worker._encode_command

    def bad_encoder(source, output_path, **kwargs):
        command = original(source, output_path, **kwargs)
        return [*command[:-1], "-c:v", "no-such-encoder", command[-1]]

    monkeypatch.setattr(worker, "_encode_command", bad_encoder)
    output = media.with_suffix(".mp4")
    with pytest.raises(worker.GpuTransformError, match="ffmpeg encode"):
        worker.transform_media(media, output, metadata(media), maximum_raw_bytes=10_000, allow_cpu_fallback=True)
    assert not output.exists()


def test_the_frame_bound_is_configurable(media, fake_cuda):
    with pytest.raises(worker.GpuTransformError, match="frame count is outside bounds"):
        worker.transform_media(media, media.with_suffix(".mp4"), metadata(media), maximum_frames=20)
    result = worker.transform_media(
        media, media.with_suffix(".mp4"), metadata(media), maximum_frames=40, maximum_raw_bytes=10_000,
    )
    assert result.frames == 30


def test_the_stream_respects_the_deadline(media, fake_cuda, monkeypatch):
    def slow(host, **kwargs):
        time.sleep(0.4)
        return host

    monkeypatch.setattr(worker, "_transform_batch", slow)
    output = media.with_suffix(".mp4")
    info = worker._cfr_info(
        media, worker._probe(media, ffprobe_path="ffprobe", deadline=time.monotonic() + 10),
        ffmpeg_path="ffmpeg", deadline=time.monotonic() + 10,
    )
    with pytest.raises(worker.GpuTransformError, match="deadline"):
        worker._gpu_stream(
            media, output, info=info, target=(64, 48), metadata=metadata(media),
            ffmpeg_path="ffmpeg", deadline=time.monotonic() + 1.0,
        )


def test_product_deadlines_up_to_an_hour_are_accepted(media):
    assert metadata(media, deadline_seconds=880).deadline_seconds == 880
    with pytest.raises(ValueError):
        metadata(media, deadline_seconds=3601)
