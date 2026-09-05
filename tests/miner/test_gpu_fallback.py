from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from vidaio.miner import gpu_worker as worker
from vidaio.miner.remote_gpu import CPU_FALLBACK_DEVICE, GPU_PROTOCOL_VERSION
from vidaio.scoring.canonicalize import build_canonicalization_plan


@pytest.fixture(params=["10", "30000/1001"])
def media(tmp_path: Path, request) -> Path:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    source = tmp_path / "vfr.mkv"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            f"testsrc2=size=64x48:rate={request.param}", "-frames:v", "10",
            "-vf", r"setpts=PTS+gte(N\,5)*0.5/TB", "-fps_mode", "vfr",
            "-c:v", "ffv1", str(source),
        ],
        check=True, capture_output=True, timeout=10,
    )
    return source


def metadata(source: Path, **updates) -> worker.GpuTaskMetadata:
    values = dict(
        protocol=GPU_PROTOCOL_VERSION, track="upscaling", solution_variant="balanced",
        input_digest=hashlib.sha256(source.read_bytes()).hexdigest(),
        input_size=source.stat().st_size, deadline_seconds=30,
        params={"target_width": 128, "target_height": 96, "upscale_factor": 2},
    )
    values.update(updates)
    return worker.GpuTaskMetadata(**values)


def canonical_info(media):
    output = media.with_suffix(".reference.y4m")
    subprocess.run(build_canonicalization_plan(str(media), str(output)), check=True, capture_output=True, timeout=10)
    return worker._probe(output, ffprobe_path="ffprobe", deadline=time.monotonic() + 10)


def test_decode_uses_the_scorer_cfr_timeline(media: Path):
    deadline = time.monotonic() + 15
    original = worker._probe(media, ffprobe_path="ffprobe", deadline=deadline)
    assert original.frame_count == 10
    info = worker._cfr_info(media, original, ffmpeg_path="ffmpeg", deadline=deadline)
    canonical = canonical_info(media)
    assert info.frame_count == canonical.frame_count > 10
    assert info.fps == pytest.approx(canonical.fps)
    decoded = worker._decode_bounded(
        media, media.with_suffix(".rgb"), ffmpeg_path="ffmpeg",
        maximum_bytes=1_000_000, deadline=deadline,
    )
    assert decoded == info.frame_count * info.width * info.height * 3


@pytest.mark.parametrize("track", ["compression", "upscaling"])
@pytest.mark.parametrize("variant", ["quality", "balanced"])
def test_raw_cap_recovery_streams_real_media_with_honest_attestation(media, track, variant):
    output = media.with_suffix(".mp4")
    result = worker.transform_media(
        media, output, metadata(media, track=track, solution_variant=variant),
        maximum_raw_bytes=100_000, allow_cpu_fallback=True,
    )
    assert result.device == CPU_FALLBACK_DEVICE
    assert result.gpu_seconds == 0
    assert (result.width, result.height, result.frames) == (128, 96, canonical_info(media).frame_count)
    assert canonical_info(output).frame_count == canonical_info(media).frame_count
    assert output.stat().st_size > 0
    assert list(media.parent.glob("*.rgb")) == []


def test_fallback_disabled_by_default_and_raw_cap_not_raised(media):
    with pytest.raises(worker.RecoverableGpuTransformError, match="cap is 100000"):
        worker.transform_media(
            media, media.with_suffix(".mp4"), metadata(media), maximum_raw_bytes=100_000,
        )
    assert not media.with_suffix(".mp4").exists()


@pytest.mark.parametrize("failure", ["invalid", "deadline", "cuda"])
def test_fallback_never_masks_invalid_input_deadline_or_cuda_fault(media, monkeypatch, failure):
    def forbidden(*args, **kwargs):
        pytest.fail("unsafe CPU fallback")

    monkeypatch.setattr(worker, "_stream_cpu_fallback", forbidden)
    values = {}
    if failure == "invalid":
        values["params"] = {"target_width": 127, "target_height": 96}
        error, message = worker.GpuTransformError, "must be even"
    elif failure == "deadline":
        values["deadline_seconds"] = 0.000001
        error, message = worker.GpuTransformError, "deadline"
    else:
        canonical_info(media)
        media = media.with_suffix(".reference.y4m")
        def broken_cuda(*args, **kwargs):
            raise RuntimeError("CUDA device fault")
        monkeypatch.setattr(worker, "_gpu_frames", broken_cuda)
        error, message = RuntimeError, "CUDA device fault"
    with pytest.raises(error, match=message):
        worker.transform_media(
            media, media.with_suffix(".mp4"), metadata(media, **values),
            allow_cpu_fallback=True,
        )


def test_recovery_uses_remaining_original_deadline_and_removes_raw_partials(media, monkeypatch):
    expected = canonical_info(media)
    media = media.with_suffix(".reference.y4m")
    original_probe = worker._probe
    deadlines = []

    def probe(*args, **kwargs):
        deadlines.append(kwargs["deadline"])
        return original_probe(*args, **kwargs)

    def mismatch(source, raw, **kwargs):
        raw.write_bytes(b"partial")
        return 7

    monkeypatch.setattr(worker, "_probe", probe)
    monkeypatch.setattr(worker, "_decode_bounded", mismatch)
    result = worker.transform_media(
        media, media.with_suffix(".mp4"), metadata(media), allow_cpu_fallback=True,
    )
    assert result.frames == expected.frame_count
    assert len(deadlines) == 2 and deadlines[0] == deadlines[1]
    assert list(media.parent.glob("*.rgb")) == []


def test_cpu_output_cap_never_returns_a_truncated_video(media):
    output = media.with_suffix(".mp4")
    with pytest.raises(worker.GpuTransformError, match="byte cap|frame count"):
        worker.transform_media(
            media, output, metadata(media), allow_cpu_fallback=True,
            maximum_raw_bytes=1, maximum_output_bytes=100,
        )
    assert not output.exists()


def test_fallback_preserves_requested_compression_codec(media):
    output = media.with_suffix(".mp4")
    worker.transform_media(
        media, output, metadata(media, track="compression", params={"codec": "h264"}),
        allow_cpu_fallback=True, maximum_raw_bytes=1,
    )
    codec = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "csv=p=0", str(output)], timeout=10,
    ).decode().strip()
    assert codec == "h264"


def test_vfr_projection_still_uses_the_larger_canonical_timeline(media):
    raw = worker._probe(media, ffprobe_path="ffprobe", deadline=time.monotonic() + 10)
    canonical = canonical_info(media)
    assert canonical.frame_count > raw.frame_count
    cap = raw.frame_count * raw.width * raw.height * 3
    with pytest.raises(worker.RecoverableGpuTransformError, match="decoded input projects"):
        worker.transform_media(media, media.with_suffix(".mp4"), metadata(media), maximum_raw_bytes=cap)


def test_fallback_removes_output_when_encode_times_out(media, monkeypatch):
    output = media.with_suffix(".mp4")
    original = worker._run_capture

    def timeout(command, **kwargs):
        if command[0] == "ffmpeg" and "-fs" in command:
            output.write_bytes(b"partial")
            raise worker.GpuTransformError("media tool exceeded deadline: ffmpeg")
        return original(command, **kwargs)

    monkeypatch.setattr(worker, "_run_capture", timeout)
    with pytest.raises(worker.GpuTransformError, match="deadline"):
        worker.transform_media(
            media, output, metadata(media), allow_cpu_fallback=True, maximum_raw_bytes=1,
        )
    assert not output.exists()


@pytest.mark.parametrize("payload", [
    b"#tb 0: 0/1\n0,0,0,1,10,0x00\n",
    b"#tb 0: 1/24\n0,0,2,1,10,0x00\n",
    b"#tb 0: 1/24\n",
    b"#tb 0: 1/24\n" + b"0,0,0,1,10,0x00\n" * 3601,
])
def test_cfr_projection_rejects_unbounded_or_nonuniform_timing(tmp_path, monkeypatch, payload):
    def capture(command, **kwargs):
        assert command[command.index("-frames:v") + 1] == "3601"
        return payload

    monkeypatch.setattr(worker, "_run_capture", capture)
    with pytest.raises(worker.GpuTransformError, match="bounded CFR geometry"):
        worker._cfr_info(
            tmp_path / "input.mkv", worker.VideoInfo(64, 48, 24, 10),
            ffmpeg_path="ffmpeg", deadline=time.monotonic() + 10,
        )
