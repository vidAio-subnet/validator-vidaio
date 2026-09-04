"""CPU video pHash primitive: deterministic distance and strict shape checks."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from vidaio.scoring import CpuVideoPhash, PerceptualHashUnavailable
from vidaio.scoring.phash_cpu import FFMPEG_PHASH_MAX_ALLOC_BYTES


def test_cpu_video_phash_hamming_distance() -> None:
    assert CpuVideoPhash.distance("0000000000000000", "0000000000000000") == 0
    assert CpuVideoPhash.distance("0000000000000000", "000000000000000f") == 4
    assert CpuVideoPhash.distance("ffffffffffffffff", "0000000000000000") == 64


@pytest.mark.parametrize("bad", ["", "0", "g" * 16, "0" * 17])
def test_cpu_video_phash_refuses_malformed_values(bad: str) -> None:
    with pytest.raises(ValueError, match="pHash"):
        CpuVideoPhash.distance(bad, "0" * 16)


def test_cpu_video_phash_decoder_is_single_threaded_and_allocation_bounded(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "untrusted.mp4"
    source.write_bytes(b"not actually decoded by this seam test")
    backend = CpuVideoPhash(sample_frames=2)
    monkeypatch.setattr(backend, "_duration", lambda _path: 1.0)
    seen: list[str] = []

    def fake_run(command, **kwargs):
        seen.extend(command)
        assert kwargs["stderr"] is subprocess.DEVNULL
        return SimpleNamespace(returncode=0, stdout=b"\0" * (2 * 32 * 32))

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert len(backend.compute_phash(str(source))) == 16
    assert "-nostdin" in seen
    assert seen[seen.index("-max_alloc") + 1] == str(FFMPEG_PHASH_MAX_ALLOC_BYTES)
    assert seen[seen.index("-threads") + 1] == "1"
    assert seen[seen.index("-filter_threads") + 1] == "1"
    assert seen[seen.index("-filter_complex_threads") + 1] == "1"


def test_cpu_video_phash_rejects_oversized_decoder_output(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "malformed.mp4"
    source.write_bytes(b"malformed")
    backend = CpuVideoPhash(sample_frames=1)
    monkeypatch.setattr(backend, "_duration", lambda _path: 1.0)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=b"\0" * (32 * 32 + 1)
        ),
    )
    with pytest.raises(PerceptualHashUnavailable, match="oversized"):
        backend.compute_phash(str(source))


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)
def test_reencoded_same_video_is_a_near_duplicate(tmp_path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    common = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x96:rate=8:duration=2",
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
    ]
    subprocess.run([*common, "-crf", "17", str(first)], check=True, timeout=30)
    subprocess.run([*common, "-crf", "31", str(second)], check=True, timeout=30)
    assert (
        hashlib.sha256(first.read_bytes()).digest()
        != hashlib.sha256(second.read_bytes()).digest()
    )

    backend = CpuVideoPhash()
    first_hash = backend.compute_phash(str(first))
    second_hash = backend.compute_phash(str(second))
    assert backend.compute_phash(str(first)) == first_hash
    assert backend.distance(first_hash, second_hash) <= 8
