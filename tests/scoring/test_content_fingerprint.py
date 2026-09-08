"""Version-1 evidence is exact, reproducible, bounded and fail-closed."""

import hashlib
import io
import os
from pathlib import Path
import random

import pytest

from vidaio.scoring import content_fingerprint as cf


def write_y4m(path: Path, frames: list[bytes], width=32, height=32, *, tag=b"") -> bytes:
    header = f"YUV4MPEG2 W{width} H{height} F30:1 Ip A1:1 C420jpeg".encode() + tag + b"\n"
    chroma = bytes([128]) * (2 * ((width + 1) // 2) * ((height + 1) // 2))
    body = header + b"".join(b"FRAME\n" + y + chroma for y in frames)
    path.write_bytes(body)
    return body


def test_whole_stream_digest_includes_headers_and_unsampled_frames(tmp_path):
    rng = random.Random(721)
    frames = [rng.randbytes(1024) for _ in range(64)]
    p = tmp_path / "measured.y4m"
    original = write_y4m(p, frames)
    first = cf.compute_canonical_content(p)
    assert first.frame_count == 64
    assert first.sample_positions == tuple(i * 63 // 31 for i in range(32))
    assert first.canonical_content_digest == hashlib.sha256(original).hexdigest()
    assert first.content_fingerprint == tuple(cf.frame_phash(frames[i]) for i in first.sample_positions)
    # Frame 1 is not sampled, but still belongs to the exact content digest.
    frames[1] = bytes(1024)
    write_y4m(p, frames)
    second = cf.compute_canonical_content(p)
    assert second.content_fingerprint == first.content_fingerprint
    assert second.canonical_content_digest != first.canonical_content_digest
    write_y4m(p, frames, tag=b" XSCORER=test")
    third = cf.compute_canonical_content(p)
    assert third.content_fingerprint == second.content_fingerprint
    assert third.canonical_content_digest != second.canonical_content_digest


@pytest.mark.parametrize("count", [1, 2, 7, 31, 32, 33])
def test_exactly_32_positions_repeat_short_clips(tmp_path, count):
    p = tmp_path / "short.y4m"
    frames = [bytes([i]) * 1024 for i in range(count)]
    write_y4m(p, frames)
    evidence = cf.compute_canonical_content(p)
    expected = tuple(i * (count - 1) // 31 for i in range(32))
    assert evidence.sample_positions == expected
    assert len(evidence.content_fingerprint) == 32
    assert evidence.content_fingerprint == tuple(cf.frame_phash(frames[i]) for i in expected)


@pytest.mark.parametrize("count", [0, -1, True, 2.0, None])
def test_no_invented_positions(count):
    with pytest.raises(cf.ContentFingerprintUnavailable):
        cf.sample_positions(count)


@pytest.mark.parametrize("width,height", [(1, 1), (3, 5), (32, 32), (37, 35), (70, 2)])
def test_area_resize_matches_independent_exact_overlap_reference(width, height):
    rng = random.Random(512)
    source = rng.randbytes(width * height)
    actual = cf._area_luma(io.BufferedReader(io.BytesIO(source)), width, height, None)
    expected = []
    # All overlap lengths use integer coordinates at the common scale 32.
    for oy in range(32):
        for ox in range(32):
            total = 0
            for iy in range(height):
                wy = max(0, min((iy + 1) * 32, (oy + 1) * height) - max(iy * 32, oy * height))
                for ix in range(width):
                    wx = max(0, min((ix + 1) * 32, (ox + 1) * width) - max(ix * 32, ox * width))
                    total += source[iy * width + ix] * wx * wy
            denominator = width * height
            expected.append((2 * total + denominator) // (2 * denominator))
    assert actual == bytes(expected)
    if width == height == 32:
        assert actual == source


def test_dct_constant_ties_dc_and_golden_word():
    assert cf.CONTENT_FINGERPRINT_VERSION == 1
    assert cf.frame_phash(bytes(1024)) == "0000000000000000"
    assert cf.frame_phash(bytes([200]) * 1024) == "8000000000000000"
    source = random.Random(302).randbytes(1024)
    # Direct 2D summation independently checks the separable implementation,
    # row-major ordering, DC inclusion and true median of 63 AC values.
    coefficients = [sum(source[y * 32 + x] * cf._DCT[u][x] * cf._DCT[v][y]
                        for y in range(32) for x in range(32))
                    for v in range(8) for u in range(8)]
    median = sorted(coefficients[1:])[31]
    bits = "".join("1" if c > median else "0" for c in coefficients)
    assert cf.frame_phash(source) == f"{int(bits, 2):016x}"
    assert cf.frame_phash(source) == "bf6a30dd2f08a607"
    assert int(cf.frame_phash(source), 16).bit_count() == 32


@pytest.mark.parametrize("body", [
    b"", b"not y4m\n", b"YUV4MPEG2 W32 H32 C420jpeg\n",
    b"YUV4MPEG2 W0 H32 C420jpeg\nFRAME\n",
    b"YUV4MPEG2 W32 H32 C420p10\nFRAME\n" + bytes(4096),
    b"YUV4MPEG2 W32 W32 H32 C420jpeg\nFRAME\n" + bytes(1536),
    b"YUV4MPEG2 W32 H32 C420jpeg\nWRONG\n" + bytes(1536),
    b"YUV4MPEG2 W32 H32 C420jpeg\nFRAME\n" + bytes(1535),
    b"YUV4MPEG2 W32 H32 C420jpeg\nFRAME\n" + bytes(1536) + b"x",
    b"YUV4MPEG2 W32 H32 C420jpeg " + b"X" * 4096 + b"\n",
    b"YUV4MPEG2 W32 H32 C420jpeg X\x00\nFRAME\n" + bytes(1536),
])
def test_invalid_stream_never_supplies_evidence(tmp_path, body):
    p = tmp_path / "bad.y4m"
    p.write_bytes(body)
    with pytest.raises(cf.ContentFingerprintUnavailable):
        cf.compute_canonical_content(p)


def test_rejects_symlink_nonregular_and_cancellation(tmp_path):
    p = tmp_path / "ok.y4m"
    write_y4m(p, [bytes(1024)])
    link = tmp_path / "link.y4m"
    link.symlink_to(p)
    pipe = tmp_path / "fifo.y4m"
    os.mkfifo(pipe)
    for invalid in (link, pipe, tmp_path, tmp_path / "absent"):
        with pytest.raises(cf.ContentFingerprintUnavailable):
            cf.compute_canonical_content(invalid)
    with pytest.raises(cf.ContentFingerprintCancelled):
        cf.compute_canonical_content(p, cancelled=lambda: True)
    calls = 0
    def cancel_during_read():
        nonlocal calls
        calls += 1
        return calls > 4
    with pytest.raises(cf.ContentFingerprintCancelled):
        cf.compute_canonical_content(p, cancelled=cancel_during_read)


def test_changed_stream_during_sampling_never_supplies_evidence(tmp_path, monkeypatch):
    p = tmp_path / "mutated.y4m"
    write_y4m(p, [bytes(1024)])
    original = cf.frame_phash
    def mutate_then_hash(data):
        with p.open("ab") as target:
            target.write(b"changed")
        return original(data)
    monkeypatch.setattr(cf, "frame_phash", mutate_then_hash)
    with pytest.raises(cf.ContentFingerprintUnavailable, match="changed"):
        cf.compute_canonical_content(p)


def test_frame_read_memory_bound_and_no_foreign_thread_pool():
    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 65536
            return super().read(size)
    body = bytes(2_000_000)
    digest = hashlib.sha256()
    cf._consume(BoundedReader(body), len(body), None, digest)
    assert digest.hexdigest() == hashlib.sha256(body).hexdigest()
