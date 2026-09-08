"""Small real-ffmpeg regressions for the accepted fixed content rule."""

import hashlib
import math
from pathlib import Path
import random
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from vidaio.scoring.backends_real import CanonicalizeExecutor
from vidaio.scoring.canonicalize import build_canonicalization_plan, plan_template_digest
from vidaio.scoring.content_duplicate_evidence import same_content
from vidaio.scoring.content_fingerprint import compute_canonical_content

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def ffmpeg(*args):
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *map(str, args)],
                   check=True, capture_output=True, timeout=30)


def evidence(path: Path):
    canonical = path.with_name(path.name + ".canonical.y4m")
    plan = build_canonicalization_plan(str(path), str(canonical))
    CanonicalizeExecutor(FFMPEG).run(plan)
    measured = compute_canonical_content(canonical)
    return SimpleNamespace(**measured.__dict__, encoded_size=path.stat().st_size,
                           canonicalization_plan_digest=plan_template_digest(plan, str(path), str(canonical))), canonical


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("content-media")
    source = root / "source.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x96:rate=16:duration=2", "-c:v", "libx264",
           "-threads", "1", "-preset", "medium", "-crf", "0", source)
    return root, source


def test_five_metadata_only_twins_have_identical_canonical_content(clips):
    root, source = clips
    original, _ = evidence(source)
    original_bytes = hashlib.sha256(source.read_bytes()).digest()
    twins = []
    for index in range(5):
        twin = root / f"metadata-{index}.mp4"
        ffmpeg("-i", source, "-map", "0:v:0", "-c:v", "copy", "-metadata", f"comment=owned-fixture-{index}", twin)
        measured, _ = evidence(twin)
        assert hashlib.sha256(twin.read_bytes()).digest() != original_bytes
        assert measured.canonical_content_digest == original.canonical_content_digest
        assert measured.content_fingerprint == original.content_fingerprint
        assert same_content(original, measured)
        twins.append(measured.canonical_content_digest)
    assert len(set(twins)) == 1


def test_psnr_over_50_pixel_noise_survives_real_remux_and_matches(clips):
    root, source = clips
    _, canonical = evidence(source)
    raw = canonical.read_bytes()
    end_header = raw.index(b"\n") + 1
    frame_bytes = 160 * 96 * 3 // 2
    noisy = bytearray(raw)
    rng = random.Random(20260906)
    squared_error = 0
    samples = 0
    offset = end_header
    while offset < len(raw):
        assert raw[offset:offset + 6] == b"FRAME\n"
        offset += 6
        for i in range(160 * 96):
            before = raw[offset + i]
            after = min(255, max(0, before + rng.choice((-1, 0, 0, 0, 1))))
            noisy[offset + i] = after
            squared_error += (after - before) ** 2
            samples += 1
        offset += frame_bytes
    assert 10 * math.log10(255 ** 2 / (squared_error / samples)) > 50
    noise_y4m = root / "noise.y4m"
    noise_y4m.write_bytes(noisy)
    # Real raw-video encodings keep the same byte size without lying about the
    # measured encoded_size or depending on codec rate-control variation.
    clean_nut, noisy_nut = root / "clean.nut", root / "noisy.nut"
    for path, output in ((canonical, clean_nut), (noise_y4m, noisy_nut)):
        ffmpeg("-i", path, "-c:v", "rawvideo", "-threads", "1", output)
    clean, _ = evidence(clean_nut)
    noisy_evidence, _ = evidence(noisy_nut)
    assert clean.canonical_content_digest != noisy_evidence.canonical_content_digest
    assert clean.encoded_size == noisy_evidence.encoded_size
    assert sum((int(a, 16) ^ int(b, 16)).bit_count() <= 6
               for a, b in zip(clean.content_fingerprint, noisy_evidence.content_fingerprint)) >= 30
    assert same_content(clean, noisy_evidence)


def test_crf22_and_crf26_legitimate_rate_choices_remain_distinct(clips):
    root, source = clips
    measured = []
    for crf in (22, 26):
        output = root / f"crf-{crf}.mp4"
        ffmpeg("-i", source, "-c:v", "libx264", "-threads", "1", "-preset", "medium", "-crf", crf, output)
        measured.append(evidence(output)[0])
    left, right = measured
    assert left.canonical_content_digest != right.canonical_content_digest
    assert 100 * abs(left.encoded_size - right.encoded_size) > max(left.encoded_size, right.encoded_size)
    assert not same_content(left, right)
