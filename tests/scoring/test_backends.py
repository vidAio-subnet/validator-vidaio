"""Deterministic PieAPP frame derivation + the fake backend's protocol conformance."""

import pytest

from vidaio.scoring import (
    DeterministicFakeBackend,
    MediaInfo,
    PerceptualCheckBackend,
    PerceptualHashBackend,
    ProbeBackend,
    VmafBackend,
    derive_pieapp_start_frame,
    usable_frames,
)


def test_start_frame_same_inputs_same_frame() -> None:
    a = derive_pieapp_start_frame("digest-a", "chal-1", 1000)
    b = derive_pieapp_start_frame("digest-a", "chal-1", 1000)
    assert a == b == 335  # sha256-derived, stable forever


def test_start_frame_differs_by_challenge_and_content() -> None:
    base = derive_pieapp_start_frame("digest-a", "chal-1", 1000)
    other_challenge = derive_pieapp_start_frame("digest-a", "chal-2", 1000)
    other_content = derive_pieapp_start_frame("digest-b", "chal-1", 1000)
    assert base != other_challenge
    assert base != other_content


def test_start_frame_in_range_and_no_separator_ambiguity() -> None:
    for n in (1, 2, 7, 97):
        frame = derive_pieapp_start_frame("d", "c", n)
        assert 0 <= frame < n
    # "ab"+"c" must not collide with "a"+"bc"
    assert derive_pieapp_start_frame("ab", "c", 10**9) != derive_pieapp_start_frame(
        "a", "bc", 10**9
    )
    with pytest.raises(ValueError):
        derive_pieapp_start_frame("d", "c", 0)


def test_usable_frames_window() -> None:
    assert usable_frames(300, 4) == 297
    assert usable_frames(4, 4) == 1
    assert usable_frames(3, 4) == 0


def test_fake_backend_serves_all_protocols() -> None:
    media = MediaInfo(
        codec="h264",
        width=640,
        height=360,
        fps=30.0,
        frame_count=90,
        duration=3.0,
        byte_size=1000,
    )
    fake = DeterministicFakeBackend(
        vmaf={("ref", "cand"): 91.5},
        pieapp={("ref", "cand"): 0.42},
        media={"ref": media},
        phashes={"cand": "ff00"},
    )
    assert isinstance(fake, VmafBackend)
    assert isinstance(fake, ProbeBackend)
    assert isinstance(fake, PerceptualCheckBackend)
    assert isinstance(fake, PerceptualHashBackend)

    assert fake.compute("ref", "cand") == 91.5
    assert fake.probe("ref") == media
    assert fake.compute_phash("cand") == "ff00"
    assert fake.distance("ff00", "ff01") == 1

    pieapp = fake.pieapp
    assert pieapp.compute("ref", "cand", start_frame=17) == 0.42
    assert fake.pieapp_calls == [("ref", "cand", 17)]

    with pytest.raises(KeyError):
        fake.compute("ref", "missing")  # a fake must never invent a metric


def test_fake_backend_perceptual_defaults_pass() -> None:
    fake = DeterministicFakeBackend()
    assert fake.check_tone_manipulation("r", "c").passed
    assert fake.check_color_grayscale("r", "c").passed
    assert fake.check_chroma_uv("r", "c").passed
