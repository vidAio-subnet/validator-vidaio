"""Real subprocess backends against genuinely generated media (ffprobe/libvmaf)."""

from __future__ import annotations

import pytest

from tests.scoring_worker.conftest import FFMPEG, FFPROBE, ClipPair, requires_media_tools
from vidaio.scoring import build_canonicalization_plan
from vidaio.scoring.backends_real import (
    CanonicalizationError,
    CanonicalizationTimeout,
    CanonicalizeExecutor,
    CpuPerceptualCheckBackend,
    FfmpegVmafBackend,
    FfprobeBackend,
    MediaToolError,
    MediaToolTimeout,
    NotConfiguredError,
    PieAppTorchBackend,
    SECONDARY_VMAF_MODEL,
    UnconfiguredPerceptualCheckBackend,
    detect_tool_versions,
)

pytestmark = requires_media_tools


@pytest.fixture(scope="module")
def probe() -> FfprobeBackend:
    return FfprobeBackend(FFPROBE, timeout=30.0)


def test_probe_maps_every_stream_property(probe: FfprobeBackend, clips: ClipPair) -> None:
    info = probe.probe(clips.reference)
    assert info.codec == "h264"
    assert (info.width, info.height) == (160, 120)
    assert info.fps == pytest.approx(10.0)
    assert info.frame_count == 10
    assert info.duration == pytest.approx(1.0, rel=0.05)
    assert info.byte_size > 0
    assert info.pix_fmt == "yuv420p"
    assert info.bit_depth == 8


def test_probe_count_frames_fallback_on_y4m(
    probe: FfprobeBackend, clips: ClipPair, tmp_path
) -> None:
    # y4m carries no nb_frames — forces the decode-and-count fallback.
    canonical = tmp_path / "ref.y4m"
    executor = CanonicalizeExecutor(FFMPEG, timeout=60.0)
    executor.run(build_canonicalization_plan(clips.reference, str(canonical)))
    info = probe.probe(str(canonical))
    assert info.codec == "rawvideo"
    assert info.frame_count == 10
    assert info.pix_fmt == "yuv420p"
    assert info.byte_size == canonical.stat().st_size


def test_probe_missing_file_is_typed_error(probe: FfprobeBackend, tmp_path) -> None:
    with pytest.raises(MediaToolError):
        probe.probe(str(tmp_path / "nope.mp4"))


def test_vmaf_deterministic_bounded_and_versioned(clips: ClipPair, tmp_path) -> None:
    backend = FfmpegVmafBackend(FFMPEG, work_dir=tmp_path, timeout=60.0)
    first = backend.compute(clips.reference, clips.candidate)
    second = backend.compute(clips.reference, clips.candidate)
    assert first == second  # full clip, single thread, pinned model — no randomness
    assert 0.0 < first <= 100.0
    assert backend.version != "unknown"  # cached from the libvmaf JSON log


def test_vmaf_secondary_model_runs_and_differs_in_model(
    clips: ClipPair, tmp_path
) -> None:
    secondary = FfmpegVmafBackend(
        FFMPEG, model=SECONDARY_VMAF_MODEL, work_dir=tmp_path, timeout=60.0
    )
    score = secondary.compute(clips.reference, clips.candidate)
    assert 0.0 < score <= 100.0
    assert secondary.model == "version=vmaf_v0.6.1neg"


def test_canonicalize_failure_captures_stderr(tmp_path) -> None:
    executor = CanonicalizeExecutor(FFMPEG, timeout=30.0)
    plan = build_canonicalization_plan(
        str(tmp_path / "missing-input.mp4"), str(tmp_path / "out.y4m")
    )
    with pytest.raises(CanonicalizationError) as excinfo:
        executor.run(plan)
    assert excinfo.value.stderr  # ffmpeg's own diagnostic is preserved
    assert excinfo.value.argv[0] == FFMPEG  # the "ffmpeg" token resolved to the binary


def test_canonicalize_timeout_is_typed(clips: ClipPair, tmp_path) -> None:
    executor = CanonicalizeExecutor(FFMPEG, timeout=30.0)
    plan = build_canonicalization_plan(clips.reference, str(tmp_path / "out.y4m"))
    with pytest.raises(CanonicalizationTimeout) as excinfo:
        executor.run(plan, timeout=0.001)
    assert isinstance(excinfo.value, MediaToolTimeout)


def test_unconfigured_pieapp_refuses_loudly() -> None:
    backend = PieAppTorchBackend()
    backend.version = "not-configured"  # independent of packages on the test host
    with pytest.raises(NotConfiguredError, match="optional 'media'"):
        backend.compute("ref", "cand", start_frame=0)
    perceptual = UnconfiguredPerceptualCheckBackend()
    for check in (
        perceptual.check_tone_manipulation,
        perceptual.check_color_grayscale,
        perceptual.check_chroma_uv,
    ):
        with pytest.raises(NotConfiguredError):
            check("ref", "cand")


def test_pieapp_cpu_backend_uses_exact_deterministic_window() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.calls = []

        def compute(self, reference, candidate, *, start_frame, sample_window):
            self.calls.append((reference, candidate, start_frame, sample_window))
            return 0.125

    runtime = Runtime()
    loaded_devices = []

    def load(device):
        loaded_devices.append(device)
        return runtime

    backend = PieAppTorchBackend(
        device="cpu",
        sample_window=4,
        _runtime_loader=load,
        _backend_version="piq/test:pieapp",
    )
    assert backend.compute("reference.y4m", "candidate.y4m", start_frame=17) == 0.125
    assert loaded_devices == ["cpu"]
    assert runtime.calls == [("reference.y4m", "candidate.y4m", 17, 4)]


def test_pieapp_rejects_invalid_window_and_start() -> None:
    with pytest.raises(ValueError, match="sample_window"):
        PieAppTorchBackend(sample_window=0)
    backend = PieAppTorchBackend(_backend_version="test")
    with pytest.raises(ValueError, match="start_frame"):
        backend.compute("ref", "cand", start_frame=-1)


def test_cpu_perceptual_backend_accepts_identity_and_rejects_manipulations(
    clips: ClipPair, tmp_path
) -> None:
    backend = CpuPerceptualCheckBackend()
    if backend.version == "not-configured":
        pytest.skip("optional media/OpenCV dependency group is not installed")
    executor = CanonicalizeExecutor(FFMPEG, timeout=60.0)
    reference = tmp_path / "reference.y4m"
    executor.run(build_canonicalization_plan(clips.reference, str(reference)))
    assert backend.check_tone_manipulation(str(reference), str(reference)).passed
    assert backend.check_color_grayscale(str(reference), str(reference)).passed
    assert backend.check_chroma_uv(str(reference), str(reference)).passed

    variants = {
        "tone": ("eq=brightness=0.2", backend.check_tone_manipulation),
        "grayscale": ("hue=s=0", backend.check_color_grayscale),
        "chroma": ("hue=h=90", backend.check_chroma_uv),
    }
    for name, (filter_graph, check) in variants.items():
        output = tmp_path / f"{name}.y4m"
        executor.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(reference),
                "-vf",
                filter_graph,
                "-pix_fmt",
                "yuv420p",
                "-f",
                "yuv4mpegpipe",
                "-y",
                str(output),
            ]
        )
        assert not check(str(reference), str(output)).passed


def test_detect_tool_versions_stamps_all_tools(tmp_path) -> None:
    backend = FfmpegVmafBackend(FFMPEG, work_dir=tmp_path, timeout=60.0)
    versions = detect_tool_versions(FFMPEG, FFPROBE, vmaf_backend=backend, timeout=30.0)
    assert set(versions) == {"ffmpeg", "ffprobe", "libvmaf"}
    for tool, stamp in versions.items():
        assert stamp.startswith(f"{tool}/")
        assert not stamp.endswith("/unknown")
