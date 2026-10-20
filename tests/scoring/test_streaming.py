"""Streaming compression: packet measurement, delivery rules and the score formula."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from vidaio.scoring import (
    ReasonCode,
    ScoringConfig,
    StreamingMeasurement,
    StreamingPacket,
    StreamingRules,
    measure_streaming,
    score_streaming,
    streaming_target_kbps,
    streaming_violations,
)

FPS = 30.0


def _packets(sizes: list[int], keyframes: set[int]) -> list[StreamingPacket]:
    return [
        StreamingPacket(size_bytes=size, pts_seconds=i / FPS, keyframe=i in keyframes)
        for i, size in enumerate(sizes)
    ]


def test_budget_is_derived_from_the_clips_own_geometry_and_rate() -> None:
    assert streaming_target_kbps(width=1920, height=1080, fps=30.0, bits_per_pixel=0.05) == pytest.approx(3110.4)
    assert streaming_target_kbps(width=1080, height=1920, fps=30.0, bits_per_pixel=0.05) == pytest.approx(3110.4)
    with pytest.raises(ValueError):
        streaming_target_kbps(width=0, height=1080, fps=30.0, bits_per_pixel=0.05)


def test_average_peak_and_keyframe_spacing_are_measured_from_packets() -> None:
    # 4 s at 30 fps, 1250 bytes/frame = 300 kbps; one 2 s burst of 2500-byte frames
    sizes = [1250] * 120
    sizes[60:120] = [2500] * 60
    m = measure_streaming(_packets(sizes, {0, 60}), fps=FPS, frames=120, peak_window_seconds=2.0)
    assert m.avg_kbps == pytest.approx(8 * (60 * 1250 + 60 * 2500) / 4.0 / 1000)
    assert m.peak_kbps == pytest.approx(8 * 60 * 2500 / 2.0 / 1000)
    assert m.max_keyframe_interval_seconds == pytest.approx(2.0)
    assert m.first_frame_is_keyframe and m.keyframes == 2 and m.packets == 120
    # one keyframe only: the tail span is the whole clip
    m = measure_streaming(_packets([1000] * 120, {0}), fps=FPS, frames=120, peak_window_seconds=2.0)
    assert m.max_keyframe_interval_seconds == pytest.approx(4.0)
    # fewer packets than the window: one window over the whole clip
    m = measure_streaming(_packets([1000] * 10, {0}), fps=FPS, frames=10, peak_window_seconds=2.0)
    assert m.peak_kbps == pytest.approx(m.avg_kbps)


def test_packets_without_timestamps_or_frames_are_unmeasurable() -> None:
    with pytest.raises(ValueError):
        measure_streaming([], fps=FPS, frames=10, peak_window_seconds=2.0)
    with pytest.raises(ValueError):
        measure_streaming([StreamingPacket(10, None, True)], fps=FPS, frames=1, peak_window_seconds=2.0)


def _m(**overrides) -> StreamingMeasurement:
    base = dict(avg_kbps=3000.0, peak_kbps=4000.0, max_keyframe_interval_seconds=2.0,
                first_frame_is_keyframe=True, keyframes=3, packets=180)
    base.update(overrides)
    return StreamingMeasurement(**base)


def test_each_delivery_rule_zeroes_on_its_own() -> None:
    config = ScoringConfig()
    rules = StreamingRules(bits_per_pixel=0.05)
    target = 3110.4
    assert streaming_violations(_m(), target_kbps=target, rules=rules, fps=FPS, config=config) == []
    # 1 % rounding slack on the average
    assert streaming_violations(_m(avg_kbps=target * 1.009), target_kbps=target, rules=rules, fps=FPS, config=config) == []
    codes = lambda **kw: [v.code for v in streaming_violations(_m(**kw), target_kbps=target, rules=rules, fps=FPS, config=config)]  # noqa: E731
    assert codes(avg_kbps=target * 1.02) == [ReasonCode.STREAMING_BITRATE_EXCEEDED]
    assert codes(peak_kbps=target * 1.51) == [ReasonCode.STREAMING_PEAK_EXCEEDED]
    assert codes(max_keyframe_interval_seconds=2.1) == [ReasonCode.STREAMING_KEYFRAME_INTERVAL]
    assert codes(max_keyframe_interval_seconds=2.0 + 0.4 / FPS) == []  # half-frame timestamp slack
    assert codes(first_frame_is_keyframe=False) == [ReasonCode.STREAMING_KEYFRAME_INTERVAL]


def test_score_maps_vmaf_from_the_floor_to_100() -> None:
    config = ScoringConfig()
    b = score_streaming(vmaf=90.0, measurement=_m(), target_kbps=3110.4, config=config)
    assert b.kind == "streaming" and b.final == pytest.approx(0.75) and b.vmaf_floor == 60.0
    assert score_streaming(vmaf=55.0, measurement=_m(), target_kbps=1.0, config=config).final == 0.0
    assert score_streaming(vmaf=100.0, measurement=_m(), target_kbps=1.0, config=config).final == 1.0
    with pytest.raises(ValueError):
        score_streaming(vmaf=float("nan"), measurement=_m(), target_kbps=1.0, config=config)


def test_rules_and_config_are_validated() -> None:
    assert StreamingRules(bits_per_pixel=0.05).as_params() == {
        "scoring": "streaming", "bits_per_pixel": 0.05, "max_keyframe_interval_s": 2.0,
        "peak_ratio": 1.5, "peak_window_s": 2.0,
    }
    for bad in ({"bits_per_pixel": 0}, {"bits_per_pixel": 0.05, "peak_ratio": 0.9},
                {"bits_per_pixel": 0.05, "max_keyframe_interval_seconds": 0}, {"bits_per_pixel": 0.05, "extra": 1}):
        with pytest.raises(ValueError):
            StreamingRules(**bad)
    with pytest.raises(ValueError):
        ScoringConfig(streaming_vmaf_floor=100.0)
    with pytest.raises(ValueError):
        ScoringConfig(streaming_rate_tolerance=-0.01)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not available")
def test_real_encodes_are_measured_from_their_packets(tmp_path: Path) -> None:
    from vidaio.scoring_worker.streaming_backend import FfprobeStreamingBackend, StreamingMeasurementError

    def encode(name: str, *x264: str) -> Path:
        out = tmp_path / name
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=4",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", *x264, "-f", "mp4", str(out)], check=True)
        return out

    backend = FfprobeStreamingBackend()
    target = streaming_target_kbps(width=320, height=180, fps=30.0, bits_per_pixel=0.1)  # 172.8 kbps
    rules = StreamingRules(bits_per_pixel=0.1)
    ok = encode("ok.mp4", "-b:v", "150k", "-maxrate", "200k", "-bufsize", "300k", "-g", "60", "-keyint_min", "60",
                "-sc_threshold", "0")
    m = backend.measure(str(ok), fps=30.0, frames=120, peak_window_seconds=2.0)
    assert m.keyframes == 2 and m.first_frame_is_keyframe and m.max_keyframe_interval_seconds == pytest.approx(2.0)
    assert streaming_violations(m, target_kbps=target, rules=rules, fps=30.0, config=ScoringConfig()) == []
    long_gop = encode("gop.mp4", "-b:v", "150k", "-maxrate", "200k", "-bufsize", "300k", "-g", "300", "-sc_threshold", "0")
    m = backend.measure(str(long_gop), fps=30.0, frames=120, peak_window_seconds=2.0)
    assert [v.code for v in streaming_violations(m, target_kbps=target, rules=rules, fps=30.0, config=ScoringConfig())] == [
        ReasonCode.STREAMING_KEYFRAME_INTERVAL
    ]
    (tmp_path / "junk.mp4").write_bytes(b"not a video")
    with pytest.raises(StreamingMeasurementError):
        backend.measure(str(tmp_path / "junk.mp4"), fps=30.0, frames=120, peak_window_seconds=2.0)
