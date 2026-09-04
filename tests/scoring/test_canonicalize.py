"""Canonicalization plan purity/digest stability + stream validation."""

import pytest

from vidaio.scoring import (
    MediaInfo,
    ReasonCode,
    build_canonicalization_plan,
    cross_check_decoders,
    plan_digest,
    plan_template_digest,
    validate_stream,
)


def info(**overrides) -> MediaInfo:
    base = dict(
        codec="h264",
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=300,
        duration=10.0,
        byte_size=1_000_000,
        pix_fmt="yuv420p",
    )
    base.update(overrides)
    return MediaInfo(**base)


def test_plan_is_pure_argv_and_digest_stable() -> None:
    a = build_canonicalization_plan("in.mp4", "out.y4m")
    b = build_canonicalization_plan("in.mp4", "out.y4m")
    assert a == b
    assert all(isinstance(x, str) for x in a)
    assert a[0] == "ffmpeg"
    assert plan_digest(a) == plan_digest(b)
    assert len(plan_digest(a)) == 64


def test_digest_changes_when_normalization_changes() -> None:
    base = build_canonicalization_plan("in.mp4", "out.y4m")
    other = build_canonicalization_plan("in.mp4", "out.y4m", pix_fmt="yuv420p10le")
    fps = build_canonicalization_plan("in.mp4", "out.y4m", fps=24.0)
    assert plan_digest(base) != plan_digest(other)
    assert plan_digest(base) != plan_digest(fps)


def test_model_delta_input_rescale_is_pinned_to_lanczos() -> None:
    plan = build_canonicalization_plan(
        "input.mp4", "input.y4m", scale_width=1920, scale_height=1080
    )
    vf = plan[plan.index("-vf") + 1]
    assert vf.startswith("scale=1920:1080:flags=lanczos,format=yuv420p,")
    with pytest.raises(ValueError, match="supplied together"):
        build_canonicalization_plan(
            "input.mp4", "input.y4m", scale_width=1920
        )


def test_template_digest_stable_across_paths() -> None:
    a = build_canonicalization_plan("/data/item1/in.mp4", "/tmp/a.y4m")
    b = build_canonicalization_plan("/data/item2/in.mp4", "/tmp/b.y4m")
    assert plan_digest(a) != plan_digest(b)
    assert plan_template_digest(
        a, "/data/item1/in.mp4", "/tmp/a.y4m"
    ) == plan_template_digest(b, "/data/item2/in.mp4", "/tmp/b.y4m")


def test_validate_stream_clean_pair() -> None:
    assert validate_stream(info(), info(byte_size=400_000)) == []


def test_validate_stream_flags_mismatches() -> None:
    violations = validate_stream(
        info(),
        info(width=1280, height=720, pix_fmt="yuv444p", frame_count=250),
    )
    codes = {v.code for v in violations}
    assert ReasonCode.STREAM_DIMENSIONS_MISMATCH in codes
    assert ReasonCode.STREAM_PIX_FMT_MISMATCH in codes
    assert ReasonCode.FRAME_COUNT_MISMATCH in codes


def test_validate_stream_duration_and_pts() -> None:
    # A duration deviation with unchanged frames/fps also makes the candidate's own
    # frame_count/fps disagree with its container duration -> both codes fire.
    dur = validate_stream(info(), info(duration=11.0))
    assert {v.code for v in dur} == {
        ReasonCode.STREAM_DURATION_MISMATCH,
        ReasonCode.STREAM_PTS_INCONSISTENT,
    }
    # frame_count/fps says 300/30 = 10s but container claims 9s -> PTS inconsistent
    pts = validate_stream(info(duration=9.0), info(duration=9.0))
    assert [v.code for v in pts] == [ReasonCode.STREAM_PTS_INCONSISTENT]


def test_cross_check_decoders_is_noop_without_secondary() -> None:
    assert cross_check_decoders("cand.mp4", info(), None) == []


def test_cross_check_decoders_with_disagreeing_secondary() -> None:
    class Secondary:
        def probe(self, path: str) -> MediaInfo:
            return info(frame_count=299)

    violations = cross_check_decoders("cand.mp4", info(), Secondary())
    assert [v.code for v in violations] == [ReasonCode.FRAME_COUNT_MISMATCH]
