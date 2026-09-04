"""Codec params -> encode plan (pure; the real-ffmpeg checks live in test_miner)."""

from __future__ import annotations

import pytest

from vidaio.miner.encoding import (
    SUPPORTED_CODECS,
    TIER_CRF,
    EncodeParamError,
    normalize_codec,
    resolve_encode,
)


def _args(plan) -> list[str]:
    return plan.ffmpeg_args


def test_no_params_is_the_launch_h264_crf_encode() -> None:
    plan = resolve_encode({}, default_crf=22, preset="medium")
    assert plan.codec == "h264" and plan.encoder == "libx264" and plan.mode == "crf"
    assert _args(plan) == ["-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p"]


@pytest.mark.parametrize(
    "alias,codec",
    [("AV1", "av1"), ("av1", "av1"), ("H264", "h264"), ("H.264", "h264"), ("h264", "h264"),
     ("HEVC", "hevc"), ("H265", "hevc"), ("H.265", "hevc"), ("VP9", "vp9"), ("", "h264"), (None, "h264")],
)
def test_codec_aliases_follow_the_old_codec_map(alias, codec) -> None:
    assert normalize_codec(alias) == codec


def test_unknown_codec_is_an_error_not_a_silent_default() -> None:
    with pytest.raises(EncodeParamError, match="unsupported target_codec"):
        resolve_encode({"target_codec": "PRORES"}, default_crf=22)


@pytest.mark.parametrize("codec", SUPPORTED_CODECS)
def test_each_codec_has_a_tier_table_and_distinct_tier_crfs(codec) -> None:
    high, medium, low = (TIER_CRF[codec][t] for t in ("High", "Medium", "Low"))
    assert high < medium < low
    for tier in ("High", "Medium", "Low"):
        plan = resolve_encode({"target_codec": codec, "compression_type": tier}, default_crf=22)
        assert plan.codec == codec
        assert _args(plan)[_args(plan).index("-crf") + 1] == str(TIER_CRF[codec][tier])


def test_vbr_uses_the_bitrate_and_drops_crf() -> None:
    plan = resolve_encode(
        {"target_codec": "HEVC", "codec_mode": "VBR", "target_bitrate": 2_500_000.0}, default_crf=22
    )
    a = _args(plan)
    assert plan.mode == "vbr" and "-crf" not in a
    assert a[a.index("-b:v") + 1] == "2500000" and a[a.index("-maxrate") + 1] == "2500000"
    assert a[a.index("-bufsize") + 1] == "5000000"
    assert a[a.index("-tag:v") + 1] == "hvc1"


def test_vbr_without_a_bitrate_falls_back_to_crf() -> None:
    plan = resolve_encode({"codec_mode": "VBR"}, default_crf=22)
    assert plan.mode == "crf" and "-crf" in _args(plan)
    plan = resolve_encode({"codec_mode": "VBR", "target_bitrate": 0}, default_crf=22)
    assert plan.mode == "crf"


def test_vp9_crf_mode_needs_zero_bitrate_and_row_mt() -> None:
    a = _args(resolve_encode({"target_codec": "VP9"}, default_crf=22))
    assert a[a.index("-b:v") + 1] == "0" and "-row-mt" in a and "-crf" in a


def test_av1_preset_is_numeric_and_default_crf_is_clamped_to_the_codec_scale() -> None:
    a = _args(resolve_encode({"target_codec": "AV1"}, default_crf=22, preset="fast"))
    assert a[a.index("-preset") + 1] == "8"
    a = _args(resolve_encode({"target_codec": "H264"}, default_crf=60))
    assert a[a.index("-crf") + 1] == "51"


@pytest.mark.parametrize(
    "params,message",
    [
        ({"codec_mode": "ABR"}, "codec_mode"),
        ({"compression_type": "Ultra"}, "compression_type"),
        ({"codec_mode": "VBR", "target_bitrate": "lots"}, "target_bitrate"),
        ({"codec_mode": "VBR", "target_bitrate": 10**12}, "500 Mbps"),
    ],
)
def test_bad_params_fail_closed(params, message) -> None:
    with pytest.raises(EncodeParamError, match=message):
        resolve_encode(params, default_crf=22)
