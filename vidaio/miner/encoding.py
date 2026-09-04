"""Codec coverage for the miner backends: task params -> one ffmpeg encode plan.

Requests may carry the legacy contract's wire fields: `target_codec`
(AV1 | H264 | HEVC | VP9 and their aliases),
`codec_mode` (CRF | VBR), `target_bitrate` (bps, VBR only) and the quality tier
`compression_type` (High | Medium | Low). Synthetic rounds carry none of them.

    params ──► resolve_encode ──► EncodePlan(codec, encoder, args)
                                     │
             FfmpegCompressBackend ──┴── gpu_worker._encode

Rules (spec = the old miner's CODEC_MAP + request model, plus the proxy's tiers):

* codec aliases are case-insensitive; a missing `target_codec` means h264 so a
  synthetic task encodes byte-for-byte as before; an UNKNOWN codec is an error
  (`bad_params` at the ingress) — the old miner silently substituted AV1, which
  is exactly the "wrong codec" a parity test must be able to catch;
* `codec_mode=VBR` with a positive `target_bitrate` -> constrained VBR
  (`-b:v N -maxrate N -bufsize 2N`); anything else -> CRF;
* the CRF comes from the tier when `compression_type` is given (each codec has
  its own scale — x264/x265 0–51, SVT-AV1 and VP9 0–63), otherwise from the
  backend's configured default (the launch calibration);
* every plan emits `yuv420p` in an mp4 with `+faststart`; audio is dropped.

Encoders are the software ones the pinned static ffmpeg ships (libx264,
libx265, libsvtav1, libvpx-vp9) — no NVENC dependency on either backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

CODEC_ALIASES: Mapping[str, str] = {
    "H264": "h264", "H.264": "h264", "AVC": "h264", "X264": "h264",
    "HEVC": "hevc", "H265": "hevc", "H.265": "hevc", "X265": "hevc",
    "AV1": "av1",
    "VP9": "vp9",
}

SUPPORTED_CODECS: tuple[str, ...] = ("h264", "hevc", "av1", "vp9")

ENCODER_BY_CODEC: Mapping[str, str] = {
    "h264": "libx264",
    "hevc": "libx265",
    "av1": "libsvtav1",
    "vp9": "libvpx-vp9",
}

#: Quality tier -> CRF, per codec. `High` is the CRF at which each encoder
#: crosses the proxy's High-tier floor (VMAF 93 vs the pristine reference) on
#: corpus footage, measured with the image's libvmaf; Medium/Low step the
#: ladder up by the same span the old
#: NVENC miner used (cq 30/35/40). The parity test's per-tier VMAF check is
#: the guard on real customer content, not this table.
TIER_CRF: Mapping[str, Mapping[str, int]] = {
    "h264": {"High": 20, "Medium": 24, "Low": 28},
    "hevc": {"High": 22, "Medium": 26, "Low": 30},
    "av1": {"High": 35, "Medium": 40, "Low": 45},
    "vp9": {"High": 28, "Medium": 33, "Low": 38},
}

CRF_RANGE: Mapping[str, tuple[int, int]] = {
    "h264": (0, 51),
    "hevc": (0, 51),
    "av1": (0, 63),
    "vp9": (0, 63),
}

#: SVT-AV1 takes a numeric preset (0 slowest .. 13 fastest); map the x264 names.
_SVTAV1_PRESET = {
    "placebo": 1, "veryslow": 2, "slower": 3, "slow": 4, "medium": 6,
    "fast": 8, "faster": 9, "veryfast": 10, "superfast": 11, "ultrafast": 12,
}
#: libvpx-vp9 `-cpu-used` (0 slowest .. 8 fastest) in "good" quality mode.
_VP9_CPU_USED = {
    "placebo": 0, "veryslow": 0, "slower": 1, "slow": 1, "medium": 2,
    "fast": 3, "faster": 4, "veryfast": 5, "superfast": 6, "ultrafast": 8,
}


class EncodeParamError(ValueError):
    """The task params ask for something this backend cannot honour."""


@dataclass(frozen=True)
class EncodePlan:
    codec: str
    encoder: str
    mode: str  # "crf" | "vbr"
    args: tuple[str, ...]  # complete video-codec argument list, `-c:v` included

    @property
    def ffmpeg_args(self) -> list[str]:
        return list(self.args)


def normalize_codec(value: object) -> str:
    """`target_codec` -> canonical codec name, or EncodeParamError."""
    if value is None:
        return "h264"
    text = str(value).strip()
    if not text:
        return "h264"
    canonical = CODEC_ALIASES.get(text.upper(), text.lower())
    if canonical not in SUPPORTED_CODECS:
        raise EncodeParamError(
            f"unsupported target_codec {text!r}; supported: {', '.join(SUPPORTED_CODECS)}"
        )
    return canonical


def _tier(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().capitalize()
    if text not in ("High", "Medium", "Low"):
        raise EncodeParamError(f"unsupported compression_type {value!r}")
    return text


def _bitrate(value: object) -> int | None:
    if value is None:
        return None
    try:
        bps = int(float(value))
    except (TypeError, ValueError):
        raise EncodeParamError(f"invalid target_bitrate {value!r}") from None
    if bps <= 0:
        return None
    if bps > 500_000_000:
        raise EncodeParamError(f"target_bitrate {bps} bps is beyond the 500 Mbps cap")
    return bps


def resolve_crf(codec: str, params: Mapping[str, object], *, default_crf: int) -> int:
    tier = _tier(params.get("compression_type"))
    lo, hi = CRF_RANGE[codec]
    if tier is not None:
        return TIER_CRF[codec][tier]
    return max(lo, min(hi, int(default_crf)))


def resolve_encode(
    params: Mapping[str, object],
    *,
    default_crf: int,
    preset: str = "medium",
) -> EncodePlan:
    """Build the codec argument list for one task.

    `default_crf` is the backend's configured CRF (the launch calibration) and
    is only used when the task names no quality tier; it is clamped into the
    chosen codec's scale.
    """
    codec = normalize_codec(params.get("target_codec"))
    encoder = ENCODER_BY_CODEC[codec]
    mode_raw = str(params.get("codec_mode") or "CRF").strip().upper()
    if mode_raw not in ("CRF", "VBR"):
        raise EncodeParamError(f"unsupported codec_mode {params.get('codec_mode')!r}")
    bitrate = _bitrate(params.get("target_bitrate"))
    mode = "vbr" if (mode_raw == "VBR" and bitrate) else "crf"
    preset_name = (preset or "medium").strip().lower()

    args: list[str] = ["-c:v", encoder]
    if codec in ("h264", "hevc"):
        args += ["-preset", preset_name]
    elif codec == "av1":
        args += ["-preset", str(_SVTAV1_PRESET.get(preset_name, 6))]
    else:  # vp9
        args += ["-deadline", "good", "-cpu-used", str(_VP9_CPU_USED.get(preset_name, 2)), "-row-mt", "1"]

    if mode == "vbr":
        assert bitrate is not None
        args += ["-b:v", str(bitrate), "-maxrate", str(bitrate), "-bufsize", str(2 * bitrate)]
    else:
        crf = resolve_crf(codec, params, default_crf=default_crf)
        args += ["-crf", str(crf)]
        if codec == "vp9":
            args += ["-b:v", "0"]  # constant-quality mode needs an explicit zero bitrate
    if codec == "hevc":
        args += ["-tag:v", "hvc1"]  # Apple/mp4 players require the hvc1 tag
    args += ["-pix_fmt", "yuv420p"]
    return EncodePlan(codec=codec, encoder=encoder, mode=mode, args=tuple(args))


def encoder_preset_value(codec: str, preset: str) -> str:
    """The encoder-native preset token for `codec` (shared with the premium path).

    x264/x265 take the named preset; SVT-AV1 takes its numeric 0-13 scale and
    libvpx-vp9 its numeric cpu-used scale (both mapped from the x264 names).
    """
    name = (preset or "medium").strip().lower()
    if codec == "av1":
        return str(_SVTAV1_PRESET.get(name, 6))
    if codec == "vp9":
        return str(_VP9_CPU_USED.get(name, 2))
    return name
