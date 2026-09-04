"""Canonicalization — run on BOTH reference and candidate before any metric.

Spec §18 (trivial-inverse prevention): container, colorspace, timebase, pixel-format and
decoder quirks must never be a scoring vector, so both sides of every comparison are
first normalized to an identical canonical representation. This module builds the
ffmpeg command *plan* as a pure argv list — execution is injected elsewhere — and
validates stream-level consistency between reference and candidate.

The plan digest is stored on every :class:`~vidaio.scoring.result.ItemScore` so an
auditor can prove which normalization produced the scored inputs (spec §08).
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

from vidaio.scoring.backends import MediaInfo
from vidaio.scoring.gates import ReasonCode, ValidityViolation

#: Canonical normalization targets. Changing any of these changes the plan digest.
CANONICAL_PIX_FMT = "yuv420p"
CANONICAL_COLOR = "bt709"
CANONICAL_TIMESCALE = 90000

#: Tokens substituted for the concrete paths when computing the *template* digest,
#: so one digest identifies the normalization recipe independent of file names.
INPUT_TOKEN = "{input}"
OUTPUT_TOKEN = "{output}"


def build_canonicalization_plan(
    input_path: str,
    output_path: str,
    *,
    pix_fmt: str = CANONICAL_PIX_FMT,
    color: str = CANONICAL_COLOR,
    timescale: int = CANONICAL_TIMESCALE,
    fps: float | None = None,
    scale_width: int | None = None,
    scale_height: int | None = None,
) -> list[str]:
    """Pure builder: the ffmpeg argv that canonicalizes one file. No execution here.

    Normalizes: first video stream only (audio/subs/data dropped), PTS rebased to zero,
    constant frame rate, pixel format, color primaries/transfer/matrix, and a fixed
    container timescale — the classic container/colorspace/timebase mismatch surface.
    Output should use a lossless-friendly container (the .y4m/.mkv choice belongs to
    the executing backend; the plan pins the stream-level normalization).
    """
    if (scale_width is None) != (scale_height is None):
        raise ValueError("scale_width and scale_height must be supplied together")
    if scale_width is not None and (scale_width < 1 or scale_height < 1):
        raise ValueError("canonical scale dimensions must be positive")
    vf = []
    if scale_width is not None:
        # The measured VMAF-delta calibration used this exact rescale for
        # Downscale challenges: compare what the miner added at output geometry.
        vf.append(f"scale={scale_width}:{scale_height}:flags=lanczos")
    vf.extend((f"format={pix_fmt}", "setpts=PTS-STARTPTS"))
    if fps is not None:
        vf.append(f"fps={fps}")
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        ",".join(vf),
        "-fps_mode",
        "cfr",
        "-color_primaries",
        color,
        "-color_trc",
        color,
        "-colorspace",
        color,
        "-video_track_timescale",
        str(timescale),
        "-y",
        output_path,
    ]


def plan_digest(argv: Sequence[str]) -> str:
    """sha256 over the argv, NUL-joined (unambiguous). Same plan -> same digest."""
    return hashlib.sha256("\x00".join(argv).encode("utf-8")).hexdigest()


def plan_template_digest(argv: Sequence[str], input_path: str, output_path: str) -> str:
    """Digest of the plan with the concrete paths replaced by stable tokens.

    Identifies the normalization *recipe* across items (audit groups by this), while
    :func:`plan_digest` identifies the exact per-item invocation.
    """
    templated = [
        INPUT_TOKEN if a == input_path else OUTPUT_TOKEN if a == output_path else a
        for a in argv
    ]
    return plan_digest(templated)


def validate_stream(
    reference_info: MediaInfo,
    candidate_info: MediaInfo,
    *,
    duration_tolerance: float = 0.05,
    pts_tolerance: float = 0.05,
) -> list[ValidityViolation]:
    """Post-canonicalization consistency checks between reference and candidate.

    Checks dims, pix_fmt, frame count, duration (relative tolerance) and internal
    PTS consistency (duration vs frame_count/fps) on the candidate. Violations use the
    shared :class:`ReasonCode` enum so they fold straight into the gate pipeline via
    ``GateContext.extra_violations``.
    """
    violations: list[ValidityViolation] = []
    ref, cand = reference_info, candidate_info

    if (cand.width, cand.height) != (ref.width, ref.height):
        violations.append(
            ValidityViolation(
                code=ReasonCode.STREAM_DIMENSIONS_MISMATCH,
                detail=f"candidate {cand.width}x{cand.height} vs reference {ref.width}x{ref.height}",
            )
        )
    if cand.pix_fmt != ref.pix_fmt:
        violations.append(
            ValidityViolation(
                code=ReasonCode.STREAM_PIX_FMT_MISMATCH,
                detail=f"candidate {cand.pix_fmt!r} vs reference {ref.pix_fmt!r}",
            )
        )
    if cand.frame_count != ref.frame_count:
        violations.append(
            ValidityViolation(
                code=ReasonCode.FRAME_COUNT_MISMATCH,
                detail=f"candidate {cand.frame_count} frames vs reference {ref.frame_count}",
                measured=float(cand.frame_count),
                limit=float(ref.frame_count),
            )
        )
    if ref.duration > 0:
        rel = abs(cand.duration - ref.duration) / ref.duration
        if rel > duration_tolerance:
            violations.append(
                ValidityViolation(
                    code=ReasonCode.STREAM_DURATION_MISMATCH,
                    detail="candidate duration deviates from reference",
                    measured=cand.duration,
                    limit=ref.duration,
                )
            )
    if cand.fps > 0 and cand.duration > 0:
        implied = cand.frame_count / cand.fps
        rel = abs(implied - cand.duration) / cand.duration
        if rel > pts_tolerance:
            violations.append(
                ValidityViolation(
                    code=ReasonCode.STREAM_PTS_INCONSISTENT,
                    detail="frame_count/fps disagrees with container duration",
                    measured=implied,
                    limit=cand.duration,
                )
            )
    return violations


class SecondaryDecoderBackend(Protocol):
    """Interface hook: an independent second decoder for untrusted bitstreams (spec §18).

    A real implementation decodes `path` with a *different* decoder implementation than
    the primary probe (e.g. gstreamer vs ffmpeg) and reports whether both agree on
    stream shape. Phase 2 — not yet wired.
    """

    def probe(self, path: str) -> MediaInfo: ...


def cross_check_decoders(
    path: str,
    primary_info: MediaInfo,
    secondary: SecondaryDecoderBackend | None = None,
) -> list[ValidityViolation]:
    """Two-decoder cross-check hook. With no secondary decoder wired it is a no-op.

    When a :class:`SecondaryDecoderBackend` is provided, its probe of `path` must agree
    with ``primary_info`` on dims/frame_count/pix_fmt; disagreement marks the bitstream
    untrusted (decoder-differential mismatch).
    """
    if secondary is None:
        return []
    return validate_stream(primary_info, secondary.probe(path))
