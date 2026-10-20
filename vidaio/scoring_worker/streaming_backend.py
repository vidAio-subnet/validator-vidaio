"""Streaming compression packet measurement for the scoring worker.

Reads the candidate's first video stream packet by packet (size, presentation time,
keyframe flag) with ffprobe — the encoded bitstream as delivered, before any decoding —
and hands it to the pure measurement in :mod:`vidaio.scoring.streaming`. The input is the
worker's private snapshot of the miner output, so an auditor recomputes exactly this.
"""

from __future__ import annotations

import json
from typing import Protocol

from vidaio.scoring.backends_real import UNTRUSTED_INPUT_ARGS, MediaToolError, _run
from vidaio.scoring.streaming import StreamingMeasurement, StreamingPacket, measure_streaming

__all__ = [
    "FfprobeStreamingBackend",
    "StreamingMeasurementError",
    "StreamingPacketBackend",
]


class StreamingMeasurementError(RuntimeError):
    """CANDIDATE-side failure: the output's packets cannot be read or carry no timestamps.
    Scored as a violation."""


class StreamingPacketBackend(Protocol):
    def measure(
        self, candidate_path: str, *, fps: float, frames: int, peak_window_seconds: float
    ) -> StreamingMeasurement: ...

    def version(self) -> str: ...


class FfprobeStreamingBackend:
    """The shipped backend: ffprobe's packet list of the first video stream."""

    VERSION = "streaming-packets/1"  # keep equal to runtime_identity.STREAMING_PACKETS_VERSION

    def __init__(self, ffprobe_path: str = "ffprobe", *, timeout: float = 300.0) -> None:
        self._ffprobe = ffprobe_path
        self._timeout = timeout

    def version(self) -> str:
        return self.VERSION

    def packets(self, path: str) -> list[StreamingPacket]:
        argv = [
            self._ffprobe, "-hide_banner", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=size,pts_time,flags", "-print_format", "json",
            *UNTRUSTED_INPUT_ARGS, path,
        ]
        try:
            doc = json.loads(_run(argv, timeout=self._timeout).stdout)
        except (MediaToolError, json.JSONDecodeError) as exc:
            raise StreamingMeasurementError(f"packets unreadable: {exc}") from exc
        out = []
        for packet in doc.get("packets", []):
            pts = packet.get("pts_time")
            try:
                pts_seconds = float(pts) if pts not in (None, "N/A") else None
                size = int(packet["size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise StreamingMeasurementError(f"malformed packet entry: {packet!r}") from exc
            out.append(
                StreamingPacket(size_bytes=size, pts_seconds=pts_seconds, keyframe="K" in str(packet.get("flags", "")))
            )
        return out

    def measure(
        self, candidate_path: str, *, fps: float, frames: int, peak_window_seconds: float
    ) -> StreamingMeasurement:
        try:
            return measure_streaming(
                self.packets(candidate_path), fps=fps, frames=frames, peak_window_seconds=peak_window_seconds
            )
        except ValueError as exc:
            raise StreamingMeasurementError(str(exc)) from exc
