"""Deterministic CPU video perceptual hash for public-corpus screening."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

CPU_VIDEO_PHASH_VERSION = "cpu-video-phash/1"
# ffmpeg's global ``-max_alloc`` bounds any single heap allocation. 64 MiB is
# ample for the canonical 32x32 output and normal source frames while refusing
# pathological dimensions before a decoder can request multi-gigabyte buffers.
FFMPEG_PHASH_MAX_ALLOC_BYTES = 64 * 1024 * 1024


class PerceptualHashUnavailable(RuntimeError):
    """The bounded CPU decoder could not produce a perceptual hash."""


class CpuVideoPhash:
    """A 64-bit temporal-majority pHash over eight evenly spaced video frames.

    Each frame is reduced with the standard low-frequency DCT pHash.  Taking the
    bitwise majority across fixed frame positions makes remuxes and modest codec
    changes collide while retaining the existing Hamming-distance threshold.
    Frames are decoded by the release-pinned ffmpeg binary and the small DCT is
    evaluated in pure Python, so this path is CPU-only and introduces no GPU or
    model dependency.
    """

    name = "cpu-video-phash"
    version = CPU_VIDEO_PHASH_VERSION

    def __init__(
        self,
        *,
        sample_frames: int = 8,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        timeout_seconds: float = 30.0,
    ) -> None:
        if sample_frames < 1 or sample_frames > 64:
            raise ValueError("sample_frames must be in [1, 64]")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.sample_frames = sample_frames
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.timeout_seconds = timeout_seconds

    def compute_phash(self, path: str) -> str:
        source = Path(path)
        if not source.is_file():
            raise PerceptualHashUnavailable(
                f"perceptual-hash input is not a file: {path}"
            )
        duration = self._duration(source)
        frame_rate = self.sample_frames / duration
        command = [
            self.ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-max_alloc",
            str(FFMPEG_PHASH_MAX_ALLOC_BYTES),
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-filter_complex_threads",
            "1",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"fps={frame_rate:.12f},scale=32:32:flags=area,format=gray",
            "-frames:v",
            str(self.sample_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                # Error text can contain attacker-controlled container metadata.
                # Do not retain an unbounded diagnostic stream in validator memory.
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PerceptualHashUnavailable(
                f"bounded ffmpeg pHash decode failed: {exc}"
            ) from exc
        if result.returncode != 0:
            raise PerceptualHashUnavailable(
                f"ffmpeg could not safely decode pHash frames from {path}"
            )
        frame_bytes = 32 * 32
        expected_bytes = frame_bytes * self.sample_frames
        if len(result.stdout) != expected_bytes:
            raise PerceptualHashUnavailable(
                "ffmpeg returned an incomplete or oversized pHash frame set "
                f"({len(result.stdout)} bytes, expected {expected_bytes})"
            )
        frames = [
            result.stdout[offset : offset + frame_bytes]
            for offset in range(0, len(result.stdout), frame_bytes)
        ]
        hashes = [self._frame_hash(frame) for frame in frames]

        majority = 0
        required = len(hashes) // 2 + 1
        for bit in range(63, -1, -1):
            votes = sum((value >> bit) & 1 for value in hashes)
            majority = (majority << 1) | int(votes >= required)
        return f"{majority:016x}"

    def _duration(self, path: Path) -> float:
        command = [
            self.ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration:format=duration",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PerceptualHashUnavailable(
                f"bounded ffprobe for pHash failed: {exc}"
            ) from exc
        if result.returncode != 0:
            raise PerceptualHashUnavailable(f"ffprobe could not safely inspect {path}")
        try:
            doc = json.loads(result.stdout or "{}")
            streams = doc.get("streams") or []
            raw = streams[0].get("duration") if streams else None
            if raw in (None, "N/A", ""):
                raw = (doc.get("format") or {}).get("duration")
            duration = float(raw)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise PerceptualHashUnavailable(
                f"ffprobe returned no finite video duration for {path}"
            ) from exc
        if not math.isfinite(duration) or duration <= 0.0:
            raise PerceptualHashUnavailable(
                f"ffprobe returned invalid video duration {duration!r} for {path}"
            )
        return duration

    @staticmethod
    def _frame_hash(frame: bytes) -> int:
        """Standard low-frequency 32x32 DCT pHash, returned as 64 bits."""
        if len(frame) != 32 * 32:
            raise ValueError("a pHash frame must be exactly 32x32 grayscale bytes")
        cosines = [
            [math.cos(math.pi * (2 * x + 1) * frequency / 64.0) for x in range(32)]
            for frequency in range(8)
        ]
        coefficients: list[float] = []
        for vertical in range(8):
            for horizontal in range(8):
                total = 0.0
                for y in range(32):
                    row = y * 32
                    cy = cosines[vertical][y]
                    total += cy * sum(
                        frame[row + x] * cosines[horizontal][x] for x in range(32)
                    )
                coefficients.append(total)
        ordered = sorted(coefficients[1:])
        midpoint = len(ordered) // 2
        threshold = (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
        value = 0
        for coefficient in coefficients:
            value = (value << 1) | int(coefficient > threshold)
        return value

    @staticmethod
    def distance(hash_a: str, hash_b: str) -> int:
        if len(hash_a) != 16 or len(hash_b) != 16:
            raise ValueError(
                "CPU video pHash values must be 16 lowercase hex characters"
            )
        try:
            left = int(hash_a, 16)
            right = int(hash_b, 16)
        except ValueError as exc:
            raise ValueError("CPU video pHash values must be hexadecimal") from exc
        return (left ^ right).bit_count()


__all__ = [
    "CPU_VIDEO_PHASH_VERSION",
    "FFMPEG_PHASH_MAX_ALLOC_BYTES",
    "CpuVideoPhash",
    "PerceptualHashUnavailable",
]
