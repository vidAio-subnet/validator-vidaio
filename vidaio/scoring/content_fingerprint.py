"""Versioned CPU evidence over the *measured* canonical Y4M stream.

Version 1 hashes every stream byte, including headers, and samples exactly 32
frames at floor(i*(N-1)/31). Repeated positions on short clips are intentional.
The fingerprint uses the original Y plane, exact area averaging to 32x32
(nearest integer, halves upward), and a separable, unnormalized DCT-II with the
fixed Q20 cosine table below. Integer arithmetic makes reductions independent
of BLAS, GPU kernels, libm, thread count and platform floating-point behavior.
The median is AC[31] after sorting all 63 AC coefficients; all 64 row-major
coefficients, including DC, use a strict greater-than comparison.

This is distinct from the public-corpus video pHash. Do not change that
primitive or substitute encoded-file bytes for a missing canonical stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import stat
from typing import BinaryIO, Callable

CONTENT_FINGERPRINT_VERSION = 1
FINGERPRINT_FRAMES = 32
_READ_BYTES = 64 * 1024
_HEADER_BYTES = 4096

# round(2**20 * cos(pi*(2*x+1)*k/64)), k=0..7, x=0..31. These integers
# are the algorithm, not a table regenerated using the host's math library.
_DCT = (
    (1048576,) * 32,
    (1047313, 1037227, 1017151, 987281, 947901, 899394, 842224, 776944, 704181, 624636, 539076, 448324, 353255, 254783, 153858, 51451, -51451, -153858, -254783, -353255, -448324, -539076, -624636, -704181, -776944, -842224, -899394, -947901, -987281, -1017151, -1037227, -1047313),
    (1043527, 1003425, 924761, 810560, 665210, 494295, 304386, 102778, -102778, -304386, -494295, -665210, -810560, -924761, -1003425, -1043527, -1043527, -1003425, -924761, -810560, -665210, -494295, -304386, -102778, 102778, 304386, 494295, 665210, 810560, 924761, 1003425, 1043527),
    (1037227, 947901, 776944, 539076, 254783, -51451, -353255, -624636, -842224, -987281, -1047313, -1017151, -899394, -704181, -448324, -153858, 153858, 448324, 704181, 899394, 1017151, 1047313, 987281, 842224, 624636, 353255, 51451, -254783, -539076, -776944, -947901, -1037227),
    (1028428, 871859, 582558, 204567, -204567, -582558, -871859, -1028428, -1028428, -871859, -582558, -204567, 204567, 582558, 871859, 1028428, 1028428, 871859, 582558, 204567, -204567, -582558, -871859, -1028428, -1028428, -871859, -582558, -204567, 204567, 582558, 871859, 1028428),
    (1017151, 776944, 353255, -153858, -624636, -947901, -1047313, -899394, -539076, -51451, 448324, 842224, 1037227, 987281, 704181, 254783, -254783, -704181, -987281, -1037227, -842224, -448324, 51451, 539076, 899394, 1047313, 947901, 624636, 153858, -353255, -776944, -1017151),
    (1003425, 665210, 102778, -494295, -924761, -1043527, -810560, -304386, 304386, 810560, 1043527, 924761, 494295, -102778, -665210, -1003425, -1003425, -665210, -102778, 494295, 924761, 1043527, 810560, 304386, -304386, -810560, -1043527, -924761, -494295, 102778, 665210, 1003425),
    (987281, 539076, -153858, -776944, -1047313, -842224, -254783, 448324, 947901, 1017151, 624636, -51451, -704181, -1037227, -899394, -353255, 353255, 899394, 1037227, 704181, 51451, -624636, -1017151, -947901, -448324, 254783, 842224, 1047313, 776944, 153858, -539076, -987281),
)


class ContentFingerprintUnavailable(RuntimeError):
    """The measured canonical stream cannot provide honest content evidence."""


class ContentFingerprintCancelled(ContentFingerprintUnavailable):
    """The request was cancelled while reading or hashing its canonical stream."""


@dataclass(frozen=True)
class CanonicalContentEvidence:
    canonical_content_digest: str
    content_fingerprint: tuple[str, ...]
    frame_count: int
    sample_positions: tuple[int, ...]


def sample_positions(frame_count: int) -> tuple[int, ...]:
    if type(frame_count) is not int or frame_count < 1:
        raise ContentFingerprintUnavailable("canonical content has no positive frame count")
    return tuple(i * (frame_count - 1) // 31 for i in range(FINGERPRINT_FRAMES))


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise ContentFingerprintCancelled("canonical content fingerprint cancelled")


def _line(stream: BinaryIO, *, eof: bool = False) -> bytes:
    line = stream.readline(_HEADER_BYTES + 1)
    if eof and not line:
        return b""
    if len(line) > _HEADER_BYTES or not line.endswith(b"\n"):
        raise ContentFingerprintUnavailable("invalid or oversized canonical Y4M header")
    if any(c < 32 or c > 126 for c in line[:-1]):
        raise ContentFingerprintUnavailable("non-ASCII canonical Y4M header")
    return line


def _header(stream: BinaryIO) -> tuple[bytes, int, int, int]:
    header = _line(stream)
    parts = header[:-1].split(b" ")
    if not parts or parts[0] != b"YUV4MPEG2":
        raise ContentFingerprintUnavailable("content fingerprint requires canonical Y4M")
    fields: dict[bytes, bytes] = {}
    for part in parts[1:]:
        if not part:
            raise ContentFingerprintUnavailable("empty canonical Y4M header token")
        key = part[:1]
        if key in (b"W", b"H", b"C"):
            if key in fields:
                raise ContentFingerprintUnavailable("duplicate canonical Y4M geometry token")
            fields[key] = part[1:]
    if any(not fields.get(key, b"").isdigit() for key in (b"W", b"H")):
        raise ContentFingerprintUnavailable("canonical Y4M geometry is missing or invalid")
    width, height = int(fields[b"W"]), int(fields[b"H"])
    if width < 1 or height < 1:
        raise ContentFingerprintUnavailable("canonical Y4M dimensions must be positive")
    if fields.get(b"C") not in (b"420", b"420jpeg", b"420mpeg2", b"420paldv"):
        raise ContentFingerprintUnavailable("content fingerprint requires 8-bit yuv420p")
    frame_bytes = width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)
    return header, width, height, frame_bytes


def _frame_header(stream: BinaryIO, *, eof: bool = False) -> bytes:
    line = _line(stream, eof=eof)
    if line and line != b"FRAME\n" and not line.startswith(b"FRAME "):
        raise ContentFingerprintUnavailable("invalid canonical Y4M frame marker")
    return line


def _consume(stream: BinaryIO, count: int, cancelled: Callable[[], bool] | None,
             digest: object | None = None) -> int:
    """Read bounded chunks, optionally hashing them; return their integer sum."""
    total = 0
    while count:
        _check_cancelled(cancelled)
        chunk = stream.read(min(count, _READ_BYTES))
        if not chunk:
            raise ContentFingerprintUnavailable("truncated canonical Y4M frame")
        if digest is not None:
            digest.update(chunk)
        else:
            total += sum(chunk)
        count -= len(chunk)
    return total


def _area_luma(stream: io.BufferedReader, width: int, height: int,
               cancelled: Callable[[], bool] | None) -> bytes:
    """Exact rational area resize, with bounded memory even for a very wide row."""
    pixels = [[0] * 32 for _ in range(32)]
    boundaries = [divmod(i * width, 32) for i in range(33)]
    for y in range(height):
        _check_cancelled(cancelled)
        integrals, consumed, cumulative = [], 0, 0
        for stop, fraction in boundaries:
            cumulative += _consume(stream, stop - consumed, cancelled)
            consumed = stop
            value = 32 * cumulative
            if fraction:
                next_byte = stream.peek(1)
                if not next_byte:
                    raise ContentFingerprintUnavailable("truncated canonical luma plane")
                value += fraction * next_byte[0]
            integrals.append(value)
        horizontal = [b - a for a, b in zip(integrals, integrals[1:])]
        first = y * 32 // height
        last = min(31, ((y + 1) * 32 - 1) // height)
        for target in range(first, last + 1):
            weight = min((target + 1) * height, (y + 1) * 32) - max(target * height, y * 32)
            for x in range(32):
                pixels[target][x] += horizontal[x] * weight
    denominator = width * height
    return bytes((2 * value + denominator) // (2 * denominator)
                 for row in pixels for value in row)


def frame_phash(luma_32x32: bytes) -> str:
    """One version-1 64-bit DCT word, in row-major frequency order including DC."""
    if len(luma_32x32) != 1024:
        raise ValueError("content pHash requires exactly 32x32 luma bytes")
    horizontal = [[sum(luma_32x32[y * 32 + x] * basis[x] for x in range(32))
                   for basis in _DCT] for y in range(32)]
    coefficients = [sum(horizontal[y][u] * _DCT[v][y] for y in range(32))
                    for v in range(8) for u in range(8)]
    median = sorted(coefficients[1:])[31]
    bits = 0
    for value in coefficients:
        bits = (bits << 1) | int(value > median)
    return f"{bits:016x}"


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def compute_canonical_content(path: str | Path, *,
                              cancelled: Callable[[], bool] | None = None) -> CanonicalContentEvidence:
    """Hash the whole measured Y4M and fingerprint its exact 32 frame positions.

    The first bounded pass verifies framing, hashes every byte and counts frames.
    The second samples at most 32 distinct frames. The same no-follow regular-file
    descriptor remains open throughout; file identity/content metadata must hold.
    No decoder, model, subprocess, network, temporary output or whole-file buffer.
    """
    _check_cancelled(cancelled)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ContentFingerprintUnavailable("canonical Y4M is not a regular file")
            header, width, height, frame_bytes = _header(stream)
            if frame_bytes > before.st_size:
                raise ContentFingerprintUnavailable("canonical frame exceeds its stream size")
            digest = hashlib.sha256(header)
            count = 0
            while marker := _frame_header(stream, eof=True):
                digest.update(marker)
                _consume(stream, frame_bytes, cancelled, digest)
                count += 1
            positions = sample_positions(count)
            stream.seek(len(header))
            wanted = set(positions)
            hashes: dict[int, str] = {}
            for index in range(count):
                _check_cancelled(cancelled)
                _frame_header(stream)
                if index in wanted:
                    hashes[index] = frame_phash(_area_luma(stream, width, height, cancelled))
                    stream.seek(frame_bytes - width * height, os.SEEK_CUR)
                else:
                    stream.seek(frame_bytes, os.SEEK_CUR)
            if stream.read(1) or _identity(os.fstat(stream.fileno())) != _identity(before):
                raise ContentFingerprintUnavailable("canonical Y4M changed during fingerprinting")
            return CanonicalContentEvidence(digest.hexdigest(), tuple(hashes[i] for i in positions), count, positions)
    except OSError as exc:
        raise ContentFingerprintUnavailable("cannot read the measured canonical Y4M") from exc


__all__ = ["CONTENT_FINGERPRINT_VERSION", "CanonicalContentEvidence", "ContentFingerprintUnavailable",
           "ContentFingerprintCancelled", "compute_canonical_content", "sample_positions", "frame_phash"]
