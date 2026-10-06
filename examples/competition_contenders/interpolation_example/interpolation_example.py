"""Frame-interpolation example contender: one item in, one item out.

Usage: python3 interpolation_example.py [--max-bytes N] INPUT_PATH OUTPUT_PATH

The input is a clip with every other frame removed (one video stream). The output is
the same clip at twice the frame rate: every input frame unchanged at the even
positions and one new frame between each pair, so N input frames give 2N - 1 output
frames. This example inserts the rounded per-plane average of the two neighbours (the
"blend" the validator measures for free), so it scores about zero by design. Replace
``insert_frame`` with a real interpolation model to earn.

Frames are streamed through ffmpeg one at a time (raw yuv420p), so memory stays at a
few frames whatever the clip length. The output is H.264 at a near-lossless CRF so the
input frames come back unchanged within the scorer's tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np


def probe(path: str) -> tuple[int, int, Fraction, int]:
    """(width, height, frame rate, frame count) of the first video stream."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
            "-of", "json", path,
        ],
        capture_output=True, text=True, check=True, timeout=300,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return (
        int(stream["width"]),
        int(stream["height"]),
        Fraction(stream["r_frame_rate"]),
        int(stream["nb_read_frames"]),
    )


def insert_frame(previous: np.ndarray, following: np.ndarray) -> np.ndarray:
    """The new frame halfway between two input frames (raw yuv420p planes, flat uint8).

    This is the free blend; a real entry estimates motion here instead."""
    return ((previous.astype(np.uint16) + following + 1) >> 1).astype(np.uint8)


def interpolate(src: str, dst: str, *, crf: int) -> None:
    width, height, rate, frames = probe(src)
    if frames < 2:
        raise SystemExit(f"{src}: needs at least two frames, has {frames}")
    frame_bytes = width * height * 3 // 2
    out_rate = rate * 2
    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", src, "-map", "0:v:0",
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        stdout=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
         "-s", f"{width}x{height}", "-r", f"{out_rate.numerator}/{out_rate.denominator}",
         "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", dst],
        stdin=subprocess.PIPE,
    )
    assert decoder.stdout is not None and encoder.stdin is not None
    written = 0
    previous = None
    try:
        while True:
            chunk = decoder.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            current = np.frombuffer(chunk, dtype=np.uint8)
            if previous is not None:
                encoder.stdin.write(insert_frame(previous, current).tobytes())
                written += 1
            encoder.stdin.write(chunk)
            written += 1
            previous = current
    finally:
        encoder.stdin.close()
        decoder.stdout.close()
        decoder.wait(timeout=60)
        encoder.wait(timeout=600)
    if decoder.returncode != 0 or encoder.returncode != 0:
        raise SystemExit(f"{src}: ffmpeg failed (decode {decoder.returncode}, encode {encoder.returncode})")
    if written != 2 * frames - 1:
        raise SystemExit(f"{src}: wrote {written} frames, expected {2 * frames - 1}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-bytes", type=int, default=500_000_000)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.output))
    fd, partial = tempfile.mkstemp(prefix=".interpolation-partial-", dir=out_dir)
    os.close(fd)
    try:
        # near-lossless first; a coarser CRF only when the file would not fit the budget
        for crf in (8, 14, 20):
            interpolate(args.input, partial, crf=crf)
            if os.path.getsize(partial) <= args.max_bytes:
                break
        else:
            raise SystemExit(f"{args.input}: output exceeds {args.max_bytes} bytes at every CRF")
        os.chmod(partial, 0o644)
        os.replace(partial, args.output)
        partial = ""
    finally:
        if partial and os.path.exists(partial):
            os.unlink(partial)
    return 0


if __name__ == "__main__":
    sys.exit(main())
