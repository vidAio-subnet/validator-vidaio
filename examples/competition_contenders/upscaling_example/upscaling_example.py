"""Upscaling example contender: one item in, one item out.

Usage: python3 upscaling_example.py [--max-bytes N] INPUT_PATH OUTPUT_PATH

The input is a degraded, downscaled clip; the hidden file
``.vidaio-next-upscale-task-<input name>`` next to it holds the task as JSON
(``upscale_factor``, ``target_width``, ``target_height``). The output is the clip at
exactly the target size with the input's frame count and frame rate.

This example upscales with ffmpeg's Lanczos filter: the free answer the validator
measures as the floor, so it scores about zero by design. Replace ``upscale_filter``
(or the whole ffmpeg step, decoding raw frames into your model as the
frame-interpolation example does) with a real restoration model to earn. The output is
H.264 at a high-quality CRF, tagged BT.709 limited range like the scorer's frames.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

SIDECAR_PREFIX = ".vidaio-next-upscale-task-"


def task(input_path: str) -> tuple[int, int, int]:
    """(upscale factor, target width, target height) from the hidden task file."""
    folder, name = os.path.split(os.path.abspath(input_path))
    with open(os.path.join(folder, SIDECAR_PREFIX + name), encoding="utf-8") as fh:
        doc = json.load(fh)
    return int(doc["upscale_factor"]), int(doc["target_width"]), int(doc["target_height"])


def frame_count(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "json", path],
        capture_output=True, text=True, check=True, timeout=300,
    )
    return int(json.loads(out.stdout)["streams"][0]["nb_read_frames"])


def upscale_filter(width: int, height: int) -> str:
    """The ffmpeg filter that makes one output frame (the free Lanczos answer)."""
    return f"scale={width}:{height}:flags=lanczos"


def upscale(src: str, dst: str, *, width: int, height: int, crf: int) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", src, "-map", "0:v:0", "-an", "-sn", "-dn",
         # every input frame once, at its own timestamp: never resample the frame rate
         "-fps_mode", "passthrough",
         "-vf", f"{upscale_filter(width, height)},format=yuv420p",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-color_range", "tv", "-movflags", "+faststart", "-f", "mp4", dst],
        check=True, timeout=3600,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-bytes", type=int, default=500_000_000)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    _factor, width, height = task(args.input)
    expected = frame_count(args.input)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    fd, partial = tempfile.mkstemp(prefix=".upscaling-partial-", dir=out_dir)
    os.close(fd)
    try:
        # high quality first; a coarser CRF only when the file would not fit the budget
        for crf in (10, 16, 22):
            upscale(args.input, partial, width=width, height=height, crf=crf)
            if os.path.getsize(partial) <= args.max_bytes:
                break
        else:
            raise SystemExit(f"{args.input}: output exceeds {args.max_bytes} bytes at every CRF")
        got = frame_count(partial)
        if got != expected:
            raise SystemExit(f"{args.input}: wrote {got} frames, expected {expected}")
        os.chmod(partial, 0o644)
        os.replace(partial, args.output)
        partial = ""
    finally:
        if partial and os.path.exists(partial):
            os.unlink(partial)
    return 0


if __name__ == "__main__":
    sys.exit(main())
