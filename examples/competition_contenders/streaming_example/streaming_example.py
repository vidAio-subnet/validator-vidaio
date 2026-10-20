"""Streaming compression example contender: one item in, one item out.

Usage: python3 streaming_example.py --bits-per-pixel BPP [--max-keyframe-interval S] INPUT_PATH OUTPUT_PATH

Encodes the clip as a streaming rendition inside the budget the competition's manifest
anchors: target_kbps = BPP * width * height * fps / 1000. x264 with a fixed GOP (a
keyframe every S seconds, no scene-cut keyframes in between), a VBV cap well under the
peak rule, and an average aimed a little below the budget; the result is measured from
its own packets and re-encoded lower if it does not fit. Replace the encoder settings (or
the encoder) to earn: the score is the VMAF reached within the budget.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction


def probe(path: str) -> tuple[int, int, Fraction, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", path],
        capture_output=True, text=True, check=True, timeout=300,
    )
    s = json.loads(out.stdout)["streams"][0]
    return int(s["width"]), int(s["height"]), Fraction(s["r_frame_rate"]), int(s["nb_read_frames"])


def packet_kbps(path: str, fps: float, frames: int, window_seconds: float) -> tuple[float, float]:
    """(average, peak over window_seconds) of the video packets, as the validator measures."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=size",
         "-of", "json", path], capture_output=True, text=True, check=True, timeout=300,
    )
    sizes = [int(p["size"]) for p in json.loads(out.stdout)["packets"]]
    avg = 8 * sum(sizes) / (frames / fps) / 1000
    w = max(1, round(window_seconds * fps))
    if len(sizes) <= w:
        return avg, 8 * sum(sizes) / (len(sizes) / fps) / 1000
    run = best = sum(sizes[:w])
    for i in range(w, len(sizes)):
        run += sizes[i] - sizes[i - w]
        best = max(best, run)
    return avg, 8 * best / (w / fps) / 1000


def encode(src: str, dst: str, *, kbps: float, gop: int) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", src, "-map", "0:v:0", "-an", "-sn", "-dn",
         "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
         "-b:v", f"{kbps:.0f}k", "-maxrate", f"{kbps * 1.3:.0f}k", "-bufsize", f"{kbps * 2:.0f}k",
         "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
         "-movflags", "+faststart", "-f", "mp4", dst],
        check=True, timeout=3600,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bits-per-pixel", type=float, required=True, help="from the competition manifest")
    parser.add_argument("--max-keyframe-interval", type=float, default=2.0, help="from the manifest")
    parser.add_argument("--peak-window", type=float, default=2.0, help="from the manifest")
    parser.add_argument("--peak-ratio", type=float, default=1.5, help="from the manifest")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    width, height, rate, frames = probe(args.input)
    fps = float(rate)
    budget = args.bits_per_pixel * width * height * fps / 1000
    # whole frames per GOP, never longer than the allowed interval
    gop = max(1, int(args.max_keyframe_interval * fps + 1e-9))
    out_dir = os.path.dirname(os.path.abspath(args.output))
    fd, partial = tempfile.mkstemp(prefix=".streaming-partial-", dir=out_dir)
    os.close(fd)
    try:
        for aim in (0.95, 0.88, 0.80, 0.70):
            encode(args.input, partial, kbps=budget * aim, gop=gop)
            avg, peak = packet_kbps(partial, fps, frames, args.peak_window)
            if avg <= budget and peak <= budget * args.peak_ratio:
                break
        else:
            raise SystemExit(f"{args.input}: no encode fits the {budget:.0f} kbps budget")
        os.chmod(partial, 0o644)
        os.replace(partial, args.output)
        partial = ""
    finally:
        if partial and os.path.exists(partial):
            os.unlink(partial)
    return 0


if __name__ == "__main__":
    sys.exit(main())
