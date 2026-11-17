"""Generative upscaling example contender: one item in, one item out (GPU when present).

Usage: python generative_example.py [--max-bytes N] INPUT_PATH OUTPUT_PATH

The input is a degraded, downscaled clip; the hidden file
``.vidaio-next-upscale-task-<input name>`` next to it holds the task as JSON
(``upscale_factor``, ``target_width``, ``target_height``). The output is the clip at
exactly the target size with the input's frame count and frame rate.

The pipeline is the one a real generative entry uses: ffmpeg decodes raw RGB frames into a
PyTorch model on the GPU, and the model's frames are piped to the encoder. The "model" here
is a stub (bicubic upscale plus a fixed, weight-free detail filter) so the example builds and
runs without downloading anything; replace :class:`StubGenerator` with your network and load
its weights from ``/app/weights`` (see the Dockerfile). Process frames in temporal windows if
your model is a video model; keep every frame, in order.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

SIDECAR_PREFIX = ".vidaio-next-upscale-task-"


def task(input_path: str) -> tuple[int, int, int]:
    """(upscale factor, target width, target height) from the hidden task file."""
    folder, name = os.path.split(os.path.abspath(input_path))
    with open(os.path.join(folder, SIDECAR_PREFIX + name), encoding="utf-8") as fh:
        doc = json.load(fh)
    return int(doc["upscale_factor"]), int(doc["target_width"]), int(doc["target_height"])


def probe(path: str) -> tuple[int, int, str, int]:
    """(width, height, frame rate as 'num/den', frame count) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", path],
        capture_output=True, text=True, check=True, timeout=600,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"]), stream["r_frame_rate"], int(stream["nb_read_frames"])


class StubGenerator(torch.nn.Module):
    """Stand-in for a generative upscaler: bicubic to the target size, then a fixed
    unsharp-mask "detail" filter. A real entry replaces this module (and its weights)."""

    def __init__(self, amount: float = 0.35) -> None:
        super().__init__()
        g = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = (g[:, None] * g[None, :]) / 256.0
        self.register_buffer("blur", kernel.expand(3, 1, 5, 5).contiguous())
        self.amount = amount

    @torch.no_grad()
    def forward(self, frames: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """``frames``: (n, 3, h, w) in [0, 1] -> (n, 3, H, W) in [0, 1]."""
        up = F.interpolate(frames, size=size, mode="bicubic", align_corners=False)
        soft = F.conv2d(F.pad(up, (2, 2, 2, 2), mode="reflect"), self.blur, groups=3)
        return (up + self.amount * (up - soft)).clamp_(0.0, 1.0)


def upscale(src: str, dst: str, *, width: int, height: int, crf: int, device: torch.device,
            batch: int = 8) -> int:
    in_w, in_h, rate, _ = probe(src)
    model = StubGenerator().to(device).eval()
    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", src, "-map", "0:v:0", "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-framerate", rate, "-i", "-",
         "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-color_range", "tv", "-movflags", "+faststart", "-f", "mp4", dst],
        stdin=subprocess.PIPE,
    )
    assert decoder.stdout is not None and encoder.stdin is not None
    frame_bytes = in_w * in_h * 3
    written = 0
    try:
        while True:
            chunk = decoder.stdout.read(frame_bytes * batch)
            if not chunk:
                break
            n = len(chunk) // frame_bytes
            x = torch.frombuffer(bytearray(chunk[: n * frame_bytes]), dtype=torch.uint8)
            x = x.view(n, in_h, in_w, 3).permute(0, 3, 1, 2).to(device).float().div_(255.0)
            y = model(x, (height, width))
            out = y.mul_(255.0).round_().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu()
            encoder.stdin.write(out.numpy().tobytes())
            written += n
    finally:
        encoder.stdin.close()
        decoder.stdout.close()
        if encoder.wait(timeout=3600) != 0 or decoder.wait(timeout=600) != 0:
            raise SystemExit(f"{src}: ffmpeg failed (decoder {decoder.returncode}, encoder {encoder.returncode})")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-bytes", type=int, default=500_000_000)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    _factor, width, height = task(args.input)
    expected = probe(args.input)[3]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("warning: no CUDA device, running on CPU", file=sys.stderr)
        device = torch.device("cpu")
    out_dir = os.path.dirname(os.path.abspath(args.output))
    fd, partial = tempfile.mkstemp(prefix=".generative-partial-", dir=out_dir)
    os.close(fd)
    try:
        # high quality first; a coarser CRF only when the file would not fit the budget
        for crf in (12, 16, 20):
            upscale(args.input, partial, width=width, height=height, crf=crf, device=device)
            if os.path.getsize(partial) <= args.max_bytes:
                break
        else:
            raise SystemExit(f"{args.input}: output exceeds {args.max_bytes} bytes at every CRF")
        got = probe(partial)[3]
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
