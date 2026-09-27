#!/usr/bin/env python3
"""Object-removal example: temporal-median fill plus Telea inpainting (CPU, deterministic).

    python3 removal_example.py INPUT OUTPUT [--max-bytes N]

INPUT is one competition item: a Matroska file with two video streams. Stream 0 is the
clip with an object on it, stream 1 is a per-frame mask (gray, luma > 127 = pixels to
reconstruct, 0 = keep). OUTPUT receives an MP4 with one H.264 stream of the same width,
height, frame count and frame rate in which only the masked pixels were replaced.

The method (the free fill every entry has to beat):

1. Decode stream 0 as planar YUV 4:2:0 (no colour conversion, so unmasked pixels are
   carried bit for bit) and stream 1 as 8-bit gray masks.
2. Every masked pixel takes the temporal median of the same pixel over all frames in
   which it is NOT masked. With a static camera this recovers the background; with
   camera or scene motion it ghosts, which is exactly what a better entry fixes.
3. Pixels masked in every frame have no clean sample. They are filled per frame with
   OpenCV's Telea inpainting (radius 5) from the median-filled frame around them.
4. Frames whose mask is empty are passed through unchanged.
5. The frames are encoded losslessly with libx264 (-qp 0, all-intra -g 1), so the
   decoded output is exactly the composited frames: pixels outside the mask equal the
   input bit for bit. Lossless H.264 in 4:2:0 is the "High 4:4:4 Predictive" profile;
   ffmpeg/libavcodec (what the scorer decodes with) reads it, many hardware and
   browser decoders do not. All-intra is about twice as fast as a normal GOP and only
   a few percent larger when lossless. If the file is larger than --max-bytes (run.sh
   passes a share of the batch's 2 GiB output cap), the clip is re-encoded with
   -crf 10, near-lossless (mean abs RGB change about 1 on heavy grain; the scorer
   tolerates 3.0 outside the mask).

Chroma: in 4:2:0 one chroma sample covers 2x2 luma pixels. A chroma sample counts as
masked when any of its four luma pixels is masked, so the object's colour does not
bleed into the fill along the mask edge. The unmasked pixels of such an edge block get
the background chroma instead, a change far below the scorer's outside-mask tolerance.

Exit status: 0 when OUTPUT was written, 1 otherwise. On failure no OUTPUT is left
behind. The encode goes to a hidden temporary file next to OUTPUT and is renamed into
place only after the result has been checked, so OUTPUT is never a partial file.
Logs go to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import cv2
import numpy as np

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")

MASK_THRESHOLD = 127  # mask luma > 127 means "reconstruct this pixel" (track contract)
INPAINT_RADIUS = 5  # Telea neighbourhood, in pixels of the plane being filled
BAND_ROWS = 64  # rows per median band; bounds the temporary to frames x 64 x width
X264_PRESET = "veryfast"  # lossless either way; the preset only trades size for speed
LOSSLESS = ("-qp", "0", "-g", "1")  # bit-exact, all-intra
FALLBACK = ("-crf", "10")  # near-lossless, normal GOP: only when lossless is too big
PARTIAL_PREFIX = ".removal-partial-"  # run.sh deletes leftovers with this prefix

# Colour tags copied from the input so the output is interpreted exactly like it.
# Only values ffmpeg is known to accept are forwarded; anything else is left unset.
_COLOUR_TAGS = (
    ("color_range", "-color_range", {"tv", "pc"}),
    (
        "color_space",
        "-colorspace",
        {"bt709", "smpte170m", "bt470bg", "bt2020nc", "bt2020c", "smpte240m", "fcc", "ycgco"},
    ),
    (
        "color_primaries",
        "-color_primaries",
        {"bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "film", "bt2020",
         "smpte428", "smpte431", "smpte432", "jedec-p22"},
    ),
    (
        "color_transfer",
        "-color_trc",
        {"bt709", "gamma22", "gamma28", "smpte170m", "smpte240m", "linear",
         "iec61966-2-1", "iec61966-2-4", "bt1361e", "bt2020-10", "bt2020-12",
         "smpte2084", "smpte428", "arib-std-b67", "log100", "log316"},
    ),
)


class ItemError(RuntimeError):
    """This item cannot be processed; no output is written for it."""


class Terminated(BaseException):
    """SIGTERM (run.sh's per-item timeout) arrived; unwind and clean up."""


def log(message: str) -> None:
    print(f"removal_example: {message}", file=sys.stderr, flush=True)


# --- probing ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    width: int
    height: int
    rate: str  # "num/den", passed to the encoder unchanged
    colour_args: tuple[str, ...]


def _parse_rate(value: object) -> str | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    num, _, den = value.partition("/")
    if not (num.isdigit() and den.isdigit()) or int(num) == 0 or int(den) == 0:
        return None
    return f"{int(num)}/{int(den)}"


def probe(path: str) -> Clip:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v",
         "-show_entries",
         "stream=index,width,height,r_frame_rate,avg_frame_rate,"
         "color_range,color_space,color_primaries,color_transfer",
         "-of", "json", path],
        capture_output=True, timeout=120,
    )
    if proc.returncode != 0:
        raise ItemError(f"ffprobe failed: {proc.stderr[-300:]!r}")
    streams = json.loads(proc.stdout or b"{}").get("streams") or []
    if len(streams) < 2:
        raise ItemError(f"expected 2 video streams (clip + mask), found {len(streams)}")
    video, mask = streams[0], streams[1]
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ItemError("video stream has no geometry")
    if (int(mask.get("width") or 0), int(mask.get("height") or 0)) != (width, height):
        raise ItemError(
            f"mask geometry {mask.get('width')}x{mask.get('height')} != video {width}x{height}"
        )
    if width % 2 or height % 2:
        raise ItemError(f"odd geometry {width}x{height} cannot be carried as H.264 4:2:0")
    rate = _parse_rate(video.get("r_frame_rate")) or _parse_rate(video.get("avg_frame_rate"))
    if rate is None:
        raise ItemError("video stream has no usable frame rate")
    colour: list[str] = []
    for key, option, allowed in _COLOUR_TAGS:
        value = video.get(key)
        if value in allowed:
            colour += [option, value]
    return Clip(width, height, rate, tuple(colour))


# --- decoding --------------------------------------------------------------------------


def _read_full(stream, buf: bytearray) -> int:
    view = memoryview(buf)
    got = 0
    while got < len(buf):
        n = stream.readinto(view[got:])
        if not n:
            break
        got += n
    return got


def decode_stream(path: str, index: int, pix_fmt: str, frame_bytes: int) -> list[bytearray]:
    """All frames of video stream `index` as raw `pix_fmt` buffers (one per frame)."""
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path,
           "-map", f"0:v:{index}", "-an", "-sn", "-dn", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]
    frames: list[bytearray] = []
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                buf = bytearray(frame_bytes)
                got = _read_full(proc.stdout, buf)
                if got == 0:
                    break
                if got != frame_bytes:
                    raise ItemError(f"stream {index}: truncated frame ({got}/{frame_bytes} bytes)")
                frames.append(buf)
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
        if code != 0:
            err.seek(0)
            raise ItemError(f"decoding stream {index} failed: {err.read()[-300:]!r}")
    return frames


def yuv_planes(buf: bytearray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Writable Y, U, V views into one yuv420p frame buffer."""
    a = np.frombuffer(buf, dtype=np.uint8)
    luma = width * height
    chroma = luma // 4
    y = a[:luma].reshape(height, width)
    u = a[luma:luma + chroma].reshape(height // 2, width // 2)
    v = a[luma + chroma:].reshape(height // 2, width // 2)
    return y, u, v


# --- the fill --------------------------------------------------------------------------


def masked_median(values: np.ndarray, invalid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column median of the valid samples.

    `values` and `invalid` are (frames, pixels). Returns the median as uint8 (the mean of
    the two middle samples for an even count, rounded half up; identical to
    np.nanmedian with NaN at the invalid samples, then rounded) and a bool array telling
    which pixels had at least one valid sample. Sorting uint16 with an out-of-range
    sentinel is several times faster than np.nanmedian and bit-identical.
    """
    frames = values.shape[0]
    work = np.ascontiguousarray(values.T, dtype=np.uint16)  # (pixels, frames)
    work[invalid.T] = 256  # sorts after every real sample
    work.sort(axis=1)
    count = frames - invalid.sum(axis=0)  # valid samples per pixel
    has = count > 0
    lo = np.take_along_axis(work, (np.maximum(count - 1, 0) // 2)[:, None], axis=1)[:, 0]
    hi = np.take_along_axis(work, (count // 2)[:, None].clip(max=frames - 1), axis=1)[:, 0]
    med = (lo.astype(np.uint32) + hi + 1) // 2
    return np.where(has, med, 0).astype(np.uint8), has


def _bbox(region: np.ndarray, pad: int = 0) -> tuple[int, int, int, int]:
    rows = np.flatnonzero(region.any(axis=1))
    cols = np.flatnonzero(region.any(axis=0))
    h, w = region.shape
    return (max(0, rows[0] - pad), min(h, rows[-1] + 1 + pad),
            max(0, cols[0] - pad), min(w, cols[-1] + 1 + pad))


def fill_plane(planes: list[np.ndarray], masks: list[np.ndarray]) -> int:
    """Fill the masked pixels of one plane in every frame, in place.

    Returns the number of pixels that were masked in every frame (Telea-filled).
    """
    union = np.zeros(planes[0].shape, dtype=bool)
    for m in masks:
        union |= m
    if not union.any():
        return 0
    y0, y1, x0, x1 = _bbox(union)

    # Step 2: temporal median of the clean samples, band by band to bound memory.
    fill = np.zeros(union.shape, dtype=np.uint8)
    hole = np.zeros(union.shape, dtype=bool)  # masked in every frame: no clean sample
    for b0 in range(y0, y1, BAND_ROWS):
        b1 = min(y1, b0 + BAND_ROWS)
        sel = union[b0:b1, x0:x1]
        if not sel.any():
            continue
        values = np.stack([p[b0:b1, x0:x1][sel] for p in planes])
        invalid = np.stack([m[b0:b1, x0:x1][sel] for m in masks])
        med, has = masked_median(values, invalid)
        fill[b0:b1, x0:x1][sel] = med
        hole[b0:b1, x0:x1][sel] = ~has

    # Composite: only masked pixels change; empty-mask frames are skipped entirely.
    box_fill = fill[y0:y1, x0:x1]
    box_clean = ~hole[y0:y1, x0:x1]
    for p, m in zip(planes, masks):
        take = m[y0:y1, x0:x1] & box_clean
        if take.any():
            p[y0:y1, x0:x1][take] = box_fill[take]

    # Step 3: pixels masked in every frame get Telea inpainting per frame, from the
    # median-filled frame around them (a crop with enough context for the radius).
    holes = int(hole.sum())
    if holes:
        if hole.all():  # nothing to inpaint from; a neutral constant is all that is left
            for p in planes:
                p[...] = 128
            return holes
        h0, h1, w0, w1 = _bbox(hole, pad=3 * INPAINT_RADIUS)
        hole_crop = hole[h0:h1, w0:w1]
        hole_u8 = hole_crop.astype(np.uint8)
        for p in planes:
            crop = np.ascontiguousarray(p[h0:h1, w0:w1])
            painted = cv2.inpaint(crop, hole_u8, INPAINT_RADIUS, cv2.INPAINT_TELEA)
            p[h0:h1, w0:w1][hole_crop] = painted[hole_crop]
    return holes


# --- encoding --------------------------------------------------------------------------


def encode(frames: list[bytearray], clip: Clip, path: str, quality: tuple[str, ...]) -> None:
    """H.264 4:2:0 with libx264; with LOSSLESS it decodes to exactly these frames."""
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{clip.width}x{clip.height}",
           "-framerate", clip.rate, "-i", "pipe:0",
           "-map", "0:v:0", "-c:v", "libx264", "-preset", X264_PRESET, *quality,
           "-pix_fmt", "yuv420p", *clip.colour_args, "-f", "mp4", "-y", path]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        try:
            try:
                for frame in frames:
                    proc.stdin.write(frame)
            except BrokenPipeError:
                pass  # ffmpeg died; its exit status and stderr say why
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if code != 0:
            err.seek(0)
            raise ItemError(f"encoding failed: {err.read()[-300:]!r}")


def check_output(path: str, clip: Clip, frames: int) -> None:
    """The encoded file must be one H.264 stream with the input's geometry and length."""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_type,codec_name,width,height,pix_fmt,nb_read_packets",
         "-of", "json", path],
        capture_output=True, timeout=120,
    )
    streams = json.loads(proc.stdout or b"{}").get("streams") or []
    if proc.returncode != 0 or len(streams) != 1:
        raise ItemError(f"output probe failed or not one stream: {proc.stderr[-300:]!r}")
    s = streams[0]
    got = (s.get("codec_name"), s.get("width"), s.get("height"), s.get("pix_fmt"),
           int(s.get("nb_read_packets") or -1))
    want = ("h264", clip.width, clip.height, "yuv420p", frames)
    if got != want:
        raise ItemError(f"output check failed: got {got}, want {want}")


# --- one item --------------------------------------------------------------------------


def process(input_path: str, output_path: str, max_bytes: int | None = None) -> None:
    name = os.path.basename(output_path)
    started = time.monotonic()
    clip = probe(input_path)
    w, h = clip.width, clip.height

    frames = decode_stream(input_path, 0, "yuv420p", w * h * 3 // 2)
    if not frames:
        raise ItemError("video stream has no frames")
    mask_frames = decode_stream(input_path, 1, "gray", w * h)
    n = len(frames)
    if len(mask_frames) != n:
        # Contract says equal; tolerate a mismatch rather than lose the item.
        log(f"{name}: mask has {len(mask_frames)} frames, video {n}; "
            "missing masks are treated as empty, extra ones are ignored")
    empty = np.zeros((h, w), dtype=bool)
    luma_masks = [
        np.frombuffer(b, dtype=np.uint8).reshape(h, w) > MASK_THRESHOLD for b in mask_frames[:n]
    ]
    del mask_frames
    luma_masks += [empty] * (n - len(luma_masks))
    # A chroma sample is masked when any of the 2x2 luma pixels it covers is masked.
    chroma_masks = [m.reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3)) for m in luma_masks]
    decoded = time.monotonic()

    planes = [yuv_planes(f, w, h) for f in frames]
    holes = fill_plane([p[0] for p in planes], luma_masks)
    for k in (1, 2):
        fill_plane([p[k] for p in planes], chroma_masks)
    masked_frames = sum(1 for m in luma_masks if m.any())
    masked_fraction = float(np.mean([m.mean() for m in luma_masks]))
    del planes, luma_masks, chroma_masks
    filled = time.monotonic()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    partial = os.path.join(out_dir, f"{PARTIAL_PREFIX}{name}.{os.getpid()}.mp4")
    try:
        encode(frames, clip, partial, LOSSLESS)
        check_output(partial, clip, n)
        mode = "lossless"
        if max_bytes is not None and os.path.getsize(partial) > max_bytes:
            log(f"{name}: lossless output is {os.path.getsize(partial)} bytes, over the "
                f"{max_bytes}-byte allowance; re-encoding with {' '.join(FALLBACK)}")
            encode(frames, clip, partial, FALLBACK)
            check_output(partial, clip, n)
            mode = "crf fallback"
            if os.path.getsize(partial) > max_bytes:
                raise ItemError(f"output is {os.path.getsize(partial)} bytes even with "
                                f"{' '.join(FALLBACK)}, over the {max_bytes}-byte allowance")
        os.chmod(partial, 0o644)
        os.replace(partial, output_path)  # same directory: atomic
    finally:
        if os.path.lexists(partial):
            os.unlink(partial)
    done = time.monotonic()
    log(f"{name}: {n} frames {w}x{h} @ {clip.rate}; {masked_frames} masked frames, "
        f"{masked_fraction:.2%} of pixels masked, {holes} never-clean luma px; "
        f"decode {decoded - started:.1f}s fill {filled - decoded:.1f}s "
        f"encode {done - filled:.1f}s; {mode}, {os.path.getsize(output_path)} bytes")


def _on_sigterm(signum, frame):  # noqa: ARG001
    raise Terminated(f"signal {signum}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Object-removal example for one item.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--max-bytes", type=int, default=None,
                        help="largest acceptable output; above it the clip is re-encoded near-losslessly")
    args = parser.parse_args(argv[1:])
    signal.signal(signal.SIGTERM, _on_sigterm)
    input_path, output_path = args.input, args.output
    try:
        process(input_path, output_path, args.max_bytes)
        return 0
    except Terminated as exc:
        log(f"{os.path.basename(output_path)}: stopped ({exc}); no output written")
    except Exception as exc:  # any failure costs this item only
        log(f"{os.path.basename(output_path)}: FAILED: {type(exc).__name__}: {exc}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
