#!/usr/bin/env python3
"""Local self-test for the removal example. No Docker needed.

    python3 test_local.py                                  # toy items, 320x240, 120 frames
    python3 test_local.py --size 1280x720                  # timing at the largest frame size
    python3 test_local.py --background some_clip.mp4       # real footage instead of the texture

It synthesises removal items exactly as the track serves them (Matroska, stream 0 FFV1
yuv420p with a coloured object on it, stream 1 FFV1 gray mask), runs `run.sh` on a
folder of sha256-named items plus one corrupt item, and checks for every item:

* run.sh exits 0, writes one 0644 regular file per good item and nothing for the
  corrupt one, and leaves no temporary files behind;
* the output decodes as one H.264 yuv420p stream with the same size, frame count and
  frame rate;
* outside the mask the mean abs RGB change is <= 1.0 in every frame (measured like the
  scorer: 4:2:0 frames converted with OpenCV's I420->RGB);
* frames with an empty mask are unchanged;
* inside the mask the output is closer to the clean background than the input was;
* the size guard: with a small --max-bytes the module falls back to the near-lossless
  encode (outside-mask change still <= 3.0, the scorer's tolerance), and with an
  impossible one it writes nothing.

It also times the module on each item and reports its peak memory. Needs ffmpeg and
ffprobe on PATH (or FFMPEG / FFPROBE), numpy and opencv-python-headless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True  # keep the contender directory free of __pycache__
import removal_example  # noqa: E402

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")
FPS = 24

# Runs one command and reports its wall time and the peak RSS of it and its children.
_MEASURE = (
    "import resource, subprocess, sys, time\n"
    "t = time.monotonic(); rc = subprocess.call(sys.argv[1:]); dt = time.monotonic() - t\n"
    "rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss\n"
    "rss = rss if sys.platform == 'darwin' else rss * 1024\n"
    "print(rc, dt, rss)\n"
)


def texture(w: int, h: int) -> np.ndarray:
    """A static procedural RGB texture with detail at several scales."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 128 + 60 * np.sin(xx / 7.0) * np.cos(yy / 11.0) + 30 * np.sin((xx + yy) / 23.0)
    g = 128 + 50 * np.cos(xx / 13.0 + yy / 5.0) + 20 * np.sin(yy / 17.0)
    b = 128 + 70 * np.sin(np.hypot(xx - w / 2, yy - h / 2) / 9.0)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def backgrounds(args, w: int, h: int, n: int):
    """Clean RGB frames: the texture with mild temporal noise, or a real clip."""
    if args.background:
        cmd = [FFMPEG, "-nostdin", "-v", "error", "-stream_loop", "-1", "-i", args.background,
               "-vf", f"scale={w}:{h}:flags=bicubic", "-frames:v", str(n),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        for _ in range(n):
            buf = proc.stdout.read(w * h * 3)
            if len(buf) != w * h * 3:
                raise SystemExit(f"--background {args.background}: too few frames")
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        proc.stdout.close()
        proc.wait()
        return
    base = texture(w, h).astype(np.int16)
    rng = np.random.default_rng(1234)
    for _ in range(n):
        yield np.clip(base + rng.integers(-2, 3, size=base.shape), 0, 255).astype(np.uint8)


def ffv1_writer(path: str, w: int, h: int, in_fmt: str, out_fmt: str) -> subprocess.Popen:
    return subprocess.Popen(
        [FFMPEG, "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", in_fmt,
         "-s", f"{w}x{h}", "-framerate", str(FPS), "-i", "pipe:0",
         "-c:v", "ffv1", "-pix_fmt", out_fmt, "-f", "matroska", "-y", path],
        stdin=subprocess.PIPE,
    )


def make_item(args, work: str, label: str, w: int, h: int, n: int, *, static_patch: bool) -> dict:
    """One served item plus the clean background it was made from."""
    obj_path = os.path.join(work, f"{label}-object.mkv")
    mask_path = os.path.join(work, f"{label}-mask.mkv")
    clean_path = os.path.join(work, f"{label}-clean.mkv")
    writers = [ffv1_writer(obj_path, w, h, "rgb24", "yuv420p"),
               ffv1_writer(mask_path, w, h, "gray", "gray"),
               ffv1_writer(clean_path, w, h, "rgb24", "yuv420p")]
    side = max(8, h // 6)
    empty = range(int(n * 0.40), int(n * 0.55))  # a span of frames with no object, empty mask
    patch = max(8, h // 10)
    px, py = w - 2 * patch, h // 8  # always-covered patch: no clean sample anywhere
    for t, clean in enumerate(backgrounds(args, w, h, n)):
        frame = clean.copy()
        mask = np.zeros((h, w), np.uint8)
        if t not in empty:
            x = int((w - side - 1) * t / max(1, n - 1))
            y = int((h - side - 1) * (0.5 + 0.4 * np.sin(t / 9.0)))
            frame[y:y + side, x:x + side] = (230, 30, 40)
            mask[y:y + side, x:x + side] = 255
        if static_patch:
            frame[py:py + patch, px:px + patch] = (20, 40, 220)
            mask[py:py + patch, px:px + patch] = 255
        writers[0].stdin.write(frame.tobytes())
        writers[1].stdin.write(mask.tobytes())
        writers[2].stdin.write(np.ascontiguousarray(clean).tobytes())
    for wr in writers:
        wr.stdin.close()
        if wr.wait() != 0:
            raise SystemExit("ffmpeg could not write the toy streams")
    item = os.path.join(work, f"{label}.mkv")
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-i", obj_path, "-i", mask_path,
                    "-map", "0:v:0", "-map", "1:v:0", "-c", "copy", "-f", "matroska", "-y", item],
                   check=True)
    with open(item, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return {"label": label, "path": item, "clean": clean_path, "digest": digest,
            "empty": set(empty) if not static_patch else set()}


def decode(path: str, w: int, h: int, index: int = 0, pix_fmt: str = "yuv420p") -> list[np.ndarray]:
    size = w * h * 3 // 2 if pix_fmt == "yuv420p" else w * h
    raw = subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-i", path, "-map", f"0:v:{index}",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"],
                         capture_output=True, check=True).stdout
    assert len(raw) % size == 0, "decoded size is not a whole number of frames"
    shape = (h * 3 // 2, w) if pix_fmt == "yuv420p" else (h, w)
    return [np.frombuffer(raw[i:i + size], np.uint8).reshape(shape) for i in range(0, len(raw), size)]


def rgb(i420: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)  # the scorer's conversion


def check_item(item: dict, output: str, w: int, h: int, n: int, *,
               outside_limit: float = 1.0, lossless: bool = True) -> list[str]:
    problems: list[str] = []
    probe = json.loads(subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,profile,width,height,pix_fmt,r_frame_rate", "-of", "json", output],
        capture_output=True, check=True).stdout)["streams"]
    if len(probe) != 1:
        problems.append(f"{len(probe)} streams in the output, want 1")
    s = probe[0]
    want = {"codec_type": "video", "codec_name": "h264", "width": w, "height": h,
            "pix_fmt": "yuv420p", "r_frame_rate": f"{FPS}/1"}
    for key, value in want.items():
        if s.get(key) != value:
            problems.append(f"{key} = {s.get(key)!r}, want {value!r}")
    out = decode(output, w, h)
    inp = decode(item["path"], w, h, 0)
    clean = decode(item["clean"], w, h)
    masks = [m > 127 for m in decode(item["path"], w, h, 1, "gray")]
    if not (len(out) == len(inp) == len(masks) == n):
        return problems + [f"frame counts: output {len(out)}, input {len(inp)}, mask {len(masks)}, want {n}"]
    worst_outside = 0.0
    fill_err = object_err = 0.0
    masked_px = 0
    for t in range(n):
        m = masks[t]
        o, i, c = rgb(out[t]), rgb(inp[t]), rgb(clean[t])
        keep = (~m).astype(np.uint8)
        if keep.any():
            worst_outside = max(worst_outside, sum(cv2.mean(cv2.absdiff(o, i), mask=keep)[:3]) / 3)
        if m.any():
            fill_err += float(np.abs(o[m].astype(np.int16) - c[m]).sum())
            object_err += float(np.abs(i[m].astype(np.int16) - c[m]).sum())
            masked_px += int(m.sum()) * 3
        if lossless and t in item["empty"] and not np.array_equal(out[t], inp[t]):
            problems.append(f"frame {t} has an empty mask but changed")
    fill_mae, object_mae = fill_err / masked_px, object_err / masked_px
    print(f"  {item['label']}: outside-mask change max {worst_outside:.3f} (limit {outside_limit}); "
          f"inside-mask MAE vs clean: fill {fill_mae:.2f}, object {object_mae:.2f}; "
          f"profile {s.get('profile')}")
    if worst_outside > outside_limit:
        problems.append(f"outside-mask change {worst_outside:.3f} > {outside_limit}")
    if not fill_mae < object_mae:
        problems.append(f"fill ({fill_mae:.2f}) is not closer to the background than the object ({object_mae:.2f})")
    return problems


def check_median() -> list[str]:
    rng = np.random.default_rng(7)
    values = rng.integers(0, 256, size=(37, 5000), dtype=np.uint8)
    invalid = rng.random((37, 5000)) < 0.6
    invalid[:, :50] = True  # no valid sample at all
    med, has = removal_example.masked_median(values, invalid)
    ref = values.astype(np.float64)
    ref[invalid] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nan_med = np.nanmedian(ref, axis=0)
    ok = ~np.isnan(nan_med)
    problems = []
    if not np.array_equal(has, ok):
        problems.append("masked_median: validity differs from np.nanmedian")
    if not np.array_equal(med[ok], np.floor(nan_med[ok] + 0.5).astype(np.uint8)):
        problems.append("masked_median differs from np.nanmedian rounded half up")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="320x240", help="WxH, both even (default 320x240)")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--background", help="a local clip to use as the clean background")
    parser.add_argument("--keep", action="store_true", help="keep the work directory")
    args = parser.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    n = args.frames
    work = tempfile.mkdtemp(prefix="removal-example-test-")
    problems = check_median()
    try:
        in_dir, out_dir = os.path.join(work, "in"), os.path.join(work, "out")
        os.mkdir(in_dir)
        os.mkdir(out_dir)
        items = [make_item(args, work, "moving-object", w, h, n, static_patch=False),
                 make_item(args, work, "always-covered-patch", w, h, n, static_patch=True)]
        for item in items:
            shutil.copyfile(item["path"], os.path.join(in_dir, item["digest"]))
        junk = os.urandom(4096)
        junk_name = hashlib.sha256(junk).hexdigest()
        with open(os.path.join(in_dir, junk_name), "wb") as fh:
            fh.write(junk)

        # 1) the module alone, timed, peak memory of the module and its ffmpeg children
        module = os.path.join(HERE, "removal_example.py")
        for item in items:
            target = os.path.join(work, item["label"] + ".mp4")
            res = subprocess.run([sys.executable, "-c", _MEASURE, sys.executable,
                                  module, item["path"], target],
                                 capture_output=True, text=True)
            rc, seconds, rss = res.stdout.split()
            print(f"  {item['label']} {w}x{h}x{n}: exit {rc}, {float(seconds):.1f} s, "
                  f"peak RSS {int(rss) / 2**20:.0f} MiB, output {os.path.getsize(target) if os.path.exists(target) else 0} bytes")
            sys.stderr.write(res.stderr)

        # 2) the size guard: fallback encode when lossless does not fit, nothing when
        #    even the fallback does not
        item = items[0]
        lossless_bytes = os.path.getsize(os.path.join(work, item["label"] + ".mp4"))
        allowance = int(lossless_bytes * 0.6)
        guarded = os.path.join(work, "guarded.mp4")
        res = subprocess.run([sys.executable, module, "--max-bytes", str(allowance), item["path"], guarded],
                             capture_output=True, text=True)
        sys.stderr.write(res.stderr)
        if res.returncode != 0 or not os.path.exists(guarded):
            problems.append(f"size guard: exit {res.returncode} with a {allowance}-byte allowance")
        else:
            if os.path.getsize(guarded) > allowance:
                problems.append("size guard: output larger than the allowance")
            problems += [f"size guard: {p}" for p in
                         check_item(item, guarded, w, h, n, outside_limit=3.0, lossless=False)]
        refused = os.path.join(work, "refused.mp4")
        res = subprocess.run([sys.executable, module, "--max-bytes", "1000", item["path"], refused],
                             capture_output=True, text=True)
        sys.stderr.write(res.stderr)
        leftovers = [f for f in os.listdir(work) if f.startswith(removal_example.PARTIAL_PREFIX)]
        if res.returncode != 1 or os.path.exists(refused) or leftovers:
            problems.append(f"size guard: impossible allowance gave exit {res.returncode}, "
                            f"output {os.path.exists(refused)}, leftovers {leftovers}")

        # 3) the full contract through run.sh
        env = dict(os.environ, CONTENDER_APP_DIR=HERE, PYTHON=sys.executable)
        res = subprocess.run(["/bin/sh", os.path.join(HERE, "run.sh"), in_dir, out_dir],
                             env=env, capture_output=True, text=True)
        sys.stderr.write(res.stderr)
        if res.returncode != 0:
            problems.append(f"run.sh exited {res.returncode}")
        names = sorted(os.listdir(out_dir))
        if names != sorted(item["digest"] for item in items):
            problems.append(f"output directory holds {names}, want exactly the two good items")
        for item in items:
            output = os.path.join(out_dir, item["digest"])
            if not os.path.exists(output):
                problems.append(f"{item['label']}: no output")
                continue
            st = os.lstat(output)
            if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode) != 0o644:
                problems.append(f"{item['label']}: output is not a 0644 regular file")
            problems += [f"{item['label']}: {p}" for p in check_item(item, output, w, h, n)]
    finally:
        if args.keep:
            print(f"work directory kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    for p in problems:
        print(f"FAIL {p}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
