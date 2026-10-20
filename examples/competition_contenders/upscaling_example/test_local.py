"""Local self-test of the upscaling example (no Docker needed; ffmpeg + numpy).

    python3 test_local.py

Builds a small synthetic original with fine texture, serves it downscaled by 2 as an
H.264 MP4 with its hidden task file (exactly what the evaluator mounts), runs ``run.sh``
and checks the contract: the target size, the input's frame count and frame rate, one
H.264 stream. When the VidAIO repository is importable it also scores the output with
the real faithful upscaling metrics (the Lanczos example scores about zero).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W, H, FRAMES, FACTOR = 384, 216, 24, 2


def scene() -> np.ndarray:
    """RGB frames: fine stripes, a checker and thin lines over a slow pan."""
    yy, xx = np.mgrid[0:H, 0:W]
    frames = np.empty((FRAMES, H, W, 3), np.uint8)
    for i in range(FRAMES):
        x = xx + 2 * i
        frames[i, ..., 0] = (x * 3 % 256).astype(np.uint8)
        frames[i, ..., 1] = ((yy * 4 + i) % 256).astype(np.uint8)
        frames[i, ..., 2] = 90
        checker = ((x // 3 + yy // 3) % 2).astype(bool)
        patch = (xx > 40) & (xx < 200) & (yy > 40) & (yy < 160)
        frames[i][patch & checker] = (235, 225, 40)
        frames[i][patch & ~checker] = (30, 35, 190)
        frames[i][:, (x[0] % 17) == 0] = (250, 250, 250)
    return frames


def ffmpeg(*args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(["ffmpeg", "-v", "error", "-y", *args], input=data,
                          capture_output=True, check=True).stdout


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)["streams"][0]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs, outputs = root / "in", root / "out"
        inputs.mkdir()
        outputs.mkdir()
        original = root / "original.mkv"
        raw = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-"]
        ffmpeg(*raw, "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               "-c:v", "ffv1", "-colorspace", "bt709", "-color_range", "tv", str(original),
               data=scene().tobytes())
        staged = root / "staged.mp4"
        ffmpeg("-i", str(original), "-vf", f"scale={W // FACTOR}:{H // FACTOR}:flags=area",
               "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
               "-color_range", "tv", "-f", "mp4", str(staged))
        name = hashlib.sha256(staged.read_bytes()).hexdigest()
        os.replace(staged, inputs / name)
        (inputs / f".vidaio-next-upscale-factor-{name}").write_text(
            json.dumps({"target_height": H, "target_width": W, "upscale_factor": FACTOR},
                       sort_keys=True, separators=(",", ":")) + "\n"
        )
        env = {**os.environ, "CONTENDER_APP_DIR": str(HERE), "PYTHON": sys.executable}
        subprocess.run(["/bin/sh", str(HERE / "run.sh"), str(inputs), str(outputs)], check=True, env=env)

        out = outputs / name
        assert out.is_file(), "no output written"
        got, served = probe(out), probe(inputs / name)
        assert got["codec_name"] == "h264", got
        assert (int(got["width"]), int(got["height"])) == (W, H), got
        assert got["nb_read_frames"] == served["nb_read_frames"] == str(FRAMES), (got, served)
        assert got["r_frame_rate"] == served["r_frame_rate"], (got, served)
        assert sorted(p.name for p in outputs.iterdir()) == [name], "only the output may remain"
        print(f"contract ok: {W}x{H}, {FRAMES} frames at {got['r_frame_rate']}")

        try:  # optional: the real metrics, when run inside the VidAIO repository
            sys.path.insert(0, str(HERE.parents[2]))
            from vidaio.scoring import ScoringConfig, score_upscaling_faithful
            from vidaio.scoring.removal import Y4MReader
            from vidaio.scoring.upscaling_faithful import upscaled_frame_metrics
        except ImportError:
            print("scoring skipped (run from the VidAIO repository to score)")
            return 0
        y4m = {}
        for label, path, vf in (("reference", original, "format=yuv420p"),
                                ("candidate", out, "format=yuv420p"),
                                ("floor", inputs / name, f"scale={W}:{H}:flags=lanczos,format=yuv420p")):
            target = root / f"{label}.y4m"
            ffmpeg("-i", str(path), "-vf", vf, "-f", "yuv4mpegpipe", str(target))
            y4m[label] = Y4MReader.open(str(target))
        cand, floor = upscaled_frame_metrics(y4m["candidate"], y4m["reference"], y4m["floor"],
                                             lpips=None, lpips_stride=1)
        print(f"psnr {cand.psnr_db:.2f} dB (floor {floor.psnr_db:.2f}), "
              f"ssim {cand.ssim:.4f} (floor {floor.ssim:.4f})")
        print("LPIPS not loaded here: the Lanczos example sits on the PSNR/SSIM floor, so it scores ~0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
