"""Local self-test of the interpolation example (no Docker needed; ffmpeg + numpy).

    python3 test_local.py

Builds a synthetic 60 fps clip with real motion, serves every other frame as a 30 fps
FFV1 Matroska input (exactly what the evaluator mounts), runs ``run.sh`` on it and checks
the contract: 2N - 1 frames at twice the rate, input frames unchanged at the even
positions. When the VidAIO repository is importable it also scores the output with the
real interpolation metrics (the blend scores about zero, the true frames score one).
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
W, H, REF_FRAMES = 192, 108, 31  # 31 frames at 60 fps -> 16 input frames at 30 fps


def scene() -> np.ndarray:
    """RGB frames: a textured disc crossing a panning gradient."""
    yy, xx = np.mgrid[0:H, 0:W]
    frames = np.empty((REF_FRAMES, H, W, 3), np.uint8)
    for i in range(REF_FRAMES):
        frames[i, ..., 0] = ((xx + 2 * i) * 3 % 256).astype(np.uint8)
        frames[i, ..., 1] = ((yy * 4 + i) % 256).astype(np.uint8)
        frames[i, ..., 2] = 80
        cx, cy = 20 + 5 * i, 54 + int(20 * np.sin(i / 5))
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 < 18**2
        stripes = ((xx + yy) // 4 % 2).astype(bool)
        frames[i][disc & stripes] = (250, 230, 40)
        frames[i][disc & ~stripes] = (30, 40, 200)
    return frames


def encode(frames: np.ndarray, path: Path, fps: int) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "ffv1",
         "-pix_fmt", "yuv420p", "-f", "matroska", str(path)],
        input=frames.tobytes(), check=True,
    )


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def frame_rate(path: Path) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,codec_name", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    assert stream["codec_name"] == "h264", stream
    return stream["r_frame_rate"]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs, outputs = root / "in", root / "out"
        inputs.mkdir()
        outputs.mkdir()
        reference = scene()
        staged = root / "staged.mkv"
        encode(reference[::2], staged, 30)
        name = hashlib.sha256(staged.read_bytes()).hexdigest()
        os.replace(staged, inputs / name)
        env = {**os.environ, "CONTENDER_APP_DIR": str(HERE), "PYTHON": sys.executable}
        subprocess.run(["/bin/sh", str(HERE / "run.sh"), str(inputs), str(outputs)], check=True, env=env)

        out = outputs / name
        assert out.is_file(), "no output written"
        assert frame_rate(out) == "60/1", frame_rate(out)
        got = decode(out)
        served = decode(inputs / name)
        assert len(got) == 2 * len(served) - 1 == REF_FRAMES, (len(got), len(served))
        kept = max(float(np.abs(got[2 * k].astype(int) - served[k]).mean()) for k in range(len(served)))
        assert kept < 3.0, f"input frames changed by {kept:.2f} (limit 3.0)"
        print(f"contract ok: {len(got)} frames at 60/1, kept-frame change {kept:.3f}")

        try:  # optional: the real metrics, when run inside the VidAIO repository
            sys.path.insert(0, str(HERE.parents[2]))
            from vidaio.scoring import ScoringConfig, score_interpolation
            from vidaio.scoring.interpolation import inserted_frame_metrics
            from vidaio.scoring.interpolation_formula import best_baseline
            from vidaio.scoring.removal import Y4MReader
        except ImportError:
            print("scoring skipped (run from the VidAIO repository to score)")
            return 0
        y4m = {}
        for label, path, fps in (("reference", None, 60), ("input", inputs / name, 30), ("candidate", out, 60)):
            target = root / f"{label}.y4m"
            if path is None:
                path = root / "reference.mkv"
                encode(reference, path, fps)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-pix_fmt", "yuv420p",
                            "-f", "yuv4mpegpipe", str(target)], check=True)
            y4m[label] = Y4MReader.open(str(target))
        cand, hold, blend = inserted_frame_metrics(
            y4m["candidate"], y4m["reference"], y4m["input"], lpips=None, lpips_stride=1
        )
        floor = best_baseline(hold, blend)
        print(f"inserted frames: psnr {cand.psnr_db:.2f} dB (floor {floor.psnr_db:.2f}), "
              f"ssim {cand.ssim:.4f} (floor {floor.ssim:.4f})")
        if all(np.isfinite([cand.lpips_vgg])):
            print(f"score {score_interpolation(metrics=cand, floor=floor, config=ScoringConfig()).final:.3f}")
        else:
            print("LPIPS not loaded here: the blend sits on the PSNR/SSIM floor, so it scores ~0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
