"""Local self-test of the streaming example (no Docker needed; ffmpeg + ffprobe).

    python3 test_local.py

Builds a 4-second lossless clip with motion, serves it as a sha256-named FFV1 file (exactly
what the evaluator mounts), runs ``run.sh`` with a 0.1 bits-per-pixel budget and checks the
contract: same size, frame count and frame rate, and, when the VidAIO repository is
importable, every streaming delivery rule measured with the validator's own packet code.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H, FPS, SECONDS, BPP = 640, 360, 30, 4, 0.1


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
        staged = root / "staged.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc2=size={W}x{H}:rate={FPS}:duration={SECONDS}", "-c:v", "ffv1",
                        "-pix_fmt", "yuv420p", "-f", "matroska", str(staged)], check=True)
        name = hashlib.sha256(staged.read_bytes()).hexdigest()
        os.replace(staged, inputs / name)
        env = {**os.environ, "CONTENDER_APP_DIR": str(HERE), "PYTHON": sys.executable, "BITS_PER_PIXEL": str(BPP)}
        subprocess.run(["/bin/sh", str(HERE / "run.sh"), str(inputs), str(outputs)], check=True, env=env)

        out = outputs / name
        assert out.is_file(), "no output written"
        got, served = probe(out), probe(inputs / name)
        assert got["codec_name"] == "h264", got
        assert (got["width"], got["height"]) == (W, H), got
        assert got["nb_read_frames"] == served["nb_read_frames"] and got["r_frame_rate"] == served["r_frame_rate"]
        print(f"contract ok: {W}x{H}, {got['nb_read_frames']} frames at {got['r_frame_rate']}")

        try:  # optional: the validator's own packet measurement, when run inside the VidAIO repository
            sys.path.insert(0, str(HERE.parents[2]))
            from vidaio.scoring import ScoringConfig, StreamingRules, streaming_target_kbps, streaming_violations
            from vidaio.scoring_worker.streaming_backend import FfprobeStreamingBackend
        except ImportError:
            print("rules not checked (run from the VidAIO repository to measure them)")
            return 0
        rules = StreamingRules(bits_per_pixel=BPP)
        target = streaming_target_kbps(width=W, height=H, fps=FPS, bits_per_pixel=BPP)
        m = FfprobeStreamingBackend().measure(str(out), fps=FPS, frames=FPS * SECONDS,
                                              peak_window_seconds=rules.peak_window_seconds)
        violations = streaming_violations(m, target_kbps=target, rules=rules, fps=FPS, config=ScoringConfig())
        print(f"budget {target:.0f} kbps: average {m.avg_kbps:.0f}, peak {m.peak_kbps:.0f}, "
              f"keyframes {m.keyframes} (max interval {m.max_keyframe_interval_seconds:.2f} s)")
        assert not violations, violations
        print("streaming rules ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
