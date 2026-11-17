"""Generative upscaling through the real worker pipeline (real ffmpeg canonicalization, the
native-geometry served input, the optional delivery encode), with deterministic stand-ins
for DISTS / LPIPS / CLIP-IQA so no model weights are needed."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import numpy as np

from tests.scoring_worker.conftest import FFMPEG, FFPROBE, requires_media_tools, sha256_file
from vidaio.scoring import ReasonCode, ScoringConfig
from vidaio.scoring_worker import ScoringWorkerConfig, effective_scorer_version, real_backends
from vidaio.scoring_worker.service import _score_sync
from vidaio.scoring_worker.upscaling_generative_backend import CpuUpscalingGenerativeBackend
from vidaio.services.protocol import ScoreRequest

SIZE, HALF = "192x128", "96x64"


class _Mad:
    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean() / 255.0)


class _Sharpness:
    def quality(self, rgb: np.ndarray) -> float:
        g = np.abs(np.diff(rgb.astype(np.float64).mean(axis=2), axis=1)).mean()
        return float(g / (g + 10.0))


def _ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", *args],
                   check=True, capture_output=True, timeout=120)


def _media(root: Path) -> dict[str, Path]:
    reference, served = root / "reference.mkv", root / "input.mp4"
    upscaled, shifted = root / "upscaled.mp4", root / "shifted.mp4"
    tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={SIZE}:rate=8:duration=2", "-pix_fmt", "yuv420p",
            "-c:v", "ffv1", *tags, "-y", str(reference))
    _ffmpeg("-i", str(reference), "-vf", f"scale={HALF}:flags=lanczos,format=yuv420p",
            "-c:v", "libx264", "-crf", "18", *tags, "-y", str(served))
    _ffmpeg("-i", str(served), "-vf", f"scale={SIZE}:flags=lanczos,format=yuv420p",
            "-c:v", "libx264", "-qp", "0", *tags, "-y", str(upscaled))
    _ffmpeg("-i", str(served), "-vf", f"scale={SIZE}:flags=lanczos,hue=h=75:s=1.4,format=yuv420p",
            "-c:v", "libx264", "-qp", "0", *tags, "-y", str(shifted))
    return {"reference": reference, "served": served, "upscaled": upscaled, "shifted": shifted}


@requires_media_tools
def test_generative_scoring_runs_the_real_pipeline_and_gates(tmp_path: Path) -> None:
    media = _media(tmp_path)
    worker_config = ScoringWorkerConfig(
        work_dir=tmp_path / "work", ffmpeg_path=FFMPEG, ffprobe_path=FFPROBE, pieapp_device="cpu",
        request_timeout=300.0, subprocess_timeout=120.0,
    )

    def run(output: Path, scoring_config: ScoringConfig):
        backends = real_backends(worker_config, scoring_config=scoring_config, pieapp_device="cpu")
        backends = dataclasses.replace(
            backends,
            upscaling_generative=CpuUpscalingGenerativeBackend(
                lpips_loader=_Mad, dists_loader=_Mad, nr_loader=_Sharpness
            ),
        )
        version = effective_scorer_version(worker_config, scoring_config)
        request = ScoreRequest(
            track="upscaling", challenge_id="chal-generative-real", item_id=sha256_file(media["served"]),
            miner_hotkey="hk", reference_path=str(media["reference"]),
            reference_digest=sha256_file(media["reference"]),
            miner_input_path=str(media["served"]), miner_input_digest=sha256_file(media["served"]),
            output_path=str(output), output_digest=sha256_file(output),
            params={"upscale_factor": 2, "target_width": 192, "target_height": 128, "scoring": "generative"},
            scorer_version=version,
        )
        return _score_sync(request, worker_config, scoring_config, backends, version)

    # the colour gate compares regional colour: production area-downscales 1080p frames to a
    # 256-px long side (7.5x); the same ratio on this 192-px clip is a 26-px long side
    config = ScoringConfig(upscaling_generative_color_max_side=26)
    good = run(media["upscaled"], config)
    assert good.gate_passed, good.violations
    assert good.breakdown is not None and good.breakdown.kind == "upscaling_generative"
    assert good.metrics["gen_lowres_psnr_db"] > 30.0
    assert good.metrics["gen_color_delta_e"] < 3.0
    assert good.metrics["delivery_encode"] == "none"
    assert good.metrics["upscale_frames"] == 16
    # byte-identical packet on a re-score (the auditor's recompute)
    assert run(media["upscaled"], config).to_json() == good.to_json()

    shifted = run(media["shifted"], config)
    assert not shifted.gate_passed and shifted.score == 0.0
    assert ReasonCode.COLOR_SHIFT_EXCEEDED in [v.code for v in shifted.violations]

    delivered_config = config.model_copy(update={"upscaling_generative_delivery_encode": True})
    delivered = run(media["upscaled"], delivered_config)
    assert delivered.gate_passed, delivered.violations
    assert delivered.metrics["delivery_encode"] == "x264-crf18"
    # the light delivery encode costs a little fidelity, never the frame alignment
    assert 30.0 < delivered.metrics["gen_lowres_psnr_db"] < good.metrics["gen_lowres_psnr_db"]
    assert delivered.scoring_config_digest != good.scoring_config_digest
