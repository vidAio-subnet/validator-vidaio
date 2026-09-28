"""Returning the served clip unchanged never scores on the removal track.

On a panning shot the free fill (per-pixel temporal median) is poor, and an object whose
colours sit close to the background can be nearer to the reference than that fill. The
floor is therefore the better of the free fill and the unchanged input."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.scoring_worker.conftest import FFMPEG, FFPROBE, requires_media_tools, sha256_file
from vidaio.scoring import ScoringConfig
from vidaio.scoring.removal_formula import RegionMetrics, best_baseline, score_removal
from vidaio.scoring_worker import ScoringWorkerConfig, effective_scorer_version, real_backends
from vidaio.scoring_worker.service import _score_sync
from vidaio.services.protocol import ScoreRequest

SIZE = "192x128"
BOX = (48, 32, 72, 48)  # w, h, x, y


def _ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", *args], check=True, capture_output=True, timeout=120)


def test_best_baseline_takes_the_better_value_of_every_scored_term() -> None:
    fill = RegionMetrics(psnr_db=14.0, ssim=0.40, lpips_vgg=0.30, warp_error=1.0, masked_pixels=300)
    unchanged = RegionMetrics(psnr_db=22.0, ssim=0.35, lpips_vgg=0.12, warp_error=9.0, masked_pixels=300)
    best = best_baseline(fill, unchanged)
    assert (best.psnr_db, best.ssim, best.lpips_vgg) == (22.0, 0.40, 0.12)
    assert (best.warp_error, best.masked_pixels) == (1.0, 300)
    # the unchanged input measured against that floor can never clear the 1.5 dB margin
    b = score_removal(metrics=unchanged, floor=best, warp_error_reference=1.0, config=ScoringConfig())
    assert b.final == 0.0 and b.zero_reason == "REGION_BASELINE_NOT_BEATEN"


def _media(root: Path) -> tuple[Path, Path, Path, Path]:
    """A horizontally scrolling reference (the median fill smears it), a served input whose
    'object' is the same patch slightly brightened, the unchanged input as an output, and a
    genuine repair."""
    w, h, x, y = BOX
    reference, served = root / "reference.mkv", root / "input.mkv"
    unchanged, repair = root / "unchanged.mp4", root / "repair.mp4"
    src = f"testsrc2=size={SIZE}:rate=8:duration=2,scroll=horizontal=0.03"
    _ffmpeg("-f", "lavfi", "-i", src, "-pix_fmt", "yuv420p", "-c:v", "ffv1", "-y", str(reference))
    _ffmpeg(
        "-i", str(reference),
        "-f", "lavfi", "-i", f"color=black:size={SIZE}:rate=8:duration=2",
        "-f", "lavfi", "-i", f"color=white:size={w}x{h}:rate=8:duration=2",
        "-filter_complex",
        f"[0:v]split[a][b];[b]crop={w}:{h}:{x}:{y},eq=brightness=0.035[p];[a][p]overlay={x}:{y},format=yuv420p[v];"
        f"[1:v][2:v]overlay={x}:{y},format=gray[m]",
        "-map", "[v]", "-map", "[m]", "-c:v", "ffv1", "-y", str(served),
    )
    _ffmpeg("-i", str(served), "-map", "0:v:0", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-y", str(unchanged))
    _ffmpeg("-i", str(reference), "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-y", str(repair))
    return reference, served, unchanged, repair


@requires_media_tools
def test_unchanged_input_scores_zero_even_when_the_free_fill_is_worse(tmp_path: Path) -> None:
    reference, served, unchanged, repair = _media(tmp_path)
    scoring_config = ScoringConfig()
    worker_config = ScoringWorkerConfig(
        work_dir=tmp_path / "work", ffmpeg_path=FFMPEG, ffprobe_path=FFPROBE, pieapp_device="cpu",
        request_timeout=300.0, subprocess_timeout=120.0,
    )
    backends = real_backends(worker_config, scoring_config=scoring_config, pieapp_device="cpu")
    if backends.removal is None:
        import pytest

        pytest.skip("removal metrics backend is not installed")
    version = effective_scorer_version(worker_config, scoring_config)

    def score(output: Path):
        request = ScoreRequest(
            track="removal", challenge_id="chal-unchanged", item_id=sha256_file(served), miner_hotkey="hk",
            reference_path=str(reference), reference_digest=sha256_file(reference),
            miner_input_path=str(served), miner_input_digest=sha256_file(served),
            output_path=str(output), output_digest=sha256_file(output),
            params={"mask_stream_index": 1}, scorer_version=version,
        )
        return _score_sync(request, worker_config, scoring_config, backends, version)

    same = score(unchanged)
    # the brightened patch is far closer to the reference than the smeared median fill,
    # which is exactly the case the unchanged-input baseline exists for
    assert same.metrics["region_floor_psnr_db"] >= same.metrics["region_psnr_db"] - 1e-6
    assert same.score == 0.0
    assert same.metrics["removal_zero_reason"] == "REGION_BASELINE_NOT_BEATEN"
    fixed = score(repair)
    assert fixed.gate_passed, fixed.violations
    assert fixed.score > 0.5
