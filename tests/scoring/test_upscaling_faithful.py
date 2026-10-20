"""Faithful upscaling metrics and composition on tiny synthetic y4m clips (no ffmpeg needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vidaio.scoring import (
    ScoringConfig,
    UpscaledFrameMetrics,
    score_upscaling_faithful,
)
from vidaio.scoring.removal import Y4MError, Y4MReader
from vidaio.scoring.upscaling_faithful import upscaled_frame_metrics
from vidaio.scoring_worker.upscaling_faithful_backend import (
    CpuUpscalingFaithfulBackend,
    UpscalingFaithfulInputError,
    UpscalingFaithfulMeasurementError,
)

W, H, N = 64, 48, 6


def _write_y4m(path: Path, frames_rgb: np.ndarray, fps: int = 30) -> None:
    """(n, h, w, 3) uint8 RGB -> 8-bit 4:2:0 y4m."""
    import cv2

    n, h, w, _ = frames_rgb.shape
    with open(path, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{w} H{h} F{fps}:1 Ip A1:1 C420jpeg\n".encode())
        for f in frames_rgb:
            fh.write(b"FRAME\n")
            fh.write(cv2.cvtColor(f, cv2.COLOR_RGB2YUV_I420).tobytes())


def _scene() -> np.ndarray:
    """Fine texture (a 2-pixel checker and thin lines) over a pan: detail a downscale destroys."""
    yy, xx = np.mgrid[0:H, 0:W]
    frames = np.empty((N, H, W, 3), np.uint8)
    for i in range(N):
        pan = 2 * i
        frames[i, ..., 0] = ((xx + pan) * 4 % 256).astype(np.uint8)
        frames[i, ..., 1] = ((yy * 5 + pan) % 256).astype(np.uint8)
        frames[i, ..., 2] = 90
        checker = ((((xx + pan) // 2) + (yy // 2)) % 2).astype(bool)
        patch = (xx >= 8) & (xx < 40) & (yy >= 10) & (yy < 34)
        frames[i][patch & checker] = (235, 235, 40)
        frames[i][patch & ~checker] = (25, 25, 200)
        frames[i][:, (xx[0] + pan) % 11 == 0] = (250, 250, 250)
    return frames


def _degrade(ref: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """The served input (area downscale) and the validator's free answer (Lanczos back up)."""
    import cv2

    served = np.stack([cv2.resize(f, (W // factor, H // factor), interpolation=cv2.INTER_AREA) for f in ref])
    lanczos = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LANCZOS4) for f in served])
    return served, lanczos


class _MadLpips:
    """A deterministic stand-in for LPIPS-VGG: mean abs difference on the 0-1 scale."""

    def __init__(self) -> None:
        self.calls = 0

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        self.calls += 1
        return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean() / 255.0)


@pytest.fixture
def clips(tmp_path: Path) -> dict[str, Path]:
    import cv2

    ref = _scene()
    served, lanczos = _degrade(ref, 2)
    halfway = ((ref.astype(np.uint16) + lanczos + 1) >> 1).astype(np.uint8)
    blurred = np.stack([cv2.GaussianBlur(f, (5, 5), 1.5) for f in lanczos])  # further from the original
    paths = {}
    for name, frames in (
        ("reference", ref),
        ("served", served),
        ("upscaled_input", lanczos),
        ("perfect", ref),
        ("lanczos", lanczos),
        ("halfway", halfway),
        ("blurred", blurred),
        ("short", ref[:-1]),
        ("small", served),
    ):
        paths[name] = tmp_path / f"{name}.y4m"
        _write_y4m(paths[name], frames)
    return paths


def _measure(paths: dict[str, Path], name: str, *, lpips=None, stride: int = 1, offset: int = 0):
    return upscaled_frame_metrics(
        Y4MReader.open(str(paths[name])),
        Y4MReader.open(str(paths["reference"])),
        Y4MReader.open(str(paths["upscaled_input"])),
        lpips=lpips if lpips is not None else _MadLpips(),
        lpips_stride=stride,
        lpips_offset=offset,
    )


def _score(paths: dict[str, Path], name: str, *, config: ScoringConfig | None = None):
    cand, floor = _measure(paths, name)
    return score_upscaling_faithful(metrics=cand, floor=floor, upscale_factor=2, config=config or ScoringConfig())


def test_the_free_answer_scores_zero_and_the_original_scores_one(clips):
    assert _score(clips, "lanczos").final == 0.0
    assert _score(clips, "blurred").final == 0.0  # no closer to the original than the floor
    perfect = _score(clips, "perfect")
    assert perfect.psnr_db == 99.0 and perfect.ssim == pytest.approx(1.0) and perfect.lpips_vgg == 0.0
    assert perfect.final == pytest.approx(1.0)
    halfway = _score(clips, "halfway")
    assert 0.0 < halfway.final < perfect.final
    assert halfway.s_psnr > 0 and halfway.s_ssim > 0 and halfway.s_lpips > 0
    assert halfway.kind == "upscaling_faithful" and halfway.upscale_factor == 2 and halfway.frames == N


def test_formula_credits_only_the_gain_over_the_floor():
    config = ScoringConfig()
    floor = UpscaledFrameMetrics(psnr_db=30.0, ssim=0.90, lpips_vgg=0.20, frames=10)
    good = UpscaledFrameMetrics(psnr_db=33.0, ssim=0.95, lpips_vgg=0.05, frames=10)
    b = score_upscaling_faithful(metrics=good, floor=floor, upscale_factor=4, config=config)
    assert b.s_psnr == pytest.approx(3.0 / 6.0)
    assert b.s_ssim == pytest.approx(0.05 / 0.10)
    assert b.s_lpips == pytest.approx(0.15 / 0.20)
    assert (b.weight_psnr, b.weight_ssim, b.weight_lpips) == (0.3, 0.3, 0.4)
    assert b.final == pytest.approx(0.3 * b.s_psnr + 0.3 * b.s_ssim + 0.4 * b.s_lpips)
    assert (b.floor_psnr_db, b.floor_ssim, b.floor_lpips_vgg) == (30.0, 0.90, 0.20)
    worse = UpscaledFrameMetrics(psnr_db=29.0, ssim=0.85, lpips_vgg=0.30, frames=10)
    assert score_upscaling_faithful(metrics=worse, floor=floor, upscale_factor=4, config=config).final == 0.0
    # each term saturates at its span / a perfect frame
    huge = UpscaledFrameMetrics(psnr_db=60.0, ssim=1.0, lpips_vgg=0.0, frames=10)
    assert score_upscaling_faithful(metrics=huge, floor=floor, upscale_factor=2, config=config).final == 1.0
    # a perfect floor leaves nothing to gain on that term (no division by zero)
    perfect_floor = UpscaledFrameMetrics(psnr_db=99.0, ssim=1.0, lpips_vgg=0.0, frames=10)
    assert score_upscaling_faithful(metrics=huge, floor=perfect_floor, upscale_factor=2, config=config).final == 0.0


def test_formula_refuses_non_finite_metrics_and_unknown_factors():
    config = ScoringConfig()
    floor = UpscaledFrameMetrics(psnr_db=30.0, ssim=0.90, lpips_vgg=0.20, frames=10)
    with pytest.raises(ValueError):
        score_upscaling_faithful(
            metrics=UpscaledFrameMetrics(float("nan"), 1.0, 0.1, 1), floor=floor, upscale_factor=2, config=config
        )
    with pytest.raises(ValueError):
        score_upscaling_faithful(
            metrics=floor, floor=UpscaledFrameMetrics(30.0, 0.9, float("inf"), 1), upscale_factor=2, config=config
        )
    for factor in (1, 3, 8):
        with pytest.raises(ValueError):
            score_upscaling_faithful(metrics=floor, floor=floor, upscale_factor=factor, config=config)


def test_metrics_refuse_a_wrong_geometry_or_frame_count(clips):
    ref = Y4MReader.open(str(clips["reference"]))
    up = Y4MReader.open(str(clips["upscaled_input"]))
    with pytest.raises(Y4MError):
        upscaled_frame_metrics(Y4MReader.open(str(clips["short"])), ref, up, lpips=None, lpips_stride=1)
    with pytest.raises(Y4MError):
        upscaled_frame_metrics(Y4MReader.open(str(clips["small"])), ref, up, lpips=None, lpips_stride=1)
    with pytest.raises(ValueError):
        upscaled_frame_metrics(ref, ref, Y4MReader.open(str(clips["short"])), lpips=None, lpips_stride=1)


def test_lpips_samples_every_kth_frame_from_an_offset(clips):
    lpips = _MadLpips()
    _measure(clips, "halfway", lpips=lpips, stride=2, offset=1)
    # 6 frames, stride 2 -> 3 sampled frames, each for the candidate and the floor
    assert lpips.calls == 6
    lpips = _MadLpips()
    _measure(clips, "halfway", lpips=lpips, stride=50, offset=7)
    assert lpips.calls == 2  # fewer frames than the stride: the last one is sampled
    # the offset picks other frames: the same answer, other sampled values
    a, _ = _measure(clips, "halfway", stride=3, offset=0)
    b, _ = _measure(clips, "halfway", stride=3, offset=1)
    assert (a.psnr_db, a.ssim) == (b.psnr_db, b.ssim)


def test_backend_separates_validator_and_candidate_failures(clips):
    lpips = _MadLpips()
    backend = CpuUpscalingFaithfulBackend(lpips_loader=lambda: lpips)
    m = backend.measure(
        canonical_reference=str(clips["reference"]),
        canonical_candidate=str(clips["halfway"]),
        canonical_upscaled_input=str(clips["upscaled_input"]),
        config=ScoringConfig(),
    )
    assert m.metrics.frames == N and m.floor.frames == N
    assert m.metrics.psnr_db > m.floor.psnr_db
    for bad in ("short", "small"):
        with pytest.raises(UpscalingFaithfulMeasurementError):
            backend.measure(
                canonical_reference=str(clips["reference"]),
                canonical_candidate=str(clips[bad]),
                canonical_upscaled_input=str(clips["upscaled_input"]),
                config=ScoringConfig(),
            )
    with pytest.raises(UpscalingFaithfulMeasurementError):
        backend.measure(
            canonical_reference=str(clips["reference"]),
            canonical_candidate=str(clips["reference"]) + ".missing",
            canonical_upscaled_input=str(clips["upscaled_input"]),
            config=ScoringConfig(),
        )
    # the validator's own files not matching is never the miner's fault
    for reference, upscaled in (("short", "upscaled_input"), ("reference", "small"), ("reference", "short")):
        with pytest.raises(UpscalingFaithfulInputError):
            backend.measure(
                canonical_reference=str(clips[reference]),
                canonical_candidate=str(clips["perfect"]),
                canonical_upscaled_input=str(clips[upscaled]),
                config=ScoringConfig(),
            )


def test_backend_measures_an_items_floor_once_for_every_candidate(clips):
    """The Lanczos floor depends on no candidate: with a floor key it is measured on the
    first candidate of an item and reused, and the measurement is identical to an uncached one."""
    lpips = _MadLpips()
    backend = CpuUpscalingFaithfulBackend(lpips_loader=lambda: lpips)
    config = ScoringConfig(upscaling_faithful_lpips_stride=N)
    item = dict(
        canonical_reference=str(clips["reference"]),
        canonical_upscaled_input=str(clips["upscaled_input"]),
        config=config,
    )
    first = backend.measure(canonical_candidate=str(clips["halfway"]), floor_key=("ref", "in"), **item)
    assert lpips.calls == 2  # candidate and floor on the one sampled frame
    second = backend.measure(canonical_candidate=str(clips["perfect"]), floor_key=("ref", "in"), **item)
    assert lpips.calls == 3  # only the candidate
    uncached = CpuUpscalingFaithfulBackend(lpips_loader=lambda: _MadLpips()).measure(
        canonical_candidate=str(clips["perfect"]), **item
    )
    assert second == uncached
    assert first.floor == second.floor
    # another LPIPS sample (offset) or another item is another floor
    backend.measure(canonical_candidate=str(clips["perfect"]), floor_key=("ref", "in"), lpips_offset=1, **item)
    backend.measure(canonical_candidate=str(clips["perfect"]), floor_key=("ref", "other"), **item)
    assert lpips.calls == 3 + 2 + 2


def test_upscaling_faithful_config_is_validated():
    with pytest.raises(ValueError):
        ScoringConfig(upscaling_faithful_psnr_span_db=0.0)
    with pytest.raises(ValueError):
        ScoringConfig(upscaling_faithful_lpips_stride=0)
    with pytest.raises(ValueError):
        ScoringConfig(upscaling_faithful_lpips_max_side=-1)
    with pytest.raises(ValueError):
        ScoringConfig(upscaling_faithful_weights={"psnr": 0.5, "ssim": 0.5, "lpips": 0.5})
