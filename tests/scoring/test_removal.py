"""Object-removal metrics and composition on tiny synthetic y4m clips (no ffmpeg needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vidaio.scoring import ScoringConfig, score_removal
from vidaio.scoring.removal import (
    RegionMetrics,
    Y4MReader,
    outside_region_change,
    region_metrics,
    temporal_median_fill,
)

W, H, N = 64, 48, 10


def _write_y4m(path: Path, frames_rgb: np.ndarray) -> None:
    """(n, h, w, 3) uint8 RGB -> 8-bit 4:2:0 y4m (BT.601 limited like ffmpeg's rgb->yuv420p)."""
    import cv2

    n, h, w, _ = frames_rgb.shape
    with open(path, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode())
        for f in frames_rgb:
            i420 = cv2.cvtColor(f, cv2.COLOR_RGB2YUV_I420)
            fh.write(b"FRAME\n")
            fh.write(i420.tobytes())


def _scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A moving-gradient background (reference), an occluded copy (served input) and
    the per-frame mask of the occluder."""
    yy, xx = np.mgrid[0:H, 0:W]
    ref = np.empty((N, H, W, 3), np.uint8)
    base = ((xx * 3 + yy * 2) % 256).astype(np.uint8)  # static camera, static background
    for i in range(N):
        ref[i, ..., 0] = base
        ref[i, ..., 1] = 255 - base
        ref[i, ..., 2] = ((xx + yy) % 256).astype(np.uint8)
        # a little frame-to-frame sensor noise so nothing is trivially identical
        ref[i] = np.clip(ref[i].astype(np.int16) + rng.integers(-2, 3, size=ref[i].shape), 0, 255).astype(np.uint8)
    masks = np.zeros((N, H, W), bool)
    served = ref.copy()
    for i in range(N):
        cx, cy = 20 + 2 * i, 24
        m = (xx - cx) ** 2 + (yy - cy) ** 2 < 8**2
        masks[i] = m
        served[i][m] = (200, 40, 40)
    return ref, served, masks


@pytest.fixture
def clips(tmp_path: Path) -> dict[str, object]:
    rng = np.random.default_rng(1)
    ref, served, masks = _scene(rng)
    _write_y4m(tmp_path / "ref.y4m", ref)
    _write_y4m(tmp_path / "served.y4m", served)
    return {"dir": tmp_path, "ref": ref, "served": served, "masks": masks}


def test_y4m_reader_round_trips_geometry_and_frames(clips) -> None:
    r = Y4MReader.open(str(clips["dir"] / "ref.y4m"))
    assert (r.width, r.height, r.frame_count) == (W, H, N)
    f0 = r.frame_rgb(0)
    assert f0.shape == (H, W, 3)
    # chroma subsampling loses a little; luma survives closely
    assert np.abs(f0.astype(int) - clips["ref"][0].astype(int)).mean() < 6
    assert r.frame_luma(3).shape == (H, W)


def test_perfect_fill_beats_served_input_and_outside_identity_holds(clips) -> None:
    d = clips["dir"]
    ref = Y4MReader.open(str(d / "ref.y4m"))
    served = Y4MReader.open(str(d / "served.y4m"))
    masks = clips["masks"]
    perfect, warp_ref = region_metrics(ref, ref, masks, lpips=None, lpips_stride=1)
    untouched, _ = region_metrics(served, ref, masks, lpips=None, lpips_stride=1)
    assert perfect.psnr_db == 99.0 and perfect.ssim == pytest.approx(1.0)
    assert untouched.psnr_db < 20.0 and untouched.ssim < perfect.ssim
    assert perfect.masked_pixels == untouched.masked_pixels == int(masks.sum() * 3)
    assert warp_ref >= 0.0
    # the served input changes nothing outside the mask relative to itself; the
    # reference (with the object removed) differs from the input only INSIDE it
    assert outside_region_change(served, served, masks) == 0.0
    assert outside_region_change(ref, served, masks) < 6.0  # chroma rounding only


def test_temporal_median_fill_reconstructs_where_background_is_visible(clips) -> None:
    d = clips["dir"]
    served = Y4MReader.open(str(d / "served.y4m"))
    ref = Y4MReader.open(str(d / "ref.y4m"))
    masks = clips["masks"]
    out = d / "median.y4m"
    temporal_median_fill(served, masks, str(out), band_rows=16)
    med = Y4MReader.open(str(out))
    assert (med.width, med.height, med.frame_count) == (W, H, N)
    filled, _ = region_metrics(med, ref, masks, lpips=None, lpips_stride=1)
    untouched, _ = region_metrics(served, ref, masks, lpips=None, lpips_stride=1)
    # the object moves 2 px per frame, so most background pixels are visible in
    # other frames: the free fill is far better than leaving the object in place
    assert filled.psnr_db > untouched.psnr_db + 6.0
    # and nothing outside the mask changed beyond 4:2:0 chroma rounding at the mask
    # edge (the validator tolerates 3.0)
    assert outside_region_change(med, served, masks) < 0.5


def test_score_removal_composition_and_zero_reasons() -> None:
    cfg = ScoringConfig()
    floor = RegionMetrics(psnr_db=29.1, ssim=0.78, lpips_vgg=0.25, warp_error=0.46, masked_pixels=1000)
    good = RegionMetrics(psnr_db=33.3, ssim=0.87, lpips_vgg=0.07, warp_error=0.83, masked_pixels=1000)
    b = score_removal(metrics=good, floor=floor, warp_error_reference=0.59, config=cfg)
    assert b.kind == "removal" and b.zero_reason is None
    assert b.s_psnr == pytest.approx((33.3 - 29.1) / 8.0)
    assert b.s_lpips == pytest.approx((0.25 - 0.07) / 0.25)
    assert b.final == pytest.approx(0.6 * b.s_psnr + 0.4 * b.s_lpips)
    # not beating the free fill by the margin -> zero
    close = RegionMetrics(psnr_db=30.0, ssim=0.8, lpips_vgg=0.2, warp_error=0.5, masked_pixels=1000)
    z = score_removal(metrics=close, floor=floor, warp_error_reference=0.59, config=cfg)
    assert z.final == 0.0 and z.zero_reason == "REGION_BASELINE_NOT_BEATEN"
    # better PSNR but worse LPIPS -> keeps the PSNR share, the LPIPS share is 0 (not a zero)
    smear = RegionMetrics(psnr_db=32.0, ssim=0.8, lpips_vgg=0.3, warp_error=0.5, masked_pixels=1000)
    z = score_removal(metrics=smear, floor=floor, warp_error_reference=0.59, config=cfg)
    assert z.zero_reason is None and z.s_lpips == 0.0
    assert z.final == pytest.approx(0.6 * min(1.0, (32.0 - 29.1) / 8.0))
    # flicker cap: 5x the reference warp error
    flicker = RegionMetrics(psnr_db=33.0, ssim=0.87, lpips_vgg=0.07, warp_error=3.5, masked_pixels=1000)
    z = score_removal(metrics=flicker, floor=floor, warp_error_reference=0.59, config=cfg)
    assert z.zero_reason == "WARP_ERROR_EXCEEDED"
    # a flat fill is perfectly consistent but must not be rewarded for it: PSNR/LPIPS decide
    flat = RegionMetrics(psnr_db=28.0, ssim=0.7, lpips_vgg=0.27, warp_error=0.05, masked_pixels=1000)
    assert score_removal(metrics=flat, floor=floor, warp_error_reference=0.59, config=cfg).final == 0.0
    with pytest.raises(ValueError):
        score_removal(
            metrics=RegionMetrics(psnr_db=float("nan"), ssim=1, lpips_vgg=0.1, warp_error=0, masked_pixels=1),
            floor=floor, warp_error_reference=0.5, config=cfg,
        )


def test_lpips_crop_is_bounded_and_deterministic():
    """removal_lpips_max_side: a large mask bbox is downscaled identically on both sides
    before LPIPS (cost bound at 1080p+); 0 leaves the crop untouched."""
    from vidaio.scoring.removal import _bound_pair

    rng = np.random.default_rng(7)
    a = rng.integers(0, 256, (600, 900, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (600, 900, 3), dtype=np.uint8)
    sa, sb = _bound_pair(a, b, 256)
    assert sa.shape == sb.shape == (171, 256, 3)
    sa2, _ = _bound_pair(a, b, 256)
    assert np.array_equal(sa, sa2)
    ua, ub = _bound_pair(a, b, 0)
    assert ua is a and ub is b
    small = a[:100, :120]
    ka, _ = _bound_pair(small, small, 256)
    assert ka is small


def test_region_metrics_on_a_wide_frame_crops_without_copy_errors(tmp_path: Path) -> None:
    """The flow crop is a strict column subset of a memory-mapped plane (non-contiguous);
    DIS must get contiguous inputs (regression: OpenCV assertion I0.isContinuous)."""
    from vidaio.scoring.removal import Y4MReader, region_metrics

    h, w, n = 32, 160, 4
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:h, 0:w]
    ref = np.empty((n, h, w, 3), np.uint8)
    for i in range(n):
        ref[i, ..., 0] = ((xx * 5 + i * 3) % 256).astype(np.uint8)
        ref[i, ..., 1] = ((yy * 7) % 256).astype(np.uint8)
        ref[i, ..., 2] = rng.integers(0, 256, (h, w), dtype=np.uint8)
    masks = np.zeros((n, h, w), bool)
    for i in range(n):
        masks[i, 10:20, 70 + i : 82 + i] = True  # far from both frame edges
    _write_y4m(tmp_path / "ref.y4m", ref)
    reader = Y4MReader.open(str(tmp_path / "ref.y4m"))
    metrics, warp_ref = region_metrics(reader, reader, masks, lpips=None, lpips_stride=1)
    assert metrics.psnr_db == 99.0 and np.isfinite(metrics.warp_error) and np.isfinite(warp_ref)


def test_candidate_geometry_must_match_reference(tmp_path: Path) -> None:
    """A candidate of another geometry is a Y4MError (candidate-side, a violation upstream),
    never an IndexError/cv2.error that would escape as a non-punitive 500."""
    from vidaio.scoring.removal import Y4MError

    rng = np.random.default_rng(5)
    ref = rng.integers(0, 256, (3, 32, 48, 3), dtype=np.uint8)
    small = rng.integers(0, 256, (3, 16, 24, 3), dtype=np.uint8)
    big = rng.integers(0, 256, (3, 64, 96, 3), dtype=np.uint8)
    _write_y4m(tmp_path / "ref.y4m", ref)
    _write_y4m(tmp_path / "small.y4m", small)
    _write_y4m(tmp_path / "big.y4m", big)
    masks = np.zeros((3, 32, 48), bool)
    masks[:, 8:20, 10:30] = True
    r = Y4MReader.open(str(tmp_path / "ref.y4m"))
    for name in ("small", "big"):
        c = Y4MReader.open(str(tmp_path / f"{name}.y4m"))
        with pytest.raises(Y4MError):
            region_metrics(c, r, masks, lpips=None, lpips_stride=1)
        with pytest.raises(Y4MError):
            outside_region_change(c, r, masks)


def test_always_masked_pixels_get_a_spatial_fill_not_black(tmp_path: Path) -> None:
    """A static mask (the object never moves, e.g. a short clip) has no unmasked frame to
    take a median from; those pixels must be inpainted from their neighbours, not 0/128."""
    yy, xx = np.mgrid[0:H, 0:W]
    n = 6
    ref = np.empty((n, H, W, 3), np.uint8)
    for i in range(n):
        ref[i, ..., 0] = 200
        ref[i, ..., 1] = 180
        ref[i, ..., 2] = 160
    masks = np.zeros((n, H, W), bool)
    masks[:, 16:32, 20:44] = True  # identical on every frame
    served = ref.copy()
    served[masks] = 0
    _write_y4m(tmp_path / "served.y4m", served)
    s = Y4MReader.open(str(tmp_path / "served.y4m"))
    out = tmp_path / "median.y4m"
    temporal_median_fill(s, masks, str(out), band_rows=16)
    med = Y4MReader.open(str(out))
    core = med.frame_rgb(2)[20:28, 26:38]  # deep inside the always-masked block
    assert core.mean() > 120, "always-masked pixels were left black"
    # and it is deterministic: the same call gives byte-identical output
    out2 = tmp_path / "median2.y4m"
    temporal_median_fill(s, masks, str(out2), band_rows=16)
    assert out.read_bytes() == out2.read_bytes()


class _CountingLpips:
    """Records which crops LPIPS saw so the sampling schedule can be asserted."""

    def __init__(self) -> None:
        self.calls = 0

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        self.calls += 1
        return 0.1


def test_lpips_samples_every_kth_masked_frame_from_an_offset(clips) -> None:
    """The stride walks MASKED frames (not frame indices) from a phase the caller derives
    from the held-out reference digest; the schedule is exact and never empty."""
    d = clips["dir"]
    ref = Y4MReader.open(str(d / "ref.y4m"))
    masks = clips["masks"].copy()
    masks[0] = False  # frame 0 has no mask: index-based sampling would count it
    masked = int(masks.any(axis=(1, 2)).sum())
    for offset in range(4):
        lp = _CountingLpips()
        region_metrics(ref, ref, masks, lpips=lp, lpips_stride=4, lpips_offset=offset)
        expected = sum(1 for k in range(masked) if (k + offset) % 4 == 0)
        assert lp.calls == expected and lp.calls >= 1
    # fewer masked frames than the stride and an offset that skips them all: still one sample
    two = masks.copy()
    two[3:] = False
    lp = _CountingLpips()
    metrics, _ = region_metrics(ref, ref, two, lpips=lp, lpips_stride=4, lpips_offset=3)
    assert lp.calls == 1 and np.isfinite(metrics.lpips_vgg)


def test_ffprobe_is_resolved_next_to_ffmpeg() -> None:
    from vidaio.scoring.removal import ffprobe_for

    assert ffprobe_for("/opt/ffmpeg/bin/ffmpeg") == "/opt/ffmpeg/bin/ffprobe"
    assert ffprobe_for("ffmpeg") == "ffprobe"
    assert ffprobe_for("/usr/local/bin/ffmpeg7") == "/usr/local/bin/ffprobe7"
    assert ffprobe_for("/opt/tools/avconv") == "ffprobe"


def test_removal_weights_must_partition_one() -> None:
    from vidaio.scoring.config import RemovalWeights

    RemovalWeights(psnr=0.6, lpips=0.4)
    with pytest.raises(ValueError):
        RemovalWeights(psnr=0.6, lpips=0.3)
    with pytest.raises(ValueError):
        RemovalWeights(psnr=0.9, lpips=0.9)


def test_score_removal_publishes_its_decision_limits() -> None:
    cfg = ScoringConfig()
    floor = RegionMetrics(psnr_db=29.1, ssim=0.78, lpips_vgg=0.25, warp_error=0.46, masked_pixels=1000)
    good = RegionMetrics(psnr_db=33.3, ssim=0.87, lpips_vgg=0.07, warp_error=0.83, masked_pixels=1000)
    b = score_removal(metrics=good, floor=floor, warp_error_reference=0.59, config=cfg)
    assert b.psnr_margin_db == cfg.removal_psnr_margin_db
    assert b.warp_cap == pytest.approx(cfg.removal_warp_cap_factor * max(0.59, cfg.removal_warp_floor))


def _hung_ffmpeg(tmp_path):
    import stat as _stat
    import sys as _sys

    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    ffprobe.write_text("#!/bin/sh\necho 64,48\n")
    ffmpeg.write_text(f"#!{_sys.executable}\nimport time\ntime.sleep(60)\n")
    for f in (ffmpeg, ffprobe):
        f.chmod(f.stat().st_mode | _stat.S_IEXEC)
    return str(ffmpeg)


def test_mask_decode_is_killed_on_cancel_and_timeout(tmp_path):
    import time as _time

    from vidaio.scoring.removal import RemovalCancelled, Y4MError, decode_mask_stream

    ffmpeg = _hung_ffmpeg(tmp_path)
    started = _time.monotonic()
    with pytest.raises(RemovalCancelled):
        decode_mask_stream(ffmpeg, str(tmp_path / "x.mkv"), cancelled=lambda: _time.monotonic() - started > 0.5)
    assert _time.monotonic() - started < 10
    started = _time.monotonic()
    with pytest.raises(Y4MError, match="exceeded"):
        decode_mask_stream(ffmpeg, str(tmp_path / "x.mkv"), timeout=1.0)
    assert _time.monotonic() - started < 10
