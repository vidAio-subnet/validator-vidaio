"""Frame-interpolation metrics and composition on tiny synthetic y4m clips (no ffmpeg needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vidaio.scoring import ScoringConfig, score_interpolation
from vidaio.scoring.interpolation import inserted_frame_metrics, kept_frame_change
from vidaio.scoring.interpolation_formula import InsertedFrameMetrics, best_baseline
from vidaio.scoring.removal import Y4MError, Y4MReader
from vidaio.scoring_worker.interpolation_backend import (
    CpuInterpolationBackend,
    InterpolationInputError,
    InterpolationMeasurementError,
)

W, H, N_REF = 64, 48, 9  # 9 reference frames -> 5 input frames, 4 inserted


def _write_y4m(path: Path, frames_rgb: np.ndarray, fps: int) -> None:
    """(n, h, w, 3) uint8 RGB -> 8-bit 4:2:0 y4m (BT.601 limited like ffmpeg's rgb->yuv420p)."""
    import cv2

    n, h, w, _ = frames_rgb.shape
    with open(path, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{w} H{h} F{fps}:1 Ip A1:1 C420jpeg\n".encode())
        for f in frames_rgb:
            fh.write(b"FRAME\n")
            fh.write(cv2.cvtColor(f, cv2.COLOR_RGB2YUV_I420).tobytes())


def _write_y4m_i420(path: Path, frames_i420: list[np.ndarray], fps: int) -> None:
    h = frames_i420[0].shape[0] * 2 // 3
    w = frames_i420[0].shape[1]
    with open(path, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{w} H{h} F{fps}:1 Ip A1:1 C420jpeg\n".encode())
        for f in frames_i420:
            fh.write(b"FRAME\n")
            fh.write(f.tobytes())


def _scene() -> np.ndarray:
    """A textured square sweeping across a panning gradient: real motion between frames."""
    yy, xx = np.mgrid[0:H, 0:W]
    frames = np.empty((N_REF, H, W, 3), np.uint8)
    for i in range(N_REF):
        pan = 3 * i
        frames[i, ..., 0] = ((xx + pan) * 4 % 256).astype(np.uint8)
        frames[i, ..., 1] = ((yy * 5 + pan) % 256).astype(np.uint8)
        frames[i, ..., 2] = 90
        x0 = 4 + 5 * i
        square = (xx >= x0) & (xx < x0 + 14) & (yy >= 16) & (yy < 30)
        checker = (((xx - x0) // 3 + (yy - 16) // 3) % 2).astype(bool)
        frames[i][square & checker] = (240, 240, 30)
        frames[i][square & ~checker] = (20, 20, 220)
    return frames


class _MadLpips:
    """A deterministic stand-in for LPIPS-VGG: mean abs difference on the 0-1 scale."""

    def __init__(self) -> None:
        self.calls = 0

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        self.calls += 1
        return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean() / 255.0)


@pytest.fixture
def clips(tmp_path: Path) -> dict[str, Path]:
    ref = _scene()
    served = ref[::2]
    hold = ref.copy()
    hold[1::2] = served[:-1]
    paths = {}
    for name, frames, fps in (
        ("reference", ref, 60),
        ("input", served, 30),
        ("perfect", ref, 60),
        ("hold", hold, 60),
    ):
        paths[name] = tmp_path / f"{name}.y4m"
        _write_y4m(paths[name], frames, fps)
    # the validator's blend: the rounded per-plane average of the two input frames (I420)
    import cv2

    served_i420 = [cv2.cvtColor(f, cv2.COLOR_RGB2YUV_I420) for f in served]
    blend_i420: list[np.ndarray] = []
    for k, frame in enumerate(served_i420):
        blend_i420.append(frame)
        if k + 1 < len(served_i420):
            nxt = served_i420[k + 1]
            blend_i420.append(((frame.astype(np.uint16) + nxt + 1) >> 1).astype(np.uint8))
    paths["blend"] = tmp_path / "blend.y4m"
    _write_y4m_i420(paths["blend"], blend_i420, 60)
    blend = np.stack([cv2.cvtColor(f, cv2.COLOR_YUV2RGB_I420) for f in blend_i420])
    halfway = ref.copy()
    halfway[1::2] = ((ref[1::2].astype(np.uint16) + blend[1::2] + 1) >> 1).astype(np.uint8)
    paths["halfway"] = tmp_path / "halfway.y4m"
    _write_y4m(paths["halfway"], halfway, 60)
    brightened = ref.copy()
    brightened[0::2] = np.clip(ref[0::2].astype(np.int16) + 12, 0, 255).astype(np.uint8)
    paths["kept_modified"] = tmp_path / "kept_modified.y4m"
    _write_y4m(paths["kept_modified"], brightened, 60)
    shifted = np.concatenate([ref[1:], ref[-1:]])
    paths["shifted"] = tmp_path / "shifted.y4m"
    _write_y4m(paths["shifted"], shifted, 60)
    paths["short"] = tmp_path / "short.y4m"
    _write_y4m(paths["short"], ref[:-2], 60)
    return paths


def _score(paths: dict[str, Path], name: str, *, config: ScoringConfig | None = None):
    config = config or ScoringConfig()
    cand, hold, blend = inserted_frame_metrics(
        Y4MReader.open(str(paths[name])),
        Y4MReader.open(str(paths["reference"])),
        Y4MReader.open(str(paths["input"])),
        lpips=_MadLpips(),
        lpips_stride=1,
    )
    return score_interpolation(metrics=cand, floor=best_baseline(hold, blend), config=config)


def test_the_free_answers_score_zero_and_the_real_frames_score_one(clips):
    assert _score(clips, "hold").final == 0.0
    assert _score(clips, "blend").final == 0.0
    perfect = _score(clips, "perfect")
    assert perfect.psnr_db == 99.0 and perfect.ssim == pytest.approx(1.0) and perfect.lpips_vgg == 0.0
    assert perfect.final == pytest.approx(1.0)
    halfway = _score(clips, "halfway")
    assert 0.0 < halfway.final < perfect.final
    assert halfway.s_psnr > 0 and halfway.s_ssim > 0 and halfway.s_lpips > 0


def test_formula_credits_only_the_gain_over_the_floor():
    config = ScoringConfig()
    floor = InsertedFrameMetrics(psnr_db=30.0, ssim=0.90, lpips_vgg=0.20, inserted_frames=10)
    good = InsertedFrameMetrics(psnr_db=34.0, ssim=0.95, lpips_vgg=0.05, inserted_frames=10)
    b = score_interpolation(metrics=good, floor=floor, config=config)
    assert b.s_psnr == pytest.approx(4.0 / 8.0)
    assert b.s_ssim == pytest.approx(0.05 / 0.10)
    assert b.s_lpips == pytest.approx(0.15 / 0.20)
    assert b.final == pytest.approx(0.3 * b.s_psnr + 0.3 * b.s_ssim + 0.4 * b.s_lpips)
    worse = InsertedFrameMetrics(psnr_db=29.0, ssim=0.85, lpips_vgg=0.30, inserted_frames=10)
    assert score_interpolation(metrics=worse, floor=floor, config=config).final == 0.0
    with pytest.raises(ValueError):
        score_interpolation(
            metrics=InsertedFrameMetrics(float("nan"), 1.0, 0.1, 1), floor=floor, config=config
        )


def test_best_baseline_takes_the_better_free_answer_per_term():
    hold = InsertedFrameMetrics(psnr_db=28.0, ssim=0.91, lpips_vgg=0.12, inserted_frames=4)
    blend = InsertedFrameMetrics(psnr_db=31.0, ssim=0.88, lpips_vgg=0.18, inserted_frames=4)
    floor = best_baseline(hold, blend)
    assert (floor.psnr_db, floor.ssim, floor.lpips_vgg) == (31.0, 0.91, 0.12)


def test_kept_frame_check_catches_modified_and_shifted_outputs(clips):
    served = Y4MReader.open(str(clips["input"]))
    assert kept_frame_change(Y4MReader.open(str(clips["perfect"])), served) < 1.0
    assert kept_frame_change(Y4MReader.open(str(clips["blend"])), served) < 1.0
    assert kept_frame_change(Y4MReader.open(str(clips["kept_modified"])), served) > 8.0
    assert kept_frame_change(Y4MReader.open(str(clips["shifted"])), served) > ScoringConfig().interpolation_kept_tolerance


def test_metrics_refuse_a_wrong_frame_grid(clips):
    ref = Y4MReader.open(str(clips["reference"]))
    served = Y4MReader.open(str(clips["input"]))
    with pytest.raises(Y4MError):
        inserted_frame_metrics(Y4MReader.open(str(clips["short"])), ref, served, lpips=None, lpips_stride=1)
    with pytest.raises(ValueError):
        # the input is not every other frame of this "reference"
        inserted_frame_metrics(ref, served, served, lpips=None, lpips_stride=1)


def test_lpips_samples_every_kth_inserted_frame_from_an_offset(clips):
    lpips = _MadLpips()
    inserted_frame_metrics(
        Y4MReader.open(str(clips["perfect"])),
        Y4MReader.open(str(clips["reference"])),
        Y4MReader.open(str(clips["input"])),
        lpips=lpips,
        lpips_stride=2,
        lpips_offset=1,
    )
    # 4 inserted frames, stride 2 -> 2 sampled positions, each for candidate/hold/blend
    assert lpips.calls == 6
    lpips = _MadLpips()
    inserted_frame_metrics(
        Y4MReader.open(str(clips["perfect"])),
        Y4MReader.open(str(clips["reference"])),
        Y4MReader.open(str(clips["input"])),
        lpips=lpips,
        lpips_stride=9,
        lpips_offset=3,
    )
    assert lpips.calls == 3  # fewer inserted frames than the stride: the last one is sampled


def test_backend_separates_validator_and_candidate_failures(clips):
    lpips = _MadLpips()
    backend = CpuInterpolationBackend(lpips_loader=lambda: lpips)
    m = backend.measure(
        canonical_reference=str(clips["reference"]),
        canonical_candidate=str(clips["halfway"]),
        canonical_input=str(clips["input"]),
        config=ScoringConfig(),
    )
    assert m.input_frames == 5 and m.metrics.inserted_frames == 4
    assert m.floor == best_baseline(m.hold, m.blend)
    assert m.kept_change < 1.0
    with pytest.raises(InterpolationMeasurementError):
        backend.measure(
            canonical_reference=str(clips["reference"]),
            canonical_candidate=str(clips["short"]),
            canonical_input=str(clips["input"]),
            config=ScoringConfig(),
        )
    with pytest.raises(InterpolationInputError):
        backend.measure(
            canonical_reference=str(clips["short"]),
            canonical_candidate=str(clips["perfect"]),
            canonical_input=str(clips["input"]),
            config=ScoringConfig(),
        )


def test_interpolation_config_is_validated():
    with pytest.raises(ValueError):
        ScoringConfig(interpolation_psnr_span_db=0.0)
    with pytest.raises(ValueError):
        ScoringConfig(interpolation_kept_tolerance=-1.0)
    with pytest.raises(ValueError):
        ScoringConfig(interpolation_lpips_stride=0)
    with pytest.raises(ValueError):
        ScoringConfig(interpolation_weights={"psnr": 0.5, "ssim": 0.5, "lpips": 0.5})


def test_digests_of_earlier_releases_stay_accepted():
    import hashlib

    from vidaio.scoring.result import (
        INTERPOLATION_CONFIG_FIELDS,
        REMOVAL_CONFIG_FIELDS,
        accepted_config_digests,
    )

    config = ScoringConfig()
    accepted = accepted_config_digests(config)
    for excluded in (INTERPOLATION_CONFIG_FIELDS, INTERPOLATION_CONFIG_FIELDS | REMOVAL_CONFIG_FIELDS):
        legacy = hashlib.sha256(config.model_dump_json(exclude=set(excluded)).encode()).hexdigest()
        assert legacy in accepted


def test_backend_measures_an_items_floor_once_for_every_candidate(clips):
    """The hold/blend floor depends on no candidate: with a floor key it is measured on
    the first candidate of an item and reused, and the measurement is identical to an
    uncached one."""
    lpips = _MadLpips()
    backend = CpuInterpolationBackend(lpips_loader=lambda: lpips)
    item = dict(
        canonical_reference=str(clips["reference"]),
        canonical_input=str(clips["input"]),
        config=ScoringConfig(),
    )
    first = backend.measure(canonical_candidate=str(clips["halfway"]), floor_key=("ref", "in"), **item)
    assert lpips.calls == 3  # candidate, hold and blend on the one sampled frame
    second = backend.measure(canonical_candidate=str(clips["perfect"]), floor_key=("ref", "in"), **item)
    assert lpips.calls == 4  # only the candidate
    uncached = CpuInterpolationBackend(lpips_loader=lambda: _MadLpips()).measure(
        canonical_candidate=str(clips["perfect"]), **item
    )
    assert second == uncached
    assert first.floor == second.floor and first.hold == second.hold
    # another LPIPS sample (offset) or another item is another floor
    backend.measure(
        canonical_candidate=str(clips["perfect"]), floor_key=("ref", "in"), lpips_offset=1, **item
    )
    backend.measure(canonical_candidate=str(clips["perfect"]), floor_key=("ref", "other"), **item)
    assert lpips.calls == 4 + 3 + 3
