"""Generative upscaling: formula, gates, colour maths, sampling and the one-pass measurement on
tiny synthetic y4m clips (no ffmpeg, no model downloads: DISTS / LPIPS / CLIP-IQA are replaced
by deterministic stand-ins with the same interface; the real networks are covered when their
pinned weights are cached)."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vidaio.scoring import (
    GenerativeFrameMetrics,
    ReasonCode,
    ScoringConfig,
    score_upscaling_generative,
)
from vidaio.scoring.config import GenerativeAnchor
from vidaio.scoring.gates import GateContext, upscaling_generative_violations
from vidaio.scoring.generative_weights import (
    GenerativeWeightsError,
    PinnedWeights,
    pinned_weights_path,
)
from vidaio.scoring.removal import Y4MReader
from vidaio.scoring.upscaling_generative import (
    ciede2000,
    fr_frame_indices,
    fr_sampling_stride,
    generative_frame_metrics,
)
from vidaio.scoring.upscaling_generative_formula import band, nr_ratio, temporal_ratio
from vidaio.scoring_worker.upscaling_generative_backend import (
    CpuUpscalingGenerativeBackend,
    UpscalingGenerativeInputError,
    UpscalingGenerativeMeasurementError,
)

W, H, N = 96, 64, 8


def _metrics(**overrides) -> GenerativeFrameMetrics:
    base = dict(
        dists=0.10, lpips_vgg=0.20, warp_error=2.0, warp_error_reference=2.0, nr=0.60,
        nr_reference=0.60, lowres_psnr_db=36.0, color_delta_e=1.0, frames=120, fr_frames=3,
    )
    base.update(overrides)
    return GenerativeFrameMetrics(**base)


# --- formula ---------------------------------------------------------------------------


def test_band_maps_floor_to_zero_and_anchor_to_one() -> None:
    a = GenerativeAnchor(floor=0.5, anchor=0.9)
    assert band(0.5, a) == 0.0 and band(0.9, a) == 1.0
    assert band(0.7, a) == pytest.approx(0.5)
    assert band(0.1, a) == 0.0 and band(1.5, a) == 1.0
    with pytest.raises(ValueError):
        GenerativeAnchor(floor=0.9, anchor=0.9)


def test_temporal_ratio_never_rewards_smoothing_below_the_reference() -> None:
    assert temporal_ratio(4.0, 2.0, 0.5) == pytest.approx(0.5)  # flickers twice as much
    assert temporal_ratio(2.0, 2.0, 0.5) == 1.0
    assert temporal_ratio(0.1, 2.0, 0.5) == 1.0  # smoother than the original: no extra credit
    # a static reference is floored, so the ratio stays defined
    assert temporal_ratio(1.0, 0.0, 0.5) == pytest.approx(0.5)
    assert temporal_ratio(0.0, 0.0, 0.5) == 1.0


def test_nr_is_capped_at_the_reference_plus_five_percent() -> None:
    assert nr_ratio(0.30, 0.60, 0.05) == pytest.approx(0.5)
    assert nr_ratio(0.63, 0.60, 0.05) == pytest.approx(1.05)
    assert nr_ratio(0.99, 0.60, 0.05) == pytest.approx(1.05)  # "sharper than reality" earns nothing
    with pytest.raises(ValueError):
        nr_ratio(0.5, 0.0, 0.05)


def test_score_is_the_weighted_composite_of_the_three_bands() -> None:
    config = ScoringConfig()
    bd = score_upscaling_generative(metrics=_metrics(), upscale_factor=4, config=config)
    bands = config.upscaling_generative_anchors[4]
    s_dists = band(0.90, bands.dists_similarity)
    s_lpips = band(0.80, bands.lpips_similarity)
    s_t = band(1.0, bands.temporal)
    s_nr = band(1.0, bands.nr_ratio)
    expected = 0.55 * (s_dists + s_lpips) / 2 + 0.15 * s_t + 0.30 * s_nr
    assert bd.final == pytest.approx(min(1.0, expected))
    assert bd.kind == "upscaling_generative"
    assert (bd.weight_fr, bd.weight_temporal, bd.weight_nr) == (0.55, 0.15, 0.30)
    assert bd.band_dists == (bands.dists_similarity.floor, bands.dists_similarity.anchor)
    # the same measurement is worth less at x2 (tighter bands): factors are normalised apart
    bd2 = score_upscaling_generative(metrics=_metrics(), upscale_factor=2, config=config)
    assert bd2.final < bd.final


def test_a_non_generative_floor_scores_zero_and_the_anchor_scores_one() -> None:
    config = ScoringConfig()
    bands = config.upscaling_generative_anchors[2]
    floor = _metrics(
        dists=1 - bands.dists_similarity.floor, lpips_vgg=1 - bands.lpips_similarity.floor,
        warp_error=2.0 / bands.temporal.floor, warp_error_reference=2.0,
        nr=0.6 * bands.nr_ratio.floor, nr_reference=0.6,
    )
    assert score_upscaling_generative(metrics=floor, upscale_factor=2, config=config).final == pytest.approx(0.0, abs=1e-12)
    anchor = _metrics(
        dists=1 - bands.dists_similarity.anchor, lpips_vgg=1 - bands.lpips_similarity.anchor,
        warp_error=2.0 / bands.temporal.anchor, warp_error_reference=2.0,
        nr=0.6 * bands.nr_ratio.anchor, nr_reference=0.6,
    )
    assert score_upscaling_generative(metrics=anchor, upscale_factor=2, config=config).final == pytest.approx(1.0)


def test_formula_rejects_bad_inputs() -> None:
    config = ScoringConfig()
    with pytest.raises(ValueError):
        score_upscaling_generative(metrics=_metrics(), upscale_factor=3, config=config)
    with pytest.raises(ValueError):
        score_upscaling_generative(metrics=_metrics(dists=float("nan")), upscale_factor=2, config=config)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ScoringConfig(upscaling_generative_weights={"fr": 0.5, "temporal": 0.3, "nr": 0.3})
    with pytest.raises(ValueError):
        ScoringConfig(upscaling_generative_fr_frames=0)
    with pytest.raises(ValueError):
        ScoringConfig(upscaling_generative_lowres_blur_sigma=0.0)
    anchors = ScoringConfig().upscaling_generative_anchors
    with pytest.raises(ValueError, match="factors 2 and 4"):
        ScoringConfig(upscaling_generative_anchors={2: anchors[2]})
    bad_nr = anchors[4].model_copy(update={"nr_ratio": GenerativeAnchor(floor=0.8, anchor=1.2)})
    with pytest.raises(ValueError, match="nr_ratio"):
        ScoringConfig(upscaling_generative_anchors={2: anchors[2], 4: bad_nr})
    # JSON round trip (int factor keys) keeps the config and its digest
    from vidaio.scoring.result import config_digest

    config = ScoringConfig()
    again = ScoringConfig.model_validate_json(config.model_dump_json())
    assert config_digest(again) == config_digest(config)


# --- gates -----------------------------------------------------------------------------


def test_gates_use_the_larger_of_minimum_and_calibrated_floor() -> None:
    config = ScoringConfig()
    assert upscaling_generative_violations(lowres_psnr_db=28.0, color_delta_e=6.0, config=config) == []
    codes = [v.code for v in upscaling_generative_violations(lowres_psnr_db=27.9, color_delta_e=6.01, config=config)]
    assert codes == [ReasonCode.LOWRES_INCONSISTENT, ReasonCode.COLOR_SHIFT_EXCEEDED]
    calibrated = ScoringConfig(upscaling_generative_lowres_psnr_floor_db=31.0)
    (violation,) = upscaling_generative_violations(lowres_psnr_db=30.0, color_delta_e=1.0, config=calibrated)
    assert violation.code is ReasonCode.LOWRES_INCONSISTENT and violation.limit == 31.0
    nan = upscaling_generative_violations(lowres_psnr_db=float("nan"), color_delta_e=float("inf"), config=config)
    assert [v.code for v in nan] == [ReasonCode.METRIC_NON_FINITE, ReasonCode.METRIC_NON_FINITE]


def test_gate_context_skips_inference_gates_for_generative_upscaling() -> None:
    from vidaio.scoring import MediaInfo
    from vidaio.scoring.gates import FileSizeCapGate, VmafFloorGate, VmafModelDeltaGate

    info = MediaInfo(width=W, height=H, fps=30.0, frame_count=N, duration=N / 30, codec="ffv1",
                     pix_fmt="yuv420p", byte_size=10_000_000)
    ctx = GateContext(track="upscaling", config=ScoringConfig(), reference_info=info,
                      candidate_info=info, upscale_factor=2, upscaling_generative=True)
    assert ctx.reference_scored_upscaling
    for gate in (FileSizeCapGate(), VmafFloorGate(), VmafModelDeltaGate()):
        assert gate.check(ctx) == []


# --- colour ----------------------------------------------------------------------------

# Sharma, Wu & Dalal (2005), "The CIEDE2000 color-difference formula", Table 1 (selection).
_SHARMA = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0009), 7.1792),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (56.0, -27.0, -3.0), 31.9030),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize(("lab1", "lab2", "expected"), _SHARMA)
def test_ciede2000_matches_the_published_test_data(lab1, lab2, expected) -> None:
    value = float(ciede2000(np.array(lab1), np.array(lab2)))
    assert value == pytest.approx(expected, abs=1e-4)
    assert float(ciede2000(np.array(lab2), np.array(lab1))) == pytest.approx(expected, abs=1e-4)


# --- sampling --------------------------------------------------------------------------


def test_full_frame_sample_is_evenly_spaced_and_phase_shifted() -> None:
    assert fr_sampling_stride(120, 3) == 40
    assert fr_frame_indices(120, 3, 0) == [0, 40, 80]
    assert fr_frame_indices(120, 3, 41) == [1, 41, 81]
    assert fr_frame_indices(2, 3, 5) == [0, 1]
    assert all(i < 121 for i in fr_frame_indices(121, 4, 29))


# --- measurement -----------------------------------------------------------------------


def _write_y4m(path: Path, frames_rgb: np.ndarray, fps: int = 30) -> None:
    import cv2

    n, h, w, _ = frames_rgb.shape
    with open(path, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{w} H{h} F{fps}:1 Ip A1:1 C420jpeg\n".encode())
        for f in frames_rgb:
            fh.write(b"FRAME\n")
            fh.write(cv2.cvtColor(f, cv2.COLOR_RGB2YUV_I420).tobytes())


def _scene() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    frames = np.empty((N, H, W, 3), np.uint8)
    for i in range(N):
        pan = 2 * i
        frames[i, ..., 0] = (60 + 120 * (np.sin((xx + pan) / 7.0) > 0)).astype(np.uint8)
        frames[i, ..., 1] = ((yy * 3 + pan) % 200 + 30).astype(np.uint8)
        frames[i, ..., 2] = (100 + 40 * np.cos((xx - yy + pan) / 9.0)).astype(np.uint8)
    return frames


class _MadDistance:
    """Deterministic stand-in for DISTS / LPIPS: mean abs difference on the 0-1 scale."""

    def __init__(self) -> None:
        self.calls = 0

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        self.calls += 1
        return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean() / 255.0)


class _SharpnessNr:
    """Deterministic stand-in for CLIP-IQA: bounded mean gradient magnitude."""

    def __init__(self) -> None:
        self.calls = 0

    def quality(self, rgb: np.ndarray) -> float:
        self.calls += 1
        g = np.abs(np.diff(rgb.astype(np.float64).mean(axis=2), axis=1)).mean()
        return float(g / (g + 10.0))


@pytest.fixture
def clips(tmp_path: Path) -> dict[str, Path]:
    import cv2

    ref = _scene()
    served = np.stack([cv2.resize(f, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for f in ref])
    lanczos = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LANCZOS4) for f in served])
    shifted = np.clip(ref.astype(np.int16) + np.array([40, -25, -30]), 0, 255).astype(np.uint8)
    invented = ref.copy()
    invented[:, 8:56, 10:70] = 255 - invented[:, 8:56, 10:70]  # structure the input does not show
    flicker = ref.copy()
    flicker[1::2] = np.clip(flicker[1::2].astype(np.int16) + 18, 0, 255).astype(np.uint8)
    paths = {}
    for name, frames in (("reference", ref), ("served", served), ("lanczos", lanczos),
                         ("shifted", shifted), ("invented", invented), ("flicker", flicker)):
        paths[name] = tmp_path / f"{name}.y4m"
        _write_y4m(paths[name], frames)
    return paths


def _measure(clips, candidate: str, config: ScoringConfig | None = None, **kwargs):
    return generative_frame_metrics(
        Y4MReader.open(str(clips[candidate])),
        Y4MReader.open(str(clips["reference"])),
        Y4MReader.open(str(clips["served"])),
        dists=_MadDistance(), lpips=_MadDistance(), nr=_SharpnessNr(),
        config=config or ScoringConfig(), **kwargs,
    )


def test_the_original_itself_passes_every_gate_and_matches_the_reference(clips) -> None:
    m = _measure(clips, "reference")
    assert m.dists == 0.0 and m.lpips_vgg == 0.0
    assert m.warp_error == pytest.approx(m.warp_error_reference)
    assert m.nr == pytest.approx(m.nr_reference)
    assert m.color_delta_e == pytest.approx(0.0, abs=1e-9)
    assert m.lowres_psnr_db > 33.0
    assert m.frames == N and m.fr_frames == 3
    assert upscaling_generative_violations(
        lowres_psnr_db=m.lowres_psnr_db, color_delta_e=m.color_delta_e, config=ScoringConfig()
    ) == []


def test_the_gates_catch_colour_shifts_and_invented_structure(clips) -> None:
    config = ScoringConfig()
    shifted = _measure(clips, "shifted")
    assert shifted.color_delta_e > config.upscaling_generative_color_delta_e_max
    invented = _measure(clips, "invented")
    assert invented.lowres_psnr_db < config.upscaling_generative_lowres_psnr_min_db
    lanczos = _measure(clips, "lanczos")  # a plain upscale is consistent with its input
    assert lanczos.lowres_psnr_db > config.upscaling_generative_lowres_psnr_min_db
    assert lanczos.color_delta_e < config.upscaling_generative_color_delta_e_max


def test_flicker_raises_the_warp_error_above_the_reference(clips) -> None:
    flicker = _measure(clips, "flicker")
    original = _measure(clips, "reference")
    assert flicker.warp_error > original.warp_error_reference + 5.0
    assert flicker.warp_error_reference == pytest.approx(original.warp_error_reference)
    assert temporal_ratio(flicker.warp_error, flicker.warp_error_reference, 0.5) < 0.5


def test_measurement_is_deterministic_and_caches_only_the_reference_side(clips) -> None:
    first = _measure(clips, "lanczos", fr_offset=1)
    again = _measure(clips, "lanczos", fr_offset=1)
    assert first == again
    cached = _measure(clips, "lanczos", fr_offset=1, nr_reference=first.nr_reference)
    assert cached == first
    other_phase = _measure(clips, "lanczos", fr_offset=2)
    assert other_phase.dists != first.dists  # a different sample of frames


def test_measurement_rejects_mismatched_inputs(clips, tmp_path) -> None:
    short = tmp_path / "short.y4m"
    _write_y4m(short, _scene()[:-1])
    with pytest.raises(Exception):
        generative_frame_metrics(
            Y4MReader.open(str(short)), Y4MReader.open(str(clips["reference"])),
            Y4MReader.open(str(clips["served"])), dists=_MadDistance(), lpips=_MadDistance(),
            nr=_SharpnessNr(), config=ScoringConfig(),
        )


def _backend(nr=None):
    return CpuUpscalingGenerativeBackend(
        lpips_loader=_MadDistance, dists_loader=_MadDistance, nr_loader=lambda: nr or _SharpnessNr()
    )


def test_backend_measures_the_reference_quality_once_per_item(clips) -> None:
    nr = _SharpnessNr()
    backend = _backend(nr)
    common = dict(
        canonical_reference=str(clips["reference"]), canonical_served_input=str(clips["served"]),
        config=ScoringConfig(), fr_offset=0, reference_key=("ref-digest", "input-digest"),
    )
    a = backend.measure(canonical_candidate=str(clips["lanczos"]), **common)
    calls_after_first = nr.calls
    b = backend.measure(canonical_candidate=str(clips["flicker"]), **common)
    assert nr.calls - calls_after_first == 3  # candidate frames only
    assert a.metrics.nr_reference == b.metrics.nr_reference
    uncached = _backend().measure(canonical_candidate=str(clips["flicker"]), **{**common, "reference_key": None})
    assert uncached.metrics == b.metrics


def test_backend_separates_validator_and_candidate_faults(clips, tmp_path) -> None:
    backend = _backend()
    common = dict(config=ScoringConfig())
    with pytest.raises(UpscalingGenerativeInputError):
        backend.measure(canonical_reference=str(tmp_path / "missing.y4m"), canonical_candidate=str(clips["lanczos"]),
                        canonical_served_input=str(clips["served"]), **common)
    with pytest.raises(UpscalingGenerativeMeasurementError):
        backend.measure(canonical_reference=str(clips["reference"]), canonical_candidate=str(clips["served"]),
                        canonical_served_input=str(clips["served"]), **common)
    short = tmp_path / "short-served.y4m"
    import cv2

    _write_y4m(short, np.stack([cv2.resize(f, (W // 2, H // 2)) for f in _scene()[:-1]]))
    with pytest.raises(UpscalingGenerativeInputError):
        backend.measure(canonical_reference=str(clips["reference"]), canonical_candidate=str(clips["lanczos"]),
                        canonical_served_input=str(short), **common)


def test_fr_max_side_bounds_the_frames_given_to_the_networks(clips) -> None:
    seen: list[tuple[int, int]] = []

    class _Recording(_MadDistance):
        def distance(self, a, b):
            seen.append(a.shape[:2])
            return super().distance(a, b)

    generative_frame_metrics(
        Y4MReader.open(str(clips["lanczos"])), Y4MReader.open(str(clips["reference"])),
        Y4MReader.open(str(clips["served"])), dists=_Recording(), lpips=_MadDistance(), nr=_SharpnessNr(),
        config=ScoringConfig(upscaling_generative_fr_max_side=48),
    )
    assert seen and all(max(shape) == 48 for shape in seen)
    seen.clear()
    generative_frame_metrics(
        Y4MReader.open(str(clips["lanczos"])), Y4MReader.open(str(clips["reference"])),
        Y4MReader.open(str(clips["served"])), dists=_Recording(), lpips=_MadDistance(), nr=_SharpnessNr(),
        config=ScoringConfig(),
    )
    assert seen and all(shape == (H, W) for shape in seen)  # native full frames by default


# --- pinned weights ----------------------------------------------------------------------


def test_pinned_weights_are_verified_and_never_downloaded_at_scoring_time(tmp_path, monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    source = tmp_path / "source.bin"
    source.write_bytes(b"pinned weights payload")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    weights = PinnedWeights(url=source.as_uri(), filename="w.bin", sha256=digest)
    with pytest.raises(GenerativeWeightsError, match="not cached"):
        pinned_weights_path(weights)
    path = pinned_weights_path(weights, download=True)
    assert path.read_bytes() == source.read_bytes()
    assert pinned_weights_path(weights) == path
    wrong = replace(weights, filename="x.bin", sha256="0" * 64)
    with pytest.raises(GenerativeWeightsError, match="digest mismatch"):
        pinned_weights_path(wrong, download=True)
    assert not (tmp_path / "hub" / "checkpoints" / "x.bin").exists()
    mirrored = replace(weights, url=(tmp_path / "gone.bin").as_uri(), filename="m.bin",
                       mirrors=(source.as_uri(),))
    assert pinned_weights_path(mirrored, download=True).read_bytes() == source.read_bytes()
    with pytest.raises(GenerativeWeightsError, match="could not fetch"):
        pinned_weights_path(replace(mirrored, filename="n.bin", mirrors=()), download=True)
    tampered = tmp_path / "hub" / "checkpoints" / "y.bin"
    tampered.write_bytes(b"something else")
    with pytest.raises(GenerativeWeightsError, match="digest mismatch"):
        pinned_weights_path(replace(weights, filename="y.bin"))


def _real_networks_or_skip():
    from vidaio.scoring.generative_weights import GENERATIVE_WEIGHTS

    try:
        for weights in GENERATIVE_WEIGHTS:
            pinned_weights_path(weights)
    except GenerativeWeightsError as exc:
        pytest.skip(f"pinned generative weights are not cached: {exc}")
    from vidaio.scoring.upscaling_generative import ClipIqa, DistsFull

    return DistsFull(), ClipIqa()


def test_real_dists_and_clipiqa_are_deterministic_on_cpu() -> None:
    dists, nr = _real_networks_or_skip()
    rng = np.random.default_rng(5)
    a = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    b = np.clip(a.astype(np.int16) + rng.integers(-25, 26, size=a.shape), 0, 255).astype(np.uint8)
    assert dists.distance(a, a) < 1e-5
    d = dists.distance(a, b)
    assert 0.0 < d < 1.0 and dists.distance(a, b) == d
    q = nr.quality(a)
    assert 0.0 < q < 1.0 and nr.quality(a) == q
