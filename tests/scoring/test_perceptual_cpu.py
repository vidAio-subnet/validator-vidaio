from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from vidaio.scoring import (
    CpuPerceptualConfig,
    PerceptualStatistics,
    chroma_uv_result,
    grayscale_result,
    tone_manipulation_result,
)
from vidaio.scoring.backends_real import (
    CpuPerceptualCheckBackend,
    NotConfiguredError,
    PieAppTorchBackend,
    _verify_pieapp_weights,
)


def _stats(**updates) -> PerceptualStatistics:
    values = {
        "sampled_pixels": 4096,
        "reference_luma_mean": 0.50,
        "candidate_luma_mean": 0.51,
        "reference_luma_std": 0.20,
        "candidate_luma_std": 0.21,
        "reference_chroma_energy": 0.20,
        "candidate_chroma_energy": 0.19,
        "chroma_mae": 0.01,
    }
    values.update(updates)
    return PerceptualStatistics(**values)


def test_cpu_perceptual_accepts_honest_codec_drift() -> None:
    config = CpuPerceptualConfig()
    stats = _stats()
    assert tone_manipulation_result(stats, config).passed
    assert grayscale_result(stats, config).passed
    assert chroma_uv_result(stats, config).passed


def test_cpu_perceptual_rejects_clear_tone_manipulation() -> None:
    result = tone_manipulation_result(
        _stats(candidate_luma_mean=0.68), CpuPerceptualConfig()
    )
    assert not result.passed
    assert result.measure is not None and result.measure > 1.0
    assert "mean delta" in result.detail


def test_cpu_perceptual_rejects_grayscale_conversion_but_not_gray_source() -> None:
    config = CpuPerceptualConfig()
    converted = grayscale_result(
        _stats(candidate_chroma_energy=0.01), config
    )
    assert not converted.passed
    naturally_gray = grayscale_result(
        _stats(reference_chroma_energy=0.01, candidate_chroma_energy=0.0), config
    )
    assert naturally_gray.passed


def test_cpu_perceptual_rejects_uv_replacement() -> None:
    result = chroma_uv_result(_stats(chroma_mae=0.25), CpuPerceptualConfig())
    assert not result.passed
    assert result.measure is not None and result.measure > 1.0
    assert "chroma MAE" in result.detail


def test_cpu_perceptual_config_is_validated_and_identity_bound() -> None:
    base = CpuPerceptualConfig()
    changed = CpuPerceptualConfig(tone_mean_delta_max=0.07)
    assert base.digest() != changed.digest()
    with pytest.raises(ValidationError, match="chroma_energy_ratio_max"):
        CpuPerceptualConfig(
            chroma_energy_ratio_min=2.0, chroma_energy_ratio_max=1.0
        )


def test_backend_reuses_one_cpu_analysis_for_all_three_gates(tmp_path) -> None:
    reference = tmp_path / "reference.y4m"
    candidate = tmp_path / "candidate.y4m"
    reference.write_bytes(b"reference")
    candidate.write_bytes(b"candidate")
    calls = []

    def analyze(ref, cand, config):
        calls.append((ref, cand, config.digest()))
        return _stats()

    backend = CpuPerceptualCheckBackend(
        _analyzer=analyze, _backend_version="opencv/test:algorithm/1"
    )
    assert backend.check_tone_manipulation(str(reference), str(candidate)).passed
    assert backend.check_color_grayscale(str(reference), str(candidate)).passed
    assert backend.check_chroma_uv(str(reference), str(candidate)).passed
    assert len(calls) == 1


def test_pieapp_preload_initializes_weights_once_without_media() -> None:
    loaded = []

    class Runtime:
        def compute(self, *args, **kwargs):
            return 0.1

    backend = PieAppTorchBackend(
        device="cpu",
        _runtime_loader=lambda device: loaded.append(device) or Runtime(),
        _backend_version="piq/test:pieapp",
    )
    backend.preload()
    backend.ensure_ready()
    assert loaded == ["cpu"]


def test_pieapp_weight_preflight_rejects_digest_mismatch(tmp_path, monkeypatch) -> None:
    weights = tmp_path / "PieAPPv0.1.pth"
    weights.write_bytes(b"pinned weights")
    monkeypatch.setattr(
        "vidaio.scoring.backends_real.PIEAPP_WEIGHTS_SHA256",
        hashlib.sha256(b"pinned weights").hexdigest(),
    )
    _verify_pieapp_weights(weights)
    weights.write_bytes(b"tampered")
    with pytest.raises(NotConfiguredError, match="digest mismatch"):
        _verify_pieapp_weights(weights)
