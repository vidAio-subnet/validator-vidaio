"""ScoringConfig defaults, validation, and core section-loader integration."""

from pathlib import Path

import pytest

from vidaio.core.config import load_raw_config, section
from vidaio.scoring import ScoringConfig
from vidaio.scoring.config import TRACK_COMPRESSION, TRACK_UPSCALING


def test_spec_defaults() -> None:
    cfg = ScoringConfig()
    assert cfg.vmaf_gate_upscaling == 0.5
    assert cfg.compression_rate_max == 0.80
    assert cfg.compression_weights.comp == 0.7
    assert cfg.compression_weights.vmaf == 0.3
    assert cfg.compression_norm == 1.12
    assert cfg.vmaf_model_delta_max == 3.0
    assert cfg.upscale_length_log_base == 321.0
    assert cfg.upscale_exponent == 6.979
    assert cfg.upscale_coefficient == 0.1
    assert cfg.file_size_caps == {2: 8.0, 4: 20.0}
    assert cfg.worst_decile_fraction == 0.1
    assert cfg.aggregate_weights.quality == 0.6
    assert cfg.aggregate_weights.cost_efficiency == 0.25
    assert cfg.aggregate_weights.length_coverage == 0.15


def test_vmaf_floor_and_threshold_per_track() -> None:
    cfg = ScoringConfig()
    assert cfg.vmaf_threshold(TRACK_UPSCALING) == 50.0
    assert cfg.vmaf_floor(TRACK_UPSCALING) == 50.0
    assert cfg.vmaf_threshold(TRACK_COMPRESSION) == 90.0
    assert cfg.vmaf_floor(TRACK_COMPRESSION) == 85.0
    with pytest.raises(KeyError):
        cfg.vmaf_threshold("interpolation")


def test_validation_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        ScoringConfig(compression_rate_max=0.0)
    with pytest.raises(ValueError):
        ScoringConfig(worst_decile_fraction=1.5)
    with pytest.raises(ValueError):
        ScoringConfig(upscale_length_log_base=1.0)
    with pytest.raises(ValueError):
        ScoringConfig(compression_norm=0.0)
    with pytest.raises(ValueError):
        ScoringConfig(pieapp_sample_window=0)
    with pytest.raises(ValueError):
        ScoringConfig(upscale_exponent=0.0)
    with pytest.raises(ValueError):
        ScoringConfig(upscale_coefficient=0.0)
    with pytest.raises(ValueError):
        ScoringConfig(vmaf_gate_upscaling=0.0)
    with pytest.raises(ValueError):
        ScoringConfig(vmaf_model_delta_max=-1.0)
    with pytest.raises(ValueError):
        ScoringConfig(file_size_caps={2: 0.0, 4: 20.0})
    with pytest.raises(ValueError):
        ScoringConfig(vmaf_thresholds={"compression": 101.0})


NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize(
    "field",
    [
        "vmaf_gate_upscaling",
        "compression_rate_max",
        "vmaf_model_delta_max",
        "compression_vmaf_band",
        "compression_norm",
        "upscale_length_log_base",
        "upscale_exponent",
        "upscale_coefficient",
        "worst_decile_fraction",
    ],
)
def test_non_finite_scalar_config_raises_at_construction(field: str, bad: float) -> None:
    # Fail closed at the config boundary: NaN compares False against every bound, so a
    # NaN compression_norm previously sailed past `<= 0` and composed score 1.0.
    with pytest.raises(ValueError):
        ScoringConfig(**{field: bad})


@pytest.mark.parametrize("bad", NON_FINITE)
def test_non_finite_dict_config_values_raise(bad: float) -> None:
    with pytest.raises(ValueError):
        ScoringConfig(file_size_caps={2: bad, 4: 20.0})
    with pytest.raises(ValueError):
        ScoringConfig(vmaf_thresholds={"compression": bad})


@pytest.mark.parametrize("bad", NON_FINITE)
def test_non_finite_weights_raise(bad: float) -> None:
    with pytest.raises(ValueError):
        ScoringConfig(compression_weights={"comp": bad, "vmaf": 0.3})
    with pytest.raises(ValueError):
        ScoringConfig(compression_weights={"comp": 0.7, "vmaf": bad})
    with pytest.raises(ValueError):
        ScoringConfig(
            aggregate_weights={
                "quality": bad, "cost_efficiency": 0.25, "length_coverage": 0.15
            }
        )


def test_weight_ranges_and_aggregate_sum() -> None:
    # Compression weights: each in [0, 1]; the pair need NOT sum to 1 (the formula's
    # min(1, .) clamp exists for exactly that case).
    ScoringConfig(compression_weights={"comp": 1.0, "vmaf": 1.0})
    with pytest.raises(ValueError):
        ScoringConfig(compression_weights={"comp": 1.5, "vmaf": 0.3})
    with pytest.raises(ValueError):
        ScoringConfig(compression_weights={"comp": -0.1, "vmaf": 0.3})
    # Aggregate weights are a convex combination: each in [0, 1], summing to 1.
    with pytest.raises(ValueError):
        ScoringConfig(
            aggregate_weights={
                "quality": 0.6, "cost_efficiency": 0.25, "length_coverage": 0.5
            }
        )
    with pytest.raises(ValueError):
        ScoringConfig(
            aggregate_weights={
                "quality": 1.0, "cost_efficiency": 0.0, "length_coverage": 0.5
            }
        )
    # Float representation error within 1e-9 is tolerated (0.3+0.3+0.4 != 1.0 exactly).
    cfg = ScoringConfig(
        aggregate_weights={
            "quality": 0.3, "cost_efficiency": 0.3, "length_coverage": 0.4
        }
    )
    assert cfg.aggregate_weights.quality == 0.3


def test_loads_via_core_section_with_yaml_and_env(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "scoring:\n  compression_rate_max: 0.75\n  vmaf_thresholds:\n    compression: 88.0\n"
    )
    monkeypatch.setenv("VIDAIO__SCORING__COMPRESSION_NORM", "1.5")
    raw = load_raw_config(cfg_file)
    cfg = section(raw, "scoring", ScoringConfig)
    assert cfg.compression_rate_max == 0.75
    assert cfg.vmaf_thresholds["compression"] == 88.0
    assert cfg.compression_norm == 1.5
    assert cfg.upscale_exponent == 6.979  # untouched default


def test_empty_section_yields_all_defaults() -> None:
    raw = load_raw_config(None)
    cfg = section(raw, "scoring", ScoringConfig)
    assert cfg == ScoringConfig()
