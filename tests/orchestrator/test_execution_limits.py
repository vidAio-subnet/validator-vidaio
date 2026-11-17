"""Execution limits for large (generative) contender images: builds get the documented
30 minutes, the image-size walk has its own configurable bound, and manifests may anchor
an image cap of up to 40 GB."""

from __future__ import annotations

import pytest

from vidaio.competition.config import CompetitionConfig
from vidaio.competition.manifest import ManifestBoundsError, validate_against_config
from vidaio.competition.orchestrator.config import OrchestratorConfig

from orchestrator_support import build_manifest


def test_orchestrator_defaults_match_the_mining_guide() -> None:
    cfg = OrchestratorConfig()
    assert cfg.build_timeout_seconds == 1800.0
    assert cfg.image_measure_timeout_seconds == 1200.0


def test_limits_stay_configurable_and_positive() -> None:
    cfg = OrchestratorConfig(build_timeout_seconds=2700, image_measure_timeout_seconds=600)
    assert (cfg.build_timeout_seconds, cfg.image_measure_timeout_seconds) == (2700.0, 600.0)
    for field in ("build_timeout_seconds", "image_measure_timeout_seconds"):
        with pytest.raises(ValueError):
            OrchestratorConfig(**{field: 0})


@pytest.mark.parametrize("cap", [25.0, 40.0])
def test_image_caps_up_to_40_gb_are_accepted(cap: float) -> None:
    assert CompetitionConfig().container_size_limit_gb_max == 40.0
    validate_against_config(build_manifest(container_size_limit_gb=cap), CompetitionConfig())


def test_image_cap_over_40_gb_is_refused() -> None:
    with pytest.raises(ManifestBoundsError, match="container_size_limit_gb 40.5 exceeds max 40"):
        validate_against_config(build_manifest(container_size_limit_gb=40.5), CompetitionConfig())


def test_shipped_configs_carry_the_new_limits() -> None:
    from pathlib import Path

    from vidaio.core import section
    from vidaio.core.config import load_raw_config

    root = Path(__file__).resolve().parents[2]
    raw = load_raw_config(root / "config" / "default.yaml")
    orchestrator = section(raw, "orchestrator", OrchestratorConfig)
    competition = section(raw, "competition", CompetitionConfig)
    assert orchestrator.build_timeout_seconds >= 1800.0
    assert orchestrator.image_measure_timeout_seconds >= 1200.0
    assert competition.container_size_limit_gb_max == 40.0
