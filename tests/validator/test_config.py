"""ValidatorConfig defaults and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vidaio.core import section
from vidaio.validator import ValidatorConfig


def test_defaults_match_spec_anchors():
    cfg = ValidatorConfig()
    assert (cfg.cycle_sleep_min_seconds, cfg.cycle_sleep_max_seconds) == (3600.0, 7200.0)
    assert cfg.metagraph_refresh_seconds == 1800.0
    assert cfg.min_stake == 0.0
    assert cfg.warrant_probe_timeout_seconds == 10.0
    assert cfg.miner_request_timeout_seconds == 300.0
    assert cfg.scoring_worker_url == "http://127.0.0.1:8201"
    assert cfg.challenge_service_url == "http://127.0.0.1:8210"
    assert cfg.miner_url_scheme == "http"
    assert cfg.miner_port == 8300
    assert cfg.metrics_port == 9101
    # EWMA decay deliberately absent: it lives in the tokenomics section only
    assert "ewma_decay" not in ValidatorConfig.model_fields


def test_section_parsing_and_overrides():
    raw = {"validator": {"min_stake": 25.0, "metrics_port": 9201}}
    cfg = section(raw, "validator", ValidatorConfig)
    assert cfg.min_stake == 25.0 and cfg.metrics_port == 9201


@pytest.mark.parametrize(
    "overrides",
    [
        {"cycle_sleep_min_seconds": 0},
        {"cycle_sleep_min_seconds": 100, "cycle_sleep_max_seconds": 50},
        {"metagraph_refresh_seconds": -1},
        {"min_stake": -0.1},
        {"warrant_probe_timeout_seconds": 0},
        {"miner_request_timeout_seconds": -5},
        {"miner_url_scheme": "ftp"},
        {"metrics_port": 0},
        {"not_a_real_key": 1},
    ],
)
def test_invalid_configs_rejected(overrides):
    with pytest.raises(ValidationError):
        ValidatorConfig.model_validate(overrides)
