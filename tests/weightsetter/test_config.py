"""WeightSetterConfig schema and defaults."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vidaio.weightsetter import WeightSetterConfig


def test_defaults():
    config = WeightSetterConfig()
    assert config.attempt_interval_seconds == 72 * 60  # design spec §01 cadence
    assert config.chain_timeout_seconds == 180.0
    assert config.chain_retry_attempts == 3
    assert config.chain_retry_base_delay_seconds == 1.0
    assert config.version_key == 16
    assert config.metrics_port == 9102
    assert config.publication_enabled is True
    assert config.max_last_success_age_seconds == 2 * 72 * 60
    assert config.reconciliation_interval_seconds == 300


@pytest.mark.parametrize(
    "field, value",
    [
        ("attempt_interval_seconds", 0),
        ("chain_timeout_seconds", -1),
        ("chain_retry_attempts", 0),
        ("chain_retry_base_delay_seconds", 0),
        ("version_key", -1),
        ("max_last_success_age_seconds", 0),
    ],
)
def test_out_of_range_values_rejected(field, value):
    with pytest.raises(ValidationError):
        WeightSetterConfig(**{field: value})


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        WeightSetterConfig(tempo=100)


def test_proposed_mainnet_reveal_grace_sets_8820_second_health_floor():
    config = WeightSetterConfig(reveal_grace_seconds=4500)
    assert config.max_last_success_age_seconds == 8640
    assert config.effective_max_last_success_age_seconds == 8820
    assert config.effective_max_last_success_age_seconds <= 2 * 4320 + 4500
