"""The streaming compression selector in the competition manifest: fenced, validated, and
the only compression manifests that may anchor a speed term."""

from __future__ import annotations

import pytest

from vidaio.competition.manifest import CompetitionManifest

from orchestrator_support import build_manifest

RULES = {"bits_per_pixel": 0.05, "max_keyframe_interval_seconds": 2.0, "peak_ratio": 1.5, "peak_window_seconds": 2.0}
SPEED = {"crown_margin": 0.05, "crown_min_score": 0.8, "podium_min_score": 0.6, "speed_weight": 0.3,
         "speed_fast_spf": 1 / 30, "speed_slow_spf": 2.0, "speed_inserted_frames": 1200}
RESOURCES = {"cpu": 32.0, "memory_mb": 65536, "batch_timeout_seconds": 3600}


def _streaming(**overrides):
    return build_manifest("comp-streaming", compression_scoring="streaming", streaming_rules=RULES, **overrides)


def test_selector_and_rules_are_fenced_out_of_classic_manifests() -> None:
    streaming = _streaming()
    assert '"compression_scoring":"streaming"' in streaming.canonical_json()
    assert '"bits_per_pixel":0.05' in streaming.canonical_json()
    classic = build_manifest("comp-streaming")
    assert "compression_scoring" not in classic.canonical_json() and "streaming_rules" not in classic.canonical_json()
    assert classic.manifest_digest() != streaming.manifest_digest()


def test_selector_and_rules_come_together_on_compression_only() -> None:
    with pytest.raises(ValueError, match="streaming_rules is set exactly"):
        build_manifest("comp-streaming", compression_scoring="streaming")
    with pytest.raises(ValueError, match="streaming_rules is set exactly"):
        build_manifest("comp-streaming", streaming_rules=RULES)
    with pytest.raises(ValueError):
        build_manifest("comp-streaming", compression_scoring="vbr", streaming_rules=RULES)


def test_speed_term_is_allowed_for_streaming_compression_only() -> None:
    assert _streaming(result_rules=SPEED, sandbox_resources=RESOURCES).result_rules.has_speed_term
    with pytest.raises(ValueError, match="speed term"):
        build_manifest("comp-streaming", result_rules=SPEED, sandbox_resources=RESOURCES)
    with pytest.raises(ValueError, match="sandbox_resources"):
        _streaming(result_rules=SPEED)
    assert isinstance(_streaming(), CompetitionManifest)
