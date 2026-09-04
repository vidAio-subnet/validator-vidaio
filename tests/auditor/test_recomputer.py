"""The REAL ScoreRecomputer over the actual scoring engine — real-media parity.

Parity + tamper tests use real ffmpeg/libvmaf on a tiny clip (skip without media
tools). The media-free honest-refusal tests live in test_recomputer_refusal.py.
"""

from __future__ import annotations

import json

from tests.auditor.conftest import (
    build_real_bundle,
    requires_media_tools,
    score_compression_item,
)
from vidaio.audit.recompute import SCORE_MISMATCH, verify_bundle
from vidaio.audit.store import LocalFsStore
from vidaio.auditor import RealScoreRecomputer


pytestmark = requires_media_tools


def test_honest_compression_packet_recomputes_clean(
    tmp_path, clips, worker_config, real_media_backends, scoring_config
) -> None:
    store = LocalFsStore(tmp_path / "audit")
    item = score_compression_item(worker_config, real_media_backends, scoring_config, clips)
    assert item.gate_passed and item.score > 0.0  # sanity: a real, non-zero score
    bundle = build_real_bundle(store, clips, item)

    recomputer = RealScoreRecomputer(
        worker_config,
        real_media_backends,
        scoring_config=scoring_config,
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )
    report = verify_bundle(bundle, store, recomputer, strict=False)

    assert report.passed, [c.model_dump() for c in report.failures()]
    # the recompute checks actually ran (not skipped away) and agreed
    names = {c.name for c in report.checks if c.passed}
    assert "score_recompute:score" in names
    assert "gate_recompute" in names
    assert "metric_set" in names
    assert "backend_versions_recompute" in names


def test_tampered_inflated_score_fails_score_mismatch(
    tmp_path, clips, worker_config, real_media_backends, scoring_config
) -> None:
    store = LocalFsStore(tmp_path / "audit")
    item = score_compression_item(worker_config, real_media_backends, scoring_config, clips)

    # Inflate the packet's top-level score AND its final_score metric/breakdown so it
    # stays internally consistent (gates-first holds) — the misreport is only caught by an
    # INDEPENDENT recompute over the real bytes, exactly the injection verify_bundle kills.
    packet = json.loads(item.to_json())
    packet["score"] = 0.99
    packet["metrics"]["final_score"] = 0.99
    if isinstance(packet.get("breakdown"), dict):
        packet["breakdown"]["final"] = 0.99
    tampered = json.dumps(packet).encode("utf-8")

    bundle = build_real_bundle(store, clips, item, packet_bytes=tampered)
    recomputer = RealScoreRecomputer(
        worker_config,
        real_media_backends,
        scoring_config=scoring_config,
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )
    report = verify_bundle(bundle, store, recomputer, strict=False)

    assert not report.passed
    codes = {c.code for c in report.failures()}
    assert SCORE_MISMATCH in codes
