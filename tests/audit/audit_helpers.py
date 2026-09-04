"""Shared builders for the audit test suite."""

from typing import Any

from vidaio.audit.bundle import AuditBundle, LifecycleStage, build_bundle
from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind, LocalFsStore

HONEST_METRICS = {"compression_rate": 0.125, "vmaf": 93.42, "final_score": 0.81}
SCORER_VERSION = "scoring-1.0.0"
BACKEND_VERSIONS = {"vmaf": "3.0.0", "ffmpeg": "7.1"}
CHALLENGE_ID = "chal-001"
ITEM_ID = "item-001"
MINER_HOTKEY = "hk-test"


def make_score_packet(metrics: dict[str, float], **overrides: Any) -> bytes:
    """Full ItemScore-shaped packet JSON; override any field via kwargs.

    The top-level score defaults to metrics["final_score"] (gates-first honest
    packet); pass e.g. score=0.99 to spoof an inconsistent packet, or
    omit={"scorer_version"} to drop required keys.
    """
    omit = overrides.pop("omit", frozenset())
    final = metrics.get("final_score", 0.0)
    packet: dict[str, Any] = {
        "item_id": ITEM_ID,
        "challenge_id": CHALLENGE_ID,
        "track": "compression",
        "score": final,
        "gate_passed": True,
        "violations": [],
        "skips": [],
        "miner_hotkey": MINER_HOTKEY,
        "content_digest": sha256_hex(b"miner restored video bytes"),
        "breakdown": {"kind": "compression", "final": final},
        "metrics": metrics,
        "scorer_version": SCORER_VERSION,
        "backend_versions": BACKEND_VERSIONS,
        "pieapp_start_frame": None,
        "scoring_config_digest": sha256_hex(b"scoring config"),
        "canonicalization_plan_digest": sha256_hex(b"canonicalization plan"),
    }
    packet.update(overrides)
    for key in omit:
        packet.pop(key, None)
    return canonical_json_bytes(packet)


def make_post_retirement_bundle(
    store: LocalFsStore,
    metrics: dict[str, float] | None = None,
    *,
    packet: bytes | None = None,
    **packet_overrides: Any,
) -> AuditBundle:
    """Store a full artifact set and return its post-retirement bundle."""
    committed_track = str(packet_overrides.get("track", "compression"))
    dag_bytes = canonical_json_bytes(
        {
            "asset_id": "asset-audit-helper",
            "dag_digest": sha256_hex(b"audit-helper-dag"),
            "dispatch_ordering_key": 0,
            "scorer_version": SCORER_VERSION,
            "seed": 424242,
            "track": committed_track,
        }
    )
    if packet is None:
        packet = make_score_packet(metrics or HONEST_METRICS, **packet_overrides)
    refs = {
        "challenge_input": store.put(b"degraded input video bytes", ArtifactKind.CHALLENGE_INPUT),
        "miner_output": store.put(b"miner restored video bytes", ArtifactKind.MINER_OUTPUT),
        "manifest": store.put(
            canonical_json_bytes({"scoring_seed": 7, "vmaf_threshold": 90}),
            ArtifactKind.MANIFEST,
        ),
        "score_packet": store.put(packet, ArtifactKind.SCORE_PACKET),
        "reference_original": store.put(
            b"pristine holdout original bytes", ArtifactKind.REFERENCE_ORIGINAL
        ),
        "dag_reveal": store.put(dag_bytes, ArtifactKind.DAG_REVEAL),
    }
    return build_bundle(
        challenge_id=CHALLENGE_ID,
        item_id=ITEM_ID,
        miner_hotkey=MINER_HOTKEY,
        commitment_hash=sha256_hex(dag_bytes),
        stage=LifecycleStage.POST_RETIREMENT,
        scorer_version=SCORER_VERSION,
        created_at="2026-08-20T12:00:00+00:00",
        backend_versions=BACKEND_VERSIONS,
        **refs,
    )
