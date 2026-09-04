import json
import operator
from pathlib import Path

import pytest

from audit_helpers import (
    BACKEND_VERSIONS,
    HONEST_METRICS,
    SCORER_VERSION,
    make_post_retirement_bundle,
    make_score_packet,
)
from vidaio.audit.bundle import LifecycleStage
from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.commitments import merkle_proof, merkle_root
from vidaio.audit.recompute import (
    ARTIFACT_CORRUPT,
    BACKEND_VERSION_MISMATCH,
    COMMITMENT_MISMATCH,
    DEFAULT_TOLERANCES,
    DIGEST_MISMATCH,
    IDENTITY_MISMATCH,
    INCOMPLETE_BUNDLE,
    MALFORMED_SCORE_PACKET,
    MERKLE_EXCLUSION,
    METRIC_SET_MISMATCH,
    MISSING_ANCHOR,
    PACKET_INCONSISTENT,
    REVEAL_INVALID,
    REVEAL_UNVERIFIED,
    SCORE_MISMATCH,
    SCORER_VERSION_MISMATCH,
    StaticRecomputer,
    verify_bundle,
)
from vidaio.audit.store import ArtifactKind, LocalFsStore
from vidaio.scoring import ScoringConfig
from tests.legacy_validator_zero import forged_validator_zero_packet

HONEST = StaticRecomputer(HONEST_METRICS, SCORER_VERSION)


def test_release_tolerance_ceilings_are_immutable() -> None:
    with pytest.raises(TypeError):
        operator.setitem(DEFAULT_TOLERANCES, "pieapp", 1.0)
    assert DEFAULT_TOLERANCES["pieapp"] == 1e-5


def _published(store: LocalFsStore, bundle, extra_digests=()):
    leaves = [bundle.score_packet.digest, *extra_digests]
    root = merkle_root(leaves)
    return root, merkle_proof(leaves, bundle.score_packet.digest)


def _verify_full(bundle, store, recomputer=HONEST, **kwargs):
    """Strict verification with every anchor supplied (the honest full path)."""
    root, proof = _published(store, bundle, [sha256_hex(b"another packet")])
    kwargs.setdefault("expected_bundle_digest", bundle.bundle_digest())
    kwargs.setdefault("published_root", root)
    kwargs.setdefault("inclusion_proof", proof)
    kwargs.setdefault("reveal_verifier", lambda dag_bytes: True)
    return verify_bundle(bundle, store, recomputer, **kwargs)


def test_honest_path_all_checks_pass(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store)
    report = _verify_full(bundle, store)  # strict default: every anchor supplied
    assert report.passed, report.failures()
    assert not report.skips()
    names = {c.name for c in report.checks}
    assert {
        "stage_recomputable",
        "bundle_digest",
        "commitment_reveal",
        "dag_reveal_generation",
        "packet_identity",
        "packet_consistency",
        "scorer_version",
        "committed_scorer_version",
        "scorer_version_recompute",
        "backend_versions_recompute",
        "merkle_inclusion",
        "metric_set",
        "score_recompute:vmaf",
        "score_recompute:score",
        "gate_recompute",
    } <= names


def test_vmaf_epsilon_tolerated_but_byte_ratio_exact(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store)
    drifted = dict(HONEST_METRICS)
    drifted["vmaf"] += 0.03  # inside the vmaf epsilon
    report = verify_bundle(
        bundle, store, StaticRecomputer(drifted, SCORER_VERSION), strict=False
    )
    assert report.passed, report.failures()

    off_ratio = dict(HONEST_METRICS)
    off_ratio["compression_rate"] += 1e-9  # byte-ratio must match exactly
    report = verify_bundle(
        bundle, store, StaticRecomputer(off_ratio, SCORER_VERSION), strict=False
    )
    assert [c.code for c in report.failures()] == [SCORE_MISMATCH]


def test_tolerance_override_may_narrow_but_not_widen_release_policy(
    store: LocalFsStore,
) -> None:
    bundle = make_post_retirement_bundle(store)
    drifted = dict(HONEST_METRICS)
    drifted["vmaf"] += 0.03

    narrowed = verify_bundle(
        bundle,
        store,
        StaticRecomputer(drifted, SCORER_VERSION),
        tolerances={"vmaf": 0.01},
        strict=False,
    )
    assert [check.code for check in narrowed.failures()] == [SCORE_MISMATCH]

    at_release_ceiling = verify_bundle(
        bundle,
        store,
        StaticRecomputer(drifted, SCORER_VERSION),
        tolerances={"vmaf": 0.05},
        strict=False,
    )
    assert at_release_ceiling.passed, at_release_ceiling.failures()


@pytest.mark.parametrize(
    "override",
    (
        {"pieapp": 1.00001e-5},
        {"pieapp": float("inf")},
        {"pieapp": float("nan")},
        {"pieapp": -1e-9},
        {"compression_rate": 1e-12},
    ),
)
def test_tolerance_override_rejects_any_local_acceptance_widening(
    store: LocalFsStore, override: dict[str, float]
) -> None:
    bundle = make_post_retirement_bundle(store)
    with pytest.raises(ValueError, match="may only narrow"):
        verify_bundle(bundle, store, HONEST, tolerances=override, strict=False)


def test_compression_threshold_flip_inside_vmaf_tolerance_uses_hysteresis(
    store: LocalFsStore,
) -> None:
    """A 90-point cross-build epsilon cannot dispute an otherwise honest epoch."""
    recorded = {"compression_rate": 0.3, "vmaf": 90.02, "final_score": 0.72}
    bundle = make_post_retirement_bundle(
        store,
        metrics=recorded,
        breakdown={
            "kind": "compression",
            "vmaf": 90.02,
            "vmaf_threshold": 90.0,
            "zero_reason": None,
            "final": 0.72,
        },
    )
    fresh = {"compression_rate": 0.3, "vmaf": 89.98, "final_score": 0.0}
    recomputer = StaticRecomputer(
        fresh,
        SCORER_VERSION,
        breakdown={
            "kind": "compression",
            "vmaf": 89.98,
            "vmaf_threshold": 90.0,
            "zero_reason": "VMAF_BELOW_THRESHOLD",
            "final": 0.0,
        },
    )
    report = verify_bundle(bundle, store, recomputer, strict=False)
    assert report.passed, report.failures()
    hysteretic = {c.name for c in report.checks if "numeric boundary" in c.reason}
    assert hysteretic == {"score_recompute:final_score", "score_recompute:score"}


def test_compression_threshold_flip_beyond_vmaf_tolerance_still_fails(
    store: LocalFsStore,
) -> None:
    recorded = {"compression_rate": 0.3, "vmaf": 90.02, "final_score": 0.72}
    bundle = make_post_retirement_bundle(
        store,
        metrics=recorded,
        breakdown={
            "kind": "compression",
            "vmaf": 90.02,
            "vmaf_threshold": 90.0,
            "zero_reason": None,
            "final": 0.72,
        },
    )
    fresh = {"compression_rate": 0.3, "vmaf": 89.96, "final_score": 0.0}
    recomputer = StaticRecomputer(
        fresh,
        SCORER_VERSION,
        breakdown={
            "kind": "compression",
            "vmaf": 89.96,
            "vmaf_threshold": 90.0,
            "zero_reason": "VMAF_BELOW_THRESHOLD",
            "final": 0.0,
        },
    )
    report = verify_bundle(bundle, store, recomputer, strict=False)
    failures = {c.name: c.code for c in report.failures()}
    assert failures["score_recompute:vmaf"] == SCORE_MISMATCH
    assert failures["score_recompute:final_score"] == SCORE_MISMATCH
    assert failures["score_recompute:score"] == SCORE_MISMATCH


def test_upscaling_vmaf_gate_flip_inside_tolerance_uses_hysteresis(
    store: LocalFsStore,
) -> None:
    recorded = {"vmaf": 50.02, "pieapp": 0.2, "final_score": 0.61}
    bundle = make_post_retirement_bundle(
        store,
        metrics=recorded,
        track="upscaling",
        breakdown={"kind": "upscaling", "final": 0.61},
    )
    fresh = {"vmaf": 49.98, "pieapp": 0.2, "final_score": 0.61}
    recomputer = StaticRecomputer(
        fresh,
        SCORER_VERSION,
        gate_passed=False,
        violations=[
            {
                "code": "VMAF_BELOW_FLOOR",
                "measured": 49.98,
                "limit": 50.0,
            }
        ],
        breakdown={"kind": "upscaling", "final": 0.61},
    )
    report = verify_bundle(bundle, store, recomputer, strict=False)
    assert report.passed, report.failures()
    gate = next(c for c in report.checks if c.name == "gate_recompute")
    assert "numeric boundary VMAF_BELOW_FLOOR" in gate.reason


def test_model_delta_gate_flip_inside_declared_tolerance_uses_hysteresis(
    store: LocalFsStore,
) -> None:
    recorded = dict(HONEST_METRICS, vmaf_model_delta=2.96)
    bundle = make_post_retirement_bundle(store, metrics=recorded)
    fresh = dict(HONEST_METRICS, vmaf_model_delta=3.04)
    recomputer = StaticRecomputer(
        fresh,
        SCORER_VERSION,
        gate_passed=False,
        violations=[
            {
                "code": "VMAF_MODEL_DELTA_EXCEEDED",
                "measured": 3.04,
                "limit": 3.0,
            }
        ],
    )
    report = verify_bundle(bundle, store, recomputer, strict=False)
    assert report.passed, report.failures()


def test_numeric_gate_flip_beyond_declared_tolerance_still_fails(
    store: LocalFsStore,
) -> None:
    recorded = dict(HONEST_METRICS, vmaf_model_delta=2.94)
    bundle = make_post_retirement_bundle(store, metrics=recorded)
    fresh = dict(HONEST_METRICS, vmaf_model_delta=3.06)
    recomputer = StaticRecomputer(
        fresh,
        SCORER_VERSION,
        gate_passed=False,
        violations=[
            {
                "code": "VMAF_MODEL_DELTA_EXCEEDED",
                "measured": 3.06,
                "limit": 3.0,
            }
        ],
    )
    report = verify_bundle(bundle, store, recomputer, strict=False)
    failures = {c.name: c.code for c in report.failures()}
    assert failures["score_recompute:vmaf_model_delta"] == SCORE_MISMATCH
    assert failures["score_recompute:score"] == SCORE_MISMATCH
    assert failures["gate_recompute"] == SCORE_MISMATCH


def test_tampered_score_packet_fails_score_mismatch(store: LocalFsStore) -> None:
    inflated = dict(HONEST_METRICS)
    inflated["final_score"] = 0.99  # adversary records a better score than earned
    bundle = make_post_retirement_bundle(store, metrics=inflated)
    report = verify_bundle(
        bundle, store, HONEST, strict=False
    )  # honest recompute disagrees
    assert not report.passed
    failures = report.failures()
    # both the tampered metric and the tampered top-level score are caught
    assert {c.code for c in failures} == {SCORE_MISMATCH}
    assert {c.name for c in failures} == {
        "score_recompute:final_score",
        "score_recompute:score",
    }


def test_codex_probe_top_level_score_tampered_metrics_honest(
    store: LocalFsStore,
) -> None:
    """review probe: metrics.final_score=0.5 (honest) but top-level score edited.

    The tampered value stays inside [0, 1] — a wilder tampered value (999, Infinity)
    is already killed at parse time as MALFORMED_SCORE_PACKET.
    """
    metrics = dict(HONEST_METRICS)
    metrics["final_score"] = 0.5
    bundle = make_post_retirement_bundle(store, metrics=metrics, score=0.99)
    honest = StaticRecomputer(metrics, SCORER_VERSION)  # agrees with the metrics block
    report = verify_bundle(bundle, store, honest, strict=False)
    assert not report.passed
    tampered = [c for c in report.failures() if c.name == "score_recompute:score"]
    assert tampered and tampered[0].code == SCORE_MISMATCH


def test_codex_probe_wrong_recomputer_version_fails(store: LocalFsStore) -> None:
    """review probe: a recomputer reporting the wrong scorer version must fail."""
    bundle = make_post_retirement_bundle(store)
    wrong_version = StaticRecomputer(HONEST_METRICS, "scoring-9.9.9")
    report = verify_bundle(bundle, store, wrong_version, strict=False)
    assert not report.passed
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("scorer_version_recompute") == SCORER_VERSION_MISMATCH


def test_packet_missing_scorer_version_is_malformed(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store, omit={"scorer_version"})
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert not report.passed
    assert MALFORMED_SCORE_PACKET in {c.code for c in report.failures()}


def test_packet_missing_required_identity_field_is_malformed(
    store: LocalFsStore,
) -> None:
    bundle = make_post_retirement_bundle(store, omit={"item_id", "gate_passed"})
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert MALFORMED_SCORE_PACKET in {c.code for c in report.failures()}


def test_packet_content_digest_must_bind_the_bundle_miner_output(
    store: LocalFsStore,
) -> None:
    bundle = make_post_retirement_bundle(store, content_digest="f" * 64)

    report = verify_bundle(bundle, store, HONEST, strict=False)

    identity = next(check for check in report.checks if check.name == "packet_identity")
    assert not identity.passed
    assert identity.code == IDENTITY_MISMATCH
    assert "content_digest" in identity.reason
    assert bundle.miner_output.digest in identity.reason


@pytest.mark.parametrize(
    "missing",
    [
        "track",
        "content_digest",
        "breakdown",
        "pieapp_start_frame",
        "miner_hotkey",
        "metrics",
        "backend_versions",
        "scoring_config_digest",
        "canonicalization_plan_digest",
        "violations",
        "score",
        "challenge_id",
    ],
)
def test_packet_missing_any_required_key_is_malformed(
    store: LocalFsStore, missing: str
) -> None:
    """Full ItemScore mirror: every required key (nullable or not) must be present."""
    bundle = make_post_retirement_bundle(store, omit={missing})
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert not report.passed
    assert MALFORMED_SCORE_PACKET in {c.code for c in report.failures()}


def test_packet_omitting_skips_still_parses(store: LocalFsStore) -> None:
    # skips defaults to empty: older packets without the key remain auditable
    bundle = make_post_retirement_bundle(store, omit={"skips"})
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert report.passed, report.failures()


def test_packet_with_skips_is_visible_and_verifiable(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(
        store, skips=[{"code": "SECONDARY_VMAF_DISABLED", "reason": "config opt-out"}]
    )
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert report.passed, report.failures()


def test_infinite_score_packet_is_malformed(store: LocalFsStore) -> None:
    """review new-issue probe: an Infinity top-level score dies at parse time."""
    payload = json.loads(make_score_packet(HONEST_METRICS))
    payload["score"] = float("inf")
    packet = json.dumps(payload, sort_keys=True).encode()  # emits bare Infinity
    bundle = make_post_retirement_bundle(store, packet=packet)
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert not report.passed
    assert MALFORMED_SCORE_PACKET in {c.code for c in report.failures()}


def test_nan_score_packet_is_malformed(store: LocalFsStore) -> None:
    payload = json.loads(make_score_packet(HONEST_METRICS))
    payload["score"] = float("nan")
    packet = json.dumps(payload, sort_keys=True).encode()
    bundle = make_post_retirement_bundle(store, packet=packet)
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert MALFORMED_SCORE_PACKET in {c.code for c in report.failures()}


@pytest.mark.parametrize("score", [999.0, 1.0000001, -0.01])
def test_out_of_range_score_packet_is_malformed(
    store: LocalFsStore, score: float
) -> None:
    bundle = make_post_retirement_bundle(store, score=score)
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert MALFORMED_SCORE_PACKET in {c.code for c in report.failures()}


def test_packet_extra_fields_are_tolerated(store: LocalFsStore) -> None:
    # loose on extras: scoring may grow the packet without breaking verifiers
    bundle = make_post_retirement_bundle(store, breakdown={"kind": "compression"})
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert report.passed, report.failures()


def test_packet_scorer_version_must_match_bundle(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store, scorer_version="scoring-0.0.1")
    report = verify_bundle(bundle, store, HONEST, strict=False)
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("scorer_version") == SCORER_VERSION_MISMATCH


def _with_dag_reveal(bundle, store: LocalFsStore, reveal: bytes):
    """Replace the committed preimage while keeping its ref/hash self-consistent."""
    return bundle.model_copy(
        update={
            "dag_reveal": store.put(reveal, ArtifactKind.DAG_REVEAL),
            "commitment_hash": sha256_hex(reveal),
        }
    )


@pytest.mark.parametrize("track", ["compression", "upscaling"])
def test_committed_scorer_identity_accepts_honest_measured_tracks(
    store: LocalFsStore, track: str
) -> None:
    bundle = make_post_retirement_bundle(store, track=track)
    report = verify_bundle(bundle, store, HONEST, strict=False)

    binding = next(c for c in report.checks if c.name == "committed_scorer_version")
    assert binding.passed, binding.reason
    assert report.passed, report.failures()


@pytest.mark.parametrize("track", ["compression", "upscaling"])
def test_committed_scorer_identity_rejects_post_commit_scorer_substitution(
    store: LocalFsStore, track: str
) -> None:
    """Packet, bundle and recomputer agreeing cannot override the pre-dispatch scorer."""
    bundle = make_post_retirement_bundle(store, track=track)
    reveal = json.loads(store.get(bundle.dag_reveal))
    reveal["scorer_version"] = "scoring-evil/9.9.9"
    tampered = _with_dag_reveal(bundle, store, canonical_json_bytes(reveal))

    report = verify_bundle(
        tampered, store, HONEST, reveal_verifier=lambda _raw: True, strict=False
    )

    failures = {c.name: c.code for c in report.failures()}
    assert failures["committed_scorer_version"] == SCORER_VERSION_MISMATCH
    assert "commitment_reveal" not in failures
    assert "dag_reveal_generation" not in failures


@pytest.mark.parametrize(
    "reveal",
    [
        b'{"scorer_version":"scoring-1.0.0","track":"compression"',
        b'{ "scorer_version": "scoring-1.0.0", "track": "compression" }',
    ],
    ids=["malformed", "noncanonical"],
)
def test_committed_scorer_identity_rejects_invalid_preimage_as_typed_finding(
    store: LocalFsStore, reveal: bytes
) -> None:
    bundle = _with_dag_reveal(make_post_retirement_bundle(store), store, reveal)

    report = verify_bundle(
        bundle, store, HONEST, reveal_verifier=lambda _raw: True, strict=False
    )

    binding = next(c for c in report.failures() if c.name == "committed_scorer_version")
    assert binding.code == REVEAL_INVALID


def _validator_zero_bundle(
    store: LocalFsStore, *, committed_scorer: str = SCORER_VERSION
):
    config = ScoringConfig()
    packet = forged_validator_zero_packet(
        item_id="item-001",
        challenge_id="chal-001",
        track="upscaling",
        miner_hotkey="hk-test",
        committed_scorer_version=committed_scorer,
        failure_reason="timeout",
        config=config,
    )
    bundle = make_post_retirement_bundle(
        store, packet=packet.to_json().encode("utf-8"), track="upscaling"
    ).model_copy(
        update={"scorer_version": packet.scorer_version, "backend_versions": {}}
    )
    recomputer = StaticRecomputer(
        {}, packet.scorer_version, score=0.0, gate_passed=False
    )
    return bundle, recomputer


def test_forged_validator_zero_bundle_cannot_pass_recompute(
    store: LocalFsStore,
) -> None:
    bundle, recomputer = _validator_zero_bundle(store)

    report = verify_bundle(bundle, store, recomputer, strict=False)

    assert not report.passed
    failures = {c.name: c.code for c in report.failures()}
    assert failures["committed_scorer_version"] == SCORER_VERSION_MISMATCH


def test_validator_zero_rejects_different_committed_worker(
    store: LocalFsStore,
) -> None:
    bundle, recomputer = _validator_zero_bundle(store)
    reveal = json.loads(store.get(bundle.dag_reveal))
    reveal["scorer_version"] = "different-worker/1+deadbeefcafe"
    tampered = _with_dag_reveal(bundle, store, canonical_json_bytes(reveal))

    report = verify_bundle(
        tampered, store, recomputer, reveal_verifier=lambda _raw: True, strict=False
    )

    failures = {c.name: c.code for c in report.failures()}
    assert failures["committed_scorer_version"] == SCORER_VERSION_MISMATCH


def test_packet_for_other_challenge_fails_identity(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store, challenge_id="chal-OTHER")
    report = verify_bundle(bundle, store, HONEST, strict=False)
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("packet_identity") == IDENTITY_MISMATCH


def test_packet_for_other_item_fails_identity(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store, item_id="item-OTHER")
    report = verify_bundle(bundle, store, HONEST, strict=False)
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("packet_identity") == IDENTITY_MISMATCH


def test_packet_for_other_miner_fails_identity(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store, miner_hotkey="hk-EVIL")
    report = verify_bundle(bundle, store, HONEST, strict=False)
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("packet_identity") == IDENTITY_MISMATCH


def test_unattributed_bundle_skips_miner_identity(store: LocalFsStore) -> None:
    # bundle without a pinned miner (e.g. calibration) does not compare hotkeys
    bundle = make_post_retirement_bundle(store).model_copy(
        update={"miner_hotkey": None}
    )
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert report.passed, report.failures()


def test_packet_backend_version_conflict_fails(store: LocalFsStore) -> None:
    tampered = dict(BACKEND_VERSIONS)
    tampered["vmaf"] = "2.9.9"  # packet claims a different backend than pinned
    bundle = make_post_retirement_bundle(store, backend_versions=tampered)
    report = verify_bundle(bundle, store, HONEST, strict=False)
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("backend_versions") == BACKEND_VERSION_MISMATCH


def test_packet_missing_pinned_backend_fails(store: LocalFsStore) -> None:
    # the bundle pins ffmpeg; a packet silent about it cannot claim parity
    bundle = make_post_retirement_bundle(store, backend_versions={"vmaf": "3.0.0"})
    report = verify_bundle(bundle, store, HONEST, strict=False)
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("backend_versions") == BACKEND_VERSION_MISMATCH


def test_packet_extra_unpinned_backend_fails(store: LocalFsStore) -> None:
    extra = dict(BACKEND_VERSIONS)
    extra["pieapp"] = "1.0"
    bundle = make_post_retirement_bundle(store, backend_versions=extra)
    report = verify_bundle(bundle, store, HONEST, strict=False)
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("backend_versions") == BACKEND_VERSION_MISMATCH


def test_recomputer_backend_version_conflict_fails(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store)
    recomputed = dict(BACKEND_VERSIONS)
    recomputed["ffmpeg"] = "ffmpeg/DIFFERENT"
    report = verify_bundle(
        bundle,
        store,
        StaticRecomputer(
            HONEST_METRICS,
            SCORER_VERSION,
            backend_versions=recomputed,
        ),
        strict=False,
    )
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("backend_versions_recompute") == BACKEND_VERSION_MISMATCH


def test_gate_passed_packet_without_breakdown_inconsistent(store: LocalFsStore) -> None:
    # breakdown is nullable ONLY when the gate failed
    bundle = make_post_retirement_bundle(store, breakdown=None)
    report = verify_bundle(bundle, store, HONEST, strict=False)
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("packet_consistency") == PACKET_INCONSISTENT


def test_gate_failed_packet_may_omit_breakdown(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(
        store,
        gate_passed=False,
        score=0.0,
        violations=[{"code": "VMAF_BELOW_THRESHOLD"}],
        breakdown=None,
    )
    gated = StaticRecomputer(HONEST_METRICS, SCORER_VERSION, gate_passed=False)
    report = verify_bundle(bundle, store, gated, strict=False)
    consistency = [c for c in report.checks if c.name == "packet_consistency"]
    assert consistency and consistency[0].passed


def test_gate_failed_packet_with_nonzero_score_inconsistent(
    store: LocalFsStore,
) -> None:
    bundle = make_post_retirement_bundle(
        store,
        gate_passed=False,
        score=0.35,
        violations=[{"code": "VMAF_BELOW_THRESHOLD"}],
    )
    report = verify_bundle(bundle, store, HONEST, strict=False)
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("packet_consistency") == PACKET_INCONSISTENT


def test_gate_passed_packet_with_violations_inconsistent(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(
        store, gate_passed=True, violations=[{"code": "VMAF_BELOW_THRESHOLD"}]
    )
    report = verify_bundle(bundle, store, HONEST, strict=False)
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("packet_consistency") == PACKET_INCONSISTENT


def test_recomputed_gate_failure_catches_doctored_gate_flag(
    store: LocalFsStore,
) -> None:
    # packet claims a clean pass, but independent recompute says the gate failed
    bundle = make_post_retirement_bundle(store)
    gate_failed = StaticRecomputer(HONEST_METRICS, SCORER_VERSION, gate_passed=False)
    report = verify_bundle(bundle, store, gate_failed, strict=False)
    assert not report.passed
    failures = {c.name: c.code for c in report.failures()}
    assert failures.get("gate_recompute") == SCORE_MISMATCH
    assert failures.get("score_recompute:score") == SCORE_MISMATCH  # 0.81 vs gated 0.0


def test_strict_default_treats_missing_anchors_as_failures(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store)
    report = verify_bundle(bundle, store, HONEST)  # strict=True default, no anchors
    assert not report.passed
    assert report.strict
    skipped = {c.name: c.code for c in report.skips()}
    assert skipped == {
        "bundle_digest": MISSING_ANCHOR,
        "merkle_inclusion": MISSING_ANCHOR,
        "dag_reveal_generation": REVEAL_UNVERIFIED,
    }
    assert all(not c.passed for c in report.skips())


def test_non_strict_records_skips_as_passing(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store)
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert report.passed, report.failures()
    assert {c.name for c in report.skips()} == {
        "bundle_digest",
        "merkle_inclusion",
        "dag_reveal_generation",
    }
    assert all(c.passed and c.skipped for c in report.skips())


def test_reveal_verifier_rejection_fails(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store)
    report = _verify_full(bundle, store, reveal_verifier=lambda dag_bytes: False)
    assert not report.passed
    failures = {c.name: c.code for c in report.failures()}
    assert failures == {"dag_reveal_generation": REVEAL_INVALID}


def test_reveal_verifier_crash_is_a_finding(store: LocalFsStore) -> None:
    def boom(dag_bytes: bytes) -> bool:
        raise RuntimeError("cannot rebuild DAG")

    bundle = make_post_retirement_bundle(store)
    report = _verify_full(bundle, store, reveal_verifier=boom)
    failures = {c.name: c.code for c in report.failures()}
    assert failures == {"dag_reveal_generation": REVEAL_INVALID}


def test_reveal_verifier_receives_dag_bytes(store: LocalFsStore) -> None:
    seen: list[bytes] = []

    def capture(dag_bytes: bytes) -> bool:
        seen.append(dag_bytes)
        return True

    bundle = make_post_retirement_bundle(store)
    report = _verify_full(bundle, store, reveal_verifier=capture)
    assert report.passed, report.failures()
    assert seen == [store.get(bundle.dag_reveal)]


def test_injected_extra_metric_fails(store: LocalFsStore) -> None:
    padded = dict(HONEST_METRICS)
    padded["bonus_metric"] = 1.0  # substituted metric the scorer never produces
    bundle = make_post_retirement_bundle(store, metrics=padded)
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert METRIC_SET_MISMATCH in {c.code for c in report.failures()}


def test_non_numeric_metric_value_cannot_hide_a_metric(store: LocalFsStore) -> None:
    hidden = dict(HONEST_METRICS)
    hidden["compression_rate"] = "not-a-number"  # dodge the value comparison
    bundle = make_post_retirement_bundle(store, metrics=hidden)
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert METRIC_SET_MISMATCH in {c.code for c in report.failures()}


def test_tampered_bundle_metadata_fails_digest_check(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store)
    anchored_digest = bundle.bundle_digest()
    tampered = bundle.model_copy(update={"created_at": "2020-01-01T00:00:00+00:00"})
    report = verify_bundle(
        tampered, store, HONEST, expected_bundle_digest=anchored_digest, strict=False
    )
    assert not report.passed
    assert DIGEST_MISMATCH in {c.code for c in report.failures()}


def test_corrupted_artifact_fails_integrity(
    store: LocalFsStore, tmp_path: Path
) -> None:
    bundle = make_post_retirement_bundle(store)
    path = tmp_path / "audit" / bundle.miner_output.backend_key
    raw = bytearray(path.read_bytes())
    raw[0] ^= 0xFF
    path.write_bytes(bytes(raw))
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert not report.passed
    corrupt = [c for c in report.failures() if c.code == ARTIFACT_CORRUPT]
    assert corrupt and corrupt[0].name == "artifact_integrity:miner_output"


def test_commitment_mismatch_detected(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store)
    # adversary swaps in a different commitment hash (pretends another DAG was committed)
    tampered = bundle.model_copy(
        update={"commitment_hash": sha256_hex(b"different dag")}
    )
    report = verify_bundle(tampered, store, HONEST, strict=False)
    assert COMMITMENT_MISMATCH in {c.code for c in report.failures()}


def test_packet_outside_published_set_fails_merkle(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store)
    # published root covers OTHER packets only — this one was injected afterwards
    others = [sha256_hex(b"legit packet 1"), sha256_hex(b"legit packet 2")]
    root = merkle_root(others)
    proof = merkle_proof(others, others[0])
    report = verify_bundle(
        bundle, store, HONEST, published_root=root, inclusion_proof=proof, strict=False
    )
    assert MERKLE_EXCLUSION in {c.code for c in report.failures()}


def test_pre_reveal_bundle_not_fully_verifiable(store: LocalFsStore) -> None:
    from vidaio.audit.bundle import build_bundle

    bundle = build_bundle(
        challenge_id="chal-001",
        item_id="item-001",
        miner_hotkey="hk-test",
        commitment_hash=sha256_hex(b"dag"),
        stage=LifecycleStage.PRE_REVEAL,
        challenge_input=store.put(b"in", ArtifactKind.CHALLENGE_INPUT),
        miner_output=store.put(b"out", ArtifactKind.MINER_OUTPUT),
        manifest=store.put(b"{}", ArtifactKind.MANIFEST),
        score_packet=store.put(
            make_score_packet(HONEST_METRICS), ArtifactKind.SCORE_PACKET
        ),
        scorer_version=SCORER_VERSION,
        created_at="2026-08-20T12:00:00+00:00",
    )
    report = verify_bundle(bundle, store, HONEST, strict=False)
    assert not report.passed
    assert INCOMPLETE_BUNDLE in {c.code for c in report.failures()}
