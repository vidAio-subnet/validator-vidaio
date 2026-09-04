"""Upscaling competition economics stay bound to publicly auditable CPU inputs."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vidaio.auditor import (
    Auditor,
    AuditorConfig,
    AuditStatus,
    InMemoryBundleSource,
    ItemVerdictKind,
    SamplePolicy,
)
from vidaio.audit import (
    ArtifactKind,
    AuditConfig,
    CompetitionCommitment,
    CompetitionItemBinding,
    LifecycleStage,
    StaticRecomputer,
    build_bundle,
    build_competition_commitment,
    canonical_json_bytes,
    make_public_store,
    make_store,
    merkle_proof,
    merkle_root,
    pin_git_sha,
    reward_parameter_digest,
    sha256_hex,
    verify_bundle,
)
from vidaio.audit.recompute import CompetitionAuditContext
from vidaio.authority import EpochFinalizer, build_audit_manifest
from vidaio.competition import (
    CompetitionManifest,
    LifecycleEngine,
    evaluation_item_commitment,
    migrate,
)
from vidaio.competition import repository as repo
from vidaio.competition.epoch_evidence import build_competition_epoch_evidence
from vidaio.competition.orchestrator.persistence import record_submission_archived
from vidaio.competition.states import Phase
from vidaio.core import connect
from vidaio.chain.adapter import ChainNeuron, InMemoryChain
from vidaio.epoch import MinerCensusEntry
from vidaio.scoring import ScoringConfig, compose_item_score, score_upscaling
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import MinerSnapshot


T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
COMPETITION_ID = "comp-upscale-economics"
CHALLENGE_ID = "chal-upscale-economics"
THRESHOLD_COMMITMENT = "f" * 64
SCORER_VERSION = "cpu-pieapp-test-v1"
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def _manifest(
    reference_digest: str,
    input_digest: str,
    *,
    baseline_artifact_digest: str,
    baseline_artifact_bytes: int,
    baseline_provenance_digest: str,
    baseline_provenance_bytes: int,
) -> CompetitionManifest:
    commitment = evaluation_item_commitment(
        competition_id=COMPETITION_ID,
        item_index=0,
        reference_sha256=reference_digest,
        input_sha256=input_digest,
        upscale_factor=2,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
    )
    return CompetitionManifest.model_validate(
        {
            "competition_id": COMPETITION_ID,
            "track": "upscaling",
            "start_time": T0 + timedelta(hours=1),
            "enrollment_deadline": T0 + timedelta(hours=2),
            "finalization_time": T0 + timedelta(hours=3),
            "end_time": T0 + timedelta(hours=4),
            "minimum_alpha_stake": 1.0,
            "scoring_factors": {
                "quality": 0.6,
                "cost_efficiency": 0.0,
                "length_coverage": 0.4,
            },
            "vmaf_threshold": 90.0,
            "sealed_vmaf_variants": [85.0, 89.0, 93.0],
            "allowed_gpus": ["L4"],
            "allowed_upscale_factors": [2],
            "evaluation_item_commitments": [commitment],
            "evaluation_batch_size": {"min": 1, "max": 1},
            "scoring_seed_commitment": "a" * 64,
            "container_size_limit_gb": 25.0,
            "scoring_version": SCORER_VERSION,
            "baseline": {
                "version": 0,
                "artifact_digest": baseline_artifact_digest,
                "artifact_bytes": baseline_artifact_bytes,
                "image_digest": "2" * 64,
                "provenance_digest": baseline_provenance_digest,
                "provenance_bytes": baseline_provenance_bytes,
                "repo_url": "https://example.invalid/vidaio/baseline",
                "commit_sha": "b" * 40,
                "tree_sha": "c" * 40,
            },
        }
    )


def _packet(
    *,
    item_id: str,
    hotkey: str | None,
    output_digest: str,
    pieapp: float,
) -> tuple[bytes, dict[str, float], dict[str, float | str]]:
    config = ScoringConfig()
    breakdown = score_upscaling(
        pieapp=pieapp,
        content_length=60.0,
        config=config,
    )
    metrics = {
        "pieapp": breakdown.pieapp,
        "content_length": breakdown.content_length,
        "final_score": breakdown.final,
    }
    packet = compose_item_score(
        item_id=item_id,
        challenge_id=CHALLENGE_ID,
        track="upscaling",
        gate_passed=True,
        violations=[],
        breakdown=breakdown,
        config=config,
        miner_hotkey=hotkey,
        content_digest=output_digest,
        metrics=metrics,
        backend_versions={"pieapp": "cpu-test-v1"},
        scorer_version=SCORER_VERSION,
    )
    return (
        packet.to_json().encode("utf-8"),
        metrics,
        breakdown.model_dump(mode="json"),
    )


def test_upscaling_epoch_economics_use_the_exact_public_cpu_audit_matrix(
    tmp_path: Path,
) -> None:
    tokenomics = TokenomicsConfig(competition_emissions_enabled=True)
    reference = b"pristine-high-resolution-reference"
    miner_input = b"low-resolution-miner-input"
    baseline_output = b"baseline-upscaled-output"
    contender_output = b"better-contender-upscaled-output"
    store = make_store(
        AuditConfig(
            backend="local",
            local_root=tmp_path / "audit",
            allow_plaintext_holdout=True,
        )
    )
    input_ref = store.put(miner_input, ArtifactKind.CHALLENGE_INPUT)
    reference_ref = store.put(reference, ArtifactKind.REFERENCE_ORIGINAL)
    baseline_output_ref = store.put(baseline_output, ArtifactKind.MINER_OUTPUT)
    contender_output_ref = store.put(contender_output, ArtifactKind.MINER_OUTPUT)
    baseline_archive_ref = store.put(
        b"sealed-baseline-source-archive", ArtifactKind.SUBMISSION_ARCHIVE
    )
    baseline_provenance_ref = store.put(
        b'{"schema":"vidaio-baseline-provenance/1"}', ArtifactKind.MANIFEST
    )
    contender_archive_ref = store.put(
        b"sealed-contender-source-archive", ArtifactKind.SUBMISSION_ARCHIVE
    )
    assert input_ref.digest != reference_ref.digest

    manifest = _manifest(
        reference_ref.digest,
        input_ref.digest,
        baseline_artifact_digest=baseline_archive_ref.digest,
        baseline_artifact_bytes=baseline_archive_ref.byte_size,
        baseline_provenance_digest=baseline_provenance_ref.digest,
        baseline_provenance_bytes=baseline_provenance_ref.byte_size,
    )
    manifest_ref = store.put(
        manifest.canonical_json().encode("utf-8"), ArtifactKind.MANIFEST
    )
    binding = CompetitionItemBinding(
        item_index=0,
        input_sha256=input_ref.digest,
        reference_sha256=reference_ref.digest,
        upscale_factor=2,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        item_commitment=(manifest.evaluation_item_commitments or [])[0],
    )

    conn = connect(":memory:")
    migrate(conn)
    repo.insert_competition(conn, manifest, T0)
    assert manifest.baseline is not None
    commitment = build_competition_commitment(
        CompetitionCommitment(
            manifest_digest=manifest.manifest_digest(),
            baseline_version=manifest.baseline.version,
            baseline_artifact_digest=manifest.baseline.artifact_digest,
            baseline_provenance_digest=manifest.baseline.provenance_digest,
            baseline_tree_digest=pin_git_sha(manifest.baseline.tree_sha),
            baseline_image_digest="2" * 64,
            dataset_selection_seed_commitment=manifest.scoring_seed_commitment,
            reward_param_digest=reward_parameter_digest(tokenomics),
        )
    )
    commitment_ref = store.put(commitment.canonical_json, ArtifactKind.MANIFEST)
    assert commitment_ref.digest == commitment.root
    anchor_at = T0 + timedelta(minutes=1)
    anchor_block = 10
    anchor_chain = InMemoryChain(
        _neurons=[
            ChainNeuron(
                uid=42,
                hotkey="hk-upscale-winner",
                coldkey="ck-upscale-winner",
                ip="203.0.113.42",
                alpha_stake=10.0,
                emission=0.0,
            )
        ],
        _block=10_000,
        anchored=[commitment.payload],
        _anchor_blocks=[anchor_block],
        block_time_anchor=(anchor_block, anchor_at),
    )
    anchor_hash = anchor_chain.block_hash(anchor_block)
    assert anchor_hash is not None
    engine = LifecycleEngine()
    engine.mark_commitment_anchored(
        conn,
        COMPETITION_ID,
        commitment.root,
        anchor_at,
        onchain_evidence={
            "root": commitment.root,
            "anchor_netuid": 85,
            "payload_hex": commitment.payload.hex(),
            "payload_digest": sha256_hex(commitment.payload),
            "anchor_block": anchor_block,
            "anchor_block_hash": anchor_hash,
            "finalized_block": anchor_block,
            "archive_verified": True,
        },
    )
    engine.tick(conn, manifest.start_time)
    baseline_id = repo.insert_calibration_contender(
        conn, COMPETITION_ID, manifest.baseline, T0
    )
    contender_id = repo.enroll_contender(
        conn,
        COMPETITION_ID,
        hotkey="hk-upscale-winner",
        repo_url="https://example.invalid/vidaio/contender",
        commit_sha="e" * 40,
        tree_sha="1" * 40,
        stake=10.0,
        now=T0,
    )
    repo.set_contender_image_digest(conn, baseline_id, "2" * 64, T0)
    repo.set_contender_image_digest(conn, contender_id, "3" * 64, T0)
    record_submission_archived(
        conn,
        COMPETITION_ID,
        baseline_id,
        baseline_archive_ref.digest,
        baseline_archive_ref.byte_size,
        T0,
    )
    record_submission_archived(
        conn,
        COMPETITION_ID,
        contender_id,
        contender_archive_ref.digest,
        contender_archive_ref.byte_size,
        T0,
    )
    item_id = repo.add_evaluation_item(
        conn,
        COMPETITION_ID,
        item_index=0,
        input_sha256=input_ref.digest,
        input_bytes=input_ref.byte_size,
        reference_sha256=reference_ref.digest,
        reference_bytes=reference_ref.byte_size,
        upscale_factor=2,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        threshold_commitment=THRESHOLD_COMMITMENT,
        challenge_id=CHALLENGE_ID,
        now=T0,
    )

    subjects = (
        (baseline_id, None, baseline_output_ref, 4.0),
        (contender_id, "hk-upscale-winner", contender_output_ref, 0.05),
    )
    bundles = []
    recomputers = []
    performance_ids = []
    for subject_id, hotkey, output_ref, pieapp in subjects:
        packet_bytes, metrics, breakdown = _packet(
            item_id=input_ref.digest,
            hotkey=hotkey,
            output_digest=output_ref.digest,
            pieapp=pieapp,
        )
        packet_ref = store.put(packet_bytes, ArtifactKind.SCORE_PACKET)
        bundle = build_bundle(
            challenge_id=CHALLENGE_ID,
            item_id=input_ref.digest,
            miner_hotkey=hotkey,
            commitment_hash=THRESHOLD_COMMITMENT,
            stage=LifecycleStage.COMPETITION_SEALED,
            challenge_input=input_ref,
            reference_original=reference_ref,
            miner_output=output_ref,
            manifest=manifest_ref,
            score_packet=packet_ref,
            competition_item=binding,
            execution_image_digest=("2" * 64 if hotkey is None else "3" * 64),
            scorer_version=SCORER_VERSION,
            backend_versions={"pieapp": "cpu-test-v1"},
            created_at=(T0 + timedelta(hours=5)).isoformat(),
        )
        bundle_ref = store.put(
            canonical_json_bytes(bundle.model_dump(mode="json")),
            ArtifactKind.AUDIT_BUNDLE,
        )
        assert bundle_ref.digest == bundle.bundle_digest()
        performance_id = repo.record_item_score(
            conn,
            COMPETITION_ID,
            contender_id=subject_id,
            item_id=item_id,
            packet_bytes=packet_bytes,
            output_bytes=output_ref.byte_size,
            now=T0 + timedelta(hours=5),
        )
        repo.set_audit_bundle_digest(conn, performance_id, bundle_ref.digest)
        bundles.append(bundle)
        performance_ids.append(performance_id)
        recomputers.append(
            StaticRecomputer(
                metrics,
                SCORER_VERSION,
                score=float(metrics["final_score"]),
                gate_passed=True,
                breakdown=breakdown,
            )
        )

    completed_at = T0 + timedelta(hours=6)
    repo.set_status(conn, COMPETITION_ID, Phase.COMPLETED, completed_at)
    repo.record_event(
        conn,
        COMPETITION_ID,
        "phase_transition",
        completed_at,
        from_phase=Phase.AWAITING_END_TIME,
        to_phase=Phase.COMPLETED,
    )
    store.release(reference_ref)

    evidence = build_competition_epoch_evidence(
        conn,
        competition_id=COMPETITION_ID,
        census_by_hotkey={
            "hk-upscale-winner": MinerCensusEntry(
                uid=42,
                hotkey="hk-upscale-winner",
                coldkey="ck-upscale-winner",
                ip="203.0.113.42",
            )
        },
        store=store,
        tokenomics=tokenomics,
        through_time=completed_at,
    )
    assert evidence is not None
    assert evidence.competition_input.track == "upscaling"
    assert evidence.competition_input.items[0].model_dump() == {
        "challenge_id": CHALLENGE_ID,
        "item_id": input_ref.digest,
        "threshold_commitment": THRESHOLD_COMMITMENT,
        "item_index": 0,
        "input_sha256": input_ref.digest,
        "reference_sha256": reference_ref.digest,
        "upscale_factor": 2,
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "item_commitment": binding.item_commitment,
    }
    assert len(evidence.scored_items) == 2
    assert evidence.result.contenders[0].uid == 42
    assert evidence.result.baseline_score is not None
    assert (
        evidence.result.contenders[0].score - evidence.result.baseline_score
    ) / evidence.result.baseline_score > 0.0

    # Every packet that produced the economic result resolves through the keyless
    # public role and independently reproduces under the same CPU recompute contract.
    public = make_public_store(
        AuditConfig(backend="local", local_root=tmp_path / "audit")
    )
    leaves = [bundle.score_packet.digest for bundle in bundles]
    context = CompetitionAuditContext(
        competition_id=COMPETITION_ID,
        track="upscaling",
        manifest_digest=manifest.manifest_digest(),
        threshold_commitment=THRESHOLD_COMMITMENT,
        item_index=0,
        input_sha256=input_ref.digest,
        reference_sha256=reference_ref.digest,
        upscale_factor=2,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        item_commitment=binding.item_commitment,
    )
    for bundle, recomputer in zip(bundles, recomputers):
        report = verify_bundle(
            bundle,
            public,
            recomputer,
            expected_bundle_digest=bundle.bundle_digest(),
            expected_miner_hotkey=bundle.miner_hotkey,
            require_expected_miner=bundle.miner_hotkey is not None,
            published_root=merkle_root(leaves),
            inclusion_proof=merkle_proof(leaves, bundle.score_packet.digest),
            strict=True,
            competition_context=context,
        )
        assert report.passed, [
            (failure.name, failure.code, failure.reason)
            for failure in report.failures()
        ]

    # Exercise the auditor's epoch-item -> CompetitionAuditContext adapter, not only
    # verify_bundle's direct context seam.  Geometry is part of the v2 commitment
    # preimage; dropping either dimension here makes every otherwise-valid upscaling
    # competition sample fail as COMPETITION_MANIFEST_INVALID.
    audit_manifest = build_audit_manifest(
        evidence.scored_items,
        store=store,
        competition_input=evidence.competition_input,
    )
    snapshots = (
        MinerSnapshot(
            uid=42,
            hotkey="hk-upscale-winner",
            coldkey="ck-upscale-winner",
            ip="203.0.113.42",
            track="upscaling",
            accumulate_score=0.0,
        ),
    )
    epoch_log = EpochFinalizer(tokenomics, scorer_version=SCORER_VERSION).build_log(
        epoch_id=1,
        close_block=359,
        snapshots=snapshots,
        burn_uid=94,
        audit_manifest=audit_manifest,
        now=completed_at,
        competition_result=evidence.result,
        competition_packet_scores=evidence.packet_scores,
    )
    bundle_source = InMemoryBundleSource()
    recomputer_by_bundle = {}
    for bundle, recomputer in zip(bundles, recomputers, strict=True):
        bundle_source.add(bundle)
        recomputer_by_bundle[bundle.bundle_digest()] = recomputer

    class _BundleRecomputer:
        def recompute(self, bundle, artifacts):
            return recomputer_by_bundle[bundle.bundle_digest()].recompute(
                bundle, artifacts
            )

    epoch_audit = Auditor(
        AuditorConfig(
            auditor_hotkey="auditor-full-upscaling-sample",
            tokenomics=tokenomics,
            burn_uid=94,
        ),
        bundle_source,
        chain=anchor_chain,
    ).audit_epoch(
        epoch_log,
        store,
        SamplePolicy(sample_rate=1.0),
        _BundleRecomputer(),
        completed_at,
    )
    competition_verdicts = [
        verdict
        for verdict in epoch_audit.item_verdicts
        if verdict.source == "competition"
    ]
    assert len(competition_verdicts) == len(bundles)
    assert all(
        verdict.verdict is ItemVerdictKind.PASS for verdict in competition_verdicts
    ), [(verdict.code, verdict.detail) for verdict in competition_verdicts]

    # The cheap competition-economics pass is exhaustive even when this auditor's
    # media sample is empty.  Substitute a fully self-consistent item preimage and
    # matching bundles while retaining the original anchored manifest: this must be
    # rejected before its unchanged score packets can enter payout arithmetic.
    substituted_input_ref = store.put(
        b"authority-substituted-low-resolution-input",
        ArtifactKind.CHALLENGE_INPUT,
    )
    substituted_reference_ref = store.put(
        b"authority-substituted-pristine-reference",
        ArtifactKind.REFERENCE_ORIGINAL,
    )
    substituted_commitment = evaluation_item_commitment(
        competition_id=COMPETITION_ID,
        item_index=0,
        reference_sha256=substituted_reference_ref.digest,
        input_sha256=substituted_input_ref.digest,
        upscale_factor=2,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
    )
    substituted_binding = CompetitionItemBinding(
        item_index=0,
        input_sha256=substituted_input_ref.digest,
        reference_sha256=substituted_reference_ref.digest,
        upscale_factor=2,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        item_commitment=substituted_commitment,
    )
    substituted_bundles = [
        type(bundle).model_validate(
            bundle.model_dump(mode="python")
            | {
                "challenge_input": substituted_input_ref,
                "reference_original": substituted_reference_ref,
                "competition_item": substituted_binding,
            }
        )
        for bundle in bundles
    ]
    substituted_bundle_digests: dict[str, str] = {}
    substituted_scored_items = []
    for scored, bundle in zip(
        evidence.scored_items, substituted_bundles, strict=True
    ):
        bundle_ref = store.put(
            canonical_json_bytes(bundle.model_dump(mode="json")),
            ArtifactKind.AUDIT_BUNDLE,
        )
        assert bundle_ref.digest == bundle.bundle_digest()
        assert scored.competition_subject is not None
        substituted_bundle_digests[scored.competition_subject] = bundle_ref.digest
        substituted_scored_items.append(
            replace(scored, bundle_digest=bundle_ref.digest)
        )
    substituted_item = evidence.competition_input.items[0].model_copy(
        update={
            "input_sha256": substituted_input_ref.digest,
            "reference_sha256": substituted_reference_ref.digest,
            "item_commitment": substituted_commitment,
        }
    )
    substituted_subjects = tuple(
        subject.model_copy(
            update={
                "audit_bundle_digests": (
                    substituted_bundle_digests[subject.subject_id],
                )
            }
        )
        for subject in evidence.competition_input.subjects
    )
    substituted_input = evidence.competition_input.model_copy(
        update={
            "items": (substituted_item,),
            "subjects": substituted_subjects,
        }
    )
    substituted_manifest = build_audit_manifest(
        substituted_scored_items,
        store=store,
        competition_input=substituted_input,
    )
    snapshots = (
        MinerSnapshot(
            uid=42,
            hotkey="hk-upscale-winner",
            coldkey="ck-upscale-winner",
            ip="203.0.113.42",
            track="upscaling",
            accumulate_score=0.0,
        ),
    )
    forged_log = EpochFinalizer(
        tokenomics, scorer_version=SCORER_VERSION
    ).build_log(
        epoch_id=1,
        close_block=359,
        snapshots=snapshots,
        burn_uid=94,
        audit_manifest=substituted_manifest,
        now=completed_at,
        competition_result=evidence.result,
        competition_packet_scores=evidence.packet_scores,
    )
    bundle_source = InMemoryBundleSource()
    for bundle in substituted_bundles:
        bundle_source.add(bundle)
    auditor = Auditor(
        AuditorConfig(
            auditor_hotkey="auditor-unsampled",
            tokenomics=tokenomics,
            burn_uid=94,
        ),
        bundle_source,
        chain=anchor_chain,
    )

    audit = auditor.audit_epoch(
        forged_log,
        store,
        SamplePolicy(sample_rate=0.0, min_samples=0),
        None,
        completed_at,
    )

    assert audit.competition_n == 0
    assert audit.overall is AuditStatus.DISPUTED
    mismatch = next(
        verdict
        for verdict in audit.earning_verdicts
        if verdict.source == "competition-economics"
        and verdict.verdict is ItemVerdictKind.FAIL
    )
    assert "does not open its anchored manifest commitment" in mismatch.detail
