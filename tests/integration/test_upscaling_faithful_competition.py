"""A faithful upscaling competition end to end: the anchored manifest selects the faithful
scorer and may price speed; the packet carries the faithful breakdown; the epoch evidence,
the published log and the auditor all agree. Manifests without the selector keep their
anchored digests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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
    make_store,
    reward_parameter_digest,
    sha256_hex,
)
from vidaio.auditor import (
    Auditor,
    AuditorConfig,
    InMemoryBundleSource,
    ItemVerdictKind,
    SamplePolicy,
)
from vidaio.authority import EpochFinalizer, build_audit_manifest
from vidaio.chain.adapter import ChainNeuron, InMemoryChain
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
from vidaio.epoch import EpochLog, MinerCensusEntry
from vidaio.scoring import (
    ScoringConfig,
    UpscaledFrameMetrics,
    compose_item_score,
    score_upscaling_faithful,
)
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import EmissionState, MinerSnapshot

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
COMPETITION_ID = "comp-upscaling-faithful"
CHALLENGE_ID = "chal-upscaling-faithful"
THRESHOLD_COMMITMENT = "f" * 64
SCORER_VERSION = "cpu-upscaling-faithful-test-v1"
HOTKEY = "hk-upscaling-winner"
FACTOR, TARGET_WIDTH, TARGET_HEIGHT = 4, 1920, 1080
BACKENDS = {"upscaling_faithful_metrics": "upscaling-faithful-metrics/1"}
RULES = {"crown_margin": 0.05, "crown_min_score": 0.50, "podium_min_score": 0.30}
SPEED_RULES = {
    **RULES,
    "speed_weight": 0.3,
    "speed_fast_spf": 0.05,
    "speed_slow_spf": 2.0,
    "speed_inserted_frames": 120,
}
RESOURCES = {"cpu": 16.0, "memory_mb": 65536, "batch_timeout_seconds": 600}


def _manifest_data(reference_digest: str, input_digest: str, **overrides) -> dict:
    commitment = evaluation_item_commitment(
        competition_id=COMPETITION_ID,
        item_index=0,
        reference_sha256=reference_digest,
        input_sha256=input_digest,
        upscale_factor=FACTOR,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
    )
    data = {
        "competition_id": COMPETITION_ID,
        "track": "upscaling",
        "start_time": T0 + timedelta(hours=1),
        "enrollment_deadline": T0 + timedelta(hours=2),
        "finalization_time": T0 + timedelta(hours=3),
        "end_time": T0 + timedelta(hours=4),
        "minimum_alpha_stake": 1.0,
        "scoring_factors": {"quality": 0.6, "cost_efficiency": 0.0, "length_coverage": 0.4},
        "vmaf_threshold": 90.0,
        "sealed_vmaf_variants": [85.0, 89.0, 93.0],
        "allowed_gpus": ["L4"],
        "allowed_upscale_factors": [2, 4],
        "upscaling_scoring": "faithful",
        "evaluation_item_commitments": [commitment],
        "evaluation_batch_size": {"min": 1, "max": 1},
        "scoring_seed_commitment": "a" * 64,
        "container_size_limit_gb": 25.0,
        "scoring_version": SCORER_VERSION,
        "baseline": None,
        "result_rules": RULES,
    }
    data.update(overrides)
    return data


def _packet(*, item_id: str, output_digest: str):
    config = ScoringConfig()
    breakdown = score_upscaling_faithful(
        metrics=UpscaledFrameMetrics(psnr_db=34.5, ssim=0.97, lpips_vgg=0.06, frames=120),
        floor=UpscaledFrameMetrics(psnr_db=30.0, ssim=0.90, lpips_vgg=0.20, frames=120),
        upscale_factor=FACTOR,
        config=config,
    )
    metrics = {
        "upscale_psnr_db": breakdown.psnr_db,
        "upscale_lpips_vgg": breakdown.lpips_vgg,
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
        miner_hotkey=HOTKEY,
        content_digest=output_digest,
        metrics=metrics,
        backend_versions=BACKENDS,
        scorer_version=SCORER_VERSION,
    )
    return packet.to_json().encode("utf-8"), metrics, breakdown.model_dump(mode="json")


def _run(tmp_path: Path, *, rules: dict = RULES, resources: dict | None = None,
         batch_walls: tuple[float, ...] | None = None, full: bool = True):
    tokenomics = TokenomicsConfig(competition_emissions_enabled=True)
    store = make_store(
        AuditConfig(backend="local", local_root=tmp_path / "audit", allow_plaintext_holdout=True)
    )
    input_ref = store.put(b"served-input-quarter-resolution", ArtifactKind.CHALLENGE_INPUT)
    reference_ref = store.put(b"sealed-1080p-original", ArtifactKind.REFERENCE_ORIGINAL)
    output_ref = store.put(b"contender-upscaled-output", ArtifactKind.MINER_OUTPUT)
    archive_ref = store.put(b"sealed-contender-source-archive", ArtifactKind.SUBMISSION_ARCHIVE)

    manifest = CompetitionManifest.model_validate(
        _manifest_data(
            reference_ref.digest, input_ref.digest, result_rules=rules,
            **({} if resources is None else {"sandbox_resources": resources}),
        )
    )
    manifest_ref = store.put(manifest.canonical_json().encode("utf-8"), ArtifactKind.MANIFEST)
    binding = CompetitionItemBinding(
        item_index=0,
        input_sha256=input_ref.digest,
        reference_sha256=reference_ref.digest,
        upscale_factor=FACTOR,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        item_commitment=(manifest.evaluation_item_commitments or [])[0],
    )

    conn = connect(":memory:")
    migrate(conn)
    repo.insert_competition(conn, manifest, T0)
    commitment = build_competition_commitment(
        CompetitionCommitment(
            manifest_digest=manifest.manifest_digest(),
            baseline_version=None,
            baseline_artifact_digest=None,
            baseline_provenance_digest=None,
            baseline_tree_digest=None,
            baseline_image_digest=None,
            dataset_selection_seed_commitment=manifest.scoring_seed_commitment,
            reward_param_digest=reward_parameter_digest(tokenomics),
        )
    )
    store.put(commitment.canonical_json, ArtifactKind.MANIFEST)
    anchor_at = T0 + timedelta(minutes=1)
    chain = InMemoryChain(
        _neurons=[
            ChainNeuron(uid=42, hotkey=HOTKEY, coldkey="ck", ip="203.0.113.42",
                        alpha_stake=10.0, emission=0.0)
        ],
        _block=10_000,
        anchored=[commitment.payload],
        _anchor_blocks=[10],
        block_time_anchor=(10, anchor_at),
    )
    engine = LifecycleEngine()
    engine.mark_commitment_anchored(
        conn, COMPETITION_ID, commitment.root, anchor_at,
        onchain_evidence={
            "root": commitment.root,
            "anchor_netuid": 85,
            "payload_hex": commitment.payload.hex(),
            "payload_digest": sha256_hex(commitment.payload),
            "anchor_block": 10,
            "anchor_block_hash": chain.block_hash(10),
            "finalized_block": 10,
            "archive_verified": True,
        },
    )
    engine.tick(conn, manifest.start_time)
    contender_id = repo.enroll_contender(
        conn, COMPETITION_ID, hotkey=HOTKEY,
        repo_url="https://example.invalid/vidaio/contender",
        commit_sha="e" * 40, tree_sha="1" * 40, stake=10.0, now=T0,
    )
    repo.set_contender_image_digest(conn, contender_id, "3" * 64, T0)
    repo.set_contender_status(conn, contender_id, "BUILT", T0)
    record_submission_archived(
        conn, COMPETITION_ID, contender_id, archive_ref.digest, archive_ref.byte_size, T0
    )
    if batch_walls is not None:
        from vidaio.competition.orchestrator import persistence as pers

        pers.ensure_batches(conn, COMPETITION_ID, contender_id, n_items=len(batch_walls),
                            batch_size=1, now=T0)
        rows = conn.execute(
            "SELECT batch_id FROM batches WHERE contender_id = ? ORDER BY batch_index",
            (contender_id,),
        ).fetchall()
        for row, wall in zip(rows, batch_walls):
            pers.complete_batch(
                conn, COMPETITION_ID, row["batch_id"], contender_id, [], T0, wall_seconds=wall
            )
    item_id = repo.add_evaluation_item(
        conn, COMPETITION_ID, item_index=0,
        input_sha256=input_ref.digest, input_bytes=input_ref.byte_size,
        reference_sha256=reference_ref.digest, reference_bytes=reference_ref.byte_size,
        upscale_factor=FACTOR, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT,
        threshold_commitment=THRESHOLD_COMMITMENT, challenge_id=CHALLENGE_ID, now=T0,
    )
    packet_bytes, metrics, breakdown = _packet(item_id=input_ref.digest, output_digest=output_ref.digest)
    packet_ref = store.put(packet_bytes, ArtifactKind.SCORE_PACKET)
    bundle = build_bundle(
        challenge_id=CHALLENGE_ID, item_id=input_ref.digest, miner_hotkey=HOTKEY,
        commitment_hash=THRESHOLD_COMMITMENT, stage=LifecycleStage.COMPETITION_SEALED,
        challenge_input=input_ref, reference_original=reference_ref,
        miner_output=output_ref, manifest=manifest_ref, score_packet=packet_ref,
        competition_item=binding, execution_image_digest="3" * 64,
        scorer_version=SCORER_VERSION, backend_versions=BACKENDS,
        created_at=(T0 + timedelta(hours=5)).isoformat(),
    )
    bundle_ref = store.put(
        canonical_json_bytes(bundle.model_dump(mode="json")), ArtifactKind.AUDIT_BUNDLE
    )
    performance_id = repo.record_item_score(
        conn, COMPETITION_ID, contender_id=contender_id, item_id=item_id,
        packet_bytes=packet_bytes, output_bytes=output_ref.byte_size,
        now=T0 + timedelta(hours=5),
    )
    repo.set_audit_bundle_digest(conn, performance_id, bundle_ref.digest)
    completed_at = T0 + timedelta(hours=6)
    repo.set_status(conn, COMPETITION_ID, Phase.COMPLETED, completed_at)
    repo.record_event(
        conn, COMPETITION_ID, "phase_transition", completed_at,
        from_phase=Phase.AWAITING_END_TIME, to_phase=Phase.COMPLETED,
    )
    store.release(reference_ref)

    evidence = build_competition_epoch_evidence(
        conn,
        competition_id=COMPETITION_ID,
        census_by_hotkey={
            HOTKEY: MinerCensusEntry(uid=42, hotkey=HOTKEY, coldkey="ck", ip="203.0.113.42")
        },
        store=store,
        tokenomics=tokenomics,
        through_time=completed_at,
    )
    assert evidence is not None
    if not full:
        return evidence, None
    comp_input = evidence.competition_input
    assert store.is_released(archive_ref)  # crown: the winning source is public

    audit_manifest = build_audit_manifest(evidence.scored_items, store=store, competition_input=comp_input)
    snapshots = (
        MinerSnapshot(uid=42, hotkey=HOTKEY, coldkey="ck", ip="203.0.113.42",
                      track="compression", accumulate_score=0.0, alpha_stake=10.0),
    )
    log = EpochFinalizer(tokenomics, scorer_version=SCORER_VERSION).build_log(
        epoch_id=1, close_block=359, snapshots=snapshots, burn_uid=94,
        audit_manifest=audit_manifest, now=completed_at,
        competition_result=evidence.result,
        competition_packet_scores=evidence.packet_scores,
    )
    assert log.reward_window_state.kind is EmissionState.CROWN
    assert log.weight_shares.get(42, 0.0) > 0.99
    reparsed = EpochLog.from_json(log.to_json())
    assert reparsed.log_digest() == log.log_digest()
    item = reparsed.audit_manifest.competition_input.items[0]
    assert (item.upscale_factor, item.target_width, item.target_height) == (FACTOR, TARGET_WIDTH, TARGET_HEIGHT)

    source = InMemoryBundleSource()
    source.add(bundle)
    recomputer = StaticRecomputer(
        metrics, SCORER_VERSION, score=float(metrics["final_score"]), gate_passed=True, breakdown=breakdown,
    )
    audit = Auditor(
        AuditorConfig(auditor_hotkey="auditor-upscaling-faithful", tokenomics=tokenomics, burn_uid=94),
        source,
        chain=chain,
    ).audit_epoch(log, store, SamplePolicy(sample_rate=1.0), recomputer, completed_at)
    competition = [v for v in audit.item_verdicts if v.source == "competition"]
    assert competition and all(v.verdict is ItemVerdictKind.PASS for v in competition), [
        (v.code, v.detail) for v in competition
    ]
    failures = [
        (v.source, v.code, v.detail)
        for v in audit.earning_verdicts
        if v.verdict is not ItemVerdictKind.PASS and v.source.startswith("competition")
    ]
    assert not failures, failures
    return evidence, log


def test_faithful_upscaling_result_audits_clean(tmp_path: Path) -> None:
    evidence, log = _run(tmp_path)
    packet = evidence.packet_scores
    assert len(packet) == 1 and 0.5 < next(iter(packet.values())) <= 1.0
    assert all(s.run_seconds is None for s in evidence.competition_input.subjects)
    assert b"speed_weight" not in log.to_json()


def test_faithful_upscaling_speed_term_is_committed_and_audits_clean(tmp_path: Path) -> None:
    # 120 output frames in 3 s = 0.025 s/frame: faster than speed_fast_spf, full credit
    evidence, log = _run(tmp_path, rules=SPEED_RULES, resources=RESOURCES, batch_walls=(3.0,))
    subject = evidence.competition_input.subjects[0]
    assert subject.run_seconds == 3.0
    assert b'"speed_weight":0.3' in log.to_json()
    quality = evidence.packet_scores[subject.packet_digests[0]]
    assert abs(evidence.result.contenders[0].score - quality) < 1e-12


def test_faithful_upscaling_speed_term_prices_slow_runs(tmp_path: Path) -> None:
    from vidaio.competition.economic_result import speed_adjusted_score, speed_score

    evidence, _ = _run(tmp_path, rules=SPEED_RULES, resources=RESOURCES, batch_walls=(60.0,), full=False)
    subject = evidence.competition_input.subjects[0]
    quality = evidence.packet_scores[subject.packet_digests[0]]
    speed = speed_score(60.0, inserted_frames=120, fast_spf=0.05, slow_spf=2.0)
    assert 0.0 < speed < 1.0
    expected = speed_adjusted_score(quality, speed, weight=0.3)
    assert abs(evidence.result.contenders[0].score - expected) < 1e-12
    assert expected < quality


def test_manifest_selector_is_fenced_and_validated() -> None:
    data = _manifest_data("2" * 64, "1" * 64)
    faithful = CompetitionManifest.model_validate(data)
    assert '"upscaling_scoring":"faithful"' in faithful.canonical_json()
    # an upscaling manifest without the selector serializes exactly as before it existed
    legacy = CompetitionManifest.model_validate({k: v for k, v in data.items() if k != "upscaling_scoring"})
    assert "upscaling_scoring" not in legacy.canonical_json()
    assert legacy.manifest_digest() != faithful.manifest_digest()
    with pytest.raises(ValueError, match="valid only for the upscaling track"):
        CompetitionManifest.model_validate(
            {**data, "track": "interpolation", "allowed_upscale_factors": None}
        )
    with pytest.raises(ValueError):
        CompetitionManifest.model_validate({**data, "upscaling_scoring": "generative"})


def test_speed_term_needs_faithful_upscaling_and_a_sandbox_envelope() -> None:
    data = _manifest_data("2" * 64, "1" * 64, result_rules=SPEED_RULES, sandbox_resources=RESOURCES)
    assert CompetitionManifest.model_validate(data).result_rules.has_speed_term
    inference = {k: v for k, v in data.items() if k != "upscaling_scoring"}
    with pytest.raises(ValueError, match="speed term"):
        CompetitionManifest.model_validate(inference)
    with pytest.raises(ValueError, match="sandbox_resources"):
        CompetitionManifest.model_validate({k: v for k, v in data.items() if k != "sandbox_resources"})
