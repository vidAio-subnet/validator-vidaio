"""A frame-interpolation competition end to end: the item commitment binds the decimated
input, the sealed full-frame-rate reference and the frame factor; the packet carries the
interpolation breakdown; the epoch evidence, the published log and the auditor all agree."""

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
    interpolation_item_commitment,
    migrate,
)
from vidaio.competition import repository as repo
from vidaio.competition.epoch_evidence import build_competition_epoch_evidence
from vidaio.competition.orchestrator.persistence import record_submission_archived
from vidaio.competition.states import Phase
from vidaio.core import connect
from vidaio.epoch import EpochLog, MinerCensusEntry
from vidaio.scoring import ScoringConfig, compose_item_score, score_interpolation
from vidaio.scoring.interpolation_formula import InsertedFrameMetrics
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import EmissionState, MinerSnapshot

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
COMPETITION_ID = "comp-interpolation-economics"
CHALLENGE_ID = "chal-interpolation-economics"
THRESHOLD_COMMITMENT = "f" * 64
SCORER_VERSION = "cpu-interpolation-test-v1"
HOTKEY = "hk-interpolation-winner"
RULES = {"crown_margin": 0.05, "crown_min_score": 0.70, "podium_min_score": 0.30}


def _manifest(
    reference_digest: str,
    input_digest: str,
    *,
    rules: dict | None = None,
    resources: dict | None = None,
) -> CompetitionManifest:
    commitment = interpolation_item_commitment(
        competition_id=COMPETITION_ID,
        item_index=0,
        input_sha256=input_digest,
        reference_sha256=reference_digest,
    )
    return CompetitionManifest.model_validate(
        {
            "competition_id": COMPETITION_ID,
            "track": "interpolation",
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
            "evaluation_item_commitments": [commitment],
            "evaluation_batch_size": {"min": 1, "max": 1},
            "scoring_seed_commitment": "a" * 64,
            "container_size_limit_gb": 25.0,
            "scoring_version": SCORER_VERSION,
            "baseline": None,
            "result_rules": RULES if rules is None else rules,
            **({} if resources is None else {"sandbox_resources": resources}),
        }
    )


def _packet(
    *, item_id: str, hotkey: str | None, output_digest: str, psnr_db: float
) -> tuple[bytes, dict[str, float], dict[str, float | str]]:
    config = ScoringConfig()
    floor = InsertedFrameMetrics(psnr_db=30.0, ssim=0.90, lpips_vgg=0.20, inserted_frames=59)
    breakdown = score_interpolation(
        metrics=InsertedFrameMetrics(psnr_db=psnr_db, ssim=0.97, lpips_vgg=0.05, inserted_frames=59),
        floor=floor,
        config=config,
    )
    metrics = {
        "interp_psnr_db": breakdown.psnr_db,
        "interp_lpips_vgg": breakdown.lpips_vgg,
        "final_score": breakdown.final,
    }
    packet = compose_item_score(
        item_id=item_id,
        challenge_id=CHALLENGE_ID,
        track="interpolation",
        gate_passed=True,
        violations=[],
        breakdown=breakdown,
        config=config,
        miner_hotkey=hotkey,
        content_digest=output_digest,
        metrics=metrics,
        backend_versions={"interpolation_metrics": "interpolation-metrics/1"},
        scorer_version=SCORER_VERSION,
    )
    return packet.to_json().encode("utf-8"), metrics, breakdown.model_dump(mode="json")


def _run_interpolation_economics(
    tmp_path: Path,
    *,
    rules: dict | None = None,
    resources: dict | None = None,
    batch_walls: tuple[float, ...] | None = None,
    failed_batches: int = 0,
    stopped_attempts: dict[int, tuple[float | None, ...]] | None = None,
    amended_timeout: int | None = None,
    full: bool = True,
):
    """One contender through evidence, the published log and the auditor.

    ``batch_walls`` (speed term only) records COMPLETED batches with those measured
    run times; ``failed_batches`` more batches fail by contender fault;
    ``stopped_attempts`` maps a completed batch's position to the run times of
    attempts whose Sandbox was stopped before it completed; ``amended_timeout``
    raises the batch timeout by an execution amendment."""
    tokenomics = TokenomicsConfig(competition_emissions_enabled=True)
    store = make_store(
        AuditConfig(backend="local", local_root=tmp_path / "audit", allow_plaintext_holdout=True)
    )
    input_ref = store.put(b"served-input-every-other-frame", ArtifactKind.CHALLENGE_INPUT)
    reference_ref = store.put(b"full-frame-rate-reference", ArtifactKind.REFERENCE_ORIGINAL)
    output_ref = store.put(b"contender-interpolated-output", ArtifactKind.MINER_OUTPUT)
    archive_ref = store.put(b"sealed-contender-source-archive", ArtifactKind.SUBMISSION_ARCHIVE)

    manifest = _manifest(reference_ref.digest, input_ref.digest, rules=rules, resources=resources)
    manifest_ref = store.put(manifest.canonical_json().encode("utf-8"), ArtifactKind.MANIFEST)
    binding = CompetitionItemBinding(
        item_index=0,
        input_sha256=input_ref.digest,
        reference_sha256=reference_ref.digest,
        frame_factor=2,
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
        conn,
        COMPETITION_ID,
        commitment.root,
        anchor_at,
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

        pers.ensure_batches(
            conn, COMPETITION_ID, contender_id,
            n_items=len(batch_walls) + failed_batches, batch_size=1, now=T0,
        )
        rows = conn.execute(
            "SELECT batch_id FROM batches WHERE contender_id = ? ORDER BY batch_index",
            (contender_id,),
        ).fetchall()
        if amended_timeout is not None:
            repo.record_event(
                conn, COMPETITION_ID, pers.EVENT_EXECUTION_AMENDED, T0,
                payload={"effective": {**(resources or {}), "batch_timeout_seconds": amended_timeout}},
            )
        for position, (row, wall) in enumerate(zip(rows, batch_walls)):
            for stopped in (stopped_attempts or {}).get(position, ()):
                pers.requeue_batch(
                    conn, COMPETITION_ID, row["batch_id"], "Sandbox stopped", T0,
                    code="SANDBOX_TERMINATED", wall_seconds=stopped,
                )
            pers.complete_batch(
                conn, COMPETITION_ID, row["batch_id"], contender_id, [], T0, wall_seconds=wall
            )
        for row in rows[len(batch_walls):]:
            pers.fail_batch_contender_fault(
                conn, COMPETITION_ID, row["batch_id"], contender_id, "SOLUTION_TIMEOUT", "timed out", T0
            )
    item_id = repo.add_evaluation_item(
        conn, COMPETITION_ID, item_index=0,
        input_sha256=input_ref.digest, input_bytes=input_ref.byte_size,
        reference_sha256=reference_ref.digest, reference_bytes=reference_ref.byte_size,
        threshold_commitment=THRESHOLD_COMMITMENT, challenge_id=CHALLENGE_ID, now=T0,
    )
    packet_bytes, metrics, breakdown = _packet(
        item_id=input_ref.digest, hotkey=HOTKEY, output_digest=output_ref.digest,
        psnr_db=36.0,
    )
    packet_ref = store.put(packet_bytes, ArtifactKind.SCORE_PACKET)
    bundle = build_bundle(
        challenge_id=CHALLENGE_ID, item_id=input_ref.digest, miner_hotkey=HOTKEY,
        commitment_hash=THRESHOLD_COMMITMENT, stage=LifecycleStage.COMPETITION_SEALED,
        challenge_input=input_ref, reference_original=reference_ref,
        miner_output=output_ref, manifest=manifest_ref, score_packet=packet_ref,
        competition_item=binding, execution_image_digest="3" * 64,
        scorer_version=SCORER_VERSION,
        backend_versions={"interpolation_metrics": "interpolation-metrics/1"},
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
    comp_input = evidence.competition_input
    if not full:
        return evidence, None, None
    assert comp_input.baseline_version is None and comp_input.baseline_artifact_digest is None
    assert [s.role for s in comp_input.subjects] == ["contender"]
    assert evidence.result.baseline_score is None and not evidence.result.has_baseline
    # the winner reached crown_min_score, so its archive is public (CROWN disclosure)
    assert store.is_released(archive_ref)

    audit_manifest = build_audit_manifest(
        evidence.scored_items, store=store, competition_input=comp_input
    )
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
    window = log.reward_window_state
    assert window.kind is EmissionState.CROWN
    assert window.podium_hotkeys == (HOTKEY,)
    assert window.baseline_version is None and window.baseline_score is None
    assert abs(window.winner_margin - window.winner_score) < 1e-12
    assert log.weight_shares.get(42, 0.0) > 0.99

    # the published bytes re-parse to the same log (null baseline, interpolation items)
    reparsed = EpochLog.from_json(log.to_json())
    assert reparsed.log_digest() == log.log_digest()
    item = reparsed.audit_manifest.competition_input.items[0]
    assert item.frame_factor == 2 and item.mask_stream_index is None
    assert reparsed.audit_manifest.competition_input.track == "interpolation"

    source = InMemoryBundleSource()
    source.add(bundle)
    recomputer = StaticRecomputer(
        metrics, SCORER_VERSION, score=float(metrics["final_score"]),
        gate_passed=True, breakdown=breakdown,
    )
    audit = Auditor(
        AuditorConfig(auditor_hotkey="auditor-interpolation", tokenomics=tokenomics, burn_uid=94),
        source,
        chain=chain,
    ).audit_epoch(log, store, SamplePolicy(sample_rate=1.0), recomputer, completed_at)
    competition = [v for v in audit.item_verdicts if v.source == "competition"]
    assert competition and all(v.verdict is ItemVerdictKind.PASS for v in competition), [
        (v.code, v.detail) for v in competition
    ]
    # the competition-economics pass (commitment, rules, items, subjects, derived
    # result and window) raises no fault; the synthetic chain's block clock is not
    # under test here
    failures = [
        (v.source, v.code, v.detail)
        for v in audit.earning_verdicts
        if v.verdict is not ItemVerdictKind.PASS and v.source.startswith("competition")
    ]
    assert not failures, failures
    economics = [v for v in audit.earning_verdicts if v.source == "competition-economics"]
    assert economics and all(v.verdict is ItemVerdictKind.PASS for v in economics), [
        (v.code, v.detail) for v in economics
    ]
    return evidence, log, audit


def test_interpolation_result_is_decided_by_absolute_bars_and_audits_clean(
    tmp_path: Path,
) -> None:
    evidence, log, audit = _run_interpolation_economics(tmp_path)
    assert all(s.run_seconds is None for s in evidence.competition_input.subjects)
    assert b"run_seconds" not in log.to_json() and b"speed_weight" not in log.to_json()


SPEED_RULES = {
    **RULES,
    "speed_weight": 0.4,
    "speed_fast_spf": 1 / 30,
    "speed_slow_spf": 2.0,
    "speed_inserted_frames": 59,
}
RESOURCES = {"cpu": 16.0, "memory_mb": 65536, "batch_timeout_seconds": 600}


def test_interpolation_speed_term_is_committed_and_audits_clean(tmp_path: Path) -> None:
    # real-time or faster: full speed credit, the score equals the quality mean, and
    # the timed subject survives the published log and the auditor's recompute
    evidence, log, audit = _run_interpolation_economics(
        tmp_path, rules=SPEED_RULES, resources=RESOURCES, batch_walls=(0.9,)
    )
    subject = evidence.competition_input.subjects[0]
    assert subject.run_seconds == 0.9
    assert b'"run_seconds":0.9' in log.to_json() and b'"speed_weight":0.4' in log.to_json()
    quality = evidence.packet_scores[subject.packet_digests[0]]
    assert abs(evidence.result.contenders[0].score - quality) < 1e-12


def test_interpolation_speed_term_prices_measured_run_time(tmp_path: Path) -> None:
    from vidaio.competition.economic_result import speed_adjusted_score, speed_score

    evidence, _, _ = _run_interpolation_economics(
        tmp_path, rules=SPEED_RULES, resources=RESOURCES, batch_walls=(12.5,), failed_batches=1,
        full=False,
    )
    subject = evidence.competition_input.subjects[0]
    # one completed batch at its measured time + one contender-failed batch at the
    # full anchored timeout: failing fast never buys speed
    assert subject.run_seconds == 612.5
    rules = evidence.competition_input.rules
    assert rules.has_speed_term and rules.speed_inserted_frames == 59
    quality = evidence.packet_scores[subject.packet_digests[0]]
    expected = speed_adjusted_score(
        quality, speed_score(612.5, inserted_frames=59, fast_spf=1 / 30, slow_spf=2.0), weight=0.4
    )
    assert speed_score(612.5, inserted_frames=59, fast_spf=1 / 30, slow_spf=2.0) == 0.0
    assert abs(evidence.result.contenders[0].score - expected) < 1e-12
    assert abs(expected - 0.6 * quality) < 1e-12


def test_interpolation_speed_term_charges_stopped_sandbox_attempts(tmp_path: Path) -> None:
    # A solution that brings its own Sandbox down cannot shop for a faster rerun: each
    # stopped attempt counts what it ran, at most the time of the run that completed
    # (an unmeasured one counts that cap).
    evidence, _, _ = _run_interpolation_economics(
        tmp_path, rules=SPEED_RULES, resources=RESOURCES, batch_walls=(12.5, 10.0),
        stopped_attempts={0: (3.0,), 1: (40.0, None)}, full=False,
    )
    assert evidence.competition_input.subjects[0].run_seconds == (12.5 + 3.0) + (10.0 + 10.0 + 10.0)


def test_interpolation_failed_batch_is_charged_the_amended_timeout(tmp_path: Path) -> None:
    # The operator raised the timeout mid-competition: a failed batch ran under the
    # raised envelope and is charged it, not the lower anchored value.
    evidence, _, _ = _run_interpolation_economics(
        tmp_path, rules=SPEED_RULES, resources=RESOURCES, batch_walls=(12.5,),
        failed_batches=1, amended_timeout=900, full=False,
    )
    assert evidence.competition_input.subjects[0].run_seconds == 912.5


def test_interpolation_item_commitment_binds_the_frame_factor() -> None:
    kwargs = dict(competition_id="c", item_index=0, input_sha256="1" * 64, reference_sha256="2" * 64)
    commitment = interpolation_item_commitment(**kwargs)
    assert commitment == interpolation_item_commitment(**kwargs, frame_factor=2)
    with pytest.raises(ValueError):
        interpolation_item_commitment(**kwargs, frame_factor=4)
    with pytest.raises(ValueError):
        interpolation_item_commitment(**{**kwargs, "reference_sha256": "1" * 64})
    from vidaio.competition import removal_item_commitment

    assert commitment != removal_item_commitment(**kwargs)  # distinct domains


def test_interpolation_manifest_requires_precommitted_items() -> None:
    manifest = _manifest("2" * 64, "1" * 64)
    with pytest.raises(ValueError, match="interpolation manifest requires precommitted"):
        CompetitionManifest.model_validate(
            {**manifest.model_dump(), "evaluation_item_commitments": None}
        )
