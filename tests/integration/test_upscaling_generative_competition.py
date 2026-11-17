"""A generative upscaling competition end to end: the anchored manifest selects the generative
scorer, may price speed and may limit failed items; every packet carries the generative
breakdown; the epoch evidence, the published log and the auditor all agree. Manifests without
the new fields keep their anchored digests."""

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
    GenerativeFrameMetrics,
    ReasonCode,
    ScoringConfig,
    compose_item_score,
    score_upscaling_generative,
)
from vidaio.scoring.gates import upscaling_generative_violations
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import EmissionState, MinerSnapshot

T0 = datetime(2026, 11, 18, tzinfo=timezone.utc)
COMPETITION_ID = "comp-upscaling-generative"
THRESHOLD_COMMITMENT = "f" * 64
SCORER_VERSION = "cpu-upscaling-generative-test-v1"
HOTKEY = "hk-upscaling-generative"
FACTOR, TARGET_WIDTH, TARGET_HEIGHT = 4, 1920, 1080
BACKENDS = {
    "upscaling_generative_metrics": "upscaling-generative-metrics/1",
    "generative_weights": "sha256:" + "6" * 64,
}
RULES = {"crown_margin": 0.05, "crown_min_score": 0.50, "podium_min_score": 0.30}
SPEED_RULES = {
    **RULES,
    "speed_weight": 0.3,
    "speed_fast_spf": 0.5,
    "speed_slow_spf": 20.0,
    "speed_inserted_frames": 120,
}
RESOURCES = {"cpu": 16.0, "memory_mb": 131072, "batch_timeout_seconds": 1200}
GOOD = dict(dists=0.08, lpips_vgg=0.18, warp_error=2.0, warp_error_reference=1.9, nr=0.63,
            nr_reference=0.62, lowres_psnr_db=36.5, color_delta_e=1.1, frames=120, fr_frames=3)


def _commitment(index: int, reference_digest: str, input_digest: str) -> str:
    return evaluation_item_commitment(
        competition_id=COMPETITION_ID,
        item_index=index,
        reference_sha256=reference_digest,
        input_sha256=input_digest,
        upscale_factor=FACTOR,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
    )


def _manifest_data(commitments: list[str], **overrides) -> dict:
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
        "allowed_gpus": ["A100-80GB"],
        "allowed_upscale_factors": [2, 4],
        "upscaling_scoring": "generative",
        "evaluation_item_commitments": commitments,
        "evaluation_batch_size": {"min": 1, "max": 1},
        "scoring_seed_commitment": "a" * 64,
        "container_size_limit_gb": 40.0,
        "scoring_version": SCORER_VERSION,
        "baseline": None,
        "result_rules": RULES,
    }
    data.update(overrides)
    return data


def _packet(*, index: int, item_id: str, output_digest: str, failed: bool):
    config = ScoringConfig()
    measured = GenerativeFrameMetrics(**{**GOOD, **({"lowres_psnr_db": 22.0} if failed else {})})
    breakdown = score_upscaling_generative(metrics=measured, upscale_factor=FACTOR, config=config)
    violations = upscaling_generative_violations(
        lowres_psnr_db=measured.lowres_psnr_db, color_delta_e=measured.color_delta_e, config=config
    )
    metrics = {
        "gen_dists": breakdown.dists,
        "gen_lowres_psnr_db": breakdown.lowres_psnr_db,
        "final_score": breakdown.final,
    }
    packet = compose_item_score(
        item_id=item_id,
        challenge_id=f"chal-generative-{index:03d}",
        track="upscaling",
        gate_passed=not violations,
        violations=violations,
        breakdown=breakdown,
        config=config,
        miner_hotkey=HOTKEY,
        content_digest=output_digest,
        metrics=metrics,
        backend_versions=BACKENDS,
        scorer_version=SCORER_VERSION,
    )
    return packet, metrics, breakdown.model_dump(mode="json")


def _run(tmp_path: Path, *, items: int = 1, failed: int = 0, rules: dict = RULES,
         resources: dict | None = None, batch_walls: tuple[float, ...] | None = None, full: bool = True):
    tokenomics = TokenomicsConfig(competition_emissions_enabled=True)
    store = make_store(
        AuditConfig(backend="local", local_root=tmp_path / "audit", allow_plaintext_holdout=True)
    )
    refs = []
    for index in range(items):
        input_ref = store.put(f"served-input-{index}".encode(), ArtifactKind.CHALLENGE_INPUT)
        reference_ref = store.put(f"sealed-original-{index}".encode(), ArtifactKind.REFERENCE_ORIGINAL)
        output_ref = store.put(f"generative-output-{index}".encode(), ArtifactKind.MINER_OUTPUT)
        refs.append((input_ref, reference_ref, output_ref))
    archive_ref = store.put(b"sealed-contender-source-archive", ArtifactKind.SUBMISSION_ARCHIVE)
    commitments = [_commitment(i, r.digest, inp.digest) for i, (inp, r, _) in enumerate(refs)]
    manifest = CompetitionManifest.model_validate(
        _manifest_data(
            commitments, result_rules=rules,
            **({} if resources is None else {"sandbox_resources": resources}),
        )
    )
    manifest_ref = store.put(manifest.canonical_json().encode("utf-8"), ArtifactKind.MANIFEST)

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
        repo_url="https://example.invalid/vidaio/generative-contender",
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
    bundles = []
    expected_by_item = {}
    for index, (input_ref, reference_ref, output_ref) in enumerate(refs):
        binding = CompetitionItemBinding(
            item_index=index,
            input_sha256=input_ref.digest,
            reference_sha256=reference_ref.digest,
            upscale_factor=FACTOR,
            target_width=TARGET_WIDTH,
            target_height=TARGET_HEIGHT,
            item_commitment=commitments[index],
        )
        item_id = repo.add_evaluation_item(
            conn, COMPETITION_ID, item_index=index,
            input_sha256=input_ref.digest, input_bytes=input_ref.byte_size,
            reference_sha256=reference_ref.digest, reference_bytes=reference_ref.byte_size,
            upscale_factor=FACTOR, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT,
            threshold_commitment=THRESHOLD_COMMITMENT, challenge_id=f"chal-generative-{index:03d}", now=T0,
        )
        packet, metrics, breakdown = _packet(
            index=index, item_id=input_ref.digest, output_digest=output_ref.digest, failed=index < failed
        )
        packet_bytes = packet.to_json().encode("utf-8")
        packet_ref = store.put(packet_bytes, ArtifactKind.SCORE_PACKET)
        bundle = build_bundle(
            challenge_id=f"chal-generative-{index:03d}", item_id=input_ref.digest, miner_hotkey=HOTKEY,
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
        bundles.append(bundle)
        expected_by_item[input_ref.digest] = (packet, metrics, breakdown)
    completed_at = T0 + timedelta(hours=6)
    repo.set_status(conn, COMPETITION_ID, Phase.COMPLETED, completed_at)
    repo.record_event(
        conn, COMPETITION_ID, "phase_transition", completed_at,
        from_phase=Phase.AWAITING_END_TIME, to_phase=Phase.COMPLETED,
    )
    for _, reference_ref, _ in refs:
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
    if not full:
        return evidence, None, expected_by_item
    assert evidence is not None
    comp_input = evidence.competition_input
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
    reparsed = EpochLog.from_json(log.to_json())
    assert reparsed.log_digest() == log.log_digest()

    for bundle in bundles:
        packet, metrics, breakdown = expected_by_item[bundle.item_id]
        source = InMemoryBundleSource()
        source.add(bundle)
        recomputer = StaticRecomputer(
            metrics, SCORER_VERSION, score=packet.score, gate_passed=packet.gate_passed,
            breakdown=breakdown, violations=[v.model_dump(mode="json") for v in packet.violations],
        ) if packet.violations else StaticRecomputer(
            metrics, SCORER_VERSION, score=packet.score, gate_passed=True, breakdown=breakdown,
        )
        audit = Auditor(
            AuditorConfig(auditor_hotkey="auditor-upscaling-generative", tokenomics=tokenomics, burn_uid=94),
            source,
            chain=chain,
        ).audit_epoch(log, store, SamplePolicy(sample_rate=1.0), recomputer, completed_at)
        competition = [
            v for v in audit.item_verdicts if v.source == "competition" and v.item_id == bundle.item_id
        ]
        assert competition and all(v.verdict is ItemVerdictKind.PASS for v in competition), [
            (v.code, v.detail) for v in competition
        ]
    return evidence, log, expected_by_item


def test_generative_upscaling_result_audits_clean(tmp_path: Path) -> None:
    evidence, log, expected = _run(tmp_path)
    assert log.reward_window_state.kind is EmissionState.CROWN
    assert log.weight_shares.get(42, 0.0) > 0.99
    (packet, _, _), = expected.values()
    assert packet.breakdown is not None and packet.breakdown.kind == "upscaling_generative"
    assert evidence.result.contenders[0].score == pytest.approx(packet.score)
    assert b"speed_weight" not in log.to_json() and b"max_failed_item_fraction" not in log.to_json()


def test_generative_upscaling_speed_term_is_committed(tmp_path: Path) -> None:
    from vidaio.competition.economic_result import speed_adjusted_score, speed_score

    evidence, log, _ = _run(tmp_path, rules=SPEED_RULES, resources=RESOURCES, batch_walls=(600.0,))
    subject = evidence.competition_input.subjects[0]
    assert subject.run_seconds == 600.0
    assert b'"speed_weight":0.3' in log.to_json()
    quality = evidence.packet_scores[subject.packet_digests[0]]
    speed = speed_score(600.0, inserted_frames=120, fast_spf=0.5, slow_spf=20.0)
    assert evidence.result.contenders[0].score == pytest.approx(speed_adjusted_score(quality, speed, weight=0.3))


def test_failed_items_up_to_the_limit_keep_the_rank(tmp_path: Path) -> None:
    rules = {**RULES, "max_failed_item_fraction": 0.10}
    evidence, log, expected = _run(tmp_path, items=10, failed=1, rules=rules)
    gated = [p for p, _, _ in expected.values() if not p.gate_passed]
    assert len(gated) == 1 and gated[0].score == 0.0
    assert any(v.code is ReasonCode.LOWRES_INCONSISTENT for v in gated[0].violations)
    assert b'"max_failed_item_fraction":0.1' in log.to_json()
    mean = sum(p.score for p, _, _ in expected.values()) / 10
    assert evidence.result.contenders[0].score == pytest.approx(mean)
    assert log.weight_shares.get(42, 0.0) > 0.99


def test_more_failed_items_than_the_limit_lose_the_paid_rank(tmp_path: Path) -> None:
    from vidaio.competition.economic_result import derive_competition_economics

    rules = {**RULES, "max_failed_item_fraction": 0.10}
    evidence, _, expected = _run(tmp_path / "over", items=10, failed=2, rules=rules, full=False)
    mean = sum(p.score for p, _, _ in expected.values()) / 10
    assert mean > RULES["podium_min_score"]  # the mean alone would have qualified
    # ranked with 0: nobody meets the podium bar, so there is no economic result to apply
    assert evidence is None
    # the same committed evidence derives a zero rank once the limit is exceeded
    evidence, _, _ = _run(tmp_path / "at", items=10, failed=1, rules=rules, full=False)
    comp_input = evidence.competition_input
    stricter = comp_input.model_copy(
        update={"rules": comp_input.rules.model_copy(update={"max_failed_item_fraction": 0.05})}
    )
    derivation = derive_competition_economics(stricter, evidence.packet_scores)
    (subject,) = derivation.subject_aggregates
    assert subject.failed_fraction == pytest.approx(0.1)
    assert subject.quality == pytest.approx(evidence.result.contenders[0].score)
    assert subject.score == 0.0 and derivation.result.contenders[0].score == 0.0


def test_manifest_selector_and_rules_are_fenced_and_validated() -> None:
    commitments = [_commitment(0, "2" * 64, "1" * 64)]
    data = _manifest_data(commitments)
    generative = CompetitionManifest.model_validate(data)
    assert '"upscaling_scoring":"generative"' in generative.canonical_json()
    assert "max_failed_item_fraction" not in generative.canonical_json()
    limited = CompetitionManifest.model_validate(
        {**data, "result_rules": {**RULES, "max_failed_item_fraction": 0.1}}
    )
    assert '"max_failed_item_fraction":0.1' in limited.canonical_json()
    assert limited.manifest_digest() != generative.manifest_digest()
    with pytest.raises(ValueError, match="podium_min_score"):
        CompetitionManifest.model_validate(
            {**data, "result_rules": {"crown_margin": 0.05, "max_failed_item_fraction": 0.1}}
        )
    with pytest.raises(ValueError, match="valid only for the upscaling track"):
        CompetitionManifest.model_validate(
            {**data, "track": "interpolation", "allowed_upscale_factors": None}
        )
    # speed term: allowed for generative upscaling with an anchored envelope
    timed = CompetitionManifest.model_validate(
        {**data, "result_rules": SPEED_RULES, "sandbox_resources": RESOURCES}
    )
    assert timed.result_rules.has_speed_term
    with pytest.raises(ValueError, match="sandbox_resources"):
        CompetitionManifest.model_validate({**data, "result_rules": SPEED_RULES})
