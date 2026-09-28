"""A competition anchored WITHOUT an executable baseline: the anchored absolute bars
(crown_min_score / podium_min_score) decide the result, the commitment and the epoch
evidence carry null baseline provenance, and the whole path still audits CLEAN."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

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
from vidaio.competition import CompetitionManifest, LifecycleEngine, migrate
from vidaio.competition import repository as repo
from vidaio.competition.epoch_evidence import build_competition_epoch_evidence
from vidaio.competition.orchestrator.persistence import record_submission_archived
from vidaio.competition.states import Phase
from vidaio.core import connect
from vidaio.epoch import EpochLog, MinerCensusEntry
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import EmissionState, MinerSnapshot

from tests.integration.test_removal_competition_epoch_evidence import (
    CHALLENGE_ID,
    COMPETITION_ID,
    SCORER_VERSION,
    T0,
    THRESHOLD_COMMITMENT,
    _manifest,
    _packet,
)

HOTKEY = "hk-removal-winner"
RULES = {"crown_margin": 0.05, "crown_min_score": 0.70, "podium_min_score": 0.30}


def test_baseline_free_removal_result_is_decided_by_absolute_bars_and_audits_clean(
    tmp_path: Path,
) -> None:
    tokenomics = TokenomicsConfig(competition_emissions_enabled=True)
    store = make_store(
        AuditConfig(backend="local", local_root=tmp_path / "audit", allow_plaintext_holdout=True)
    )
    input_ref = store.put(b"served-input-with-mask", ArtifactKind.CHALLENGE_INPUT)
    reference_ref = store.put(b"clean-reference", ArtifactKind.REFERENCE_ORIGINAL)
    output_ref = store.put(b"contender-repaired-output", ArtifactKind.MINER_OUTPUT)
    archive_ref = store.put(b"sealed-contender-source-archive", ArtifactKind.SUBMISSION_ARCHIVE)

    with_baseline = _manifest(
        reference_ref.digest,
        input_ref.digest,
        baseline_artifact_digest="9" * 64,
        baseline_artifact_bytes=1,
        baseline_provenance_digest="8" * 64,
        baseline_provenance_bytes=1,
    )
    manifest = CompetitionManifest.model_validate(
        {**with_baseline.model_dump(), "baseline": None, "result_rules": RULES}
    )
    manifest_ref = store.put(manifest.canonical_json().encode("utf-8"), ArtifactKind.MANIFEST)
    binding = CompetitionItemBinding(
        item_index=0,
        input_sha256=input_ref.digest,
        reference_sha256=reference_ref.digest,
        mask_stream_index=1,
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
    item_id = repo.add_evaluation_item(
        conn, COMPETITION_ID, item_index=0,
        input_sha256=input_ref.digest, input_bytes=input_ref.byte_size,
        reference_sha256=reference_ref.digest, reference_bytes=reference_ref.byte_size,
        threshold_commitment=THRESHOLD_COMMITMENT, challenge_id=CHALLENGE_ID, now=T0,
    )
    packet_bytes, metrics, breakdown = _packet(
        item_id=input_ref.digest, hotkey=HOTKEY, output_digest=output_ref.digest,
        region_psnr_db=23.0,
    )
    packet_ref = store.put(packet_bytes, ArtifactKind.SCORE_PACKET)
    bundle = build_bundle(
        challenge_id=CHALLENGE_ID, item_id=input_ref.digest, miner_hotkey=HOTKEY,
        commitment_hash=THRESHOLD_COMMITMENT, stage=LifecycleStage.COMPETITION_SEALED,
        challenge_input=input_ref, reference_original=reference_ref,
        miner_output=output_ref, manifest=manifest_ref, score_packet=packet_ref,
        competition_item=binding, execution_image_digest="3" * 64,
        scorer_version=SCORER_VERSION,
        backend_versions={"removal_metrics": "removal-metrics/2"},
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

    # the published bytes re-parse to the same log (null baseline, removal items)
    reparsed = EpochLog.from_json(log.to_json())
    assert reparsed.log_digest() == log.log_digest()
    assert reparsed.audit_manifest.competition_input.items[0].mask_stream_index == 1

    source = InMemoryBundleSource()
    source.add(bundle)
    recomputer = StaticRecomputer(
        metrics, SCORER_VERSION, score=float(metrics["final_score"]),
        gate_passed=True, breakdown=breakdown,
    )
    audit = Auditor(
        AuditorConfig(auditor_hotkey="auditor-baseline-free", tokenomics=tokenomics, burn_uid=94),
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


def test_below_the_crown_bar_is_a_podium_and_below_the_podium_bar_pays_nobody() -> None:
    from datetime import datetime, timezone

    from vidaio.tokenomics.breakthrough import resolve_reward_window
    from vidaio.tokenomics.state import (
        CompetitionResult,
        CompetitionRules,
        ContenderResult,
        RewardWindowState,
    )

    config = TokenomicsConfig(competition_emissions_enabled=True)
    rules = CompetitionRules(crown_margin=0.05, crown_min_score=0.70, podium_min_score=0.30)

    def result(scores: list[float]) -> CompetitionResult:
        return CompetitionResult(
            competition_id="c", track="removal", cycle=1,
            applied_at=datetime(2026, 10, 4, tzinfo=timezone.utc),
            contenders=tuple(
                ContenderResult(hotkey=f"hk{i}", uid=i, score=s) for i, s in enumerate(scores)
            ),
            baseline_score=None, baseline_version=None, baseline_artifact_digest=None,
            rules=rules,
        )

    podium = resolve_reward_window(config, RewardWindowState(), result([0.65, 0.5, 0.2]))
    assert podium.kind is EmissionState.PODIUM
    assert podium.podium_hotkeys == ("hk0", "hk1")  # 0.2 is below podium_min_score
    nobody = resolve_reward_window(config, RewardWindowState(), result([0.25]))
    assert nobody.kind is EmissionState.IDLE  # v3 default: the result closes the window
