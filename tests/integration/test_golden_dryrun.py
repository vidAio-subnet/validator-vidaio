"""Golden dry-run: the honest end-to-end path across all five foundation modules.

One deterministic in-memory run (built once in integration_support.build_golden_world)
is asserted stage by stage: challenge -> scoring -> competition -> audit -> tokenomics.
The run is chronologically genuine: the pre-commitment is anchored while SCHEDULED
(before enrollment opens), scores enter only as packet bytes, the challenge resolves
before the asset's commitment is revealed, and bundles verify STRICT with every anchor
and the deep reveal verifier wired in.
"""

from __future__ import annotations

import json

import pytest

from vidaio.audit import (
    CommitmentStatus,
    ScorePacketShape,
    merkle_proof,
    sha256_hex,
    verify_bundle,
    verify_merkle_proof,
)
from vidaio.challenge import get_asset, provenance_log, verify_reveal, verify_reveal_deep
from vidaio.competition import Phase, verify_review_chain
from vidaio.competition import repository as comp_repo
from vidaio.scoring import (
    TRACK_COMPRESSION,
    GateContext,
    ItemScore,
    ReasonCode,
    ScoringConfig,
    compose_item_score,
    default_pipeline,
    score_compression,
)
from vidaio.tokenomics import (
    MinerSnapshot,
    build_weight_vector,
)

from integration_support import (
    COMPETITION_ID,
    GOOD_DATA,
    PRIVATE_SEED,
    REF_DATA,
    SCORER_VERSION,
    GoldenWorld,
    _media_info,
    make_backend,
    packet_metrics,
)


# ---- (a) challenge -----------------------------------------------------------------


def test_dispatch_payload_is_clean(world: GoldenWorld) -> None:
    payload = world.challenge.dispatch.model_dump(mode="json")
    assert set(payload) == {"challenge_id", "task_type", "input_ref"}
    assert payload["task_type"] == "compression"
    assert payload["input_ref"] == f"challenges/{world.challenge.challenge_id}/input.mp4"
    text = json.dumps(payload)
    # nothing validator-private leaks to the miner
    assert str(PRIVATE_SEED) not in text
    assert world.asset_id not in text
    assert world.challenge.commitment.dag_digest not in text


def test_ingest_confirmed_and_metadata_stripped(world: GoldenWorld) -> None:
    # register_asset records plans only; confirm_ingest_step recorded the completion
    # facts — including the metadata strip the transcode step implies.
    asset = get_asset(world.challenge_conn, world.asset_id)
    assert asset.metadata_stripped
    events = [row["event"] for row in provenance_log(world.challenge_conn, world.asset_id)]
    for step in ("fetch_completed", "transcode_completed", "segment_completed"):
        assert step in events
    assert events.index("transcode_planned") < events.index("transcode_completed")
    assert "metadata_stripped" in events


def test_commitment_recorded_before_challenge(world: GoldenWorld) -> None:
    conn = world.challenge_conn
    commitment = conn.execute(
        "SELECT * FROM challenge_commitments WHERE commit_hash = ?",
        (world.challenge.commitment.commit_hash,),
    ).fetchone()
    assert commitment is not None
    assert commitment["clean_asset_id"] == world.asset_id
    assert int(commitment["seed"]) == PRIVATE_SEED
    challenge_row = conn.execute(
        "SELECT * FROM challenges WHERE challenge_id = ?",
        (world.challenge.challenge_id,),
    ).fetchone()
    assert challenge_row is not None
    assert challenge_row["commit_hash"] == commitment["commit_hash"]
    assert challenge_row["dag_digest"] == world.challenge.commitment.dag_digest


def test_challenge_resolved_before_reveal(world: GoldenWorld) -> None:
    row = world.challenge_conn.execute(
        "SELECT status, resolved_at FROM challenges WHERE challenge_id = ?",
        (world.challenge.challenge_id,),
    ).fetchone()
    assert row["status"] == "resolved"
    assert row["resolved_at"] is not None
    # reveal happened strictly after resolution and retirement
    assert world.revealed.revealed_at > row["resolved_at"]


def test_holdout_split_never_issued(world: GoldenWorld) -> None:
    holdout = get_asset(world.challenge_conn, world.holdout_asset_id)
    assert holdout.split == "holdout"
    assert holdout.status == "fresh"  # untouched by checkout
    assert world.asset_id != world.holdout_asset_id


# ---- (b) scoring -------------------------------------------------------------------


def test_good_response_scores_by_the_spec_formula(world: GoldenWorld) -> None:
    good = world.item_scores["hk-a"]
    assert good.gate_passed
    assert good.violations == []
    # rate 0.5 -> comp 0.5; vmaf 93 -> quality 0.93; (0.7*0.5 + 0.3*0.93)/1.12
    assert good.score == pytest.approx((0.7 * 0.5 + 0.3 * 0.93) / 1.12)
    assert good.breakdown is not None and good.breakdown.zero_reason is None
    assert good.scorer_version == "scorer-v1"
    # nothing was consciously disabled on the golden path: no gate skips recorded
    assert good.skips == []


def test_gate_failed_response_zeroes(world: GoldenWorld) -> None:
    bad = world.item_scores["hk-b"]
    assert not bad.gate_passed
    assert bad.score == 0.0
    codes = {v.code for v in bad.violations}
    assert ReasonCode.COMPRESSION_RATE_TOO_HIGH in codes
    assert bad.breakdown is not None
    assert bad.breakdown.zero_reason == ReasonCode.COMPRESSION_RATE_TOO_HIGH
    assert bad.breakdown.compression_rate == pytest.approx(0.9)


def test_duplicate_response_zeroes_via_dedup(world: GoldenWorld) -> None:
    dup = world.item_scores["hk-e"]
    assert not dup.gate_passed
    assert dup.score == 0.0
    assert ReasonCode.REPLAY_DUPLICATE in {v.code for v in dup.violations}
    # the replay is byte-identical to hk-a's output
    assert dup.content_digest == world.item_scores["hk-a"].content_digest


def test_baseline_calibration_scores_without_identity(world: GoldenWorld) -> None:
    baseline = world.baseline_item_score
    assert baseline.miner_hotkey is None
    assert baseline.gate_passed
    # rate 0.56 -> comp 0.44; vmaf 91 -> (0.7*0.44 + 0.3*0.91)/1.12
    assert baseline.score == pytest.approx((0.7 * 0.44 + 0.3 * 0.91) / 1.12)
    # The winner genuinely out-scored the reference baseline on the same item.
    assert world.item_scores["hk-a"].score > baseline.score


def test_packets_roundtrip_json(world: GoldenWorld) -> None:
    for hk, packet in world.packets.items():
        restored = ItemScore.from_json(packet.decode())
        assert restored == world.item_scores[hk]
        # audit-parseable: metrics present and numeric-only
        metrics = json.loads(packet)["metrics"]
        assert metrics and all(isinstance(v, (int, float)) for v in metrics.values())


def test_gate_skips_flow_from_packet_to_audit_parse() -> None:
    """A config-disabled check (require_secondary_vmaf=False) records a GateSkip
    that flows GateContext -> compose_item_score -> ItemScore JSON -> the audit
    module's ScorePacketShape parse — the skip is auditable, never silently lost."""
    config = ScoringConfig(require_secondary_vmaf=False)
    backend = make_backend()
    ref_digest = sha256_hex(REF_DATA)
    cand_digest = sha256_hex(GOOD_DATA)
    ctx = GateContext(
        track=TRACK_COMPRESSION,
        config=config,
        reference_info=_media_info("ffv1", len(REF_DATA)),
        candidate_info=_media_info("h264", len(GOOD_DATA)),
        reference_path=ref_digest,
        candidate_path=cand_digest,
        vmaf_primary=backend.compute(ref_digest, cand_digest),
        vmaf_secondary=None,  # absent run: a skip (config off), never a silent pass
    )
    passed, violations = default_pipeline(backend).run(ctx)
    assert passed and violations == []
    assert [s.gate for s in ctx.skips] == ["vmaf_model_delta"]

    breakdown = score_compression(
        candidate_bytes=len(GOOD_DATA),
        reference_bytes=len(REF_DATA),
        vmaf=ctx.vmaf_primary,
        config=config,
    )
    item = compose_item_score(
        item_id=cand_digest,
        challenge_id="chal-skips",
        track=TRACK_COMPRESSION,
        gate_passed=passed,
        violations=violations,
        skips=ctx.skips,
        breakdown=breakdown,
        config=config,
        content_digest=cand_digest,
        metrics=packet_metrics(breakdown),
        backend_versions=backend.versions(),
        scorer_version=SCORER_VERSION,
    )
    assert [s.gate for s in item.skips] == ["vmaf_model_delta"]
    # lossless scoring-side JSON roundtrip
    assert ItemScore.from_json(item.to_json()).skips == item.skips
    # the audit-side packet contract carries the skips through unchanged
    shape = ScorePacketShape.model_validate(json.loads(item.to_json()))
    assert len(shape.skips) == 1
    assert shape.skips[0]["gate"] == "vmaf_model_delta"
    assert "require_secondary_vmaf=False" in shape.skips[0]["detail"]


# ---- (c) competition ---------------------------------------------------------------


def test_lifecycle_reached_completed(world: GoldenWorld) -> None:
    comp = comp_repo.get_competition(world.comp_conn, COMPETITION_ID)
    assert comp is not None and comp.status is Phase.COMPLETED
    transitions = [
        (row["from_phase"], row["to_phase"])
        for row in comp_repo.list_events(world.comp_conn, COMPETITION_ID)
        if row["event_type"] == "transition"
    ]
    assert transitions == [
        ("SCHEDULED", "ENROLLING"),
        ("ENROLLING", "FINALIZING_SUBMISSIONS"),
        ("FINALIZING_SUBMISSIONS", "VALIDATING"),
        ("VALIDATING", "BUILDING"),
        ("BUILDING", "EVALUATING"),
        ("EVALUATING", "SCORING"),
        ("SCORING", "AWAITING_END_TIME"),
        ("AWAITING_END_TIME", "COMPLETED"),
    ]
    assert verify_review_chain(world.comp_conn, COMPETITION_ID)


def test_commitment_anchored_before_enrollment_chronologically(world: GoldenWorld) -> None:
    """review #4: the anchoring event precedes the enrolling transition in BOTH the
    event order and real (fake-clock) time — no backdating after the fact."""
    events = comp_repo.list_events(world.comp_conn, COMPETITION_ID)
    kinds = [row["event_type"] for row in events]
    anchored_idx = kinds.index("commitment_anchored")
    enrolling_idx = next(
        i
        for i, row in enumerate(events)
        if row["event_type"] == "transition" and row["to_phase"] == "ENROLLING"
    )
    assert anchored_idx < enrolling_idx
    assert events[anchored_idx]["created_at"] < events[enrolling_idx]["created_at"]
    # the whole event log is chronologically monotonic — nothing was backdated
    stamps = [row["created_at"] for row in events]
    assert stamps == sorted(stamps)
    # the anchored root is the REAL competition commitment recorded in the ledger
    comp = comp_repo.get_competition(world.comp_conn, COMPETITION_ID)
    assert comp is not None and comp.commitment_root == world.pre_commitment.root
    payload = json.loads(events[anchored_idx]["payload_json"])
    assert payload["commitment_root"] == world.pre_commitment.root


def test_backup_guard_carries_audit_store_evidence(world: GoldenWorld) -> None:
    events = comp_repo.list_events(world.comp_conn, COMPETITION_ID)
    backup = next(
        row
        for row in events
        if row["event_type"] == "transition" and row["to_phase"] == "VALIDATING"
    )
    payload = json.loads(backup["payload_json"])
    assert payload["backup_ref"] == world.backup_ref
    assert len(world.backup_ref) == 64  # a real audit-store artifact digest


def test_podium_excludes_calibration(world: GoldenWorld) -> None:
    podium = comp_repo.podium(world.comp_conn, COMPETITION_ID)
    assert [c.hotkey for c in podium] == ["hk-a", "hk-b"]
    assert all(not c.is_calibration for c in podium)
    baseline = next(
        c for c in comp_repo.list_contenders(world.comp_conn, COMPETITION_ID) if c.is_calibration
    )
    # The baseline is scored for calibration but never ranked or attributed.
    assert baseline.final_score is not None
    assert baseline.final_rank is None
    assert baseline.hotkey is None


def test_item_scores_are_packet_bound_and_audit_linked(world: GoldenWorld) -> None:
    rows = {
        row["contender_id"]: row
        for row in world.comp_conn.execute(
            "SELECT * FROM performance_history WHERE competition_id = ?",
            (COMPETITION_ID,),
        )
    }
    for hk in ("hk-a", "hk-b"):
        row = rows[world.contender_ids[hk]]
        # score/validity/digest all derive from the recorded packet bytes
        assert row["score_packet_digest"] == world.packet_refs[hk].digest
        assert row["item_score"] == world.item_scores[hk].score
        assert bool(row["valid"]) == world.item_scores[hk].gate_passed
        # per-(contender, item) audit bundle linkage
        assert row["audit_bundle_digest"] == world.bundle_digests[hk]
    baseline_row = rows[world.baseline_contender_id]
    assert baseline_row["score_packet_digest"] == world.baseline_packet_ref.digest
    # The calibration row is audit-linked exactly like a miner's row.
    assert baseline_row["audit_bundle_digest"] == world.baseline_bundle_digest


def test_completion_stalls_until_baseline_bundle_linked(world: GoldenWorld) -> None:
    """The audit-linkage completion gate, observed on the golden run itself: the
    tick at end_time found the baseline calibration row unlinked and applied NOTHING
    (the run stayed in AWAITING_END_TIME); only after set_audit_bundle_digest for
    the baseline row did the re-tick complete the competition."""
    assert world.stall_gaps == [(world.baseline_contender_id, world.item_ids[0])]
    # the due tick deferred: no transition applied, phase held
    assert world.stalled_transitions == []
    assert world.phase_during_stall is Phase.AWAITING_END_TIME
    # After baseline linkage the gate opened and the run completed for real.
    comp = comp_repo.get_competition(world.comp_conn, COMPETITION_ID)
    assert comp is not None and comp.status is Phase.COMPLETED
    assert world.engine.audit_linkage_gaps(world.comp_conn, COMPETITION_ID) == []


# ---- (d) audit ---------------------------------------------------------------------


def test_asset_retired_and_reveal_verifies_deep(world: GoldenWorld) -> None:
    asset = get_asset(world.challenge_conn, world.asset_id)
    assert asset.status == "retired"
    assert verify_reveal(world.revealed)
    # deep check: the committed DAG genuinely regenerates from the revealed seed
    assert verify_reveal_deep(world.revealed)
    assert world.revealed.seed == PRIVATE_SEED
    assert world.revealed.dag_digest == world.challenge.dag.canonical_digest()


def test_bundles_fully_verify_strict_against_real_recompute(world: GoldenWorld) -> None:
    recomputer = world.recomputer()
    for hk, bundle in world.bundles.items():
        report = verify_bundle(
            bundle, world.store, recomputer, **world.verify_kwargs(hk)
        )
        assert report.strict  # default strict=True: skipped anchors would fail
        assert report.passed, f"{hk}: {[c.model_dump() for c in report.failures()]}"
        assert report.skips() == []  # every anchor + the deep verifier were supplied
        names = {c.name for c in report.checks}
        assert {
            "stage_recomputable",
            "bundle_digest",
            "commitment_reveal",
            "dag_reveal_generation",
            "merkle_inclusion",
            "metric_set",
            "score_recompute:final_score",
            "score_recompute:score",
            "gate_recompute",
            "packet_consistency",
        } <= names
        # every miner bundle is attributed: the packet's hotkey is pinned
        assert bundle.miner_hotkey == hk


def test_baseline_bundle_verifies_strict_without_identity(world: GoldenWorld) -> None:
    """The unattributed baseline bundle is independently recomputable."""
    bundle = world.baseline_bundle
    assert bundle.miner_hotkey is None
    assert json.loads(world.baseline_packet)["miner_hotkey"] is None
    report = verify_bundle(
        bundle, world.store, world.recomputer(), **world.verify_kwargs("baseline")
    )
    assert report.strict
    assert report.passed, [c.model_dump() for c in report.failures()]
    assert report.skips() == []  # every anchor + the deep verifier were supplied
    # the identity check ran (hotkey comparison skipped for unattributed bundles)
    # The baseline packet is a leaf of the same published Merkle root.
    assert "packet_identity" in {c.name for c in report.checks}
    assert bundle.score_packet.digest in world.packet_digests


def test_publication_record_and_ledger(world: GoldenWorld) -> None:
    # Every score packet, including calibration, is in the published root.
    assert world.baseline_packet_ref.digest in world.packet_digests
    for digest in world.packet_digests:
        proof = merkle_proof(world.packet_digests, digest)
        assert verify_merkle_proof(digest, proof, world.published_root)
    # chain payloads: domain-tagged, anchorable (<= 128 bytes)
    assert world.publication.kind == "publication"
    assert len(world.publication.payload) <= 128
    assert world.pre_commitment.kind == "competition"
    # ledger status history is append-only and forward-only
    assert world.ledger.current_status(world.pre_commitment_id) is CommitmentStatus.ANCHORED
    assert world.ledger.current_status(world.publication_id) is CommitmentStatus.PUBLISHED
    assert [s for s, _ in world.ledger.history(world.publication_id)] == [
        "pending_chain", "anchored", "published",
    ]
    # chronology: recorded -> anchored -> published timestamps advance
    history = world.ledger.history(world.publication_id)
    assert [at for _, at in history] == sorted(at for _, at in history)


# ---- (e) tokenomics ----------------------------------------------------------------
# These golden cases supply no earning competition result. Dedicated schema-v14
# integration tests cover PODIUM/CROWN allocation and recomputation.


def test_idle_weight_vector_has_inference_and_canonical_sink(world: GoldenWorld) -> None:
    vector = world.weights
    assert sum(vector.values()) == pytest.approx(1.0)
    # IDLE is 80% inference and 20% canonical sink. Inference remains split
    # compression/upscaling 0.8/0.2 and uses the 5:4 rank curve here.
    assert vector[0] == pytest.approx(0.20)
    assert vector[10] == pytest.approx(0.80 * 0.8 * 5 / 9)
    assert vector[11] == pytest.approx(0.80 * 0.8 * 4 / 9)
    assert vector[12] == pytest.approx(0.80 * 0.2 * 5 / 9)
    assert vector[13] == pytest.approx(0.80 * 0.2 * 4 / 9)


def test_baseline_has_no_identity_anywhere(world: GoldenWorld) -> None:
    assert world.baseline_final_score is not None
    # The only extra uid is the configured canonical sink, not a baseline identity.
    assert set(world.weights) == {0, *(m.uid for m in world.miners)}


def test_ineligible_miner_takes_nothing(world: GoldenWorld) -> None:
    assert world.weights[14] == 0.0  # duplicate IP of uid 11: deduped out of ranking


# (test_retention_lever_reshapes_within_track REMOVED with the retention multiplier for v1
# — retention removed — owner decision; an internal review — rank weight
# no longer depends on any windowed retention input.)


def test_no_inference_eligible_miners_routes_everything_to_sink(world: GoldenWorld) -> None:
    """No eligible miner never creates a normalizable partial/empty vector."""
    cfg = world.tokenomics_config

    def _idle(uid: int, hotkey: str, ip: str) -> MinerSnapshot:
        # zero accumulated score: known to the metagraph, but inference-ineligible
        return MinerSnapshot(
            uid=uid, hotkey=hotkey, coldkey=f"ck-{uid}", ip=ip, track="compression",
            accumulate_score=0.0,
        )

    idle_fleet = [_idle(10, "hk-a", "10.0.0.1"), _idle(11, "hk-b", "10.0.0.2")]
    vector = build_weight_vector(cfg, idle_fleet, burn_uid=0)
    assert vector == {10: 0.0, 11: 0.0, 0: 1.0}
