from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import json

import pytest

from vidaio.audit import (
    ArtifactKind,
    ArtifactRef,
    backend_key,
    make_public_store,
)
from vidaio.auditor import Auditor, AuditorConfig, InMemoryBundleSource
from vidaio.auditor.report import (
    COMPETITION_MISMATCH,
    COMPETITION_UNVERIFIED,
    ItemVerdictKind,
)
from vidaio.authority.finalizer import (
    AuditFileMissingError,
    EpochFinalizer,
    build_audit_manifest,
    epoch_prefix,
)
from vidaio.authority import EpochIndex, ScoringAuthority
from vidaio.competition.epoch_evidence import (
    CompetitionEvidenceError,
    build_competition_epoch_evidence,
)
from vidaio.competition import repository as competition_repo
from vidaio.competition.orchestrator.persistence import EVENT_COMMITMENT_ANCHORED
from vidaio.competition.economic_result import derive_competition_result
from vidaio.chain.adapter import InMemoryChain
from vidaio.epoch import AuditManifest, EpochLogInvalid, MinerCensusEntry
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import MinerSnapshot

from integration_support import COMPLETED_AT, GoldenWorld


def _competition_miners() -> tuple[MinerSnapshot, ...]:
    return (
        MinerSnapshot(
            uid=10,
            hotkey="hk-a",
            coldkey="ck-a",
            ip="203.0.113.10",
            track="compression",
            accumulate_score=0.0,
        ),
        MinerSnapshot(
            uid=11,
            hotkey="hk-b",
            coldkey="ck-b",
            ip="203.0.113.11",
            track="compression",
            accumulate_score=0.0,
        ),
    )


def _competition_census(*hotkeys: str) -> dict[str, MinerCensusEntry]:
    by_hotkey = {miner.hotkey: miner for miner in _competition_miners()}
    return {
        hotkey: MinerCensusEntry.from_miner(by_hotkey[hotkey]) for hotkey in hotkeys
    }


class _InferenceCommitmentMustNotHandleCompetition:
    def committed_dispatch(self, challenge_id: str):
        raise AssertionError(
            f"competition challenge {challenge_id} reached inference commitment source"
        )


def _append_receipt(
    world: GoldenWorld, *, update: dict[str, object], remove: tuple[str, ...] = ()
) -> None:
    row = world.comp_conn.execute(
        "SELECT payload_json FROM events WHERE competition_id = ? AND event_type = ? "
        "ORDER BY event_id DESC LIMIT 1",
        (world.manifest.competition_id, EVENT_COMMITMENT_ANCHORED),
    ).fetchone()
    assert row is not None
    payload = json.loads(row["payload_json"])
    payload.update(update)
    for name in remove:
        payload.pop(name, None)
    competition_repo.record_event(
        world.comp_conn,
        world.manifest.competition_id,
        EVENT_COMMITMENT_ANCHORED,
        COMPLETED_AT,
        payload=payload,
    )


def test_authority_evidence_refuses_incomplete_persisted_anchor_receipt(
    fresh_world: GoldenWorld,
) -> None:
    _append_receipt(fresh_world, update={}, remove=("anchor_block_hash",))

    with pytest.raises(CompetitionEvidenceError, match="receipt is incomplete"):
        build_competition_epoch_evidence(
            fresh_world.comp_conn,
            census_by_hotkey=_competition_census("hk-a", "hk-b"),
            store=fresh_world.store,
            through_time=COMPLETED_AT,
        )


def test_authority_evidence_refuses_root_mismatched_persisted_anchor_receipt(
    fresh_world: GoldenWorld,
) -> None:
    _append_receipt(fresh_world, update={"root": "0" * 64})

    with pytest.raises(CompetitionEvidenceError, match="receipt root does not match"):
        build_competition_epoch_evidence(
            fresh_world.comp_conn,
            census_by_hotkey=_competition_census("hk-a", "hk-b"),
            store=fresh_world.store,
            through_time=COMPLETED_AT,
        )


def test_scoring_authority_independently_refuses_archive_payload_mismatch(
    fresh_world: GoldenWorld, tmp_path
) -> None:
    cfg = TokenomicsConfig(competition_emissions_enabled=True)
    evidence = build_competition_epoch_evidence(
        fresh_world.comp_conn,
        census_by_hotkey=_competition_census("hk-a", "hk-b"),
        store=fresh_world.store,
        tokenomics=cfg,
        through_time=COMPLETED_AT,
    )
    assert evidence is not None
    manifest = build_audit_manifest(
        evidence.scored_items,
        store=fresh_world.store,
        competition_input=evidence.competition_input,
    )
    receipt = evidence.competition_input
    wrong_chain = InMemoryChain(
        _block=fresh_world.anchor_chain.current_block(),
        anchored=[b"readable-but-wrong-authority-archive-record"],
        _anchor_blocks=[receipt.anchor_block],
        block_time_anchor=(
            receipt.anchor_block,
            fresh_world.anchor_chain.block_time(receipt.anchor_block),
        ),
    )
    index = EpochIndex.open(tmp_path / "authority-anchor-check.db")
    service = ScoringAuthority(
        {
            "core": {"metrics_port": 0},
            "authority": {
                "http_host": "127.0.0.1",
                "http_port": 0,
                "metrics_port": 0,
                "netuid": 85,
                "burn_uid": 94,
                "scorer_version": "scorer-v1",
            },
        },
        metrics_port=0,
        store=fresh_world.store,
        public_store=make_public_store(fresh_world.audit_config),
        chain=wrong_chain,
        index=index,
        finalizer=EpochFinalizer(cfg, scorer_version="scorer-v1"),
    )
    try:
        with pytest.raises(EpochLogInvalid, match="independently proven"):
            service._verify_competition_anchor(manifest, epoch_close_block=359)
    finally:
        service.close()


def test_completed_competition_drives_packet_derived_epoch_emissions(
    fresh_world: GoldenWorld,
) -> None:
    world = fresh_world
    evidence = build_competition_epoch_evidence(
        world.comp_conn,
        census_by_hotkey=_competition_census("hk-a", "hk-b"),
        store=world.store,
        through_time=COMPLETED_AT,
    )
    assert evidence is not None
    assert {subject.subject_id for subject in evidence.competition_input.subjects} == {
        "baseline",
        "contender:hk-a",
        "contender:hk-b",
    }
    assert len(evidence.scored_items) == 3
    assert set(evidence.packet_scores) == {
        world.packet_refs["hk-a"].digest,
        world.packet_refs["hk-b"].digest,
        world.baseline_packet_ref.digest,
    }

    # Manual-review state is deliberately not an economic input.  Even if the DB
    # review columns are changed, exact packet-derived ordering stays unchanged.
    before = evidence.result
    world.comp_conn.execute(
        "UPDATE contenders SET manual_disqualified = 1, eligible = 0 "
        "WHERE competition_id = ? AND hotkey = 'hk-a'",
        (evidence.competition_input.competition_id,),
    )
    after = build_competition_epoch_evidence(
        world.comp_conn,
        census_by_hotkey=_competition_census("hk-a", "hk-b"),
        store=world.store,
        through_time=COMPLETED_AT,
    )
    assert after is not None
    assert after.result == before

    manifest = build_audit_manifest(
        evidence.scored_items,
        store=world.store,
        competition_input=evidence.competition_input,
        commitment_source=_InferenceCommitmentMustNotHandleCompetition(),
    )
    cfg = TokenomicsConfig(competition_emissions_enabled=True)
    log = EpochFinalizer(cfg, scorer_version="scorer-v1").build_log(
        epoch_id=1,
        close_block=359,
        snapshots=_competition_miners(),
        burn_uid=94,
        audit_manifest=manifest,
        now=COMPLETED_AT,
        competition_result=evidence.result,
        competition_packet_scores=evidence.packet_scores,
    )
    assert log.competition_result == evidence.result
    assert log.weight_shares[10] > log.weight_shares[11] > 0.0
    assert log.weight_shares[94] > 0.0

    # Both paid contenders have a zero inference EWMA. Their positive weights are
    # backed by the exact competition packets/result above, so the inference
    # earning pass must not misclassify them as unaudited genesis carry-forwards.
    auditor = Auditor(
        AuditorConfig(
            auditor_hotkey="auditor-test", tokenomics=cfg, burn_uid=94
        ),
        InMemoryBundleSource(),
    )
    assert auditor._earning_verdicts(log, world.store, None, True) == []


def test_registered_but_non_economic_podium_rank_routes_to_sink(
    fresh_world: GoldenWorld,
) -> None:
    """A fresh podium identity needs census binding, not inference eligibility.

    A BUILT contender can remain registered at the close block while being absent
    from the narrower economic snapshot (for example, no currently resolved
    inference track).  The competition result must stay complete so the authority
    cannot cherry-pick the ranked field; its missing payout share goes to the sink.
    """
    world = fresh_world
    census_by_hotkey = _competition_census("hk-a", "hk-b")
    evidence = build_competition_epoch_evidence(
        world.comp_conn,
        census_by_hotkey=census_by_hotkey,
        store=world.store,
        through_time=COMPLETED_AT,
    )
    assert evidence is not None
    manifest = build_audit_manifest(
        evidence.scored_items,
        store=world.store,
        competition_input=evidence.competition_input,
        commitment_source=_InferenceCommitmentMustNotHandleCompetition(),
    )
    cfg = TokenomicsConfig(competition_emissions_enabled=True)
    finalizer = EpochFinalizer(cfg, scorer_version="scorer-v1")
    all_miners = _competition_miners()
    missing_rank = evidence.result.contenders[1]
    retained = tuple(
        miner for miner in all_miners if miner.hotkey != missing_rank.hotkey
    )
    full_census = tuple(census_by_hotkey.values())

    full = finalizer.build_log(
        epoch_id=1,
        close_block=359,
        snapshots=all_miners,
        miner_census=full_census,
        burn_uid=94,
        audit_manifest=manifest,
        now=COMPLETED_AT,
        competition_result=evidence.result,
        competition_packet_scores=evidence.packet_scores,
    )
    partial = finalizer.build_log(
        epoch_id=1,
        close_block=359,
        snapshots=retained,
        miner_census=full_census,
        burn_uid=94,
        audit_manifest=manifest,
        now=COMPLETED_AT,
        competition_result=evidence.result,
        competition_packet_scores=evidence.packet_scores,
    )

    competition_pool = 0.9 if partial.reward_window_state.kind.value == "CROWN" else 0.4
    assert full.weight_shares[missing_rank.uid] == pytest.approx(
        competition_pool * 0.20
    )
    assert missing_rank.uid not in partial.weight_shares
    assert partial.weight_shares[94] - full.weight_shares[94] == pytest.approx(
        competition_pool * 0.20
    )

    # Registration is still mandatory. Omitting the contender from the complete
    # census is a malformed/cherry-picked result, not another sink case.
    with pytest.raises(EpochLogInvalid, match="absent from the complete close-block"):
        finalizer.build_log(
            epoch_id=1,
            close_block=359,
            snapshots=retained,
            miner_census=(MinerCensusEntry.from_miner(retained[0]),),
            burn_uid=94,
            audit_manifest=manifest,
            now=COMPLETED_AT,
            competition_result=evidence.result,
            competition_packet_scores=evidence.packet_scores,
        )

    # The committed hotkey is the reward recipient across the finite window;
    # winner_uid remains source-result provenance. If that same hotkey later
    # occupies a new registered uid, composition pays the new slot and the
    # weight-setter binds it to the new close-block census identity.
    winner = evidence.result.contenders[0]
    moved = MinerSnapshot(
        uid=42,
        hotkey=winner.hotkey,
        coldkey="ck-moved",
        ip="203.0.113.42",
        track="compression",
        accumulate_score=0.0,
    )
    carried = finalizer.build_log(
        epoch_id=2,
        close_block=719,
        snapshots=(moved,),
        miner_census=(MinerCensusEntry.from_miner(moved),),
        burn_uid=94,
        audit_manifest=AuditManifest(),
        now=COMPLETED_AT + timedelta(minutes=1),
        prior_reward_window_state=partial.reward_window_state,
    )
    assert carried.reward_window_state.winner_uid == winner.uid
    assert carried.reward_window_state.winner_hotkey == moved.hotkey
    assert carried.weight_shares[moved.uid] == pytest.approx(competition_pool * 0.70)


def test_authority_refuses_crown_until_winner_archive_is_publicly_readable(
    fresh_world: GoldenWorld,
) -> None:
    world = fresh_world
    cfg = TokenomicsConfig(competition_emissions_enabled=True)
    # Omitting tokenomics here deliberately leaves every submission sealed.  The
    # finalizer must independently refuse the otherwise-valid CROWN result.
    evidence = build_competition_epoch_evidence(
        world.comp_conn,
        census_by_hotkey=_competition_census("hk-a", "hk-b"),
        store=world.store,
        through_time=COMPLETED_AT,
    )
    assert evidence is not None
    assert evidence.result.baseline_score is not None
    assert evidence.result.contenders[0].score >= (
        evidence.result.baseline_score * 1.05
    )
    manifest = build_audit_manifest(
        evidence.scored_items,
        store=world.store,
        competition_input=evidence.competition_input,
    )
    finalizer = EpochFinalizer(cfg, scorer_version="scorer-v1")
    public = make_public_store(world.audit_config)
    kwargs = {
        "epoch_id": 20,
        "close_block": 7_199,
        "snapshots": _competition_miners(),
        "burn_uid": 94,
        "audit_manifest": manifest,
        "store": world.store,
        "public_store": public,
        "now": COMPLETED_AT,
        "competition_result": evidence.result,
        "competition_packet_scores": evidence.packet_scores,
    }

    with pytest.raises(AuditFileMissingError, match="not anonymously readable"):
        finalizer.finalize(**kwargs)
    assert world.store.is_finalized(epoch_prefix(20)) is False

    winner = evidence.result.contenders[0]
    winner_subject = next(
        subject
        for subject in evidence.competition_input.subjects
        if subject.role == "contender" and subject.hotkey == winner.hotkey
    )
    assert winner_subject.submission_archive_digest is not None
    assert winner_subject.submission_archive_bytes is not None
    winner_archive = ArtifactRef(
        digest=winner_subject.submission_archive_digest,
        kind=ArtifactKind.SUBMISSION_ARCHIVE,
        byte_size=winner_subject.submission_archive_bytes,
        backend_key=backend_key(
            ArtifactKind.SUBMISSION_ARCHIVE,
            winner_subject.submission_archive_digest,
        ),
    )
    world.store.release(winner_archive)

    credentialed_kwargs = {**kwargs, "public_store": world.store}
    with pytest.raises(AuditFileMissingError, match="credentialed/private"):
        finalizer.finalize(**credentialed_kwargs)
    assert world.store.is_finalized(epoch_prefix(20)) is False

    finalized = finalizer.finalize(**kwargs)
    assert finalized.log.reward_window_state.kind.value == "CROWN"
    assert public.is_released(winner_archive) is True
    loser_subject = next(
        subject
        for subject in evidence.competition_input.subjects
        if subject.role == "contender" and subject.hotkey != winner.hotkey
    )
    assert loser_subject.submission_archive_digest is not None
    assert loser_subject.submission_archive_bytes is not None
    loser_archive = ArtifactRef(
        digest=loser_subject.submission_archive_digest,
        kind=ArtifactKind.SUBMISSION_ARCHIVE,
        byte_size=loser_subject.submission_archive_bytes,
        backend_key=backend_key(
            ArtifactKind.SUBMISSION_ARCHIVE,
            loser_subject.submission_archive_digest,
        ),
    )
    assert public.is_released(loser_archive) is False


def test_podium_epoch_keeps_every_contender_source_sealed(
    fresh_world: GoldenWorld,
) -> None:
    world = fresh_world
    cfg = TokenomicsConfig(competition_emissions_enabled=True)
    evidence = build_competition_epoch_evidence(
        world.comp_conn,
        census_by_hotkey=_competition_census("hk-a", "hk-b"),
        store=world.store,
        through_time=COMPLETED_AT,
    )
    assert evidence is not None
    # Reuse the complete committed matrix but feed the pure finalizer derivation a
    # sub-threshold 4% winner.  Source release is not a scoring input, so all
    # contender archives must remain sealed for this PODIUM result.
    packet_scores: dict[str, float] = {}
    for subject in evidence.competition_input.subjects:
        score = 0.50 if subject.role == "baseline" else (
            0.52 if subject.hotkey == "hk-a" else 0.40
        )
        packet_scores.update(dict.fromkeys(subject.packet_digests, score))
    podium_result = derive_competition_result(
        evidence.competition_input,
        packet_scores,
    )
    manifest = build_audit_manifest(
        evidence.scored_items,
        store=world.store,
        competition_input=evidence.competition_input,
    )
    public = make_public_store(world.audit_config)

    finalized = EpochFinalizer(cfg, scorer_version="scorer-v1").finalize(
        epoch_id=21,
        close_block=7_559,
        snapshots=_competition_miners(),
        burn_uid=94,
        audit_manifest=manifest,
        store=world.store,
        public_store=public,
        now=COMPLETED_AT,
        competition_result=podium_result,
        competition_packet_scores=packet_scores,
    )

    assert finalized.log.reward_window_state.kind.value == "PODIUM"
    for subject in evidence.competition_input.subjects:
        if subject.role != "contender":
            continue
        assert subject.submission_archive_digest is not None
        assert subject.submission_archive_bytes is not None
        ref = ArtifactRef(
            digest=subject.submission_archive_digest,
            kind=ArtifactKind.SUBMISSION_ARCHIVE,
            byte_size=subject.submission_archive_bytes,
            backend_key=backend_key(
                ArtifactKind.SUBMISSION_ARCHIVE,
                subject.submission_archive_digest,
            ),
        )
        assert public.is_released(ref) is False


def test_competition_cannot_be_applied_before_its_completion_time(
    fresh_world: GoldenWorld,
) -> None:
    evidence = build_competition_epoch_evidence(
        fresh_world.comp_conn,
        census_by_hotkey=_competition_census("hk-a", "hk-b"),
        store=fresh_world.store,
        through_time=COMPLETED_AT,
    )
    assert evidence is not None
    assert (
        build_competition_epoch_evidence(
            fresh_world.comp_conn,
            census_by_hotkey=_competition_census("hk-a", "hk-b"),
            store=fresh_world.store,
            through_time=COMPLETED_AT - timedelta(seconds=1),
        )
        is None
    )
    manifest = build_audit_manifest(
        evidence.scored_items,
        store=fresh_world.store,
        competition_input=evidence.competition_input,
    )
    with pytest.raises(EpochLogInvalid, match="applied_at must equal"):
        EpochFinalizer(
            TokenomicsConfig(competition_emissions_enabled=True),
            scorer_version="scorer-v1",
        ).build_log(
            epoch_id=1,
            close_block=359,
            snapshots=_competition_miners(),
            burn_uid=94,
            audit_manifest=manifest,
            now=COMPLETED_AT - timedelta(seconds=1),
            competition_result=evidence.result,
            competition_packet_scores=evidence.packet_scores,
        )


def test_built_contender_missing_from_close_census_fails_closed(
    fresh_world: GoldenWorld,
) -> None:
    with pytest.raises(CompetitionEvidenceError, match="absent from the close-block census"):
        build_competition_epoch_evidence(
            fresh_world.comp_conn,
            census_by_hotkey=_competition_census("hk-a"),
            store=fresh_world.store,
            through_time=COMPLETED_AT,
        )


def test_authority_and_auditor_rederive_competition_coldkey_dedup(
    fresh_world: GoldenWorld,
) -> None:
    cfg = TokenomicsConfig(competition_emissions_enabled=True)
    census = _competition_census("hk-a", "hk-b")
    census["hk-b"] = census["hk-b"].model_copy(
        update={"coldkey": census["hk-a"].coldkey}
    )
    evidence = build_competition_epoch_evidence(
        fresh_world.comp_conn,
        census_by_hotkey=census,
        store=fresh_world.store,
        tokenomics=cfg,
        through_time=COMPLETED_AT,
    )
    assert evidence is not None
    by_hotkey = {
        subject.hotkey: subject
        for subject in evidence.competition_input.subjects
        if subject.role == "contender"
    }
    assert not by_hotkey["hk-a"].dedup_excluded
    assert by_hotkey["hk-b"].dedup_excluded
    assert [(entry.uid, entry.hotkey) for entry in evidence.result.contenders] == [
        (10, "hk-a")
    ]

    manifest = build_audit_manifest(
        evidence.scored_items,
        store=fresh_world.store,
        competition_input=evidence.competition_input,
        commitment_source=_InferenceCommitmentMustNotHandleCompetition(),
    )
    snapshots = tuple(
        replace(
            miner,
            coldkey=census[miner.hotkey].coldkey,
            ip=census[miner.hotkey].ip,
        )
        for miner in _competition_miners()
    )
    log = EpochFinalizer(cfg, scorer_version="scorer-v1").build_log(
        epoch_id=2,
        close_block=719,
        snapshots=snapshots,
        burn_uid=94,
        audit_manifest=manifest,
        now=COMPLETED_AT,
        competition_result=evidence.result,
        competition_packet_scores=evidence.packet_scores,
    )
    bundle_source = InMemoryBundleSource()
    for bundle in (*fresh_world.bundles.values(), fresh_world.baseline_bundle):
        bundle_source.add(bundle)
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="auditor-test", tokenomics=cfg, burn_uid=94),
        bundle_source,
        chain=fresh_world.anchor_chain,
    )

    derived, _reward_window, verdicts = auditor._competition_verdicts(
        log, fresh_world.store, None, True
    )
    assert derived == evidence.result
    assert log.reward_window_state.ends_at == COMPLETED_AT + timedelta(hours=168)
    assert verdicts
    assert all(verdict.verdict is ItemVerdictKind.PASS for verdict in verdicts), [
        (verdict.verdict, verdict.code, verdict.detail) for verdict in verdicts
    ]

    # The result-window duration is not an auditor-local exception. It is part of
    # the pre-enrollment reward-parameter commitment and the pure reward-state fold.
    # An auditor composed with the testnet 2h override cannot PASS a 168h commitment.
    mismatched_window_cfg = cfg.model_copy(update={"result_window_hours": 2.0})
    mismatched_window_auditor = Auditor(
        AuditorConfig(
            auditor_hotkey="auditor-window-mismatch",
            tokenomics=mismatched_window_cfg,
            burn_uid=94,
        ),
        bundle_source,
        chain=fresh_world.anchor_chain,
    )
    _result, _window, window_mismatch = mismatched_window_auditor._competition_verdicts(
        log, fresh_world.store, None, True
    )
    assert window_mismatch[0].verdict is ItemVerdictKind.FAIL
    assert window_mismatch[0].code == COMPETITION_MISMATCH
    assert "reward_param_digest" in window_mismatch[0].detail

    # The receipt is independently re-read through the auditor's archive adapter.
    # A missing RPC seam is availability (INCONCLUSIVE/report-only), while readable
    # exact-block state carrying different raw bytes is a provable FAIL.
    unavailable_auditor = Auditor(
        AuditorConfig(auditor_hotkey="auditor-test", tokenomics=cfg, burn_uid=94),
        bundle_source,
        chain=None,
    )
    _result, _window, unavailable = unavailable_auditor._competition_verdicts(
        log, fresh_world.store, None, True
    )
    assert unavailable[0].verdict is ItemVerdictKind.SKIP
    assert unavailable[0].code == COMPETITION_UNVERIFIED
    assert "missing" in unavailable[0].detail

    receipt = evidence.competition_input
    mismatch_chain = InMemoryChain(
        _block=fresh_world.anchor_chain.current_block(),
        anchored=[b"different-readable-raw-commitment"],
        _anchor_blocks=[receipt.anchor_block],
        block_time_anchor=(
            receipt.anchor_block,
            fresh_world.anchor_chain.block_time(receipt.anchor_block),
        ),
    )
    mismatch_auditor = Auditor(
        AuditorConfig(auditor_hotkey="auditor-test", tokenomics=cfg, burn_uid=94),
        bundle_source,
        chain=mismatch_chain,
    )
    _result, _window, mismatch = mismatch_auditor._competition_verdicts(
        log, fresh_world.store, None, True
    )
    assert mismatch[0].verdict is ItemVerdictKind.FAIL
    assert mismatch[0].code == COMPETITION_MISMATCH
    assert "exact committed payload" in mismatch[0].detail

    missing_receipt_shape = json.loads(log.to_json())
    del missing_receipt_shape["audit_manifest"]["competition_input"][
        "anchor_block_hash"
    ]
    with pytest.raises(EpochLogInvalid, match="v14_fields"):
        type(log).from_json(json.dumps(missing_receipt_shape))

    legacy_shape = json.loads(log.to_json())
    del legacy_shape["audit_manifest"]["competition_input"]["subjects"][0][
        "dedup_excluded"
    ]
    with pytest.raises(EpochLogInvalid, match="v14_fields"):
        type(log).from_json(json.dumps(legacy_shape))

    forged_subjects = tuple(
        subject.model_copy(update={"dedup_excluded": False})
        for subject in evidence.competition_input.subjects
    )
    forged_input = evidence.competition_input.model_copy(
        update={"subjects": forged_subjects}
    )
    forged_manifest = log.audit_manifest.model_copy(
        update={"competition_input": forged_input}
    )
    forged_log = log.model_copy(update={"audit_manifest": forged_manifest})
    _derived, _reward_window, forged_verdicts = auditor._competition_verdicts(
        forged_log, fresh_world.store, None, True
    )
    assert forged_verdicts[0].verdict is ItemVerdictKind.FAIL
    assert forged_verdicts[0].code == COMPETITION_MISMATCH
    assert "dedup exclusions differ" in forged_verdicts[0].detail


def test_already_applied_cycle_is_not_replayed_or_reopened(
    fresh_world: GoldenWorld,
) -> None:
    conn = fresh_world.comp_conn
    census = _competition_census("hk-a", "hk-b")
    cfg = TokenomicsConfig(competition_emissions_enabled=True)
    evidence = build_competition_epoch_evidence(
        conn,
        census_by_hotkey=census,
        store=fresh_world.store,
        tokenomics=cfg,
        through_time=COMPLETED_AT,
    )
    assert evidence is not None

    # Once this global cycle is chained into reward-window state, the authority
    # must not read or reapply it on later epochs. The cycle boundary is checked
    # before any archive/commitment read, so retained evidence can age independently
    # without poisoning inference finalization after it has already been applied.
    assert (
        build_competition_epoch_evidence(
            conn,
            census_by_hotkey=census,
            store=fresh_world.store,
            tokenomics=cfg,
            through_time=COMPLETED_AT + timedelta(days=1),
            after_cycle=evidence.result.cycle,
        )
        is None
    )


def test_reward_window_carries_without_replaying_the_competition(
    fresh_world: GoldenWorld,
) -> None:
    cfg = TokenomicsConfig(competition_emissions_enabled=True)
    evidence = build_competition_epoch_evidence(
        fresh_world.comp_conn,
        census_by_hotkey=_competition_census("hk-a", "hk-b"),
        store=fresh_world.store,
        tokenomics=cfg,
        through_time=COMPLETED_AT,
    )
    assert evidence is not None
    manifest = build_audit_manifest(
        evidence.scored_items,
        store=fresh_world.store,
        competition_input=evidence.competition_input,
        commitment_source=_InferenceCommitmentMustNotHandleCompetition(),
    )
    first = EpochFinalizer(cfg, scorer_version="scorer-v1").build_log(
        epoch_id=3,
        close_block=1_079,
        snapshots=_competition_miners(),
        burn_uid=94,
        audit_manifest=manifest,
        now=COMPLETED_AT,
        competition_result=evidence.result,
        competition_packet_scores=evidence.packet_scores,
    )
    carry_manifest = build_audit_manifest(
        (),
        prior_fold_cursors=first.audit_manifest.fold_cursors,
        current_census_uids=(10, 11),
    )
    carry = EpochFinalizer(cfg, scorer_version="scorer-v1").build_log(
        epoch_id=4,
        close_block=1_439,
        snapshots=_competition_miners(),
        miner_census=first.miner_census,
        burn_uid=94,
        audit_manifest=carry_manifest,
        now=COMPLETED_AT + timedelta(hours=1),
        prior_log_digest=first.log_digest(),
        prior_earning={10: ("hk-a", 0.0), 11: ("hk-b", 0.0)},
        prior_fold_cursors=first.audit_manifest.fold_cursors,
        prior_reward_window_state=first.reward_window_state,
    )
    assert carry.competition_result is None
    assert carry.reward_window_state == first.reward_window_state
    assert carry.weight_shares == first.weight_shares

    auditor = Auditor(
        AuditorConfig(auditor_hotkey="auditor-test", tokenomics=cfg, burn_uid=94),
        InMemoryBundleSource(),
    )

    derived, reward_state, verdicts = auditor._competition_verdicts(
        carry, fresh_world.store, first, False
    )

    assert derived is None
    assert reward_state == first.reward_window_state
    assert verdicts
    assert all(verdict.verdict is ItemVerdictKind.PASS for verdict in verdicts)
