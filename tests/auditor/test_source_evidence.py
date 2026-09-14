"""Auditors reproduce source-proximity zeros from archived originals, never from the authority's word."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.auditor.test_duplicate_evidence import _case, _signed_receipt
from vidaio.audit import ArtifactKind, canonical_json_bytes, verify_bundle
from vidaio.audit.recompute import RecomputedScore
from vidaio.auditor.report import ItemVerdictKind
from vidaio.auditor.source_evidence import audit_source_rounds, validate_source_packet, verify_roster
from vidaio.challenge import deep_reveal_verifier
from vidaio.epoch import AuditFileKind, AuditFileRef, AuditManifest
from vidaio.scoring.compression import score_compression
from vidaio.scoring.result import compose_item_score, config_digest
from vidaio.scoring.source_proximity_evidence import (
    InvalidSourceEvidence, SourceMember, SourceProximityWitness, SourceRoundEvidence,
    derive_flags, derive_medians, mint_source_proximity_packet, source_witness_from_packet,
)

RESIDUALS = {1: (-0.20, -0.9), 2: (-0.10, -1.1), 3: (-0.30, -0.7), 4: (-0.05, -1.0), 9: (0.45, 0.30)}


def source_case(tmp_path, residuals=RESIDUALS):
    store, scoring, real, commitment, template, *_ = _case(tmp_path)
    roster, originals = [], {}
    for uid, (vres, cres) in sorted(residuals.items()):
        hotkey = f"miner-{uid}"
        output = store.put(f"encode-{uid}".encode(), ArtifactKind.MINER_OUTPUT)
        receipt = _signed_receipt(validator="validator-hotkey", miner=hotkey, uid=uid,
            challenge_id=template.challenge_id, track="compression", anchor=template.challenge_anchor,
            input_digest=template.challenge_input.digest, input_size=template.challenge_input.byte_size,
            output_digest=output.digest, output_size=output.byte_size)
        packet = compose_item_score(item_id=receipt.metadata.task_id, challenge_id=template.challenge_id,
            track="compression", miner_hotkey=hotkey, content_digest=output.digest, gate_passed=True,
            violations=[], breakdown=score_compression(candidate_bytes=output.byte_size, reference_bytes=1000,
                vmaf=95.0, config=scoring), config=scoring,
            metrics={"vmaf": 95.0, "vmaf_residual": vres, "chroma_residual": cres,
                     "psnr_uv_reference": 40.0 + cres, "psnr_uv_input": 40.0, "final_score": 0.5},
            scorer_version=real.scorer_version)
        obj = json.loads(packet.to_json())
        ref = store.put(canonical_json_bytes(obj), ArtifactKind.SCORE_PACKET)
        originals[uid] = obj
        roster.append(SourceMember(uid=uid, hotkey=hotkey, score_packet=ref, output=output, receipt=receipt,
                                   vmaf_residual=vres, chroma_residual=cres))
    roster = tuple(roster)
    vmed, cmed = derive_medians(roster)
    context = SourceRoundEvidence(challenge_id=template.challenge_id, item_id=template.challenge_id,
        track="compression", commitment_anchor=template.challenge_anchor, challenge_input=template.challenge_input,
        reference_original=template.reference_original, committed_scorer_version=real.scorer_version,
        scoring_config_digest=config_digest(scoring), roster=roster, vmaf_residual_median=vmed,
        chroma_residual_median=cmed, flagged=derive_flags(roster))
    witness = SourceProximityWitness(round_evidence=context, member_uid=9)
    zero = mint_source_proximity_packet(witness=witness, config=scoring)
    member = context.member(9)
    zero_ref = store.put(canonical_json_bytes(zero.model_dump(mode="json")), ArtifactKind.SCORE_PACKET)
    bundle = template.model_copy(update=dict(item_id=member.receipt.metadata.task_id, miner_hotkey=member.hotkey,
        miner_output=member.output, miner_receipt=member.receipt, score_packet=zero_ref,
        scorer_version=zero.scorer_version, backend_versions={}))
    return SimpleNamespace(store=store, scoring=scoring, real=real, template=template, context=context,
                           originals=originals, witness=witness, zero=zero, bundle=bundle)


def residual_recompute(case):
    """Stand-in for the media rerun: answers with the archived original's published metrics."""
    def recompute(bundle, artifacts):
        packet = json.loads(artifacts[ArtifactKind.SCORE_PACKET])
        return RecomputedScore(metrics={k: v for k, v in packet["metrics"].items() if isinstance(v, (int, float))},
            scorer_version=case.real.scorer_version, backend_versions={}, score=packet["score"],
            gate_passed=True, breakdown=packet["breakdown"])
    return recompute


def test_zero_packet_validates_and_tampering_is_detected(tmp_path):
    case = source_case(tmp_path)
    validate_source_packet(case.zero, case.bundle, case.witness)
    verify_roster(case.context, case.store)
    obj = case.zero.model_dump(mode="json")
    for change in ({"score": 0.1}, {"gate_passed": True}, {"miner_hotkey": "miner-1"}, {"backend_versions": {"x": "1"}},
                   {"violations": [{**obj["violations"][0], "measured": 1.0}]}):
        with pytest.raises(InvalidSourceEvidence):
            validate_source_packet({**obj, **change}, case.bundle, case.witness)


def test_roster_values_must_match_archived_originals(tmp_path):
    case = source_case(tmp_path)
    member = case.context.member(2)
    forged = member.model_copy(update={"chroma_residual": member.chroma_residual + 0.01})
    roster = tuple(forged if m.uid == 2 else m for m in case.context.roster)
    vmed, cmed = derive_medians(roster)
    forged_context = case.context.model_copy(update={"roster": roster, "vmaf_residual_median": vmed,
                                                     "chroma_residual_median": cmed, "flagged": derive_flags(roster)})
    with pytest.raises(InvalidSourceEvidence, match="residuals differ"):
        verify_roster(forged_context, case.store)


def test_verify_bundle_reproduces_the_zero_through_the_real_recomputer(tmp_path, monkeypatch):
    case = source_case(tmp_path)
    monkeypatch.setattr(case.real, "recompute", residual_recompute(case))
    report = verify_bundle(case.bundle, case.store, case.real, expected_bundle_digest=case.bundle.bundle_digest(),
        reveal_verifier=deep_reveal_verifier, strict=False)
    assert report.passed, report.failures()


def test_verify_bundle_disputes_when_the_flagged_original_does_not_reproduce(tmp_path, monkeypatch):
    case = source_case(tmp_path)
    honest = residual_recompute(case)

    def drifted(bundle, artifacts):
        fresh = honest(bundle, artifacts)
        return fresh.model_copy(update={"metrics": {**fresh.metrics, "chroma_residual": fresh.metrics["chroma_residual"] - 0.5}})
    monkeypatch.setattr(case.real, "recompute", drifted)
    report = verify_bundle(case.bundle, case.store, case.real, expected_bundle_digest=case.bundle.bundle_digest(),
        reveal_verifier=deep_reveal_verifier, strict=False)
    assert not report.passed
    assert any("do not reproduce" in c.reason for c in report.failures())


def test_verify_bundle_rejects_a_witness_for_a_different_committed_scorer(tmp_path, monkeypatch):
    case = source_case(tmp_path)
    monkeypatch.setattr(case.real, "recompute", residual_recompute(case))
    obj = case.zero.model_dump(mode="json")
    foreign = case.context.model_copy(update={"committed_scorer_version": "vidaio-scorer/1+000000000000"})
    obj["metrics"] = {"source_proximity_witness": SourceProximityWitness(round_evidence=foreign, member_uid=9).to_json()}
    ref = case.store.put(canonical_json_bytes(obj), ArtifactKind.SCORE_PACKET)
    bundle = case.bundle.model_copy(update={"score_packet": ref})
    report = verify_bundle(bundle, case.store, case.real, expected_bundle_digest=bundle.bundle_digest(),
        reveal_verifier=deep_reveal_verifier, strict=False)
    assert not report.passed


def epoch_log(case, *, drop_uid=None, extra_uid=None):
    per_uid = {}
    ordering = case.context.commitment_anchor.dispatch_ordering_key
    for member in case.context.roster:
        if member.uid == drop_uid:
            continue
        packet = case.zero.model_dump(mode="json") if member.uid == 9 else case.originals[member.uid]
        packet = dict(packet, item_id=f"{member.receipt.metadata.task_id}-c{ordering}", cycle_sequence=ordering, excluded=False)
        ref = case.store.put(canonical_json_bytes(packet), ArtifactKind.SCORE_PACKET)
        per_uid[member.uid] = (AuditFileRef(kind=AuditFileKind.SCORE_PACKET, digest=ref.digest,
            challenge_id=case.context.challenge_id, item_id=packet["item_id"], committed_track="compression"),)
    if extra_uid is not None:
        packet = dict(next(iter(case.originals.values())), item_id=f"{case.context.challenge_id}:{extra_uid}-c{ordering}",
                      miner_hotkey=f"miner-{extra_uid}", cycle_sequence=ordering, excluded=False)
        ref = case.store.put(canonical_json_bytes(packet), ArtifactKind.SCORE_PACKET)
        per_uid[extra_uid] = (AuditFileRef(kind=AuditFileKind.SCORE_PACKET, digest=ref.digest,
            challenge_id=case.context.challenge_id, item_id=packet["item_id"], committed_track="compression"),)
    return SimpleNamespace(schema_version=17, audit_manifest=AuditManifest(per_uid=per_uid, content_rounds=()))


def test_epoch_pass_accepts_a_complete_roster(tmp_path):
    case = source_case(tmp_path)
    verdicts = audit_source_rounds(epoch_log(case), case.store)
    assert [v.verdict for v in verdicts] == [ItemVerdictKind.PASS]


def test_epoch_pass_disputes_an_omitted_or_invented_member(tmp_path):
    case = source_case(tmp_path)
    for log in (epoch_log(case, extra_uid=5),):
        verdicts = audit_source_rounds(log, case.store)
        assert [v.verdict for v in verdicts] == [ItemVerdictKind.FAIL], verdicts
        assert "eligible fold set" in verdicts[0].detail
    # A member the authority left out of the roster but still folded positively.
    smaller = {uid: RESIDUALS[uid] for uid in (2, 3, 4, 9)}
    case2 = source_case(tmp_path / "b", smaller)
    log = epoch_log(case2, extra_uid=7)
    verdicts = audit_source_rounds(log, case2.store)
    assert [v.verdict for v in verdicts] == [ItemVerdictKind.FAIL]


def test_epoch_pass_ignores_logs_without_source_zeros(tmp_path):
    case = source_case(tmp_path)
    log = epoch_log(case, drop_uid=9)
    assert audit_source_rounds(log, case.store) == ()
