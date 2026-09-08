"""Content auditors derive evidence from archived bytes, including suppressed originals."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.auditor.test_duplicate_evidence import _case, _signed_receipt, _receipt_ok, _ChronologyChain
from vidaio.audit import ArtifactKind, canonical_json_bytes, verify_bundle
from vidaio.audit.recompute import RecomputedScore
from vidaio.auditor.chronology import verify_challenge_chronology, ChronologyKind
from vidaio.auditor.content_evidence import verify_round, audit_manifest_rounds
from vidaio.auditor.report import ItemVerdictKind
from vidaio.challenge import deep_reveal_verifier
from vidaio.epoch import ContentRoundInput, AuditManifest, AuditFileRef, AuditFileKind
from vidaio.scoring.content_duplicate_evidence import (
    ContentMember, ContentRoundEvidence, ContentDuplicateWitness, derive_edges,
    derive_components, mint_content_duplicate_packet,
)
from vidaio.scoring.content_fingerprint import compute_canonical_content
from vidaio.scoring.result import config_digest


class ByteRecomputer:
    requires_content_evidence = True
    """Test media adapter strips an encoded metadata prefix; real fingerprint primitive."""
    def __init__(self, root, version):
        self.root, self.scorer_version = root, version

    def recompute(self, bundle, artifacts):
        output=artifacts[ArtifactKind.MINER_OUTPUT]
        raw=output.read_bytes() if isinstance(output,Path) else output
        canonical=self.root/'actual.y4m'; canonical.write_bytes(raw.split(b'\n',1)[1])
        evidence=compute_canonical_content(canonical)
        return RecomputedScore(metrics={'final_score':0.5},scorer_version=self.scorer_version,
            backend_versions={},score=0.5,gate_passed=True,breakdown={'final_score':0.5},
            canonical_content_digest=evidence.canonical_content_digest,
            content_fingerprint=evidence.content_fingerprint,encoded_size=len(raw),
            canonicalization_plan_digest='c'*64)


def content_case(tmp_path):
    store,scoring,real,commitment,template,_,_= _case(tmp_path)
    y4m=b'YUV4MPEG2 W4 H4 F1:1 Ip A1:1 C420jpeg\nFRAME\n'+bytes(range(16))+b'\x80'*8
    path=tmp_path/'measured.y4m';path.write_bytes(y4m)
    evidence=compute_canonical_content(path)
    roster=[]; originals={}
    for uid in (4,9):
        hotkey=f'miner-{uid}'
        output=store.put(f'metadata-{uid}\n'.encode()+y4m,ArtifactKind.MINER_OUTPUT)
        receipt=_signed_receipt(validator='validator-hotkey',miner=hotkey,uid=uid,
            challenge_id=template.challenge_id,track='compression',anchor=template.challenge_anchor,
            input_digest=template.challenge_input.digest,input_size=template.challenge_input.byte_size,
            output_digest=output.digest,output_size=output.byte_size)
        packet=dict(item_id=receipt.metadata.task_id,challenge_id=template.challenge_id,track='compression',
            score=0.5,gate_passed=True,violations=[],skips=[],miner_hotkey=hotkey,content_digest=output.digest,
            canonical_content_digest=evidence.canonical_content_digest,content_fingerprint=list(evidence.content_fingerprint),
            encoded_size=output.byte_size,breakdown={'final_score':0.5},metrics={'final_score':0.5},
            scorer_version=real.scorer_version,backend_versions={},pieapp_start_frame=None,
            scoring_config_digest=config_digest(scoring),canonicalization_plan_digest='c'*64)
        ref=store.put(canonical_json_bytes(packet),ArtifactKind.SCORE_PACKET)
        originals[uid]=packet
        roster.append(ContentMember(uid=uid,hotkey=hotkey,score_packet=ref,output=output,receipt=receipt,
            canonical_content_digest=evidence.canonical_content_digest,content_fingerprint=evidence.content_fingerprint,
            encoded_size=output.byte_size,canonicalization_plan_digest='c'*64))
    context=ContentRoundEvidence(challenge_id=template.challenge_id,item_id=template.challenge_id,track='compression',
        challenge_input=template.challenge_input,reference_original=template.reference_original,
        commitment_anchor=template.challenge_anchor,committed_scorer_version=real.scorer_version,
        scoring_config_digest=config_digest(scoring),roster=tuple(roster),
        edges=derive_edges(roster),components=derive_components(roster))
    byte=ByteRecomputer(tmp_path,real.scorer_version)
    winner=context.winner(context.components[0]);loser=next(uid for uid in (4,9) if uid!=winner)
    witness=ContentDuplicateWitness(round_evidence=context,component_uids=(4,9),winner_uid=winner,loser_uid=loser)
    zero=mint_content_duplicate_packet(witness=witness,config=scoring)
    member=next(m for m in roster if m.uid==loser)
    zero_ref=store.put(canonical_json_bytes(zero.model_dump(mode='json')),ArtifactKind.SCORE_PACKET)
    zero_bundle=template.model_copy(update=dict(item_id=member.receipt.metadata.task_id,miner_hotkey=member.hotkey,
        miner_output=member.output,miner_receipt=member.receipt,score_packet=zero_ref,
        scorer_version=zero.scorer_version,backend_versions={}))
    return store,scoring,real,template,context,byte,originals,witness,zero_bundle


def test_archived_originals_rederive_component_and_signatures(tmp_path):
    store,scoring,real,template,context,byte,*_=content_case(tmp_path)
    results=verify_round(context,template,store,byte,
        chronology=lambda b:verify_challenge_chronology(b,store,_ChronologyChain(b.challenge_anchor),
            require_anchor=True,expected_netuid=85,scoring=scoring,receipt_verifier=_receipt_ok),
        reveal_verifier=deep_reveal_verifier)
    assert [(r.uids,r.status) for r in results]==[((4,9),'verified')]


@pytest.mark.parametrize('field,value',[
    ('canonical_content_digest','e'*64),('content_fingerprint',('0'*16,)*32),
    ('canonicalization_plan_digest','d'*64),
])
def test_matching_forged_packet_and_witness_fail_actual_media_recompute(tmp_path,field,value):
    store,scoring,real,template,context,byte,originals,*_=content_case(tmp_path)
    roster=[]
    for member in context.roster:
        packet=dict(originals[member.uid]);packet[field]=list(value) if isinstance(value,tuple) else value
        ref=store.put(canonical_json_bytes(packet),ArtifactKind.SCORE_PACKET)
        roster.append(member.model_copy(update={field:value,'score_packet':ref}))
    context=context.model_copy(update=dict(roster=tuple(roster),edges=derive_edges(roster),components=derive_components(roster)))
    results=verify_round(context,template,store,byte,reveal_verifier=deep_reveal_verifier)
    assert results[0].status=='invalid'
    assert 'CONTENT_EVIDENCE_MISMATCH' in results[0].detail


def test_missing_archived_member_never_verifies(tmp_path,monkeypatch):
    store,_,_,template,context,byte,*_=content_case(tmp_path)
    real_get=store.get_limited
    def get(ref,limit):
        if ref==context.roster[0].score_packet:raise FileNotFoundError('removed original')
        return real_get(ref,limit)
    monkeypatch.setattr(store,'get_limited',get)
    assert verify_round(context,template,store,byte)[0].status=='unavailable'


def test_content_zero_checks_all_component_signatures(tmp_path):
    store,scoring,_,_,context,_,_,_,bundle=content_case(tmp_path)
    result=verify_challenge_chronology(bundle,store,_ChronologyChain(bundle.challenge_anchor),
        require_anchor=True,expected_netuid=85,scoring=scoring,receipt_verifier=_receipt_ok)
    assert result.kind is ChronologyKind.PASS
    result=verify_challenge_chronology(bundle,store,_ChronologyChain(bundle.challenge_anchor),
        require_anchor=True,expected_netuid=85,scoring=scoring,
        receipt_verifier=lambda receipt:receipt==bundle.miner_receipt)
    assert result.kind is ChronologyKind.FAIL


def finalized_case(case):
    store,scoring,real,template,context,byte,originals,witness,bundle=case
    per_uid={};key=context.commitment_anchor.dispatch_ordering_key
    for member in context.roster:
        packet=(json.loads(store.get(bundle.score_packet)) if member.uid==witness.loser_uid else originals[member.uid])
        packet=dict(packet,item_id=f'{member.receipt.metadata.task_id}-c{key}',cycle_sequence=key,excluded=False)
        ref=store.put(canonical_json_bytes(packet),ArtifactKind.SCORE_PACKET)
        per_uid[member.uid]=(AuditFileRef(kind=AuditFileKind.SCORE_PACKET,digest=ref.digest,
            challenge_id=context.challenge_id,item_id=packet['item_id'],committed_track='compression'),)
    template_ref=store.put(canonical_json_bytes(template.model_dump(mode="json")),ArtifactKind.AUDIT_BUNDLE)
    entry=ContentRoundInput(challenge_id=context.challenge_id,item_id=context.item_id,track=context.track,
        round_json=context.to_json(),round_digest=context.digest(),template_bundle=template_ref)
    log=SimpleNamespace(schema_version=17,prior_log_digest=None,
        miner_census=tuple(SimpleNamespace(uid=m.uid,hotkey=m.hotkey) for m in context.roster),
        audit_manifest=AuditManifest(per_uid=per_uid,content_rounds=(entry,)))
    service=SimpleNamespace(_reveal_verifier=deep_reveal_verifier,
        _challenge_chronology=lambda b,s:verify_challenge_chronology(b,s,_ChronologyChain(b.challenge_anchor),
            require_anchor=True,expected_netuid=85,scoring=scoring,receipt_verifier=_receipt_ok))
    return service,log,store,byte


def test_complete_round_index_reproduces_winner_and_zero(tmp_path):
    service,log,store,byte=finalized_case(content_case(tmp_path))
    verdicts=audit_manifest_rounds(service,log,store,byte)
    assert verdicts and all(v.verdict is ItemVerdictKind.PASS for v in verdicts),verdicts


def test_removed_entire_index_and_removed_loser_are_disputed(tmp_path):
    service,log,store,byte=finalized_case(content_case(tmp_path))
    omitted=SimpleNamespace(**(vars(log)|{'audit_manifest':log.audit_manifest.model_copy(update={'content_rounds':()})}))
    assert all(v.verdict is ItemVerdictKind.FAIL for v in audit_manifest_rounds(service,omitted,store,byte))
    per_uid=dict(log.audit_manifest.per_uid);per_uid.pop(next(iter(per_uid)))
    omitted=SimpleNamespace(**(vars(log)|{'audit_manifest':log.audit_manifest.model_copy(update={'per_uid':per_uid})}))
    assert any(v.verdict is ItemVerdictKind.FAIL for v in audit_manifest_rounds(service,omitted,store,byte))


def test_reserved_content_zero_runs_real_component_dispatch(tmp_path,monkeypatch):
    store,_,real,_,_,byte,_,_,bundle=content_case(tmp_path)
    monkeypatch.setattr(real,'recompute',byte.recompute)
    report=verify_bundle(bundle,store,real,expected_bundle_digest=bundle.bundle_digest(),
        reveal_verifier=deep_reveal_verifier,strict=False)
    assert report.passed,report.failures()


def test_noncanonical_zero_extra_violation_is_disputed(tmp_path,monkeypatch):
    store,_,real,_,_,byte,_,_,bundle=content_case(tmp_path)
    monkeypatch.setattr(real,'recompute',byte.recompute)
    packet=json.loads(store.get(bundle.score_packet));packet['violations'][0]['measured']=1.0
    ref=store.put(canonical_json_bytes(packet),ArtifactKind.SCORE_PACKET)
    bundle=bundle.model_copy(update={'score_packet':ref})
    report=verify_bundle(bundle,store,real,expected_bundle_digest=bundle.bundle_digest(),
        reveal_verifier=deep_reveal_verifier,strict=False)
    assert not report.passed
    assert any('canonical signed-loser' in c.reason for c in report.failures())


def test_unavailable_component_skips_only_itself_other_component_proceeds(tmp_path,monkeypatch):
    case=list(content_case(tmp_path));store,scoring,real,template,context,byte,originals,witness,bundle=case
    uid=20;hotkey='miner-20'
    original_bytes=store.get(context.roster[0].output).split(b'\n',1)[1]
    frame=original_bytes.split(b'FRAME\n',1)[1]
    y4m=original_bytes+(b'FRAME\n'+frame)*4
    path=tmp_path/'distinct.y4m';path.write_bytes(y4m);evidence=compute_canonical_content(path)
    output=store.put(b'metadata-20\n'+y4m,ArtifactKind.MINER_OUTPUT)
    receipt=_signed_receipt(validator='validator-hotkey',miner=hotkey,uid=uid,
        challenge_id=context.challenge_id,track=context.track,anchor=context.commitment_anchor,
        input_digest=context.challenge_input.digest,input_size=context.challenge_input.byte_size,
        output_digest=output.digest,output_size=output.byte_size)
    packet=dict(originals[4],item_id=receipt.metadata.task_id,miner_hotkey=hotkey,content_digest=output.digest,
        canonical_content_digest=evidence.canonical_content_digest,content_fingerprint=list(evidence.content_fingerprint),
        encoded_size=output.byte_size)
    ref=store.put(canonical_json_bytes(packet),ArtifactKind.SCORE_PACKET)
    member=ContentMember(uid=uid,hotkey=hotkey,score_packet=ref,output=output,receipt=receipt,
        canonical_content_digest=evidence.canonical_content_digest,content_fingerprint=evidence.content_fingerprint,
        encoded_size=output.byte_size,canonicalization_plan_digest='c'*64)
    roster=context.roster+(member,)
    context=ContentRoundEvidence.model_validate(context.model_dump()|dict(roster=roster,
        edges=derive_edges(roster),components=derive_components(roster),skipped_components=((4,9),)))
    assert context.components==((4,9),(20,))
    originals[20]=packet;case[4]=context
    service,log,_,_=finalized_case(case)
    log.audit_manifest=log.audit_manifest.model_copy(update={'per_uid':{20:log.audit_manifest.per_uid[20]}})
    original_get=store.get_limited
    def unavailable(ref,limit):
        if ref==context.roster[0].score_packet:raise FileNotFoundError('archived member missing')
        return original_get(ref,limit)
    monkeypatch.setattr(store,'get_limited',unavailable)
    verdicts=audit_manifest_rounds(service,log,store,byte)
    assert {v.verdict for v in verdicts}=={ItemVerdictKind.PASS,ItemVerdictKind.SKIP},verdicts


def test_authenticated_v16_identity_is_separate_from_current_recomputer(tmp_path):
    from tests.epoch.test_schema17_history import legacy_bytes,current_log
    from vidaio.epoch import EpochLog
    from vidaio.audit.canonical import sha256_hex
    from vidaio.scoring_worker.service import historical_v16_scorer_version
    _,_,real,*_=content_case(tmp_path)
    raw=legacy_bytes();history=EpochLog.from_history_json(raw,expected_digest=sha256_hex(raw),expected_epoch_id=42)
    prior=real.for_epoch_log(history)
    assert prior is not real and prior.scorer_version!=real.scorer_version
    assert prior.scorer_version==historical_v16_scorer_version(real._config,real._scoring_config,runtime_attestation=real._runtime_attestation)
    assert real.for_epoch_log(current_log()) is real


@pytest.mark.parametrize('tamper',[False,True])
def test_declared_archive_write_skip_stays_inconclusive_but_tamper_disputes(tmp_path,tamper):
    case=list(content_case(tmp_path));store,_,_,_,context,_,originals,*_=case
    if tamper:
        roster=[]
        for member in context.roster:
            packet=dict(originals[member.uid],canonical_content_digest='e'*64)
            ref=store.put(canonical_json_bytes(packet),ArtifactKind.SCORE_PACKET)
            roster.append(member.model_copy(update={'score_packet':ref,'canonical_content_digest':'e'*64}))
        context=context.model_copy(update={'roster':tuple(roster)})
    context=context.model_copy(update={'skipped_components':context.components});case[4]=context
    service,log,store,byte=finalized_case(case)
    log.audit_manifest=log.audit_manifest.model_copy(update={'per_uid':{}})
    verdicts=audit_manifest_rounds(service,log,store,byte)
    assert verdicts and {v.verdict for v in verdicts}=={ItemVerdictKind.FAIL if tamper else ItemVerdictKind.SKIP},verdicts


def test_epoch_memo_scores_five_originals_once_across_four_zeros_and_winner(tmp_path,monkeypatch):
    from vidaio.auditor.content_cache import EpochRecomputer
    from vidaio.auditor.content_evidence import measured_bundle
    from vidaio.audit.recompute import ScorePacketShape
    store,scoring,real,template,context,byte,originals,*_=content_case(tmp_path)
    roster=list(context.roster)
    media=store.get(roster[0].output).split(b'\n',1)[1]
    for uid in (20,21,22):
        output=store.put(f'metadata-{uid}\n'.encode()+media,ArtifactKind.MINER_OUTPUT)
        receipt=_signed_receipt(validator='validator-hotkey',miner=f'miner-{uid}',uid=uid,
            challenge_id=context.challenge_id,track=context.track,anchor=context.commitment_anchor,
            input_digest=context.challenge_input.digest,input_size=context.challenge_input.byte_size,
            output_digest=output.digest,output_size=output.byte_size)
        packet=dict(originals[4],item_id=receipt.metadata.task_id,miner_hotkey=receipt.miner_hotkey,
                    content_digest=output.digest,encoded_size=output.byte_size)
        ref=store.put(canonical_json_bytes(packet),ArtifactKind.SCORE_PACKET)
        originals[uid]=packet
        roster.append(roster[0].model_copy(update=dict(uid=uid,hotkey=receipt.miner_hotkey,
            score_packet=ref,output=output,receipt=receipt,encoded_size=output.byte_size)))
    context=ContentRoundEvidence.model_validate(context.model_dump()|dict(roster=tuple(roster),
        edges=derive_edges(roster),components=derive_components(roster)))
    component=context.components[0];winner=context.winner(component)
    calls=[]
    def actual(bundle,artifacts):
        calls.append(bundle.miner_output.digest)
        return byte.recompute(bundle,artifacts)
    monkeypatch.setattr(real,'recompute',actual)
    memo=EpochRecomputer(real)
    zero_bundles=[]
    for member in roster:
        if member.uid==winner:continue
        witness=ContentDuplicateWitness(round_evidence=context,component_uids=component,
            winner_uid=winner,loser_uid=member.uid)
        packet=mint_content_duplicate_packet(witness=witness,config=scoring)
        ref=store.put(canonical_json_bytes(packet.model_dump(mode='json')),ArtifactKind.SCORE_PACKET)
        bundle=template.model_copy(update=dict(item_id=member.receipt.metadata.task_id,miner_hotkey=member.hotkey,
            miner_output=member.output,miner_receipt=member.receipt,score_packet=ref,
            scorer_version=packet.scorer_version,backend_versions={}))
        report=verify_bundle(bundle,store,memo,expected_bundle_digest=bundle.bundle_digest(),
            reveal_verifier=deep_reveal_verifier,strict=False)
        assert report.passed,report.failures()
        zero_bundles.append(bundle)
    member=next(m for m in roster if m.uid==winner)
    packet=dict(originals[winner],item_id=f'{member.receipt.metadata.task_id}-c7',cycle_sequence=7,excluded=False)
    ref=store.put(canonical_json_bytes(packet),ArtifactKind.SCORE_PACKET)
    winner_bundle=measured_bundle(template,member,ScorePacketShape.model_validate(packet)).model_copy(update={'score_packet':ref})
    report=verify_bundle(winner_bundle,store,memo,expected_bundle_digest=winner_bundle.bundle_digest(),
        reveal_verifier=deep_reveal_verifier,strict=False)
    assert report.passed,report.failures()
    assert all(r.status=='verified' for r in verify_round(context,template,store,memo))
    assert len(calls)==5 and len(set(calls))==5
    # A new audit invocation recomputes fresh rather than sharing availability/state.
    assert all(r.status=='verified' for r in verify_round(context,template,store,EpochRecomputer(real)))
    assert len(calls)==10
    old_get=store.get_limited
    def missing(ref,limit):
        if ref==roster[0].score_packet:raise FileNotFoundError('became unavailable after cached score')
        return old_get(ref,limit)
    monkeypatch.setattr(store,'get_limited',missing)
    result=verify_round(context,template,store,memo)
    assert result[0].status=='unavailable'
    assert len(calls)==10


def test_stripping_all_current_fields_fails_recompute_and_missing_index(tmp_path):
    from vidaio.auditor.content_evidence import measured_bundle
    from vidaio.audit.recompute import ScorePacketShape
    case=content_case(tmp_path);store,_,_,template,context,byte,originals,*_=case
    member=context.roster[0];packet=dict(originals[member.uid])
    for field in ('canonical_content_digest','content_fingerprint','encoded_size'):packet.pop(field)
    ref=store.put(canonical_json_bytes(packet),ArtifactKind.SCORE_PACKET)
    bundle=measured_bundle(template,member,ScorePacketShape.model_validate(packet)).model_copy(update={'score_packet':ref})
    report=verify_bundle(bundle,store,byte,expected_bundle_digest=bundle.bundle_digest(),
        reveal_verifier=deep_reveal_verifier,strict=False)
    assert any(c.code=='CONTENT_EVIDENCE_MISMATCH' for c in report.failures())
    service,log,store,byte=finalized_case(case)
    final=dict(packet,item_id=member.receipt.metadata.task_id+'-c7',cycle_sequence=7,excluded=False)
    final_ref=store.put(canonical_json_bytes(final),ArtifactKind.SCORE_PACKET)
    ref=AuditFileRef(kind=AuditFileKind.SCORE_PACKET,digest=final_ref.digest,
        challenge_id=context.challenge_id,item_id=final['item_id'],committed_track='compression')
    log.audit_manifest=AuditManifest(per_uid={member.uid:(ref,)})
    verdicts=audit_manifest_rounds(service,log,store,byte)
    assert verdicts and all(v.verdict is ItemVerdictKind.FAIL for v in verdicts)


@pytest.mark.parametrize('reason',['hotkey_changed','deregistered','already_folded'])
def test_original_roster_stays_complete_when_member_cannot_refold(tmp_path,reason):
    service,log,store,byte=finalized_case(content_case(tmp_path))
    uid=log.miner_census[0].uid
    rows=dict(log.audit_manifest.per_uid);rows.pop(uid)
    log.audit_manifest=log.audit_manifest.model_copy(update={'per_uid':rows})
    prior=None
    if reason=='hotkey_changed':
        log.miner_census=(SimpleNamespace(uid=uid,hotkey='replacement'),)+log.miner_census[1:]
    elif reason=='deregistered':log.miner_census=log.miner_census[1:]
    else:
        log.prior_log_digest='a'*64
        prior=SimpleNamespace(log_digest=lambda:'a'*64,audit_manifest=AuditManifest(fold_cursors={uid:7}))
    verdicts=audit_manifest_rounds(service,log,store,byte,prior_log=prior)
    assert verdicts and all(v.verdict is ItemVerdictKind.PASS for v in verdicts),verdicts


def test_split_domain_rejected_even_when_wrapper_item_ids_differ(tmp_path):
    from vidaio.epoch import EpochLogInvalid
    service,log,store,byte=finalized_case(content_case(tmp_path))
    entry=log.audit_manifest.content_rounds[0]
    forged=entry.model_copy(update={'item_id':'other-local-subset'})
    with pytest.raises(EpochLogInvalid,match='one complete roster|item_id'):
        AuditManifest(content_rounds=(entry,forged))
    log.audit_manifest=log.audit_manifest.model_copy(update={'content_rounds':(entry,forged)})
    assert any(v.verdict is ItemVerdictKind.FAIL for v in audit_manifest_rounds(service,log,store,byte))


def test_positive_score_cannot_hide_from_index_by_claiming_gate_failed(tmp_path):
    service,log,store,byte=finalized_case(content_case(tmp_path))
    uid,refs=next(iter(log.audit_manifest.per_uid.items()))
    ref=refs[0];packet=json.loads(store.get_digest_limited(ArtifactKind.SCORE_PACKET,ref.digest,max_bytes=1000000))
    packet.update(score=0.5,gate_passed=False)
    for field in ('canonical_content_digest','content_fingerprint','encoded_size'):packet.pop(field,None)
    raw=store.put(canonical_json_bytes(packet),ArtifactKind.SCORE_PACKET)
    ref=ref.model_copy(update={'digest':raw.digest})
    log.audit_manifest=AuditManifest(per_uid={uid:(ref,)})
    verdicts=audit_manifest_rounds(service,log,store,byte)
    assert any(v.verdict is ItemVerdictKind.FAIL and 'positive score' in v.detail for v in verdicts)


def test_conclusive_corruption_dominates_simultaneous_missing_artifact(tmp_path,monkeypatch):
    from vidaio.audit.store import IntegrityError
    store,_,_,template,context,byte,*_=content_case(tmp_path)
    get=store.get_limited; materialize=store.materialize
    def missing(ref,limit):
        if ref==template.manifest:raise FileNotFoundError('manifest unavailable')
        return get(ref,limit)
    def corrupt(ref,directory,*,max_bytes):
        if ref==context.roster[0].output:raise IntegrityError('proven media digest mismatch')
        return materialize(ref,directory,max_bytes=max_bytes)
    monkeypatch.setattr(store,'get_limited',missing)
    monkeypatch.setattr(store,'materialize',corrupt)
    result=verify_round(context,template,store,byte)
    assert result[0].status=='invalid'
    assert 'ARTIFACT_CORRUPT' in result[0].detail


def test_sampled_zero_missing_original_is_unavailable_not_proven_fraud(tmp_path,monkeypatch):
    store,_,real,_,context,byte,_,_,bundle=content_case(tmp_path)
    monkeypatch.setattr(real,'recompute',byte.recompute)
    get=store.get_limited
    def missing(ref,limit):
        if ref==context.roster[0].score_packet:raise FileNotFoundError('original temporarily missing')
        return get(ref,limit)
    monkeypatch.setattr(store,'get_limited',missing)
    report=verify_bundle(bundle,store,real,expected_bundle_digest=bundle.bundle_digest(),
        reveal_verifier=deep_reveal_verifier,strict=False)
    assert any(c.name=='score_recompute' and c.skipped and 'original temporarily missing' in c.reason for c in report.checks)
    assert not any(c.code=='RECOMPUTE_ERROR' for c in report.checks)


def test_zero_score_original_is_never_a_roster_member(tmp_path):
    from vidaio.scoring.content_duplicate_evidence import InvalidContentEvidence, validate_member_packet
    store,scoring,real,template,context,byte,originals,*_=content_case(tmp_path)
    member=context.roster[0]
    zero=dict(originals[member.uid],score=0.0,breakdown={'final_score':0.0},metrics={'final_score':0.0})
    with pytest.raises(InvalidContentEvidence,match='positively scored'):
        validate_member_packet(member,zero,context)
    validate_member_packet(member,originals[member.uid],context)


@pytest.mark.parametrize('score,expect_fail',[(0.0,False),(0.5,True)])
def test_index_owes_positive_packets_only_not_gate_passed_zeros(tmp_path,score,expect_fail):
    """A published gate-passed zero with content evidence is not owed a round-index entry;
    the same packet with a positive score omitted from the index is still a FAIL."""
    service,log,store,byte=finalized_case(content_case(tmp_path))
    uid,refs=next(iter(log.audit_manifest.per_uid.items()))
    ref=refs[0];packet=json.loads(store.get_digest_limited(ArtifactKind.SCORE_PACKET,ref.digest,max_bytes=1000000))
    stray=dict(packet,score=score,miner_hotkey='miner-77',item_id='duplicate-challenge:77-c7')
    raw=store.put(canonical_json_bytes(stray),ArtifactKind.SCORE_PACKET)
    stray_ref=ref.model_copy(update={'digest':raw.digest,'item_id':stray['item_id']})
    rows=dict(log.audit_manifest.per_uid);rows[77]=(stray_ref,)
    log.audit_manifest=log.audit_manifest.model_copy(update={'per_uid':rows})
    verdicts=audit_manifest_rounds(service,log,store,byte)
    omitted=[v for v in verdicts if v.verdict is ItemVerdictKind.FAIL and 'omitted' in v.detail]
    assert bool(omitted)==expect_fail,verdicts
