"""Commit membership uses archived evidence and survives empty/all-skip epochs."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind, LocalFsStore
from vidaio.auditor import (
    Auditor, AuditorConfig, AuditMode, AuditStatus, InMemoryBundleSource,
    ItemVerdictKind, ROUND_COMMIT_MISMATCH, ROUND_COMMIT_STALE, SamplePolicy,
)
from vidaio.auditor.round_membership import audit_round_membership
from vidaio.authority.finalizer import EpochFinalizer, build_audit_manifest
from vidaio.challenge import ChallengeAnchor
from vidaio.epoch import AuditManifest, ContentRoundInput, EpochLog, RoundCommitInput
from vidaio.tokenomics import TokenomicsConfig
from tests.auditor.fakes import make_fake_bundle, scored_item, folded_miner, NOW
from tests.auditor.test_availability_earning import (
    _observation, _build_log, _ArchiveChain, _verify, BURN_UID,
)


def empty_log(*, epoch=11, close=120, prior=None, cursor=None, content=(), commits=()):
    manifest = AuditManifest(content_rounds=content, round_commits=commits, round_commit_cursor=cursor)
    return EpochFinalizer(TokenomicsConfig(), scorer_version="membership-test").build_log(
        epoch_id=epoch, close_block=close, snapshots=(), burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW,
        prior_log_digest=prior.log_digest() if prior is not None else None,
    )


def membership(log, entries, *, cursor=None):
    entries = tuple(entries)
    if cursor is None:
        cursor = max((e.ordering_key for e in entries), default=None)
    return log.model_copy(update={"audit_manifest": log.audit_manifest.model_copy(
        update={"round_commits": entries, "round_commit_cursor": cursor})})


def availability_case(tmp_path, *, mode=AuditMode.BEACON, commit=150, maximum=2880):
    observation = _observation()
    log, evidence = _build_log(observation)
    prior = empty_log()
    log = log.model_copy(update={"prior_log_digest": prior.log_digest()})
    anchor = observation.attempt.request.metadata.commitment_anchor
    entry = RoundCommitInput(round_id="ordinary-round", challenge_id=evidence.challenge_id,
        commit_block=commit, anchor_block=anchor.block, ordering_key=anchor.dispatch_ordering_key)
    log = membership(log, [entry])
    service = Auditor(AuditorConfig(auditor_hotkey="auditor", burn_uid=BURN_UID,
        audit_mode=mode, round_commit_max_blocks=maximum), InMemoryBundleSource(),
        chain=_ArchiveChain(observation), availability_verify_fn=_verify)
    return service, LocalFsStore(tmp_path / "store"), prior, log, entry


@pytest.mark.parametrize("mode", list(AuditMode))
def test_straddle_is_clean_in_both_modes_with_no_media_sample(tmp_path, mode):
    service, store, prior, log, entry = availability_case(tmp_path, mode=mode)
    assert entry.anchor_block < prior.close_block < entry.commit_block <= log.close_block
    report = service.audit_epoch(log, store, SamplePolicy(sample_rate=0, min_samples=0),
                                None, NOW, prior_log=prior, is_genesis=False)
    assert report.overall is AuditStatus.CLEAN, report.earning_verdicts
    assert report.audit_mode is mode


@pytest.mark.parametrize("mode", list(AuditMode))
@pytest.mark.parametrize("commit", [99, 120, 201])
def test_before_anchor_repeated_or_future_commit_disputed_both_modes(tmp_path, mode, commit):
    service, store, prior, log, _ = availability_case(tmp_path, mode=mode, commit=commit)
    report = service.audit_epoch(log, store, SamplePolicy(sample_rate=0, min_samples=0),
                                None, NOW, prior_log=prior, is_genesis=False)
    assert report.overall is AuditStatus.DISPUTED
    assert any(v.code == ROUND_COMMIT_MISMATCH and v.verdict is ItemVerdictKind.FAIL
               for v in report.earning_verdicts)


@pytest.mark.parametrize("mode", list(AuditMode))
def test_slow_commit_is_report_only_even_when_all_checks_run(tmp_path, mode):
    service, store, prior, log, _ = availability_case(tmp_path, mode=mode, maximum=49)
    report = service.audit_epoch(log, store, SamplePolicy(sample_rate=0, min_samples=0),
                                None, NOW, prior_log=prior, is_genesis=False)
    finding = [v for v in report.earning_verdicts if v.code == ROUND_COMMIT_STALE]
    assert report.overall is AuditStatus.CLEAN
    assert len(finding) == 1 and finding[0].verdict is ItemVerdictKind.PASS
    service._config = service._config.model_copy(update={"round_commit_max_blocks": 50})
    assert not any(v.code == ROUND_COMMIT_STALE for v in audit_round_membership(service, log, store, prior, False))


@pytest.mark.parametrize("mutation", ["missing-index", "extra-entry", "anchor", "key", "cursor-drop", "cursor-invent"])
def test_index_and_anchor_integrity(tmp_path, mutation):
    service, store, prior, log, entry = availability_case(tmp_path)
    entries, cursor = [entry], entry.ordering_key
    if mutation == "missing-index": entries = []
    elif mutation == "extra-entry": entries.append(entry.model_copy(update={"challenge_id": "unbacked", "ordering_key": 10}))
    elif mutation == "anchor": entries = [entry.model_copy(update={"anchor_block": 101})]
    elif mutation == "key": entries = [entry.model_copy(update={"ordering_key": 10})]; cursor = 10
    elif mutation == "cursor-drop": cursor = 8
    elif mutation == "cursor-invent": cursor = 10
    bad = membership(log, entries, cursor=cursor)
    verdicts = audit_round_membership(service, bad, store, prior, False)
    assert any(v.verdict is ItemVerdictKind.FAIL for v in verdicts), verdicts


def test_unreadable_predecessor_is_inconclusive_not_genesis(tmp_path):
    service, store, _, log, _ = availability_case(tmp_path)
    verdicts = audit_round_membership(service, log, store, None, False)
    assert [v.verdict for v in verdicts] == [ItemVerdictKind.SKIP]


def test_same_dispatch_cannot_reappear_after_empty_epoch(tmp_path):
    service, store, prior, log, entry = availability_case(tmp_path)
    empty = empty_log(epoch=13, close=300, prior=log, cursor=entry.ordering_key)
    assert audit_round_membership(service, empty, store, log, False) == ()
    replay = membership(log.model_copy(update={"epoch_id": 14, "close_block": 400,
        "prior_log_digest": empty.log_digest()}), [entry.model_copy(update={"commit_block": 350})])
    verdicts = audit_round_membership(service, replay, store, empty, False)
    assert any(v.verdict is ItemVerdictKind.FAIL and "consumed" in v.detail for v in verdicts)
    reset = empty.model_copy(update={"audit_manifest": empty.audit_manifest.model_copy(update={"round_commit_cursor": None})})
    assert any(v.verdict is ItemVerdictKind.FAIL for v in audit_round_membership(service, reset, store, log, False))


def test_media_anchor_and_preimage_bind_the_index_and_missing_bytes_hold(tmp_path):
    store = LocalFsStore(tmp_path / "store")
    bundle = make_fake_bundle(store, challenge_id="challenge", item_id="item", miner_hotkey="hk1",
                              dispatch_ordering_key=7)
    anchor = ChallengeAnchor(netuid=85, dispatch_ordering_key=7, commitment_hash=bundle.commitment_hash,
                             block=90, block_hash="a" * 64)
    bundle = bundle.model_copy(update={"challenge_anchor": anchor})
    store.put(canonical_json_bytes(bundle.model_dump(mode="json")), ArtifactKind.AUDIT_BUNDLE)
    source = InMemoryBundleSource(); source.add(bundle)
    entry = RoundCommitInput(round_id="round", challenge_id=bundle.challenge_id,
                              commit_block=150, anchor_block=90, ordering_key=7)
    manifest = build_audit_manifest([scored_item(bundle, 1, seq=7)],
                                   round_commits=(entry,), store=store)
    prior = empty_log()
    log = EpochFinalizer(TokenomicsConfig(), scorer_version=bundle.scorer_version).build_log(
        epoch_id=12, close_block=200, snapshots=(folded_miner(1),), audit_manifest=manifest,
        burn_uid=BURN_UID, now=NOW, prior_log_digest=prior.log_digest())
    service = Auditor(AuditorConfig(), source)
    assert audit_round_membership(service, log, store, prior, False) == ()
    wrong = membership(log, [entry.model_copy(update={"anchor_block": 89})])
    assert any(v.verdict is ItemVerdictKind.FAIL for v in audit_round_membership(service, wrong, store, prior, False))
    (store._root / bundle.dag_reveal.backend_key).unlink()
    verdicts = audit_round_membership(service, log, store, prior, False)
    assert any(v.verdict is ItemVerdictKind.SKIP for v in verdicts)


def test_all_skip_content_consumes_cursor_without_earning_cycles(tmp_path):
    from tests.auditor.test_content_evidence import content_case
    store, _, _, template, context, *_ = content_case(tmp_path)
    context = context.model_copy(update={"skipped_components": context.components})
    template_ref = store.put(canonical_json_bytes(template.model_dump(mode="json")), ArtifactKind.AUDIT_BUNDLE)
    content = ContentRoundInput(challenge_id=context.challenge_id, item_id=context.item_id, track=context.track,
        round_json=context.to_json(), round_digest=context.digest(), template_bundle=template_ref)
    anchor = context.commitment_anchor
    entry = RoundCommitInput(round_id="all-skip-round", challenge_id=context.challenge_id,
        commit_block=anchor.block + 5, anchor_block=anchor.block, ordering_key=anchor.dispatch_ordering_key)
    log = empty_log(close=anchor.block + 10, content=(content,), commits=(entry,), cursor=entry.ordering_key)
    service = Auditor(AuditorConfig(), InMemoryBundleSource())
    assert not log.audit_manifest.earning_inputs
    assert audit_round_membership(service, log, store) == ()
    empty = empty_log(epoch=12, close=log.close_block + 10, prior=log, cursor=entry.ordering_key)
    assert audit_round_membership(service, empty, store, log, False) == ()
    replay = log.model_copy(update={"epoch_id": 13, "close_block": empty.close_block + 10,
        "prior_log_digest": empty.log_digest()})
    replay = membership(replay, [entry.model_copy(update={"commit_block": empty.close_block + 1})])
    assert any(v.verdict is ItemVerdictKind.FAIL for v in audit_round_membership(service, replay, store, empty, False))


@pytest.mark.parametrize("prior_key", [0, 8])
def test_authenticated_v16_history_keeps_exact_bytes_and_seeds_boundary(tmp_path, prior_key):
    service, store, _, current, entry = availability_case(tmp_path)
    prior17 = empty_log(close=120)
    obj = __import__("json").loads(prior17.to_json()); obj["schema_version"] = 16
    for key in ("payout_min_alpha_stake", "round_membership"): obj.pop(key)
    for key in ("content_rounds", "round_commits", "round_commit_cursor"): obj["audit_manifest"].pop(key)
    obj["audit_manifest"]["fold_cursors"] = {"7": prior_key}
    for value in obj["miners"] + obj["miner_census"]: value.pop("alpha_stake")
    raw = canonical_json_bytes(obj)
    historical = EpochLog.from_history_json(raw, expected_digest=sha256_hex(raw), expected_epoch_id=prior17.epoch_id)
    assert historical.to_json() == raw
    assert audit_round_membership(service, historical, store) == ()
    carried = empty_log(epoch=12, close=200, prior=historical, cursor=prior_key)
    assert audit_round_membership(service, carried, store, historical, False) == ()
    current = current.model_copy(update={"prior_log_digest": historical.log_digest()})
    assert audit_round_membership(service, current, store, historical, False) == ()
    replay = membership(current, [entry.model_copy(update={"ordering_key": prior_key})], cursor=prior_key)
    assert any(v.verdict is ItemVerdictKind.FAIL for v in audit_round_membership(service, replay, store, historical, False))


def test_new_genesis_cannot_invent_historical_zero_cursor(tmp_path):
    service = Auditor(AuditorConfig(), InMemoryBundleSource())
    verdicts = audit_round_membership(service, empty_log(cursor=0), LocalFsStore(tmp_path / "store"))
    assert any(v.verdict is ItemVerdictKind.FAIL for v in verdicts)
