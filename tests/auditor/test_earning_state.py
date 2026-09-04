"""The EARNING-STATE re-derivation (#1) — the honesty crux, now EVIDENCE-BOUND.

The auditor must RECONSTRUCT each nonzero-weight uid's `accumulate_score` from the
audited packet scores + the chained prior-epoch carry-in, so an authority cannot
publish honest packets while assigning a SUBSTITUTED earning state. The fold is now
bound to committed evidence: each cycle score is tied to a committed
SCORE_PACKET (its recorded score + monotonic `cycle_sequence` fold-order anchor), so a
REORDER, an unbacked padded 0.0, or a substituted -1 exclusion all FAIL — not just a
substituted accumulate. These tests are media-free (`sample_rate=0`) so the ONLY signal
is the earning re-fold.
"""

from __future__ import annotations

import pytest

from vidaio.audit.recompute import IDENTITY_MISMATCH
from vidaio.audit.store import ArtifactKind, LocalFsStore
from vidaio.auditor import (
    Auditor,
    AuditorConfig,
    AuditStatus,
    EARNING_PACKET_REPLAY,
    EARNING_STATE_MISMATCH,
    EARNING_STATE_RESET,
    EARNING_STATE_UNVERIFIED,
    FOLD_CURSOR_MISMATCH,
    InMemoryBundleSource,
    ItemVerdictKind,
    PREDECESSOR_CHAIN_BROKEN,
    PREDECESSOR_UNVERIFIED,
    SamplePolicy,
    persist_bundle,
)
from vidaio.authority import EpochFinalizer, ScoredItem, build_audit_manifest
from vidaio.epoch.log import (
    AuditManifest,
    CycleScore,
    EarningInput,
    EpochLog,
    MinerCensusEntry,
    weight_vector_digest,
)
from vidaio.tokenomics import TokenomicsConfig, quantize_u16
from vidaio.tokenomics.ewma import accumulate
from vidaio.tokenomics.weights import build_weight_vector

from tests.auditor.fakes import (
    BURN_UID,
    NOW,
    SCORER,
    MetagraphAuditor,
    make_fake_bundle,
    make_miner,
    make_packet,
    metagraph_chain,
)

DECAY = TokenomicsConfig().ewma_decay
CFG = TokenomicsConfig()
NO_SAMPLE = SamplePolicy(sample_rate=0.0, min_samples=0)  # earning-only, no media recompute


def _census(miners):
    """Identity-only census matching a direct-construction adversarial fixture.

    ``MinerSnapshot`` predates pydantic and can carry a deliberately malformed null hotkey in
    one defense-in-depth test, so use ``model_construct`` here instead of weakening the real
    schema-v11 census model's string types.
    """
    return tuple(
        MinerCensusEntry.model_construct(
            uid=m.uid, hotkey=m.hotkey, coldkey=m.coldkey, ip=m.ip
        )
        for m in miners
    )


def _fold(prior: float, scores) -> float:
    v = prior
    for s in scores:
        v = accumulate(v, s, DECAY)
    return v


def _bundle(store, source, uid, item_id, score, *, seq=0, excluded=False):
    packet = make_packet(
        challenge_id="c1", item_id=item_id, miner_hotkey=f"hk{uid}", score=score,
        cycle_sequence=seq, excluded=excluded,
        metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": score},
    )
    b = make_fake_bundle(
        store, challenge_id="c1", item_id=item_id, miner_hotkey=f"hk{uid}", packet=packet,
        dispatch_ordering_key=seq,
    )
    persist_bundle(store, b)  # resolvable + stored (the finalizer probes it)
    source.add(b)
    return b


def _scored(b, uid, score, *, seq=0, excluded=False):
    return ScoredItem(
        uid=uid, hotkey=f"hk{uid}", challenge_id=b.challenge_id, item_id=b.item_id,
        bundle_digest=b.bundle_digest(), packet_digest=b.score_packet.digest,
        committed_track="compression", score=score, cycle_sequence=seq,
        excluded_cycle=excluded,
    )


def _auditor(source) -> Auditor:
    # MetagraphAuditor auto-wires the close-block metagraph from the log's honest
    # identities so the SNAPSHOT binding passes for honest fixtures;
    # the adversarial EARNING-path faults these tests exercise still dominate the roll-up.
    return MetagraphAuditor(
        AuditorConfig(
            auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID
        ),
        source,
    )


def _honest_genesis(store, source, per_uid_scores: dict[int, float]):
    """An honest epoch: each uid scored once this epoch from a zero carry-in."""
    items, miners = [], []
    for uid, score in per_uid_scores.items():
        b = _bundle(store, source, uid, f"i{uid}", score)
        items.append(_scored(b, uid, score))
        miners.append(make_miner(uid, _fold(0.0, [score])))
    manifest = build_audit_manifest(items, store=store)
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    log = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=tuple(miners),
        burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW,
    )
    return log, manifest, miners


# --- the crux: substituted accumulate_score + honest packets is CAUGHT ----------------


def test_honest_earning_state_passes(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, _, _ = _honest_genesis(store, source, {1: 0.8, 2: 0.6})

    report = _auditor(source).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    # every uid's accumulate_score re-folds from its audited packet + zero carry-in
    uid_verdicts = [v for v in report.earning_verdicts if v.uid is not None and v.item_id != "crown"]
    assert uid_verdicts and all(v.verdict is ItemVerdictKind.PASS for v in uid_verdicts)
    assert report.weight_verdict.verdict is ItemVerdictKind.PASS
    # an honest genesis epoch is CLEAN (not washed, and not spuriously INCONCLUSIVE).
    # This genesis fixture intentionally has no earning competition input.
    assert report.overall is AuditStatus.CLEAN


def test_substituted_accumulate_with_honest_packets_disputes(tmp_path) -> None:
    """The exact #1 scenario: honest packets, a substituted accumulate_score, weights
    that DO follow from the (substituted) inputs — only the re-fold catches it."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, manifest, miners = _honest_genesis(store, source, {1: 0.8, 2: 0.6})

    # Substitute uid 1's accumulate_score HIGH; weights follow from it (so the weight
    # re-derivation PASSES), but the honest packet (score 0.8) folds to _fold(0,[0.8]),
    # not the substituted value — so ONLY the earning re-fold disputes it.
    substituted = 0.95
    fab_miners = (make_miner(1, substituted), miners[1])
    shares = build_weight_vector(CFG, fab_miners, burn_uid=BURN_UID)
    assert shares.get(1, 0.0) > 0.0
    u16 = quantize_u16(shares)
    fab_log = EpochLog(
        schema_version=log.schema_version, epoch_id=100, close_block=360_000,
        scorer_version=SCORER, created_at=NOW, burn_uid=BURN_UID, miners=fab_miners,
        miner_census=_census(fab_miners),
        weight_shares=shares,
        weight_u16=u16, weight_vector_digest=weight_vector_digest(u16),
        audit_manifest=manifest,
    )

    report = _auditor(source).audit_epoch(fab_log, store, NO_SAMPLE, None, NOW)

    earning = {
        v.uid: v
        for v in report.earning_verdicts
        if v.item_id != "crown" and v.source == "earning"
    }
    assert earning[1].verdict is ItemVerdictKind.FAIL
    assert earning[1].code == EARNING_STATE_MISMATCH
    assert earning[2].verdict is ItemVerdictKind.PASS  # uid 2 untouched
    # the dispute is reflected through the weight verdict so the roll-up sees it
    assert report.weight_verdict.verdict is ItemVerdictKind.FAIL
    assert report.weight_verdict.code == EARNING_STATE_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_substituted_bundle_backing_a_foreign_packet_is_disputed(tmp_path) -> None:
    """an internal review: the manifest pairs an AUDIT_BUNDLE ref and a SCORE_PACKET ref by
    AUTHORITY-supplied (challenge_id, item_id) LABELS. Pointing the bundle ref at an
    UNRELATED but resolvable bundle (whose own DAG_REVEAL is well-formed) that
    authenticates a DIFFERENT packet must NOT let the foreign packet's score fold to
    PASS/CLEAN at zero media sampling — the resolved bundle has to AUTHENTICATE the packet
    it backs (its own score_packet.digest == the ref's, and its identity must match)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # An honest hk1 packet (score 0.8) AND an UNRELATED resolvable bundle (hk99, its own
    # packet scores 0.1) — both fully stored/resolvable, each with a well-formed DAG_REVEAL.
    b1 = _bundle(store, source, 1, "i1", 0.8)
    b99 = _bundle(store, source, 99, "i99", 0.1)
    # The substitution: uid 1's item KEEPS hk1's honest packet (so the packet-bound backing
    # + re-fold would otherwise PASS) but points its BUNDLE ref at the foreign b99. Absent
    # the identity binding this folds 0.8 straight to CLEAN.
    sub = ScoredItem(
        uid=1, hotkey="hk1", challenge_id=b1.challenge_id, item_id=b1.item_id,
        bundle_digest=b99.bundle_digest(), packet_digest=b1.score_packet.digest,
        committed_track="compression", score=0.8, cycle_sequence=0,
    )
    miner = make_miner(1, _fold(0.0, [0.8]))  # accumulate follows honestly from 0.8
    manifest = build_audit_manifest([sub], store=store)
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    log = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(miner,),
        burn_uid=BURN_UID, audit_manifest=manifest, now=NOW,
    )

    report = _auditor(source).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    v = next(v for v in report.earning_verdicts if v.uid == 1)
    assert v.verdict is ItemVerdictKind.FAIL
    assert v.code == IDENTITY_MISMATCH
    # a proven substitution is a CONCLUSIVE fault, not a wash-to-INCONCLUSIVE SKIP.
    assert report.overall is AuditStatus.DISPUTED


def test_earning_bundle_with_null_miner_is_disputed(tmp_path) -> None:
    """an internal review(a): the bundle backing uid 1's packet carries miner_hotkey=None
    (the AuditBundle model permits None), while the packet + manifest labels otherwise look
    valid. A score whose miner cannot be attributed to the uid must NOT fold in — the prior
    fix SKIPPED the miner check when the bundle miner was null, so nulling it dodged
    attribution. It is now a conclusive IDENTITY_MISMATCH ⇒ DISPUTED."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # A valid-looking hk1 packet, but the bundle wrapping it pins NO miner (None).
    packet = make_packet(
        challenge_id="c1", item_id="i1", miner_hotkey="hk1", score=0.8, cycle_sequence=0,
        metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": 0.8},
    )
    b = make_fake_bundle(
        store, challenge_id="c1", item_id="i1", miner_hotkey=None, packet=packet,
        dispatch_ordering_key=0,
    )
    persist_bundle(store, b)
    source.add(b)
    item = ScoredItem(
        uid=1, hotkey="hk1", challenge_id="c1", item_id="i1",
        bundle_digest=b.bundle_digest(), packet_digest=b.score_packet.digest,
        committed_track="compression", score=0.8, cycle_sequence=0,
    )
    manifest = build_audit_manifest([item], store=store)
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    log = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(make_miner(1, _fold(0.0, [0.8])),),
        burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW,
    )

    report = _auditor(source).audit_epoch(log, store, NO_SAMPLE, None, NOW)
    v = next(v for v in report.earning_verdicts if v.uid == 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == IDENTITY_MISMATCH
    assert "earning-backing bundle" in v.detail  # pinned to the null/foreign-miner check
    assert report.overall is AuditStatus.DISPUTED


def test_earning_packet_minted_for_foreign_miner_is_disputed(tmp_path) -> None:
    """an internal review(b): the manifest ref labels and the bundle digest all line up for
    uid 1 / hk1, and the bundle even PINS hk1 — but the SCORE PACKET's OWN internal
    miner_hotkey is hk99. The authority wrapped hk99's high-scoring packet in a bundle
    LABELLED for the targeted uid. Reading the packet's self-declared identity catches it:
    IDENTITY_MISMATCH ⇒ DISPUTED."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # The packet inside is minted for hk99; everything OUTER (bundle, ScoredItem, snapshot)
    # is uid 1 / hk1. Only reading the packet's internal identity exposes the reassignment.
    packet = make_packet(
        challenge_id="c1", item_id="i1", miner_hotkey="hk99", score=0.8, cycle_sequence=0,
        metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": 0.8},
    )
    b = make_fake_bundle(
        store, challenge_id="c1", item_id="i1", miner_hotkey="hk1", packet=packet,
        dispatch_ordering_key=0,
    )
    persist_bundle(store, b)
    source.add(b)
    item = ScoredItem(
        uid=1, hotkey="hk1", challenge_id="c1", item_id="i1",
        bundle_digest=b.bundle_digest(), packet_digest=b.score_packet.digest,
        committed_track="compression", score=0.8, cycle_sequence=0,
    )
    manifest = build_audit_manifest([item], store=store)
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    log = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(make_miner(1, _fold(0.0, [0.8])),),
        burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW,
    )

    report = _auditor(source).audit_epoch(log, store, NO_SAMPLE, None, NOW)
    v = next(v for v in report.earning_verdicts if v.uid == 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == IDENTITY_MISMATCH
    assert "minted for miner" in v.detail
    assert report.overall is AuditStatus.DISPUTED


#


@pytest.mark.parametrize("bad_hotkey", [None, ""])
def test_null_or_empty_log_hotkey_for_nonzero_uid_is_disputed(tmp_path, bad_hotkey) -> None:
    """an internal review: a NONZERO-weight uid whose log hotkey is missing/empty is itself a
    fault. `MinerSnapshot` is an unchecked dataclass (`_miner_from_obj` accepts JSON null),
    so nulling the hotkey used to SKIP every identity check (they only fire when the expected
    hotkey is non-null) — letting a null-hotkey uid wrap another miner's packet. It is now a
    conclusive IDENTITY_MISMATCH ⇒ DISPUTED, not a wash-to-CLEAN skip."""
    from dataclasses import replace

    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, manifest, miners = _honest_genesis(store, source, {1: 0.8})

    null_miner = replace(miners[0], hotkey=bad_hotkey)  # nonzero-weight uid, null/empty hk
    shares = build_weight_vector(CFG, (null_miner,), burn_uid=BURN_UID)
    assert shares.get(1, 0.0) > 0.0  # still a nonzero-weight uid
    u16 = quantize_u16(shares)
    bad = EpochLog(
        schema_version=log.schema_version, epoch_id=100, close_block=360_000,
        scorer_version=SCORER, created_at=NOW, burn_uid=BURN_UID, miners=(null_miner,),
        miner_census=_census((null_miner,)),
        weight_shares=shares,
        weight_u16=u16, weight_vector_digest=weight_vector_digest(u16), audit_manifest=manifest,
    )

    report = _auditor(source).audit_epoch(bad, store, NO_SAMPLE, None, NOW)
    v = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id != "crown")
    assert v.verdict is ItemVerdictKind.FAIL and v.code == IDENTITY_MISMATCH
    assert "missing/empty" in v.detail
    assert report.overall is AuditStatus.DISPUTED


# --- B: the fold order and the sentinels are EVIDENCE-BOUND ---------------------------


def _multi_cycle_log(store, source, *, scores_by_seq):
    """A single uid (1) folded over MULTIPLE cycles (seq -> score), honestly."""
    items = []
    for seq, score in scores_by_seq.items():
        b = _bundle(store, source, 1, f"i{seq}", score, seq=seq)
        items.append(_scored(b, 1, score, seq=seq))
    ordered = [scores_by_seq[s] for s in sorted(scores_by_seq)]
    miner = make_miner(1, _fold(0.0, ordered))
    manifest = build_audit_manifest(items, store=store)
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    log = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(miner,),
        burn_uid=BURN_UID, audit_manifest=manifest, now=NOW,
    )
    return log, manifest


def test_reordered_cycle_scores_fold_is_caught(tmp_path) -> None:
    """EWMA is order-dependent: [0.1,0.9]->0.24375 but [0.9,0.1]->0.19375. An authority
    that swaps the VALUES against their committed ordering keys is caught by the
    value cross-check (the packet at seq 0 records 0.1, not 0.9)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, manifest = _multi_cycle_log(store, source, scores_by_seq={0: 0.1, 1: 0.9})
    packets = {c.ordering_key: c.packet_digest for c in manifest.earning_for(1).cycle_scores}

    # Reorder the VALUES while keeping ordering keys ascending: claim seq0->0.9, seq1->0.1
    # (the reverse fold) — but committed packet seq0 records 0.1, so it FAILs.
    swapped = EarningInput(
        cycle_scores=(
            CycleScore(packet_digest=packets[0], ordering_key=0, score=0.9),
            CycleScore(packet_digest=packets[1], ordering_key=1, score=0.1),
        )
    )
    tampered = manifest.model_copy(update={"earning_inputs": {1: swapped}})
    # miner accumulate = the REVERSED fold the authority is trying to sneak in
    miner = make_miner(1, _fold(0.0, [0.9, 0.1]))
    shares = build_weight_vector(CFG, (miner,), burn_uid=BURN_UID)
    u16 = quantize_u16(shares)
    bad = EpochLog(
        schema_version=log.schema_version, epoch_id=100, close_block=360_000,
        scorer_version=SCORER, created_at=NOW, burn_uid=BURN_UID, miners=(miner,),
        miner_census=_census((miner,)),
        weight_shares=shares,
        weight_u16=u16, weight_vector_digest=weight_vector_digest(u16), audit_manifest=tampered,
    )

    report = _auditor(source).audit_epoch(bad, store, NO_SAMPLE, None, NOW)
    v = next(v for v in report.earning_verdicts if v.uid == 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == EARNING_STATE_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_unbacked_padded_zero_is_rejected(tmp_path) -> None:
    """An extra 0.0 cycle (extra decay) that no committed packet backs must FAIL."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, manifest = _multi_cycle_log(store, source, scores_by_seq={0: 0.8})
    real = manifest.earning_for(1).cycle_scores[0]

    from vidaio.audit.canonical import sha256_hex

    padded = EarningInput(
        cycle_scores=(
            real,
            # a 0.0 at seq 1 referencing a packet that is NOT one of uid 1's committed leaves
            CycleScore(packet_digest=sha256_hex(b"ghost-packet"), ordering_key=1, score=0.0),
        )
    )
    tampered = manifest.model_copy(
        update={"earning_inputs": {1: padded}, "fold_cursors": {1: 1}}
    )
    miner = make_miner(1, _fold(0.0, [0.8, 0.0]))  # the extra-decayed value
    shares = build_weight_vector(CFG, (miner,), burn_uid=BURN_UID)
    u16 = quantize_u16(shares)
    bad = EpochLog(
        schema_version=log.schema_version, epoch_id=100, close_block=360_000,
        scorer_version=SCORER, created_at=NOW, burn_uid=BURN_UID, miners=(miner,),
        miner_census=_census((miner,)),
        weight_shares=shares,
        weight_u16=u16, weight_vector_digest=weight_vector_digest(u16), audit_manifest=tampered,
    )

    report = _auditor(source).audit_epoch(bad, store, NO_SAMPLE, None, NOW)
    v = next(v for v in report.earning_verdicts if v.uid == 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == EARNING_STATE_MISMATCH


def test_substituted_exclusion_sentinel_is_rejected(tmp_path) -> None:
    """A -1 exclusion cycle over a packet that records NO exclusion must FAIL."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, manifest = _multi_cycle_log(store, source, scores_by_seq={0: 0.8})
    real = manifest.earning_for(1).cycle_scores[0]

    # Claim the committed (non-excluded) packet at seq 0 is a -1 exclusion.
    substituted = EarningInput(
        cycle_scores=(CycleScore(packet_digest=real.packet_digest, ordering_key=0, score=-1.0),)
    )
    tampered = manifest.model_copy(update={"earning_inputs": {1: substituted}})
    # accumulate -1 = excluded -> zero weight, so give the uid a real weight another way:
    # keep the honest miner (0.2), so weight is nonzero and the uid IS audited.
    report = _auditor(source).audit_epoch(
        log.model_copy(update={"audit_manifest": tampered}), store, NO_SAMPLE, None, NOW
    )
    v = next(v for v in report.earning_verdicts if v.uid == 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == EARNING_STATE_MISMATCH


def test_evidenced_exclusion_then_recovery_passes(tmp_path) -> None:
    """A -1 exclusion BACKED by a committed exclusion packet, then recovery, verifies."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    b0 = _bundle(store, source, 1, "i0", 0.5, seq=0)
    bx = _bundle(store, source, 1, "ix", 0.0, seq=1, excluded=True)  # committed exclusion
    b2 = _bundle(store, source, 1, "i2", 0.8, seq=2)
    items = [
        _scored(b0, 1, 0.5, seq=0),
        _scored(bx, 1, 0.0, seq=1, excluded=True),
        _scored(b2, 1, 0.8, seq=2),
    ]
    acc = _fold(0.0, [0.5, -1.0, 0.8])
    miner = make_miner(1, acc)
    manifest = build_audit_manifest(items, store=store)
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    log = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(miner,),
        burn_uid=BURN_UID, audit_manifest=manifest, now=NOW,
    )
    report = _auditor(source).audit_epoch(log, store, NO_SAMPLE, None, NOW)
    v = next(v for v in report.earning_verdicts if v.uid == 1)
    assert v.verdict is ItemVerdictKind.PASS


def test_dropped_committed_cycle_is_caught(tmp_path) -> None:
    """Omitting a committed low-score cycle to keep accumulate high must FAIL."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, manifest = _multi_cycle_log(store, source, scores_by_seq={0: 0.9, 1: 0.1})
    keep = manifest.earning_for(1).cycle_scores[0]  # only the high cycle

    dropped = EarningInput(cycle_scores=(keep,))
    tampered = manifest.model_copy(update={"earning_inputs": {1: dropped}})
    miner = make_miner(1, _fold(0.0, [0.9]))  # inflated: the 0.1 cycle omitted
    shares = build_weight_vector(CFG, (miner,), burn_uid=BURN_UID)
    u16 = quantize_u16(shares)
    bad = EpochLog(
        schema_version=log.schema_version, epoch_id=100, close_block=360_000,
        scorer_version=SCORER, created_at=NOW, burn_uid=BURN_UID, miners=(miner,),
        miner_census=_census((miner,)),
        weight_shares=shares,
        weight_u16=u16, weight_vector_digest=weight_vector_digest(u16), audit_manifest=tampered,
    )

    report = _auditor(source).audit_epoch(bad, store, NO_SAMPLE, None, NOW)
    v = next(v for v in report.earning_verdicts if v.uid == 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == EARNING_STATE_MISMATCH


#


def test_reordered_fold_via_authority_sequences_is_caught(tmp_path) -> None:
    """The exact round-3 CRITICAL #1 scenario, which the round-2 packet-only binding
    MISSED: the authority reorders the fold by assigning matching ascending sequences to
    the packets it controls. Every packet is internally self-consistent (its own
    cycle_sequence == the CycleScore ordering_key, its own value matches), so the round-2
    checks PASS — but the CHALLENGE-committed dispatch order (fixed pre-scoring, in the
    anchored DAG_REVEAL) does not match, so the reordered fold is now CAUGHT."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()

    # Item A: CHALLENGE-committed dispatch key 0 (folds FIRST), honest score 0.1. Item B:
    # committed key 1 (folds SECOND), honest score 0.9. The honest fold is [0.1, 0.9].
    # The authority wants the REVERSE fold [0.9, 0.1] (a different accumulator), so it
    # stamps packet B's cycle_sequence=0 and packet A's cycle_sequence=1 (the ordering it
    # controls at scoring) — but the committed dispatch keys in the DAG_REVEALs are fixed.
    pa = make_packet(challenge_id="c1", item_id="iA", miner_hotkey="hk1", score=0.1,
                     cycle_sequence=1, metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": 0.1})
    pb = make_packet(challenge_id="c1", item_id="iB", miner_hotkey="hk1", score=0.9,
                     cycle_sequence=0, metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": 0.9})
    ba = make_fake_bundle(store, challenge_id="c1", item_id="iA", miner_hotkey="hk1",
                          packet=pa, dispatch_ordering_key=0)  # committed FIRST
    bb = make_fake_bundle(store, challenge_id="c1", item_id="iB", miner_hotkey="hk1",
                          packet=pb, dispatch_ordering_key=1)  # committed SECOND
    for b in (ba, bb):
        persist_bundle(store, b)
        source.add(b)

    # Build the honest manifest (per_uid refs), then override the earning input with the
    # authority's reordered fold: B first (ordering_key 0), A second (ordering_key 1).
    items = [_scored(ba, 1, 0.1, seq=1), _scored(bb, 1, 0.9, seq=0)]
    manifest = build_audit_manifest(items, store=store)
    reordered = EarningInput(
        cycle_scores=(
            CycleScore(packet_digest=bb.score_packet.digest, ordering_key=0, score=0.9),
            CycleScore(packet_digest=ba.score_packet.digest, ordering_key=1, score=0.1),
        )
    )
    tampered = manifest.model_copy(update={"earning_inputs": {1: reordered}})
    miner = make_miner(1, _fold(0.0, [0.9, 0.1]))  # the reversed accumulator it sneaks in
    shares = build_weight_vector(CFG, (miner,), burn_uid=BURN_UID)
    u16 = quantize_u16(shares)
    bad = EpochLog(
        schema_version=EpochLog.model_fields["schema_version"].default, epoch_id=100,
        close_block=360_000, scorer_version=SCORER, created_at=NOW, burn_uid=BURN_UID,
        miners=(miner,),
        miner_census=_census((miner,)),
        weight_shares=shares, weight_u16=u16, weight_vector_digest=weight_vector_digest(u16),
        audit_manifest=tampered,
    )

    report = _auditor(source).audit_epoch(bad, store, NO_SAMPLE, None, NOW)
    v = next(v for v in report.earning_verdicts if v.uid == 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == EARNING_STATE_MISMATCH
    assert "CHALLENGE-committed dispatch order" in v.detail
    assert report.overall is AuditStatus.DISPUTED


def test_substituted_track_in_earning_path_is_caught(tmp_path) -> None:
    """#9 in the EARNING path: a committed-compression challenge, but the manifest's
    SCORE_PACKET ref stamps committed_track='upscaling' (to force a GPU SKIP / dodge
    recompute). With NO media sampled, only the earning path's challenge-commitment track
    check catches it — IDENTITY_MISMATCH — because the CHALLENGE-committed track is
    'compression' (fixed pre-dispatch in the DAG_REVEAL)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()

    packet = make_packet(challenge_id="c1", item_id="iT", miner_hotkey="hk1", score=0.8,
                         cycle_sequence=0)  # the packet itself declares track=compression
    b = make_fake_bundle(store, challenge_id="c1", item_id="iT", miner_hotkey="hk1",
                         packet=packet, committed_track="compression", dispatch_ordering_key=0)
    persist_bundle(store, b)
    source.add(b)

    # The ScoredItem substitutes the ref's committed_track (upscaling) over a
    # committed-compression challenge — the finalizer stamps it onto the SCORE_PACKET ref.
    item = ScoredItem(
        uid=1, hotkey="hk1", challenge_id="c1", item_id="iT",
        bundle_digest=b.bundle_digest(), packet_digest=b.score_packet.digest,
        committed_track="upscaling", score=0.8, cycle_sequence=0,
    )
    manifest = build_audit_manifest([item], store=store)
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    log = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(make_miner(1, _fold(0.0, [0.8])),),
        burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW,
    )

    report = _auditor(source).audit_epoch(log, store, NO_SAMPLE, None, NOW)
    v = next(v for v in report.earning_verdicts if v.uid == 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == IDENTITY_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


#


def test_omitted_committed_challenge_evidence_for_nonzero_uid_is_inconclusive(tmp_path) -> None:
    """The exact an internal review hole: a misreporting authority OMITS the committed challenge
    evidence (the bundle / DAG_REVEAL / commitment preimage is unresolvable at audit time)
    while publishing self-consistent, packet-bound-backed cycle scores, with ZERO media
    sampled. The packet-bound fallback alone would fold to PASS/CLEAN — never validating the
    committed ordering/track. It must FAIL CLOSED instead: EARNING_STATE_UNVERIFIED (a SKIP)
    for the nonzero uid ⇒ the report is INCONCLUSIVE (HOLD), never CLEAN."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # Honest genesis (packets stored + self-consistent, bundle added to `source`).
    log, _, _ = _honest_genesis(store, source, {1: 0.8})

    # The auditor is handed a source in which the committed challenge evidence is
    # UNRESOLVABLE (the DAG_REVEAL/bundle was omitted) — the packet store is untouched, so
    # the round-2 packet-bound backing still passes; only the round-3 committed binding is
    # unreachable. Fail closed rather than wash to CLEAN.
    unresolvable = InMemoryBundleSource()  # no bundle → committed challenge unresolvable
    report = _auditor(unresolvable).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    v = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id != "crown")
    assert v.verdict is ItemVerdictKind.SKIP
    assert v.code == EARNING_STATE_UNVERIFIED
    # never a provable FAIL (nothing was disproven), and NEVER washed to CLEAN.
    assert not any(v.verdict is ItemVerdictKind.FAIL for v in report.earning_verdicts)
    assert report.overall is AuditStatus.INCONCLUSIVE

    # Contrast: the SAME log, audited against the RESOLVABLE source, verifies the committed
    # binding and re-folds cleanly → CLEAN (the fail-closed rule is specific to
    # unresolvability, it does not spuriously hold an honest, resolvable epoch).
    honest_report = _auditor(source).audit_epoch(log, store, NO_SAMPLE, None, NOW)
    hv = next(v for v in honest_report.earning_verdicts if v.uid == 1 and v.item_id != "crown")
    assert hv.verdict is ItemVerdictKind.PASS
    assert honest_report.overall is AuditStatus.CLEAN


def test_unreadable_dag_reveal_for_nonzero_uid_is_inconclusive(tmp_path) -> None:
    """A resolvable bundle whose DAG_REVEAL bytes are gone from the store (so the committed
    preimage cannot be read/verified) is still UNVERIFIED for a nonzero uid — fail closed,
    not a packet-bound-fallback PASS."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    b = _bundle(store, source, 1, "i1", 0.8)  # bundle stays resolvable via `source`
    items = [_scored(b, 1, 0.8)]
    manifest = build_audit_manifest(items, store=store)
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    log = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(make_miner(1, _fold(0.0, [0.8])),),
        burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW,
    )
    # Delete the DAG_REVEAL artifact from the store: the bundle ref still points at it, but
    # its committed preimage is now unreadable → committed binding UNRESOLVABLE.
    store._path(ArtifactKind.DAG_REVEAL, b.dag_reveal.digest).unlink()

    report = _auditor(source).audit_epoch(log, store, NO_SAMPLE, None, NOW)
    v = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id != "crown")
    assert v.verdict is ItemVerdictKind.SKIP and v.code == EARNING_STATE_UNVERIFIED
    assert report.overall is AuditStatus.INCONCLUSIVE


# --- C/D: earning FAIL/SKIP factor into the overall verdict ---------------------------


def test_earning_fail_never_derives_clean(tmp_path) -> None:
    """A report carrying an earning FAIL can never derive CLEAN (even weight PASS)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, manifest, miners = _honest_genesis(store, source, {1: 0.8})
    # substitute uid 1's accumulate; weights follow, only earning disputes.
    fab_miners = (make_miner(1, 0.95),)
    shares = build_weight_vector(CFG, fab_miners, burn_uid=BURN_UID)
    u16 = quantize_u16(shares)
    fab = EpochLog(
        schema_version=log.schema_version, epoch_id=100, close_block=360_000,
        scorer_version=SCORER, created_at=NOW, burn_uid=BURN_UID, miners=fab_miners,
        miner_census=_census(fab_miners),
        weight_shares=shares,
        weight_u16=u16, weight_vector_digest=weight_vector_digest(u16), audit_manifest=manifest,
    )
    report = _auditor(source).audit_epoch(fab, store, NO_SAMPLE, None, NOW)
    assert any(v.verdict is ItemVerdictKind.FAIL for v in report.earning_verdicts)
    assert report.overall is AuditStatus.DISPUTED


def test_missing_earning_input_for_nonzero_uid_at_genesis_is_disputed(tmp_path) -> None:
    """A nonzero-weight uid at GENESIS whose earning state the log does not carry is a
    provable FAIL (DISPUTED), never a SKIP that washes CLEAN (#2/D, round-20 #2).

    Round-20 #2 routes a nonzero-weight uid with NO EarningInput through the carry-forward
    chain check (it may be a legitimate idle prior earner). At GENESIS there is no prior to
    carry from, so a positive accumulator with no committed fold cannot exist — it is a
    substituted accumulator ⇒ FAIL (a strictly stronger verdict than the old INCONCLUSIVE)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, manifest, miners = _honest_genesis(store, source, {1: 0.8})
    # Strip the earning input for uid 1 (an untrusted/legacy log the finalizer would refuse,
    # but the auditor must still flag rather than wash clean).
    stripped = manifest.model_copy(update={"earning_inputs": {}})
    untrusted = log.model_copy(update={"audit_manifest": stripped})

    report = _auditor(source).audit_epoch(untrusted, store, NO_SAMPLE, None, NOW)
    v = next(v for v in report.earning_verdicts if v.uid == 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == EARNING_STATE_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


# --- carry-in chaining across epochs (back to genesis) -------------------------------


def _epoch(store, source, *, epoch_id, close_block, per_uid, priors, prior_log, seq=0):
    items, miners = [], []
    # The committed dispatch ordering_key is MONOTONIC per uid across epochs (the producer only
    # folds a packet whose key exceeds the highest already folded), so a LATER epoch chaining the
    # SAME uid must use a strictly higher `seq` than the prior epoch — otherwise the auditor's
    # cross-epoch REPLAY guard (round-22 #1) correctly flags the shared ordering_key as a re-fold.
    # Default 0 for a single-epoch fixture; multi-epoch same-uid chains pass increasing `seq`.
    for uid, (score, acc) in per_uid.items():
        b = _bundle(store, source, uid, f"e{epoch_id}i{uid}", score, seq=seq)
        items.append(_scored(b, uid, score, seq=seq))
        miners.append(make_miner(uid, acc))
    manifest = build_audit_manifest(
        items,
        store=store,
        prior_accumulate=priors,
        prior_fold_cursors=(
            prior_log.audit_manifest.fold_cursors if prior_log is not None else {}
        ),
    )
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    return fin.build_log(
        epoch_id=epoch_id, close_block=close_block, snapshots=tuple(miners),
        burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW,
        prior_log_digest=prior_log.log_digest() if prior_log is not None else None,
    )


def test_carry_in_chains_against_the_prior_log(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid={1: (0.8, _fold(0.0, [0.8]))}, priors={}, prior_log=None,
    )
    carry = _fold(0.0, [0.8])
    cur = _epoch(
        store, source, epoch_id=100, close_block=360_000,
        per_uid={1: (0.5, _fold(carry, [0.5]))}, priors={1: carry}, prior_log=prior, seq=1,
    )

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)
    uid_v = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id != "crown")
    assert uid_v.verdict is ItemVerdictKind.PASS
    assert report.overall is AuditStatus.CLEAN


def test_prior_epoch_packet_replay_is_disputed_even_when_fold_and_weights_match(tmp_path) -> None:
    """an internal review (CRITICAL): a genuine packet already folded in a PRIOR epoch, re-folded
    in a later epoch, is DISPUTED even though backing + fold + carry-in + weights all match.

    The committed dispatch ordering_key is MONOTONIC per uid; the replayed packet's key is
    at/below the prior epoch's max folded key, so the auditor rejects the re-fold
    (EARNING_PACKET_REPLAY) before it can inflate the accumulator and double-award the work."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid={1: (0.8, _fold(0.0, [0.8]))}, priors={}, prior_log=None, seq=5,
    )
    carry = _fold(0.0, [0.8])  # uid 1 ended the prior epoch here (its cycle folded at key 5)
    # E+1 REPLAYS a cycle at key 3 (<= the prior max key 5) — a re-fold of already-counted work
    # that inflates the accumulator. The carry-in (== the prior accumulator) chains correctly and
    # the fold reproduces the stated accumulate_score, so ONLY the replay guard catches it.
    b = _bundle(store, source, 1, "e100i1replay", 0.5, seq=3)
    manifest = build_audit_manifest([_scored(b, 1, 0.5, seq=3)], store=store, prior_accumulate={1: carry})
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(make_miner(1, _fold(carry, [0.5])),),
        burn_uid=BURN_UID, audit_manifest=manifest, now=NOW, prior_log_digest=prior.log_digest(),
    )

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)

    v1 = next(v for v in report.earning_verdicts if v.uid == 1 and v.source == "earning")
    assert v1.verdict is ItemVerdictKind.FAIL and v1.code == EARNING_PACKET_REPLAY
    assert report.overall is AuditStatus.DISPUTED


def test_reactivation_after_carry_only_epoch_uses_cumulative_watermark(tmp_path) -> None:
    """Schema v11 carries the prior boundary through an idle epoch, so an honest later cycle
    is provably non-replay and CLEAN rather than permanently INCONCLUSIVE."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # Prior epoch: uid 1 carried FORWARD (positive accumulator, NO earning input) alongside uid 2.
    prior_acc1 = _fold(0.0, [0.8])
    b2 = _bundle(store, source, 2, "e99i2", 0.7, seq=1)
    manifest99 = build_audit_manifest(
        [_scored(b2, 2, 0.7, seq=1)], store=store, prior_accumulate={2: 0.0},
        prior_fold_cursors={1: 0},
    )
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    prior = fin.build_log(
        epoch_id=99, close_block=359_640,
        snapshots=(make_miner(1, prior_acc1), make_miner(2, _fold(0.0, [0.7]))),
        burn_uid=BURN_UID, audit_manifest=manifest99, now=NOW,
        prior_log_digest="c" * 64, prior_earning={1: ("hk1", prior_acc1)},
    )
    assert prior.audit_manifest.earning_for(1) is None  # uid 1 carried forward, no EI

    assert prior.audit_manifest.fold_cursors[1] == 0
    # E+1: uid 1 is ACTIVE again above its carried boundary.
    b1 = _bundle(store, source, 1, "e100i1", 0.5, seq=2)
    manifest100 = build_audit_manifest(
        [_scored(b1, 1, 0.5, seq=2)], store=store,
        prior_accumulate={1: prior_acc1},
        prior_fold_cursors=prior.audit_manifest.fold_cursors,
    )
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(make_miner(1, _fold(prior_acc1, [0.5])),),
        burn_uid=BURN_UID, audit_manifest=manifest100, now=NOW, prior_log_digest=prior.log_digest(),
    )

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)

    v1 = next(v for v in report.earning_verdicts if v.uid == 1 and v.source == "earning")
    assert v1.verdict is ItemVerdictKind.PASS
    assert report.overall is AuditStatus.CLEAN


def test_first_fold_after_explicit_null_cursor_is_clean(tmp_path) -> None:
    """An observed uid with no prior cycle has a conclusive, auditable first fold."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    prior_manifest = build_audit_manifest(
        (), prior_fold_cursors={}, current_census_uids=(1,)
    )
    prior = fin.build_log(
        epoch_id=99,
        close_block=359_640,
        snapshots=(make_miner(1, 0.0),),
        burn_uid=BURN_UID,
        audit_manifest=prior_manifest,
        now=NOW,
        prior_fold_cursors={},
    )
    assert prior.audit_manifest.fold_cursors == {1: None}

    bundle = _bundle(store, source, 1, "e100-first", 0.8, seq=0)
    manifest = build_audit_manifest(
        [_scored(bundle, 1, 0.8, seq=0)],
        store=store,
        prior_accumulate={1: 0.0},
        prior_fold_cursors=prior.audit_manifest.fold_cursors,
        current_census_uids=(1,),
    )
    cur = fin.build_log(
        epoch_id=100,
        close_block=360_000,
        snapshots=(make_miner(1, _fold(0.0, [0.8])),),
        burn_uid=BURN_UID,
        audit_manifest=manifest,
        now=NOW,
        prior_log_digest=prior.log_digest(),
        prior_fold_cursors=prior.audit_manifest.fold_cursors,
    )

    report = _auditor(source).audit_epoch(
        cur, store, NO_SAMPLE, None, NOW, prior_log=prior, is_genesis=False
    )
    first_fold = next(
        verdict
        for verdict in report.earning_verdicts
        if verdict.uid == 1 and verdict.item_id == "uid:1"
    )
    assert first_fold.verdict is ItemVerdictKind.PASS
    assert report.overall is AuditStatus.CLEAN


def test_replay_after_excluded_carry_only_epoch_is_not_clean(tmp_path) -> None:
    """an internal review (CRITICAL): a uid that earned, was EXCLUDED (-1 sentinel), then carried
    the -1 forward through an epoch with NO earning input, must NOT audit CLEAN when it re-folds an
    old packet — non-replay is UNPROVABLE (its watermark is not in the prior log) ⇒ HOLD.

    The -1 exclusion sentinel is a CONTINUING identity (the uid already earned), not a genuinely
    new uid; accumulate() restarts from 0.0 after exclusion, so refolding an earlier packet over
    -1 would re-award it. The unavailable-watermark path must fail closed to INCONCLUSIVE for ANY
    continuing identity, not PASS just because the accumulator is non-positive."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # Prior epoch: uid 1 is EXCLUDED (accumulator latched to the -1 sentinel) and carried FORWARD
    # with NO earning input; uid 2 keeps the epoch non-empty with fresh evidence.
    excluded_acc = _fold(_fold(0.0, [0.8]), [-1.0])  # earned, then excluded → -1
    b2 = _bundle(store, source, 2, "e99i2", 0.7, seq=1)
    manifest99 = build_audit_manifest(
        [_scored(b2, 2, 0.7, seq=1)], store=store, prior_accumulate={2: 0.0},
        prior_fold_cursors={1: 0},
    )
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    prior = fin.build_log(
        epoch_id=99, close_block=359_640,
        snapshots=(make_miner(1, excluded_acc), make_miner(2, _fold(0.0, [0.7]))),
        burn_uid=BURN_UID, audit_manifest=manifest99, now=NOW,
        prior_log_digest="d" * 64, prior_earning={1: ("hk1", excluded_acc)},
    )
    assert prior.audit_manifest.earning_for(1) is None  # carried -1 forward, no EI
    assert prior.audit_manifest.fold_cursors[1] == 0

    # E+1: uid 1 RE-FOLDS an old packet (key 0) over the -1 restart — a replay of already-awarded
    # work. The fold reproduces the stated accumulator and the carry-in chains, so ONLY the
    # non-replay guard can catch it; the carried watermark makes the replay conclusive.
    b1 = _bundle(store, source, 1, "e100i1", 0.5, seq=0)
    manifest100 = build_audit_manifest([_scored(b1, 1, 0.5, seq=0)], store=store, prior_accumulate={1: 0.0})
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(make_miner(1, _fold(0.0, [0.5])),),
        burn_uid=BURN_UID, audit_manifest=manifest100, now=NOW, prior_log_digest=prior.log_digest(),
    )

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)

    v1 = next(v for v in report.earning_verdicts if v.uid == 1 and v.source == "earning")
    assert v1.verdict is ItemVerdictKind.FAIL and v1.code == EARNING_PACKET_REPLAY
    assert report.overall is AuditStatus.DISPUTED


@pytest.mark.parametrize("omission_epochs", [1, 3])
def test_replay_after_excluded_identity_omitted_for_one_epoch_is_not_clean(
    tmp_path, omission_epochs: int
) -> None:
    """Round 24 + generalized omission: a census gap cannot erase replay history.

    uid 1 earns at key 0 and is then excluded at key 1.  The authority omits it from one
    (and, parametrically, several) chained censuses, then reintroduces it and re-folds the
    old key-0 packet over a zero restart.  Every omitted epoch carries the anchored v11
    tombstone, so the eventual replay is conclusively rejected regardless of accumulator
    sign or omission duration.
    """
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    fin = EpochFinalizer(CFG, scorer_version=SCORER)

    earned = _bundle(store, source, 1, "e98-earned", 0.8, seq=0)
    excluded = _bundle(store, source, 1, "e98-excluded", 0.0, seq=1, excluded=True)
    keeper = _bundle(store, source, 2, "e98-keeper", 0.7, seq=0)
    history_manifest = build_audit_manifest(
        [
            _scored(earned, 1, 0.8, seq=0),
            _scored(excluded, 1, 0.0, seq=1, excluded=True),
            _scored(keeper, 2, 0.7, seq=0),
        ],
        store=store,
        prior_fold_cursors={},
    )
    excluded_acc = _fold(0.0, [0.8, -1.0])
    keeper_acc = _fold(0.0, [0.7])
    prior = fin.build_log(
        epoch_id=98,
        close_block=359_280,
        snapshots=(make_miner(1, excluded_acc), make_miner(2, keeper_acc)),
        burn_uid=BURN_UID,
        audit_manifest=history_manifest,
        now=NOW,
        prior_fold_cursors={},
    )
    assert prior.audit_manifest.fold_cursors[1] == 1

    for offset in range(omission_epochs):
        carried = build_audit_manifest(
            (), prior_fold_cursors=prior.audit_manifest.fold_cursors
        )
        prior = fin.build_log(
            epoch_id=99 + offset,
            close_block=359_640 + 360 * offset,
            snapshots=(make_miner(2, keeper_acc),),  # uid 1 deliberately omitted
            burn_uid=BURN_UID,
            audit_manifest=carried,
            now=NOW,
            prior_log_digest=prior.log_digest(),
            prior_earning={2: ("hk2", keeper_acc)},
            prior_fold_cursors=prior.audit_manifest.fold_cursors,
        )
        assert prior.audit_manifest.fold_cursors[1] == 1

    # Re-use the exact previously-folded packet at key 0.  A dishonest producer can bypass
    # its own refusal, but cannot make the auditor forget the anchored predecessor tombstone.
    replay_manifest = build_audit_manifest(
        [_scored(earned, 1, 0.8, seq=0)],
        store=store,
        prior_accumulate={1: 0.0},
    ).model_copy(update={"fold_cursors": prior.audit_manifest.fold_cursors})
    cur = fin.build_log(
        epoch_id=99 + omission_epochs,
        close_block=359_640 + 360 * omission_epochs,
        snapshots=(make_miner(1, _fold(0.0, [0.8])), make_miner(2, keeper_acc)),
        burn_uid=BURN_UID,
        audit_manifest=replay_manifest,
        now=NOW,
        prior_log_digest=prior.log_digest(),
        prior_earning={2: ("hk2", keeper_acc)},
    )

    report = _auditor(source).audit_epoch(
        cur, store, NO_SAMPLE, None, NOW, prior_log=prior, is_genesis=False
    )

    replay = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id == "uid:1")
    assert replay.verdict is ItemVerdictKind.FAIL and replay.code == EARNING_PACKET_REPLAY
    assert report.overall is AuditStatus.DISPUTED


def test_hotkey_ping_pong_cannot_reset_uid_fold_cursor(tmp_path) -> None:
    """Miner-reachable A -> B -> A registration churn does not reopen A's old packet."""
    from dataclasses import replace

    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    fin = EpochFinalizer(CFG, scorer_version=SCORER)

    old = _bundle(store, source, 1, "e98-a", 0.8, seq=5)
    keeper0 = _bundle(store, source, 2, "e98-keeper", 0.7, seq=0)
    first_manifest = build_audit_manifest(
        [_scored(old, 1, 0.8, seq=5), _scored(keeper0, 2, 0.7, seq=0)],
        store=store,
        prior_fold_cursors={},
    )
    a_acc, keeper_acc = _fold(0.0, [0.8]), _fold(0.0, [0.7])
    first_a = fin.build_log(
        epoch_id=98,
        close_block=359_280,
        snapshots=(make_miner(1, a_acc), make_miner(2, keeper_acc)),
        burn_uid=BURN_UID,
        audit_manifest=first_manifest,
        now=NOW,
        prior_fold_cursors={},
    )

    # The uid is genuinely re-registered to B; its numeric accumulator resets, but the uid-slot
    # replay boundary remains 5 while another miner supplies this epoch's new evidence.
    keeper1 = _bundle(store, source, 2, "e99-keeper", 0.7, seq=1)
    b_manifest = build_audit_manifest(
        [_scored(keeper1, 2, 0.7, seq=1)],
        store=store,
        prior_accumulate={2: keeper_acc},
        prior_fold_cursors=first_a.audit_manifest.fold_cursors,
    )
    hotkey_b = replace(
        make_miner(1, 0.0), hotkey="hkB", coldkey="ckB", ip="10.0.1.1"
    )
    keeper_acc_1 = _fold(keeper_acc, [0.7])
    middle_b = fin.build_log(
        epoch_id=99,
        close_block=359_640,
        snapshots=(hotkey_b, make_miner(2, keeper_acc_1)),
        burn_uid=BURN_UID,
        audit_manifest=b_manifest,
        now=NOW,
        prior_log_digest=first_a.log_digest(),
        prior_fold_cursors=first_a.audit_manifest.fold_cursors,
    )
    assert middle_b.audit_manifest.fold_cursors[1] == 5

    # A registers again and reuses its original key-5 packet.  Numeric carry-in 0 is correct for
    # B -> A, so only the uid-slot replay watermark exposes the double award.
    replay_manifest = build_audit_manifest(
        [_scored(old, 1, 0.8, seq=5)],
        store=store,
        prior_accumulate={1: 0.0},
    ).model_copy(update={"fold_cursors": middle_b.audit_manifest.fold_cursors})
    back_to_a = fin.build_log(
        epoch_id=100,
        close_block=360_000,
        snapshots=(make_miner(1, a_acc), make_miner(2, keeper_acc_1)),
        burn_uid=BURN_UID,
        audit_manifest=replay_manifest,
        now=NOW,
        prior_log_digest=middle_b.log_digest(),
        prior_earning={2: ("hk2", keeper_acc_1)},
    )

    report = _auditor(source).audit_epoch(
        back_to_a, store, NO_SAMPLE, None, NOW, prior_log=middle_b, is_genesis=False
    )

    replay = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id == "uid:1")
    assert replay.verdict is ItemVerdictKind.FAIL and replay.code == EARNING_PACKET_REPLAY
    assert report.overall is AuditStatus.DISPUTED


def test_deregistered_uid_fold_cursor_tombstone_cannot_be_deleted(tmp_path) -> None:
    """A real deregistration removes the census row, not the cumulative replay tombstone."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior = _epoch(
        store,
        source,
        epoch_id=99,
        close_block=359_640,
        per_uid={1: (0.8, _fold(0.0, [0.8]))},
        priors={},
        prior_log=None,
        seq=4,
    )
    # uid 1 is genuinely absent from the current metagraph/census; uid 2 is new.  Dropping
    # uid 1 from the watermark map is still tampering because it would let the slot replay key 4
    # if uid 1 later returned.
    b2 = _bundle(store, source, 2, "e100i2", 0.7, seq=0)
    dropped = build_audit_manifest([_scored(b2, 2, 0.7, seq=0)], store=store)
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    cur = fin.build_log(
        epoch_id=100,
        close_block=360_000,
        snapshots=(make_miner(2, _fold(0.0, [0.7])),),
        burn_uid=BURN_UID,
        audit_manifest=dropped,
        now=NOW,
        prior_log_digest=prior.log_digest(),
    )

    report = _auditor(source).audit_epoch(
        cur, store, NO_SAMPLE, None, NOW, prior_log=prior, is_genesis=False
    )

    watermark = next(
        v for v in report.earning_verdicts if v.item_id == "fold-cursor:1"
    )
    assert watermark.verdict is ItemVerdictKind.FAIL
    assert watermark.code == FOLD_CURSOR_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_substituted_carry_in_is_caught_when_chained(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid={1: (0.8, _fold(0.0, [0.8]))}, priors={}, prior_log=None,
    )
    # Claim a carry-in of 0.9 (the prior epoch actually ended at ~0.2); the fold is made
    # internally consistent, but chaining against the real prior log exposes the lie.
    lie = 0.9
    cur = _epoch(
        store, source, epoch_id=100, close_block=360_000,
        per_uid={1: (0.5, _fold(lie, [0.5]))}, priors={1: lie}, prior_log=prior, seq=1,
    )

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)
    uid_v = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id != "crown")
    assert uid_v.verdict is ItemVerdictKind.FAIL and uid_v.code == EARNING_STATE_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_referenced_but_unavailable_prior_zero_carry_in_is_inconclusive(tmp_path) -> None:
    """an internal review: when the log EXPLICITLY references a prior epoch (prior_log_digest
    set) whose log could not be loaded, a ZERO carry-in is UNVERIFIABLE, NOT safe to PASS —
    removing the prior object is exactly how earnings would be reset/censored to zero. It
    is an earning SKIP (EARNING_STATE_UNVERIFIED) -> INCONCLUSIVE, never a false CLEAN."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid={1: (0.8, _fold(0.0, [0.8]))}, priors={}, prior_log=None,
    )
    # `cur` references `prior` (prior_log_digest set) but declares a ZERO carry-in.
    cur = _epoch(
        store, source, epoch_id=100, close_block=360_000,
        per_uid={1: (0.8, _fold(0.0, [0.8]))}, priors={1: 0.0}, prior_log=prior, seq=1,
    )
    # Audit WITHOUT supplying prior_log -> the referenced prior is unavailable.
    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=None)
    uid_v = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id != "crown")
    assert uid_v.verdict is ItemVerdictKind.SKIP and uid_v.code == EARNING_STATE_UNVERIFIED
    assert report.overall is AuditStatus.INCONCLUSIVE


def test_genuine_genesis_zero_carry_in_still_clean(tmp_path) -> None:
    """an internal review: a GENUINE genesis (prior_log_digest is None) with a zero carry-in
    still PASSes/CLEANs — only a referenced-but-unavailable prior is unverifiable, so the
    fix does not over-block honest genesis epochs."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    genesis = _epoch(
        store, source, epoch_id=100, close_block=360_000,
        per_uid={1: (0.8, _fold(0.0, [0.8]))}, priors={}, prior_log=None,
    )
    report = _auditor(source).audit_epoch(genesis, store, NO_SAMPLE, None, NOW, prior_log=None)
    uid_v = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id != "crown")
    assert uid_v.verdict is ItemVerdictKind.PASS
    assert report.overall is AuditStatus.CLEAN


#


def test_omitted_prior_digest_at_non_genesis_disputes(tmp_path) -> None:
    """an internal review: a log that OMITS prior_log_digest (None) at an epoch the LOOP knows is
    NON-genesis is a broken chain — omitting the digest resets the earning carry-in to zero at
    an arbitrary epoch. The auditor threads is_genesis=False from the loop ⇒ DISPUTED, NOT
    re-treated as genesis (round-8 #6 only closed 'digest present but prior unavailable')."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # A self-consistent log with a ZERO carry-in and NO prior_log_digest (looks like genesis).
    tampered = _epoch(
        store, source, epoch_id=500, close_block=360_000,
        per_uid={1: (0.8, _fold(0.0, [0.8]))}, priors={}, prior_log=None,
    )
    assert tampered.prior_log_digest is None

    # The loop KNOWS epoch 500 is not the genesis floor -> is_genesis=False.
    report = _auditor(source).audit_epoch(
        tampered, store, NO_SAMPLE, None, NOW, prior_log=None, is_genesis=False
    )
    uid_v = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id != "crown")
    assert uid_v.verdict is ItemVerdictKind.FAIL and uid_v.code == EARNING_STATE_MISMATCH
    assert "OMITS prior_log_digest" in uid_v.detail
    assert report.overall is AuditStatus.DISPUTED


def test_true_genesis_with_none_digest_still_clean(tmp_path) -> None:
    """an internal review: the SAME log at the genuine genesis floor (is_genesis=True) still
    CLEANs — the fix distinguishes a true genesis from a deliberately-omitted prior, so honest
    genesis epochs are not over-blocked."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    genesis = _epoch(
        store, source, epoch_id=500, close_block=360_000,
        per_uid={1: (0.8, _fold(0.0, [0.8]))}, priors={}, prior_log=None,
    )
    report = _auditor(source).audit_epoch(
        genesis, store, NO_SAMPLE, None, NOW, prior_log=None, is_genesis=True
    )
    uid_v = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id != "crown")
    assert uid_v.verdict is ItemVerdictKind.PASS
    # This fixture intentionally has no reward window; schema-v14 window verdicts are covered in
    # the dedicated competition-cycle tests.
    assert report.overall is AuditStatus.CLEAN


#


def _empty_burn_log(*, epoch_id, close_block, prior_log_digest):
    """A canonical EMPTY burn log (miners=[], {burn_uid:1.0}) with the given prior digest."""
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    return fin.build_log(
        epoch_id=epoch_id, close_block=close_block, snapshots=(),
        burn_uid=BURN_UID, audit_manifest=AuditManifest(), now=NOW,
        prior_log_digest=prior_log_digest,
    )


def _burn_auditor(source, *, chain=None):
    """An Auditor whose canonical burn_uid == the fixtures' BURN_UID (so an honest burn log is
    not spuriously flagged BURN_UID_MISMATCH — isolating the predecessor-chain verdict)."""
    return Auditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID),
        source, chain=chain,
    )


def test_non_genesis_empty_burn_cannot_break_prior_chain_and_hide_active_reset(tmp_path) -> None:
    """an internal review (CRITICAL): a NON-genesis EMPTY burn log that BREAKS the predecessor
    chain (omits prior_log_digest) is DISPUTED, even though it selects NO earning uid.

    An empty canonical burn log (miners=[], {burn_uid:1.0}) selects no earning uid, so every
    per-uid carry-in / carry-forward / reset check is skipped. Before round-20 #1 a non-genesis
    authority could OMIT prior_log_digest, publish the empty burn vector, and audit CLEAN —
    silently RESETTING a still-registered prior earner's accrued earnings (uid 1 here) and, via
    the own-audit gate, advancing the cursor past the erased history. The LOG-LEVEL predecessor
    check fires regardless of the audited set: an omitted digest at a non-genesis epoch ⇒ FAIL
    PREDECESSOR_CHAIN_BROKEN ⇒ DISPUTED."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # A prior epoch where uid 1 accrued a positive accumulator.
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid={1: (0.8, _fold(0.0, [0.8]))}, priors={}, prior_log=None,
    )
    # The current empty-burn log OMITS prior_log_digest at a NON-genesis epoch (the chain reset).
    tampered = _empty_burn_log(epoch_id=100, close_block=360_000, prior_log_digest=None)
    assert tampered.prior_log_digest is None and tampered.burn_uid == BURN_UID and not tampered.miners

    # uid 1 is STILL registered in the close-block metagraph — its earnings are being erased.
    chain = metagraph_chain([make_miner(1, _fold(0.0, [0.8]))], close_block=360_000)
    auditor = _burn_auditor(source, chain=chain)
    report = auditor.audit_epoch(
        tampered, store, NO_SAMPLE, None, NOW, prior_log=prior, is_genesis=False
    )
    pv = next(v for v in report.earning_verdicts if v.item_id == "predecessor-chain")
    assert pv.verdict is ItemVerdictKind.FAIL and pv.code == PREDECESSOR_CHAIN_BROKEN
    assert report.overall is AuditStatus.DISPUTED


def test_non_genesis_empty_burn_referencing_unavailable_prior_is_inconclusive(tmp_path) -> None:
    """an internal review: an empty burn log that REFERENCES a predecessor (prior_log_digest set)
    which cannot be loaded is UNVERIFIABLE at the log level ⇒ INCONCLUSIVE (HOLD), never a CLEAN
    that would advance the cursor past an unverifiable reset (round-8 #6 at the log level)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid={1: (0.8, _fold(0.0, [0.8]))}, priors={}, prior_log=None,
    )
    # References the real prior digest, but the prior is NOT supplied (unavailable / pruned).
    tampered = _empty_burn_log(
        epoch_id=100, close_block=360_000, prior_log_digest=prior.log_digest()
    )
    report = _burn_auditor(source).audit_epoch(
        tampered, store, NO_SAMPLE, None, NOW, prior_log=None, is_genesis=False
    )
    pv = next(v for v in report.earning_verdicts if v.item_id == "predecessor-chain")
    assert pv.verdict is ItemVerdictKind.SKIP and pv.code == PREDECESSOR_UNVERIFIED
    assert report.overall is AuditStatus.INCONCLUSIVE


def test_genuine_empty_epoch_maintaining_the_chain_is_clean(tmp_path) -> None:
    """an internal review: a GENUINELY empty epoch that MAINTAINS the chain (valid
    prior_log_digest matching an available prior) is CLEAN — only BREAKING/censoring the chain
    is faulted, so honest empty epochs are not over-blocked."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior = _empty_burn_log(epoch_id=99, close_block=359_640, prior_log_digest=None)
    cur = _empty_burn_log(
        epoch_id=100, close_block=360_000, prior_log_digest=prior.log_digest()
    )
    # A metagraph with the block clock anchored so the created_at binding PASSES (an empty
    # census has nothing else to bind); production always has one wired.
    chain = metagraph_chain([], close_block=360_000, close_block_time=NOW)
    report = _burn_auditor(source, chain=chain).audit_epoch(
        cur, store, NO_SAMPLE, None, NOW, prior_log=prior, is_genesis=False
    )
    assert not any(v.item_id == "predecessor-chain" for v in report.earning_verdicts)
    assert report.overall is AuditStatus.CLEAN


#; carry-in is (uid, hotkey)-keyed


def test_substituted_accumulator_on_zero_weight_miner_disputes(tmp_path) -> None:
    """an internal review(a): a ZERO-weight miner's accumulator is now RE-DERIVED from evidence.

    top_n_per_track is 5: uids 1..5 (score 0.9) make the top-5; uid 6 (real committed evidence
    0.1) ranks below the cutoff => ZERO weight. Its STATED accumulator is SUBSTITUTED (it does not
    fold from its committed packet). A zero-weight miner used to be entirely unaudited, so this
    substitution rode free until uid 6 later entered top-N and the carry-in check accepted the
    prior STATED accumulator verbatim — re-attributing it into paid weight. The auditor now re-folds
    every evidenced uid regardless of weight => EARNING_STATE_MISMATCH => DISPUTED."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    per_uid = {uid: (0.9, _fold(0.0, [0.9])) for uid in range(1, 6)}
    per_uid[6] = (0.1, _fold(0.0, [0.5]))  # evidence 0.1, but stated accumulator = fold([0.5])
    log = _epoch(
        store, source, epoch_id=100, close_block=360_000,
        per_uid=per_uid, priors={}, prior_log=None,
    )
    assert log.weight_shares.get(6, 0.0) == 0.0  # uid 6 is a zero-weight rank loser

    report = _auditor(source).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    v6 = next(v for v in report.earning_verdicts if v.uid == 6 and v.item_id != "crown")
    assert v6.verdict is ItemVerdictKind.FAIL and v6.code == EARNING_STATE_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_honest_zero_weight_carry_forward_stays_clean(tmp_path) -> None:
    """an internal review(a): an HONEST zero-weight miner that carries a positive accumulator
    forward with NO new evidence this epoch (a real below-cutoff loser not re-scored) is verified
    as a chained carry-forward of the prior epoch's value => PASS (no false-HOLD)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # Prior epoch: uids 1..5 (0.9) + uid 6 (0.1). uid 6 earns fold([0.1]) but ranks below top-5.
    prior_per_uid = {uid: (0.9, _fold(0.0, [0.9])) for uid in range(1, 6)}
    prior_per_uid[6] = (0.1, _fold(0.0, [0.1]))
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid=prior_per_uid, priors={}, prior_log=None,
    )
    # Current epoch: uids 1..5 re-scored; uid 6 carries its SAME accumulator forward with NO new
    # cycle (no EarningInput) and stays a zero-weight loser.
    items, miners = [], []
    top_prior = {uid: _fold(0.0, [0.9]) for uid in range(1, 6)}
    for uid in range(1, 6):
        b = _bundle(store, source, uid, f"e100i{uid}", 0.9, seq=1)
        items.append(_scored(b, uid, 0.9, seq=1))
        miners.append(make_miner(uid, _fold(top_prior[uid], [0.9])))
    carried = _fold(0.0, [0.1])
    miners.append(make_miner(6, carried))  # uid 6 present, positive accumulator, NO new item
    manifest = build_audit_manifest(
        items, store=store, prior_accumulate=top_prior,
        prior_fold_cursors=prior.audit_manifest.fold_cursors,
    )
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=tuple(miners),
        burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW, prior_log_digest=prior.log_digest(),
    )
    assert cur.weight_shares.get(6, 0.0) == 0.0
    assert cur.audit_manifest.earning_for(6) is None  # no committed evidence this epoch

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)

    v6 = next(v for v in report.earning_verdicts if v.uid == 6 and v.item_id != "crown")
    assert v6.verdict is ItemVerdictKind.PASS
    assert report.overall is AuditStatus.CLEAN


def test_honest_nonzero_weight_carry_forward_stays_clean(tmp_path) -> None:
    """an internal review: an ACTIVE (NONZERO-weight) prior earner that IDLES this epoch — carries
    its accumulator forward with NO new packet (no EarningInput) but stays weighted — is verified
    as a chained carry-forward ⇒ PASS, not the pre-round-20 INCONCLUSIVE-HOLD. The auditor side of
    the carry-forward fix (it routes a nonzero-weight no-EI uid through `_carry_forward_verdict`)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior_per_uid = {uid: (0.9, _fold(0.0, [0.9])) for uid in range(1, 6)}
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid=prior_per_uid, priors={}, prior_log=None,
    )
    # Current epoch: uids 1..4 re-scored; uid 5 IDLES — carries its accumulator forward with NO
    # new cycle (no EarningInput) yet stays NONZERO-weight (still within the top-N pool).
    items, miners = [], []
    top_prior = {uid: _fold(0.0, [0.9]) for uid in range(1, 6)}
    for uid in range(1, 5):
        b = _bundle(store, source, uid, f"e100i{uid}", 0.9, seq=1)
        items.append(_scored(b, uid, 0.9, seq=1))
        miners.append(make_miner(uid, _fold(top_prior[uid], [0.9])))
    miners.append(make_miner(5, top_prior[5]))  # nonzero-weight idle earner, NO new item
    manifest = build_audit_manifest(
        items, store=store, prior_accumulate={u: top_prior[u] for u in range(1, 5)},
        prior_fold_cursors=prior.audit_manifest.fold_cursors,
    )
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    prior_earning = {m.uid: (m.hotkey, float(m.accumulate_score)) for m in prior.miners}
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=tuple(miners), burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW, prior_log_digest=prior.log_digest(),
        prior_earning=prior_earning,
    )
    assert cur.weight_shares.get(5, 0.0) > 0.0  # NONZERO weight — an active idle earner
    assert cur.audit_manifest.earning_for(5) is None  # no committed evidence this epoch

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)

    v5 = next(v for v in report.earning_verdicts if v.uid == 5 and v.item_id != "crown")
    assert v5.verdict is ItemVerdictKind.PASS
    assert report.overall is AuditStatus.CLEAN


def test_injected_nonzero_weight_carry_forward_disputes(tmp_path) -> None:
    """an internal review: the symmetric integrity guard — a NONZERO-weight uid with NO EarningInput
    whose accumulator does NOT match the prior epoch's value is an INJECTED accumulator (nothing
    folds it) ⇒ DISPUTED, never washed to CLEAN by the carry-forward path."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior_per_uid = {uid: (0.9, _fold(0.0, [0.9])) for uid in range(1, 6)}
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid=prior_per_uid, priors={}, prior_log=None,
    )
    items, miners = [], []
    top_prior = {uid: _fold(0.0, [0.9]) for uid in range(1, 6)}
    for uid in range(1, 5):
        b = _bundle(store, source, uid, f"e100i{uid}", 0.9, seq=1)
        items.append(_scored(b, uid, 0.9, seq=1))
        miners.append(make_miner(uid, _fold(top_prior[uid], [0.9])))
    # uid 5 stays nonzero-weight but its accumulator JUMPS with no new evidence and != prior.
    miners.append(make_miner(5, _fold(top_prior[5], [0.9])))
    manifest = build_audit_manifest(
        items, store=store, prior_accumulate={u: top_prior[u] for u in range(1, 5)},
        prior_fold_cursors=prior.audit_manifest.fold_cursors,
    )
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    # Deliberately tell the finalizer the wrong prior (so it accepts) — the AUDITOR, chaining the
    # REAL prior log, is what must catch the injected jump.
    prior_earning = {m.uid: (m.hotkey, float(m.accumulate_score)) for m in prior.miners}
    prior_earning[5] = ("hk5", _fold(top_prior[5], [0.9]))  # lie to the producer-side check
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=tuple(miners), burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW, prior_log_digest=prior.log_digest(),
        prior_earning=prior_earning,
    )
    assert cur.weight_shares.get(5, 0.0) > 0.0 and cur.audit_manifest.earning_for(5) is None

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)

    v5 = next(v for v in report.earning_verdicts if v.uid == 5 and v.item_id != "crown")
    assert v5.verdict is ItemVerdictKind.FAIL and v5.code == EARNING_STATE_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_injected_zero_weight_carry_forward_disputes(tmp_path) -> None:
    """an internal review(a): a zero-weight miner whose positive accumulator does NOT match the
    prior epoch's value AND has no committed evidence this epoch is an INJECTED accumulator (a
    value materialized with nothing folding it) => DISPUTED."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior_per_uid = {uid: (0.9, _fold(0.0, [0.9])) for uid in range(1, 6)}
    prior_per_uid[6] = (0.1, _fold(0.0, [0.1]))
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid=prior_per_uid, priors={}, prior_log=None,
    )
    items, miners = [], []
    top_prior = {uid: _fold(0.0, [0.9]) for uid in range(1, 6)}
    for uid in range(1, 6):
        b = _bundle(store, source, uid, f"e100i{uid}", 0.9, seq=1)
        items.append(_scored(b, uid, 0.9, seq=1))
        miners.append(make_miner(uid, _fold(top_prior[uid], [0.9])))
    # uid 6's accumulator JUMPS to fold([0.5]) with no new evidence and != the prior fold([0.1]).
    miners.append(make_miner(6, _fold(0.0, [0.5])))
    manifest = build_audit_manifest(
        items, store=store, prior_accumulate=top_prior,
        prior_fold_cursors=prior.audit_manifest.fold_cursors,
    )
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=tuple(miners),
        burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW, prior_log_digest=prior.log_digest(),
    )
    assert cur.weight_shares.get(6, 0.0) == 0.0

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)

    v6 = next(v for v in report.earning_verdicts if v.uid == 6 and v.item_id != "crown")
    assert v6.verdict is ItemVerdictKind.FAIL and v6.code == EARNING_STATE_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_reregistered_hotkey_inheriting_carry_in_disputes(tmp_path) -> None:
    """an internal review(b): the carry-in is keyed by (uid, hotkey). A re-registered uid (hotkey
    changed vs the prior epoch) is a FRESH identity — the validator registry resets its
    accumulate_score to 0.0 — so it carries in 0.0. A log that instead claims the PREVIOUS owner's
    nonzero carry-in for the reused uid is a re-attributed inheritance => DISPUTED."""
    from dataclasses import replace

    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid={1: (0.8, _fold(0.0, [0.8]))}, priors={}, prior_log=None,
    )
    carry = _fold(0.0, [0.8])  # the PREVIOUS owner (hk1) ended the prior epoch here
    # E+1: uid 1 is RE-REGISTERED as hkNEW; its committed evidence is under hkNEW, but the log
    # claims the previous owner's carry-in.
    packet = make_packet(
        challenge_id="c1", item_id="e100i1", miner_hotkey="hkNEW", score=0.5, cycle_sequence=1,
        metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": 0.5},
    )
    b = make_fake_bundle(
        store, challenge_id="c1", item_id="e100i1", miner_hotkey="hkNEW", packet=packet,
        dispatch_ordering_key=1,
    )
    persist_bundle(store, b)
    source.add(b)
    item = ScoredItem(
        uid=1, hotkey="hkNEW", challenge_id="c1", item_id="e100i1",
        bundle_digest=b.bundle_digest(), packet_digest=b.score_packet.digest,
        committed_track="compression", score=0.5, cycle_sequence=1,
    )
    reregistered = replace(
        make_miner(1, _fold(carry, [0.5])), hotkey="hkNEW", coldkey="ckNEW", ip="10.0.1.1"
    )
    manifest = build_audit_manifest(
        [item], store=store, prior_accumulate={1: carry},
        prior_fold_cursors=prior.audit_manifest.fold_cursors,
    )
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(reregistered,),
        burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW, prior_log_digest=prior.log_digest(),
    )

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)

    v1 = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id != "crown")
    assert v1.verdict is ItemVerdictKind.FAIL and v1.code == EARNING_STATE_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_reregistered_hotkey_with_zero_carry_in_passes(tmp_path) -> None:
    """an internal review(b): the SAME re-registration is HONEST when the fresh identity carries in
    0.0 (the validator's reset) — a re-registered uid starting from scratch is not a fault."""
    from dataclasses import replace

    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid={1: (0.8, _fold(0.0, [0.8]))}, priors={}, prior_log=None,
    )
    packet = make_packet(
        challenge_id="c1", item_id="e100i1", miner_hotkey="hkNEW", score=0.5, cycle_sequence=1,
        metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": 0.5},
    )
    b = make_fake_bundle(
        store, challenge_id="c1", item_id="e100i1", miner_hotkey="hkNEW", packet=packet,
        dispatch_ordering_key=1,
    )
    persist_bundle(store, b)
    source.add(b)
    item = ScoredItem(
        uid=1, hotkey="hkNEW", challenge_id="c1", item_id="e100i1",
        bundle_digest=b.bundle_digest(), packet_digest=b.score_packet.digest,
        committed_track="compression", score=0.5, cycle_sequence=1,
    )
    fresh = replace(
        make_miner(1, _fold(0.0, [0.5])), hotkey="hkNEW", coldkey="ckNEW", ip="10.0.1.1"
    )
    manifest = build_audit_manifest(
        [item], store=store, prior_accumulate={1: 0.0},
        prior_fold_cursors=prior.audit_manifest.fold_cursors,
    )
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(fresh,),
        burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW, prior_log_digest=prior.log_digest(),
    )

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)

    v1 = next(v for v in report.earning_verdicts if v.uid == 1 and v.item_id != "crown")
    assert v1.verdict is ItemVerdictKind.PASS
    assert report.overall is AuditStatus.CLEAN


#


def test_active_epoch_cannot_reset_prior_miner_to_zero_without_evidence(tmp_path) -> None:
    """an internal review: a STILL-REGISTERED, prior-POSITIVE miner cannot be SILENTLY RESET to
    0.0 (or the exclusion sentinel, or dropped) this epoch with NO evidenced reason.

    The earning re-fold + census only look at CURRENT positive accumulators / current evidence,
    and `_earning_verdicts` only audits uids with positive weight, a current EarningInput, or a
    positive CURRENT accumulator — so a miner reset to 0.0 with no current evidence is in NONE of
    those sets and is silently skipped while another miner's new evidence re-derives the vector
    CLEAN. That erases the reset miner's accrued earnings.

    Prior epoch: uid1 and uid2 both earn positively. Current epoch: uid2 is re-scored (new
    evidence ⇒ weight) while uid1 is reset to 0.0 with NO current evidence, STILL registered under
    the same (uid, hotkey). The auditor chains against the prior log + binds to the close-block
    metagraph ⇒ EARNING_STATE_RESET ⇒ DISPUTED."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid={1: (0.6, _fold(0.0, [0.6])), 2: (0.6, _fold(0.0, [0.6]))},
        priors={}, prior_log=None,
    )
    prior2 = _fold(0.0, [0.6])
    # Current: only uid2 carries committed evidence; uid1 is present (same hk1) but reset to 0.0.
    b2 = _bundle(store, source, 2, "e100i2", 0.6, seq=1)
    manifest = build_audit_manifest(
        [_scored(b2, 2, 0.6, seq=1)], store=store, prior_accumulate={2: prior2},
        prior_fold_cursors=prior.audit_manifest.fold_cursors,
    )
    miners = (make_miner(1, 0.0), make_miner(2, _fold(prior2, [0.6])))
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=miners, burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW, prior_log_digest=prior.log_digest(),
    )
    assert cur.weight_shares.get(1, 0.0) == 0.0  # uid1 zeroed
    assert cur.weight_shares.get(2, 0.0) > 0.0  # uid2 still earns (its new evidence re-derives)
    assert cur.audit_manifest.earning_for(1) is None  # NO evidence justifies the drop

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)

    v = next(v for v in report.earning_verdicts if v.uid == 1 and v.code == EARNING_STATE_RESET)
    assert v.verdict is ItemVerdictKind.FAIL
    assert report.weight_verdict.verdict is ItemVerdictKind.FAIL
    assert report.overall is AuditStatus.DISPUTED


def test_genuine_deregistration_of_prior_positive_miner_is_not_falsely_disputed(tmp_path) -> None:
    """an internal review (false-positive guard): a prior-positive miner that genuinely
    DEREGISTERED — absent from the close-block metagraph — carries in 0.0 as a fresh identity and
    is NOT a reset. uid1 (positive in the prior epoch) is gone this epoch; uid2 re-earns. The
    close-block metagraph (auto-wired from the current miners) does not carry uid1 ⇒ no reset
    verdict ⇒ CLEAN."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid={1: (0.6, _fold(0.0, [0.6])), 2: (0.6, _fold(0.0, [0.6]))},
        priors={}, prior_log=None,
    )
    prior2 = _fold(0.0, [0.6])
    b2 = _bundle(store, source, 2, "e100i2", 0.6, seq=1)  # monotonic: > the prior epoch's key 0
    manifest = build_audit_manifest(
        [_scored(b2, 2, 0.6, seq=1)], store=store, prior_accumulate={2: prior2},
        prior_fold_cursors=prior.audit_manifest.fold_cursors,
    )
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(make_miner(2, _fold(prior2, [0.6])),),
        burn_uid=BURN_UID, audit_manifest=manifest, now=NOW, prior_log_digest=prior.log_digest(),
    )

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)

    assert not [v for v in report.earning_verdicts if v.code == EARNING_STATE_RESET]
    assert report.overall is AuditStatus.CLEAN


def test_evidenced_exclusion_of_prior_positive_miner_is_not_falsely_disputed(tmp_path) -> None:
    """an internal review (false-positive guard): a prior-positive miner EXCLUDED this epoch via a
    COMMITTED exclusion packet (its accumulator latches to the -1 sentinel) carries a CURRENT
    EarningInput, so reset-detection DEFERS to the earning fold (which validates the evidenced
    exclusion) — it is NOT a silent reset. uid1 is excluded on evidence; uid2 re-earns ⇒ CLEAN."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    prior = _epoch(
        store, source, epoch_id=99, close_block=359_640,
        per_uid={1: (0.6, _fold(0.0, [0.6])), 2: (0.6, _fold(0.0, [0.6]))},
        priors={}, prior_log=None,
    )
    prior1, prior2 = _fold(0.0, [0.6]), _fold(0.0, [0.6])
    # monotonic keys: this epoch's cycles (seq=1) exceed the prior epoch's key 0 for each uid
    bx = _bundle(store, source, 1, "e100ix", 0.0, seq=1, excluded=True)  # committed exclusion
    b2 = _bundle(store, source, 2, "e100i2", 0.6, seq=1)
    items = [_scored(bx, 1, 0.0, seq=1, excluded=True), _scored(b2, 2, 0.6, seq=1)]
    manifest = build_audit_manifest(
        items, store=store, prior_accumulate={1: prior1, 2: prior2},
        prior_fold_cursors=prior.audit_manifest.fold_cursors,
    )
    excluded_acc = _fold(prior1, [-1.0])  # the -1 sentinel latches
    miners = (make_miner(1, excluded_acc), make_miner(2, _fold(prior2, [0.6])))
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=miners, burn_uid=BURN_UID,
        audit_manifest=manifest, now=NOW, prior_log_digest=prior.log_digest(),
    )

    report = _auditor(source).audit_epoch(cur, store, NO_SAMPLE, None, NOW, prior_log=prior)

    assert not [v for v in report.earning_verdicts if v.code == EARNING_STATE_RESET]
    v1 = next(v for v in report.earning_verdicts if v.uid == 1 and v.source == "earning")
    assert v1.verdict is ItemVerdictKind.PASS  # the evidenced exclusion verifies
    assert report.overall is AuditStatus.CLEAN
