"""Close-block metagraph binding of the SNAPSHOT-derivable weight inputs.

The auditor re-derives the weight vector via ``build_weight_vector(config, log.miners, ...)``
and today only independently verifies each miner's ``accumulate_score`` (the earning fold).
This suite covers the OTHER SNAPSHOT-derivable MinerSnapshot fields the authority previously
self-attested unchecked, by binding them to the close-block METAGRAPH the auditor reads ITSELF:

  - uid -> hotkey/coldkey/ip   (identity / relabel)   -> IDENTITY_MISMATCH ⇒ DISPUTED
  - the IP/coldkey DEDUP outcome (the ``excluded`` flag) -> METAGRAPH_DEDUP_MISMATCH ⇒ DISPUTED
  - the declared scoring ``track``                    -> METAGRAPH_TRACK_MISMATCH ⇒ DISPUTED

Fail-closed: an unreadable/unavailable metagraph ⇒ INCONCLUSIVE (HOLD), never a PASS on the
authority's word. The WINDOWED inputs (alpha_stake_delta_window / emission_window /
has_full_retention_window) are an internal review and OUT OF SCOPE here.

These tests wire an EXPLICIT metagraph (the TRUE identities) into a plain ``Auditor`` while the
authority log lies — exercising the real read seam, not the auto-deriving test double.
"""

from __future__ import annotations

from dataclasses import replace

from vidaio.audit.recompute import IDENTITY_MISMATCH
from vidaio.audit.store import LocalFsStore
from vidaio.auditor import (
    METAGRAPH_DEDUP_MISMATCH,
    METAGRAPH_TRACK_MISMATCH,
    SNAPSHOT_UNVERIFIED,
    UNKNOWN_TRACK,
    Auditor,
    AuditorConfig,
    AuditStatus,
    InMemoryBundleSource,
    ItemVerdictKind,
    SamplePolicy,
    persist_bundle,
)
from vidaio.authority import EpochFinalizer, build_audit_manifest
from vidaio.chain.adapter import ChainStateUnavailable
from vidaio.epoch.log import EpochLog, MinerCensusEntry, weight_vector_digest
from vidaio.tokenomics import TokenomicsConfig, quantize_u16
from vidaio.tokenomics.weights import build_weight_vector

from tests.auditor.fakes import (
    BURN_UID,
    CLOSE_BLOCK,
    NOW,
    SCORER,
    folded_miner,
    honest_log,
    make_fake_bundle,
    make_packet,
    metagraph_chain,
    scored_item,
)

CFG = TokenomicsConfig()
NO_SAMPLE = SamplePolicy(sample_rate=0.0, min_samples=0)  # snapshot/earning only, no media


def _burn_log(miners, manifest, *, epoch_id=100, close_block=CLOSE_BLOCK, prior_log_digest=None):
    """A burn-only EpochLog built via ``model_construct`` — BYPASSES ``_validate``.

    an internal review refuses an OUT-OF-PROTOCOL ``MinerSnapshot.track`` (e.g. "unknown") at the
    construction / from_json boundary, so a test that needs to seat such a track (to exercise the
    auditor's DEFENSE-IN-DEPTH track binding on bytes that dodged the finalizer) constructs the
    log directly. The burn vector is the real ``build_weight_vector`` output for these miners
    (an out-of-protocol track takes zero share ⇒ ``{burn_uid: 1.0}``)."""
    shares = build_weight_vector(CFG, list(miners), burn_uid=BURN_UID)
    u16 = quantize_u16(shares)
    return EpochLog.model_construct(
        schema_version=EpochLog.model_fields["schema_version"].default,
        epoch_id=epoch_id, close_block=close_block, scorer_version=SCORER, created_at=NOW,
        prior_log_digest=prior_log_digest, burn_uid=BURN_UID, miners=tuple(miners),
        weight_shares=shares, weight_u16=u16,
        weight_vector_digest=weight_vector_digest(u16), audit_manifest=manifest,
    )


def _auditor(source, chain):
    """A plain Auditor with an EXPLICIT close-block metagraph read seam.

    burn_uid=BURN_UID: the CANONICAL burn recipient the auditor binds a burn log's burn uid
    to — the SAME value these fixtures finalize burn epochs with, so the
    burn-only tests exercise ONLY their intended snapshot fault (not a spurious burn mismatch).
    """
    return Auditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID),
        source, chain=chain,
    )


def _honest(store, source, miners):
    """An honest genesis epoch: one committed cycle per miner, weights follow."""
    items = []
    for m in miners:
        b = make_fake_bundle(
            store, challenge_id="c1", item_id=f"i{m.uid}", miner_hotkey=m.hotkey,
        )
        persist_bundle(store, b)
        source.add(b)
        items.append(scored_item(b, m.uid))
    manifest = build_audit_manifest(items, store=store)
    return honest_log(list(miners), manifest)


def _snapshot(report, uid):
    return next(
        v for v in report.earning_verdicts if v.source == "snapshot" and v.uid == uid
    )


# --- 6. honest epoch with a matching metagraph ⇒ CLEAN --------------------------------


def test_honest_epoch_with_matching_metagraph_is_clean(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    miners = [folded_miner(1), folded_miner(2)]
    log = _honest(store, source, miners)

    report = _auditor(source, metagraph_chain(miners)).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    assert all(
        v.verdict is ItemVerdictKind.PASS
        for v in report.earning_verdicts
        if v.source == "snapshot"
    )
    assert report.weight_verdict.verdict is ItemVerdictKind.PASS
    assert report.overall is AuditStatus.CLEAN


def test_zero_weight_miner_with_tampered_identity_is_disputed(tmp_path) -> None:
    """an internal review: a below-cutoff ZERO-weight census miner whose LOG identity is tampered
    must STILL be bound to the close-block metagraph. Snapshot identity binding used to cover
    only nonzero-weight uids, so a tampered-identity zero-weight record (which earns nothing this
    epoch but sits in the census and could later become load-bearing / re-attribute a carry-in)
    audited CLEAN. It must now be IDENTITY_MISMATCH ⇒ DISPUTED."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # 6 miners on one track with DISTINCT descending scores ⇒ uid 6 is below the top-5 cutoff
    # (zero weight). uid 6's LOG identity is tampered; the TRUE metagraph binds its real identity.
    scores = {1: 0.90, 2: 0.80, 3: 0.70, 4: 0.60, 5: 0.50, 6: 0.40}
    TAMPERED = {"hotkey": "hkTAMPERED", "coldkey": "ckTAMPERED", "ip": "203.0.113.66"}
    items, log_miners, true_miners = [], [], []
    for uid, s in scores.items():
        true = folded_miner(uid, score=s)
        # The log (and uid 6's committed evidence) speak the tampered identity for uid 6.
        logged = replace(true, **TAMPERED) if uid == 6 else true
        packet = make_packet(
            challenge_id="c1", item_id=f"i{uid}", miner_hotkey=logged.hotkey, score=s,
            cycle_sequence=0, metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": s},
        )
        b = make_fake_bundle(
            store, challenge_id="c1", item_id=f"i{uid}", miner_hotkey=logged.hotkey,
            packet=packet, dispatch_ordering_key=0,
        )
        persist_bundle(store, b)
        source.add(b)
        items.append(replace(scored_item(b, uid, score=s, seq=0), hotkey=logged.hotkey))
        log_miners.append(logged)
        true_miners.append(true)
    manifest = build_audit_manifest(items, store=store)
    log = honest_log(log_miners, manifest)
    assert log.weight_shares.get(6, 0.0) == 0.0  # uid 6 is a zero-weight rank loser

    report = _auditor(source, metagraph_chain(true_miners)).audit_epoch(
        log, store, NO_SAMPLE, None, NOW
    )

    v = _snapshot(report, 6)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == IDENTITY_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def _burn_only_epoch(store, source, *, spoof_uid=None, exclude_all=True):
    """A BURN-ONLY epoch ({burn_uid: 1.0}): every census miner is zero-weight (excluded).

    Returns (log, true_miners). When ``spoof_uid`` is set, that miner's LOG identity is tampered
    (its committed evidence speaks the tampered identity) while the metagraph keeps its true one.
    """
    scores = {1: 0.90, 2: 0.80, 3: 0.70}
    TAMPERED = {"hotkey": "hkTAMPERED", "coldkey": "ckTAMPERED", "ip": "203.0.113.9"}
    items, log_miners, true_miners = [], [], []
    for uid, s in scores.items():
        true = folded_miner(uid, score=s)
        logged = replace(true, **TAMPERED) if uid == spoof_uid else true
        if exclude_all:
            logged = replace(logged, excluded=True)
        packet = make_packet(
            challenge_id="c1", item_id=f"i{uid}", miner_hotkey=logged.hotkey, score=s,
            cycle_sequence=0, excluded=exclude_all,
            metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": s},
        )
        b = make_fake_bundle(
            store, challenge_id="c1", item_id=f"i{uid}", miner_hotkey=logged.hotkey,
            packet=packet, dispatch_ordering_key=0,
        )
        persist_bundle(store, b)
        source.add(b)
        items.append(replace(scored_item(b, uid, score=s, seq=0), hotkey=logged.hotkey,
                             excluded_cycle=exclude_all))
        log_miners.append(logged)
        true_miners.append(true)
    manifest = build_audit_manifest(items, store=store)
    return honest_log(log_miners, manifest), true_miners


def test_burn_only_zero_weight_tampered_identity_is_disputed(tmp_path) -> None:
    """an internal review: a BURN-ONLY vector must NOT bypass the zero-weight census binding — a
    tampered identity on an all-excluded (zero-weight) census miner is still IDENTITY_MISMATCH."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, true_miners = _burn_only_epoch(store, source, spoof_uid=1)
    assert all(w == 0.0 for uid, w in log.weight_shares.items() if uid != log.burn_uid)

    report = _auditor(source, metagraph_chain(true_miners)).audit_epoch(
        log, store, NO_SAMPLE, None, NOW
    )

    v = _snapshot(report, 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == IDENTITY_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_burn_only_wrongly_excluded_distinct_miner_is_disputed(tmp_path) -> None:
    """an internal review: an authority that over-excludes DISTINCT miners to substitute a
    100%-burn vector is caught — the re-derived dedup (distinct ⇒ not excluded) disagrees with
    the log's excluded=True ⇒ METAGRAPH_DEDUP_MISMATCH ⇒ DISPUTED, even for a burn-only epoch."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, true_miners = _burn_only_epoch(store, source, spoof_uid=None)
    assert all(w == 0.0 for uid, w in log.weight_shares.items() if uid != log.burn_uid)

    report = _auditor(source, metagraph_chain(true_miners)).audit_epoch(
        log, store, NO_SAMPLE, None, NOW
    )

    dedup = [v for v in report.earning_verdicts if v.source == "snapshot"
             and v.code == METAGRAPH_DEDUP_MISMATCH]
    assert dedup, "a wrongly-excluded distinct miner must raise METAGRAPH_DEDUP_MISMATCH"
    assert report.overall is AuditStatus.DISPUTED


def test_burn_only_evidenced_uid_absent_from_metagraph_is_disputed(tmp_path) -> None:
    """an internal review: an authority RELABELS evidenced, excluded census miners to uids that
    are ABSENT from the close-block metagraph, publishes a 100%-burn vector, and — before this
    fix — the zero-weight census pass SILENTLY SKIPPED any ``m.uid not in metagraph``, washing to
    CLEAN. A census miner that is NOT the burn uid and carries COMMITTED evidence but whose uid is
    absent from the metagraph is an unbindable/relabelled identity ⇒ IDENTITY_MISMATCH ⇒ DISPUTED
    (a zero-weight NO-evidence absent miner stays a benign skip — covered by the honest cases)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # log uids 1,2,3: evidenced, all excluded ⇒ a 100%-burn vector (the substituted burn).
    log, _ = _burn_only_epoch(store, source, spoof_uid=None)
    assert all(w == 0.0 for uid, w in log.weight_shares.items() if uid != log.burn_uid)
    # Every evidenced census uid is RELABELLED to a uid ABSENT from the close-block metagraph
    # (the metagraph binds a DISJOINT identity set), so none can be bound — the relabel-to-
    # absent-uid burn hole. Before round-17 #2 these were SILENTLY SKIPPED and washed to CLEAN.
    disjoint_metagraph = metagraph_chain([folded_miner(90)])

    report = _auditor(source, disjoint_metagraph).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    absent = [
        v for v in report.earning_verdicts
        if v.source == "snapshot" and v.code == IDENTITY_MISMATCH
        and v.verdict is ItemVerdictKind.FAIL
    ]
    assert sorted(v.uid for v in absent) == [1, 2, 3]  # each evidenced absent uid is caught
    assert report.weight_verdict.verdict is ItemVerdictKind.FAIL
    assert report.weight_verdict.code == IDENTITY_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_burn_only_tampered_nonpaying_track_is_disputed(tmp_path) -> None:
    """an internal review: evidence-backed positive-score miners are assigned a bogus/NON-PAYING
    track — ``build_weight_vector`` gives a track absent from ``track_weights`` ZERO share, so the
    whole vector collapses to a {burn_uid:1.0} burn — while their committed challenge evidence
    commits a valid (paying) track. Before this fix the zero-weight census pass bound identity +
    dedup but NEVER the track, and the burn-only reconstruct preserved the tampered track, so the
    substituted burn audited CLEAN. The zero-weight pass must BIND the track to committed evidence ⇒
    METAGRAPH_TRACK_MISMATCH ⇒ DISPUTED (propagated to the weight verdict)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    scores = {1: 0.90, 2: 0.80, 3: 0.70}
    items, miners = [], []
    for uid, s in scores.items():
        # Committed evidence commits the real PAYING track "compression".
        packet = make_packet(
            challenge_id="c1", item_id=f"i{uid}", miner_hotkey=f"hk{uid}", score=s,
            cycle_sequence=0, metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": s},
        )
        b = make_fake_bundle(
            store, challenge_id="c1", item_id=f"i{uid}", miner_hotkey=f"hk{uid}",
            packet=packet, committed_track="compression", dispatch_ordering_key=0,
        )
        persist_bundle(store, b)
        source.add(b)
        items.append(scored_item(b, uid, score=s, seq=0, committed_track="compression"))
        # The LOG snapshot declares a NON-PAYING track (absent from track_weights) ⇒ zero
        # inference share ⇒ the vector collapses to the burn.
        miners.append(folded_miner(uid, score=s, track="unknown"))
    manifest = build_audit_manifest(items, store=store)
    # `_validate` now refuses the out-of-protocol "unknown" track (round-19 #2), so build the
    # burn log via model_construct — a tampered log that dodged the finalizer, handed to the
    # auditor to exercise the zero-weight track binding (round-17 #3) defense in depth.
    log = _burn_log(miners, manifest)
    assert log.burn_uid == BURN_UID  # the tampered-track collapse produced a burn-only vector
    assert all(w == 0.0 for uid, w in log.weight_shares.items() if uid != log.burn_uid)

    report = _auditor(source, metagraph_chain(miners)).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    track = [
        v for v in report.earning_verdicts
        if v.source == "snapshot" and v.code == METAGRAPH_TRACK_MISMATCH
    ]
    assert track, "a tampered non-paying track on evidenced miners must raise METAGRAPH_TRACK_MISMATCH"
    assert report.weight_verdict.verdict is ItemVerdictKind.FAIL
    assert report.weight_verdict.code == METAGRAPH_TRACK_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_burn_only_implicit_carry_tampered_track_is_disputed(tmp_path) -> None:
    """an internal review: an IMPLICIT carry-forward miner (positive accumulator, NO current
    evidence) had NO track binding. The zero-weight snapshot pass only binds EVIDENCED miners
    (round-17 #3), and the carry-forward earning path checked the prior digest / (uid, hotkey) /
    accumulator but NOT the track; the burn-only reconstruct preserves the raw log track. So an
    authority could carry a CLEAN predecessor's positive accumulator forward under the SAME
    (uid, hotkey), switch its paying track to a NON-PAYING `unknown`, omit current evidence, and
    publish the canonical burn vector — the snapshot pass skips the unevidenced track, carry-forward
    passes, reconstruct keeps `unknown`, and the vector collapses to burn ⇒ CLEAN. The carry-forward
    track must CHAIN to the prior epoch's carried track for this (uid, hotkey) ⇒
    METAGRAPH_TRACK_MISMATCH ⇒ DISPUTED (propagated to the weight verdict)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # Prior epoch (a CLEAN predecessor): uid 1 scored on the PAYING "compression" track.
    b = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    persist_bundle(store, b)
    source.add(b)
    prior_manifest = build_audit_manifest([scored_item(b, 1)], store=store)
    prior = honest_log([folded_miner(1)], prior_manifest, epoch_id=99, close_block=359_640)
    carried = next(m for m in prior.miners if m.uid == 1).accumulate_score

    # Current epoch: uid 1 carries the SAME accumulator forward under the SAME hotkey with NO new
    # evidence, but its track is switched to the NON-PAYING "unknown" ⇒ zero share ⇒ burn vector.
    cur_miner = replace(folded_miner(1, track="unknown"), accumulate_score=carried)
    # `_validate` now refuses the out-of-protocol "unknown" track (round-19 #2), so build the
    # burn log via model_construct — a tampered log that dodged the finalizer, handed to the
    # auditor to exercise the carry-forward track binding (round-18 #1) defense in depth.
    cur = _burn_log(
        [cur_miner], build_audit_manifest([], store=store),
        epoch_id=100, close_block=360_000, prior_log_digest=prior.log_digest(),
    )
    assert cur.burn_uid == BURN_UID  # the switched track collapsed the vector to a burn
    assert cur.audit_manifest.earning_for(1) is None  # a pure carry-forward, no current evidence

    report = _auditor(source, metagraph_chain([cur_miner])).audit_epoch(
        cur, store, NO_SAMPLE, None, NOW, prior_log=prior
    )

    track = [
        v for v in report.earning_verdicts
        if v.code == METAGRAPH_TRACK_MISMATCH and v.uid == 1
    ]
    assert track, "a tampered carry-forward track must raise METAGRAPH_TRACK_MISMATCH"
    assert report.weight_verdict.verdict is ItemVerdictKind.FAIL
    assert report.overall is AuditStatus.DISPUTED


#


def _unknown_bundle(store, source, uid, score, track):
    """A committed bundle whose committed_track + DAG_REVEAL track + packet track are ``track``."""
    packet = make_packet(
        challenge_id="c1", item_id=f"i{uid}", miner_hotkey=f"hk{uid}", score=score,
        cycle_sequence=0, track=track,
        metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": score},
    )
    b = make_fake_bundle(
        store, challenge_id="c1", item_id=f"i{uid}", miner_hotkey=f"hk{uid}",
        packet=packet, committed_track=track, dispatch_ordering_key=0,
    )
    persist_bundle(store, b)
    source.add(b)
    return b


def test_burn_only_genesis_unknown_committed_track_is_disputed(tmp_path) -> None:
    """an internal review: positive evidence committed CONSISTENTLY under an OUT-OF-PROTOCOL track.

    ``AuditFileRef`` requires a non-null committed track but never that it be a MEMBER of the
    protocol set; commitment parsing accepts any non-empty string; and tokenomics silently drops a
    miner whose track is absent from ``track_weights`` — so committing positive evidence under
    "unknown" (committed_track + DAG_REVEAL track + log track all "unknown", self-consistent)
    collapses the vector to ``{burn_uid: 1.0}`` and audited CLEAN: the existing track binding only
    compares self-consistent DECLARATIONS to each other (both "unknown" ⇒ they AGREE). The auditor
    now validates every committed/log track against the protocol set ⇒ UNKNOWN_TRACK ⇒ DISPUTED.
    ``_validate`` refuses such a log, so this exercises the auditor's defense in depth on bytes that
    dodged the finalizer (built via model_construct)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    scores = {1: 0.9, 2: 0.8}
    items, miners = [], []
    for uid, s in scores.items():
        b = _unknown_bundle(store, source, uid, s, "unknown")
        items.append(scored_item(b, uid, score=s, committed_track="unknown"))
        miners.append(folded_miner(uid, score=s, track="unknown"))
    manifest = build_audit_manifest(items, store=store)
    log = _burn_log(miners, manifest)
    assert log.burn_uid == BURN_UID  # out-of-protocol tracks took zero share ⇒ burn vector

    report = _auditor(source, metagraph_chain(miners)).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    unknown = [v for v in report.earning_verdicts if v.code == UNKNOWN_TRACK]
    assert unknown, "an out-of-protocol committed/log track must raise UNKNOWN_TRACK"
    assert report.weight_verdict.verdict is ItemVerdictKind.FAIL
    assert report.overall is AuditStatus.DISPUTED


def test_in_protocol_track_genesis_stays_clean(tmp_path) -> None:
    """an internal review (false-positive guard): the SAME shape with a VALID protocol track
    ("compression") is a normal, honest epoch — no UNKNOWN_TRACK verdict, CLEAN."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    scores = {1: 0.9, 2: 0.8}
    items, miners = [], []
    for uid, s in scores.items():
        b = _unknown_bundle(store, source, uid, s, "compression")
        items.append(scored_item(b, uid, score=s, committed_track="compression"))
        miners.append(folded_miner(uid, score=s, track="compression"))
    manifest = build_audit_manifest(items, store=store)
    log = honest_log(miners, manifest)  # a valid track ⇒ nonzero weights ⇒ normal epoch

    report = _auditor(source, metagraph_chain(miners)).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    assert not [v for v in report.earning_verdicts if v.code == UNKNOWN_TRACK]
    assert report.overall is AuditStatus.CLEAN


#


def test_burn_only_implicit_carry_absent_from_metagraph_and_excluded_is_disputed(tmp_path) -> None:
    """an internal review: a POSITIVE implicit carry-forward miner (positive accumulator, NO current
    evidence) with a correct prior value/hotkey/track but a TAMPERED ``excluded=True`` and ABSENT
    from the close-block metagraph used to audit CLEAN: the zero-weight snapshot pass only bound
    EVIDENCED miners, so this unevidenced carry was a benign skip, and the burn-only reconstruct
    preserved the tampered exclusion ⇒ substituted burn. A positive carry-forward miner is BINDABLE
    evidence — absent from the metagraph its identity/dedup (`excluded`) cannot be bound ⇒ FAIL
    IDENTITY_MISMATCH ⇒ DISPUTED (never a wash on the authority's self-attested exclusion)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # Prior (CLEAN predecessor): uid1 scored on the PAYING compression track.
    b = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    persist_bundle(store, b)
    source.add(b)
    prior = honest_log([folded_miner(1)], build_audit_manifest([scored_item(b, 1)], store=store),
                       epoch_id=99, close_block=359_640)
    carried = next(m for m in prior.miners if m.uid == 1).accumulate_score

    # Current: uid1 carries the SAME accumulator forward (same hk1, VALID compression track) with
    # NO current evidence, but spoofs excluded=True ⇒ zero share ⇒ burn vector — AND is ABSENT
    # from the close-block metagraph.
    cur_miner = replace(folded_miner(1), accumulate_score=carried, excluded=True)
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    cur = fin.build_log(
        epoch_id=100, close_block=360_000, snapshots=(cur_miner,), burn_uid=BURN_UID,
        audit_manifest=build_audit_manifest([], store=store), now=NOW,
        prior_log_digest=prior.log_digest(),
    )
    assert cur.burn_uid == BURN_UID  # the tampered exclusion collapsed the vector to a burn
    assert cur.audit_manifest.earning_for(1) is None  # a pure carry-forward, no current evidence

    # The close-block metagraph does NOT carry uid1 (absent — but the LOG seats it).
    report = _auditor(source, metagraph_chain([])).audit_epoch(
        cur, store, NO_SAMPLE, None, NOW, prior_log=prior
    )

    v = _snapshot(report, 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == IDENTITY_MISMATCH
    assert report.weight_verdict.verdict is ItemVerdictKind.FAIL
    assert report.overall is AuditStatus.DISPUTED


# --- 1. relabelled uid -> hotkey ⇒ DISPUTED -------------------------------------------


def test_relabelled_uid_hotkey_is_disputed(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    miners = [folded_miner(1), folded_miner(2)]
    log = _honest(store, source, miners)

    # The TRUE metagraph binds uid 1 to a DIFFERENT hotkey than the log claims.
    tampered = [replace(miners[0], hotkey="hkEVIL"), miners[1]]
    report = _auditor(source, metagraph_chain(tampered)).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    v = _snapshot(report, 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == IDENTITY_MISMATCH
    assert report.weight_verdict.verdict is ItemVerdictKind.FAIL
    assert report.weight_verdict.code == IDENTITY_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_uid_absent_from_metagraph_is_disputed(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    miners = [folded_miner(1), folded_miner(2)]
    log = _honest(store, source, miners)

    # The metagraph does not carry uid 1 at all — its identity cannot be bound.
    report = _auditor(source, metagraph_chain([miners[1]])).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    v = _snapshot(report, 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == IDENTITY_MISMATCH
    assert "ABSENT" in v.detail
    assert report.overall is AuditStatus.DISPUTED


# --- 2. tampered dedup: a real collision the authority did NOT exclude ⇒ DISPUTED --------


def test_unexcluded_ip_collision_is_disputed(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # uid 2 SHARES uid 1's IP (a real dedup collision); NEITHER is marked excluded.
    m1 = folded_miner(1)
    m2 = replace(folded_miner(2), ip=m1.ip)  # SAME ip as uid 1
    log = _honest(store, source, [m1, m2])
    # build_weight_vector dedups uid 2 (higher uid) dynamically -> uid 2 earns ZERO.
    assert log.weight_shares.get(2, 0.0) == 0.0 and log.weight_shares.get(1, 0.0) > 0.0

    report = _auditor(source, metagraph_chain([m1, m2])).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    # the zero-weight dup's `excluded` should have been True; the log left it False.
    v = _snapshot(report, 2)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == METAGRAPH_DEDUP_MISMATCH
    assert report.weight_verdict.code == METAGRAPH_DEDUP_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_unexcluded_coldkey_collision_is_disputed(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    m1 = folded_miner(1)
    m2 = replace(folded_miner(2), coldkey=m1.coldkey)  # SAME coldkey as uid 1
    log = _honest(store, source, [m1, m2])
    assert log.weight_shares.get(2, 0.0) == 0.0

    report = _auditor(source, metagraph_chain([m1, m2])).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    v = _snapshot(report, 2)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == METAGRAPH_DEDUP_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


# --- 3. wrongly-excluded distinct miner ⇒ DISPUTED ------------------------------------


def test_wrongly_excluded_distinct_miner_is_disputed(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # uid 1 is a DISTINCT miner (no IP/coldkey collision) but the log excludes it.
    m1 = replace(folded_miner(1), excluded=True)
    m2 = folded_miner(2)
    log = _honest(store, source, [m1, m2])
    # the excluded flag zeroed uid 1's weight (uid 2 still earns).
    assert log.weight_shares.get(1, 0.0) == 0.0 and log.weight_shares.get(2, 0.0) > 0.0

    report = _auditor(source, metagraph_chain([m1, m2])).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    v = _snapshot(report, 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == METAGRAPH_DEDUP_MISMATCH
    assert "wrongly-excluded" in v.detail
    assert report.overall is AuditStatus.DISPUTED


# --- 4. track mismatch ⇒ DISPUTED -----------------------------------------------------


def test_track_mismatch_is_disputed(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # The committed earning evidence is a COMPRESSION challenge, but the miner DECLARES
    # the upscaling track (a different pool). Identity + dedup are honest.
    honest = folded_miner(1, track="compression")
    b = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    persist_bundle(store, b)
    source.add(b)
    manifest = build_audit_manifest([scored_item(b, 1)], store=store)  # committed_track=compression

    mis = replace(honest, track="upscaling")  # declared track != committed track
    # This track-substitution case intentionally has no earning competition input.
    shares = build_weight_vector(CFG, (mis,), burn_uid=BURN_UID)
    u16 = quantize_u16(shares)
    log = EpochLog(
        schema_version=EpochLog.model_fields["schema_version"].default,
        epoch_id=100, close_block=360_000, scorer_version=honest_log([honest], manifest).scorer_version,
        created_at=NOW, burn_uid=BURN_UID, miners=(mis,),
        miner_census=(MinerCensusEntry.from_miner(mis),),
        weight_shares=shares, weight_u16=u16,
        weight_vector_digest=weight_vector_digest(u16), audit_manifest=manifest,
    )

    report = _auditor(source, metagraph_chain([mis])).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    v = _snapshot(report, 1)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == METAGRAPH_TRACK_MISMATCH
    assert report.weight_verdict.code == METAGRAPH_TRACK_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


# --- 5. metagraph unavailable ⇒ INCONCLUSIVE (fail closed) ----------------------------


class _UnavailableChain:
    """A chain whose metagraph read RAISES — the auditor must fail closed, not PASS."""

    def neurons(self):
        raise ChainStateUnavailable("no chain snapshot yet")


def test_metagraph_unavailable_is_inconclusive(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    miners = [folded_miner(1), folded_miner(2)]
    log = _honest(store, source, miners)

    report = _auditor(source, _UnavailableChain()).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    snaps = [v for v in report.earning_verdicts if v.source == "snapshot"]
    assert snaps and all(
        v.verdict is ItemVerdictKind.SKIP and v.code == SNAPSHOT_UNVERIFIED for v in snaps
    )
    # never a PASS on the authority's word; the weight verdict HOLDs, not PASSes.
    assert report.weight_verdict.verdict is ItemVerdictKind.SKIP
    assert report.weight_verdict.code == SNAPSHOT_UNVERIFIED
    assert not any(v.verdict is ItemVerdictKind.FAIL for v in report.earning_verdicts)
    assert report.overall is AuditStatus.INCONCLUSIVE


def test_no_chain_wired_is_inconclusive(tmp_path) -> None:
    """No metagraph read seam wired at all is also UNVERIFIABLE (fail closed)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    miners = [folded_miner(1)]
    log = _honest(store, source, miners)

    report = _auditor(source, None).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    assert report.weight_verdict.verdict is ItemVerdictKind.SKIP
    assert report.overall is AuditStatus.INCONCLUSIVE
