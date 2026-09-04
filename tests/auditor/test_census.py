"""The MINER-CENSUS vs committed-evidence cross-check.

The snapshot / earning re-derivations are all scoped to the POSITIVE-weight set, so a log whose
positive set is empty or only ``burn_uid`` (``miners=[], {burn_uid:1.0}``) BYPASSES every one of
them — letting a misreporting authority STORE earning evidence for real miners and then OMIT (or
zero-out) every one of them from the census and still receive CLEAN (a censored empty-burn epoch
indistinguishable from a genuinely-empty one).

The auditor now cross-checks the census against the log's OWN committed manifest evidence — a
check provable from the log's bytes (no metagraph): any uid the manifest carries committed
EARNING evidence for, but which is OMITTED from ``log.miners`` (or zeroed without an evidenced
exclusion), is a censored miner ⇒ DISPUTED. A GENUINELY empty epoch (no committed evidence at
all → burn) stays CLEAN.

Schema-v13 competition-specific census completeness is covered by the earning-evidence
tests; this module isolates inference and empty/burn census cases.
"""

from __future__ import annotations

from dataclasses import replace

from vidaio.audit.store import LocalFsStore
from vidaio.auditor import (
    Auditor,
    AuditorConfig,
    AuditStatus,
    BURN_UID_MISMATCH,
    BURN_UID_UNVERIFIED,
    CENSUS_MISMATCH,
    InMemoryBundleSource,
    ItemVerdictKind,
    SamplePolicy,
    persist_bundle,
)
from vidaio.authority import EpochFinalizer, build_audit_manifest
from vidaio.epoch.log import AuditManifest, EpochLog, MinerCensusEntry, weight_vector_digest
from vidaio.tokenomics import TokenomicsConfig, quantize_u16

from tests.auditor.fakes import (
    BURN_UID,
    CLOSE_BLOCK,
    NOW,
    SCORER,
    MetagraphAuditor,
    fold,
    make_fake_bundle,
    make_miner,
    make_packet,
    metagraph_chain,
    rebuild_log,
    scored_item,
)

CFG = TokenomicsConfig()
NO_SAMPLE = SamplePolicy(sample_rate=0.0, min_samples=0)  # earning/census-only, no media


def _auditor(source) -> MetagraphAuditor:
    # burn_uid=BURN_UID: the canonical burn recipient the auditor binds the log's burn uid to
    # — the SAME value these fixtures finalize burn epochs with.
    return MetagraphAuditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID), source
    )


def _honest(store, source, scores: dict[int, float], *, epoch_id: int = 100):
    """An honest genesis epoch: each uid scored once this epoch from a zero carry-in."""
    items, miners = [], []
    for uid, sc in scores.items():
        packet = make_packet(
            challenge_id="c1", item_id=f"i{uid}", miner_hotkey=f"hk{uid}", score=sc,
            cycle_sequence=0, metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": sc},
        )
        b = make_fake_bundle(
            store, challenge_id="c1", item_id=f"i{uid}", miner_hotkey=f"hk{uid}",
            packet=packet, dispatch_ordering_key=0,
        )
        persist_bundle(store, b)
        source.add(b)
        items.append(scored_item(b, uid, score=sc, seq=0))
        miners.append(make_miner(uid, fold(0.0, [sc])))
    manifest = build_audit_manifest(items, store=store)
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    log = fin.build_log(
        epoch_id=epoch_id, close_block=CLOSE_BLOCK, snapshots=tuple(miners),
        burn_uid=BURN_UID, audit_manifest=manifest, now=NOW,
    )
    return log, manifest, miners


def _census_fails(report):
    return [
        v for v in report.earning_verdicts
        if v.code == CENSUS_MISMATCH and v.verdict is ItemVerdictKind.FAIL
    ]


# --- the crux: committed evidence for a miner OMITTED from the census is DISPUTED -----


def test_manifest_evidence_for_omitted_uid_disputes(tmp_path) -> None:
    """Manifest carries committed EARNING evidence for uid 7, but log.miners OMITS uid 7 (and
    it gets no weight) — the authority stored the evidence then censored the miner ⇒ DISPUTED."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, manifest, miners = _honest(store, source, {1: 0.8, 7: 0.6})

    # Drop uid 7 from the census + give all weight to uid 1, but KEEP the committed evidence
    # (earning_inputs[7] / per_uid[7]) in the manifest — the censorship the check must catch.
    shares = {1: 1.0}
    u16 = quantize_u16(shares)
    censored = rebuild_log(
        log,
        miners=tuple(m for m in miners if m.uid != 7),
        burn_uid=None,
        weight_shares=shares,
        weight_u16=u16,
        weight_vector_digest=weight_vector_digest(u16),
    )
    assert 7 in censored.audit_manifest.earning_inputs  # evidence retained
    assert all(m.uid != 7 for m in censored.miners)  # but censored from the census

    report = _auditor(source).audit_epoch(censored, store, NO_SAMPLE, None, NOW)

    fails = _census_fails(report)
    assert [v.uid for v in fails] == [7]
    assert report.overall is AuditStatus.DISPUTED


def test_evidenced_omission_still_disputes_during_metagraph_outage(tmp_path) -> None:
    """Unavailable chain state cannot hide a fault already proven by the log itself."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, _manifest, miners = _honest(store, source, {1: 0.8, 7: 0.6})
    shares = {1: 1.0}
    u16 = quantize_u16(shares)
    censored = rebuild_log(
        log,
        miners=tuple(m for m in miners if m.uid != 7),
        weight_shares=shares,
        weight_u16=u16,
        weight_vector_digest=weight_vector_digest(u16),
    )
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID),
        source,
        chain=None,
    )

    report = auditor.audit_epoch(censored, store, NO_SAMPLE, None, NOW)

    omitted = next(
        v for v in report.earning_verdicts if v.item_id == "census-economic:7"
    )
    unavailable = next(
        v for v in report.earning_verdicts if v.item_id == "census:metagraph"
    )
    assert omitted.verdict is ItemVerdictKind.FAIL and omitted.code == CENSUS_MISMATCH
    assert unavailable.verdict is ItemVerdictKind.SKIP
    assert report.overall is AuditStatus.DISPUTED


def test_empty_burn_log_with_nonempty_manifest_disputes(tmp_path) -> None:
    """An EMPTY/burn log (miners=[], {burn_uid:1.0}) whose manifest nonetheless carries
    committed earning evidence ⇒ DISPUTED. This is the exact case: publish an
    apparently-honest empty-burn epoch while the manifest proves real miners were scored."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, manifest, _ = _honest(store, source, {7: 0.6})

    shares = {BURN_UID: 1.0}
    u16 = quantize_u16(shares)
    burn = rebuild_log(
        log,
        miners=(),  # every real miner censored
        burn_uid=BURN_UID,
        weight_shares=shares,
        weight_u16=u16,
        weight_vector_digest=weight_vector_digest(u16),
    )
    assert 7 in burn.audit_manifest.earning_inputs

    report = _auditor(source).audit_epoch(burn, store, NO_SAMPLE, None, NOW)

    fails = _census_fails(report)
    assert [v.uid for v in fails] == [7]
    assert report.overall is AuditStatus.DISPUTED


def test_genuinely_empty_epoch_burn_stays_clean(tmp_path) -> None:
    """A GENUINELY empty epoch (no committed evidence at all → burn) carries no evidenced uids,
    so NO census verdict fires and the legitimate empty-epoch burn stays CLEAN (DECISIONS #11)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    burn = fin.build_log(
        epoch_id=100, close_block=CLOSE_BLOCK, snapshots=(),
        burn_uid=BURN_UID, audit_manifest=AuditManifest(), now=NOW,
    )
    assert burn.burn_uid == BURN_UID and not burn.audit_manifest.earning_inputs

    report = _auditor(source).audit_epoch(burn, store, NO_SAMPLE, None, NOW)

    assert _census_fails(report) == []
    assert report.overall is AuditStatus.CLEAN


def test_empty_census_holds_when_metagraph_is_unavailable(tmp_path) -> None:
    """A transient chain outage is unknown, not proof that the subnet has no miners.

    The old empty-census early return ran before the ``metagraph is None`` guard and let a
    ``miners=[]`` burn log audit CLEAN while registered earning state could be erased.
    """
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    burn = EpochFinalizer(CFG, scorer_version=SCORER).build_log(
        epoch_id=100,
        close_block=CLOSE_BLOCK,
        snapshots=(),
        burn_uid=BURN_UID,
        audit_manifest=AuditManifest(),
        now=NOW,
    )
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID),
        source,
        chain=None,
    )

    report = auditor.audit_epoch(burn, store, NO_SAMPLE, None, NOW)

    census = next(v for v in report.earning_verdicts if v.item_id == "census:metagraph")
    assert census.verdict is ItemVerdictKind.SKIP
    assert report.overall is AuditStatus.INCONCLUSIVE


def test_registered_uid_missing_from_empty_census_is_disputed(tmp_path) -> None:
    """The close-block metagraph is the independent census source even with no evidence.

    A registered zero-state uid omitted from ``log.miners`` used to be invisible because the
    evidence-only census check had nothing to inspect.  Exact metagraph binding makes the
    omission conclusive and prevents it from erasing a cumulative replay boundary.
    """
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    burn = EpochFinalizer(CFG, scorer_version=SCORER).build_log(
        epoch_id=100,
        close_block=CLOSE_BLOCK,
        snapshots=(),
        burn_uid=BURN_UID,
        audit_manifest=AuditManifest(),
        now=NOW,
    )
    registered = make_miner(7, 0.0)
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID),
        source,
        chain=metagraph_chain([registered], close_block=CLOSE_BLOCK, close_block_time=NOW),
    )

    report = auditor.audit_epoch(burn, store, NO_SAMPLE, None, NOW)

    omitted = next(v for v in report.earning_verdicts if v.item_id == "census:7")
    assert omitted.verdict is ItemVerdictKind.FAIL and omitted.code == CENSUS_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_census_uses_exact_close_block_metagraph_when_available(tmp_path) -> None:
    """Post-close registration churn must not be relabelled as the epoch's census."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, _manifest, miners = _honest(store, source, {1: 0.8})
    chain = metagraph_chain(miners, close_block=CLOSE_BLOCK, close_block_time=NOW)
    historical = chain.neurons()
    calls: list[int] = []

    def neurons_at(block: int):
        calls.append(block)
        return historical

    def wrong_head():
        raise AssertionError("auditor read current-head neurons instead of close-block state")

    chain.neurons_at = neurons_at  # type: ignore[attr-defined]
    chain.neurons = wrong_head  # type: ignore[method-assign]
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID),
        source,
        chain=chain,
    )

    report = auditor.audit_epoch(log, store, NO_SAMPLE, None, NOW)

    assert calls == [CLOSE_BLOCK]
    assert report.overall is AuditStatus.CLEAN


def test_burn_to_noncanonical_uid_is_disputed(tmp_path) -> None:
    """an internal review: EpochLog._validate only checks the burn uid is the SOLE positive uid,
    so it lets an UNTRUSTED authority CHOOSE the recipient — an empty log burning 100% to a
    registered beneficiary IT controls. The auditor resolves the CANONICAL burn uid INDEPENDENTLY
    of the log (from config, BURN_UID here) and DISPUTES a log burning to ANY other uid
    (BURN_UID_MISMATCH) — the authority does not get to pick the burn recipient."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    honest_burn = fin.build_log(
        epoch_id=100, close_block=CLOSE_BLOCK, snapshots=(),
        burn_uid=BURN_UID, audit_manifest=AuditManifest(), now=NOW,
    )
    # Re-anchor the (genuinely-empty) burn to a NON-canonical uid the authority controls.
    noncanonical = 5
    shares = {noncanonical: 1.0}
    u16 = quantize_u16(shares)
    substituted = rebuild_log(
        honest_burn, burn_uid=noncanonical, weight_shares=shares, weight_u16=u16,
        weight_vector_digest=weight_vector_digest(u16),
    )

    report = _auditor(source).audit_epoch(substituted, store, NO_SAMPLE, None, NOW)

    burn = [
        v for v in report.earning_verdicts
        if v.code == BURN_UID_MISMATCH and v.verdict is ItemVerdictKind.FAIL
    ]
    assert [v.uid for v in burn] == [noncanonical]
    assert report.overall is AuditStatus.DISPUTED


def test_burn_to_canonical_uid_stays_clean(tmp_path) -> None:
    """an internal review: the honest empty-epoch burn to the CANONICAL uid (the value the auditor
    is configured with, matching the authority) audits CLEAN (DECISIONS #11 preserved)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    fin = EpochFinalizer(CFG, scorer_version=SCORER)
    burn = fin.build_log(
        epoch_id=100, close_block=CLOSE_BLOCK, snapshots=(),
        burn_uid=BURN_UID, audit_manifest=AuditManifest(), now=NOW,
    )

    report = _auditor(source).audit_epoch(burn, store, NO_SAMPLE, None, NOW)

    assert not [v for v in report.earning_verdicts if v.code == BURN_UID_MISMATCH]
    assert report.overall is AuditStatus.CLEAN


def test_burn_uid_uses_chain_state_over_stale_config(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    burn = EpochFinalizer(CFG, scorer_version=SCORER).build_log(
        epoch_id=100, close_block=CLOSE_BLOCK, snapshots=(),
        burn_uid=BURN_UID, audit_manifest=AuditManifest(), now=NOW,
    )
    chain = metagraph_chain([], close_block=CLOSE_BLOCK, close_block_time=NOW)
    chain.get_burn_uid = lambda: BURN_UID  # type: ignore[attr-defined]
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=999),
        source,
        chain=chain,
    )
    report = auditor.audit_epoch(burn, store, NO_SAMPLE, None, NOW)
    assert not [v for v in report.earning_verdicts if v.code == BURN_UID_MISMATCH]
    assert report.overall is AuditStatus.CLEAN


def test_chain_resolved_burn_sink_is_allowed_in_registered_census(tmp_path) -> None:
    """Exact census binding includes a burn sink that is itself registered on the subnet.

    The sink remains absent from the economic ``miners`` rows and earning evidence, but it must
    not disappear from the independent close-block registration set.
    """
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    sink = make_miner(BURN_UID, 0.0)
    burn = EpochFinalizer(CFG, scorer_version=SCORER).build_log(
        epoch_id=100, close_block=CLOSE_BLOCK, snapshots=(),
        miner_census=(MinerCensusEntry.from_miner(sink),),
        burn_uid=BURN_UID, audit_manifest=AuditManifest(), now=NOW,
    )
    chain = metagraph_chain(
        [sink], close_block=CLOSE_BLOCK, close_block_time=NOW
    )
    chain.get_burn_uid = lambda: BURN_UID  # type: ignore[attr-defined]
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=999),
        source,
        chain=chain,
    )

    report = auditor.audit_epoch(burn, store, NO_SAMPLE, None, NOW)

    assert not [v for v in report.earning_verdicts if v.code == CENSUS_MISMATCH]
    assert report.overall is AuditStatus.CLEAN


def test_registered_unknown_track_census_only_uid_stays_clean(tmp_path) -> None:
    """Offline/new registrations do not need a fabricated economic track.

    They remain exact-bindable in ``miner_census`` while the empty economic set legitimately
    burns, avoiding an unknown-track or offline-registration denial of service.
    """
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    registered = make_miner(7, 0.0, track="unknown")
    census = (MinerCensusEntry.from_miner(registered),)
    burn = EpochFinalizer(CFG, scorer_version=SCORER).build_log(
        epoch_id=100,
        close_block=CLOSE_BLOCK,
        snapshots=(),
        miner_census=census,
        burn_uid=BURN_UID,
        audit_manifest=AuditManifest(),
        now=NOW,
    )
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID),
        source,
        chain=metagraph_chain(
            [registered], close_block=CLOSE_BLOCK, close_block_time=NOW
        ),
    )

    report = auditor.audit_epoch(burn, store, NO_SAMPLE, None, NOW)

    bound = next(v for v in report.earning_verdicts if v.item_id == "census:7")
    assert bound.verdict is ItemVerdictKind.PASS
    assert report.overall is AuditStatus.CLEAN


def test_validator_permitted_miner_remains_in_exact_census(tmp_path) -> None:
    """Permit acquisition cannot erase a previously earning miner's replay identity."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    registered = make_miner(7, 0.0, track="unknown")
    burn = EpochFinalizer(CFG, scorer_version=SCORER).build_log(
        epoch_id=100,
        close_block=CLOSE_BLOCK,
        snapshots=(),
        miner_census=(MinerCensusEntry.from_miner(registered),),
        burn_uid=BURN_UID,
        audit_manifest=AuditManifest(),
        now=NOW,
    )
    chain = metagraph_chain(
        [registered], close_block=CLOSE_BLOCK, close_block_time=NOW
    )
    chain._neurons[0] = replace(chain._neurons[0], is_validator=True)
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID),
        source,
        chain=chain,
    )

    report = auditor.audit_epoch(burn, store, NO_SAMPLE, None, NOW)

    bound = next(v for v in report.earning_verdicts if v.item_id == "census:7")
    assert bound.verdict is ItemVerdictKind.PASS
    assert not [v for v in report.earning_verdicts if v.code == CENSUS_MISMATCH]
    assert report.overall is AuditStatus.CLEAN


def test_census_only_identity_mismatch_is_disputed(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    registered = make_miner(7, 0.0)
    false_census = (
        MinerCensusEntry(uid=7, hotkey="hk-impostor", coldkey="ck7", ip="10.0.0.7"),
    )
    burn = EpochFinalizer(CFG, scorer_version=SCORER).build_log(
        epoch_id=100,
        close_block=CLOSE_BLOCK,
        snapshots=(),
        miner_census=false_census,
        burn_uid=BURN_UID,
        audit_manifest=AuditManifest(),
        now=NOW,
    )
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID),
        source,
        chain=metagraph_chain(
            [registered], close_block=CLOSE_BLOCK, close_block_time=NOW
        ),
    )

    report = auditor.audit_epoch(burn, store, NO_SAMPLE, None, NOW)

    mismatch = next(v for v in report.earning_verdicts if v.item_id == "census:7")
    assert mismatch.verdict is ItemVerdictKind.FAIL
    assert mismatch.code == CENSUS_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_burn_uid_chain_read_failure_is_inconclusive_not_clean(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    burn = EpochFinalizer(CFG, scorer_version=SCORER).build_log(
        epoch_id=100, close_block=CLOSE_BLOCK, snapshots=(),
        burn_uid=BURN_UID, audit_manifest=AuditManifest(), now=NOW,
    )
    chain = metagraph_chain([], close_block=CLOSE_BLOCK, close_block_time=NOW)

    def unavailable():
        raise RuntimeError("owner registry unavailable")

    chain.get_burn_uid = unavailable  # type: ignore[attr-defined]
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=0),
        source,
        chain=chain,
    )
    report = auditor.audit_epoch(burn, store, NO_SAMPLE, None, NOW)
    skipped = [v for v in report.earning_verdicts if v.code == BURN_UID_UNVERIFIED]
    assert len(skipped) == 1 and skipped[0].verdict is ItemVerdictKind.SKIP
    assert report.overall is AuditStatus.INCONCLUSIVE


def test_below_cutoff_rank_losers_stay_clean(tmp_path) -> None:
    """REGRESSION: an HONEST epoch with MORE miners than ``top_n_per_track``
    (5) — so the lowest-scored miners rank below the cutoff and correctly receive ZERO weight —
    must stay CLEAN. The finalizer carries earning evidence for EVERY scored miner (not just the
    positive-weight ones), so those honest, PRESENT, zero-weight losers are evidenced; the census
    check must NOT mistake their legitimate zeroing for censorship (the round-9 #1 false-DISPUTE)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # 6 distinct miners on the same track, strictly descending scores → uids 1..5 make the top-5,
    # uid 6 ranks below the cutoff and gets zero weight (present + evidenced, but NOT paid).
    scores = {1: 0.90, 2: 0.80, 3: 0.70, 4: 0.60, 5: 0.50, 6: 0.40}
    log, _, _ = _honest(store, source, scores)

    assert 6 in log.audit_manifest.earning_inputs  # the loser IS evidenced
    assert any(m.uid == 6 for m in log.miners)  # and PRESENT in the census
    assert log.weight_shares.get(6, 0.0) == 0.0  # yet correctly UNPAID (below top_n)
    assert not any(m.uid == 6 and m.excluded for m in log.miners)  # not a dedup exclusion

    report = _auditor(source).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    assert _census_fails(report) == []  # the honest loser is NOT a census fault
    census = [v for v in report.earning_verdicts if v.item_id == "census:6"]
    assert census and all(v.verdict is ItemVerdictKind.PASS for v in census)
    assert report.overall is AuditStatus.CLEAN


def test_evidenced_canonical_burn_uid_is_disputed(tmp_path) -> None:
    """an internal review: the RESERVED burn uid must NOT double as a census/evidence identity.
    `EpochLog._validate` exempts burn_uid from manifest coverage, the auditor excludes it from the
    snapshot identity/dedup/track pass AND from the earning fold, and the census check only records
    that an evidenced row is PRESENT — so an untrusted log could seat the CANONICAL burn uid in
    `miners` carrying another miner's committed evidence + a self-attested hotkey and publish
    {burn_uid:1.0}: no binding or fold ever runs for the reserved uid ⇒ CLEAN. `_validate` now
    REFUSES such a log; this exercises the auditor's DEFENSE IN DEPTH on a tampered log that
    BYPASSED the finalizer (built via model_construct) ⇒ DISPUTED via the burn verdict."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    # An honest epoch scored a real target (uid 7); harvest its committed evidence.
    log, manifest, _ = _honest(store, source, {7: 0.6})
    target_refs = manifest.per_uid[7]
    target_ei = manifest.earning_inputs[7]

    # Tamper: re-key the target's evidence under the CANONICAL burn uid, seat it as a census miner
    # with a self-attested hotkey, and publish the burn vector. _validate refuses this, so bypass
    # it with model_construct — a log that dodged the finalizer/validation, handed to the auditor.
    tampered_manifest = AuditManifest(
        per_uid={BURN_UID: target_refs},
        earning_inputs={BURN_UID: target_ei},
        score_packet_merkle_root=manifest.score_packet_merkle_root,
    )
    seated = make_miner(BURN_UID, fold(0.0, [0.6]))  # self-attested hk{BURN_UID}
    shares = {BURN_UID: 1.0}
    u16 = quantize_u16(shares)
    tampered = EpochLog.model_construct(
        schema_version=log.schema_version,
        epoch_id=100, close_block=CLOSE_BLOCK, scorer_version=SCORER, created_at=NOW,
        prior_log_digest=None, burn_uid=BURN_UID, miners=(seated,),
        weight_shares=shares, weight_u16=u16,
        weight_vector_digest=weight_vector_digest(u16), audit_manifest=tampered_manifest,
    )
    assert tampered.burn_uid == BURN_UID  # the recipient IS canonical — only the overlap is the fault

    report = _auditor(source).audit_epoch(tampered, store, NO_SAMPLE, None, NOW)

    burn = [
        v for v in report.earning_verdicts
        if v.code == BURN_UID_MISMATCH and v.verdict is ItemVerdictKind.FAIL
    ]
    assert [v.uid for v in burn] == [BURN_UID]
    assert report.overall is AuditStatus.DISPUTED


def test_honest_full_census_is_clean(tmp_path) -> None:
    """An HONEST epoch whose census contains every evidenced miner passes the census check
    (and the epoch stays CLEAN)."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    log, _, _ = _honest(store, source, {1: 0.8, 7: 0.6})

    report = _auditor(source).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    assert _census_fails(report) == []
    # every evidenced uid is present -> census PASS for each, and the epoch is CLEAN.
    census = [v for v in report.earning_verdicts if v.item_id.startswith("census:")]
    assert census and all(v.verdict is ItemVerdictKind.PASS for v in census)
    assert report.overall is AuditStatus.CLEAN
