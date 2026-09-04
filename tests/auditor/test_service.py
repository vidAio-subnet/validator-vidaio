"""Auditor.audit_epoch: weight re-derivation, item aggregation, and the client seam.

All media-free: a StaticRecomputer stands in for the scoring engine so PASS/FAIL is
driven by whether a packet matches the fixed recompute (the real recompute has its
own real-ffmpeg tests in test_recomputer.py).
"""

from __future__ import annotations

from vidaio.audit.recompute import SCORE_MISMATCH, StaticRecomputer
from vidaio.audit.store import LocalFsStore
from vidaio.epoch.log import AuditFileKind, AuditManifest, EpochLog
from vidaio.auditor import (
    WEIGHT_DERIVATION_MISMATCH,
    Auditor,
    AuditorConfig,
    AuditMode,
    AuditStatus,
    InMemoryBundleSource,
    ItemVerdictKind,
    RecordingAuditResultsClient,
    SamplePolicy,
    Sha256Signer,
    overall_status,
)
from vidaio.auditor.report import ItemVerdict, WeightVerdict
from vidaio.tokenomics import quantize_u16
from vidaio.epoch.log import weight_vector_digest

from tests.auditor.fakes import (
    NOW,
    SCORER,
    BURN_UID,
    MetagraphAuditor,
    folded_miner,
    make_fake_bundle,
    honest_log,
    make_packet,
    refs_for,
    scored_item,
)


def _static_recomputer() -> StaticRecomputer:
    from tests.auditor.fakes import HONEST_METRICS, HONEST_SCORE

    return StaticRecomputer(HONEST_METRICS, SCORER, score=HONEST_SCORE)


def _epoch_with_items(store: LocalFsStore, bundles) -> tuple[EpochLog, InMemoryBundleSource]:
    """Honest epoch: each bundle's committed packet (score + cycle_sequence) drives a
    consistent ScoredItem + fold-matched miner, so the evidence-bound earning re-fold
    and the manifest coverage both hold."""
    import json

    from vidaio.authority import build_audit_manifest

    source = InMemoryBundleSource()
    items, miners = [], []
    for uid, bundle in enumerate(bundles, start=1):
        source.add(bundle)
        payload = json.loads(store.get(bundle.score_packet))
        score, seq = float(payload["score"]), int(payload["cycle_sequence"])
        items.append(scored_item(bundle, uid, score=score, seq=seq))
        miners.append(folded_miner(uid, score))
    manifest = build_audit_manifest(items)
    log = honest_log(miners, manifest)
    return log, source


def _auditor(source, **kw) -> Auditor:
    kw.setdefault("burn_uid", BURN_UID)
    config = AuditorConfig(auditor_hotkey="hkAuditor", **kw)
    # MetagraphAuditor auto-wires the close-block metagraph from the log's honest
    # identities so the SNAPSHOT binding passes for honest fixtures.
    # an internal review: a strict MISSING reveal verifier is now an INCONCLUSIVE skip (no
    # longer washed to PASS), so media-sampling fixtures wire a trivial reveal verifier — the
    # fake bundles' DAGs are not real build_dag outputs (the deep verifier has its own tests
    # in tests/audit/test_recompute.py); here the media recompute / merkle checks are the point.
    return MetagraphAuditor(config, source, reveal_verifier=lambda dag_bytes: True)


# --- weight re-derivation ------------------------------------------------------------


def test_honest_log_passes_weight_derivation(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    b1 = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    log, source = _epoch_with_items(store, [b1])

    auditor = _auditor(source)
    report = auditor.audit_epoch(log, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW)

    assert report.weight_verdict.verdict is ItemVerdictKind.PASS
    assert report.weight_verdict.code == ""


def test_substituted_weight_fails_derivation_without_media(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    b1 = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    b2 = make_fake_bundle(store, challenge_id="c1", item_id="i2", miner_hotkey="hk2")
    honest, source = _epoch_with_items(store, [b1, b2])

    # Substitute a log whose weight_shares do NOT follow from build_weight_vector, but
    # are still internally consistent (u16 = quantize(shares), digest binds it, every
    # nonzero uid backed) so it constructs — only an independent re-derivation catches it.
    substituted_shares = {1: 0.99, 2: 0.01}
    substituted_u16 = quantize_u16(substituted_shares)
    substituted = EpochLog(
        schema_version=honest.schema_version,
        epoch_id=honest.epoch_id,
        close_block=honest.close_block,
        scorer_version=honest.scorer_version,
        created_at=honest.created_at,
        burn_uid=None,
        miners=honest.miners,
        miner_census=honest.miner_census,
        # This substitution case intentionally uses an inference-only log.
        weight_shares=substituted_shares,
        weight_u16=substituted_u16,
        weight_vector_digest=weight_vector_digest(substituted_u16),
        audit_manifest=honest.audit_manifest,
    )
    assert substituted.weight_u16 != honest.weight_u16  # a genuinely different vector

    auditor = _auditor(source)
    # No items sampled (rate 0, min 0) → the dispute is PURELY the substituted weight.
    report = auditor.audit_epoch(
        substituted, store, SamplePolicy(sample_rate=0.0, min_samples=0), _static_recomputer(), NOW
    )

    assert report.weight_verdict.verdict is ItemVerdictKind.FAIL
    assert report.weight_verdict.code == WEIGHT_DERIVATION_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_omitted_sink_can_never_audit_pass_even_if_partial_vector_quantizes(
    tmp_path,
) -> None:
    """Defense in depth for a malformed/legacy producer bypassing EpochLog validation."""
    store = LocalFsStore(tmp_path / "s")
    bundles = [
        make_fake_bundle(
            store,
            challenge_id="c1",
            item_id=f"i{uid}",
            miner_hotkey=f"hk{uid}",
        )
        for uid in (1, 2)
    ]
    honest, source = _epoch_with_items(store, bundles)
    assert honest.burn_uid == BURN_UID

    partial_shares = {
        uid: share
        for uid, share in honest.weight_shares.items()
        if uid != honest.burn_uid
    }
    assert 0.0 < sum(partial_shares.values()) < 1.0
    # This is the historical hazard: an internally normalized u16 vector can look
    # complete even though the raw fixed allocation omitted its sink remainder.
    partial_u16 = quantize_u16(partial_shares)
    assert sum(partial_u16.values()) == 65535
    malformed = honest.model_copy(
        update={
            "burn_uid": None,
            "weight_shares": partial_shares,
            "weight_u16": partial_u16,
            "weight_vector_digest": weight_vector_digest(partial_u16),
        }
    )

    report = _auditor(source).audit_epoch(
        malformed,
        store,
        SamplePolicy(sample_rate=0.0, min_samples=0),
        _static_recomputer(),
        NOW,
    )

    assert report.weight_verdict.verdict is ItemVerdictKind.FAIL
    assert report.weight_verdict.code == WEIGHT_DERIVATION_MISMATCH
    assert "canonical burn_uid is required" in report.weight_verdict.detail
    assert report.overall is AuditStatus.DISPUTED


# --- item aggregation: CLEAN vs DISPUTED --------------------------------------------


def test_all_honest_items_is_clean(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    bundles = [
        make_fake_bundle(store, challenge_id="c1", item_id=f"i{u}", miner_hotkey=f"hk{u}")
        for u in (1, 2, 3)
    ]
    log, source = _epoch_with_items(store, bundles)

    auditor = _auditor(source)
    report = auditor.audit_epoch(log, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW)

    assert len(report.item_verdicts) == 3
    assert all(v.verdict is ItemVerdictKind.PASS for v in report.item_verdicts)
    assert report.overall is AuditStatus.CLEAN
    assert report.inference_n == 3


def test_one_tampered_item_disputes_the_epoch(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    honest_b = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    tampered_packet = make_packet(
        challenge_id="c1", item_id="i2", miner_hotkey="hk2", score=0.99,
        metrics={"compression_rate": 0.125, "vmaf": 93.42, "final_score": 0.99},
    )
    tampered_b = make_fake_bundle(
        store, challenge_id="c1", item_id="i2", miner_hotkey="hk2", packet=tampered_packet
    )
    log, source = _epoch_with_items(store, [honest_b, tampered_b])

    auditor = _auditor(source)
    report = auditor.audit_epoch(log, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW)

    verdicts = {v.item_id: v for v in report.item_verdicts}
    assert verdicts["i1"].verdict is ItemVerdictKind.PASS
    assert verdicts["i2"].verdict is ItemVerdictKind.FAIL
    assert verdicts["i2"].code == SCORE_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


# --- merkle inclusion: strict membership against the committed root -----------------


def _scored(bundle, uid):
    return scored_item(bundle, uid)


def _epoch_with_committed_manifest(store, bundles):
    """Epoch log whose manifest carries a real merkle root + per-item inclusion proofs."""
    from vidaio.authority import build_audit_manifest

    source = InMemoryBundleSource()
    items, miners = [], []
    for uid, bundle in enumerate(bundles, start=1):
        source.add(bundle)
        items.append(scored_item(bundle, uid))
        miners.append(folded_miner(uid))
    manifest = build_audit_manifest(items)  # computes root + proofs
    return honest_log(miners, manifest), source


def test_honest_committed_items_pass_strict_merkle_inclusion(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    bundles = [
        make_fake_bundle(store, challenge_id="c1", item_id=f"i{u}", miner_hotkey=f"hk{u}")
        for u in (1, 2, 3)
    ]
    log, source = _epoch_with_committed_manifest(store, bundles)
    assert log.audit_manifest.score_packet_merkle_root is not None

    auditor = _auditor(source)  # strict=True by default now
    report = auditor.audit_epoch(
        log, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW
    )
    assert all(v.verdict is ItemVerdictKind.PASS for v in report.item_verdicts)
    assert report.overall is AuditStatus.CLEAN


def test_item_outside_committed_root_disputes_with_merkle_exclusion(tmp_path) -> None:
    from vidaio.audit.recompute import MERKLE_EXCLUSION
    from vidaio.authority import build_audit_manifest

    store = LocalFsStore(tmp_path / "s")
    b1 = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    b2 = make_fake_bundle(store, challenge_id="c1", item_id="i2", miner_hotkey="hk2")
    # item 3 is a fully honest, resolvable, recomputable bundle — but it was NOT part
    # of the committed score-packet set (its packet is not a leaf of the anchored root).
    b3 = make_fake_bundle(store, challenge_id="c1", item_id="i3", miner_hotkey="hk3")

    from vidaio.epoch.log import CycleScore, EarningInput
    from tests.auditor.fakes import HONEST_SCORE

    committed = build_audit_manifest([_scored(b1, 1), _scored(b2, 2)])
    per_uid = dict(committed.per_uid)
    per_uid[3] = refs_for(b3)  # injected: refs carry no inclusion proof
    earning = dict(committed.earning_inputs)
    # uid 3 carries a consistent earning input (so the finalizer accepts the log); only
    # its MEDIA membership is tampered (packet outside the committed root).
    earning[3] = EarningInput(
        cycle_scores=(
            CycleScore(packet_digest=b3.score_packet.digest, ordering_key=0, score=HONEST_SCORE),
        )
    )
    tampered_manifest = AuditManifest(
        per_uid=per_uid,
        score_packet_merkle_root=committed.score_packet_merkle_root,
        earning_inputs=earning,
        fold_cursors={**committed.fold_cursors, 3: 0},
    )
    miners = [folded_miner(u) for u in (1, 2, 3)]
    log = honest_log(miners, tampered_manifest)

    source = InMemoryBundleSource()
    for b in (b1, b2, b3):
        source.add(b)

    auditor = _auditor(source)
    report = auditor.audit_epoch(
        log, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW
    )
    verdicts = {v.item_id: v for v in report.item_verdicts}
    assert verdicts["i1"].verdict is ItemVerdictKind.PASS
    assert verdicts["i2"].verdict is ItemVerdictKind.PASS
    assert verdicts["i3"].verdict is ItemVerdictKind.FAIL
    assert verdicts["i3"].code == MERKLE_EXCLUSION
    assert report.overall is AuditStatus.DISPUTED


def test_unresolvable_bundle_is_skip_not_fail(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    b1 = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    log, source = _epoch_with_items(store, [b1])
    # Drop the bundle from the source so it cannot be resolved (unreachable).
    source._by_digest.clear()

    auditor = _auditor(source)
    report = auditor.audit_epoch(log, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW)

    assert report.item_verdicts[0].verdict is ItemVerdictKind.SKIP
    # a SKIP never DISPUTES, but an all-SKIP media sample was never recomputed, so it
    # is INCONCLUSIVE (needs-attention), never washed to CLEAN (#8).
    assert report.overall is AuditStatus.INCONCLUSIVE


# --- the client seam + signing -------------------------------------------------------


def test_client_seam_records_the_submitted_report(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    b1 = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    log, source = _epoch_with_items(store, [b1])

    client = RecordingAuditResultsClient()
    signer = Sha256Signer("auditor-secret")
    auditor = MetagraphAuditor(
        AuditorConfig(auditor_hotkey="hkAuditor"), source, signer=signer, client=client
    )
    report, ack = auditor.audit_and_submit(
        log, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW
    )

    assert client.submitted == [report]
    assert ack.accepted and ack.report_id == report.report_digest()
    # the report is signed over its canonical (unsigned) bytes
    assert report.auditor_signature == signer.sign(report.canonical_bytes())
    assert signer.verify(report.canonical_bytes(), report.auditor_signature)


def test_report_is_deterministic(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    bundles = [
        make_fake_bundle(store, challenge_id="c1", item_id=f"i{u}", miner_hotkey=f"hk{u}")
        for u in (1, 2)
    ]
    log, source = _epoch_with_items(store, bundles)
    auditor = _auditor(source)

    r1 = auditor.audit_epoch(log, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW)
    r2 = auditor.audit_epoch(log, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW)
    assert r1.canonical_bytes() == r2.canonical_bytes()
    assert r1.report_digest() == r2.report_digest()


def test_report_mode_preserves_legacy_beacon_signature_and_separates_own_audit(
    tmp_path,
) -> None:
    """Beacon is wire-compatible with old reports; own-audit is a signed namespace."""
    store = LocalFsStore(tmp_path / "s")
    bundle = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    log, source = _epoch_with_items(store, [bundle])
    signer = Sha256Signer("auditor-secret")

    beacon = MetagraphAuditor(
        AuditorConfig(auditor_hotkey="hkAuditor"),
        source,
        signer=signer,
        reveal_verifier=lambda _dag: True,
    ).audit_epoch(log, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW)

    # A historical body with no field parses as beacon, and beacon's canonical
    # payload still omits the new field so its old signature/digest remain valid.
    historical_body = beacon.model_dump(mode="json")
    historical_body.pop("audit_mode")
    parsed = beacon.__class__.model_validate(historical_body)
    assert parsed.audit_mode is AuditMode.BEACON
    assert parsed.canonical_bytes() == beacon.canonical_bytes()
    assert b'"audit_mode"' not in beacon.canonical_bytes()
    assert signer.verify(parsed.canonical_bytes(), parsed.auditor_signature)

    own = MetagraphAuditor(
        AuditorConfig(auditor_hotkey="hkAuditor", audit_mode=AuditMode.OWN_AUDIT),
        source,
        signer=signer,
        reveal_verifier=lambda _dag: True,
    ).audit_epoch(log, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW)
    assert own.audit_mode is AuditMode.OWN_AUDIT
    assert b'"audit_mode":"own_audit"' in own.canonical_bytes()
    assert own.report_digest() != beacon.report_digest()
    assert signer.verify(own.canonical_bytes(), own.auditor_signature)


def test_substituted_track_cannot_dodge_verification_via_skip(tmp_path) -> None:
    """#9: a packet that declares track=upscaling (to force a GPU-unavailable SKIP) over
    a committed-compression item is a FAIL, not a dodge — the committed track governs."""
    from vidaio.audit.recompute import IDENTITY_MISMATCH
    from vidaio.auditor import persist_bundle
    from vidaio.authority import ScoredItem, build_audit_manifest

    store = LocalFsStore(tmp_path / "s")
    packet = make_packet(
        challenge_id="c1", item_id="i1", miner_hotkey="hk1", track="upscaling", score=0.81,
    )
    b = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1", packet=packet)
    persist_bundle(store, b)
    item = ScoredItem(
        uid=1, hotkey="hk1", challenge_id="c1", item_id="i1",
        bundle_digest=b.bundle_digest(), packet_digest=b.score_packet.digest,
        committed_track="compression",  # the COMMITTED track (packet substitutes upscaling)
        score=0.81, cycle_sequence=0,
    )
    manifest = build_audit_manifest([item], store=store)
    log = honest_log([folded_miner(1, 0.81)], manifest)

    source = InMemoryBundleSource()
    source.add(b)
    report = _auditor(source).audit_epoch(
        log, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW
    )
    v = report.item_verdicts[0]
    assert v.verdict is ItemVerdictKind.FAIL
    assert v.code == IDENTITY_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_report_overall_is_derived_never_trusted() -> None:
    """#7: a report can never CLAIM clean while carrying a fault — overall is derived."""
    from vidaio.audit.recompute import SCORE_MISMATCH as _SM
    from vidaio.auditor import AuditReport
    from datetime import datetime, timezone

    wv = WeightVerdict(
        recomputed_weight_vector_digest="a", published_weight_vector_digest="a",
        verdict=ItemVerdictKind.PASS,
    )
    fail_item = ItemVerdict(
        source="inference", challenge_id="c", item_id="i", bundle_digest="b",
        packet_digest="p", verdict=ItemVerdictKind.FAIL, code=_SM,
    )
    report = AuditReport(
        auditor_hotkey="hk", epoch_id=1, snapshot_digest="d", pipeline_version="v",
        sampled_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
        item_verdicts=(fail_item,), weight_verdict=wv,
        overall=AuditStatus.CLEAN,  # a LIE — construction overwrites it
    )
    assert report.overall is AuditStatus.DISPUTED


def test_overall_status_helper() -> None:
    wv_pass = WeightVerdict(
        recomputed_weight_vector_digest="a", published_weight_vector_digest="a",
        verdict=ItemVerdictKind.PASS,
    )
    wv_fail = wv_pass.model_copy(update={"verdict": ItemVerdictKind.FAIL})

    def item(v: ItemVerdictKind) -> ItemVerdict:
        return ItemVerdict(
            source="inference", challenge_id="c", item_id="i", bundle_digest="b",
            packet_digest="p", verdict=v,
        )

    assert overall_status((item(ItemVerdictKind.PASS),), wv_pass) is AuditStatus.CLEAN
    # an all-SKIP media sample is INCONCLUSIVE, not CLEAN (#8)
    assert overall_status((item(ItemVerdictKind.SKIP),), wv_pass) is AuditStatus.INCONCLUSIVE
    # Launch production selects every media row. A PASS cannot wash an independently
    # selected, CPU-unrecomputable GPU score to CLEAN.
    assert overall_status(
        (item(ItemVerdictKind.PASS), item(ItemVerdictKind.SKIP)), wv_pass
    ) is AuditStatus.INCONCLUSIVE
    assert overall_status((item(ItemVerdictKind.FAIL),), wv_pass) is AuditStatus.DISPUTED
    assert overall_status((item(ItemVerdictKind.PASS),), wv_fail) is AuditStatus.DISPUTED

    # an internal review: the coverage floor is LABEL-INDEPENDENT. A sampled item that
    # all-SKIPs is INCONCLUSIVE regardless of the (spoofable) source label — an off-list
    # source can no longer let an all-SKIP sample wash to CLEAN.
    def item_src(v: ItemVerdictKind, src: str) -> ItemVerdict:
        return ItemVerdict(
            source=src, challenge_id="c", item_id="i", bundle_digest="b",
            packet_digest="p", verdict=v,
        )

    assert overall_status(
        (item_src(ItemVerdictKind.SKIP, "totally-made-up"),), wv_pass
    ) is AuditStatus.INCONCLUSIVE
    for source in ("inference", "competition"):
        assert overall_status(
            (
                item_src(ItemVerdictKind.PASS, source),
                item_src(ItemVerdictKind.SKIP, source),
            ),
            wv_pass,
        ) is AuditStatus.INCONCLUSIVE
    assert overall_status(
        (
            item_src(ItemVerdictKind.PASS, "inference"),
            item_src(ItemVerdictKind.SKIP, "competition"),
        ),
        wv_pass,
    ) is AuditStatus.INCONCLUSIVE

    wv_skip = wv_pass.model_copy(update={"verdict": ItemVerdictKind.SKIP})
    assert overall_status((item(ItemVerdictKind.PASS),), wv_skip) is AuditStatus.INCONCLUSIVE


#


def test_manifest_incomplete_yields_signed_disputed(tmp_path) -> None:
    """an internal review: a manifest with an item missing its bundle/packet pair cannot be
    sampled. The auditor emits a SIGNED DISPUTED report (MANIFEST_INCOMPLETE) instead of
    letting the ManifestIncomplete exception escape and block the cursor forever."""
    from vidaio.auditor.report import MANIFEST_INCOMPLETE

    store = LocalFsStore(tmp_path / "s")
    b1 = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    log, source = _epoch_with_items(store, [b1])
    # Drop the AUDIT_BUNDLE ref for uid 1 -> its item is unpaired -> ManifestIncomplete at
    # sampling. model_copy keeps the (finalizer-produced) earning inputs intact, so the ONLY
    # defect is the structural one. This is the malformed manifest an auditor is handed.
    packet_only = tuple(
        r for r in log.audit_manifest.per_uid[1] if r.kind is AuditFileKind.SCORE_PACKET
    )
    incomplete_manifest = log.audit_manifest.model_copy(update={"per_uid": {1: packet_only}})
    incomplete = log.model_copy(update={"audit_manifest": incomplete_manifest})

    signer = Sha256Signer("auditor-secret")
    auditor = MetagraphAuditor(
        AuditorConfig(
            auditor_hotkey="hkAuditor", audit_mode=AuditMode.OWN_AUDIT
        ),
        source,
        signer=signer,
    )
    report = auditor.audit_epoch(
        incomplete, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW
    )

    assert report.weight_verdict.verdict is ItemVerdictKind.FAIL
    assert report.weight_verdict.code == MANIFEST_INCOMPLETE
    assert report.audit_mode is AuditMode.OWN_AUDIT
    assert report.overall is AuditStatus.DISPUTED
    # a SIGNED report, not an uncaught exception
    assert report.auditor_signature == signer.sign(report.canonical_bytes())


def test_malformed_inclusion_proof_hex_yields_signed_disputed(tmp_path) -> None:
    """an internal review: a malformed inclusion-proof sibling hex must not crash the audit.
    It is a structural defect (the packet is not provably in the committed set), so the
    sampled item FAILs MERKLE_EXCLUSION and the epoch rolls up to a SIGNED DISPUTED report,
    never an uncaught bytes.fromhex ValueError that blocks the cursor."""
    from vidaio.audit.recompute import MERKLE_EXCLUSION

    store = LocalFsStore(tmp_path / "s")
    b1 = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    log, source = _epoch_with_items(store, [b1])

    # Tamper ONLY the SCORE_PACKET ref's inclusion proof with a non-hex sibling. model_copy
    # bypasses model validation (inclusion-proof hex is not model-validated — the hole), so
    # this is exactly the malformed manifest an auditor is handed on the wire.
    good_refs = log.audit_manifest.per_uid[1]
    tampered_refs = tuple(
        r.model_copy(update={"inclusion_proof": (("zz", "left"),)})
        if r.kind is AuditFileKind.SCORE_PACKET
        else r
        for r in good_refs
    )
    tampered_manifest = log.audit_manifest.model_copy(update={"per_uid": {1: tampered_refs}})
    tampered = log.model_copy(update={"audit_manifest": tampered_manifest})

    signer = Sha256Signer("auditor-secret")
    auditor = MetagraphAuditor(AuditorConfig(auditor_hotkey="hkAuditor"), source, signer=signer)
    report = auditor.audit_epoch(
        tampered, store, SamplePolicy(sample_rate=1.0), _static_recomputer(), NOW
    )

    assert any(
        v.verdict is ItemVerdictKind.FAIL and v.code == MERKLE_EXCLUSION
        for v in report.item_verdicts
    )
    assert report.overall is AuditStatus.DISPUTED
    assert report.auditor_signature == signer.sign(report.canonical_bytes())


#


def test_strict_missing_merkle_root_is_not_clean(tmp_path) -> None:
    """an internal review: under strict=True `verify_bundle` records a MISSING published merkle
    root as a FAILED-but-`skipped` check (`passed=False, skipped=True`) — "strict mode treats
    skipped checks as failures". `_verdict_from` previously excluded EVERY `skipped` check
    regardless of `passed`, so this strict FAILURE mapped to PASS. It must NOT: an unverifiable
    anchor fails CLOSED to SKIP (INCONCLUSIVE), never PASS. Only the merkle root is missing here
    (recompute / digest / reveal all pass), so the report's sole failure is the strict skip."""
    from vidaio.audit.recompute import verify_bundle

    store = LocalFsStore(tmp_path / "s")
    b = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    report = verify_bundle(
        b, store, _static_recomputer(),
        expected_bundle_digest=b.bundle_digest(),
        expected_miner_hotkey="hk1", require_expected_miner=True,
        published_root=None,  # MISSING published merkle root
        inclusion_proof=None,
        reveal_verifier=lambda dag_bytes: True,  # reveal wired — only the root is missing
        strict=True,
    )
    assert report.passed is False  # strict treats the missing root as a failure
    verdict, _code, _detail = Auditor._verdict_from(report)
    assert verdict is not ItemVerdictKind.PASS  # a strict FAIL must never wash to PASS
    assert verdict is ItemVerdictKind.SKIP  # unverifiable ⇒ INCONCLUSIVE, fail closed


def test_strict_missing_reveal_verifier_is_not_clean(tmp_path) -> None:
    """an internal review: under strict=True a MISSING reveal verifier is a FAILED-but-`skipped`
    check (`passed=False, skipped=True`). It must NOT map to PASS — an unverified DAG regeneration
    fails CLOSED to SKIP (INCONCLUSIVE). Only the reveal verifier is missing here (merkle
    inclusion / recompute / digest all pass), so the report's sole failure is the strict skip."""
    from vidaio.audit.recompute import verify_bundle
    from vidaio.authority import build_audit_manifest

    store = LocalFsStore(tmp_path / "s")
    b = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    manifest = build_audit_manifest([scored_item(b, 1)])  # real merkle root + inclusion proof
    packet_ref = next(
        r for r in manifest.per_uid[1] if r.kind is AuditFileKind.SCORE_PACKET
    )
    report = verify_bundle(
        b, store, _static_recomputer(),
        expected_bundle_digest=b.bundle_digest(),
        expected_miner_hotkey="hk1", require_expected_miner=True,
        published_root=manifest.score_packet_merkle_root,
        inclusion_proof=packet_ref.inclusion_proof,
        reveal_verifier=None,  # MISSING reveal verifier
        strict=True,
    )
    assert report.passed is False  # strict treats the missing verifier as a failure
    verdict, _code, _detail = Auditor._verdict_from(report)
    assert verdict is not ItemVerdictKind.PASS  # a strict FAIL must never wash to PASS
    assert verdict is ItemVerdictKind.SKIP  # unverifiable ⇒ INCONCLUSIVE, fail closed
