"""EpochFinalizer + build_audit_manifest: the _FINALIZED producer + manifest contract."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from vidaio.audit.canonical import sha256_hex
from vidaio.audit.store import (
    ArtifactKind,
    LocalFsStore,
    SetNotFinalizedError,
    set_member_key,
)
from vidaio.authority import (
    AuditFileMissingError,
    EpochFinalizer,
    ScoredItem,
    build_audit_manifest,
    epoch_prefix,
)
from vidaio.epoch import AuditFileKind, AuditManifest, EpochLog
from vidaio.tokenomics import MinerSnapshot, TokenomicsConfig

from vidaio.tokenomics.ewma import accumulate

NOW = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
SCORER = "scoring-1.0.0+abc123def456"
DECAY = TokenomicsConfig().ewma_decay


def _acc(score: float) -> float:
    """The genesis EWMA accumulate for a single ``score`` cycle."""
    return accumulate(0.0, score, DECAY)


@pytest.fixture
def store(tmp_path: Path) -> LocalFsStore:
    return LocalFsStore(tmp_path / "audit")


@pytest.fixture
def finalizer() -> EpochFinalizer:
    return EpochFinalizer(TokenomicsConfig(), scorer_version=SCORER)


def _miner(uid: int, score: float, track: str = "compression") -> MinerSnapshot:
    return MinerSnapshot(
        uid=uid,
        hotkey=f"hk{uid}",
        coldkey=f"ck{uid}",
        ip=f"10.0.0.{uid}",
        track=track,
        accumulate_score=score,
    )


def _item(
    uid: int, store: LocalFsStore | None = None, *, score: float = 0.8, seq: int = 0
) -> ScoredItem:
    packet = f"packet-bytes-{uid}".encode()
    bundle = f"bundle-bytes-{uid}".encode()
    if store is not None:
        store.put(packet, ArtifactKind.SCORE_PACKET)  # make it a REAL stored file
        store.put(bundle, ArtifactKind.AUDIT_BUNDLE)  # a resolvable bundle object (#8)
    return ScoredItem(
        uid=uid,
        hotkey=f"hk{uid}",
        challenge_id="c1",
        item_id=f"i{uid}",
        bundle_digest=sha256_hex(bundle),
        packet_digest=sha256_hex(packet),
        committed_track="compression",  # REQUIRED (#9)
        score=score,
        cycle_sequence=seq,
    )


# --------------------------------------------------------------------------------------
# build_audit_manifest — the manifest-assembly contract.
# --------------------------------------------------------------------------------------


def test_authority_refuses_incomplete_allocation_without_canonical_sink(
    finalizer: EpochFinalizer, store: LocalFsStore
) -> None:
    """A producer cannot publish a vector whose fixed pools sum below one.

    IDLE assigns only part of total emissions to this lone compression miner.  If
    the canonical sink is absent, the chain would normalize that partial vector and
    donate all withheld pools to the miner.  Refuse before an EpochLog can exist.
    """
    manifest = build_audit_manifest([_item(1, store, score=0.8)], store=store)

    with pytest.raises(ValueError, match="canonical burn_uid is required"):
        finalizer.build_log(
            epoch_id=1,
            close_block=359,
            snapshots=(_miner(1, _acc(0.8)),),
            burn_uid=None,  # type: ignore[arg-type] - exercise the runtime boundary
            audit_manifest=manifest,
            now=NOW,
        )


def test_manifest_lists_exactly_the_files_backing_each_uid(store: LocalFsStore) -> None:
    items = [_item(1, store), _item(2, store)]
    manifest = build_audit_manifest(items, store=store)
    assert set(manifest.per_uid) == {1, 2}
    for uid in (1, 2):
        refs = manifest.refs_for(uid)
        kinds = {r.kind for r in refs}
        assert kinds == {AuditFileKind.AUDIT_BUNDLE, AuditFileKind.SCORE_PACKET}
        digests = {r.digest for r in refs}
        assert sha256_hex(f"bundle-bytes-{uid}".encode()) in digests
        assert sha256_hex(f"packet-bytes-{uid}".encode()) in digests
        assert all(r.item_id == f"i{uid}" for r in refs)


class _FakeCommitmentSource:
    """A pre-dispatch challenge-commitment lookup. Maps a
    challenge_id -> its committed (track, dispatch_ordering_key); unknown -> None."""

    def __init__(self, committed: dict[str, tuple[str, int]]) -> None:
        self._committed = committed

    def committed_dispatch(self, challenge_id: str) -> tuple[str, int] | None:
        return self._committed.get(challenge_id)


def test_manifest_sources_track_and_ordering_key_from_commitment(store: LocalFsStore) -> None:
    """With a commitment_source the finalizer SOURCES the fold order + track from the
    pre-dispatch commitment, NOT from the (finalization-time) ScoredItem — so it cannot
    invent them. Here the item states cycle_sequence=0 but the
    committed dispatch key is 7: the manifest carries the COMMITTED 7."""
    item = _item(1, store, seq=0)  # the item's OWN stated sequence is 0
    source = _FakeCommitmentSource({"c1": ("compression", 7)})
    manifest = build_audit_manifest([item], store=store, commitment_source=source)

    cs = manifest.earning_for(1).cycle_scores
    assert [c.ordering_key for c in cs] == [7]  # the CHALLENGE-committed key, not the item's 0
    packet_ref = next(
        r for r in manifest.refs_for(1) if r.kind is AuditFileKind.SCORE_PACKET
    )
    assert packet_ref.committed_track == "compression"  # sourced from the commitment


def test_finalizer_refuses_item_without_pre_dispatch_commitment(store: LocalFsStore) -> None:
    """An item whose challenge has NO pre-dispatch commitment cannot enter an auditable
    earning fold: the finalizer REFUSES rather than invent an order."""
    item = _item(1, store)
    empty = _FakeCommitmentSource({})  # no commitment for challenge "c1"
    with pytest.raises(AuditFileMissingError, match="NO pre-dispatch challenge commitment"):
        build_audit_manifest([item], store=store, commitment_source=empty)


def test_manifest_rejects_unstored_packet(store: LocalFsStore) -> None:
    # item's packet was never put into the store -> honest refusal.
    ghost = ScoredItem(
        uid=1,
        hotkey="hk1",
        challenge_id="c1",
        item_id="i1",
        bundle_digest=sha256_hex(b"bundle-1"),
        packet_digest=sha256_hex(b"never-stored"),
        committed_track="compression",
    )
    with pytest.raises(AuditFileMissingError, match="not in"):
        build_audit_manifest([ghost], store=store)


def test_manifest_rejects_unresolvable_bundle(store: LocalFsStore) -> None:
    # #8: the packet is stored but the BUNDLE object is not -> a packet-only ref an
    # auditor could never recompute is a finalization error, not a silent dead link.
    packet = store.put(b"real-packet", ArtifactKind.SCORE_PACKET)
    packet_only = ScoredItem(
        uid=1,
        hotkey="hk1",
        challenge_id="c1",
        item_id="i1",
        bundle_digest=sha256_hex(b"never-stored-bundle"),
        packet_digest=packet.digest,
        committed_track="compression",
    )
    with pytest.raises(AuditFileMissingError, match="audit_bundle"):
        build_audit_manifest([packet_only], store=store)


def test_manifest_carries_merkle_root_and_verifiable_inclusion_proofs(
    store: LocalFsStore,
) -> None:
    """Every SCORE_PACKET ref opens against the committed root (the tamper-evidence gap)."""
    from vidaio.audit.commitments import merkle_root, verify_merkle_proof

    items = [_item(1, store), _item(2, store), _item(3, store)]
    manifest = build_audit_manifest(items, store=store)

    # The root is the merkle root over ALL score-packet digests (earning + baseline).
    all_packet_digests = [sha256_hex(f"packet-bytes-{u}".encode()) for u in (1, 2, 3)]
    assert manifest.score_packet_merkle_root == merkle_root(all_packet_digests)

    # Each item's SCORE_PACKET ref carries an inclusion proof that verifies.
    for uid in (1, 2, 3):
        packet_ref = next(
            r
            for r in manifest.refs_for(uid)
            if r.kind is AuditFileKind.SCORE_PACKET
        )
        assert packet_ref.inclusion_proof is not None
        assert verify_merkle_proof(
            packet_ref.digest,
            packet_ref.inclusion_proof,
            manifest.score_packet_merkle_root,
        )
        # the AUDIT_BUNDLE ref carries no proof (proven by recompute, not leaf inclusion)
        bundle_ref = next(
            r for r in manifest.refs_for(uid) if r.kind is AuditFileKind.AUDIT_BUNDLE
        )
        assert bundle_ref.inclusion_proof is None


def test_manifest_proofs_are_deterministic_regardless_of_item_order(
    store: LocalFsStore,
) -> None:
    """Proofs derive from the SORTED leaves, so item order cannot change the bytes."""
    a = build_audit_manifest([_item(1, store), _item(2, store), _item(3, store)])
    b = build_audit_manifest([_item(3, store), _item(1, store), _item(2, store)])
    assert a.score_packet_merkle_root == b.score_packet_merkle_root
    assert a._canonical_obj() == b._canonical_obj()


def test_manifest_carries_cumulative_fold_cursors_through_empty_epoch(
    store: LocalFsStore,
) -> None:
    """Schema-v14 integer and null tombstones survive idle epochs."""
    prior = {1: 7, 9: None}

    idle = build_audit_manifest(
        (), prior_fold_cursors=prior, current_census_uids=(1, 2)
    )
    advanced = build_audit_manifest(
        [_item(1, store, seq=8)],
        store=store,
        prior_fold_cursors=idle.fold_cursors,
        current_census_uids=(1, 2),
    )

    assert idle.fold_cursors == {1: 7, 2: None, 9: None}
    assert advanced.fold_cursors == {1: 8, 2: None, 9: None}


def test_manifest_allows_first_fold_after_explicit_null_cursor(
    store: LocalFsStore,
) -> None:
    manifest = build_audit_manifest(
        [_item(7, store, seq=0)],
        store=store,
        prior_fold_cursors={7: None},
        current_census_uids=(7,),
    )

    assert manifest.fold_cursors == {7: 0}


def test_manifest_refuses_cycle_at_or_below_cumulative_watermark(
    store: LocalFsStore,
) -> None:
    """The producer refuses a replay before it can publish self-consistent weights."""
    with pytest.raises(AuditFileMissingError, match="cross-epoch packet replay"):
        build_audit_manifest(
            [_item(1, store, seq=7)],
            store=store,
            prior_fold_cursors={1: 7},
        )


def test_manifest_baseline_rows_are_non_earning(store: LocalFsStore) -> None:
    baseline_packet = store.put(b"baseline-packet", ArtifactKind.SCORE_PACKET)
    baseline_bundle = store.put(b"baseline-bundle", ArtifactKind.AUDIT_BUNDLE)
    baseline = ScoredItem(
        uid=999,
        hotkey="",
        challenge_id="c1",
        item_id="baseline",
        bundle_digest=baseline_bundle.digest,
        packet_digest=baseline_packet.digest,
        committed_track="compression",
        source="competition",
        baseline=True,
    )
    manifest = build_audit_manifest([baseline], store=store)
    assert manifest.per_uid == {}  # baseline never maps to a weight
    assert len(manifest.baseline_bundles) == 2  # bundle + packet, still audited


# --------------------------------------------------------------------------------------
# finalize — writes a readable _FINALIZED set + returns the pointer.
# --------------------------------------------------------------------------------------


def test_finalize_writes_readable_set_and_returns_pointer(
    finalizer: EpochFinalizer, store: LocalFsStore
) -> None:
    miners = [_miner(1, _acc(0.8)), _miner(2, _acc(0.8), track="upscaling")]
    manifest = build_audit_manifest(
        [_item(1, store), _item(2, store)], store=store,
    )
    res = finalizer.finalize(
        # This finalizer case intentionally supplies no earning competition input.
        epoch_id=41822,
        close_block=15057191,
        snapshots=miners,
        burn_uid=0,
        audit_manifest=manifest,
        store=store,
        now=NOW,
    )
    prefix = epoch_prefix(41822)
    assert res.snapshot_key == set_member_key(prefix, "log.json")
    assert res.finalized is True
    assert res.already_finalized is False
    # the set is readable (marker present) and the bytes verify against the digest
    assert store.is_finalized(prefix)
    data = store.get_set_member(prefix, "log.json", expected_digest=res.log_digest)
    assert sha256_hex(data) == res.log_digest
    # the fetched bytes reconstruct the same log a validator would converge from
    log = EpochLog.from_json(data)
    assert log.log_digest() == res.log_digest
    assert log.weight_vector_digest == res.weight_vector_digest
    assert log.weight_u16 == res.log.weight_u16


def test_half_written_set_is_unreadable(finalizer: EpochFinalizer, store: LocalFsStore) -> None:
    """A member written WITHOUT the marker must be unreadable by a mirroring reader."""
    prefix = epoch_prefix(7)
    store.put_set_member(prefix, "log.json", b'{"partial": true}', ArtifactKind.EPOCH_LOG)
    # marker NOT written -> a mirror refuses to read the half-written set
    assert not store.is_finalized(prefix)
    with pytest.raises(SetNotFinalizedError):
        store.get_set_member(prefix, "log.json")


def test_finalize_is_idempotent(finalizer: EpochFinalizer, store: LocalFsStore) -> None:
    miners = [_miner(1, _acc(0.8)), _miner(2, _acc(0.8), track="upscaling")]
    manifest = build_audit_manifest(
        [_item(1, store), _item(2, store)], store=store,
    )
    kw = dict(
        # This finalizer case intentionally supplies no earning competition input.
        epoch_id=41822,
        close_block=15057191,
        snapshots=miners,
        burn_uid=0,
        audit_manifest=manifest,
        store=store,
        now=NOW,
    )
    first = finalizer.finalize(**kw)
    second = finalizer.finalize(**kw)
    assert second.already_finalized is True
    assert second.log_digest == first.log_digest
    assert second.snapshot_key == first.snapshot_key
    assert second.weight_vector_digest == first.weight_vector_digest


def test_finalize_refuses_nonzero_uid_without_earning_input(
    finalizer: EpochFinalizer, store: LocalFsStore
) -> None:
    """#1: every nonzero-weight uid MUST carry a complete earning input, over ALL nonzero
    uids — a missing one is refused (it can no longer become a silent auditor SKIP)."""
    from vidaio.epoch import EpochLogInvalid

    # An item with NO attested score -> build_audit_manifest emits no earning input for
    # its uid, but the uid still takes weight -> the finalizer refuses.
    item = _item(1, store, score=0.8)
    unscored = ScoredItem(
        uid=1, hotkey="hk1", challenge_id="c1", item_id="i1",
        bundle_digest=item.bundle_digest, packet_digest=item.packet_digest,
        committed_track="compression",  # no score / cycle_sequence -> no earning input
    )
    manifest = build_audit_manifest([unscored], store=store)
    assert manifest.earning_for(1) is None
    with pytest.raises(EpochLogInvalid, match="NO earning input"):
        finalizer.build_log(
            epoch_id=1, close_block=359, snapshots=(_miner(1, _acc(0.8)),),
            burn_uid=0, audit_manifest=manifest, now=NOW,
        )


def test_report_finalizer_carries_active_prior_earner_without_new_packet(
    finalizer: EpochFinalizer, store: LocalFsStore
) -> None:
    """an internal review: an IDLE prior earner (positive weight, NO new packet this epoch ⇒ no
    EarningInput) is CARRIED FORWARD, not refused.

    With ANOTHER miner's new evidence present, the pre-round-20 finalizer RAISED — the idle
    earner had no earning input and `_require_complete_earning` refused the whole log, stalling
    a normal carry epoch. It is now accepted as a verifiable pure carry-forward: its (uid,
    hotkey, accumulate_score) matches the prior epoch (`prior_earning`), exactly the chain the
    auditor re-derives via `_carry_forward_verdict`."""
    acc1 = _acc(0.8)  # uid 1's carried accumulator (unchanged — an idle earner does not re-fold)
    manifest = build_audit_manifest([_item(2, store, score=0.8)], store=store)
    assert manifest.earning_for(1) is None  # no new packet for the idle earner
    prior_earning = {1: ("hk1", acc1), 2: ("hk2", 0.0)}
    log = finalizer.build_log(
        epoch_id=100, close_block=359,
        snapshots=(_miner(1, acc1), _miner(2, _acc(0.8))),
        burn_uid=0, audit_manifest=manifest, now=NOW,
        prior_log_digest="a" * 64, prior_earning=prior_earning,
    )
    # The idle earner is present with nonzero weight and NO earning input — a pure carry-forward.
    assert log.weight_shares.get(1, 0.0) > 0.0
    assert log.audit_manifest.earning_for(1) is None
    # Fixed track pools do not renormalize: the canonical sink may coexist with
    # real earners for the absent upscaling allocation.
    assert log.burn_uid == 0
    assert log.weight_shares[0] > 0.0


def test_report_finalizer_all_carry_epoch_does_not_publish_disputed_burn(
    finalizer: EpochFinalizer, store: LocalFsStore
) -> None:
    """an internal review: an ALL-CARRY epoch (NOBODY has a new packet this epoch, but prior
    earners still hold positive accumulators) publishes a real CARRY-FORWARD log — miners carried
    at their accumulators — NOT an empty {burn_uid:1.0} vector.

    Before round-20 the producer discarded the snapshots on empty `items` and burned; with a
    valid predecessor, round-19's reset detector then DISPUTED that burn (every still-registered
    positive earner reset to nothing), so the honest idle path produced a known-disputed vector.
    Now the log carries the miners forward and stays auditable/CLEAN."""
    acc1, acc2 = _acc(0.8), _acc(0.6)
    prior_earning = {1: ("hk1", acc1), 2: ("hk2", acc2)}
    log = finalizer.build_log(
        epoch_id=100, close_block=359,
        snapshots=(_miner(1, acc1), _miner(2, acc2)),
        burn_uid=0, audit_manifest=AuditManifest(),  # NO new items this epoch
        now=NOW, prior_log_digest="b" * 64, prior_earning=prior_earning,
    )
    assert log.burn_uid == 0  # conditional withheld-pool sink, not an empty epoch
    assert log.weight_shares[0] > 0.0
    assert {m.uid for m in log.miners} == {1, 2}  # carried forward, not censored to an empty burn
    assert log.weight_shares.get(1, 0.0) > 0.0 and log.weight_shares.get(2, 0.0) > 0.0
    assert log.audit_manifest.earning_for(1) is None
    assert log.audit_manifest.earning_for(2) is None


def test_finalizer_refuses_dropped_predecessor_fold_cursor(
    finalizer: EpochFinalizer,
) -> None:
    """Live finalization requires the complete predecessor map, including deregistered uids."""
    from vidaio.epoch import EpochLogInvalid

    with pytest.raises(EpochLogInvalid, match="do not exactly carry"):
        finalizer.build_log(
            epoch_id=100,
            close_block=359,
            snapshots=(),
            burn_uid=0,
            audit_manifest=AuditManifest(),
            now=NOW,
            prior_log_digest="c" * 64,
            prior_fold_cursors={7: 12},
        )


def test_finalize_refuses_inconsistent_earning_fold(
    finalizer: EpochFinalizer, store: LocalFsStore
) -> None:
    """#1: a present earning input whose EWMA fold does not reproduce the snapshot's
    accumulate_score is refused before publication."""
    from vidaio.epoch import EpochLogInvalid

    manifest = build_audit_manifest([_item(1, store, score=0.8)], store=store)
    with pytest.raises(EpochLogInvalid, match="EWMA-folds"):
        finalizer.build_log(
            epoch_id=1, close_block=359, snapshots=(_miner(1, 0.5),),  # 0.5 != fold(0,[0.8])
            burn_uid=0, audit_manifest=manifest, now=NOW,
        )


def test_empty_epoch_finalizes_to_burn_vector(
    finalizer: EpochFinalizer, store: LocalFsStore
) -> None:
    res = finalizer.finalize(
        epoch_id=9,
        close_block=100,
        snapshots=[],
        burn_uid=7,
        audit_manifest=AuditManifest(),  # possibly-empty manifest
        store=store,
        now=NOW,
    )
    assert res.log.burn_uid == 7
    assert res.log.weight_u16 == {7: 65535}
    assert store.is_finalized(epoch_prefix(9))
    # still fully readable + verifiable
    data = store.get_set_member(epoch_prefix(9), "log.json", expected_digest=res.log_digest)
    assert EpochLog.from_json(data).weight_u16 == {7: 65535}


def test_finalize_determinism_two_finalizers(store: LocalFsStore, tmp_path: Path) -> None:
    """Two independent finalizers over the same state produce the same digest."""
    miners = [_miner(1, _acc(0.8)), _miner(2, _acc(0.8), track="upscaling")]
    manifest = build_audit_manifest(
        [_item(1, store), _item(2, store)], store=store,
    )
    kw = dict(
        epoch_id=1000,
        close_block=5000,
        snapshots=miners,
        burn_uid=0,
        audit_manifest=manifest,
        now=NOW,
    )
    store_b = LocalFsStore(tmp_path / "audit-b")
    a = EpochFinalizer(TokenomicsConfig(), scorer_version=SCORER).finalize(store=store, **kw)
    b = EpochFinalizer(TokenomicsConfig(), scorer_version=SCORER).finalize(store=store_b, **kw)
    assert a.log_digest == b.log_digest
    assert a.weight_vector_digest == b.weight_vector_digest
