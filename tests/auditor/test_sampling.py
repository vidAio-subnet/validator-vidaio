"""Deterministic, reproducible, non-cherry-pickable sampling."""

from __future__ import annotations

import pytest

from vidaio.epoch.log import AuditManifest
from vidaio.auditor import (
    DuplicateAuditIdentity,
    SamplePolicy,
    manifest_items,
    sample_items,
)
from vidaio.auditor.sampling import AuditItem, _rank, _seed

from tests.auditor.fakes import make_fake_bundle, refs_for
from vidaio.audit.store import LocalFsStore


def _manifest_with(store: LocalFsStore, n: int, source: str = "inference") -> AuditManifest:
    per_uid = {}
    for uid in range(1, n + 1):
        bundle = make_fake_bundle(
            store, challenge_id="c1", item_id=f"i{uid}", miner_hotkey=f"hk{uid}"
        )
        per_uid[uid] = refs_for(bundle, source=source)
    return AuditManifest(per_uid=per_uid)


def test_same_epoch_and_auditor_gives_the_same_sample(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    manifest = _manifest_with(store, 20)
    policy = SamplePolicy(sample_rate=0.25, min_samples=1, max_samples=50)

    a = sample_items(manifest, epoch_id=42, auditor_hotkey="hkA", policy=policy)
    b = sample_items(manifest, epoch_id=42, auditor_hotkey="hkA", policy=policy)

    def keys(items):
        return [item.key() for item in items]
    assert keys(a) == keys(b)  # reproducible, order and all
    assert len(a) == 5  # ceil(20 * 0.25)


def test_different_auditor_gives_a_different_sample(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    manifest = _manifest_with(store, 20)
    policy = SamplePolicy(sample_rate=0.25, min_samples=1, max_samples=50)

    a = set(it.key() for it in sample_items(manifest, epoch_id=42, auditor_hotkey="hkA", policy=policy))
    b = set(it.key() for it in sample_items(manifest, epoch_id=42, auditor_hotkey="hkB", policy=policy))

    assert a != b  # a different identity can't be steered to the same items


def test_different_epoch_gives_a_different_sample(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    manifest = _manifest_with(store, 20)
    policy = SamplePolicy(sample_rate=0.25)
    a = set(it.key() for it in sample_items(manifest, epoch_id=1, auditor_hotkey="hkA", policy=policy))
    b = set(it.key() for it in sample_items(manifest, epoch_id=2, auditor_hotkey="hkA", policy=policy))
    assert a != b


def test_beacon_changes_the_sample_but_stays_reproducible(tmp_path) -> None:
    """#10: the post-finalization beacon (on-chain anchor) steers the sample away from
    anything the authority could precompute at manifest-build time, yet a fixed beacon
    is fully reproducible."""
    store = LocalFsStore(tmp_path / "s")
    manifest = _manifest_with(store, 20)
    policy = SamplePolicy(sample_rate=0.25, min_samples=1, max_samples=50)

    no_beacon = set(
        it.key() for it in sample_items(manifest, epoch_id=42, auditor_hotkey="hkA", policy=policy)
    )
    anchored = set(
        it.key()
        for it in sample_items(
            manifest, epoch_id=42, auditor_hotkey="hkA", policy=policy, beacon="0xanchorTXID"
        )
    )
    other = set(
        it.key()
        for it in sample_items(
            manifest, epoch_id=42, auditor_hotkey="hkA", policy=policy, beacon="0xdifferentblockhash"
        )
    )
    assert anchored != no_beacon  # the anchor moves the draw (unpredictable pre-anchor)
    assert anchored != other  # a different beacon -> a different sample
    # reproducible given the SAME (public, post-hoc) beacon
    repeat = set(
        it.key()
        for it in sample_items(
            manifest, epoch_id=42, auditor_hotkey="hkA", policy=policy, beacon="0xanchorTXID"
        )
    )
    assert repeat == anchored


def test_min_and_max_clamps(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    manifest = _manifest_with(store, 20)

    # min floor: a tiny rate still audits min_samples
    lots = sample_items(
        manifest, epoch_id=1, auditor_hotkey="hkA",
        policy=SamplePolicy(sample_rate=0.0, min_samples=3, max_samples=50),
    )
    assert len(lots) == 3

    # max ceiling: a full rate is capped at max_samples
    capped = sample_items(
        manifest, epoch_id=1, auditor_hotkey="hkA",
        policy=SamplePolicy(sample_rate=1.0, min_samples=1, max_samples=7),
    )
    assert len(capped) == 7

    # never more than the population
    small = _manifest_with(LocalFsStore(tmp_path / "t"), 2)
    all_of_it = sample_items(
        small, epoch_id=1, auditor_hotkey="hkA",
        policy=SamplePolicy(sample_rate=1.0, min_samples=5, max_samples=50),
    )
    assert len(all_of_it) == 2


def test_explicit_all_items_mode_bypasses_the_fifty_item_sample_cap(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "all")
    manifest = _manifest_with(store, 51)
    selected = sample_items(
        manifest,
        epoch_id=1,
        auditor_hotkey="submit-gate",
        policy=SamplePolicy(
            sample_rate=1.0,
            min_samples=0,
            max_samples=1,
            all_items=True,
        ),
    )
    assert len(selected) == 51


def test_stratified_by_source(tmp_path) -> None:
    # 10 inference + 6 competition; each source sampled independently.
    store = LocalFsStore(tmp_path / "s")
    per_uid = {}
    for uid in range(1, 11):
        b = make_fake_bundle(store, challenge_id="c1", item_id=f"inf{uid}", miner_hotkey=f"hk{uid}")
        per_uid[uid] = refs_for(b, source="inference")
    for uid in range(11, 17):
        b = make_fake_bundle(store, challenge_id="c1", item_id=f"comp{uid}", miner_hotkey=f"hk{uid}")
        per_uid[uid] = refs_for(b, source="competition")
    manifest = AuditManifest(per_uid=per_uid)

    sampled = sample_items(
        manifest, epoch_id=7, auditor_hotkey="hkA",
        policy=SamplePolicy(sample_rate=0.5, min_samples=1, max_samples=50),
    )
    by_source = {}
    for it in sampled:
        by_source.setdefault(it.source, 0)
        by_source[it.source] += 1
    assert by_source["inference"] == 5  # ceil(10 * 0.5)
    assert by_source["competition"] == 3  # ceil(6 * 0.5)


#


def test_colliding_identity_items_do_not_share_a_rank(tmp_path) -> None:
    """#1(a): two DISTINCT items reusing the same (source, challenge_id, item_id) but with
    different uid + committed digests must NOT share a beacon-seeded rank (the old rank key
    excluded the uid and both digests, so colliding items ranked identically for everyone)."""
    store = LocalFsStore(tmp_path / "s")
    # Same challenge/item, different miners -> same identity tuple, different digests.
    b1 = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    b2 = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk2")
    bundle1, packet1 = refs_for(b1)
    bundle2, packet2 = refs_for(b2)
    it1 = AuditItem(source="inference", challenge_id="c1", item_id="i1", uid=1,
                    bundle_ref=bundle1, packet_ref=packet1)
    it2 = AuditItem(source="inference", challenge_id="c1", item_id="i1", uid=2,
                    bundle_ref=bundle2, packet_ref=packet2)

    assert it1.key() == it2.key()  # SAME identity tuple (what the old rank keyed on)
    assert it1.rank_key() != it2.rank_key()  # but the tie-free rank keys differ
    # ranks differ for the same seed AND for a different beacon/auditor seed
    for beacon in ("0xanchorTXID", "0xdifferent"):
        seed = _seed(beacon, 42, "hkA")
        assert _rank(seed, it1.rank_key()) != _rank(seed, it2.rank_key())


def test_duplicate_audit_identity_is_refused(tmp_path) -> None:
    """#1(b): two uids reusing the same (source, challenge_id, item_id) is a tamper signal —
    the manifest is REFUSED (DuplicateAuditIdentity), never sampled/audited as if honest."""
    store = LocalFsStore(tmp_path / "s")
    b1 = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    b2 = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk2")
    manifest = AuditManifest(per_uid={1: refs_for(b1), 2: refs_for(b2)})
    policy = SamplePolicy(sample_rate=1.0, min_samples=1, max_samples=50)

    with pytest.raises(DuplicateAuditIdentity):
        sample_items(manifest, epoch_id=42, auditor_hotkey="hkA", policy=policy)
    with pytest.raises(DuplicateAuditIdentity):
        manifest_items(manifest)


def test_distinct_identities_still_sample_deterministically_from_the_beacon(tmp_path) -> None:
    """#1: the honest distinct-identity case is unaffected — a fixed beacon draws the SAME
    sample every run, and a different beacon reshuffles it."""
    store = LocalFsStore(tmp_path / "s")
    manifest = _manifest_with(store, 20)  # distinct item_ids per uid
    policy = SamplePolicy(sample_rate=0.25, min_samples=1, max_samples=50)

    def keys(items):
        return [item.key() for item in items]
    a = keys(sample_items(manifest, epoch_id=42, auditor_hotkey="hkA", policy=policy, beacon="0xB"))
    b = keys(sample_items(manifest, epoch_id=42, auditor_hotkey="hkA", policy=policy, beacon="0xB"))
    other = set(
        it.key()
        for it in sample_items(manifest, epoch_id=42, auditor_hotkey="hkA", policy=policy, beacon="0xC")
    )
    assert a == b and len(a) == 5  # reproducible from the same beacon
    assert set(a) != other  # a different beacon reshuffles the draw


def test_manifest_items_pairs_bundle_and_packet(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    manifest = _manifest_with(store, 3)
    items = manifest_items(manifest)
    assert len(items) == 3
    for it in items:
        assert it.bundle_ref.digest != it.packet_ref.digest
        assert it.uid in {1, 2, 3}
