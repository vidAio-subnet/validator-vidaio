"""StoredBundleSource — the REAL BundleSource resolving bundles from the object store.

The authority persists each epoch's bundles content-addressed (persist_bundle), so the
store digest IS the bundle_digest the manifest names; StoredBundleSource resolves them
back verify-on-read, and honestly SKIPs (BundleUnavailable) on missing/corrupt objects.
"""

from __future__ import annotations

import pytest

from vidaio.audit.store import ArtifactKind, LocalFsStore
from vidaio.auditor import (
    BundleUnavailable,
    InMemoryBundleSource,
    StoredBundleSource,
    persist_bundle,
)

from tests.auditor.fakes import make_fake_bundle, refs_for


def test_persist_bundle_content_addresses_to_the_bundle_digest(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    bundle = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    ref = persist_bundle(store, bundle)
    assert ref.kind is ArtifactKind.AUDIT_BUNDLE
    assert ref.digest == bundle.bundle_digest()


def test_stored_source_resolves_a_persisted_bundle(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    bundle = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    persist_bundle(store, bundle)

    source = StoredBundleSource(store)
    bundle_ref, _packet_ref = refs_for(bundle)
    resolved = source.bundle_for(bundle_ref)
    assert resolved.bundle_digest() == bundle.bundle_digest()
    assert resolved == bundle


def test_stored_source_matches_the_inmemory_fake_for_a_persisted_bundle(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    bundle = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    persist_bundle(store, bundle)
    bundle_ref, _ = refs_for(bundle)

    fake = InMemoryBundleSource()
    fake.add(bundle)
    real = StoredBundleSource(store)
    assert real.bundle_for(bundle_ref) == fake.bundle_for(bundle_ref)


def test_unpersisted_bundle_is_unavailable_not_a_crash(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    bundle = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    # bundle was NEVER persisted -> the source cannot resolve it (SKIP, not FAIL).
    bundle_ref, _ = refs_for(bundle)
    with pytest.raises(BundleUnavailable, match="no stored bundle"):
        StoredBundleSource(store).bundle_for(bundle_ref)


def test_corrupt_stored_bundle_is_unavailable(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    bundle = make_fake_bundle(store, challenge_id="c1", item_id="i1", miner_hotkey="hk1")
    ref = persist_bundle(store, bundle)
    # Overwrite the stored object's bytes so sha256(bytes) != digest.
    path = store._path(ArtifactKind.AUDIT_BUNDLE, ref.digest)  # type: ignore[attr-defined]
    path.write_bytes(b"not a bundle")

    bundle_ref, _ = refs_for(bundle)
    with pytest.raises(BundleUnavailable):
        StoredBundleSource(store).bundle_for(bundle_ref)


def test_auditor_over_store_uses_the_stored_source(tmp_path) -> None:
    from vidaio.auditor import Auditor, AuditorConfig

    store = LocalFsStore(tmp_path / "s")
    auditor = Auditor.over_store(AuditorConfig(auditor_hotkey="hkAuditor"), store)
    assert isinstance(auditor._bundle_source, StoredBundleSource)  # type: ignore[attr-defined]
