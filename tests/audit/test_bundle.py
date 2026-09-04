import pytest
from pydantic import ValidationError

from audit_helpers import make_post_retirement_bundle
from vidaio.audit.bundle import LifecycleStage, build_bundle
from vidaio.audit.canonical import sha256_hex
from vidaio.audit.store import ArtifactKind, LocalFsStore


def _refs(store: LocalFsStore) -> dict:
    return {
        "challenge_input": store.put(b"in", ArtifactKind.CHALLENGE_INPUT),
        "miner_output": store.put(b"out", ArtifactKind.MINER_OUTPUT),
        "manifest": store.put(b"{}", ArtifactKind.MANIFEST),
        "score_packet": store.put(b'{"metrics":{}}', ArtifactKind.SCORE_PACKET),
    }


COMMON = {
    "challenge_id": "chal-1",
    "item_id": "item-1",
    "miner_hotkey": "hk-test",
    "commitment_hash": sha256_hex(b"dag"),
    "scorer_version": "scoring-1.0.0",
    "created_at": "2026-08-20T12:00:00+00:00",
}


def test_pre_reveal_bundle_lacks_holdout(store: LocalFsStore) -> None:
    bundle = build_bundle(stage=LifecycleStage.PRE_REVEAL, **_refs(store), **COMMON)
    assert bundle.reference_original is None
    assert bundle.dag_reveal is None


def test_pre_reveal_rejects_holdout_refs(store: LocalFsStore) -> None:
    original = store.put(b"orig", ArtifactKind.REFERENCE_ORIGINAL)
    with pytest.raises(ValidationError, match="pre-reveal"):
        build_bundle(
            stage=LifecycleStage.PRE_REVEAL,
            reference_original=original,
            **_refs(store),
            **COMMON,
        )


def test_post_retirement_requires_everything(store: LocalFsStore) -> None:
    with pytest.raises(ValidationError, match="reference_original, dag_reveal"):
        build_bundle(stage=LifecycleStage.POST_RETIREMENT, **_refs(store), **COMMON)
    with pytest.raises(ValidationError, match="dag_reveal"):
        build_bundle(
            stage=LifecycleStage.POST_RETIREMENT,
            reference_original=store.put(b"orig", ArtifactKind.REFERENCE_ORIGINAL),
            **_refs(store),
            **COMMON,
        )


def test_slot_kind_mismatch_rejected(store: LocalFsStore) -> None:
    refs = _refs(store)
    refs["miner_output"] = refs["challenge_input"]  # wrong kind in the slot
    with pytest.raises(ValidationError, match="miner_output"):
        build_bundle(stage=LifecycleStage.PRE_REVEAL, **refs, **COMMON)


def test_bundle_digest_stable(store: LocalFsStore) -> None:
    a = make_post_retirement_bundle(store)
    b = make_post_retirement_bundle(store)
    assert a.bundle_digest() == b.bundle_digest()
    assert len(a.bundle_digest()) == 64


def test_bundle_digest_ignores_dict_ordering(store: LocalFsStore) -> None:
    refs = _refs(store)
    a = build_bundle(
        stage=LifecycleStage.PRE_REVEAL,
        backend_versions={"vmaf": "3.0.0", "ffmpeg": "7.1"},
        **refs,
        **COMMON,
    )
    b = build_bundle(
        stage=LifecycleStage.PRE_REVEAL,
        backend_versions={"ffmpeg": "7.1", "vmaf": "3.0.0"},
        **refs,
        **COMMON,
    )
    assert a.bundle_digest() == b.bundle_digest()


def test_any_metadata_change_changes_digest(store: LocalFsStore) -> None:
    bundle = make_post_retirement_bundle(store)
    tampered_ts = bundle.model_copy(update={"created_at": "2026-08-21T00:00:00+00:00"})
    tampered_ver = bundle.model_copy(update={"scorer_version": "scoring-9.9.9"})
    tampered_item = bundle.model_copy(update={"item_id": "item-OTHER"})
    tampered_miner = bundle.model_copy(update={"miner_hotkey": "hk-OTHER"})
    assert tampered_ts.bundle_digest() != bundle.bundle_digest()
    assert tampered_ver.bundle_digest() != bundle.bundle_digest()
    assert tampered_item.bundle_digest() != bundle.bundle_digest()
    assert tampered_miner.bundle_digest() != bundle.bundle_digest()
