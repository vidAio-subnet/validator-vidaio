"""SharedSnapshotProvider — pointer -> mirror -> three-way verify -> inputs (wave 5).

Every test drives the REAL authority producer (`EpochFinalizer` + `LocalFsStore` +
`InMemoryChain` anchor + `EpochIndex`) so the pointer/bytes/anchor a validator
mirrors are exactly what the honest pipeline wrote — only the HTTP transport is a
fake `ScoringAuthorityClient`. No boto3, no bittensor, no ports, no sleeps.
"""

from __future__ import annotations

import json

import httpx
import pytest

from vidaio.audit import merkle_root
from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.store import finalized_marker_key, set_member_key
from vidaio.authority import EPOCH_LOG_MEMBER, build_audit_manifest, epoch_prefix
from vidaio.tokenomics import TokenomicsConfig
from vidaio.tokenomics.quantize import quantize_u16
from vidaio.tokenomics.weights import build_weight_vector
from vidaio.epoch import weight_vector_digest
from vidaio.weightsetter.config import WeightSetterConfig
from vidaio.weightsetter.shared_snapshot import (
    HttpScoringAuthorityClient,
    SharedSnapshotProvider,
    SnapshotDigestMismatch,
    SnapshotUnavailable,
    make_snapshot_provider,
)

from weightsetter_support import (
    NETUID,
    NOW,
    AuthorityHarness as Authority,
    FakeScoringAuthorityClient,
    make_item,
    make_miner,
)


# --------------------------------------------------------------------------------------
# Happy path: pointer -> mirror -> three-way verify -> the log's snapshots + inputs.
# --------------------------------------------------------------------------------------


async def test_provider_yields_log_snapshots_and_inputs(tmp_path) -> None:
    a = Authority(tmp_path)
    try:
        miners = [make_miner(1, 0.9), make_miner(2, 0.7, track="upscaling")]
        finalized = await a.finalize(
            epoch_id=100,
            close_block=36359,
            miners=miners,
            items=[make_item(1, a.store), make_item(2, a.store)],
        )
        provider = a.provider(epoch_id=100)

        snaps = list(provider.miner_snapshots())
        assert {m.uid for m in snaps} == {1, 2}
        assert snaps == list(finalized.log.miners)

        inputs = provider.epoch_inputs()
        assert inputs.epoch_id == 100
        assert inputs.close_block == 36359
        assert inputs.weight_u16 == finalized.log.weight_u16
        assert inputs.weight_shares == finalized.log.weight_shares
        assert inputs.miner_census_hotkeys == {1: "hk1", 2: "hk2"}
        assert inputs.burn_uid == 0
        assert inputs.composed_at == finalized.log.created_at
        assert inputs.competition_result == finalized.log.competition_result
        assert inputs.reward_window_state == finalized.log.reward_window_state
        assert not hasattr(inputs, "current_cycle")
        canonical = json.loads(finalized.log.to_json())
        assert "current_cycle" not in canonical
        assert "crown_state" not in canonical
        assert "reward_window_state" in canonical
        packet_digests = list(provider.score_packet_digests())
        assert packet_digests
        assert merkle_root(packet_digests) == (
            finalized.log.audit_manifest.score_packet_merkle_root
        )
        assert provider.resolved_snapshot_digest() == finalized.log_digest
    finally:
        a.close()


async def test_rederivation_matches_the_logs_stated_vector(tmp_path) -> None:
    """The validator RE-DERIVES the vector from the log's inputs and it equals the
    log's STATED u16 vector (the convergence cross-check that gates submission)."""
    a = Authority(tmp_path)
    try:
        miners = [
            make_miner(1, 0.9),
            make_miner(2, 0.3),
            make_miner(3, 0.5, "upscaling"),
        ]
        finalized = await a.finalize(
            epoch_id=5,
            close_block=2159,
            miners=miners,
            items=[make_item(u, a.store) for u in (1, 2, 3)],
        )
        provider = a.provider(epoch_id=5)
        snaps = list(provider.miner_snapshots())
        inputs = provider.epoch_inputs()

        # This fixture contains no earning competition input.
        rederived = build_weight_vector(
            TokenomicsConfig(), snaps, burn_uid=inputs.burn_uid
        )
        assert quantize_u16(rederived) == inputs.weight_u16 == finalized.log.weight_u16
    finally:
        a.close()


# --------------------------------------------------------------------------------------
# The convergence property, at provider level: TWO providers, SAME finalized epoch,
# byte-identical u16 through the real quantizer.
# --------------------------------------------------------------------------------------


async def test_two_providers_same_epoch_are_byte_identical(tmp_path) -> None:
    a = Authority(tmp_path)
    try:
        miners = [
            make_miner(1, 0.91),
            make_miner(2, 0.44),
            make_miner(3, 0.62, "upscaling"),
        ]
        finalized = await a.finalize(
            epoch_id=41822,
            close_block=15057191,
            miners=miners,
            items=[make_item(u, a.store) for u in (1, 2, 3)],
        )

        def one_validator_vector():
            provider = a.provider(epoch_id=41822)
            snaps = list(provider.miner_snapshots())
            inputs = provider.epoch_inputs()
            # This fixture contains no earning competition input.
            vec = build_weight_vector(
                TokenomicsConfig(), snaps, burn_uid=inputs.burn_uid
            )
            return quantize_u16(vec)

        u16_a = one_validator_vector()
        u16_b = one_validator_vector()
        assert u16_a == u16_b == finalized.log.weight_u16
        assert sum(u16_a.values()) == 65535
    finally:
        a.close()


# --------------------------------------------------------------------------------------
# Tamper-evidence chain: any broken leg REFUSES (SnapshotDigestMismatch), never yields.
# --------------------------------------------------------------------------------------


async def test_tampered_mirror_bytes_refused(tmp_path) -> None:
    """sha256(mirrored bytes) != pointer digest -> refuse (leg 1)."""
    a = Authority(tmp_path)
    try:
        await a.finalize(
            epoch_id=7,
            close_block=2879,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        # Overwrite the finalized member bytes underneath the pointer's digest.
        a.store._raw_put(
            set_member_key(epoch_prefix(7), EPOCH_LOG_MEMBER), b'{"tampered": true}'
        )
        provider = a.provider(epoch_id=7)
        with pytest.raises(SnapshotDigestMismatch):
            provider.miner_snapshots()
    finally:
        a.close()


async def test_packet_refs_must_reproduce_the_manifest_root(tmp_path) -> None:
    """Publication cannot trust a root that is not openable by the log's packet refs."""
    a = Authority(tmp_path)
    try:
        finalized = await a.finalize(
            epoch_id=8,
            close_block=3239,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        bad_manifest = finalized.log.audit_manifest.model_copy(
            update={"score_packet_merkle_root": "f" * 64}
        )
        bad_log = finalized.log.model_copy(update={"audit_manifest": bad_manifest})
        bad_bytes = bad_log.to_json()
        a.store._raw_put(set_member_key(epoch_prefix(8), EPOCH_LOG_MEMBER), bad_bytes)
        digest = sha256_hex(bad_bytes)
        pointer = a.pointer(8).model_copy(
            update={
                "snapshot_digest": digest,
                "anchor": a.pointer(8).anchor.model_copy(update={"digest": digest}),
            }
        )
        provider = SharedSnapshotProvider(
            client=FakeScoringAuthorityClient(latest=pointer),
            store=a.store,
            netuid=NETUID,
            verify_anchor=False,
        )

        assert provider.miner_snapshots()
        with pytest.raises(SnapshotDigestMismatch, match="merkle root"):
            provider.score_packet_digests()
    finally:
        a.close()


class _WrongAnchorReader:
    def __init__(self, digest: str) -> None:
        self._digest = digest

    def read_epoch_anchor(self, *, netuid: int, epoch_id: int):
        return self._digest


class _NoAnchorReader:
    def read_epoch_anchor(self, *, netuid: int, epoch_id: int):
        return None


async def test_non_canonical_anchored_bytes_refused(tmp_path) -> None:
    """an internal review: the shared-snapshot path REFUSES non-canonical anchored bytes —
    byte-for-byte the SAME requirement the auditor enforces (log.to_json() == data). Without
    it a validator could approve+submit from bytes the auditor would refuse, and the own-
    audit report digest would name the canonical reserialization, not the anchored bytes."""
    from vidaio.audit.canonical import sha256_hex
    from vidaio.epoch import EpochLog

    a = Authority(tmp_path)
    try:
        await a.finalize(
            epoch_id=7,
            close_block=2879,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        data = a.store.get_set_member(epoch_prefix(7), EPOCH_LOG_MEMBER)
        noncanon = (
            data + b" "
        )  # parses to the SAME log, but the bytes are not canonical
        assert EpochLog.from_json(noncanon).to_json() == data != noncanon
        a.store._raw_put(set_member_key(epoch_prefix(7), EPOCH_LOG_MEMBER), noncanon)

        # Re-point the pointer + its anchor field at sha256(noncanon) so the digest chain
        # PASSES (leg 1 sha256(bytes)==digest, leg 2 anchor==digest) and the ONLY remaining
        # objection is non-canonicality — the check this finding adds. verify_anchor=False
        # skips the independent leg (not under test here).
        nd = sha256_hex(noncanon)
        p = a.pointer(7)
        p2 = p.model_copy(
            update={
                "snapshot_digest": nd,
                "anchor": p.anchor.model_copy(update={"digest": nd}),
            }
        )
        provider = SharedSnapshotProvider(
            client=FakeScoringAuthorityClient(latest=p2),
            store=a.store,
            netuid=NETUID,
            anchor_reader=None,
            verify_anchor=False,
        )
        with pytest.raises(SnapshotDigestMismatch, match="canonical"):
            provider.miner_snapshots()
    finally:
        a.close()


async def test_narrow_submission_view_still_refuses_an_unsafe_u16_grid(
    tmp_path,
) -> None:
    """Report-only economics never weakens exact chain-input feasibility checks."""
    a = Authority(tmp_path)
    try:
        finalized = await a.finalize(
            epoch_id=8,
            close_block=3239,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        raw = json.loads(finalized.log.to_json())
        raw["weight_u16"][0][1] += 1
        raw["weight_vector_digest"] = weight_vector_digest(
            {int(uid): int(value) for uid, value in raw["weight_u16"]}
        )
        bad_bytes = canonical_json_bytes(raw)
        a.store._raw_put(set_member_key(epoch_prefix(8), EPOCH_LOG_MEMBER), bad_bytes)
        digest = sha256_hex(bad_bytes)
        old_pointer = a.pointer(8)
        pointer = old_pointer.model_copy(
            update={
                "snapshot_digest": digest,
                "weight_vector_digest": raw["weight_vector_digest"],
                "anchor": old_pointer.anchor.model_copy(update={"digest": digest}),
            }
        )
        provider = SharedSnapshotProvider(
            client=FakeScoringAuthorityClient(latest=pointer),
            store=a.store,
            netuid=NETUID,
            verify_anchor=False,
        )

        with pytest.raises(SnapshotDigestMismatch, match="sum .* is not 65535"):
            provider.miner_snapshots()
    finally:
        a.close()


async def test_on_chain_anchor_mismatch_refused(tmp_path) -> None:
    """on-chain anchored digest != pointer digest -> refuse (leg 3)."""
    a = Authority(tmp_path)
    try:
        await a.finalize(
            epoch_id=9,
            close_block=3599,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        provider = a.provider(epoch_id=9, anchor_reader=_WrongAnchorReader("ab" * 32))
        with pytest.raises(SnapshotDigestMismatch):
            provider.miner_snapshots()
    finally:
        a.close()


async def test_chain_holds_no_anchor_while_pointer_claims_one_refused(tmp_path) -> None:
    a = Authority(tmp_path)
    try:
        await a.finalize(
            epoch_id=11,
            close_block=4319,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        provider = a.provider(epoch_id=11, anchor_reader=_NoAnchorReader())
        with pytest.raises(SnapshotDigestMismatch):
            provider.miner_snapshots()
    finally:
        a.close()


async def test_real_anchor_reader_confirms_honest_epoch(tmp_path) -> None:
    """The real InMemoryChainAnchorReader reads back the exact anchored digest."""
    a = Authority(tmp_path)
    try:
        finalized = await a.finalize(
            epoch_id=13,
            close_block=5039,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        reader = a.anchor_reader()
        assert (
            reader.read_epoch_anchor(netuid=NETUID, epoch_id=13) == finalized.log_digest
        )
        # and the honest provider resolves cleanly through it
        assert list(a.provider(epoch_id=13).miner_snapshots())
    finally:
        a.close()


async def test_provider_verifies_anchor_at_pointer_inclusion_block(tmp_path) -> None:
    """A single-slot production reader must use archive state named by the pointer."""
    a = Authority(tmp_path)
    try:
        finalized = await a.finalize(
            epoch_id=14,
            close_block=5399,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        pointer = a.pointer(14)
        assert pointer.anchor.block is not None

        class HistoricalReader:
            def __init__(self):
                self.calls = []

            def read_epoch_anchor(self, *, netuid, epoch_id):
                raise AssertionError(
                    "head state must not verify a block-bearing pointer"
                )

            def read_epoch_anchor_at(self, *, netuid, epoch_id, block_number):
                self.calls.append((netuid, epoch_id, block_number))
                return finalized.log_digest

        reader = HistoricalReader()
        provider = a.provider(epoch_id=14, anchor_reader=reader)
        assert list(provider.miner_snapshots())
        assert reader.calls == [(NETUID, 14, pointer.anchor.block)]
    finally:
        a.close()


async def test_snapshot_key_not_matching_layout_refused(tmp_path) -> None:
    a = Authority(tmp_path)
    try:
        await a.finalize(
            epoch_id=15,
            close_block=5759,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        pointer = finalized_pointer_with_key(a.pointer(15), "adversary/chosen/key")
        provider = SharedSnapshotProvider(
            client=FakeScoringAuthorityClient(latest=pointer),
            store=a.store,
            netuid=NETUID,
            anchor_reader=a.anchor_reader(),
        )
        with pytest.raises(SnapshotDigestMismatch):
            provider.miner_snapshots()
    finally:
        a.close()


def finalized_pointer_with_key(pointer, key):
    return pointer.model_copy(update={"snapshot_key": key})


# --------------------------------------------------------------------------------------
# Unavailable (not tamper) -> HOLD (SnapshotUnavailable).
# --------------------------------------------------------------------------------------


async def test_no_finalized_pointer_is_unavailable(tmp_path) -> None:
    a = Authority(tmp_path)
    try:
        provider = SharedSnapshotProvider(
            client=FakeScoringAuthorityClient(latest=None),
            store=a.store,
            netuid=NETUID,
            anchor_reader=a.anchor_reader(),
        )
        with pytest.raises(SnapshotUnavailable):
            provider.miner_snapshots()
    finally:
        a.close()


async def test_missing_finalized_marker_is_unavailable(tmp_path) -> None:
    """A set whose _FINALIZED marker is gone must never be mirrored (half-write guard)."""
    a = Authority(tmp_path)
    try:
        await a.finalize(
            epoch_id=17,
            close_block=6479,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        (a.store._root / finalized_marker_key(epoch_prefix(17))).unlink()
        provider = a.provider(epoch_id=17)
        with pytest.raises(SnapshotUnavailable):
            provider.miner_snapshots()
    finally:
        a.close()


async def test_not_yet_anchored_pointer_holds(tmp_path) -> None:
    """finalized but txid is None (anchor pending): HOLD, don't refuse or submit."""
    a = Authority(tmp_path)
    try:
        # finalize + index but DO NOT anchor: the pointer carries txid=None.
        manifest = build_audit_manifest(
            [make_item(1, a.store)],
            store=a.store,
        )
        # This case intentionally finalizes an inference-only epoch.
        finalized = a.finalizer.finalize(
            epoch_id=19,
            close_block=7199,
            snapshots=[make_miner(1, 0.9)],
            burn_uid=0,
            audit_manifest=manifest,
            store=a.store,
            now=NOW,
        )
        a.index.record_finalized(finalized, finalized_at=NOW.isoformat())
        provider = a.provider(epoch_id=19)
        with pytest.raises(SnapshotUnavailable):
            provider.miner_snapshots()
    finally:
        a.close()


# --------------------------------------------------------------------------------------
# Empty-epoch burn convergence: the log carries {burn_uid: 65535}; the provider passes
# burn_uid through so the validator re-derives the identical burn vector (rule 11).
# --------------------------------------------------------------------------------------


async def test_empty_epoch_burn_inputs(tmp_path) -> None:
    a = Authority(tmp_path, burn_uid=7)
    try:
        finalized = await a.finalize(
            epoch_id=21, close_block=7919, miners=[], items=None
        )
        assert finalized.log.weight_u16 == {7: 65535}
        provider = a.provider(epoch_id=21)
        assert list(provider.miner_snapshots()) == []
        inputs = provider.epoch_inputs()
        assert inputs.burn_uid == 7
        # With neither inference nor competition earnings, burn_uid is the only input.
        rederived = build_weight_vector(
            TokenomicsConfig(), [], burn_uid=inputs.burn_uid
        )
        assert quantize_u16(rederived) == {7: 65535} == inputs.weight_u16
    finally:
        a.close()


# --------------------------------------------------------------------------------------
# The ScoringAuthorityClient HTTP seam (via httpx.MockTransport — bearer, parse, 404).
# --------------------------------------------------------------------------------------


async def test_http_client_sends_bearer_parses_pointer_and_maps_404(tmp_path) -> None:
    a = Authority(tmp_path)
    try:
        await a.finalize(
            epoch_id=42,
            close_block=15479,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        pointer_body = a.pointer(42).model_dump(mode="json")
        seen_auth: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_auth.append(request.headers.get("Authorization"))
            if request.url.path == "/epoch/latest":
                return httpx.Response(200, json=pointer_body)
            return httpx.Response(404, json={"detail": {"error": "epoch_not_found"}})

        transport = httpx.MockTransport(handler)
        client = HttpScoringAuthorityClient(
            "http://authority.test",
            token="tok-123",
            client=httpx.Client(transport=transport),
        )
        pointer = client.latest_pointer()
        assert pointer.epoch_id == 42
        assert seen_auth == ["Bearer tok-123"]
        with pytest.raises(SnapshotUnavailable):
            client.pointer_for(9999)
    finally:
        a.close()


async def test_http_client_signed_pointer_polls_pass_the_enforcing_guard(tmp_path) -> None:
    """P2 client half: with a signer wired, every pointer read carries Scheme A
    headers that the authority's guard accepts in `enforce` mode — including a
    base-url path prefix, which the server-side `request.url.path` would contain."""
    import hashlib

    from vidaio.services.hotkey_auth import (
        HotkeyAuthConfig,
        HotkeyAuthGuard,
        RegisteredHotkeyRegistry,
    )

    vali = "5" + "F" * 47

    class Signer:
        hotkey = vali

        def sign(self, payload: bytes) -> str:
            return hashlib.sha256(vali.encode() + payload).hexdigest()

    def verify(hotkey: str, payload: bytes, signature: str) -> bool:
        return signature == hashlib.sha256(hotkey.encode() + payload).hexdigest()

    class Neuron:
        hotkey = vali
        is_validator = True
        alpha_stake = 1000.0

    class Chain:
        def refresh(self) -> None: ...
        def neurons(self):
            return [Neuron()]

    guard = HotkeyAuthGuard(
        RegisteredHotkeyRegistry(Chain(), ttl_seconds=45),
        HotkeyAuthConfig(mode="enforce"),
        verify_fn=verify,
    )

    a = Authority(tmp_path)
    try:
        await a.finalize(
            epoch_id=42,
            close_block=15479,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        pointer_body = a.pointer(42).model_dump(mode="json")

        def handler(request: httpx.Request) -> httpx.Response:
            # The server verifies over ITS OWN view of the request: full URL path.
            guard.require(
                dict(request.headers),
                method=request.method,
                path=request.url.path,
                body=b"",
                require_validator_permit=True,
            )
            return httpx.Response(200, json=pointer_body)

        client = HttpScoringAuthorityClient(
            "http://authority.test/behind/prefix",
            token="tok-123",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            signer=Signer(),
        )
        assert client.latest_pointer().epoch_id == 42

        from vidaio.services.hotkey_auth import HotkeyAuthMissing

        unsigned = HttpScoringAuthorityClient(
            "http://authority.test",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(HotkeyAuthMissing):
            unsigned.latest_pointer()  # same guard refuses the headerless poll
    finally:
        a.close()


# --------------------------------------------------------------------------------------
# make_snapshot_provider: config selects local (back-compat) vs shared.
# --------------------------------------------------------------------------------------


def test_make_snapshot_provider_local_returns_local_unchanged() -> None:
    sentinel = object()
    cfg = WeightSetterConfig(provider="local")
    got = make_snapshot_provider(cfg, local_provider=sentinel)
    assert got is sentinel


async def test_make_snapshot_provider_shared_builds_shared(tmp_path) -> None:
    a = Authority(tmp_path)
    try:
        await a.finalize(
            epoch_id=23,
            close_block=8639,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        cfg = WeightSetterConfig(provider="shared", authority_netuid=NETUID)
        provider = make_snapshot_provider(
            cfg,
            store=a.store,
            local_provider=object(),
            authority_client=FakeScoringAuthorityClient(latest=a.pointer(23)),
            anchor_reader=a.anchor_reader(),
        )
        assert isinstance(provider, SharedSnapshotProvider)
        assert {m.uid for m in provider.miner_snapshots()} == {1}
    finally:
        a.close()
