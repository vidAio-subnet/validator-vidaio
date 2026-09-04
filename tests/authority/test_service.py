"""ScoringAuthority — the thin pointer API + the finalize->anchor epoch-close.

Every test drives the REAL `finalize_and_anchor` (finalizer + object store +
InMemoryChain anchor + epoch index) and reads pointers back over an ASGI client.
The central assertion running through all of them: the API returns KEYS + DIGESTS +
ANCHOR, never the epoch-log bytes — validators mirror the bytes from the object
store and verify the tamper-evidence chain themselves.
"""

from __future__ import annotations

import httpx
import pytest

from vidaio.audit.canonical import sha256_hex
from vidaio.authority import epoch_prefix
from vidaio.authority.finalizer import EPOCH_LOG_MEMBER
from vidaio.epoch import AuditManifest, EpochLog, EpochLogInvalid, MinerCensusEntry

from authority_support import NOW, Authority, make_item, make_miner

#: Response keys that would mean the API leaked snapshot BYTES (it must not).
#: Competition/crown economic state stays in the committed log and is not leaked here.
_SNAPSHOT_BODY_KEYS = {"miners", "weight_shares", "weight_u16", "audit_manifest", "log"}


# --------------------------------------------------------------------------------------
# finalize_and_anchor — writes a readable _FINALIZED set + anchors + indexes.
# --------------------------------------------------------------------------------------


async def test_finalize_and_anchor_writes_set_anchors_and_indexes(authority: Authority) -> None:
    miners = [make_miner(1, 0.9), make_miner(2, 0.8, track="upscaling")]
    items = [make_item(1, authority.store), make_item(2, authority.store)]
    finalized = await authority.seed_epoch(
        epoch_id=41822, close_block=15057191, miners=miners, items=items
    )

    # (1) a readable _FINALIZED set whose bytes verify against the pointer digest.
    prefix = epoch_prefix(41822)
    assert authority.store.is_finalized(prefix)
    data = authority.store.get_set_member(
        prefix, EPOCH_LOG_MEMBER, expected_digest=finalized.log_digest
    )
    assert sha256_hex(data) == finalized.log_digest

    # (2) the digest was anchored on chain (InMemoryChain recorded the payload).
    assert len(authority.chain.anchored) == 1
    anchored_payload = authority.chain.anchored[0].decode("ascii")
    assert anchored_payload.endswith(finalized.log_digest)  # the digest is IN the anchor

    # (3) it is indexed with the anchor txid.
    record = authority.index.get(41822)
    assert record is not None
    assert record.anchored and record.anchor_txid is not None
    assert record.log_digest == finalized.log_digest
    assert record.anchor_block == authority.chain.current_block()


# --------------------------------------------------------------------------------------
# GET /epoch/latest — the pointer (key + digests + anchor), NOT the bytes.
# --------------------------------------------------------------------------------------


async def test_latest_returns_pointer_not_bytes(
    authority: Authority, client: httpx.AsyncClient
) -> None:
    finalized = await authority.seed_epoch(
        epoch_id=100, close_block=36359, miners=[make_miner(1, 0.9)],
        items=[make_item(1, authority.store)],
    )
    resp = await client.get("/epoch/latest")
    assert resp.status_code == 200
    body = resp.json()

    # It is a POINTER: key + digests + anchor.
    assert body["epoch_id"] == 100
    assert body["close_block"] == 36359
    assert body["snapshot_key"] == f"{epoch_prefix(100)}/{EPOCH_LOG_MEMBER}"
    assert body["snapshot_digest"] == finalized.log_digest
    assert body["weight_vector_digest"] == finalized.weight_vector_digest
    assert body["anchor"]["digest"] == finalized.log_digest
    assert body["anchor"]["txid"] is not None

    # It carries NO snapshot payload — the bytes live only in the object store.
    assert _SNAPSHOT_BODY_KEYS.isdisjoint(body.keys())
    assert set(body.keys()) == {
        "epoch_id", "close_block", "snapshot_key", "snapshot_digest",
        "weight_vector_digest", "finalized", "anchor",
    }


async def test_latest_is_the_newest_finalized_epoch(
    authority: Authority, client: httpx.AsyncClient
) -> None:
    for eid in (10, 12, 11):  # out of order; latest() is by epoch_id
        await authority.seed_epoch(
            epoch_id=eid, close_block=eid * 360 + 359, miners=[make_miner(eid, 0.9)],
            items=[make_item(eid, authority.store)],
        )
    resp = await client.get("/epoch/latest")
    assert resp.json()["epoch_id"] == 12


async def test_latest_404_when_no_finalized_epoch(client: httpx.AsyncClient) -> None:
    resp = await client.get("/epoch/latest")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "epoch_not_found"


# --------------------------------------------------------------------------------------
# The validator-style tamper-evidence chain.
# --------------------------------------------------------------------------------------


async def test_validator_flow_pointer_then_mirror_then_verify(
    authority: Authority, client: httpx.AsyncClient
) -> None:
    """fetch pointer -> pull bytes from the store by key -> sha256(bytes) ==
    pointer digest == on-chain anchored digest (the whole tamper-evidence chain)."""
    await authority.seed_epoch(
        epoch_id=777, close_block=279719, miners=[make_miner(1, 0.9), make_miner(2, 0.7)],
        items=[make_item(1, authority.store), make_item(2, authority.store)],
    )

    # 1. pointer
    pointer = (await client.get("/epoch/777")).json()
    snapshot_digest = pointer["snapshot_digest"]

    # 2. MIRROR the bytes directly from the object store by the pointer's key.
    mirrored = authority.store.get_set_member(epoch_prefix(777), EPOCH_LOG_MEMBER)

    # 3. verify: sha256(bytes) == pointer digest == on-chain anchored digest.
    anchor = (await client.get("/epoch/777/anchor")).json()
    on_chain_digest = authority.chain.anchored[-1].decode("ascii").split(":")[-1]
    assert sha256_hex(mirrored) == snapshot_digest == anchor["digest"] == on_chain_digest

    # the mirrored bytes reconstruct the log a validator converges from.
    log = EpochLog.from_json(mirrored)
    assert log.log_digest() == snapshot_digest
    assert log.weight_vector_digest == pointer["weight_vector_digest"]


async def test_anchor_record_has_txid_digest_block(
    authority: Authority, client: httpx.AsyncClient
) -> None:
    finalized = await authority.seed_epoch(
        epoch_id=5, close_block=2159, miners=[make_miner(1, 0.9)],
        items=[make_item(1, authority.store)],
    )
    body = (await client.get("/epoch/5/anchor")).json()
    assert body == {
        "epoch_id": 5,
        "digest": finalized.log_digest,
        "txid": authority.index.get(5).anchor_txid,
        "block": authority.chain.current_block(),
    }


# --------------------------------------------------------------------------------------
# 404s — unknown / unfinalized epochs are never distinguished (no in-progress leak).
# --------------------------------------------------------------------------------------


async def test_unknown_epoch_404(
    authority: Authority, client: httpx.AsyncClient
) -> None:
    await authority.seed_epoch(
        epoch_id=1, close_block=719, miners=[make_miner(1, 0.9)],
        items=[make_item(1, authority.store)],
    )
    assert (await client.get("/epoch/9999")).status_code == 404
    assert (await client.get("/epoch/9999/anchor")).status_code == 404


async def test_unfinalized_epoch_is_not_served(
    authority: Authority, client: httpx.AsyncClient
) -> None:
    """An epoch whose log exists in the store but was NEVER indexed/finalized here
    (e.g. a half-write) 404s exactly like an unknown one — no in-progress leak."""
    # write a half-written set directly (member, but NO _FINALIZED marker, NOT indexed)
    from vidaio.audit.store import ArtifactKind

    authority.store.put_set_member(
        epoch_prefix(42), EPOCH_LOG_MEMBER, b'{"partial": true}', ArtifactKind.EPOCH_LOG
    )
    resp = await client.get("/epoch/42")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "epoch_not_found"


# --------------------------------------------------------------------------------------
# Auth gating (401 missing / 403 wrong) — and healthz stays open.
# --------------------------------------------------------------------------------------


async def test_auth_gating_401_403_and_ok(authority_authed: Authority) -> None:
    a = authority_authed
    await a.seed_epoch(
        epoch_id=3, close_block=1439, miners=[make_miner(1, 0.9)],
        items=[make_item(1, a.store)],
    )
    transport = httpx.ASGITransport(app=a.service.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://authority.test") as c:
        # missing bearer -> 401
        assert (await c.get("/epoch/latest")).status_code == 401
        assert (await c.get("/epoch/3")).status_code == 401
        assert (await c.get("/epoch/3/anchor")).status_code == 401
        # wrong token -> 403
        wrong = {"Authorization": "Bearer not-the-token"}
        assert (await c.get("/epoch/latest", headers=wrong)).status_code == 403
        # right token -> 200
        ok = {"Authorization": "Bearer s3cr3t-validator-token"}
        assert (await c.get("/epoch/latest", headers=ok)).status_code == 200
        assert (await c.get("/epoch/3", headers=ok)).status_code == 200
        # healthz is always open
        assert (await c.get("/healthz")).status_code == 200


# --------------------------------------------------------------------------------------
# Idempotent re-finalize + empty-epoch burn.
# --------------------------------------------------------------------------------------


async def test_finalize_and_anchor_is_idempotent(
    authority: Authority, client: httpx.AsyncClient
) -> None:
    kw = dict(
        epoch_id=8, close_block=3239, miners=[make_miner(1, 0.9)],
        items=[make_item(1, authority.store)],
    )
    first = await authority.seed_epoch(**kw)
    # a fresh item set would re-store the same packet bytes (write-once) -> same manifest
    second = await authority.seed_epoch(
        epoch_id=8, close_block=3239, miners=[make_miner(1, 0.9)],
        items=[make_item(1, authority.store)],
    )
    assert second.already_finalized is True
    assert second.log_digest == first.log_digest
    # anchored exactly ONCE (no second on-chain write), one index row.
    assert len(authority.chain.anchored) == 1
    resp = await client.get("/epoch/8")
    assert resp.json()["snapshot_digest"] == first.log_digest


async def test_successive_finalize_and_anchor_chains_a_spine_the_gate_accepts(
    authority: Authority,
) -> None:
    """an internal review: the public finalize_and_anchor now threads prior_log_digest — derived
    from the authority's OWN durable index (the newest finalized epoch before this one) — so two
    successive epochs chain a spine: epoch N+1's prior_log_digest == epoch N's log_digest, while
    the FIRST epoch stays genesis (None). Before the fix EVERY epoch produced this way was genesis
    (None), which the own-audit genesis gate correctly DISPUTES for a non-genesis epoch."""
    from vidaio.auditor import Auditor, AuditorConfig, SamplePolicy
    from vidaio.tokenomics import TokenomicsConfig

    first = await authority.seed_epoch(
        epoch_id=10, close_block=10 * 360 + 359, miners=[make_miner(1)],
        items=[make_item(1, authority.store)],
    )
    second = await authority.seed_epoch(
        epoch_id=11, close_block=11 * 360 + 359, miners=[make_miner(2)],
        items=[make_item(2, authority.store)],
    )
    log_first = EpochLog.from_json(
        authority.store.get_set_member(epoch_prefix(10), EPOCH_LOG_MEMBER)
    )
    log_second = EpochLog.from_json(
        authority.store.get_set_member(epoch_prefix(11), EPOCH_LOG_MEMBER)
    )
    # (1) the chained spine: genesis stays None; the successor commits the predecessor's digest.
    assert log_first.prior_log_digest is None
    assert log_second.prior_log_digest == first.log_digest == log_first.log_digest()
    assert second.log_digest == log_second.log_digest()

    # (2) the own-audit genesis gate ACCEPTS the chained non-genesis epoch: audited WITH the prior
    # log, no broken-chain "OMITS prior_log_digest" DISPUTE (round-9 #3) is raised.
    auditor = Auditor.over_store(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=TokenomicsConfig()),
        authority.store, chain=authority.chain,
    )
    policy = SamplePolicy(sample_rate=0.0, min_samples=0)
    report = auditor.audit_epoch(
        log_second, authority.store, policy, None, NOW,
        prior_log=log_first, is_genesis=False,
    )
    assert not any(
        "OMITS prior_log_digest" in (v.detail or "") for v in report.earning_verdicts
    )

    # NOTE: the negative contrast (a forced-genesis non-genesis epoch DISPUTED as a broken
    # chain) is not asserted here — the shared authority_support.make_item stores opaque
    # placeholder packet bytes (fine for finalize/anchor, but the auditor's fold verifier
    # SKIPs at the packet-read gate before it can reach the "OMITS prior_log_digest" genesis
    # gate). That gate's firing is covered in tests/auditor/ with real, readable packets.


async def test_service_carries_omitted_uid_watermark_and_refuses_later_replay(
    authority: Authority,
) -> None:
    """The advertised production close path must not bypass schema-v11 continuity.

    Callers historically build a current-only manifest before ``finalize_and_anchor`` resolves
    its predecessor.  The service fills the predecessor tombstone during an omitted epoch, then
    refuses the same uid's old ordering key when it returns.
    """
    first = await authority.seed_epoch(
        epoch_id=20,
        close_block=20 * 360 + 359,
        miners=[make_miner(1)],
        items=[make_item(1, authority.store)],
    )
    omitted = await authority.seed_epoch(
        epoch_id=21,
        close_block=21 * 360 + 359,
        miners=[make_miner(2)],
        items=[make_item(2, authority.store)],
    )

    assert first.log.audit_manifest.fold_cursors == {1: 0}
    assert omitted.log.audit_manifest.fold_cursors == {1: 0, 2: 0}

    with pytest.raises(EpochLogInvalid, match="cross-epoch packet replay"):
        await authority.seed_epoch(
            epoch_id=22,
            close_block=22 * 360 + 359,
            miners=[make_miner(1)],
            items=[make_item(1, authority.store)],  # uid 1 reuses committed key 0
        )


async def test_service_persists_full_registered_census_separately_from_economic_miners(
    authority: Authority,
) -> None:
    """The production service path must not collapse the close-block census to snapshots."""
    registered = MinerCensusEntry(
        uid=7, hotkey="hk7", coldkey="ck7", ip="10.0.0.7"
    )

    finalized = await authority.service.finalize_and_anchor(
        epoch_id=30,
        close_block=30 * 360 + 359,
        snapshots=(),
        miner_census=(registered,),
        audit_manifest=AuditManifest(),
    )

    assert finalized.log.miners == ()
    assert finalized.log.miner_census == (registered,)
    persisted = EpochLog.from_json(
        authority.store.get_set_member(epoch_prefix(30), EPOCH_LOG_MEMBER)
    )
    assert persisted.miner_census == (registered,)


async def test_empty_epoch_finalizes_to_burn_pointer(tmp_path) -> None:
    a = Authority(tmp_path, burn_uid=7)
    try:
        finalized = await a.seed_epoch(epoch_id=9, close_block=3599, miners=[], items=None)
        assert finalized.log.weight_u16 == {7: 65535}
        transport = httpx.ASGITransport(app=a.service.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://a.test") as c:
            body = (await c.get("/epoch/latest")).json()
        assert body["snapshot_digest"] == finalized.log_digest
        # the burn pointer still points at real, mirrorable, anchored bytes.
        data = a.store.get_set_member(epoch_prefix(9), EPOCH_LOG_MEMBER)
        assert EpochLog.from_json(data).weight_u16 == {7: 65535}
        assert a.chain.anchored[-1].decode("ascii").endswith(finalized.log_digest)
    finally:
        a.close()


async def test_empty_epoch_burn_uid_comes_from_chain_not_config(tmp_path) -> None:
    a = Authority(tmp_path, burn_uid=0)
    try:
        # uid 0 is an ordinary configured fallback; live chain state names uid 7.
        a.chain.get_burn_uid = lambda: 7  # type: ignore[attr-defined]
        finalized = await a.seed_epoch(
            epoch_id=9, close_block=3599, miners=[], items=None
        )
        assert finalized.log.burn_uid == 7
        assert finalized.log.weight_u16 == {7: 65535}
    finally:
        a.close()
