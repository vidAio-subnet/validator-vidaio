"""End-to-end WeightSetter over the shared provider + convergence-health (wave 5).

Drives the FULL WeightSetter loop (compose -> set_weights -> publish) with the
SharedSnapshotProvider as its snapshot SOURCE, proving:

- two validators reading the SAME finalized epoch submit BYTE-IDENTICAL u16 vectors
  (the convergence property, end to end through the real quantizer);
- a tampered mirror still HOLDs, while the authenticated authority vector is submitted
  without running any local audit/re-derivation in the weight process;
- the empty-epoch 100%-burn vector converges;
- provider = "local" still works (report-mode / back-compat);
- the `vidaio_weightsetter_convergence` gauge reflects agreeing vs divergent peers.

No boto3, no bittensor, no ports, no sleeps.
"""

from __future__ import annotations

from vidaio.audit.store import ArtifactKind, set_member_key
from vidaio.audit.canonical import sha256_hex
from vidaio.authority import EPOCH_LOG_MEMBER, epoch_prefix
from vidaio.authority.anchoring import anchor_epoch
from vidaio.authority.finalizer import FinalizedEpoch
from vidaio.chain import ChainNeuron
from vidaio.core.db import connect
from vidaio.epoch import (
    AuditFileKind,
    AuditFileRef,
    AuditManifest,
    EpochLog,
    MinerCensusEntry,
    weight_vector_digest,
)
from vidaio.tokenomics import TokenomicsConfig
from vidaio.tokenomics.quantize import quantize_u16
from vidaio.tokenomics.weights import build_weight_vector
from vidaio.weightsetter import WeightSetter

from weightsetter_support import (
    NETUID,
    NOW,
    SCORER,
    AuthorityHarness,
    FakeSnapshots,
    PeerChain,
    make_item,
    make_miner,
)
from weightsetter_support import InMemoryChain


def owner_bound_chain(*, burn_uid: int = 0) -> InMemoryChain:
    """A report-chain fixture with the canonical subnet-owner sink bound.

    Tokenomics V2 intentionally burns an inactive reward-window allocation.  Even
    inference-only fixtures therefore carry an owner-sink share, and the production
    weight setter refuses to submit that share without a live uid/hotkey binding.
    """

    return InMemoryChain(
        _neurons=[
            ChainNeuron(
                burn_uid,
                f"owner-hk-{burn_uid}",
                "owner-coldkey",
                "0.0.0.0",
                0.0,
                0.0,
            )
        ]
    )


def build_setter(tmp_path, name, *, chain, snapshots, **overrides) -> WeightSetter:
    """A WeightSetter with its OWN sqlite db + InMemoryChain, publication off."""
    conn = connect(tmp_path / f"{name}.db")
    raw = {
        "core": {"metrics_port": 0},
        "weightsetter": {
            "metrics_port": 0,
            "chain_timeout_seconds": 0.5,
            "chain_retry_attempts": 3,
            "chain_retry_base_delay_seconds": 0.01,
            "publication_enabled": False,
            # This dependency-free report fixture has no owner-registry RPC.
            # Production resolves the same uid from the live chain adapter.
            "burn_uid": 0,
            **overrides,
        },
    }
    return WeightSetter(raw, chain=chain, snapshots=snapshots, conn=conn)


def rederive_u16(miners, inputs) -> dict[int, int]:
    # These legacy convergence fixtures provide no earning competition input.
    vec = build_weight_vector(TokenomicsConfig(), miners, burn_uid=inputs.burn_uid)
    return quantize_u16(vec)


# --------------------------------------------------------------------------------------
# Convergence, end to end: two validators, same epoch, byte-identical u16 on chain.
# --------------------------------------------------------------------------------------


async def test_two_validators_submit_byte_identical_u16(tmp_path) -> None:
    a = AuthorityHarness(tmp_path)
    try:
        miners = [make_miner(1, 0.9), make_miner(2, 0.55), make_miner(3, 0.7, "upscaling")]
        finalized = await a.finalize(
            epoch_id=100,
            close_block=36359,
            miners=miners,
            items=[make_item(u, a.store) for u in (1, 2, 3)],
        )

        chain_a, chain_b = owner_bound_chain(), owner_bound_chain()
        ws_a = build_setter(tmp_path, "va", chain=chain_a, snapshots=a.provider(epoch_id=100))
        ws_b = build_setter(tmp_path, "vb", chain=chain_b, snapshots=a.provider(epoch_id=100))

        assert await ws_a.attempt_once() is True
        assert await ws_b.attempt_once() is True

        # each validator submitted its OWN honest vector to its OWN chain ...
        _, vec_a = chain_a.weight_calls[-1]
        _, vec_b = chain_b.weight_calls[-1]
        # ... and they quantize BYTE-IDENTICALLY, and equal the log's stated vector.
        u16_a, u16_b = quantize_u16(vec_a), quantize_u16(vec_b)
        assert u16_a == u16_b == finalized.log.weight_u16
        assert sum(u16_a.values()) == 65535
    finally:
        a.close()


# --------------------------------------------------------------------------------------
# Tamper -> HOLD (no submission), and the tamper metric fires.
# --------------------------------------------------------------------------------------


async def test_tampered_mirror_holds_no_submission(tmp_path) -> None:
    a = AuthorityHarness(tmp_path)
    try:
        await a.finalize(
            epoch_id=7,
            close_block=2879,
            miners=[make_miner(1, 0.9)],
            items=[make_item(1, a.store)],
        )
        a.store._raw_put(
            set_member_key(epoch_prefix(7), EPOCH_LOG_MEMBER), b'{"tampered": true}'
        )
        chain = InMemoryChain()
        ws = build_setter(tmp_path, "vt", chain=chain, snapshots=a.provider(epoch_id=7))
        assert await ws.attempt_once() is False
        assert chain.weight_calls == []  # NOTHING submitted
        assert ws.metric_snapshot_tampered._value.get() == 1
    finally:
        a.close()


# --------------------------------------------------------------------------------------
# WEIGHT_DERIVATION_MISMATCH: report it, but submit the authenticated authority vector.
# --------------------------------------------------------------------------------------


async def test_weight_derivation_mismatch_reports_and_submits_authority_vector(
    tmp_path,
) -> None:
    a = AuthorityHarness(tmp_path)
    try:
        miners = (make_miner(1, 0.9), make_miner(2, 0.1))
        # A vector that is INTERNALLY consistent (u16 == quantize(shares)) but does NOT
        # follow from build_weight_vector(miners): the honest composition pays BOTH
        # eligible miners, so a 100%-to-uid-1 vector could never be its output.
        shares = {1: 1.0}
        u16 = quantize_u16(shares)

        def ref(uid: int) -> AuditFileRef:
            return AuditFileRef(
                kind=AuditFileKind.SCORE_PACKET,
                digest=sha256_hex(f"p{uid}".encode()),
                challenge_id="c",
                item_id=f"i{uid}",
                source="inference",
                committed_track="compression",  # REQUIRED on SCORE_PACKET refs (#9)
            )

        manifest = AuditManifest(
            per_uid={1: (ref(1),)}, fold_cursors={1: None, 2: None}
        )
        # This case intentionally builds an inference-only epoch.
        log = EpochLog(
            epoch_id=8,
            close_block=3239,
            scorer_version=SCORER,
            created_at=NOW,
            burn_uid=None,
            miners=miners,
            miner_census=tuple(MinerCensusEntry.from_miner(m) for m in miners),
            weight_shares=shares,
            weight_u16=u16,
            weight_vector_digest=weight_vector_digest(u16),
            audit_manifest=manifest,
        )
        # sanity: the honest re-derivation genuinely differs from the crafted vector.
        honest = quantize_u16(
            build_weight_vector(TokenomicsConfig(), list(miners), burn_uid=999)
        )
        assert honest != u16

        # store it as a real _FINALIZED set, index + anchor its digest.
        prefix = epoch_prefix(8)
        data = log.to_json()
        a.store.put_set_member(prefix, EPOCH_LOG_MEMBER, data, ArtifactKind.EPOCH_LOG)
        a.store.finalize_set(prefix)
        finalized = FinalizedEpoch(
            epoch_id=8,
            close_block=3239,
            snapshot_key=set_member_key(prefix, EPOCH_LOG_MEMBER),
            log_digest=sha256_hex(data),
            weight_vector_digest=log.weight_vector_digest,
            log=log,
        )
        a.index.record_finalized(finalized, finalized_at=NOW.isoformat())
        await anchor_epoch(finalized, chain=a.chain, index=a.index, netuid=NETUID, now=NOW)

        chain = InMemoryChain()
        # This deliberately malformed log omits the 20% upscaling pool. The thin
        # setter still submits its authenticated vector; the isolated audit workers
        # own every re-derivation/finding.
        ws = build_setter(
            tmp_path,
            "vd",
            chain=chain,
            snapshots=a.provider(epoch_id=8),
            burn_uid=999,
        )
        assert await ws.attempt_once() is True
        assert len(chain.weight_calls) == 1
        assert quantize_u16(chain.weight_calls[0][1]) == u16
        assert (
            ws.metric_chain_state_skips.labels(
                reason="weight_derivation_mismatch"
            )._value.get()
            == 0
        )
    finally:
        a.close()


# --------------------------------------------------------------------------------------
# Empty epoch -> the 100%-burn convergence vector is submitted (rule 11).
# --------------------------------------------------------------------------------------


async def test_empty_epoch_submits_burn_vector(tmp_path) -> None:
    from vidaio.chain import ChainNeuron

    a = AuthorityHarness(tmp_path, burn_uid=7)
    try:
        finalized = await a.finalize(epoch_id=9, close_block=3599, miners=[], items=None)
        assert finalized.log.weight_u16 == {7: 65535}
        chain = InMemoryChain(
            _neurons=[ChainNeuron(7, "owner-hk", "owner-ck", "0.0.0.0", 0.0, 0.0)]
        )
        # Chain state is authoritative even when a stale report fallback says uid 0.
        chain.get_burn_uid = lambda: 7  # type: ignore[attr-defined]
        ws = build_setter(
            tmp_path, "ve", chain=chain, snapshots=a.provider(epoch_id=9), burn_uid=0
        )
        assert await ws.attempt_once() is True
        _, vec = chain.weight_calls[-1]
        assert quantize_u16(vec) == {7: 65535}
    finally:
        a.close()


async def test_shared_empty_epoch_holds_when_live_burn_identity_is_unavailable(
    tmp_path,
) -> None:
    a = AuthorityHarness(tmp_path, burn_uid=7)
    try:
        await a.finalize(epoch_id=9, close_block=3599, miners=[], items=None)
        chain = InMemoryChain()

        def unavailable():
            raise RuntimeError("owner registry unavailable")

        chain.get_burn_uid = unavailable  # type: ignore[attr-defined]
        ws = build_setter(
            tmp_path, "burn-unavailable", chain=chain,
            snapshots=a.provider(epoch_id=9), burn_uid=0,
        )
        assert await ws.attempt_once() is False
        assert chain.weight_calls == []
        assert ws.metric_chain_state_skips.labels(
            reason="burn_identity_unavailable"
        )._value.get() == 1
    finally:
        a.close()


# --------------------------------------------------------------------------------------
# provider = "local" (report-mode / back-compat): the local path still submits.
# --------------------------------------------------------------------------------------


async def test_local_provider_still_submits(tmp_path) -> None:
    chain = owner_bound_chain()
    miners = [make_miner(1, 0.9), make_miner(2, 0.6, "upscaling")]
    ws = build_setter(tmp_path, "vl", chain=chain, snapshots=FakeSnapshots(miners))
    # FakeSnapshots has no epoch_inputs(): the report-mode weight-setter falls back to
    # local composition of this inference-only fixture.
    assert await ws.attempt_once() is True
    assert len(chain.weight_calls) == 1


# --------------------------------------------------------------------------------------
# Convergence gauge: agreeing vs divergent peers (observe-only, after we submit).
# --------------------------------------------------------------------------------------


async def test_convergence_gauge_reflects_peer_agreement(tmp_path) -> None:
    a = AuthorityHarness(tmp_path)
    try:
        miners = [make_miner(1, 0.9), make_miner(2, 0.6, "upscaling")]
        finalized = await a.finalize(
            epoch_id=200,
            close_block=72359,
            miners=miners,
            items=[make_item(u, a.store) for u in (1, 2)],
        )
        provider = a.provider(epoch_id=200)
        # our honest float vector (what we will submit)
        # This peer-comparison fixture has no earning competition input.
        our_vec = dict(finalized.log.weight_shares)
        inner = owner_bound_chain()
        chain = PeerChain(
            inner,
            peer_vectors={
                "peerA": dict(our_vec),  # agrees with us (same vector)
                "peerB": {1: 1.0},       # a DIFFERENT vector (different uid set)
            },
        )
        ws = build_setter(
            tmp_path,
            "vc",
            chain=chain,
            snapshots=provider,
            validator_hotkey="me",
            convergence_observe_enabled=True,
            convergence_peer_hotkeys=["peerA", "peerB"],
        )
        assert await ws.attempt_once() is True
        # one of two observed peers agrees -> 0.5
        assert ws.metric_convergence._value.get() == 0.5
    finally:
        a.close()


async def test_convergence_gauge_all_agree_is_one(tmp_path) -> None:
    a = AuthorityHarness(tmp_path)
    try:
        miners = [make_miner(1, 0.9), make_miner(2, 0.6, "upscaling")]
        finalized = await a.finalize(
            epoch_id=201,
            close_block=72719,
            miners=miners,
            items=[make_item(u, a.store) for u in (1, 2)],
        )
        provider = a.provider(epoch_id=201)
        # This peer-comparison fixture has no earning competition input.
        our_vec = dict(finalized.log.weight_shares)
        inner = owner_bound_chain()
        # peerA agrees; peerC hasn't submitted (None -> not observed); peerD unreadable.
        chain = PeerChain(
            inner,
            peer_vectors={"peerA": dict(our_vec)},
            no_record=["peerC"],
            unreadable=["peerD"],
        )
        ws = build_setter(
            tmp_path,
            "vc2",
            chain=chain,
            snapshots=provider,
            validator_hotkey="me",
            convergence_observe_enabled=True,
            convergence_peer_hotkeys=["peerA", "peerC", "peerD"],
        )
        assert await ws.attempt_once() is True
        # only peerA is observed (peerC has no record, peerD unreadable) and it agrees.
        assert ws.metric_convergence._value.get() == 1.0
    finally:
        a.close()
