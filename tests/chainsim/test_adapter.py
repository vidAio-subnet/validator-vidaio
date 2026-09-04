"""HttpChainAdapter: Protocol conformance, factory semantics, and the real
WeightSetter service driven end-to-end against a chainsim ASGI app."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from chainsim_support import Clock, FakeSnapshots, mk_miner

from vidaio.audit import COMMITMENT_DOMAIN, CommitmentLedger, CommitmentStatus, LocalFsStore
from vidaio.chain import (
    ChainAdapter,
    ChainStateUnavailable,
    EmbeddedReportingChain,
    HttpChainAdapter,
    make_chain_adapter,
)
from vidaio.chainsim.report import build_report
from vidaio.chainsim.service import RegisterRequest, WeightsRequest
from vidaio.core.db import connect
from vidaio.tokenomics import build_weight_vector
from vidaio.weightsetter import WeightSetter
from vidaio.weightsetter import migrate as ws_migrate


def test_http_adapter_satisfies_the_chain_adapter_protocol(make_adapter):
    assert isinstance(make_adapter(), ChainAdapter)


async def test_refresh_pulls_neurons_and_block_into_the_cache(sim, make_adapter):
    adapter = make_adapter("val")
    assert adapter.current_block() == 0
    with pytest.raises(ChainStateUnavailable):
        adapter.neurons()  # never refreshed != "the subnet is empty"
    assert adapter.register(alpha_stake=1000.0) == 0
    assert adapter.auth_token  # registration claimed the hotkey and kept its token
    sim.register(RegisterRequest(hotkey="hk1", coldkey="ck1", ip="10.0.0.1", role="miner"))
    sim.advance(3)

    adapter.refresh()

    assert adapter.current_block() == 4
    neurons = adapter.neurons()
    assert [(n.uid, n.hotkey, n.is_validator) for n in neurons] == [
        (0, "val", True),
        (1, "hk1", False),
    ]
    assert neurons[0].alpha_stake == 1000.0


def test_burn_uid_resolves_anchor_authority_from_refreshed_registry(sim, make_adapter):
    # An ordinary miner deliberately takes uid 0 first.  Burn identity follows the
    # configured authority hotkey, not registration order or a hardcoded zero.
    sim.register(RegisterRequest(hotkey="miner-first", coldkey="ck0", ip="10.0.0.1"))
    adapter = make_adapter("authority")
    assert adapter.register() == 1
    adapter.refresh()
    assert adapter.get_burn_uid() == 1


def test_burn_uid_fails_closed_without_a_fresh_registered_authority(make_adapter):
    adapter = make_adapter("authority")
    with pytest.raises(ChainStateUnavailable, match="no chain snapshot"):
        adapter.get_burn_uid()

    adapter.refresh()
    with pytest.raises(ChainStateUnavailable, match="anchor authority"):
        adapter.get_burn_uid()


async def test_refresh_failure_keeps_the_cached_snapshot(sim, make_adapter):
    adapter = make_adapter("val")
    adapter.register()
    adapter.refresh()
    cached = adapter.neurons()
    assert cached  # populated

    class Boom(httpx.BaseTransport):
        def handle_request(self, request):
            raise httpx.ConnectError("sim down")

    adapter._client = httpx.Client(transport=Boom())
    adapter.refresh()  # must not raise — reads stay snapshots
    assert adapter.neurons() == cached


async def test_set_weights_and_anchor_roundtrip(sim, make_adapter):
    adapter = make_adapter("val")
    adapter.register()

    result = await adapter.set_weights({1: 0.75, 2: 0.25}, version_key=7)
    assert result.success and result.block == 1

    gated = await adapter.set_weights({1: 1.0}, version_key=7)
    assert not gated.success and "tempo" in gated.message

    payload = b"vidaio.commitment.v1:publication:" + b"a" * 64
    txid = await adapter.anchor_commitment(payload)
    assert txid == "0x" + hashlib.sha256(payload).hexdigest()[:16]  # InMemoryChain parity

    with pytest.raises(ValueError):
        await adapter.anchor_commitment(b"x" * 129)

    state = sim.state()
    assert state["weight_calls"][0]["vector"] == {"1": 0.75, "2": 0.25}
    assert state["weight_calls"][0]["version_key"] == 7
    assert state["anchors"][0]["txid"] == txid
    assert state["anchors"][0]["hotkey"] == "val"


async def test_set_weights_reports_the_exact_submitted_u16(sim, make_adapter):
    """Round-4 #3: HttpChainAdapter reports the EXACT u16 that lands on the chain's
    grid as `SetWeightsResult.submitted`, so the weight-setter publishes/anchors
    chain state rather than the pre-quantization float intent — even scale-equivalent."""
    from vidaio.tokenomics.quantize import max_normalize_u16, quantize_u16

    adapter = make_adapter("val")
    adapter.register()

    intent = {1: 0.4, 2: 0.6}
    result = await adapter.set_weights(intent, version_key=7)
    assert result.success
    assert result.submitted == max_normalize_u16(quantize_u16(intent))
    assert result.submitted != intent  # the u16 grid, NOT the float intent
    assert max(result.submitted.values()) == 65535

    # a rejected (tempo-gated) submit reports NO submitted vector — nothing landed
    gated = await adapter.set_weights({1: 1.0}, version_key=7)
    assert not gated.success
    assert gated.submitted == {}


async def test_submitted_weights_round_trips_through_the_sim(sim, make_adapter):
    """The an internal review read: prove OUR vector is what the chain holds.

    Before any write the sim answers a POSITIVE "no weights for this hotkey"
    (None, not an error and not an empty vector); afterwards it hands back the
    exact vector and the block it was recorded at.
    """
    adapter = make_adapter("val")
    adapter.register()

    assert adapter.submitted_weights("val") is None  # a positive "none on chain"

    result = await adapter.set_weights({1: 0.75, 2: 0.25}, version_key=7)
    assert result.success

    reported = adapter.submitted_weights("val")
    assert reported is not None
    assert reported.weights == {1: 0.75, 2: 0.25}
    assert reported.block == result.block
    # a hotkey that never set weights is answered, not guessed at
    assert adapter.submitted_weights("somebody-else") is None
    # ... and the sim's own view agrees (the endpoint is a thin read over it)
    assert sim.latest_weights("val")["vector"] == {"1": 0.75, "2": 0.25}


async def test_the_submitted_weights_read_is_open_like_the_rest_of_chain_state(
    client, sim
):
    """Chain state is public: reading a hotkey's vector needs no credential."""
    sim.register(RegisterRequest(hotkey="val", coldkey="ck", ip="10.0.0.1"))
    sim.submit_weights(WeightsRequest(hotkey="val", vector={1: 1.0}, version_key=0))

    response = await client.get("/weights/val")  # NO Authorization header

    assert response.status_code == 200
    body = response.json()
    assert body["hotkey"] == "val"
    assert body["vector"] == {"1": 1.0}
    assert body["block"] == 1

    missing = await client.get("/weights/nobody")
    assert missing.status_code == 200
    assert missing.json()["vector"] is None  # explicit "none", never a 404 guess


async def test_block_hash_agrees_with_inmemory_and_is_none_for_a_future_block(sim, make_adapter):
    """The round-6 beacon seam: HttpChainAdapter reads block_hash(n)
    from the sim's /block_hash endpoint — the SAME `synthetic_block_hash` derivation as
    InMemoryChain (so report-mode and in-memory agree byte-for-byte) — and None for a
    block the sim has NOT produced yet (the beacon block is then not finalized)."""
    from vidaio.chain import InMemoryChain
    from vidaio.chain.adapter import synthetic_block_hash

    adapter = make_adapter("val")
    adapter.register()
    sim.advance(9)  # produce some blocks
    current = sim.current_block()

    # A PRODUCED block: the sim's hash == synthetic_block_hash == InMemoryChain's.
    produced = current - 2
    h = adapter.block_hash(produced)
    assert h == synthetic_block_hash(produced)
    assert len(h) == 64 and int(h, 16) >= 0

    mem = InMemoryChain()
    mem.advance_blocks(current - 1)  # same head as the sim
    assert mem.block_hash(produced) == h  # in-memory and report-mode AGREE

    # A NOT-YET-PRODUCED block: a POSITIVE None on both, never a substituted hash.
    assert adapter.block_hash(current + 5) is None
    assert mem.block_hash(current + 5) is None


async def test_read_anchor_block_reads_the_inclusion_block_from_state(sim, make_adapter):
    """The anchor-inclusion-block seam: HttpChainAdapter reads
    the block the epoch's anchor landed at from /state — a positive None before any
    anchor, the real block after, and the SAME value InMemoryChain records."""
    from vidaio.authority.anchoring import ANCHOR_DOMAIN, anchor_payload
    from vidaio.chain import InMemoryChain

    netuid, epoch, digest = 85, 42, "d" * 64
    adapter = make_adapter("val")
    adapter.register()

    # No anchor yet -> a POSITIVE "none", never a substituted value.
    assert adapter.read_anchor_block(netuid=netuid, epoch_id=epoch, domain=ANCHOR_DOMAIN) is None

    sim.advance(4)  # the anchor lands at a block > 1
    payload = anchor_payload(epoch, netuid, digest)
    await adapter.anchor_commitment(payload)
    block = int(sim.state()["anchors"][-1]["block"])

    assert adapter.read_anchor_block(netuid=netuid, epoch_id=epoch, domain=ANCHOR_DOMAIN) == block
    # read_anchor still returns the anchored digest (the finding-#4 tamper leg).
    assert adapter.read_anchor(netuid=netuid, epoch_id=epoch, domain=ANCHOR_DOMAIN) == digest
    raw = adapter.read_commitment_record(netuid=netuid)
    assert raw is not None and (raw.payload, raw.block) == (payload, block)
    archived = adapter.read_commitment_record(netuid=netuid, block_number=block)
    assert archived == raw

    # An InMemoryChain that anchored the SAME payload at the SAME block agrees exactly.
    mem = InMemoryChain()
    mem.advance_blocks(block - 1)
    await mem.anchor_commitment(payload)
    assert mem.current_block() == block
    assert mem.read_anchor_block(netuid=netuid, epoch_id=epoch, domain=ANCHOR_DOMAIN) == block
    assert mem.read_commitment_record(netuid=netuid, block_number=block) == raw


async def test_read_anchor_block_and_block_hash_raise_on_an_unreadable_state():
    """A transport failure must HOLD (raise), never be read as 'no anchor'/'no block'."""
    class Down(httpx.BaseTransport):
        def handle_request(self, request):
            raise httpx.ConnectError("nobody home")

    adapter = HttpChainAdapter(
        "http://sim", validator_hotkey="val", client=httpx.Client(transport=Down())
    )
    with pytest.raises(ChainStateUnavailable):
        adapter.read_anchor_block(netuid=85, epoch_id=42, domain="vidaio.epoch.anchor.v1")
    with pytest.raises(ChainStateUnavailable):
        adapter.block_hash(5)
    adapter.close()


async def test_an_unreadable_weights_read_raises_instead_of_denying():
    """A failed read must NEVER look like "the chain holds no weights".

    None denies a weight intent, and a denied intent is eventually abandoned
    unpublished — so transport failure has to raise.
    """

    class Down(httpx.BaseTransport):
        def handle_request(self, request):
            raise httpx.ConnectError("nobody home")

    adapter = HttpChainAdapter(
        "http://sim", validator_hotkey="val", client=httpx.Client(transport=Down())
    )
    with pytest.raises(ChainStateUnavailable):
        adapter.submitted_weights("val")
    adapter.close()


async def test_transport_errors_surface_as_oserror():
    class Down(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("nobody home")

    adapter = HttpChainAdapter(
        "http://sim",
        validator_hotkey="val",
        async_client=httpx.AsyncClient(transport=Down()),
    )
    with pytest.raises(OSError):
        await adapter.set_weights({0: 1.0}, version_key=0)
    with pytest.raises(OSError):
        await adapter.anchor_commitment(b"x")
    adapter.close()


#


async def test_read_anchor_honors_only_the_scoring_authority(sim):
    """A COMPETING anchor written by a non-authority hotkey is IGNORED by read_anchor /
    read_anchor_block — only the Scoring Authority's anchor is honored.

    The sim's `/anchor` accepts ANY registered hotkey, and the reader used to select the LAST
    matching-epoch payload regardless of who wrote it — so any participant could REPLACE the
    effective anchor. Binding the READ path to the authority account closes that (matching the
    production adapter, which reads only the authority's commitment)."""
    from chainsim_support import SyncASGITransport

    from vidaio.authority.anchoring import ANCHOR_DOMAIN, anchor_payload

    netuid, epoch = 85, 55
    auth_digest, evil_digest = "a" * 64, "e" * 64

    def mk(hotkey: str, anchor_hotkey: str | None = None) -> HttpChainAdapter:
        return HttpChainAdapter(
            "http://sim",
            validator_hotkey=hotkey,
            anchor_hotkey=anchor_hotkey,
            client=httpx.Client(transport=SyncASGITransport(sim.app)),
            async_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=sim.app)),
        )

    authority = mk("authority-hk")
    adversary = mk("adversary-hk")
    authority.register()  # each claims its own hotkey + token so it may anchor
    adversary.register()

    # The AUTHORITY anchors the real digest first (at some block).
    await authority.anchor_commitment(anchor_payload(epoch, netuid, auth_digest))
    auth_block = int(sim.state()["anchors"][-1]["block"])
    sim.advance(3)
    # A NON-authority participant anchors a COMPETING digest LATER — it would be the
    # "last matching" payload an unscoped reader picks.
    await adversary.anchor_commitment(anchor_payload(epoch, netuid, evil_digest))

    # A reader BOUND to the authority ignores the adversary's anchor on BOTH read legs.
    reader = mk("some-validator", anchor_hotkey="authority-hk")
    assert reader.read_anchor(netuid=netuid, epoch_id=epoch, domain=ANCHOR_DOMAIN) == auth_digest
    assert (
        reader.read_anchor_block(netuid=netuid, epoch_id=epoch, domain=ANCHOR_DOMAIN)
        == auth_block
    )

    # Control: the filter IS what protects the binding — a reader keyed to the adversary (its own
    # hotkey, no authority binding) sees the adversary's competing digest.
    evil_reader = mk("adversary-hk")  # anchor_hotkey falls back to validator_hotkey
    assert (
        evil_reader.read_anchor(netuid=netuid, epoch_id=epoch, domain=ANCHOR_DOMAIN)
        == evil_digest
    )

    for adapter in (authority, adversary, reader, evil_reader):
        adapter.close()


def test_factory_passes_the_anchor_hotkey_through():
    """chain.anchor_hotkey binds the report-mode anchor READ path to the authority account —
    it must reach the adapter, not be silently dropped."""
    adapter = make_chain_adapter(
        {"chain": {"validator_hotkey": "val-9", "anchor_hotkey": "authority-ss58"}}
    )
    assert isinstance(adapter, HttpChainAdapter)
    assert adapter.anchor_authority_hotkey == "authority-ss58"
    adapter.close()

    # Empty anchor_hotkey falls back to validator_hotkey (self-anchoring single node).
    fallback = make_chain_adapter({"chain": {"validator_hotkey": "solo"}})
    assert fallback.anchor_authority_hotkey == "solo"
    fallback.close()


# ---- factory --------------------------------------------------------------------


def test_factory_default_is_report_mode_http_adapter():
    adapter = make_chain_adapter({})  # empty config: report mode IS the default
    assert isinstance(adapter, HttpChainAdapter)
    assert isinstance(adapter, ChainAdapter)
    adapter.close()


def test_factory_embedded_sentinel_builds_the_jsonl_chain(tmp_path):
    raw = {"chain": {"chainsim_url": "embedded", "report_dir": str(tmp_path)}}
    adapter = make_chain_adapter(raw)
    assert isinstance(adapter, EmbeddedReportingChain)
    assert Path(adapter.journal_path) == tmp_path / "embedded-chain.jsonl"


def test_factory_bittensor_mode_selects_the_real_transport_path(monkeypatch):
    """The explicit mode must never fall back to the report adapter.

    Stub the transport boundary so this chainless factory suite neither opens an
    RPC socket nor imports the provider SDK (whose global logging setup would
    contaminate unrelated tests in this process).
    """
    import vidaio.chain.bittensor_adapter as adapter_module

    def selected(config):
        assert config.netuid == 85
        raise RuntimeError("real bittensor transport selected")

    monkeypatch.setattr(adapter_module, "_connect_real_transport", selected)
    with pytest.raises(RuntimeError, match="real bittensor transport selected"):
        make_chain_adapter({"chain": {"mode": "bittensor"}})


def test_factory_passes_the_auth_token_through():
    """chain.auth_token IS the adapter's credential — a config typo must not
    silently produce an unauthenticated adapter."""
    adapter = make_chain_adapter(
        {"chain": {"validator_hotkey": "val-9", "auth_token": "s3cret-token"}}
    )
    assert isinstance(adapter, HttpChainAdapter)
    assert adapter.validator_hotkey == "val-9"
    assert adapter.auth_token == "s3cret-token"
    adapter.close()

    default = make_chain_adapter({})  # unset means "register and capture", not ""
    assert default.auth_token is None
    default.close()


# ---- the real WeightSetter against the sim --------------------------------------


async def test_weightsetter_happy_path_against_chainsim(sim, make_adapter, tmp_path):
    adapter = make_adapter("local-validator")
    assert adapter.register() == 0
    # Registration claimed the identity: every mutation below is SIGNED with the
    # token the sim issued (the real adapter will sign with a keypair instead).
    assert adapter.auth_token is not None
    for uid in (1, 2, 3):
        sim.register(
            RegisterRequest(hotkey=f"hk{uid}", coldkey=f"ck{uid}", ip=f"10.0.0.{uid}")
        )
    miners = [
        mk_miner(1, score=0.9),
        mk_miner(2, score=0.4),
        mk_miner(3, track="upscaling", score=0.7),
    ]

    conn = connect(":memory:")
    ws_migrate(conn)
    store = LocalFsStore(tmp_path / "audit")
    ledger = CommitmentLedger.open(tmp_path / "ledger.db")
    clock = Clock()
    setter = WeightSetter(
        {
            "core": {"metrics_port": 0},
            "weightsetter": {
                "metrics_port": 0,
                "chain_timeout_seconds": 5.0,
                "chain_retry_attempts": 2,
                "chain_retry_base_delay_seconds": 0.01,
            },
        },
        chain=adapter,
        snapshots=FakeSnapshots(miners),
        conn=conn,
        store=store,
        ledger=ledger,
        clock=clock,
    )

    assert await setter.attempt_once() is True

    # the sim recorded EXACTLY the tokenomics vector, under the validator's hotkey
    state = sim.state()
    assert len(state["weight_calls"]) == 1
    call = state["weight_calls"][0]
    assert call["hotkey"] == "local-validator"
    # This report-mode fixture supplies no active reward window. Tokenomics V2 keeps
    # IDLE's fixed 20% sink share explicit instead of renormalizing it across miners;
    # the live adapter resolves that sink to the registered anchor authority (uid 0).
    expected = build_weight_vector(
        setter.tokenomics, miners, burn_uid=adapter.get_burn_uid()
    )
    assert {int(uid): w for uid, w in call["vector"].items()} == expected

    # publication was anchored on the sim and the ledger advanced to anchored
    assert len(state["anchors"]) == 1
    assert [status for status, _ in ledger.history(1)] == [
        CommitmentStatus.PENDING_CHAIN.value,
        CommitmentStatus.ANCHORED.value,
    ]

    # the report shows the submitted vector (ranked) and the decoded anchor
    report = build_report(state)
    ranked = report["latest_vector"]["ranked"]
    assert {r["uid"] for r in ranked} == set(expected)
    assert ranked[0]["weight"] == max(expected.values())
    assert report["anchors"][0]["decoded"]["domain"] == COMMITMENT_DOMAIN
    assert report["anchors"][0]["decoded"]["kind"] == "publication"

    conn.close()


async def test_weightsetter_without_a_credential_sets_nothing(
    sim, make_adapter, tmp_path
):
    """The same service, same sim — but the adapter never claimed its hotkey.

    The attempt must FAIL (and say why), not silently look like a successful set.
    """
    registrar = make_adapter("local-validator")
    registrar.register()  # somebody else owns the identity
    adapter = make_adapter("local-validator")  # no token
    adapter.refresh()

    conn = connect(":memory:")
    ws_migrate(conn)
    setter = WeightSetter(
        {
            "core": {"metrics_port": 0},
            "weightsetter": {
                "metrics_port": 0,
                "chain_timeout_seconds": 5.0,
                "chain_retry_attempts": 1,
                "chain_retry_base_delay_seconds": 0.01,
            },
        },
        chain=adapter,
        snapshots=FakeSnapshots([mk_miner(1, score=0.9)]),
        conn=conn,
        store=LocalFsStore(tmp_path / "audit"),
        ledger=CommitmentLedger.open(tmp_path / "ledger.db"),
        clock=Clock(),
    )

    assert await setter.attempt_once() is False
    assert sim.state()["weight_calls"] == []
    conn.close()
