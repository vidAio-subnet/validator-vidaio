"""Fake-transport unit tests for the REAL bittensor ChainAdapter.

Every logic path of vidaio/chain/bittensor_adapter.py is exercised against a FAKE
transport — no live chain, no heavy bittensor dep. The real transport (the only
part that imports bittensor) is validated on testnet, not here; these tests prove
the ADAPTER's decisions: u16 quantization determinism, metagraph->ChainNeuron
mapping, set_weights classification, submitted-weights readback + commit-reveal
awareness, freshness, hotkey->uid reconciliation, and the reconnect discipline.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from datetime import datetime, timezone

import pytest

from vidaio.chain import (
    BittensorAdapterConfig,
    BittensorChainAdapter,
    BittensorHotkeySigner,
    BittensorReadOnlyChainAdapter,
    ChainStateUnavailable,
    CommitmentRecordReadable,
    EpochBoundary,
    HistoricalEpochAnchorReadable,
    SubmittedWeights,
    make_chain_adapter,
    make_read_only_chain_adapter,
    quantize_u16,
)
from vidaio.chain.bittensor_adapter import (
    EpochScheduleView,
    MetagraphView,
    ReadOnlyChainError,
    U16_MAX,
    _CommitmentUsageView,
    _RealSubtensorTransport,
    _hotkey_in_timelocked_commits,
    _parse_chain_result,
)
from vidaio.audit import NotConfiguredError
from vidaio.tokenomics.quantize import max_normalize_u16
from vidaio.weightsetter.intents import quantize_weights, weights_match


# --------------------------------------------------------------------------------------
# v10 SDK result shapes (ExtrinsicResponse + its receipt)
# --------------------------------------------------------------------------------------


class FakeReceipt:
    """A stand-in for a v10 ExtrinsicReceipt — carries the extrinsic hash."""

    def __init__(self, extrinsic_hash: str) -> None:
        self.extrinsic_hash = extrinsic_hash


class FakeExtrinsicResponse:
    """A stand-in for bittensor v10 `ExtrinsicResponse` (duck-typed by `.success`).

    Crucially it is TRUTHY even for a rejection (mirrors the real object), so a
    `bool(response)` classifier would read a failure as a success.
    """

    def __init__(
        self,
        success: bool,
        message: str = "",
        receipt=None,
        error: BaseException | dict | None = None,
    ) -> None:
        self.success = success
        self.message = message
        self.extrinsic_receipt = receipt
        self.error = error

    def __bool__(self) -> bool:  # a rejection object can still be truthy
        return True


# --------------------------------------------------------------------------------------
# Fake transport
# --------------------------------------------------------------------------------------


class FakeTransport:
    """A programmable stand-in for _SubtensorTransport — records + can raise."""

    def __init__(
        self,
        *,
        view: MetagraphView | None = None,
        uid_map: dict[str, int] | None = None,
        subnet_owner_hotkey: str | None = None,
        weights: dict[int, list[tuple[int, int]]] | None = None,
        last_update: dict[int, int] | None = None,
        timelocked_commits: dict[int, list[str]] | None = None,
        epoch: int = 0,
        commit_reveal: bool = False,
        rate_limit: int = 0,
        block: int = 1000,
        finalized_block: int | None = None,
        epoch_closes: dict[int, int] | None = None,
        set_result=(True, "included"),
        store_successful_non_cr_write: bool = True,
        commit_result=(True, "0xdeadbeef"),
        commitment_history: dict[int, tuple[str, int] | None] | None = None,
        commitment_max_space: int = 3100,
        commitment_usage_epoch: int | None = None,
        commitment_used_space: int = 0,
    ) -> None:
        self._view = view
        self._uid_map = uid_map or {}
        self._subnet_owner_hotkey = subnet_owner_hotkey
        self._weights = weights or {}
        self._last_update = last_update or {}
        #: v10 timelocked commits, keyed by (implicit netuid) EPOCH -> [hotkeys
        #: with a commit pending reveal].
        self._timelocked_commits = timelocked_commits or {}
        self._epoch = epoch
        self._commit_reveal = commit_reveal
        self._rate_limit = rate_limit
        self._block = block
        self._finalized_block = block if finalized_block is None else finalized_block
        self._epoch_closes = dict(epoch_closes or {})
        # A raw SDK result: a legacy (bool, msg) tuple, a bare bool, a v10
        # FakeExtrinsicResponse, or an exception instance (raised). Runs through the
        # real `_parse_chain_result`, exactly like the production transport.
        self.set_result = set_result
        self.store_successful_non_cr_write = store_successful_non_cr_write
        self.commit_result = commit_result
        self.commitment_history = commitment_history or {}
        self._commitment_max_space = commitment_max_space
        self._commitment_usage_epoch = commitment_usage_epoch
        self._commitment_used_space = commitment_used_space
        self.metagraph_calls: list[tuple[int, int | None]] = []
        self.set_calls: list[tuple[list[int], list[int], int]] = []
        self.submitted_weights_calls = 0
        #: v10.5.0 set_commitment(data: str) — the payload crosses this seam as a
        #: STR, never bytes. Records exactly what the adapter passed.
        self.commit_calls: list[str] = []
        #: the last-committed payload, served back by get_commitment as ascii bytes
        #: (parse_anchor_digest decodes ascii) so the anchor round-trips: str in ->
        #: str out -> the same digest.
        self._committed: str | None = None
        #: the block the last commitment landed at (an internal review — the beacon binds
        #: to it). Recorded by set_commitment as the then-current block.
        self._commit_block: int | None = None
        self.commitment_read_calls: list[tuple[str, int | None]] = []
        self.commitment_usage_calls: list[tuple[int, str, int]] = []
        self.closed = 0
        self.raise_on: set[str] = set()
        self.raise_exc: BaseException = RuntimeError("transport boom")

    def _maybe(self, name: str) -> None:
        if name in self.raise_on:
            raise self.raise_exc

    def current_block(self) -> int:
        self._maybe("current_block")
        return self._block

    def finalized_block(self) -> int:
        self._maybe("finalized_block")
        return self._finalized_block

    def epoch_index(self, netuid: int, block_number: int) -> int:
        self._maybe("epoch_index")
        return max(
            (
                epoch
                for epoch, close in self._epoch_closes.items()
                if close <= block_number
            ),
            default=0,
        )

    def epoch_schedule(self, netuid: int, block_number: int) -> EpochScheduleView:
        self._maybe("epoch_schedule")
        index = self.epoch_index(netuid, block_number)
        last = self._epoch_closes.get(index, 0)
        return EpochScheduleView(
            block=block_number,
            last_epoch_block=last,
            pending_epoch_at=0,
            subnet_epoch_index=index,
            tempo=100,
            blocks_since_last_step=max(0, block_number - last),
        )

    def metagraph(self, netuid: int, block_number: int | None = None) -> MetagraphView:
        self._maybe("metagraph")
        self.metagraph_calls.append((netuid, block_number))
        assert self._view is not None
        if block_number is None:
            return self._view
        return MetagraphView(
            block=block_number,
            hotkeys=self._view.hotkeys,
            coldkeys=self._view.coldkeys,
            axon_ips=self._view.axon_ips,
            alpha_stake=self._view.alpha_stake,
            emission=self._view.emission,
            validator_permit=self._view.validator_permit,
            last_update=self._view.last_update,
            registration_block=self._view.registration_block,
            axon_ports=self._view.axon_ports,
        )

    def set_weights(self, *, netuid, uids, weights, version_key):
        self.set_calls.append((list(uids), list(weights), version_key))
        if isinstance(self.set_result, BaseException):
            raise self.set_result
        outcome = _parse_chain_result(self.set_result)  # the real classifier
        if (
            outcome.success
            and not self._commit_reveal
            and self.store_successful_non_cr_write
        ):
            source_uid = next(
                (
                    uid
                    for uid, permitted in enumerate(self._view.validator_permit)
                    if permitted
                ),
                None,
            )
            if source_uid is not None:
                emitted = max_normalize_u16(dict(zip(uids, weights, strict=True)))
                self._weights[source_uid] = list(emitted.items())
                self._last_update[source_uid] = (
                    max(self._block, int(self._view.block)) + 1
                )
        return outcome.success, outcome.message, self._commit_reveal

    def commit_reveal_enabled(self, netuid):
        self._maybe("commit_reveal_enabled")
        return self._commit_reveal

    def query_weights(self, netuid, uid):
        self._maybe("query_weights")
        return self._weights.get(uid, [])

    def query_last_update(self, netuid, uid):
        self._maybe("query_last_update")
        return self._last_update.get(uid, 0)

    def submitted_weights_at_finalized_head(self, netuid, hotkey):
        self.submitted_weights_calls += 1
        self._maybe("submitted_weights_at_finalized_head")
        self._maybe("query_weights")
        self._maybe("query_last_update")
        uid = self._uid_map.get(hotkey)
        if uid is None:
            raise LookupError(f"unregistered hotkey {hotkey}")
        raw = self._weights.get(uid, [])
        if not raw:
            return None
        return SubmittedWeights(
            weights={int(target): float(weight) for target, weight in raw},
            block=int(self._last_update.get(uid, 0)),
        )

    def pending_timelocked_commit(self, netuid, hotkey):
        self._maybe("pending_timelocked_commit")
        return hotkey in self._timelocked_commits.get(self._epoch, [])

    def uid_for_hotkey(self, hotkey, netuid):
        self._maybe("uid_for_hotkey")
        return self._uid_map.get(hotkey)

    def subnet_owner_hotkey(self, netuid):
        self._maybe("subnet_owner_hotkey")
        return self._subnet_owner_hotkey

    def weights_rate_limit(self, netuid):
        self._maybe("weights_rate_limit")
        return self._rate_limit

    def commitment_rate_limit(self):
        self._maybe("commitment_rate_limit")
        return 7

    def commitment_usage(self, *, netuid, ss58, block_number):
        self._maybe("commitment_usage")
        self.commitment_usage_calls.append((netuid, ss58, block_number))
        return _CommitmentUsageView(
            block=block_number,
            max_space=self._commitment_max_space,
            usage_epoch=self._commitment_usage_epoch,
            used_space=self._commitment_used_space,
        )

    def tempo(self, netuid):
        return 100

    def block_time(self, block_number):
        self._maybe("block_time")
        if block_number > self._block:
            return None
        return datetime.fromtimestamp(block_number * 12, tz=timezone.utc)

    def set_commitment(self, *, netuid, payload):
        # v10.5.0 set_commitment(data: str) does data.encode() itself: the real SDK
        # RAISES on bytes. Model that strictly so a regression that hands bytes to
        # the transport is caught here rather than mid-extrinsic.
        if not isinstance(payload, str):
            raise TypeError(
                f"set_commitment(data) must be str, got {type(payload).__name__}"
            )
        self.commit_calls.append(payload)
        self._committed = payload
        self._commit_block = self._block  # the inclusion block
        if isinstance(self.commit_result, BaseException):
            raise self.commit_result
        outcome = _parse_chain_result(self.commit_result)
        if not outcome.success:
            raise OSError(f"set_commitment rejected: {outcome.message}")
        return outcome.receipt_hash or outcome.message or "committed"

    def get_commitment(self, *, netuid, ss58, block_number=None):
        self._maybe("get_commitment")
        self.commitment_read_calls.append(("payload", block_number))
        if block_number is not None:
            record = self.commitment_history.get(block_number)
            return None if record is None else record[0].encode("ascii")
        if self._committed is None:
            return None
        # The chain stores/returns the ascii payload bytes (parse_anchor_digest
        # decodes ascii) — mirrors _RealSubtensorTransport.get_commitment.
        return self._committed.encode("ascii")

    def get_commitment_block(self, *, netuid, ss58, block_number=None):
        self._maybe("get_commitment_block")
        self.commitment_read_calls.append(("block", block_number))
        if block_number is not None:
            record = self.commitment_history.get(block_number)
            return None if record is None else record[1]
        return self._commit_block  # None until something is committed

    def get_block_hash(self, block_number):
        self._maybe("get_block_hash")
        # A deterministic 0x-prefixed hash of the block height (the real substrate hash
        # is 0x-prefixed too; the adapter strips 0x before returning the beacon).
        return "0x" + f"{block_number:064x}"

    def close(self):
        self.closed += 1


def make_view(n: int = 3, *, own: int = 0) -> MetagraphView:
    """A metagraph with uid `own` as the validator and the rest as miners."""
    return MetagraphView(
        block=1000,
        hotkeys=[f"hk-{i}" for i in range(n)],
        coldkeys=[f"ck-{i}" for i in range(n)],
        axon_ips=[f"10.0.0.{i}" for i in range(n)],
        alpha_stake=[100.0 * i for i in range(n)],
        emission=[0.5 * i for i in range(n)],
        validator_permit=[i == own for i in range(n)],
        last_update=[900 + i for i in range(n)],
        registration_block=[800 + i for i in range(n)],
        axon_ports=[8300 + i for i in range(n)],
    )


def make_adapter(transport: FakeTransport, **cfg_overrides) -> BittensorChainAdapter:
    cfg = BittensorAdapterConfig(
        validator_hotkey=cfg_overrides.pop("validator_hotkey", "hk-0"),
        netuid=1,
        metagraph_ttl_seconds=cfg_overrides.pop("metagraph_ttl_seconds", 120.0),
        reconnect_after_consecutive_failures=cfg_overrides.pop("reconnect_after", 3),
        # Keep proof-exhaustion tests instant. Production/default-contract tests
        # below independently pin the real one-block delay.
        weight_readback_delay_seconds=cfg_overrides.pop(
            "weight_readback_delay_seconds", 0.0
        ),
        **cfg_overrides,
    )
    return BittensorChainAdapter(cfg, transport=transport)


def make_read_only_adapter(
    transport: FakeTransport,
    *,
    connect_transport=None,
    reconnect_after: int = 3,
) -> BittensorReadOnlyChainAdapter:
    return BittensorReadOnlyChainAdapter(
        BittensorAdapterConfig(
            netuid=1,
            validator_hotkey="",
            hotkey_seed_env="",
            read_only=True,
            reconnect_after_consecutive_failures=reconnect_after,
        ),
        transport=transport,
        connect_transport=connect_transport,
    )


@pytest.mark.asyncio
async def test_wallet_free_adapter_reads_metagraph_but_rejects_all_key_operations() -> (
    None
):
    transport = FakeTransport(view=make_view(3))
    # A registration reader must not query a fictitious "own" validator identity.
    transport.raise_on.update({"uid_for_hotkey", "query_last_update"})
    adapter = make_read_only_adapter(transport)

    adapter.refresh()
    assert [neuron.hotkey for neuron in adapter.neurons()] == ["hk-0", "hk-1", "hk-2"]
    assert adapter.own_uid is None
    assert adapter.last_refresh_error is None

    with pytest.raises(ReadOnlyChainError, match="cannot sign"):
        adapter.sign(b"report")
    with pytest.raises(ReadOnlyChainError, match="cannot submit weights"):
        await adapter.set_weights({1: 1.0}, version_key=12)
    with pytest.raises(ReadOnlyChainError, match="cannot anchor"):
        await adapter.anchor_commitment(b"anchor")
    assert transport.set_calls == []
    assert transport.commit_calls == []


def test_wallet_free_adapter_retains_failure_reconnect_discipline() -> None:
    failed = FakeTransport(view=make_view(1))
    failed.raise_on.add("metagraph")
    recovered = FakeTransport(view=make_view(2))
    adapter = make_read_only_adapter(
        failed, connect_transport=lambda: recovered, reconnect_after=1
    )

    adapter.refresh()
    assert adapter.last_refresh_error is not None
    adapter.refresh()

    assert failed.closed == 1
    assert [neuron.hotkey for neuron in adapter.neurons()] == ["hk-0", "hk-1"]
    assert adapter.last_refresh_error is None


def test_burn_uid_is_resolved_from_current_subnet_owner_hotkey() -> None:
    transport = FakeTransport(
        view=make_view(4),
        uid_map={"validator": 0, "owner-current": 3},
        subnet_owner_hotkey="owner-current",
    )
    adapter = make_adapter(transport, validator_hotkey="validator")
    assert adapter.get_burn_uid() == 3


@pytest.mark.parametrize("owner,uid_map", [(None, {}), ("owner-missing", {})])
def test_burn_uid_fails_closed_when_owner_identity_is_unresolvable(
    owner: str | None, uid_map: dict[str, int]
) -> None:
    transport = FakeTransport(
        view=make_view(), uid_map=uid_map, subnet_owner_hotkey=owner
    )
    adapter = make_adapter(transport)
    with pytest.raises(ChainStateUnavailable, match="owner"):
        adapter.get_burn_uid()


def test_burn_uid_chain_read_failure_is_not_replaced_with_uid_zero() -> None:
    transport = FakeTransport(
        view=make_view(), uid_map={"owner": 2}, subnet_owner_hotkey="owner"
    )
    transport.raise_on.add("subnet_owner_hotkey")
    adapter = make_adapter(transport)
    with pytest.raises(ChainStateUnavailable, match="cannot resolve"):
        adapter.get_burn_uid()


# --------------------------------------------------------------------------------------
# quantize_u16 — the convergence-critical function
# --------------------------------------------------------------------------------------


def test_quantize_sums_to_exactly_65535():
    for vec in ({1: 0.5, 2: 0.3, 3: 0.2}, {7: 1.0}, {1: 0.1, 2: 0.1, 3: 0.8}):
        assert sum(quantize_u16(vec).values()) == U16_MAX


def test_quantize_is_deterministic_and_byte_identical_across_validators():
    # Two "validators" holding the SAME float vector (built in DIFFERENT key
    # orders) must emit BYTE-IDENTICAL u16 — the DECISIONS rule-9 property.
    vec_a = {3: 0.2, 1: 0.5, 2: 0.3}
    vec_b = {1: 0.5, 2: 0.3, 3: 0.2}
    q_a = quantize_u16(vec_a)
    q_b = quantize_u16(vec_b)
    assert q_a == q_b
    # identical ordered items (byte-identical serialization)
    assert sorted(q_a.items()) == sorted(q_b.items())
    # and stable across repeated calls
    assert quantize_u16(vec_a) == q_a


def test_quantize_drops_non_positive_and_empty_in_empty_out():
    assert quantize_u16({1: 0.0, 2: -3.0}) == {}
    assert quantize_u16({}) == {}
    q = quantize_u16({1: 1.0, 2: 0.0, 3: -1.0})
    assert set(q) == {1}


def test_quantize_floors_tiny_positive_share_to_at_least_one_unit():
    # A sub-1/65535 share must not vanish to zero (it must stay comparable).
    q = quantize_u16({1: 1e-12, 2: 1.0})
    assert q[1] >= 1
    assert sum(q.values()) == U16_MAX


def test_quantize_refuses_more_positive_miners_than_grid_units():
    with pytest.raises(ValueError):
        quantize_u16({uid: 1.0 for uid in range(U16_MAX + 1)})


def test_quantize_matches_the_weightsetters_grid_for_comparison():
    # The adapter's quantize and the weight-setter's quantize_weights must agree
    # up to the weight-setter's 1-step tolerance, so a submitted vector read back
    # as raw u16 still matches its intent.
    vec = {1: 0.6, 2: 0.4}
    raw_u16 = quantize_u16(vec)
    reported_as_floats = {uid: float(w) for uid, w in raw_u16.items()}
    assert weights_match(reported_as_floats, vec)
    # sanity: the weight-setter's own quantization of the same vector
    assert quantize_weights(vec).keys() == raw_u16.keys()


# --------------------------------------------------------------------------------------
# freshness
# --------------------------------------------------------------------------------------


def test_never_refreshed_neurons_raises_and_age_is_none():
    adapter = make_adapter(FakeTransport(view=make_view()))
    with pytest.raises(ChainStateUnavailable):
        adapter.neurons()
    assert adapter.snapshot_age(now=10.0) is None
    assert adapter.has_fresh_snapshot(now=10.0, max_age_seconds=100.0) is False
    assert adapter.current_block() == 0


def test_refresh_populates_snapshot_and_freshness():
    clock = [1000.0]
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0}, last_update={0: 950})
    adapter = BittensorChainAdapter(
        BittensorAdapterConfig(validator_hotkey="hk-0", netuid=1),
        transport=t,
        clock=lambda: clock[0],
    )
    adapter.refresh()
    assert adapter.current_block() == 1000
    assert len(adapter.neurons()) == 3
    assert adapter.own_uid == 0
    assert adapter.blocks_since_last_update() == 50
    assert adapter.has_fresh_snapshot(now=1000.0, max_age_seconds=120.0)
    assert adapter.snapshot_age(now=1030.0) == 30.0


def test_refresh_honours_ttl_and_never_raises_on_transport_failure():
    clock = [1000.0]
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = BittensorChainAdapter(
        BittensorAdapterConfig(
            validator_hotkey="hk-0", netuid=1, metagraph_ttl_seconds=60.0
        ),
        transport=t,
        clock=lambda: clock[0],
    )
    adapter.refresh()
    first = adapter.snapshot_age(now=clock[0])
    assert first == 0.0
    # inside TTL: skipped (no error even if transport would raise)
    t.raise_on = {"metagraph"}
    clock[0] = 1030.0
    adapter.refresh()  # skipped by TTL
    assert adapter.last_refresh_error is None
    # past TTL: attempts, transport raises, snapshot KEPT, error recorded, NO raise
    clock[0] = 1100.0
    adapter.refresh()
    assert adapter.last_refresh_error is not None
    assert len(adapter.neurons()) == 3  # last-good kept


# --------------------------------------------------------------------------------------
# metagraph -> ChainNeuron mapping
# --------------------------------------------------------------------------------------


def test_metagraph_maps_every_field_including_validator_permit():
    t = FakeTransport(view=make_view(3, own=1), uid_map={"hk-1": 1})
    adapter = make_adapter(t, validator_hotkey="hk-1")
    adapter.refresh()
    neurons = {n.uid: n for n in adapter.neurons()}
    assert neurons[1].is_validator is True
    assert neurons[0].is_validator is False
    assert neurons[2].hotkey == "hk-2"
    assert neurons[2].coldkey == "ck-2"
    assert neurons[2].ip == "10.0.0.2"
    assert neurons[2].axon_port == 8302
    assert neurons[2].alpha_stake == 200.0
    assert neurons[1].last_update == 901
    assert neurons[2].registration_block == 802


def test_neurons_at_is_exact_block_pinned_and_does_not_mutate_head_cache():
    transport = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(transport)
    adapter.refresh()

    historical = adapter.neurons_at(777)
    assert transport.metagraph_calls[-1] == (1, 777)
    assert [neuron.uid for neuron in historical] == [0, 1, 2]
    assert historical[2].registration_block == 802
    assert adapter.current_block() == 1000
    assert adapter.neurons()[2].last_update == 902

    transport.raise_on = {"metagraph"}
    with pytest.raises(ChainStateUnavailable, match="at block 777"):
        adapter.neurons_at(777)
    # No silent fallback to the still-good current-head cache.
    assert adapter.neurons()[2].hotkey == "hk-2"


def test_neurons_at_rejects_transport_head_fallback():
    class HeadFallbackTransport(FakeTransport):
        def metagraph(self, netuid, block_number=None):
            # Models a pruned/old SDK silently ignoring the historical block.
            return self._view

    transport = HeadFallbackTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(transport)
    with pytest.raises(ChainStateUnavailable, match="not the requested"):
        adapter.neurons_at(777)


def test_block_time_and_tempo_use_live_chain_semantics():
    transport = FakeTransport(view=make_view(3), uid_map={"hk-0": 0}, block=1000)
    adapter = make_adapter(transport)

    assert adapter.block_time(10) == datetime.fromtimestamp(120, tz=timezone.utc)
    assert adapter.block_time(1001) is None
    # In pinned v10.5.0 tempo is already the runtime's exact period.
    assert adapter.tempo() == 100

    transport.raise_on = {"block_time"}
    with pytest.raises(ChainStateUnavailable, match="block_time"):
        adapter.block_time(10)
    with pytest.raises(ValueError, match="non-negative"):
        adapter.block_time(-1)


def test_finalized_block_is_independent_of_best_head() -> None:
    transport = FakeTransport(
        view=make_view(3), uid_map={"hk-0": 0}, block=1_020, finalized_block=1_013
    )
    adapter = make_adapter(transport)
    adapter.refresh()

    assert adapter.current_block() == 1_020
    assert adapter.finalized_block() == 1_013

    transport.raise_on = {"finalized_block"}
    with pytest.raises(ChainStateUnavailable, match="GRANDPA-finalized"):
        adapter.finalized_block()


def test_epoch_boundaries_follow_exact_historical_index_transitions() -> None:
    # Irregular gaps model a tempo reset, an owner-triggered early epoch and a
    # deferred epoch. There is deliberately no zero-offset arithmetic relation.
    transport = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        block=360,
        finalized_block=350,
        epoch_closes={39: 100, 40: 250, 41: 310, 42: 333},
    )
    adapter = make_adapter(transport)

    assert adapter.latest_closed_epoch(netuid=1) == EpochBoundary(42, 333)
    assert adapter.epoch_close_block(netuid=1, epoch_id=40) == 250
    assert adapter.epoch_close_block(netuid=1, epoch_id=43) is None


def test_epoch_boundary_search_stays_above_pre_migration_storage() -> None:
    """Old blocks legitimately lack SubnetEpochIndex under their old metadata.

    A post-migration transition remains provable from a post-migration
    predecessor; boundary discovery must not insist on querying block 1.
    """

    class MigratedScheduleTransport(FakeTransport):
        migration_block = 500

        def epoch_index(self, netuid, block_number):
            if block_number < self.migration_block:
                raise LookupError("SubnetEpochIndex did not exist before migration")
            if block_number < 650:
                return 40  # the migration-seeded counter (not an epoch close)
            if block_number < 800:
                return 41
            return 42

        def epoch_schedule(self, netuid, block_number):
            index = self.epoch_index(netuid, block_number)
            last = 500 if index == 40 else 650 if index == 41 else 800
            return EpochScheduleView(
                block=block_number,
                last_epoch_block=last,
                pending_epoch_at=0,
                subnet_epoch_index=index,
                tempo=100,
                blocks_since_last_step=max(0, block_number - last),
            )

    adapter = make_adapter(
        MigratedScheduleTransport(
            view=make_view(3),
            uid_map={"hk-0": 0},
            block=900,
            finalized_block=900,
        )
    )

    assert adapter.latest_closed_epoch(netuid=1) == EpochBoundary(42, 800)
    assert adapter.epoch_close_block(netuid=1, epoch_id=41) == 650
    with pytest.raises(ChainStateUnavailable, match="migration-seeded"):
        adapter.epoch_close_block(netuid=1, epoch_id=40)


def test_epoch_boundary_rejects_a_migration_counter_jump() -> None:
    class JumpTransport(FakeTransport):
        def epoch_index(self, netuid, block_number):
            return 0 if block_number < 500 else 10

        def epoch_schedule(self, netuid, block_number):
            index = self.epoch_index(netuid, block_number)
            return EpochScheduleView(
                block=block_number,
                last_epoch_block=500 if block_number >= 500 else 0,
                pending_epoch_at=0,
                subnet_epoch_index=index,
                tempo=100,
                blocks_since_last_step=max(0, block_number - 500),
            )

    adapter = make_adapter(
        JumpTransport(
            view=make_view(3), uid_map={"hk-0": 0}, block=600, finalized_block=600
        )
    )
    with pytest.raises(ChainStateUnavailable, match=r"expected 9->10"):
        adapter.epoch_close_block(netuid=1, epoch_id=10)


def test_epoch_boundary_rejects_last_epoch_block_that_did_not_move_with_index() -> None:
    class BadLastEpochTransport(FakeTransport):
        def epoch_schedule(self, netuid, block_number):
            state = super().epoch_schedule(netuid, block_number)
            if block_number == 333:
                return EpochScheduleView(
                    block=state.block,
                    last_epoch_block=320,
                    pending_epoch_at=state.pending_epoch_at,
                    subnet_epoch_index=state.subnet_epoch_index,
                    tempo=state.tempo,
                    blocks_since_last_step=state.blocks_since_last_step,
                )
            return state

    adapter = make_adapter(
        BadLastEpochTransport(
            view=make_view(3),
            uid_map={"hk-0": 0},
            block=350,
            finalized_block=350,
            epoch_closes={40: 250, 41: 310, 42: 333},
        )
    )
    with pytest.raises(ChainStateUnavailable, match="LastEpochBlock=320"):
        adapter.latest_closed_epoch(netuid=1)


def test_epoch_boundary_requires_archive_history() -> None:
    transport = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        block=350,
        finalized_block=350,
        epoch_closes={40: 250, 41: 310, 42: 333},
    )
    transport.raise_on = {"epoch_index"}
    adapter = make_adapter(transport)
    with pytest.raises(ChainStateUnavailable, match="historical SubnetEpochIndex"):
        adapter.epoch_close_block(netuid=1, epoch_id=42)


def test_wallet_identity_and_signing_seams_fail_fast():
    class SigningTransport(FakeTransport):
        def __init__(self, *args, signer="hk-0", **kwargs):
            super().__init__(*args, **kwargs)
            self.signer = signer

        def signer_hotkey(self):
            return self.signer

        def sign_hotkey(self, payload):
            assert payload == b"canonical-report"
            return b"\x01" * 64

    transport = SigningTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(transport)
    assert adapter.sign(b"canonical-report") == "01" * 64
    assert BittensorHotkeySigner(adapter).sign(b"canonical-report") == "01" * 64

    wrong = SigningTransport(
        view=make_view(3), uid_map={"other-hotkey": 0}, signer="other-hotkey"
    )
    with pytest.raises(RuntimeError, match="does not match"):
        make_adapter(wrong)
    assert wrong.closed == 1

    replacement = SigningTransport(
        view=make_view(3), uid_map={"rotated-hotkey": 0}, signer="rotated-hotkey"
    )
    adapter._connect_transport = lambda: replacement
    with pytest.raises(RuntimeError, match="does not match"):
        adapter._reconnect()
    assert replacement.closed == 1


@pytest.mark.parametrize("raw_signature", (b"\x01" * 63, b"\x01" * 65, "not-bytes"))
def test_wallet_signer_requires_an_exact_64_byte_signature(raw_signature):
    class MalformedSigningTransport(FakeTransport):
        def signer_hotkey(self):
            return "hk-0"

        def sign_hotkey(self, payload):  # noqa: ARG002
            return raw_signature

    adapter = make_adapter(
        MalformedSigningTransport(view=make_view(3), uid_map={"hk-0": 0})
    )
    with pytest.raises(RuntimeError, match="exact 64-byte"):
        adapter.sign(b"canonical-report")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"validator_hotkey": ""}, "validator_hotkey"),
        ({"validator_hotkey": "hk", "netuid": -1}, "netuid"),
        ({"validator_hotkey": "hk", "version_key": -1}, "version_key"),
        (
            {"validator_hotkey": "hk", "wallet_name": "wallet-only"},
            "wallet_name and wallet_hotkey",
        ),
        ({"validator_hotkey": "hk", "hotkey_seed_env": ""}, "hotkey_seed_env"),
        ({"validator_hotkey": "hk", "rpc_timeout_seconds": 0}, "positive"),
        (
            {"validator_hotkey": "hk", "weight_readback_attempts": 0},
            "at least 1",
        ),
        (
            {"validator_hotkey": "hk", "weight_readback_delay_seconds": -1},
            "non-negative",
        ),
        (
            {"validator_hotkey": "hk", "reconnect_after_consecutive_failures": 0},
            "at least 1",
        ),
    ],
)
def test_bittensor_adapter_config_rejects_invalid_startup_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BittensorAdapterConfig(**kwargs)


def test_bittensor_adapter_default_version_fences_current_schema():
    config = BittensorAdapterConfig(validator_hotkey="hk")
    assert config.version_key == 16
    assert config.weight_readback_attempts == 5
    assert config.weight_readback_delay_seconds == 12.0


@pytest.mark.parametrize(
    "fallbacks,match",
    (
        (("",), "empty"),
        (("wss://archive-b", "wss://archive-b"), "duplicates"),
        (("wss://archive-a",), "primary endpoint"),
    ),
)
def test_bittensor_adapter_config_rejects_invalid_fallback_endpoints(fallbacks, match):
    with pytest.raises(ValueError, match=match):
        BittensorAdapterConfig(
            validator_hotkey="hk",
            endpoint="wss://archive-a",
            fallback_endpoints=fallbacks,
        )


def test_bittensor_adapter_config_normalizes_fallback_endpoints():
    config = BittensorAdapterConfig(
        validator_hotkey="hk",
        endpoint="wss://archive-a",
        fallback_endpoints=("  wss://archive-b  ",),
    )
    assert config.fallback_endpoints == ("wss://archive-b",)


def test_metagraph_ip_defaults_to_zero_when_axon_absent():
    view = make_view(2)
    stripped = MetagraphView(
        block=view.block,
        hotkeys=view.hotkeys,
        coldkeys=view.coldkeys,
        axon_ips=[],  # no axons served (the [PENDING DECISION] path)
        alpha_stake=view.alpha_stake,
        emission=view.emission,
        validator_permit=view.validator_permit,
        last_update=view.last_update,
    )
    t = FakeTransport(view=stripped, uid_map={"hk-0": 0})
    adapter = make_adapter(t)
    adapter.refresh()
    assert all(n.ip == "0.0.0.0" for n in adapter.neurons())


# --------------------------------------------------------------------------------------
# set_weights classification + quantization + version_key
# --------------------------------------------------------------------------------------


async def test_set_weights_success_quantizes_and_resets_failure_counter():
    t = FakeTransport(view=make_view(4), uid_map={"hk-0": 0}, set_result=(True, "ok"))
    adapter = make_adapter(t)
    adapter.refresh()
    adapter._consecutive_failures = 2  # a prior read failure
    result = await adapter.set_weights({1: 0.5, 2: 0.3, 3: 0.2}, version_key=0)
    assert result.success is True
    uids, vals, _vk = t.set_calls[-1]
    assert uids == [1, 2, 3]
    assert sum(vals) == U16_MAX
    assert result.submitted == max_normalize_u16(dict(zip(uids, vals, strict=True)))
    assert max(result.submitted.values()) == U16_MAX
    assert adapter._consecutive_failures == 0  # clean submit resets


async def test_set_weights_version_key_passthrough_and_config_default():
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(t, version_key=5)
    adapter.refresh()
    await adapter.set_weights({1: 1.0}, version_key=7)  # explicit wins
    assert t.set_calls[-1][2] == 7
    await adapter.set_weights({1: 1.0}, version_key=0)  # falls back to config 5
    assert t.set_calls[-1][2] == 5


async def test_set_weights_tempo_pregate_message_contains_tempo():
    # rate_limit window open -> pre-gate, message MUST contain "tempo".
    t = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        last_update={0: 995},
        rate_limit=50,
        block=1000,
    )
    adapter = make_adapter(t)
    adapter.refresh()  # blocks_since = 1000-995 = 5 <= 50
    result = await adapter.set_weights({1: 1.0}, version_key=0)
    assert result.success is False
    assert "tempo" in result.message.lower()
    assert not t.set_calls  # never fired the doomed commit


async def test_set_weights_chain_rate_limit_rejection_mapped_to_tempo():
    t = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        set_result=(False, "SettingWeightsTooFast"),
    )
    adapter = make_adapter(t)
    adapter.refresh()
    result = await adapter.set_weights({1: 1.0}, version_key=0)
    assert result.success is False
    assert "tempo" in result.message.lower()
    assert adapter._consecutive_failures == 0  # a chain answer is NOT transport trouble


async def test_set_weights_other_chain_rejection_preserved_not_tempo():
    t = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        set_result=(False, "Unregistered hotkey"),
    )
    adapter = make_adapter(t)
    adapter.refresh()
    result = await adapter.set_weights({1: 1.0}, version_key=0)
    assert result.success is False
    assert "tempo" not in result.message.lower()
    assert "Unregistered" in result.message
    assert adapter._consecutive_failures == 0  # healthy socket, counter untouched


async def test_set_weights_transport_raise_counts_toward_reconnect():
    t = FakeTransport(
        view=make_view(3), uid_map={"hk-0": 0}, set_result=RuntimeError("socket")
    )
    adapter = make_adapter(t)
    adapter.refresh()
    with pytest.raises(OSError, match="transport failure"):
        await adapter.set_weights({1: 1.0}, version_key=0)
    assert adapter._consecutive_failures == 1  # a RAISE counts
    assert adapter._condemned is True  # readback reconnects before deciding fate


# ------------------------------------------------------------------------------------)
# --------------------------------------------------------------------------------------


def test_parse_chain_result_classifies_every_shape_without_implicit_success():
    # 1. v10 ExtrinsicResponse: read .success/.message/.extrinsic_receipt — a
    #    truthy REJECTION object must classify as a FAILURE, never bool(obj)==True.
    reject = FakeExtrinsicResponse(False, "chain rejected", FakeReceipt("0xabc"))
    assert bool(reject) is True  # would have been a false success under bool(result)
    out = _parse_chain_result(reject)
    assert out.success is False and out.message == "chain rejected"
    ok = FakeExtrinsicResponse(True, "included", FakeReceipt("0xdeadbeef"))
    out = _parse_chain_result(ok)
    assert out.success is True and out.receipt_hash == "0xdeadbeef"
    # 2. legacy (bool, msg) tuple; 3. bare bool
    assert _parse_chain_result((True, "ok")).success is True
    assert _parse_chain_result((False, "no")).success is False
    assert _parse_chain_result(True).success is True
    assert _parse_chain_result(False).success is False
    # 4. anything else -> RAISE (a FAILURE, never an implicit success)
    with pytest.raises(TypeError):
        _parse_chain_result(object())
    with pytest.raises(TypeError):
        _parse_chain_result(None)


async def test_set_weights_v10_rejection_object_is_never_a_false_success():
    # The exact an internal review regression: a v10 ExtrinsicResponse rejection reaching the
    # adapter must yield SetWeightsResult(success=False) — publishing a vector that
    # never landed is the failure mode.
    t = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        set_result=FakeExtrinsicResponse(False, "SubtensorApi.rejected"),
    )
    adapter = make_adapter(t)
    adapter.refresh()
    result = await adapter.set_weights({1: 1.0}, version_key=0)
    assert result.success is False
    assert "rejected" in result.message.lower()
    assert adapter._consecutive_failures == 0  # a chain answer, not transport trouble


async def test_structured_rate_limit_error_is_safely_rendered_and_normalized():
    class OpaqueSecret:
        def __str__(self):
            raise AssertionError("opaque dispatch fields must never be stringified")

        def __repr__(self):
            raise AssertionError("opaque dispatch fields must never be repr'd")

    transport = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        set_result=FakeExtrinsicResponse(
            False,
            "",
            error={
                "name": "SettingWeightsTooFast",
                "module": "SubtensorModule",
                "credential": "must-not-appear",
                "opaque": OpaqueSecret(),
            },
        ),
    )
    adapter = make_adapter(transport)
    adapter.refresh()

    result = await adapter.set_weights({1: 1.0}, version_key=0)

    assert result.success is False
    assert result.message.startswith("tempo gate: chain rate-limit rejection")
    assert "SettingWeightsTooFast" in result.message
    assert "SubtensorModule" in result.message
    assert "must-not-appear" not in result.message


async def test_set_weights_v10_success_object_is_a_success():
    t = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        set_result=FakeExtrinsicResponse(True, "included", FakeReceipt("0xfeed")),
    )
    adapter = make_adapter(t)
    adapter.refresh()
    result = await adapter.set_weights({1: 1.0}, version_key=0)
    assert result.success is True
    assert result.block == 1001  # exact finalized LastUpdate, not a best-head guess
    assert result.submitted == {1: U16_MAX}


async def test_non_cr_success_polls_finalized_proof_without_a_second_write():
    """Archive lag is waited out inside one submission, never retried as a write."""

    class LaggedProofTransport(FakeTransport):
        def submitted_weights_at_finalized_head(self, netuid, hotkey):
            self.submitted_weights_calls += 1
            if self.submitted_weights_calls < 3:
                return SubmittedWeights(weights={2: float(U16_MAX)}, block=900)
            return SubmittedWeights(weights={1: float(U16_MAX)}, block=1001)

    transport = LaggedProofTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        set_result=FakeExtrinsicResponse(True, "SDK says included"),
        store_successful_non_cr_write=False,
    )
    adapter = make_adapter(
        transport,
        weight_readback_attempts=5,
        weight_readback_delay_seconds=0.0,
    )
    adapter.refresh()

    result = await adapter.set_weights({1: 1.0}, version_key=0)

    assert result.success is True
    assert result.block == 1001
    assert result.submitted == {1: U16_MAX}
    assert transport.submitted_weights_calls == 3
    assert len(transport.set_calls) == 1  # polling never emits another extrinsic


@pytest.mark.parametrize(
    ("stored", "last_update", "message"),
    [
        ([], 1000, "no weight record"),
        ([(2, U16_MAX)], 1001, "differs from the emitted max-grid"),
        ([(1, U16_MAX)], 999, "LastUpdate did not advance"),
        # The same bytes at the pre-submit block prove only that an older write
        # already existed. They must not turn an SDK no-op into a fresh success.
        ([(1, U16_MAX)], 1000, "LastUpdate did not advance"),
    ],
)
async def test_non_cr_sdk_success_requires_exact_fresh_finalized_storage_proof(
    stored, last_update, message
):
    """A claimed success cannot reproduce a false-success incident seen in production."""
    transport = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        weights={0: stored},
        last_update={0: last_update},
        set_result=FakeExtrinsicResponse(True, "SDK says included"),
        store_successful_non_cr_write=False,
    )
    adapter = make_adapter(
        transport,
        weight_readback_attempts=3,
        weight_readback_delay_seconds=0.0,
    )
    adapter.refresh()

    with pytest.raises(OSError, match=message):
        await adapter.set_weights({1: 1.0}, version_key=0)

    # The write's fate is UNKNOWN, never an explicit rejection/success. The
    # weight-setter therefore retains its durable intent for later reconciliation.
    assert adapter._condemned is True
    assert adapter._consecutive_failures >= 1
    assert transport.submitted_weights_calls == 3
    assert len(transport.set_calls) == 1


async def test_commit_reveal_acceptance_stays_pending_until_vector_readback():
    transport = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        commit_reveal=True,
        timelocked_commits={0: ["hk-0"]},
        set_result=FakeExtrinsicResponse(True, "commit finalized"),
    )
    adapter = make_adapter(transport)
    adapter.refresh()

    result = await adapter.set_weights({1: 1.0}, version_key=12)

    assert result.success is False
    assert result.pending_reveal is True
    assert result.submitted == {1: U16_MAX}
    assert "awaiting automatic reveal" in result.message
    assert adapter.commit_reveal_enabled() is True
    assert adapter.weight_commit_pending("hk-0") is True
    assert adapter.commitment_rate_limit() == 7
    with pytest.raises(ChainStateUnavailable, match="pending reveal"):
        adapter.submitted_weights("hk-0")


def test_commitment_capacity_uses_current_epoch_tracker_and_exact_pallet_charge():
    transport = FakeTransport(
        view=make_view(3),
        block=1000,
        epoch_closes={7: 900},
        commitment_max_space=3100,
        commitment_usage_epoch=7,
        commitment_used_space=2875,
    )
    capacity = make_adapter(transport).commitment_capacity(1, "hk-authority")

    assert capacity.block == 1000
    assert capacity.current_epoch == 7
    assert capacity.usage_epoch == 7
    assert capacity.reported_used_space == 2875
    assert capacity.used_space == 2875
    assert capacity.remaining_space == 225
    assert capacity.required_space(0) == 100
    assert capacity.required_space(99) == 100
    assert capacity.required_space(128) == 128
    assert capacity.can_fit(128) is True
    assert capacity.writes_remaining(100) == 2
    assert transport.commitment_usage_calls == [(1, "hk-authority", 1000)]


@pytest.mark.parametrize("usage_epoch", [None, 6])
def test_commitment_capacity_absent_or_stale_tracker_resets_effective_usage(
    usage_epoch,
):
    transport = FakeTransport(
        view=make_view(3),
        block=1000,
        epoch_closes={7: 900},
        commitment_max_space=3100,
        commitment_usage_epoch=usage_epoch,
        commitment_used_space=0 if usage_epoch is None else 3000,
    )
    capacity = make_adapter(transport).commitment_capacity(1, "hk-authority")

    assert capacity.current_epoch == 7
    assert capacity.usage_epoch == usage_epoch
    assert capacity.reported_used_space == (0 if usage_epoch is None else 3000)
    assert capacity.used_space == 0
    assert capacity.remaining_space == 3100
    assert capacity.writes_remaining(128) == 24


def test_commitment_capacity_fails_closed_on_unreadable_or_inconsistent_state():
    unreadable = FakeTransport(view=make_view(3), epoch_closes={7: 900})
    unreadable.raise_on.add("commitment_usage")
    with pytest.raises(ChainStateUnavailable, match="Commitments capacity"):
        make_adapter(unreadable).commitment_capacity(1, "hk-authority")

    overdrawn = FakeTransport(
        view=make_view(3),
        epoch_closes={7: 900},
        commitment_max_space=100,
        commitment_usage_epoch=7,
        commitment_used_space=101,
    )
    with pytest.raises(ChainStateUnavailable, match="exceeds MaxSpace"):
        make_adapter(overdrawn).commitment_capacity(1, "hk-authority")


def test_commitment_capacity_rejects_wrong_subnet_and_empty_account_locally():
    adapter = make_adapter(FakeTransport(view=make_view(3)))
    with pytest.raises(ValueError, match="bound to subnet"):
        adapter.commitment_capacity(85, "hk-authority")
    with pytest.raises(ValueError, match="must be non-empty"):
        adapter.commitment_capacity(1, "")


async def test_unknown_commit_reveal_mode_fails_closed_before_success():
    transport = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        set_result=FakeExtrinsicResponse(True, "accepted"),
        commit_reveal=None,
    )
    adapter = make_adapter(transport)

    with pytest.raises(OSError, match="boolean commit-reveal mode"):
        await adapter.set_weights({1: 1.0}, version_key=12)
    with pytest.raises(ChainStateUnavailable, match="not bool"):
        adapter.commit_reveal_enabled()


async def test_set_weights_unrecognized_result_is_a_transport_failure_not_success():
    # An unrecognized SDK shape must NOT become an implicit success.
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0}, set_result=object())
    adapter = make_adapter(t)
    adapter.refresh()
    with pytest.raises(OSError, match="transport failure"):
        await adapter.set_weights({1: 1.0}, version_key=0)
    assert adapter._consecutive_failures == 1  # the raise counts toward reconnect


async def test_set_weights_refuses_orphan_uid_without_donating_to_survivors():
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})  # uids 0,1,2
    adapter = make_adapter(t)
    adapter.refresh()
    # uid 99 is not in the metagraph. Dropping it would turn uid 1's 50% into
    # 100%, so the exact authority vector must fail before any transport write.
    result = await adapter.set_weights({1: 0.5, 99: 0.5}, version_key=0)
    assert result.success is False
    assert "orphan_uids=[99]" in result.message
    assert result.submitted == {}
    assert t.set_calls == []


async def test_set_weights_refuses_uid_recycled_to_a_different_hotkey():
    # an internal review: epoch uid 2 was hotkey A, but by submission uid 2 is recycled to
    # a NEW hotkey (metagraph now binds uid 2 -> "hk-2"). Reconciling on the
    # (uid, hotkey) BINDING must reject the whole attempt: paying uid 2 would pay
    # the wrong occupant, while dropping it would donate its share to uid 1.
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})  # uid 2 -> "hk-2" now
    adapter = make_adapter(t)
    adapter.refresh()
    result = await adapter.set_weights(
        {1: 0.5, 2: 0.5},
        version_key=0,
        hotkeys={1: "hk-1", 2: "hk-OLD-A"},  # intended: uid 2 was hotkey A
    )
    assert result.success is False
    assert "recycled_uids=[2]" in result.message
    assert result.submitted == {}
    assert t.set_calls == []


async def test_set_weights_keeps_uid_whose_hotkey_binding_still_matches():
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(t)
    adapter.refresh()
    result = await adapter.set_weights(
        {1: 0.5, 2: 0.5},
        version_key=0,
        hotkeys={1: "hk-1", 2: "hk-2"},  # both bindings still hold
    )
    assert result.success is True
    uids, vals, _ = t.set_calls[-1]
    assert uids == [1, 2]
    assert dict(zip(uids, vals, strict=True)) == quantize_u16({1: 0.5, 2: 0.5})
    assert result.submitted == max_normalize_u16(
        quantize_u16({1: 0.5, 2: 0.5})
    )


async def test_set_weights_preserves_exact_authority_sum_grid_arrays():
    """Authenticated u16 pairs cross the adapter without drop or requantization."""
    authority_u16 = {0: 13_107, 1: 34_952, 2: 17_476}
    assert sum(authority_u16.values()) == U16_MAX
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(t)
    adapter.refresh()

    result = await adapter.set_weights(
        {uid: float(value) for uid, value in authority_u16.items()},
        version_key=16,
        hotkeys={0: "hk-0", 1: "hk-1", 2: "hk-2"},
    )

    assert result.success is True
    assert t.set_calls == [([0, 1, 2], [13_107, 34_952, 17_476], 16)]
    assert result.submitted == max_normalize_u16(authority_u16)


async def test_set_weights_rejects_a_partial_binding_map_before_write():
    # Once binding safety is requested, every positive target (including burn)
    # must be named. Falling back to uid liveness can pay a recycled occupant.
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})  # uids 0,1,2 live
    adapter = make_adapter(t)
    adapter.refresh()
    result = await adapter.set_weights(
        {2: 1.0},  # a live uid the binding map does not mention
        version_key=0,
        hotkeys={1: "hk-1"},  # partial binding: uid 2 absent
    )
    assert result.success is False
    assert "incomplete uid/hotkey binding map" in result.message
    assert "2" in result.message
    assert t.set_calls == []


async def test_set_weights_uses_a_fresh_metagraph_for_binding_reconciliation():
    stale = make_view(3)
    t = FakeTransport(view=stale, uid_map={"hk-0": 0})
    adapter = make_adapter(t)
    adapter.refresh()  # cache says uid 2 -> hk-2

    t._view = MetagraphView(
        block=stale.block + 1,
        hotkeys=["hk-0", "hk-1", "hk-NEW"],
        coldkeys=stale.coldkeys,
        axon_ips=stale.axon_ips,
        alpha_stake=stale.alpha_stake,
        emission=stale.emission,
        validator_permit=stale.validator_permit,
        last_update=stale.last_update,
    )
    result = await adapter.set_weights(
        {1: 0.5, 2: 0.5},
        version_key=0,
        hotkeys={1: "hk-1", 2: "hk-2"},
    )

    assert result.success is False
    assert "recycled_uids=[2]" in result.message
    assert result.submitted == {}
    assert t.set_calls == []


async def test_set_weights_fails_cleanly_when_fresh_metagraph_is_unavailable():
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(t)
    adapter.refresh()
    t.raise_on.add("metagraph")

    result = await adapter.set_weights({1: 1.0}, version_key=0, hotkeys={1: "hk-1"})

    assert result.success is False
    assert "fresh metagraph unavailable" in result.message
    assert t.set_calls == []


async def test_set_weights_empty_after_quantization_is_a_clean_failure():
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(t)
    adapter.refresh()
    result = await adapter.set_weights({1: 0.0}, version_key=0)  # no positive weight
    assert result.success is False
    assert not t.set_calls


# --------------------------------------------------------------------------------------
# submitted_weights readback + commit-reveal awareness
# --------------------------------------------------------------------------------------


def test_submitted_weights_returns_raw_u16_with_block():
    t = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        weights={0: [(1, 40000), (2, 25535)]},
        last_update={0: 1234},
    )
    adapter = make_adapter(t)
    got = adapter.submitted_weights("hk-0")
    assert isinstance(got, SubmittedWeights)
    assert got.weights == {1: 40000.0, 2: 25535.0}  # RAW u16, not renormalized
    assert got.block == 1234


def test_submitted_weights_none_when_registered_but_no_weights():
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0}, weights={0: []})
    adapter = make_adapter(t)
    assert adapter.submitted_weights("hk-0") is None  # positive "no weights"


def test_submitted_weights_pending_commit_raises_unknown_not_denied():
    # v10: a timelocked commit keyed (netuid, epoch) is pending for our hotkey while
    # the stale pre-commit Weights are still on chain -> UNKNOWN, never DENIED.
    t = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        weights={0: [(1, 65535)]},  # stale pre-commit vector present...
        timelocked_commits={7: ["hk-0"]},  # ...but a timelocked commit pends...
        epoch=7,  # ...in the current epoch
    )
    adapter = make_adapter(t)
    with pytest.raises(ChainStateUnavailable):
        adapter.submitted_weights("hk-0")


def test_submitted_weights_pending_commit_for_other_epoch_does_not_block():
    # A timelocked commit in a DIFFERENT epoch is not ours-pending-now: read normally.
    t = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        weights={0: [(1, 65535)]},
        timelocked_commits={6: ["hk-0"]},  # previous epoch only
        epoch=7,
    )
    adapter = make_adapter(t)
    got = adapter.submitted_weights("hk-0")
    assert got is not None
    assert got.weights == {1: 65535.0}


def test_submitted_weights_unregistered_hotkey_raises():
    t = FakeTransport(view=make_view(3), uid_map={})  # not registered
    adapter = make_adapter(t)
    with pytest.raises(ChainStateUnavailable):
        adapter.submitted_weights("hk-nope")


def test_submitted_weights_transport_failure_raises_and_counts():
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0}, weights={0: [(1, 1)]})
    t.raise_on = {"query_weights"}
    adapter = make_adapter(t)
    with pytest.raises(ChainStateUnavailable):
        adapter.submitted_weights("hk-0")
    assert adapter._consecutive_failures == 1


def test_submitted_weights_round_trips_a_submitted_vector():
    # Submit -> chain stores our u16 -> read back raw -> weights_match our intent.
    vec = {1: 0.6, 2: 0.4}
    u16 = quantize_u16(vec)
    t = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        weights={0: list(u16.items())},
        last_update={0: 1234},
    )
    adapter = make_adapter(t)
    got = adapter.submitted_weights("hk-0")
    assert got is not None
    assert weights_match(got.weights, vec)


# ------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------


async def test_a_cancelled_slow_submit_isolated_in_an_old_transport_generation():
    # asyncio.to_thread cannot kill A's SDK wait. Reconnect must atomically install
    # a fresh socket for B without closing or waiting on A's still-live generation.
    gate = threading.Event()
    entered = threading.Event()
    state = {"live": 0, "max": 0}
    guard = threading.Lock()

    class SlowTransport(FakeTransport):
        def set_weights(self, *, netuid, uids, weights, version_key):
            self.set_calls.append((list(uids), list(weights), version_key))
            entered.set()
            with guard:
                state["live"] += 1
                state["max"] = max(state["max"], state["live"])
            gate.wait(5)
            with guard:
                state["live"] -= 1
            return (True, "ok", False)

    t = SlowTransport(view=make_view(4), uid_map={"hk-0": 0})
    replacement = FakeTransport(view=make_view(4), uid_map={"hk-0": 0})
    adapter = BittensorChainAdapter(
        BittensorAdapterConfig(validator_hotkey="hk-0", netuid=1),
        transport=t,
        connect_transport=lambda: replacement,
    )
    adapter.refresh()

    # Submit A — let its worker thread enter transport.set_weights and hold the mutex.
    a = asyncio.create_task(adapter.set_weights({1: 1.0}, version_key=0))
    await asyncio.to_thread(entered.wait, 5)
    assert state["live"] == 1

    # The caller's timeout fires: cancel A. Its worker thread lives on.
    a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await a
    assert adapter._condemned is True
    assert t.closed == 0  # the live submit's socket is NOT closed under it

    # Submit B completes on a genuinely separate transport while A is still live.
    b = await asyncio.wait_for(
        adapter.set_weights({1: 1.0}, version_key=0), timeout=0.5
    )
    assert b.success is True
    assert replacement.set_calls
    assert state["live"] == 1
    assert t.closed == 0  # background retirement waits for A, never closes under it
    assert adapter.unreaped_transport_generations

    gate.set()
    for _ in range(100):
        if t.closed == 1 and not adapter.unreaped_transport_generations:
            break
        await asyncio.sleep(0.01)
    assert t.closed == 1
    assert adapter.unreaped_transport_generations == ()


async def test_refresh_and_readback_escape_an_abandoned_submit_worker():
    gate = threading.Event()
    submit_entered = threading.Event()
    refresh_entered = threading.Event()
    readback_entered = threading.Event()

    class SlowTransport(FakeTransport):
        def set_weights(self, *, netuid, uids, weights, version_key):
            submit_entered.set()
            gate.wait(5)
            return (True, "ok", False)

        def metagraph(self, netuid, block_number=None):
            refresh_entered.set()
            return super().metagraph(netuid, block_number)

        def pending_timelocked_commit(self, netuid, hotkey):
            readback_entered.set()
            return super().pending_timelocked_commit(netuid, hotkey)

    clock = [1000.0]
    transport = SlowTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        weights={0: [(1, U16_MAX)]},
        last_update={0: 999},
    )
    replacement = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        weights={0: [(1, U16_MAX)]},
        last_update={0: 999},
    )
    adapter = BittensorChainAdapter(
        BittensorAdapterConfig(
            validator_hotkey="hk-0", netuid=1, metagraph_ttl_seconds=120
        ),
        transport=transport,
        connect_transport=lambda: replacement,
        clock=lambda: clock[0],
    )
    adapter.refresh()
    refresh_entered.clear()
    clock[0] += 121

    submit = asyncio.create_task(adapter.set_weights({1: 1.0}, version_key=0))
    await asyncio.to_thread(submit_entered.wait, 5)
    # The submit's mandatory fresh metagraph legitimately touched the old lane;
    # clear that observation before testing post-timeout reads.
    refresh_entered.clear()
    submit.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submit

    refresh = asyncio.to_thread(adapter.refresh)
    readback = asyncio.to_thread(adapter.submitted_weights, "hk-0")
    _, submitted = await asyncio.wait_for(
        asyncio.gather(refresh, readback), timeout=0.5
    )
    # Neither operation re-entered the abandoned old transport.
    assert not refresh_entered.is_set()
    assert not readback_entered.is_set()
    assert submitted is not None
    assert submitted.weights == {1: float(U16_MAX)}
    assert transport.closed == 0
    assert adapter.unreaped_transport_generations

    gate.set()
    for _ in range(100):
        if transport.closed == 1 and not adapter.unreaped_transport_generations:
            break
        await asyncio.sleep(0.01)
    assert transport.closed == 1


async def test_adapter_close_never_waits_for_a_busy_transport_generation():
    entered = threading.Event()
    release = threading.Event()

    class BusyTransport(FakeTransport):
        def finalized_block(self):
            entered.set()
            release.wait(5)
            return super().finalized_block()

    transport = BusyTransport(view=make_view(3), finalized_block=987)
    adapter = make_adapter(transport)
    live_read = asyncio.create_task(asyncio.to_thread(adapter.finalized_block))
    assert await asyncio.to_thread(entered.wait, 5)

    # Shutdown schedules retirement and returns; it must not wait on the live
    # generation mutex or on transport.close().
    await asyncio.wait_for(asyncio.to_thread(adapter.close), timeout=0.25)
    assert transport.closed == 0
    assert any(
        key.startswith("close-main:") for key in adapter.unreaped_transport_generations
    )

    release.set()
    assert await live_read == 987
    for _ in range(100):
        if transport.closed == 1 and not adapter.unreaped_transport_generations:
            break
        await asyncio.sleep(0.01)
    assert transport.closed == 1
    assert adapter.unreaped_transport_generations == ()


# --------------------------------------------------------------------------------------
# reconnect discipline
# --------------------------------------------------------------------------------------


def test_reconnect_only_after_n_consecutive_raised_failures():
    built: list[FakeTransport] = []

    def factory() -> FakeTransport:
        t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
        # the FIRST socket always raises on metagraph; the reconnect is healthy
        if not built:
            t.raise_on = {"metagraph"}
        built.append(t)
        return t

    adapter = BittensorChainAdapter(
        BittensorAdapterConfig(
            validator_hotkey="hk-0", netuid=1, reconnect_after_consecutive_failures=3
        ),
        connect_transport=factory,
    )
    assert len(built) == 1  # initial socket built via the factory
    for _ in range(3):
        adapter.refresh()  # each raises -> counts, but no reconnect yet
    assert adapter._consecutive_failures == 3
    assert len(built) == 1  # still the original socket
    adapter.refresh()  # entry sees >=3 -> reconnect to a healthy socket
    assert len(built) == 2
    assert built[0].closed == 1  # retired socket was closed
    assert adapter._consecutive_failures == 0  # reset after a clean refresh
    assert len(adapter.neurons()) == 3


def test_condemned_socket_reconnects_before_the_next_call():
    built: list[FakeTransport] = []

    def factory() -> FakeTransport:
        t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
        built.append(t)
        return t

    adapter = BittensorChainAdapter(
        BittensorAdapterConfig(validator_hotkey="hk-0", netuid=1),
        connect_transport=factory,
    )
    adapter.refresh()
    assert len(built) == 1
    adapter._condemned = True  # simulate a fired set_weights timeout
    # any call that actually touches the transport must reconnect first
    adapter.submitted_weights("hk-0")
    assert len(built) == 2
    assert adapter._condemned is False


# --------------------------------------------------------------------------------------
# anchor_commitment
# --------------------------------------------------------------------------------------


async def test_anchor_commitment_writes_and_rejects_oversize():
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(t)
    txid = await adapter.anchor_commitment(b"vidaio-anchor")
    assert txid == "0xdeadbeef"
    # v10.5.0: the transport received a STR (the adapter decoded the ascii bytes),
    # never bytes — passing bytes to set_commitment(data=...) raises before submission.
    assert t.commit_calls == ["vidaio-anchor"]
    assert all(isinstance(p, str) for p in t.commit_calls)
    with pytest.raises(ValueError):
        await adapter.anchor_commitment(b"x" * 129)


async def test_anchor_txid_is_the_extrinsic_receipt_hash_not_the_message():
    # an internal review tail: with a v10 ExtrinsicResponse, the anchor txid must be the
    # extrinsic RECEIPT hash, NOT the response message.
    t = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        commit_result=FakeExtrinsicResponse(
            True, "committed at block 42", FakeReceipt("0xRECEIPTHASH")
        ),
    )
    adapter = make_adapter(t)
    txid = await adapter.anchor_commitment(b"vidaio-anchor")
    assert txid == "0xRECEIPTHASH"


async def test_anchor_commitment_rejection_object_raises():
    t = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        commit_result=FakeExtrinsicResponse(False, "commitment rate-limited"),
    )
    adapter = make_adapter(t)
    with pytest.raises(OSError):
        await adapter.anchor_commitment(b"vidaio-anchor")


async def test_cancelled_anchor_wait_does_not_starve_chain_reads_and_reconnects():
    entered = threading.Event()
    release = threading.Event()

    class SlowAnchorTransport(FakeTransport):
        def set_commitment(self, *, netuid, payload):
            entered.set()
            release.wait(5)
            return super().set_commitment(netuid=netuid, payload=payload)

    main = FakeTransport(view=make_view(3), uid_map={"hk-0": 0}, finalized_block=987)
    slow = SlowAnchorTransport(view=make_view(3), uid_map={"hk-0": 0})
    replacement = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    transports = iter((slow, replacement))
    adapter = BittensorChainAdapter(
        BittensorAdapterConfig(validator_hotkey="hk-0", netuid=1),
        transport=main,
        connect_transport=lambda: next(transports),
    )

    anchor = asyncio.create_task(adapter.anchor_commitment(b"first-anchor"))
    assert await asyncio.to_thread(entered.wait, 5)

    # The commitment SDK wait is synchronous and cannot be killed. It runs on a
    # distinct socket, so an ordinary read remains responsive while it is stuck.
    assert (
        await asyncio.wait_for(asyncio.to_thread(adapter.finalized_block), timeout=0.25)
        == 987
    )
    anchor.cancel()
    await asyncio.sleep(0)
    assert (
        not anchor.done()
    )  # cancellation holds the mutable-slot lane until worker exit
    assert adapter._anchor_condemned is True
    assert main.closed == 0

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await anchor

    # The next write retires only the dedicated commitment socket and succeeds;
    # the main read socket was neither closed nor reused by the abandoned worker.
    assert await adapter.anchor_commitment(b"second-anchor") == "0xdeadbeef"
    assert slow.closed == 1
    assert replacement.commit_calls == ["second-anchor"]
    assert main.closed == 0


async def test_anchor_reconnect_does_not_wait_for_a_wedged_real_close(monkeypatch):
    """A timed-out real SDK generation is swapped before its close is attempted."""
    close_entered = threading.Event()
    release_close = threading.Event()

    class TimedOutAnchorTransport(FakeTransport):
        def set_commitment(self, *, netuid, payload):  # noqa: ARG002
            raise TimeoutError("SDK RPC timed out")

        def close(self):
            close_entered.set()
            release_close.wait(5)
            self.closed += 1

    # Mark this injected type as the real transport class for the retirement
    # branch; no SDK/network import or connection occurs.
    import vidaio.chain.bittensor_adapter as adapter_module

    monkeypatch.setattr(
        adapter_module, "_RealSubtensorTransport", TimedOutAnchorTransport
    )
    main = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    timed_out = TimedOutAnchorTransport(view=make_view(3), uid_map={"hk-0": 0})
    recovered = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    transports = iter((timed_out, recovered))
    adapter = BittensorChainAdapter(
        BittensorAdapterConfig(validator_hotkey="hk-0", netuid=1),
        transport=main,
        connect_transport=lambda: next(transports),
    )

    with pytest.raises(TimeoutError, match="SDK RPC timed out"):
        await adapter.anchor_commitment(b"first")

    # The next anchor uses the fresh generation even though retiring the old
    # generation has entered a close() that cannot return yet.
    assert (
        await asyncio.wait_for(adapter.anchor_commitment(b"second"), timeout=0.5)
        == "0xdeadbeef"
    )
    assert close_entered.is_set()
    assert timed_out.closed == 0
    assert recovered.commit_calls == ["second"]
    assert any(
        key.startswith("anchor:") for key in adapter.unreaped_transport_generations
    )

    release_close.set()
    for _ in range(100):
        if timed_out.closed == 1 and not adapter.unreaped_transport_generations:
            break
        await asyncio.sleep(0.01)
    assert timed_out.closed == 1
    assert adapter.unreaped_transport_generations == ()


async def test_anchor_failures_reconnect_only_the_dedicated_commitment_socket():
    main = FakeTransport(view=make_view(3), uid_map={"hk-0": 0}, finalized_block=987)
    failed = FakeTransport(
        view=make_view(3), uid_map={"hk-0": 0}, commit_result=OSError("socket down")
    )
    recovered = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    transports = iter((failed, recovered))
    adapter = BittensorChainAdapter(
        BittensorAdapterConfig(
            validator_hotkey="hk-0",
            netuid=1,
            reconnect_after_consecutive_failures=3,
        ),
        transport=main,
        connect_transport=lambda: next(transports),
    )

    for _ in range(3):
        with pytest.raises(OSError, match="socket down"):
            await adapter.anchor_commitment(b"anchor")
    assert adapter._anchor_consecutive_failures == 3
    assert adapter._consecutive_failures == 0
    assert main.closed == 0

    assert await adapter.anchor_commitment(b"anchor") == "0xdeadbeef"
    assert failed.closed == 1
    assert recovered.commit_calls == ["anchor"]
    assert adapter._anchor_consecutive_failures == 0
    assert main.closed == 0


# --------------------------------------------------------------------------------------
# set_commitment passes a STR (v10.5.0) + anchor read round-trips
# --------------------------------------------------------------------------------------


async def test_set_commitment_passes_a_str_and_read_anchor_round_trips():
    # v10.5.0 set_commitment(data: str) does data.encode() itself; the adapter must
    # hand the transport a STR, and read_anchor must decode the same ascii bytes back
    # to the SAME digest (str in -> str out -> the same digest, the third leg #3).
    netuid, epoch_id, digest = 1, 42, "a" * 64
    domain = "vidaio.epoch.anchor.v1"
    payload = f"{domain}:{netuid}:{epoch_id}:{digest}".encode("ascii")
    assert len(payload) <= 128  # the anchor fits the commitment budget

    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(t)  # netuid=1, validator_hotkey hk-0 (anchor self-account)

    txid = await adapter.anchor_commitment(payload)
    assert txid == "0xdeadbeef"
    assert t.commit_calls == [payload.decode("ascii")]  # a STR reached the transport
    assert all(isinstance(p, str) for p in t.commit_calls)

    got = adapter.read_anchor(netuid=netuid, epoch_id=epoch_id, domain=domain)
    assert got == digest  # the anchored digest reads straight back


async def test_read_anchor_block_returns_the_inclusion_block():
    """an internal review (step 3): read_anchor_block returns the block the epoch's anchor
    landed at — a POSITIVE None before any anchor, and None for a different epoch."""
    netuid, epoch_id, digest = 1, 42, "a" * 64
    domain = "vidaio.epoch.anchor.v1"
    payload = f"{domain}:{netuid}:{epoch_id}:{digest}".encode("ascii")

    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0}, block=1234)
    adapter = make_adapter(t)  # netuid=1, validator_hotkey hk-0 (anchor self-account)

    # No commitment yet -> POSITIVE None.
    assert (
        adapter.read_anchor_block(netuid=netuid, epoch_id=epoch_id, domain=domain)
        is None
    )

    await adapter.anchor_commitment(payload)  # lands at block 1234
    assert (
        adapter.read_anchor_block(netuid=netuid, epoch_id=epoch_id, domain=domain)
        == 1234
    )
    # read_anchor still returns the anchored digest (the finding-#4 tamper leg).
    assert (
        adapter.read_anchor(netuid=netuid, epoch_id=epoch_id, domain=domain) == digest
    )

    # A DIFFERENT epoch cannot be disproved from this single-slot head state: the
    # requested historical anchor may have been overwritten. Fail closed / HOLD.
    with pytest.raises(ChainStateUnavailable, match="single current slot"):
        adapter.read_anchor_block(netuid=netuid, epoch_id=999, domain=domain)


def test_read_anchor_at_recovers_overwritten_slot_from_exact_archive_block():
    domain, netuid = "vidaio.epoch.anchor.v1", 1
    old_digest, new_digest = "a" * 64, "b" * 64
    old_payload = f"{domain}:{netuid}:41:{old_digest}"
    new_payload = f"{domain}:{netuid}:42:{new_digest}"
    transport = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        commitment_history={
            1100: (old_payload, 1100),
            # At a later block the same old record remains in the slot. It must
            # not authenticate 1200 as that record's inclusion block.
            1200: (old_payload, 1100),
            1300: (new_payload, 1300),
        },
    )
    transport._committed = new_payload  # head state has overwritten epoch 41
    transport._commit_block = 1300
    adapter = make_adapter(transport)

    assert isinstance(adapter, HistoricalEpochAnchorReadable)
    with pytest.raises(ChainStateUnavailable, match="single current slot"):
        adapter.read_anchor(netuid=netuid, epoch_id=41, domain=domain)
    assert (
        adapter.read_anchor_at(
            netuid=netuid, epoch_id=41, domain=domain, block_number=1100
        )
        == old_digest
    )
    assert transport.commitment_read_calls[-2:] == [
        ("payload", 1100),
        ("block", 1100),
    ]

    # Exact archive state is authoritative: a nonmatching payload or an older
    # record merely carried forward to this height is a definitive absence.
    assert (
        adapter.read_anchor_at(
            netuid=netuid, epoch_id=41, domain=domain, block_number=1200
        )
        is None
    )
    assert (
        adapter.read_anchor_at(
            netuid=netuid, epoch_id=41, domain=domain, block_number=1300
        )
        is None
    )


def test_raw_commitment_record_preserves_v1_payload_and_original_archive_block():
    old_payload = "vidaio.commitment.v1:competition:" + "a" * 64
    new_payload = "vidaio.commitment.v1:competition:" + "b" * 64
    transport = FakeTransport(
        view=make_view(3),
        uid_map={"hk-0": 0},
        commitment_history={
            1100: (old_payload, 1100),
            1200: (old_payload, 1100),
            1300: (new_payload, 1300),
        },
    )
    transport._committed = new_payload
    transport._commit_block = 1300
    adapter = make_adapter(transport)

    assert isinstance(adapter, CommitmentRecordReadable)
    head = adapter.read_commitment_record(netuid=1)
    assert head is not None
    assert head.payload == new_payload.encode("ascii")
    assert head.block == 1300

    included = adapter.read_commitment_record(netuid=1, block_number=1100)
    assert included is not None
    assert included.payload == old_payload.encode("ascii")
    assert included.block == 1100

    carried = adapter.read_commitment_record(netuid=1, block_number=1200)
    assert carried is not None
    assert carried.payload == old_payload.encode("ascii")
    assert carried.block == 1100


def test_read_anchor_at_empty_or_unreadable_archive_state_is_fail_closed():
    domain = "vidaio.epoch.anchor.v1"
    transport = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(transport)
    assert (
        adapter.read_anchor_at(netuid=1, epoch_id=41, domain=domain, block_number=1100)
        is None
    )

    transport.raise_on = {"get_commitment"}
    with pytest.raises(ChainStateUnavailable, match="at block 1100"):
        adapter.read_anchor_at(netuid=1, epoch_id=41, domain=domain, block_number=1100)


def test_every_anchor_reader_rejects_a_foreign_netuid_before_rpc():
    transport = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(transport)
    calls = (
        lambda: adapter.read_anchor(netuid=85, epoch_id=1, domain="domain"),
        lambda: adapter.read_anchor_at(
            netuid=85, epoch_id=1, domain="domain", block_number=100
        ),
        lambda: adapter.read_anchor_block(netuid=85, epoch_id=1, domain="domain"),
        lambda: adapter.read_commitment_record(netuid=85),
    )
    for call in calls:
        with pytest.raises(ValueError, match="bound to subnet 1, not 85"):
            call()
    assert transport.commitment_read_calls == []


async def test_block_hash_returns_the_substrate_hash_0x_stripped():
    """an internal review: block_hash(n) returns the real substrate hash, 0x-stripped and
    lowercased, for the round-6 beacon block_hash(close_block + K)."""
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0}, block=1234)
    adapter = make_adapter(t)
    h = adapter.block_hash(41)
    assert (
        h == f"{41:064x}"
    )  # the transport's 0x-prefixed hash, 0x-stripped + lowercased
    assert len(h) == 64


async def test_read_anchor_block_raises_when_the_block_read_fails():
    """The commitment exists but its inclusion block is unreadable => HOLD (raise), never a
    substituted None (which would read as 'no anchor')."""
    from vidaio.chain.adapter import ChainStateUnavailable

    netuid, epoch_id, digest = 1, 42, "a" * 64
    domain = "vidaio.epoch.anchor.v1"
    payload = f"{domain}:{netuid}:{epoch_id}:{digest}".encode("ascii")

    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0}, block=1234)
    adapter = make_adapter(t)
    await adapter.anchor_commitment(payload)
    t.raise_on = {"get_commitment_block"}  # the inclusion-block read now fails
    with pytest.raises(ChainStateUnavailable):
        adapter.read_anchor_block(netuid=netuid, epoch_id=epoch_id, domain=domain)


async def test_block_hash_raises_when_the_read_fails():
    """An unreadable block-hash RPC => HOLD (raise ChainStateUnavailable)."""
    from vidaio.chain.adapter import ChainStateUnavailable

    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0}, block=1234)
    adapter = make_adapter(t)
    t.raise_on = {"get_block_hash"}
    with pytest.raises(ChainStateUnavailable):
        adapter.block_hash(41)


async def test_anchor_non_ascii_payload_is_caught_at_our_boundary():
    # We now ALWAYS pass a str: a payload that is not decodable ascii is caught at
    # the adapter boundary (never reaching the transport / SDK mid-extrinsic).
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(t)
    with pytest.raises(OSError):
        await adapter.anchor_commitment(b"\xff\xfe not ascii")
    assert t.commit_calls == []  # the extrinsic was never submitted


def test_fake_transport_set_commitment_requires_str_like_v10():
    # The v10.5.0 SDK raises on bytes (it does data.encode()); the fake models that
    # strictly, so a regression handing the transport bytes is caught immediately.
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    with pytest.raises(TypeError):
        t.set_commitment(netuid=1, payload=b"bytes-not-str")
    assert t.set_commitment(netuid=1, payload="ok") == "0xdeadbeef"


# --------------------------------------------------------------------------------------
# v10.5.0 CRv4 storage pagination
# --------------------------------------------------------------------------------------


class FakeSubtensor:
    """Small SDK stand-in used by the private real-transport boundary tests."""

    def __init__(self, *, commits, block: int = 1000, tempo: int = 100) -> None:
        self._commits = commits
        self._block = block
        self._tempo = tempo
        self.timelocked_calls: list[tuple[int, int, int | None]] = []

    def get_current_block(self) -> int:
        return self._block

    def tempo(self, netuid: int) -> int:
        return self._tempo

    def get_timelocked_weight_commits(self, netuid, mechid=0, block=None):
        self.timelocked_calls.append((netuid, mechid, block))
        return self._commits


class FakeSubstrate:
    def __init__(self, *, timelocked_rows=()):
        self.timelocked_rows = list(timelocked_rows)
        self.timelocked_calls = []

    def get_chain_head(self):
        return "0xhead"

    def get_chain_finalised_head(self):
        return "0xfinalized"

    def get_block_number(self, block_hash=None):
        assert block_hash == "0xfinalized"
        return 999

    def get_block_hash(self, block_id):
        return f"0x{block_id:064x}"

    def query(
        self, module, storage_function, params=None, block_hash=None
    ):  # pragma: no cover - specialized capacity fakes implement this
        raise AssertionError(f"unexpected storage query {module}.{storage_function}")

    def query_map(
        self,
        module,
        storage_function,
        params=None,
        block_hash=None,
        max_results=None,
        start_key=None,
        page_size=100,
        ignore_decoding_errors=False,
    ):
        self.timelocked_calls.append(
            {
                "module": module,
                "storage_function": storage_function,
                "params": list(params or []),
                "block_hash": block_hash,
                "page_size": page_size,
                "ignore_decoding_errors": ignore_decoding_errors,
            }
        )
        return iter(self.timelocked_rows)


def _real_transport_with(
    subtensor: FakeSubtensor, *, substrate=None
) -> _RealSubtensorTransport:
    # Build the real transport WITHOUT its bittensor-importing __init__ (bittensor is
    # not installed): only _subtensor/_config are needed for pending_timelocked_commit.
    transport = object.__new__(_RealSubtensorTransport)
    transport._subtensor = subtensor
    transport._config = BittensorAdapterConfig(validator_hotkey="hk-0", netuid=1)
    transport._wallet = object()
    transport._sdk_lock = threading.RLock()
    transport._substrate = substrate or FakeSubstrate()
    return transport


def test_pending_timelocked_commit_exhausts_all_epoch_buckets_at_pinned_head():
    class ScaleValue:
        def __init__(self, value):
            self.value = value

    class PagedRows:
        """Model query_map: .records is page one; iteration exhausts all pages."""

        def __init__(self):
            self.records = [(ScaleValue(40), ScaleValue([("hk-9", 900, b"a", 1)]))]
            self.iterated = False

        def __iter__(self):
            self.iterated = True
            yield from self.records
            # Our commit is deliberately NOT in .records/page one. The pinned SDK
            # getter reads only records[0] and therefore returns a false negative.
            yield ScaleValue(41), ScaleValue([("hk-0", 999, b"b", 2)])

    rows = PagedRows()

    class PagedSubstrate(FakeSubstrate):
        def query_map(self, **kwargs):
            self.timelocked_calls.append(kwargs)
            return rows

    substrate = PagedSubstrate()
    sub = FakeSubtensor(commits=[])  # its broken convenience getter is not consulted
    transport = _real_transport_with(sub, substrate=substrate)
    assert transport.pending_timelocked_commit(1, "hk-0") is True
    assert rows.iterated is True
    assert sub.timelocked_calls == []
    assert substrate.timelocked_calls == [
        {
            "module": "SubtensorModule",
            "storage_function": "TimelockedWeightCommits",
            "params": [1],
            "block_hash": "0xhead",
            "page_size": 100,
            "ignore_decoding_errors": False,
        }
    ]


def test_pending_timelocked_commit_false_when_hotkey_absent():
    substrate = FakeSubstrate(
        timelocked_rows=[
            (40, [("hk-9", 900, b"a", 1)]),
            (41, [("hk-8", 999, b"b", 2)]),
        ]
    )
    transport = _real_transport_with(FakeSubtensor(commits=[]), substrate=substrate)
    assert transport.pending_timelocked_commit(1, "hk-0") is False


def test_pending_timelocked_commit_finds_previous_bucket_after_epoch_rollover():
    # The runtime can tag a fire-block commit to the next epoch, and reconciliation
    # can run after the head advances again. Presence in ANY live bucket must HOLD.
    substrate = FakeSubstrate(
        timelocked_rows=[
            (42, [("hk-0", 1000, b"c", 123)]),
            (43, [("hk-9", 1001, b"d", 124)]),
        ]
    )
    transport = _real_transport_with(FakeSubtensor(commits=[]), substrate=substrate)
    assert transport.pending_timelocked_commit(1, "hk-0") is True


def test_pending_timelocked_commit_matches_live_raw_account_id_shape():
    # async-substrate-interface 2.2.1 has returned AccountId32 from this storage
    # as ((b0, ..., b31),) in live production. Comparing str(row[0]) to
    # SS58 would silently miss our commit.
    validator_account = bytes(range(32))
    other_account = bytes(reversed(range(32)))
    substrate = FakeSubstrate(
        timelocked_rows=[
            (
                42,
                [
                    ((tuple(other_account),), 999, b"a", 1),
                    ((tuple(validator_account),), 1000, b"b", 2),
                ],
            )
        ]
    )
    transport = _real_transport_with(FakeSubtensor(commits=[]), substrate=substrate)
    transport._hotkey = types.SimpleNamespace(
        ss58_address="hk-0", public_key=validator_account
    )
    assert transport.pending_timelocked_commit(1, "hk-0") is True


def test_pending_timelocked_commit_raw_account_without_decoder_is_unknown():
    substrate = FakeSubstrate(
        timelocked_rows=[(42, [((tuple(range(32)),), 1000, b"b", 2)])]
    )
    transport = _real_transport_with(FakeSubtensor(commits=[]), substrate=substrate)
    with pytest.raises(TypeError, match="could not be decoded"):
        transport.pending_timelocked_commit(1, "hk-0")


def test_pending_probe_does_not_use_sdk_getter_or_derive_epoch_from_tempo():
    class ExactV105Subtensor(FakeSubtensor):
        def get_current_block(self):
            raise AssertionError("raw prefix scan pins substrate head directly")

        def tempo(self, netuid):
            raise AssertionError("raw prefix scan must not derive runtime epochs")

        def get_timelocked_weight_commits(self, netuid, mechid=0, block=None):
            raise AssertionError("the pinned SDK getter truncates query-map pagination")

    substrate = FakeSubstrate(
        timelocked_rows=[(999, [("hk-0", 999, "0xcommit", 123456)])]
    )
    sub = ExactV105Subtensor(commits=[], block=1000, tempo=100)
    assert (
        _real_transport_with(sub, substrate=substrate).pending_timelocked_commit(
            1, "hk-0"
        )
        is True
    )


def test_hotkey_in_timelocked_commits_liberal_shapes():
    # dict keyed by hotkey, flat list of hotkeys, list of (hotkey, ...) records, and
    # objects with a `.hotkey` — all recognized; the epoch key never collides.
    class _Rec:
        hotkey = "hk-0"

    assert _hotkey_in_timelocked_commits({"hk-0": [1]}, "hk-0") is True
    assert _hotkey_in_timelocked_commits({"hk-1": [1]}, "hk-0") is False
    assert _hotkey_in_timelocked_commits(["hk-0"], "hk-0") is True
    assert _hotkey_in_timelocked_commits([("hk-0", b"c", 1)], "hk-0") is True
    assert _hotkey_in_timelocked_commits([_Rec()], "hk-0") is True
    assert _hotkey_in_timelocked_commits([], "hk-0") is False


@pytest.mark.parametrize(
    "rows",
    [
        [object()],
        [(True, [])],
        [(1, object())],
        [(1, [object()])],
        [(-1, [])],
    ],
)
def test_pending_timelocked_commit_fails_closed_on_unknown_storage_shapes(rows):
    transport = _real_transport_with(
        FakeSubtensor(commits=[]), substrate=FakeSubstrate(timelocked_rows=rows)
    )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        transport.pending_timelocked_commit(1, "hk-0")


def test_pending_timelocked_commit_fails_closed_on_paging_error():
    class BrokenSubstrate(FakeSubstrate):
        def query_map(self, **kwargs):
            def rows():
                yield 1, []
                raise OSError("page two unavailable")

            return rows()

    transport = _real_transport_with(
        FakeSubtensor(commits=[]), substrate=BrokenSubstrate()
    )
    with pytest.raises(OSError, match="page two unavailable"):
        transport.pending_timelocked_commit(1, "hk-0")


# --------------------------------------------------------------------------------------
# pinned v10.5 SDK boundary contract
# --------------------------------------------------------------------------------------


def test_sdk_caught_transport_error_stays_ambiguous_but_rejections_do_not():
    socket_error = OSError("websocket closed after submit")
    ambiguous = _parse_chain_result(
        FakeExtrinsicResponse(False, "socket failed", error=socket_error)
    )
    assert ambiguous.transport_error is socket_error

    dispatch = _parse_chain_result(
        FakeExtrinsicResponse(
            False,
            "SettingWeightsTooFast",
            error={"name": "SettingWeightsTooFast"},
        )
    )
    assert dispatch.success is False
    assert dispatch.transport_error is None

    pool_rejection = _parse_chain_result(
        FakeExtrinsicResponse(
            False,
            "Transaction is already imported",
            error=RuntimeError("Transaction is already imported"),
        )
    )
    assert pool_rejection.transport_error is None


def test_real_transport_set_weights_disables_mev_and_preserves_error_boundary():
    class WriteSdk:
        def __init__(self, response):
            self.response = response
            self.calls = []

        def set_weights(self, **kwargs):
            self.calls.append(kwargs)
            return self.response

        def commit_reveal_enabled(self, *, netuid):
            return False

    socket_error = OSError("websocket closed")
    sdk = WriteSdk(FakeExtrinsicResponse(False, "lost", error=socket_error))
    transport = _real_transport_with(sdk)
    with pytest.raises(OSError, match="websocket closed"):
        transport.set_weights(netuid=1, uids=[2], weights=[U16_MAX], version_key=12)
    assert sdk.calls[-1]["mev_protection"] is False
    assert sdk.calls[-1]["raise_error"] is False
    assert sdk.calls[-1]["wait_for_inclusion"] is True
    assert sdk.calls[-1]["wait_for_finalization"] is True
    assert sdk.calls[-1]["wait_for_revealed_execution"] is True
    assert sdk.calls[-1]["commit_reveal_version"] == 4
    assert sdk.calls[-1]["max_attempts"] == 1
    assert sdk.calls[-1]["version_key"] == 12

    sdk.response = FakeExtrinsicResponse(
        False, "Unregistered hotkey", error={"name": "HotKeyNotRegisteredInSubNet"}
    )
    assert transport.set_weights(
        netuid=1, uids=[2], weights=[U16_MAX], version_key=12
    ) == (False, "Unregistered hotkey", False)

    # Pinned v10.5 returns a default ExtrinsicResponse(False) with no message
    # or error when its internal rate-limit precheck skips the write loop. This
    # is narrowly normalized here; no extrinsic was attempted.
    sdk.response = FakeExtrinsicResponse(False)
    assert transport.set_weights(
        netuid=1, uids=[2], weights=[U16_MAX], version_key=12
    ) == (
        False,
        "tempo gate: pinned SDK rate-limit precheck made no set_weights attempt",
        False,
    )


def test_real_transport_weight_proof_pins_all_storage_to_one_finalized_hash():
    class ScaleValue:
        def __init__(self, value):
            self.value = value

    class FinalizedStorage(FakeSubstrate):
        def __init__(self):
            super().__init__()
            self.queries = []

        def get_chain_finalised_head(self):
            return "0xfinalized-proof"

        def get_block_number(self, block_hash=None):
            assert block_hash == "0xfinalized-proof"
            return 1200

        def query(self, module, storage_function, params=None, block_hash=None):
            self.queries.append(
                (module, storage_function, list(params or []), block_hash)
            )
            assert module == "SubtensorModule"
            if storage_function == "Uids":
                return ScaleValue(0)
            if storage_function == "Weights":
                return ScaleValue([(2, U16_MAX), (3, 32768)])
            if storage_function == "LastUpdate":
                return ScaleValue([1199])
            raise AssertionError(storage_function)

    substrate = FinalizedStorage()
    transport = _real_transport_with(FakeSubtensor(commits=[]), substrate=substrate)

    report = transport.submitted_weights_at_finalized_head(1, "hk-0")

    assert report == SubmittedWeights(weights={2: 65535.0, 3: 32768.0}, block=1199)
    assert substrate.queries == [
        ("SubtensorModule", "Uids", [1, "hk-0"], "0xfinalized-proof"),
        ("SubtensorModule", "Weights", [1, 0], "0xfinalized-proof"),
        ("SubtensorModule", "LastUpdate", [1], "0xfinalized-proof"),
    ]


def test_real_transport_historical_commitment_uses_exact_sdk_block_kwarg():
    class MetadataSdk:
        def __init__(self):
            self.value = "anchor-ascii"
            self.calls = []

        def get_commitment_metadata(self, *, netuid, hotkey_ss58, block=None):
            self.calls.append(
                {"netuid": netuid, "hotkey_ss58": hotkey_ss58, "block": block}
            )
            return self.value

    sdk = MetadataSdk()
    transport = _real_transport_with(sdk)
    archive_checks = []

    class ArchiveSubstrate:
        def get_block_hash(self, block_number):
            archive_checks.append(block_number)
            return "0xarchive"

    transport._substrate = ArchiveSubstrate()
    assert (
        transport.get_commitment(
            netuid=85, ss58="authority-hotkey", block_number=123456
        )
        == b"anchor-ascii"
    )
    sdk.value = {"block": 123456}
    assert (
        transport.get_commitment_block(
            netuid=85, ss58="authority-hotkey", block_number=123456
        )
        == 123456
    )
    assert sdk.calls == [
        {"netuid": 85, "hotkey_ss58": "authority-hotkey", "block": 123456},
        {"netuid": 85, "hotkey_ss58": "authority-hotkey", "block": 123456},
    ]
    assert archive_checks == [123456, 123456]


def test_real_transport_historical_commitment_refuses_pruned_head_fallback():
    class MetadataSdk:
        def get_commitment_metadata(self, *, netuid, hotkey_ss58, block=None):
            raise AssertionError("must not silently query head")

    class PrunedSubstrate:
        def get_block_hash(self, block_number):
            return None

    transport = _real_transport_with(MetadataSdk())
    transport._substrate = PrunedSubstrate()
    with pytest.raises(LookupError, match="requires archive state"):
        transport.get_commitment(
            netuid=85, ss58="authority-hotkey", block_number=123456
        )


def test_real_transport_commitment_usage_reads_both_storage_items_at_exact_hash():
    class ScaleValue:
        def __init__(self, value):
            self.value = value

    class CapacitySubstrate:
        def __init__(self):
            self.calls = []

        def get_block_hash(self, block_number):
            assert block_number == 123456
            return "0xarchive"

        def query(self, module, storage_function, params=None, block_hash=None):
            self.calls.append(
                (module, storage_function, list(params or []), block_hash)
            )
            if storage_function == "MaxSpace":
                return ScaleValue(3100)
            if storage_function == "UsedSpaceOf":
                return ScaleValue({"last_epoch": 42, "used_space": 777})
            raise AssertionError(storage_function)

    substrate = CapacitySubstrate()
    transport = _real_transport_with(FakeSubtensor(commits=[]))
    transport._substrate = substrate

    assert transport.commitment_usage(
        netuid=85, ss58="authority-hotkey", block_number=123456
    ) == _CommitmentUsageView(
        block=123456,
        max_space=3100,
        usage_epoch=42,
        used_space=777,
    )
    assert substrate.calls == [
        ("Commitments", "MaxSpace", [], "0xarchive"),
        (
            "Commitments",
            "UsedSpaceOf",
            [85, "authority-hotkey"],
            "0xarchive",
        ),
    ]


def test_real_transport_commitment_usage_accepts_only_none_as_missing_tracker():
    class ScaleValue:
        def __init__(self, value):
            self.value = value

    class CapacitySubstrate:
        def __init__(self, usage):
            self.usage = usage

        def get_block_hash(self, block_number):
            return "0xarchive"

        def query(self, module, storage_function, params=None, block_hash=None):
            return ScaleValue(3100 if storage_function == "MaxSpace" else self.usage)

    transport = _real_transport_with(FakeSubtensor(commits=[]))
    transport._substrate = CapacitySubstrate(None)
    assert transport.commitment_usage(
        netuid=85, ss58="authority-hotkey", block_number=9
    ) == _CommitmentUsageView(
        block=9,
        max_space=3100,
        usage_epoch=None,
        used_space=0,
    )

    for malformed in ({}, (), {"last_epoch": 1}, {"used_space": 1}, ""):
        transport._substrate = CapacitySubstrate(malformed)
        with pytest.raises(TypeError, match="UsageTracker"):
            transport.commitment_usage(
                netuid=85, ss58="authority-hotkey", block_number=9
            )


def test_real_transport_commitment_usage_fails_closed_without_archive_hash():
    class PrunedSubstrate:
        def get_block_hash(self, block_number):
            return None

        def query(self, **kwargs):
            raise AssertionError("must not silently query head")

    transport = _real_transport_with(FakeSubtensor(commits=[]))
    transport._substrate = PrunedSubstrate()
    with pytest.raises(LookupError, match="requires archive state"):
        transport.commitment_usage(
            netuid=85, ss58="authority-hotkey", block_number=123456
        )


def test_real_transport_block_time_uses_timestamp_at_exact_height():
    expected = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)

    class TimestampSdk:
        def __init__(self):
            self.calls = []

        def get_timestamp(self, *, block=None):
            self.calls.append(block)
            return expected

    class Substrate:
        def __init__(self):
            self.hashes = {99: "0xabc"}
            self.calls = []

        def get_block_hash(self, block_number):
            self.calls.append(block_number)
            return self.hashes.get(block_number)

    sdk = TimestampSdk()
    substrate = Substrate()
    transport = _real_transport_with(sdk)
    transport._substrate = substrate

    assert transport.block_time(99) == expected
    assert sdk.calls == [99]
    assert transport.block_time(100) is None
    assert sdk.calls == [99]  # a future/nonexistent block never reads Timestamp.Now
    assert substrate.calls == [99, 100]


def test_real_transport_finality_and_epoch_schedule_are_exactly_pinned() -> None:
    class Schedule:
        current_block = 333
        last_epoch_block = 333
        pending_epoch_at = 0
        subnet_epoch_index = 42
        tempo = 100
        blocks_since_last_step = 0

    class EpochSdk:
        def __init__(self):
            self.index_calls = []
            self.schedule_calls = []

        def get_subnet_epoch_index(self, *, netuid, block=None):
            self.index_calls.append((netuid, block))
            return 42

        def get_epoch_schedule_state(self, *, netuid, block=None):
            self.schedule_calls.append((netuid, block))
            return Schedule()

    class Substrate:
        def __init__(self):
            self.finalized_hash_calls = 0
            self.number_calls = []
            self.hash_calls = []

        def get_chain_finalised_head(self):
            self.finalized_hash_calls += 1
            return "0xfinal"

        def get_block_number(self, block_hash=None):
            self.number_calls.append(block_hash)
            return 350

        def get_block_hash(self, block_number):
            self.hash_calls.append(block_number)
            return "0xarchive" if block_number == 333 else None

    sdk = EpochSdk()
    substrate = Substrate()
    transport = _real_transport_with(sdk)
    transport._substrate = substrate

    assert transport.finalized_block() == 350
    assert transport.epoch_index(85, 333) == 42
    assert transport.epoch_schedule(85, 333) == EpochScheduleView(
        block=333,
        last_epoch_block=333,
        pending_epoch_at=0,
        subnet_epoch_index=42,
        tempo=100,
        blocks_since_last_step=0,
    )
    assert substrate.finalized_hash_calls == 1
    assert substrate.number_calls == ["0xfinal"]
    assert substrate.hash_calls == [333, 333]
    assert sdk.index_calls == [(85, 333)]
    assert sdk.schedule_calls == [(85, 333)]


def test_real_transport_epoch_reads_refuse_pruned_head_fallback() -> None:
    class EpochSdk:
        def get_subnet_epoch_index(self, *, netuid, block=None):
            raise AssertionError("must not silently query head")

    class PrunedSubstrate(FakeSubstrate):
        def get_block_hash(self, block_number):
            return None

    transport = _real_transport_with(EpochSdk())
    transport._substrate = PrunedSubstrate()
    with pytest.raises(LookupError, match="requires archive state"):
        transport.epoch_index(85, 333)


def test_real_transport_metagraph_pins_block_and_maps_registration_height():
    class Axon:
        def __init__(self, ip, port):
            self.ip = ip
            self.port = port

    class Metagraph:
        n = 2
        block = 777
        hotkeys = ["hk-0", "hk-1"]
        coldkeys = ["ck-0", "ck-1"]
        axons = [Axon("10.0.0.1", 9101), Axon("10.0.0.2", 9102)]
        alpha_stake = [1.0, 2.0]
        emission = [0.1, 0.2]
        validator_permit = [True, False]
        last_update = [700, 701]
        block_at_registration = [600, 601]

    class MetagraphSdk:
        def __init__(self):
            self.calls = []

        def metagraph(self, netuid, *, lite=True, block=None):
            self.calls.append((netuid, lite, block))
            return Metagraph()

    sdk = MetagraphSdk()
    view = _real_transport_with(sdk).metagraph(85, block_number=777)
    assert sdk.calls == [(85, False, 777)]
    assert view.block == 777
    assert view.registration_block == [600, 601]
    assert view.axon_ips == ["10.0.0.1", "10.0.0.2"]
    assert view.axon_ports == [9101, 9102]


class CompatibleV105Sdk:
    def set_weights(
        self,
        wallet,
        netuid,
        uids,
        weights,
        commit_reveal_version=4,
        max_attempts=5,
        version_key=0,
        *,
        mev_protection=False,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    ): ...

    def commit_reveal_enabled(self, netuid, block=None):
        return False

    def set_commitment(
        self,
        wallet,
        netuid,
        data,
        *,
        mev_protection=False,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    ): ...

    def metagraph(self, netuid, mechid=0, lite=True, block=None): ...

    def get_timelocked_weight_commits(self, netuid, mechid=0, block=None): ...

    def tx_rate_limit(self, block=None):
        return 0

    def get_timestamp(self, block=None): ...

    def get_commitment_metadata(self, netuid, hotkey_ss58, block=None): ...

    def get_epoch_schedule_state(self, netuid, block=None): ...

    def get_subnet_epoch_index(self, netuid, block=None): ...


def test_real_read_only_transport_never_invokes_wallet_loader(monkeypatch) -> None:
    sdk = CompatibleV105Sdk()
    sdk.substrate = FakeSubstrate()
    fake_bt = types.ModuleType("bittensor")
    connect_args = {}

    def subtensor(*, network, fallback_endpoints=None, archive_endpoints=None):
        connect_args.update(
            network=network,
            fallback_endpoints=fallback_endpoints,
            archive_endpoints=archive_endpoints,
        )
        return sdk

    fake_bt.Subtensor = subtensor
    monkeypatch.setitem(sys.modules, "bittensor", fake_bt)

    wallet_loads = 0

    def forbidden_wallet_load(self, config):  # type: ignore[no-untyped-def]
        nonlocal wallet_loads
        wallet_loads += 1
        raise AssertionError("read-only construction touched a signing key")

    monkeypatch.setattr(_RealSubtensorTransport, "_load_hotkey", forbidden_wallet_load)
    transport = _RealSubtensorTransport(
        BittensorAdapterConfig(
            read_only=True,
            validator_hotkey="",
            wallet_name="must-not-be-opened",
            wallet_hotkey="must-not-be-opened",
            wallet_path="/must/not/be/read",
            hotkey_seed_env="MUST_NOT_BE_READ",
            endpoint="wss://archive-a",
            fallback_endpoints=("wss://archive-b",),
        )
    )

    assert wallet_loads == 0
    assert connect_args == {
        "network": "wss://archive-a",
        "fallback_endpoints": ["wss://archive-b"],
        "archive_endpoints": ["wss://archive-b"],
    }
    assert transport._hotkey is None
    assert transport._wallet is None
    with pytest.raises(ReadOnlyChainError):
        transport.sign_hotkey(b"payload")
    with pytest.raises(ReadOnlyChainError):
        transport.signer_hotkey()
    with pytest.raises(ReadOnlyChainError):
        transport.set_weights(netuid=85, uids=[1], weights=[U16_MAX], version_key=12)
    with pytest.raises(ReadOnlyChainError):
        transport.set_commitment(netuid=85, payload="anchor")


def test_startup_sdk_contract_accepts_pinned_v105_shape():
    _real_transport_with(CompatibleV105Sdk())._validate_sdk_contract()


def test_startup_sdk_contract_requires_pinned_block_id_spelling():
    class LegacySubstrate(FakeSubstrate):
        def get_block_hash(self, block_number):
            return f"0x{block_number:064x}"

    transport = _real_transport_with(CompatibleV105Sdk())
    transport._substrate = LegacySubstrate()
    with pytest.raises(RuntimeError, match=r"get_block_hash lacks block_id"):
        transport._validate_sdk_contract()


def test_startup_sdk_contract_rejects_missing_mev_protection_kwarg():
    class IncompatibleSdk(CompatibleV105Sdk):
        def set_weights(
            self,
            wallet,
            netuid,
            uids,
            weights,
            version_key=0,
            *,
            raise_error=False,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        ): ...

    with pytest.raises(RuntimeError, match="mev_protection"):
        _real_transport_with(IncompatibleSdk())._validate_sdk_contract()


# --------------------------------------------------------------------------------------
# factory + fail-fast on missing deps
# --------------------------------------------------------------------------------------


def test_factory_builds_the_bittensor_adapter_with_an_injected_fake_transport():
    # The factory path itself needs the real deps (raises NotConfiguredError here),
    # so constructing the adapter directly with a fake transport is how the
    # bittensor-mode adapter is exercised end to end without bittensor.
    t = FakeTransport(view=make_view(3), uid_map={"hk-0": 0})
    adapter = make_adapter(t)
    from vidaio.chain import ChainAdapter, SubmittedWeightsReader

    assert isinstance(adapter, ChainAdapter)
    assert isinstance(adapter, SubmittedWeightsReader)


def test_read_only_factory_discards_all_wallet_configuration(monkeypatch) -> None:
    import vidaio.chain.bittensor_adapter as adapter_module

    captured: list[BittensorAdapterConfig] = []
    transport = FakeTransport(view=make_view(2))

    def connect(config: BittensorAdapterConfig):
        captured.append(config)
        return transport

    monkeypatch.setattr(adapter_module, "_connect_real_transport", connect)
    monkeypatch.delenv("MISSING_SEED_IS_FINE", raising=False)
    adapter = make_read_only_chain_adapter(
        {
            "chain": {
                "mode": "bittensor",
                "validator_hotkey": "configured-writer",
                "wallet_name": "configured-wallet",
                "wallet_hotkey": "configured-hotkey",
                "wallet_path": "/configured/wallet/path",
                "hotkey_seed_env": "MISSING_SEED_IS_FINE",
            }
        }
    )

    assert isinstance(adapter, BittensorReadOnlyChainAdapter)
    assert len(captured) == 1
    cfg = captured[0]
    assert cfg.read_only is True
    assert cfg.validator_hotkey == ""
    assert cfg.wallet_name == cfg.wallet_hotkey == cfg.wallet_path == ""
    assert cfg.hotkey_seed_env == ""
    adapter.refresh()
    assert len(adapter.neurons()) == 2


def test_factory_bittensor_mode_fails_fast_when_deps_absent(monkeypatch):
    # Simulate the base image explicitly: this remains deterministic even when a
    # developer has already installed the chain extra into the shared venv.
    monkeypatch.setitem(sys.modules, "bittensor", None)
    with pytest.raises(NotConfiguredError) as exc:
        make_chain_adapter({"chain": {"mode": "bittensor"}})
    assert ".[chain]" in str(exc.value)
