"""Snapshot freshness: refresh() fails OPEN, but never SILENTLY (finding #21).

refresh() must not raise — service loops depend on that — so the adapter carries
the evidence instead: `last_refresh_error`, `snapshot_age()`, and
`has_fresh_snapshot()`. And a NEVER-refreshed adapter refuses to answer
`neurons()` at all: an empty list there is indistinguishable from "no miners are
registered", which is how a startup race becomes a successful-looking empty
round and a silently omitted weight vector.
"""

from __future__ import annotations

import httpx
import pytest
from chainsim_support import SyncASGITransport

from vidaio.chain import (
    ChainAdapter,
    ChainStateUnavailable,
    EmbeddedReportingChain,
    HttpChainAdapter,
    InMemoryChain,
)
from vidaio.chainsim.service import RegisterRequest


class Clock:
    """Injected wall clock (epoch seconds — the Protocol's freshness clock)."""

    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class Down(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sim down")


class Garbage(httpx.BaseTransport):
    """200 OK with a payload that does not decode into neurons."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"block": "not-an-int", "neurons": [{}]})


def make_adapter_with(sim, transport, clock) -> HttpChainAdapter:
    return HttpChainAdapter(
        "http://sim",
        validator_hotkey="val",
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=sim.app)),
        clock=clock,
    )


# ---- never refreshed --------------------------------------------------------------


def test_never_refreshed_neurons_raises_instead_of_faking_an_empty_subnet(sim):
    clock = Clock()
    adapter = make_adapter_with(sim, Down(), clock)

    assert adapter.snapshot_age(clock()) is None
    assert adapter.has_fresh_snapshot(clock(), 60.0) is False
    with pytest.raises(ChainStateUnavailable):
        adapter.neurons()

    adapter.refresh()  # fails — and STILL nothing to serve
    assert adapter.last_refresh_error is not None
    assert "ConnectError" in adapter.last_refresh_error
    assert adapter.last_successful_refresh is None
    assert adapter.has_fresh_snapshot(clock(), 60.0) is False
    with pytest.raises(ChainStateUnavailable) as excinfo:
        adapter.neurons()
    assert "ConnectError" in str(excinfo.value)  # the reason travels with the refusal
    adapter.close()


def test_a_malformed_payload_is_a_failed_refresh_not_a_half_applied_one(sim):
    """A 200 that does not decode must leave the cache untouched and be reported."""
    clock = Clock()
    adapter = make_adapter_with(sim, Garbage(), clock)
    adapter.refresh()
    assert adapter.last_successful_refresh is None
    assert adapter.current_block() == 0
    assert adapter.last_refresh_error is not None
    with pytest.raises(ChainStateUnavailable):
        adapter.neurons()
    adapter.close()


# ---- staleness + recovery ---------------------------------------------------------


def test_staleness_is_measured_from_the_last_SUCCESSFUL_refresh(sim):
    clock = Clock()
    live = SyncASGITransport(sim.app)
    adapter = make_adapter_with(sim, live, clock)
    sim.register(RegisterRequest(hotkey="hk1", role="miner"))

    adapter.refresh()
    assert adapter.snapshot_age(clock()) == 0.0
    assert adapter.has_fresh_snapshot(clock(), 60.0) is True
    assert [n.hotkey for n in adapter.neurons()] == ["hk1"]

    clock.advance(30.0)
    assert adapter.snapshot_age(clock()) == 30.0
    assert adapter.has_fresh_snapshot(clock(), 60.0) is True
    assert adapter.has_fresh_snapshot(clock(), 10.0) is False  # older than allowed

    # the sim goes away: reads keep working (cached), freshness keeps ageing
    adapter._client = httpx.Client(transport=Down())
    clock.advance(40.0)
    adapter.refresh()
    assert adapter.last_refresh_error is not None
    assert adapter.snapshot_age(clock()) == 70.0
    assert adapter.has_fresh_snapshot(clock(), 60.0) is False
    assert [n.hotkey for n in adapter.neurons()] == ["hk1"]  # last known state

    # recovery: a successful refresh clears the error and resets the age
    adapter._client = httpx.Client(transport=live)
    sim.register(RegisterRequest(hotkey="hk2", role="miner"))
    adapter.refresh()
    assert adapter.last_refresh_error is None
    assert adapter.last_successful_refresh == clock()
    assert adapter.snapshot_age(clock()) == 0.0
    assert adapter.has_fresh_snapshot(clock(), 1.0) is True
    assert [n.hotkey for n in adapter.neurons()] == ["hk1", "hk2"]
    adapter.close()


def test_boundary_age_is_still_fresh(sim):
    clock = Clock()
    adapter = make_adapter_with(sim, SyncASGITransport(sim.app), clock)
    adapter.refresh()
    clock.advance(60.0)
    assert adapter.has_fresh_snapshot(clock(), 60.0) is True
    clock.advance(0.001)
    assert adapter.has_fresh_snapshot(clock(), 60.0) is False
    adapter.close()


# ---- the surface is uniform across adapters ---------------------------------------


def test_in_memory_adapters_are_trivially_fresh(tmp_path):
    """Same Protocol surface for the fakes, so callers need no isinstance checks."""
    for chain in (
        InMemoryChain(),
        EmbeddedReportingChain(journal_path=tmp_path / "journal.jsonl"),
    ):
        assert isinstance(chain, ChainAdapter)
        assert chain.has_fresh_snapshot(0.0, 0.0) is True
        assert chain.snapshot_age(12345.0) == 0.0
        assert chain.last_refresh_error is None
        assert chain.neurons() == []  # genuinely empty, not "unknown"


def test_http_adapter_still_satisfies_the_protocol(sim):
    adapter = make_adapter_with(sim, SyncASGITransport(sim.app), Clock())
    assert isinstance(adapter, ChainAdapter)
    adapter.close()
