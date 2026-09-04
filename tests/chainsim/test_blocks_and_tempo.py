"""Lazy block clock, /advance, and tempo-gate parity with InMemoryChain."""

from __future__ import annotations

from chainsim_support import bearer, register

from vidaio.chain import InMemoryChain
from vidaio.chainsim.service import RegisterRequest, WeightsRequest


def test_blocks_advance_lazily_from_the_injected_clock(make_sim, fake_time):
    sim = make_sim(block_seconds=2.0)
    assert sim.current_block() == 1
    fake_time.advance(5.0)
    assert sim.current_block() == 3  # 1 + floor(5/2), no background task involved
    sim.advance(4)
    assert sim.current_block() == 7  # wall clock and /advance offsets compose


def test_block_time_of_a_past_block_is_stable_across_advance(make_sim, fake_time):
    """an internal review: a PRODUCED block's time must NOT change when the clock later advances.

    The finalizer pins EpochLog.created_at to block_time(close_block); a later /advance must not
    move that value, or the auditor's created_at binding turns an honest CLEAN epoch DISPUTED.
    """
    sim = make_sim(block_seconds=2.0)
    sim.advance(10)  # produce blocks 2..11
    close_block = 5
    before = sim.block_time(close_block)
    assert before is not None
    # advancing the clock a lot must not retroactively shift an already-produced block's time.
    sim.advance(1000)
    assert sim.block_time(close_block) == before
    # and it is the pure function of the block number (start + (b-1)*block_seconds).
    fake_time.advance(4.0)  # natural time passing must not shift it either
    assert sim.block_time(close_block) == before
    # a not-yet-produced block is still None (offset governs only the produced gate).
    assert sim.block_time(sim.current_block() + 1) is None


async def test_advance_endpoint_is_deterministic_and_forward_only(client, operator):
    def go(blocks: int):
        return client.post("/advance", json={"blocks": blocks}, headers=operator)

    assert (await go(5)).json() == {"block": 6}
    assert (await go(0)).json() == {"block": 6}
    assert (await go(-1)).status_code == 422


async def test_tempo_gate_parity_with_inmemorychain(make_sim):
    tempo = 10
    sim = make_sim(tempo=tempo)
    mem = InMemoryChain(tempo=tempo)
    sim.register(RegisterRequest(hotkey="val", role="validator"))

    async def both() -> tuple[tuple[bool, str], tuple[bool, str]]:
        s = sim.submit_weights(WeightsRequest(hotkey="val", vector={0: 1.0}))
        m = await mem.set_weights({0: 1.0}, version_key=0)
        return (s["success"], s["message"]), (m.success, m.message)

    # first call at block 1: both accept
    s, m = await both()
    assert s == m == (True, "")
    # same block again: both reject with the exact same message
    s, m = await both()
    assert s == m == (False, "tempo gate: too soon")
    # boundary block last + tempo (11): still gated in both
    sim.advance(tempo)
    mem.advance_blocks(tempo)
    s, m = await both()
    assert s == m == (False, "tempo gate: too soon")
    # one block past the boundary: both open
    sim.advance(1)
    mem.advance_blocks(1)
    s, m = await both()
    assert s == m == (True, "")


async def test_tempo_gate_is_per_validator_hotkey(make_sim):
    sim = make_sim(tempo=10)
    sim.register(RegisterRequest(hotkey="val-a", role="validator"))
    sim.register(RegisterRequest(hotkey="val-b", role="validator"))
    assert sim.submit_weights(WeightsRequest(hotkey="val-a", vector={0: 1.0}))["success"]
    # a DIFFERENT validator is not blocked by val-a's gate
    assert sim.submit_weights(WeightsRequest(hotkey="val-b", vector={0: 1.0}))["success"]
    # but val-a itself is
    assert not sim.submit_weights(WeightsRequest(hotkey="val-a", vector={0: 1.0}))["success"]


async def test_unregistered_hotkey_cannot_set_weights(sim, client):
    """Over HTTP an unknown hotkey cannot even authenticate; the chain rule that
    unregistered hotkeys have no weight permit is enforced independently."""
    _uid, token = await register(client, "val", role="validator")
    response = await client.post(
        "/weights",
        json={"hotkey": "ghost", "vector": {"0": 1.0}, "version_key": 0},
        headers=bearer(token),
    )
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "auth_unknown_hotkey"

    # the rule itself, at the model layer (in-process callers are trusted)
    result = sim.submit_weights(WeightsRequest(hotkey="ghost", vector={0: 1.0}))
    assert result["success"] is False and "not registered" in result["message"]


async def test_rejected_calls_are_not_recorded(make_sim):
    sim = make_sim(tempo=10)
    sim.register(RegisterRequest(hotkey="val", role="validator"))
    assert sim.submit_weights(WeightsRequest(hotkey="val", vector={0: 1.0}))["success"]
    assert not sim.submit_weights(WeightsRequest(hotkey="val", vector={0: 2.0}))["success"]
    calls = sim.state()["weight_calls"]
    assert len(calls) == 1 and calls[0]["vector"] == {"0": 1.0}
