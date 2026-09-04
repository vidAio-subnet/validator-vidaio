"""Registration: sequential uids, hotkey idempotency, restart safety."""

from __future__ import annotations

from chainsim_support import bearer, register

from vidaio.chainsim.service import RegisterRequest, WeightsRequest


async def test_uids_are_sequential_and_reregistration_is_idempotent(client):
    uid_a, token_a = await register(
        client, "hk-a", coldkey="ck-a", ip="10.0.0.1", role="miner"
    )
    uid_b, _token_b = await register(
        client, "hk-b", coldkey="ck-b", ip="10.0.0.2", role="validator"
    )
    assert (uid_a, uid_b) == (0, 1)

    # same hotkey again (with its token): same uid, not a new slot, fields updated
    again = await client.post(
        "/register",
        json={"hotkey": "hk-a", "coldkey": "ck-a2", "ip": "10.9.9.9", "role": "miner"},
        headers=bearer(token_a),
    )
    assert (again.json()["uid"], again.json()["new"]) == (0, False)
    # the token is issued ONCE — a later registration never re-reveals it
    assert again.json()["auth_token"] is None

    neurons = (await client.get("/neurons")).json()["neurons"]
    assert [n["uid"] for n in neurons] == [0, 1]
    assert neurons[0]["ip"] == "10.9.9.9"
    assert neurons[0]["coldkey"] == "ck-a2"
    assert neurons[1]["is_validator"] is True

    # a third distinct hotkey still gets the next sequential uid
    uid_c, _ = await register(client, "hk-c", role="miner")
    assert uid_c == 2


async def test_alpha_stake_only_updates_when_supplied(client):
    _uid, token = await register(client, "hk-a", role="validator", alpha_stake=500.0)
    # re-register WITHOUT alpha_stake: stake untouched
    await client.post(
        "/register", json={"hotkey": "hk-a", "role": "validator"}, headers=bearer(token)
    )
    neurons = (await client.get("/neurons")).json()["neurons"]
    assert neurons[0]["alpha_stake"] == 500.0
    # re-register WITH alpha_stake: stake moves (the sim's only stake lever)
    await client.post(
        "/register",
        json={"hotkey": "hk-a", "role": "validator", "alpha_stake": 750.0},
        headers=bearer(token),
    )
    neurons = (await client.get("/neurons")).json()["neurons"]
    assert neurons[0]["alpha_stake"] == 750.0


def test_registry_blocks_and_history_survive_restart(make_sim):
    sim1 = make_sim()
    assert sim1.register(RegisterRequest(hotkey="hk-a", role="miner"))["uid"] == 0
    sim1.advance(7)
    assert sim1.submit_weights(WeightsRequest(hotkey="hk-a", vector={0: 1.0}))["success"]
    sim1.close()

    sim2 = make_sim()  # same tmp SQLite file = a restart
    assert sim2.current_block() == 8
    assert sim2.register(RegisterRequest(hotkey="hk-a", role="miner"))["new"] is False
    assert sim2.register(RegisterRequest(hotkey="hk-b", role="miner"))["uid"] == 1
    assert len(sim2.state()["weight_calls"]) == 1


def test_tokens_survive_restart(make_sim):
    """The credential is stored (hashed), so a restarted sim still knows it."""
    sim1 = make_sim()
    token = sim1.register(RegisterRequest(hotkey="hk-a", role="validator"))["auth_token"]
    sim1.close()

    sim2 = make_sim()
    row = sim2._neuron_row("hk-a")
    assert row["token_sha256"] and token not in row["token_sha256"]  # only the hash
    from vidaio.chainsim.service import _token_matches

    assert _token_matches(token, row["token_sha256"])
