"""POST /reset: full wipe when enabled, 403 when config-disabled."""

from __future__ import annotations

import httpx
from chainsim_support import OPERATOR_TOKEN, bearer, register


async def test_reset_wipes_state_and_rewinds_the_clock(sim, client, operator):
    _uid, token = await register(client, "val", role="validator")
    await client.post(
        "/weights",
        json={"hotkey": "val", "vector": {"0": 1.0}, "version_key": 0},
        headers=bearer(token),
    )
    await client.post("/advance", json={"blocks": 10}, headers=operator)
    payload_hex = b"x".hex()
    await client.post(
        "/anchor", json={"payload_hex": payload_hex, "hotkey": "val"}, headers=bearer(token)
    )

    response = await client.post("/reset", headers=operator)
    assert response.status_code == 200 and response.json() == {"block": 1}

    state = (await client.get("/state")).json()
    assert state["block"] == 1
    assert state["neurons"] == []
    assert state["weight_calls"] == []
    assert state["anchors"] == []
    assert state["emission"]["distributed"] == 0.0

    # uid numbering restarts too, and identities are re-issued from scratch
    again = await client.post("/register", json={"hotkey": "someone-new"})
    assert again.json()["uid"] == 0
    # the operator credential belongs to the node, not the run — it survived
    assert (await client.post("/reset", headers=operator)).status_code == 200


async def test_reset_is_config_gated(make_sim):
    sim = make_sim(enable_reset=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=sim.app), base_url="http://sim"
    ) as client:
        response = await client.post("/reset", headers=bearer(OPERATOR_TOKEN))
    assert response.status_code == 403
    assert response.json()["detail"] == "reset is disabled by config"
