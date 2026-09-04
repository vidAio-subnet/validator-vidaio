"""Authorization: the sim's stand-in for the real chain's signature model.

Every mutation must prove WHO it is (a bearer token issued at registration —
"token ~ hotkey signature") and that the identity is ALLOWED the action (only a
validator sets weights; only the operator produces blocks or destroys history).
Reads stay open — chain state is public.
"""

from __future__ import annotations

import httpx
import pytest
from chainsim_support import OPERATOR_TOKEN, bearer, register

from vidaio.chainsim.service import OPERATOR_TOKEN_FILE

WEIGHTS = {"vector": {"0": 1.0}, "version_key": 0}


# ---- identity: you can only act as yourself -------------------------------------


async def test_weights_require_a_token(client):
    await register(client, "val", role="validator")
    response = await client.post("/weights", json={"hotkey": "val", **WEIGHTS})
    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "auth_token_missing"


@pytest.mark.parametrize(
    "header",
    [{"Authorization": "wrong-scheme abc"}, {"Authorization": "Bearer "}],
    ids=["not-bearer", "empty-bearer"],
)
async def test_malformed_credentials_are_treated_as_absent(client, header):
    await register(client, "val", role="validator")
    response = await client.post("/weights", json={"hotkey": "val", **WEIGHTS}, headers=header)
    assert response.status_code == 401


async def test_a_wrong_token_cannot_submit_as_the_validator(client):
    await register(client, "val", role="validator")
    _uid, other = await register(client, "hk-miner", role="miner")

    # somebody else's token, and a made-up one, both naming the validator
    for token in (other, "totally-made-up"):
        response = await client.post(
            "/weights", json={"hotkey": "val", **WEIGHTS}, headers=bearer(token)
        )
        assert response.status_code == 403
        assert response.json()["detail"]["error"] == "auth_token_invalid"
    assert (await client.get("/state")).json()["weight_calls"] == []


async def test_a_miner_cannot_set_weights_even_with_its_own_token(client):
    """The validator permit, enforced: a registered miner submitting a
    self-paying vector under its OWN identity is refused."""
    _uid, miner_token = await register(client, "hk-miner", role="miner")
    response = await client.post(
        "/weights", json={"hotkey": "hk-miner", **WEIGHTS}, headers=bearer(miner_token)
    )
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "auth_wrong_role"
    assert (await client.get("/state")).json()["weight_calls"] == []


async def test_a_validator_sets_weights_with_its_own_token(client):
    _uid, token = await register(client, "val", role="validator")
    response = await client.post(
        "/weights", json={"hotkey": "val", **WEIGHTS}, headers=bearer(token)
    )
    assert response.status_code == 200 and response.json()["success"] is True
    assert len((await client.get("/state")).json()["weight_calls"]) == 1


async def test_anchors_are_bound_to_the_named_hotkey(client, operator):
    _uid, val_token = await register(client, "val", role="validator")
    _uid2, miner_token = await register(client, "hk-miner", role="miner")
    payload = {"payload_hex": b"vidaio".hex()}

    assert (await client.post("/anchor", json={**payload, "hotkey": "val"})).status_code == 401
    tampered = await client.post(
        "/anchor", json={**payload, "hotkey": "val"}, headers=bearer(miner_token)
    )
    assert tampered.status_code == 403

    # a miner may anchor as ITSELF (anchoring is not a validator-only power)
    mine = await client.post(
        "/anchor", json={**payload, "hotkey": "hk-miner"}, headers=bearer(miner_token)
    )
    assert mine.status_code == 200
    ok = await client.post(
        "/anchor", json={**payload, "hotkey": "val"}, headers=bearer(val_token)
    )
    assert ok.status_code == 200

    # unattributed anchors are an operator action, not a participant one
    unattributed = await client.post("/anchor", json=payload, headers=bearer(val_token))
    assert unattributed.status_code == 403
    assert (await client.post("/anchor", json=payload, headers=operator)).status_code == 200

    hotkeys = [a["hotkey"] for a in (await client.get("/state")).json()["anchors"]]
    assert hotkeys == ["hk-miner", "val", None]


# ---- operator powers -------------------------------------------------------------


@pytest.mark.parametrize(
    "path,body",
    [("/advance", {"blocks": 3}), ("/reset", None)],
    ids=["advance", "reset"],
)
async def test_operator_endpoints_reject_participant_tokens(client, operator, path, body):
    _uid, val_token = await register(client, "val", role="validator")

    def call(**kw):
        return client.post(path, json=body, **kw)

    assert (await call()).status_code == 401
    forbidden = await call(headers=bearer(val_token))
    assert forbidden.status_code == 403
    assert forbidden.json()["detail"]["error"] == "auth_token_invalid"
    # block production / history destruction did NOT happen
    assert (await client.get("/state")).json()["block"] == 1
    assert [n["hotkey"] for n in (await client.get("/neurons")).json()["neurons"]] == ["val"]

    assert (await call(headers=operator)).status_code == 200


async def test_a_generated_operator_token_is_written_to_the_report_dir(tmp_path, fake_time):
    """No configured token: the sim mints one, persists its hash, and drops the
    plaintext in the report dir so the operator can actually use it."""
    from vidaio.chainsim import ChainSim

    def build() -> ChainSim:
        return ChainSim(
            {
                "core": {"metrics_port": 0},
                "chainsim": {
                    "port": 0,
                    "metrics_port": 0,
                    "db_path": str(tmp_path / "sim.db"),
                    "report_dir": str(tmp_path / "reports"),
                },
            },
            now=fake_time,
        )

    sim = build()
    token_file = tmp_path / "reports" / OPERATOR_TOKEN_FILE
    token = token_file.read_text().strip()
    assert token and sim._is_operator(token) and not sim._is_operator("nope")
    sim.close()

    # a restart keeps the SAME credential (the file stays valid)
    restarted = build()
    assert restarted._is_operator(token)
    assert token_file.read_text().strip() == token
    restarted.close()

    sim2 = build()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=sim2.app), base_url="http://sim"
    ) as client:
        advanced = await client.post(
            "/advance", json={"blocks": 1}, headers=bearer(token)
        )
    assert advanced.status_code == 200  # the written token really is the operator's
    sim2.close()


async def test_a_configured_operator_token_overrides_a_generated_one(tmp_path, fake_time):
    """Recovery path: pinning chainsim.operator_token re-establishes access."""
    from vidaio.chainsim import ChainSim

    raw = {
        "core": {"metrics_port": 0},
        "chainsim": {
            "port": 0,
            "metrics_port": 0,
            "db_path": str(tmp_path / "sim.db"),
            "report_dir": str(tmp_path / "reports"),
        },
    }
    first = ChainSim(raw, now=fake_time)
    generated = (tmp_path / "reports" / OPERATOR_TOKEN_FILE).read_text().strip()
    first.close()

    pinned = ChainSim(
        {**raw, "chainsim": {**raw["chainsim"], "operator_token": "pinned-token"}},
        now=fake_time,
    )
    assert pinned._is_operator("pinned-token")
    assert not pinned._is_operator(generated)  # rotated out
    pinned.close()


# ---- identity takeover -----------------------------------------------------------


async def test_a_claimed_hotkey_cannot_be_taken_over(client):
    _uid, token = await register(client, "val", role="validator", alpha_stake=100.0)

    # no credential: the hotkey is spoken for
    hijack = await client.post(
        "/register", json={"hotkey": "val", "role": "validator", "alpha_stake": 999.0}
    )
    assert hijack.status_code == 409
    assert hijack.json()["detail"]["error"] == "auth_hotkey_claimed"

    # wrong credential, header or body: still no
    for attempt in (
        client.post(
            "/register", json={"hotkey": "val", "role": "validator"}, headers=bearer("guess")
        ),
        client.post("/register", json={"hotkey": "val", "auth_token": "guess-again"}),
    ):
        assert (await attempt).status_code == 403

    # nothing moved, and the real holder still works
    neurons = (await client.get("/neurons")).json()["neurons"]
    assert [(n["hotkey"], n["alpha_stake"], n["role"]) for n in neurons] == [
        ("val", 100.0, "validator")
    ]
    ok = await client.post(
        "/register",
        json={"hotkey": "val", "role": "validator", "alpha_stake": 999.0},
        headers=bearer(token),
    )
    assert ok.status_code == 200 and ok.json()["new"] is False


async def test_a_client_supplied_token_claims_a_new_hotkey(client):
    """How a multi-process local fleet shares one configured secret: the first
    registration adopts it; later ones must present it."""
    claimed = await client.post(
        "/register",
        json={"hotkey": "val", "role": "validator", "auth_token": "fleet-secret"},
    )
    assert claimed.status_code == 200 and claimed.json()["auth_token"] == "fleet-secret"

    weights = await client.post(
        "/weights", json={"hotkey": "val", **WEIGHTS}, headers=bearer("fleet-secret")
    )
    assert weights.json()["success"] is True
    # restart-style re-registration with the same secret is idempotent
    again = await client.post(
        "/register",
        json={"hotkey": "val", "role": "validator", "auth_token": "fleet-secret"},
    )
    assert again.status_code == 200 and again.json()["new"] is False


async def test_pre_auth_rows_are_unclaimed_and_claimable_once(sim, client):
    """A neuron migrated from a pre-auth sim database (token_sha256 = '') cannot
    authenticate anything until a registration claims it."""
    sim._conn.execute(
        "INSERT INTO neurons (uid, hotkey, coldkey, ip, role, alpha_stake,"
        " registered_block, token_sha256) VALUES (0, 'legacy', '', '', 'validator', 0, 1, '')"
    )
    denied = await client.post(
        "/weights", json={"hotkey": "legacy", **WEIGHTS}, headers=bearer("")
    )
    assert denied.status_code == 401  # an empty bearer is no bearer
    assert (
        await client.post(
            "/weights", json={"hotkey": "legacy", **WEIGHTS}, headers=bearer("anything")
        )
    ).status_code == 403

    claim = await client.post("/register", json={"hotkey": "legacy", "role": "validator"})
    token = claim.json()["auth_token"]
    assert claim.json()["new"] is False and token
    assert (
        await client.post(
            "/weights", json={"hotkey": "legacy", **WEIGHTS}, headers=bearer(token)
        )
    ).json()["success"] is True


# ---- reads stay open --------------------------------------------------------------


async def test_reads_need_no_credential(client):
    _uid, token = await register(client, "val", role="validator")
    await client.post("/weights", json={"hotkey": "val", **WEIGHTS}, headers=bearer(token))
    for path in ("/healthz", "/neurons", "/state", "/report"):
        assert (await client.get(path)).status_code == 200, path


async def test_auth_failures_are_counted(sim, client):
    await register(client, "val", role="validator")
    await client.post("/weights", json={"hotkey": "val", **WEIGHTS})  # 401
    await client.post("/advance", json={"blocks": 1}, headers=bearer("nope"))  # 403

    samples = {
        (s.labels.get("error"), s.name): s.value
        for metric in sim.health.registry.collect()
        for s in metric.samples
        if metric.name == "vidaio_chainsim_auth_failures"
    }
    assert samples[("auth_token_missing", "vidaio_chainsim_auth_failures_total")] == 1.0
    assert samples[("auth_token_invalid", "vidaio_chainsim_auth_failures_total")] == 1.0
    assert OPERATOR_TOKEN  # the pinned fixture credential, unused by the failures above
