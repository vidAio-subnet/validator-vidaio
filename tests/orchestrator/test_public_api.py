"""Public competitions API: discovery + self-signed, chain-checked enrollment."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from vidaio.competition import repository as repo
from vidaio.competition.orchestrator.public_api import (
    SlidingWindowLimiter,
    create_public_app,
)
from vidaio.services.hotkey_auth import (
    HotkeyAuthConfig,
    HotkeyAuthGuard,
    RegisteredHotkeyRegistry,
    sign_request_headers,
)

from orchestrator_support import CONTENDER_SHAS, M, START, build_manifest, start_and_enroll

REPO = "https://github.com/example/solution.git"
HK_A = "5" + "A" * 47
HK_B = "5" + "B" * 47
HK_OTHER = "5" + "C" * 47


class _Chain:
    """Registry seam: registered hotkeys with their on-chain alpha stake."""

    def __init__(self, stakes: dict[str, float]) -> None:
        self.stakes = stakes

    def refresh(self) -> None:
        return None

    def neurons(self):
        return [
            SimpleNamespace(hotkey=hk, is_validator=False, alpha_stake=stake)
            for hk, stake in self.stakes.items()
        ]


class _Signer:
    def __init__(self, hotkey: str, *, forge: bool = False) -> None:
        self.hotkey = hotkey
        self._forge = forge

    def sign(self, payload: bytes) -> str:
        return "bad" if self._forge else f"sig:{self.hotkey}:{payload.hex()}"


def _verify(hotkey: str, payload: bytes, signature: str) -> bool:
    return signature == f"sig:{hotkey}:{payload.hex()}"


def _guard(stakes: dict[str, float], mode: str = "enforce") -> HotkeyAuthGuard:
    return HotkeyAuthGuard(
        RegisteredHotkeyRegistry(_Chain(stakes)),
        HotkeyAuthConfig(mode=mode, min_enroll_alpha_stake=500.0),
        verify_fn=_verify,
    )


async def _world(orchestrator_factory, fixture_repos, *, guard, **kwargs):
    orch = orchestrator_factory(repos=fixture_repos, clock=lambda: START + 5 * M)
    cid = await start_and_enroll(orch, build_manifest("comp-public"), [])
    app = create_public_app(
        orch,
        hotkey_guard=guard,
        min_enroll_alpha_stake=500.0,
        allowed_repo_hosts=("github.com",),
        **kwargs,
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://public"
    )
    return orch, cid, client


def _enroll_request(cid: str, signer: _Signer, body: dict | None = None):
    commit_sha, tree_sha = CONTENDER_SHAS["hk-a"]
    raw = json.dumps(
        body
        if body is not None
        else {"repo_url": REPO, "commit_sha": commit_sha, "tree_sha": tree_sha}
    ).encode()
    path = f"/v1/competitions/{cid}/enroll"
    headers = sign_request_headers(signer, method="POST", path=path, body=raw)
    headers["content-type"] = "application/json"
    return path, raw, headers


async def test_registered_staked_miner_enrolls_itself(orchestrator_factory, fixture_repos):
    orch, cid, client = await _world(
        orchestrator_factory, fixture_repos, guard=_guard({HK_A: 750.0})
    )
    async with client:
        path, raw, headers = _enroll_request(cid, _Signer(HK_A))
        response = await client.post(path, content=raw, headers=headers)
        assert response.status_code == 201, response.text
        assert response.json()["hotkey"] == HK_A
        assert response.json()["alpha_stake"] == 750.0
        contenders = repo.list_contenders(orch.conn, cid)
        assert [(c.hotkey, c.enrollment_stake) for c in contenders] == [(HK_A, 750.0)]

        listing = (await client.get("/v1/competitions")).json()
        assert listing["min_enroll_alpha_stake"] == 500.0
        entry = listing["competitions"][0]
        assert entry["enrollment_open"] is True
        assert entry["enrolled"] == [{"hotkey": HK_A, "status": "ENROLLED"}]
        assert entry["manifest"]["competition_id"] == cid

        # One submission per hotkey per competition.
        path, raw, headers = _enroll_request(cid, _Signer(HK_A))
        again = await client.post(path, content=raw, headers=headers)
        assert again.status_code == 409


@pytest.mark.parametrize(
    ("stakes", "signer", "status", "code"),
    [
        ({HK_OTHER: 900.0}, _Signer(HK_A), 403, "hotkey_not_registered"),
        ({HK_A: 499.0}, _Signer(HK_A), 403, "hotkey_below_stake_floor"),
        ({HK_A: 900.0}, _Signer(HK_A, forge=True), 401, None),
    ],
)
async def test_enrollment_refusals(
    orchestrator_factory, fixture_repos, stakes, signer, status, code
):
    orch, cid, client = await _world(
        orchestrator_factory, fixture_repos, guard=_guard(stakes)
    )
    async with client:
        path, raw, headers = _enroll_request(cid, signer)
        response = await client.post(path, content=raw, headers=headers)
    assert response.status_code == status, response.text
    if code is not None:
        assert response.json()["detail"]["code"] == code
    assert repo.list_contenders(orch.conn, cid) == []


async def test_unsigned_request_is_refused(orchestrator_factory, fixture_repos):
    orch, cid, client = await _world(
        orchestrator_factory, fixture_repos, guard=_guard({HK_A: 900.0})
    )
    async with client:
        response = await client.post(
            f"/v1/competitions/{cid}/enroll",
            json={"repo_url": REPO, "commit_sha": "a" * 40, "tree_sha": "b" * 40},
        )
    assert response.status_code == 401
    assert repo.list_contenders(orch.conn, cid) == []


async def test_body_cannot_name_another_hotkey_or_a_stake(
    orchestrator_factory, fixture_repos
):
    orch, cid, client = await _world(
        orchestrator_factory, fixture_repos, guard=_guard({HK_A: 900.0, HK_B: 900.0})
    )
    commit_sha, tree_sha = CONTENDER_SHAS["hk-a"]
    async with client:
        path, raw, headers = _enroll_request(
            cid,
            _Signer(HK_A),
            {"repo_url": REPO, "commit_sha": commit_sha, "tree_sha": tree_sha,
             "hotkey": HK_B, "stake": 1e9},
        )
        response = await client.post(path, content=raw, headers=headers)
    assert response.status_code == 422
    assert repo.list_contenders(orch.conn, cid) == []


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.example/x/y.git",
        "http://github.com/x/y.git",
        "https://token@github.com/x/y.git",
        "https://github.com:8443/x/y.git",
    ],
)
async def test_repo_host_allow_list(orchestrator_factory, fixture_repos, url):
    orch, cid, client = await _world(
        orchestrator_factory, fixture_repos, guard=_guard({HK_A: 900.0})
    )
    async with client:
        path, raw, headers = _enroll_request(
            cid, _Signer(HK_A),
            {"repo_url": url, "commit_sha": "a" * 40, "tree_sha": "b" * 40},
        )
        response = await client.post(path, content=raw, headers=headers)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "repo_host_not_allowed"


@pytest.mark.parametrize("guard", [None, "log"])
async def test_enrollment_needs_an_enforcing_guard(
    orchestrator_factory, fixture_repos, guard
):
    built = None if guard is None else _guard({HK_A: 900.0}, mode="log")
    orch, cid, client = await _world(orchestrator_factory, fixture_repos, guard=built)
    async with client:
        path, raw, headers = _enroll_request(cid, _Signer(HK_A))
        response = await client.post(path, content=raw, headers=headers)
        assert response.status_code == 503
        assert (await client.get(f"/v1/competitions/{cid}")).status_code == 200
    assert repo.list_contenders(orch.conn, cid) == []


async def test_enrollment_closed_outside_the_window(orchestrator_factory, fixture_repos):
    orch = orchestrator_factory(repos=fixture_repos)
    manifest = build_manifest("comp-scheduled")
    orch.create_competition(manifest, START.replace(year=START.year - 1))
    app = create_public_app(
        orch, hotkey_guard=_guard({HK_A: 900.0}), min_enroll_alpha_stake=500.0,
        allowed_repo_hosts=("github.com",),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://public"
    ) as client:
        path, raw, headers = _enroll_request("comp-scheduled", _Signer(HK_A))
        response = await client.post(path, content=raw, headers=headers)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "enrollment_closed"


async def test_enrollment_is_rate_limited_per_address(orchestrator_factory, fixture_repos):
    orch, cid, client = await _world(
        orchestrator_factory, fixture_repos, guard=_guard({HK_A: 900.0}),
        enroll_limit_per_hour=2,
    )
    async with client:
        codes = []
        for _ in range(3):
            path, raw, headers = _enroll_request(cid, _Signer(HK_A, forge=True))
            codes.append((await client.post(path, content=raw, headers=headers)).status_code)
    assert codes == [401, 401, 429]


def test_limiter_window_and_key_flood() -> None:
    now = [0.0]
    limiter = SlidingWindowLimiter(2, 10.0, clock=lambda: now[0], max_keys=2)
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    now[0] = 11.0
    assert limiter.allow("a")
    assert limiter.allow("b")
    assert not limiter.allow("c")  # key table full of live keys: fail closed
