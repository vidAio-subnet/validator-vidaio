"""Self-serve entries on the public API: replace and withdraw before the deadline, the
signed entry status, and the public per-entry summary."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest

from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import entries
from vidaio.competition.orchestrator.public_api import create_public_app
from vidaio.services.hotkey_auth import (
    HotkeyAuthConfig,
    HotkeyAuthGuard,
    RegisteredHotkeyRegistry,
    sign_request_headers,
)

from orchestrator_support import CONTENDER_SHAS, M, START, build_manifest, start_and_enroll

REPO = "https://github.com/example/solution.git"
REPO_2 = "https://github.com/example/solution-v2.git"
HK_A = "5" + "A" * 47
HK_B = "5" + "B" * 47
COMMIT_1, TREE_1 = CONTENDER_SHAS["hk-a"]
COMMIT_2, TREE_2 = CONTENDER_SHAS["hk-b"]


class _Chain:
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
    def __init__(self, hotkey: str) -> None:
        self.hotkey = hotkey

    def sign(self, payload: bytes) -> str:
        return f"sig:{self.hotkey}:{payload.hex()}"


def _verify(hotkey: str, payload: bytes, signature: str) -> bool:
    return signature == f"sig:{hotkey}:{payload.hex()}"


class _World:
    def __init__(self, orch, cid: str, client: httpx.AsyncClient, chain: _Chain, now):
        self.orch, self.cid, self.client, self.chain, self.now = orch, cid, client, chain, now

    async def signed(self, method: str, action: str, hotkey: str, body: dict | None = None):
        path = f"/v1/competitions/{self.cid}/{action}"
        raw = b"" if body is None else json.dumps(body).encode()
        headers = sign_request_headers(_Signer(hotkey), method=method, path=path, body=raw)
        if body is not None:
            headers["content-type"] = "application/json"
        return await self.client.request(method, path, content=raw, headers=headers)

    async def enroll(self, hotkey: str, commit: str = COMMIT_1, tree: str = TREE_1, *,
                     repo_url: str = REPO, replace: bool | None = None):
        body = {"repo_url": repo_url, "commit_sha": commit, "tree_sha": tree}
        if replace is not None:
            body["replace"] = replace
        return await self.signed("POST", "enroll", hotkey, body)

    async def view(self) -> dict:
        return (await self.client.get(f"/v1/competitions/{self.cid}")).json()


@pytest.fixture
async def world(orchestrator_factory, fixture_repos):
    now = [START + 5 * M]
    orch = orchestrator_factory(repos=fixture_repos, clock=lambda: now[0])
    cid = await start_and_enroll(orch, build_manifest("comp-entries"), [])
    chain = _Chain({HK_A: 750.0, HK_B: 900.0})
    guard = HotkeyAuthGuard(
        RegisteredHotkeyRegistry(chain, ttl_seconds=1e-6),
        HotkeyAuthConfig(mode="enforce", min_enroll_alpha_stake=500.0),
        verify_fn=_verify,
    )
    app = create_public_app(
        orch, hotkey_guard=guard, min_enroll_alpha_stake=500.0,
        allowed_repo_hosts=("github.com",),
    )
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://public")
    async with client:
        yield _World(orch, cid, client, chain, now)


def _events(orch, cid: str, event_type: str) -> list[dict]:
    return [
        json.loads(e["payload_json"])
        for e in repo.list_events(orch.conn, cid)
        if e["event_type"] == event_type
    ]


async def test_replace_changes_the_entry_in_place_and_is_idempotent(world: _World) -> None:
    first = await world.enroll(HK_A)
    assert first.status_code == 201, first.text
    contender_id = first.json()["contender_id"]
    assert first.json()["replaced"] is False

    again = await world.enroll(HK_A, COMMIT_2, TREE_2)
    assert again.status_code == 409
    detail = again.json()["detail"]
    assert detail["code"] == "already_enrolled"
    assert detail["contender_id"] == contender_id and detail["commit_sha"] == COMMIT_1
    assert f"contender {contender_id}" in detail["message"]
    assert COMMIT_1[:7] in detail["message"] and "--replace" in detail["message"]

    world.now[0] += M
    replaced = await world.enroll(HK_A, COMMIT_2, TREE_2, repo_url=REPO_2, replace=True)
    assert replaced.status_code == 200, replaced.text
    body = replaced.json()
    assert body["contender_id"] == contender_id
    assert body["replaced"] is True and body["unchanged"] is False
    assert body["previous_commit_sha"] == COMMIT_1
    record = repo.get_contender(world.orch.conn, contender_id)
    assert (record.repo_url, record.commit_sha, record.tree_sha) == (REPO_2, COMMIT_2, TREE_2)
    assert record.status == "ENROLLED" and record.enrollment_stake == 750.0
    assert record.updated_at == world.now[0].isoformat()
    assert record.created_at != record.updated_at
    assert len(repo.list_contenders(world.orch.conn, world.cid)) == 1
    (event,) = _events(world.orch, world.cid, entries.EVENT_ENTRY_REPLACED)
    assert event["contender_id"] == contender_id and event["hotkey"] == HK_A
    assert event["previous_commit_sha"] == COMMIT_1 and event["commit_sha"] == COMMIT_2
    assert event["previous_repo_url"] == REPO and event["repo_url"] == REPO_2

    same = await world.enroll(HK_A, COMMIT_2, TREE_2, repo_url=REPO_2, replace=True)
    assert same.status_code == 200
    assert same.json()["replaced"] is False and same.json()["unchanged"] is True
    assert len(_events(world.orch, world.cid, entries.EVENT_ENTRY_REPLACED)) == 1
    assert body["entry_commitment"] == entries.entry_commitment(
        world.cid, HK_A, REPO_2, COMMIT_2, TREE_2
    )


async def test_replace_without_an_entry_enrolls(world: _World) -> None:
    response = await world.enroll(HK_B, replace=True)
    assert response.status_code == 201, response.text
    assert response.json()["replaced"] is False
    assert [c.hotkey for c in repo.list_contenders(world.orch.conn, world.cid)] == [HK_B]


async def test_replace_keeps_the_stake_floor(world: _World) -> None:
    assert (await world.enroll(HK_A)).status_code == 201
    world.chain.stakes[HK_A] = 100.0
    response = await world.enroll(HK_A, COMMIT_2, TREE_2, replace=True)
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "hotkey_below_stake_floor"
    assert repo.list_contenders(world.orch.conn, world.cid)[0].commit_sha == COMMIT_1


async def test_withdraw_is_final_and_shows_its_reason(world: _World) -> None:
    contender_id = (await world.enroll(HK_A)).json()["contender_id"]
    for body in ({}, {"confirm": False}, {"confirm": True, "hotkey": HK_B}):
        refused = await world.signed("POST", "withdraw", HK_A, body)
        assert refused.status_code == 422, body
    # withdrawing needs a registered hotkey, not the enrollment stake
    world.chain.stakes[HK_A] = 1.0
    done = await world.signed("POST", "withdraw", HK_A, {"confirm": True})
    assert done.status_code == 200, done.text
    assert done.json()["withdrawn"] is True and done.json()["final"] is True
    assert "final" in done.json()["message"]
    assert repo.get_contender(world.orch.conn, contender_id).status == "REJECTED"
    (event,) = _events(world.orch, world.cid, entries.EVENT_WITHDRAWN)
    assert event == {
        "contender_id": contender_id, "hotkey": HK_A, "reason": "withdrawn by the miner"
    }
    (listed,) = (await world.view())["enrolled"]
    assert listed["status"] == "REJECTED" and listed["reason"] == "withdrawn by the miner"

    world.chain.stakes[HK_A] = 750.0
    for replace in (False, True):
        again = await world.enroll(HK_A, COMMIT_2, TREE_2, replace=replace)
        assert again.status_code == 409
        assert again.json()["detail"]["code"] == "entry_not_replaceable"
    twice = await world.signed("POST", "withdraw", HK_A, {"confirm": True})
    assert twice.status_code == 409
    assert twice.json()["detail"]["code"] == "entry_not_replaceable"


async def test_withdraw_without_an_entry_is_404(world: _World) -> None:
    response = await world.signed("POST", "withdraw", HK_B, {"confirm": True})
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_enrolled"


async def test_operator_rejected_entry_cannot_be_replaced(world: _World) -> None:
    contender_id = (await world.enroll(HK_A)).json()["contender_id"]
    world.orch.reject_contender(
        world.cid, contender_id, "operator", world.now[0], reason="licence review failed"
    )
    response = await world.enroll(HK_A, COMMIT_2, TREE_2, replace=True)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "entry_not_replaceable" and detail["status"] == "REJECTED"


async def test_replace_and_withdraw_close_at_the_deadline(world: _World) -> None:
    assert (await world.enroll(HK_A)).status_code == 201
    deadline = repo.get_competition(world.orch.conn, world.cid).enrollment_deadline
    world.now[0] = deadline + timedelta(seconds=1)
    replaced = await world.enroll(HK_A, COMMIT_2, TREE_2, replace=True)
    assert replaced.status_code == 409
    assert replaced.json()["detail"]["code"] == "enrollment_closed"
    withdrawn = await world.signed("POST", "withdraw", HK_A, {"confirm": True})
    assert withdrawn.status_code == 409
    assert withdrawn.json()["detail"]["code"] == "enrollment_closed"
    assert repo.list_contenders(world.orch.conn, world.cid)[0].status == "ENROLLED"
    # the entrant can still read its own entry after the deadline
    status = await world.signed("GET", "entry", HK_A)
    assert status.status_code == 200
    assert status.json()["enrollment_open"] is False


async def test_entry_status_is_the_signers_own_and_carries_the_full_reason(
    world: _World,
) -> None:
    contender_id = (await world.enroll(HK_A)).json()["contender_id"]
    (await world.enroll(HK_B, COMMIT_2, TREE_2)).raise_for_status()

    unsigned = await world.client.get(f"/v1/competitions/{world.cid}/entry")
    assert unsigned.status_code == 401

    mine = await world.signed("GET", "entry", HK_A)
    assert mine.status_code == 200, mine.text
    body = mine.json()
    assert body["hotkey"] == HK_A and body["contender_id"] == contender_id
    assert (body["repo_url"], body["commit_sha"], body["tree_sha"]) == (REPO, COMMIT_1, TREE_1)
    assert body["status"] == "ENROLLED" and body["replacements"] == 0
    assert body["reason"] is None and body["enrollment_open"] is True
    assert body["checks"] == {"fetch": "pending", "build": "disabled"}
    assert body["precheck"] is None and body["trial_build"] is None
    assert body["created_at"] and body["updated_at"]
    assert body["entry_commitment"] == entries.entry_commitment(
        world.cid, HK_A, REPO, COMMIT_1, TREE_1
    )
    other = (await world.signed("GET", "entry", HK_B)).json()
    assert other["hotkey"] == HK_B and other["commit_sha"] == COMMIT_2

    (await world.enroll(HK_A, COMMIT_2, TREE_2, replace=True)).raise_for_status()
    assert (await world.signed("GET", "entry", HK_A)).json()["replacements"] == 1

    # A multi-line build failure: public first line, entrant the full text.
    reason = "build failed (BUILD_FAILED): step 4/9 RUN pip install\nERROR: no matching dist"
    repo.record_event(
        world.orch.conn, world.cid, "contender_build_failed", world.now[0],
        payload={"contender_id": contender_id, "reason": reason},
    )
    assert (await world.signed("GET", "entry", HK_A)).json()["reason"] == reason
    listed = {e["hotkey"]: e for e in (await world.view())["enrolled"]}
    assert listed[HK_A]["reason"] == reason.splitlines()[0]


async def test_entry_status_for_a_hotkey_without_an_entry_is_404(world: _World) -> None:
    response = await world.signed("GET", "entry", HK_B)
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_enrolled"


async def test_public_listing_shows_ids_commitments_checks_and_pause(world: _World) -> None:
    contender_id = (await world.enroll(HK_A)).json()["contender_id"]
    view = await world.view()
    assert view["paused"] is False
    assert view["entry_checks"]["licenses_required"] is False  # compression
    assert view["entry_checks"]["max_worktree_bytes"] == 512 * 1024 * 1024
    (listed,) = view["enrolled"]
    expected = hashlib.sha256(
        f"{world.cid}\n{HK_A}\n{REPO}\n{COMMIT_1}\n{TREE_1}".encode()
    ).hexdigest()
    assert listed == {
        "contender_id": contender_id,
        "hotkey": HK_A,
        "status": "ENROLLED",
        "updated_at": world.now[0].isoformat(),
        "entry_commitment": expected,
        "checks": {"fetch": "pending", "build": "disabled"},
    }
    assert "results" not in view  # only once COMPLETED

    world.orch.pause(world.cid, "operator", world.now[0], reason="moving the sandbox")
    assert (await world.view())["paused"] is True


async def test_entry_reads_have_their_own_rate_limit(orchestrator_factory, fixture_repos) -> None:
    orch = orchestrator_factory(repos=fixture_repos, clock=lambda: START + 5 * M)
    cid = await start_and_enroll(orch, build_manifest("comp-entry-limit"), [])
    guard = HotkeyAuthGuard(
        RegisteredHotkeyRegistry(_Chain({HK_A: 900.0})),
        HotkeyAuthConfig(mode="enforce", min_enroll_alpha_stake=500.0),
        verify_fn=_verify,
    )
    app = create_public_app(
        orch, hotkey_guard=guard, min_enroll_alpha_stake=500.0,
        allowed_repo_hosts=("github.com",), entry_limit_per_hour=2,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://public"
    ) as client:
        codes = []
        for _ in range(3):
            path = f"/v1/competitions/{cid}/entry"
            headers = sign_request_headers(_Signer(HK_A), method="GET", path=path, body=b"")
            codes.append((await client.get(path, headers=headers)).status_code)
    assert codes == [404, 404, 429]


async def test_results_appear_only_once_completed_and_are_cached(
    orchestrator_factory, fixture_repos, tmp_path, monkeypatch
) -> None:
    from orchestrator_support import END, FINALIZATION, drive_to_completion, phase
    from vidaio.competition.states import Phase

    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest("comp-results"), ["hk-a", "hk-b"])
    app = create_public_app(
        orch, hotkey_guard=None, min_enroll_alpha_stake=500.0,
        allowed_repo_hosts=("github.com",),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://public"
    ) as client:
        from orchestrator_support import seed_items

        seed_items(orch, cid, tmp_path / "item-src")
        for at in (FINALIZATION, FINALIZATION + 2 * M, FINALIZATION + 3 * M,
                   FINALIZATION + 10 * M, FINALIZATION + 15 * M):
            await orch.step(at)
            view = (await client.get(f"/v1/competitions/{cid}")).json()
            assert "results" not in view, view["status"]  # never before COMPLETED
        assert phase(orch, cid) is Phase.AWAITING_END_TIME
        await orch.step(END + M)
        assert phase(orch, cid) is Phase.COMPLETED
        view = (await client.get(f"/v1/competitions/{cid}")).json()
        listing = (await client.get("/v1/competitions")).json()["competitions"][0]
    assert "results" not in listing  # the per-competition view only
    rows = view["results"]
    assert [r["rank"] for r in rows] == [1, 2]
    assert {r["hotkey"] for r in rows} == {"hk-a", "hk-b"}
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    for row in rows:
        assert row["contender_id"] == by_hotkey[row["hotkey"]].contender_id
        assert "items" not in row and row["speed"] is None  # no per-clip scores
        assert row["failed_items"] == 0 and row["failure_codes"] == {}
        assert row["final_score"] == pytest.approx(row["quality"])
    assert rows[0]["final_score"] >= rows[1]["final_score"]
    assert view["results_baseline"] is None and view["results_speed_term"] is None
    assert "duplicate-operator" in view["results_note"]
    # immutable once COMPLETED: computed once, served from the cache
    from vidaio.competition.orchestrator import public_results

    calls = []
    real = public_results.collect
    monkeypatch.setattr(
        public_results, "collect", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://public"
    ) as client:
        again = (await client.get(f"/v1/competitions/{cid}")).json()
    assert again["results"] == rows and calls == []
