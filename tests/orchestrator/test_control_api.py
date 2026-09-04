"""Competition control API + the single anchor path + tokenomics results (#11).

Before this the live stack ran an orchestrator nobody could reach, anchoring only
wrote SQLite, and no code turned a finished competition into a CompetitionResult.
These tests drive the whole graph through HTTP exactly as an operator (or the
local stack) would, and assert that the anchor payload actually reached the
INJECTED ChainAdapter — the same call path chainless report mode and the future
real chain share.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from vidaio.audit import NotConfiguredError
from vidaio.audit.canonical import sha256_hex
from vidaio.audit.commitments import (
    COMMITMENT_DOMAIN,
    pin_git_sha,
    reward_parameter_digest,
)
from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.orchestrator.control import create_control_app
from vidaio.competition.orchestrator.results import (
    ResultNotReady,
    build_competition_result,
)
from vidaio.competition.states import Phase
from vidaio.tokenomics.config import TokenomicsConfig

from orchestrator_support import (
    BASELINE,
    CONTENDER_SHAS,
    END,
    FINALIZATION,
    M,
    START,
    T0,
    RecordingChain,
    build_manifest,
    materialize_baseline,
    phase,
    repo_url,
    seed_items,
)

TOKEN = "control-token-for-tests"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BASELINE_IMAGE_DIGEST = str(BASELINE["image_digest"])
REWARD_PARAM_DIGEST = reward_parameter_digest(TokenomicsConfig())
ACTIVE_BASELINE = SimpleNamespace(version=0, artifact_digest="ab" * 32)


class Clock:
    """Explicit, movable test clock — the orchestrator never reads wall time."""

    def __init__(self, value=T0) -> None:
        self.value = value

    def __call__(self):
        return self.value


@pytest.fixture
def control(orchestrator_factory, fixture_repos):
    clock = Clock()
    chain = RecordingChain()
    orch = orchestrator_factory(
        repos=fixture_repos, chain=chain, clock=clock, control_token=TOKEN
    )
    assert orch.control_app is not None
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=orch.control_app),
        base_url="http://control",
    )
    return orch, chain, clock, client


# ---- auth -----------------------------------------------------------------------


async def test_every_route_requires_the_token(control):
    _orch, _chain, _clock, client = control
    async with client:
        for method, path in [
            ("post", "/competitions"),
            ("post", "/competitions/comp-orch/contenders"),
            ("post", "/competitions/comp-orch/anchor"),
            ("post", "/competitions/comp-orch/anchor/release"),
            ("post", "/competitions/comp-orch/halt/clear"),
            ("get", "/competitions/comp-orch"),
            ("post", "/competitions/comp-orch/review"),
            ("get", "/competitions/comp-orch/result"),
        ]:
            response = await client.request(method.upper(), path, json={})
            assert response.status_code == 401, (method, path)
        wrong = await client.get(
            "/competitions/comp-orch", headers={"Authorization": "Bearer nope"}
        )
        assert wrong.status_code == 401


def test_control_app_refuses_to_exist_without_a_token(
    orchestrator_factory, fixture_repos
):
    orch = orchestrator_factory(repos=fixture_repos)
    assert orch.control_app is None  # fail closed: not served at all
    with pytest.raises(NotConfiguredError):
        create_control_app(orch, token="")


async def test_authenticated_clear_halt_records_operator_and_reason(control) -> None:
    orch, _chain, clock, client = control
    manifest = build_manifest("comp-clear-halt-control")
    cid = manifest.competition_id
    orch.create_competition(manifest, T0)
    assert pers.record_halt(orch.conn, cid, "synthetic infrastructure blocker", T0)

    async with client:
        blank = await client.post(
            f"/competitions/{cid}/halt/clear",
            headers=AUTH,
            json={"operator": "   ", "reason": "   "},
        )
        assert blank.status_code == 422
        assert pers.is_halted(orch.conn, cid)

        clock.value = T0 + M
        response = await client.post(
            f"/competitions/{cid}/halt/clear",
            headers=AUTH,
            json={
                "operator": "  ops@vidaio  ",
                "reason": "  Modal capacity and input pool integrity verified  ",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json() == {"competition_id": cid, "cleared": True}
        assert not pers.is_halted(orch.conn, cid)
        status = await client.get(f"/competitions/{cid}", headers=AUTH)
        assert status.status_code == 200
        assert status.json()["halted"] is False

        repeated = await client.post(
            f"/competitions/{cid}/halt/clear",
            headers=AUTH,
            json={"operator": "ops@vidaio", "reason": "duplicate acknowledgement"},
        )
        assert repeated.status_code == 200
        assert repeated.json()["cleared"] is False

    events = [
        event
        for event in repo.list_events(orch.conn, cid)
        if event["event_type"] == pers.EVENT_HALT_CLEARED
    ]
    assert len(events) == 1
    assert json.loads(events[0]["payload_json"]) == {
        "operator": "ops@vidaio",
        "reason": "Modal capacity and input pool integrity verified",
    }


# ---- happy path -----------------------------------------------------------------


async def test_full_competition_through_the_control_api(
    control, fixture_repos, tmp_path
):
    orch, chain, clock, client = control
    manifest = build_manifest(baseline=materialize_baseline(orch, fixture_repos))
    cid = manifest.competition_id

    async with client:
        # -- create ----------------------------------------------------------
        created = await client.post(
            "/competitions",
            headers=AUTH,
            json={"manifest": manifest.model_dump(mode="json")},
        )
        assert created.status_code == 201, created.text
        assert created.json() == {"competition_id": cid, "status": "SCHEDULED"}

        # Evaluation bytes are provisioned before the manifest commitment is
        # anchored; the deployed control path refuses an empty item matrix.
        seed_items(orch, cid, tmp_path / "item-src")

        bad = await client.post(
            "/competitions", headers=AUTH, json={"manifest": {"competition_id": "x"}}
        )
        assert bad.status_code == 422

        # -- anchor THROUGH the injected ChainAdapter ------------------------
        anchored = await client.post(
            f"/competitions/{cid}/anchor",
            headers=AUTH,
            json={
                "reward_param_digest": REWARD_PARAM_DIGEST,
            },
        )
        assert anchored.status_code == 200, anchored.text
        body = anchored.json()

        # The adapter received the exact payload bytes, and they are the
        # domain-tagged root of the PERSISTED manifest — not anything the caller
        # supplied.
        assert len(chain.anchor_calls) == 1
        payload = chain.anchor_calls[0]
        assert payload == bytes.fromhex(body["payload_hex"])
        assert payload.decode() == f"{COMMITMENT_DOMAIN}:competition:{body['root']}"
        assert chain.anchored == [payload]
        assert body["tx_id"].startswith("0x")
        assert body["baseline_image_digest"] == BASELINE_IMAGE_DIGEST
        assert body["anchor_block"] == 1
        assert body["anchor_block_hash"] == chain.block_hash(1)
        assert body["finalized_block"] >= body["anchor_block"]
        assert body["archive_verified"] is True
        assert body["write_response_recovered"] is False
        assert manifest.manifest_digest() in body["canonical_json"]
        assert pin_git_sha(BASELINE["tree_sha"]) in body["canonical_json"]
        assert manifest.scoring_seed_commitment in body["canonical_json"]

        # ... and only then was it recorded in SQLite.
        comp = repo.get_competition(orch.conn, cid)
        assert comp.commitment_root == body["root"]
        assert len(pers.contender_fault_events(orch.conn, cid)) == 0
        anchor_events = [
            e
            for e in repo.list_events(orch.conn, cid)
            if e["event_type"] == pers.EVENT_COMMITMENT_ANCHORED
        ]
        assert (
            len(anchor_events) == 1
            and body["tx_id"] in anchor_events[0]["payload_json"]
        )
        status = (await client.get(f"/competitions/{cid}", headers=AUTH)).json()
        assert status["anchor_receipt"]["anchor_block"] == body["anchor_block"]
        assert status["anchor_receipt"]["archive_verified"] is True

        # A second anchor is refused BEFORE the chain is touched (review round 2,
        # new-7): it used to write again and only then discover it was recorded,
        # leaving a second untracked commitment on chain.
        again = await client.post(
            f"/competitions/{cid}/anchor",
            headers=AUTH,
            json={
                "baseline_image_digest": BASELINE_IMAGE_DIGEST,
                "reward_param_digest": REWARD_PARAM_DIGEST,
            },
        )
        assert again.status_code == 409
        assert again.json()["detail"]["code"] == "already_anchored"
        assert len(chain.anchor_calls) == 1  # still exactly one chain write

        # -- enroll ----------------------------------------------------------
        clock.value = START
        await orch.step(START)
        assert phase(orch, cid) is Phase.ENROLLING
        clock.value = START + 5 * M
        ids = {}
        for hotkey in ("hk-a", "hk-b"):
            commit_sha, tree_sha = CONTENDER_SHAS[hotkey]
            enrolled = await client.post(
                f"/competitions/{cid}/contenders",
                headers=AUTH,
                json={
                    "hotkey": hotkey,
                    "repo_url": repo_url(hotkey),
                    "commit_sha": commit_sha,
                    "tree_sha": tree_sha,
                    "stake": 1000.0,
                },
            )
            assert enrolled.status_code == 201, enrolled.text
            ids[hotkey] = enrolled.json()["contender_id"]
        assert len(set(ids.values())) == 2

        duplicate = await client.post(
            f"/competitions/{cid}/contenders",
            headers=AUTH,
            json={
                "hotkey": "hk-a",
                "repo_url": repo_url("hk-a"),
                "commit_sha": CONTENDER_SHAS["hk-a"][0],
                "tree_sha": CONTENDER_SHAS["hk-a"][1],
                "stake": 1000.0,
            },
        )
        assert duplicate.status_code == 409

        # -- status ----------------------------------------------------------
        status = (await client.get(f"/competitions/{cid}", headers=AUTH)).json()
        assert status["status"] == "ENROLLING"
        assert status["halted"] is False
        assert status["commitment_root"] == body["root"]
        assert {c["hotkey"] for c in status["contenders"]} == {"hk-a", "hk-b"}
        assert status["podium"] == []

        assert (await client.get("/competitions/nope", headers=AUTH)).status_code == 404

        # -- result is refused before completion ------------------------------
        early = await client.get(f"/competitions/{cid}/result", headers=AUTH)
        assert early.status_code == 409
        assert early.json()["detail"]["code"] == "not_completed"

        # -- run the competition ---------------------------------------------
        await orch.step(FINALIZATION)
        await orch.step(FINALIZATION + 2 * M)
        await orch.step(FINALIZATION + 3 * M)
        await orch.step(FINALIZATION + 10 * M)
        await orch.step(FINALIZATION + 15 * M)
        assert phase(orch, cid) is Phase.AWAITING_END_TIME

        # -- review ------------------------------------------------------------
        clock.value = FINALIZATION + 16 * M
        reviewed = await client.post(
            f"/competitions/{cid}/review",
            headers=AUTH,
            json={
                "contender_id": ids["hk-b"],
                "action": "DISQUALIFY",
                "reviewer": "ops@vidaio",
                "reason": "manual audit found a licence violation",
            },
        )
        assert reviewed.status_code == 201, reviewed.text
        assert reviewed.json()["review_id"] >= 1
        assert repo.verify_review_chain(orch.conn, cid)

        bogus = await client.post(
            f"/competitions/{cid}/review",
            headers=AUTH,
            json={
                "contender_id": ids["hk-a"],
                "action": "NOT-AN-ACTION",
                "reviewer": "ops@vidaio",
                "reason": "typo",
            },
        )
        assert bogus.status_code == 409

        # -- complete + result --------------------------------------------------
        from vidaio.chain.adapter import ChainNeuron

        chain.set_neurons(
            [
                ChainNeuron(
                    uid=7,
                    hotkey="hk-a",
                    coldkey="ck-a",
                    ip="1.2.3.4",
                    alpha_stake=1.0,
                    emission=0.0,
                ),
                ChainNeuron(
                    uid=8,
                    hotkey="hk-b",
                    coldkey="ck-b",
                    ip="1.2.3.5",
                    alpha_stake=1.0,
                    emission=0.0,
                ),
            ]
        )
        clock.value = END + M
        await orch.step(END + M)
        assert phase(orch, cid) is Phase.COMPLETED
        chain.block_time_anchor = (chain.finalized_block(), END + M)

        result = await client.get(f"/competitions/{cid}/result", headers=AUTH)
        assert result.status_code == 200, result.text
        payload = result.json()
        assert payload["cycle"] == 1
        assert payload["competition_id"] == cid
        assert payload["track"] == "compression"
        assert payload["baseline_score"] is not None and payload["baseline_score"] > 0
        assert payload["baseline_version"] >= 0
        assert len(payload["baseline_artifact_digest"]) == 64
        # Manual review is operational only; earning order comes from packets.
        assert {c["hotkey"] for c in payload["contenders"]} == {"hk-a", "hk-b"}
        assert {c["uid"] for c in payload["contenders"]} == {7, 8}
        assert payload["source"] == "packet_mean.current_census_preview.v1"
        assert payload["authoritative_emitted_result"] is False
        assert payload["contenders"][0]["score"] is not None
        status = (await client.get(f"/competitions/{cid}", headers=AUTH)).json()
        assert [p["hotkey"] for p in status["podium"]] == ["hk-a"]


# ---- anchoring failure modes ----------------------------------------------------


async def test_anchor_without_a_chain_adapter_is_503_and_records_nothing(
    orchestrator_factory, fixture_repos
):
    clock = Clock()
    orch = orchestrator_factory(
        repos=fixture_repos, clock=clock, control_token=TOKEN
    )  # no chain
    manifest = build_manifest(baseline=BASELINE)
    cid = manifest.competition_id
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=orch.control_app), base_url="http://control"
    ) as client:
        await client.post(
            "/competitions",
            headers=AUTH,
            json={"manifest": manifest.model_dump(mode="json")},
        )
        response = await client.post(
            f"/competitions/{cid}/anchor",
            headers=AUTH,
            json={
                "baseline_image_digest": BASELINE_IMAGE_DIGEST,
                "reward_param_digest": REWARD_PARAM_DIGEST,
            },
        )
    assert response.status_code == 503
    assert repo.get_competition(orch.conn, cid).commitment_root is None
    # Enrollment stays closed: the SCHEDULED->ENROLLING guard needs a root.
    await orch.step(START)
    assert phase(orch, cid) is Phase.SCHEDULED


async def test_schema_v14_anchor_refuses_a_manifest_without_a_baseline(
    orchestrator_factory, fixture_repos, tmp_path
):
    clock = Clock()
    chain = RecordingChain()
    orch = orchestrator_factory(
        repos=fixture_repos, chain=chain, clock=clock, control_token=TOKEN
    )
    manifest = build_manifest()  # baseline=None
    cid = manifest.competition_id
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=orch.control_app), base_url="http://control"
    ) as client:
        await client.post(
            "/competitions",
            headers=AUTH,
            json={"manifest": manifest.model_dump(mode="json")},
        )
        missing = await client.post(
            f"/competitions/{cid}/anchor",
            headers=AUTH,
            json={
                "baseline_image_digest": BASELINE_IMAGE_DIGEST,
                "reward_param_digest": REWARD_PARAM_DIGEST,
            },
        )
        assert missing.status_code == 422
        assert chain.anchor_calls == []

        seed_items(orch, cid, tmp_path / "item-src")

        supplied = await client.post(
            f"/competitions/{cid}/anchor",
            headers=AUTH,
            json={
                "baseline_image_digest": BASELINE_IMAGE_DIGEST,
                "reward_param_digest": REWARD_PARAM_DIGEST,
                "baseline_tree_digest": sha256_hex(b"explicit-baseline-tree"),
            },
        )
        assert supplied.status_code == 422
        assert "schema-v14 anchoring requires" in supplied.text
        assert chain.anchor_calls == []


async def test_a_failed_chain_write_leaves_nothing_anchored(
    orchestrator_factory, fixture_repos
):
    class BrokenChain(RecordingChain):
        async def anchor_commitment(self, payload: bytes) -> str:
            self.anchor_calls.append(bytes(payload))
            raise OSError("substrate node unreachable")

    from vidaio.competition.orchestrator import AnchorError

    chain = BrokenChain()
    orch = orchestrator_factory(
        repos=fixture_repos, chain=chain, clock=Clock(), control_token=TOKEN
    )
    manifest = build_manifest(baseline=BASELINE)
    cid = manifest.competition_id
    orch.create_competition(manifest, T0)
    with pytest.raises(AnchorError):
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_PARAM_DIGEST,
            now=T0,
        )
    assert len(chain.anchor_calls) == 1  # ambiguous writes are never blind-retried
    assert repo.get_competition(orch.conn, cid).commitment_root is None


# ---- build_competition_result semantics -----------------------------------------


async def test_build_competition_result_values(control, tmp_path):
    orch, chain, clock, client = control
    manifest = build_manifest(baseline=BASELINE)
    cid = manifest.competition_id
    orch.create_competition(manifest, T0)
    await orch.anchor_competition(
        cid,
        baseline_image_digest=BASELINE_IMAGE_DIGEST,
        reward_param_digest=REWARD_PARAM_DIGEST,
        now=T0,
    )
    await orch.step(START)
    for hotkey in ("hk-a", "hk-b"):
        commit_sha, tree_sha = CONTENDER_SHAS[hotkey]
        orch.enroll_contender(
            cid,
            hotkey=hotkey,
            repo_url=repo_url(hotkey),
            commit_sha=commit_sha,
            tree_sha=tree_sha,
            stake=1000.0,
            now=START + 5 * M,
        )
    seed_items(orch, cid, tmp_path / "item-src")

    with pytest.raises(ResultNotReady):
        build_competition_result(
            orch.conn,
            cid,
            applied_at=END + 2 * M,
            active_baseline=ACTIVE_BASELINE,
            uid_by_hotkey={"hk-a": 7, "hk-b": 9},
        )

    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)
    await orch.step(FINALIZATION + 10 * M)
    await orch.step(FINALIZATION + 15 * M)
    await orch.step(END + M)
    assert phase(orch, cid) is Phase.COMPLETED

    result = build_competition_result(
        orch.conn,
        cid,
        applied_at=END + 2 * M,
        active_baseline=ACTIVE_BASELINE,
        uid_by_hotkey={"hk-a": 7, "hk-b": 9},
    )
    assert result.cycle == 1
    assert result.applied_at == END + 2 * M
    assert result.baseline_version == 0
    assert result.baseline_artifact_digest == "ab" * 32

    # Ranked best-first, baseline absent by construction, uids from the snapshot.
    assert [c.hotkey for c in result.contenders] == ["hk-a", "hk-b"]
    assert [c.uid for c in result.contenders] == [7, 9]
    assert all("baseline" not in (c.hotkey or "") for c in result.contenders)

    # baseline_score is the archived baseline row's persisted final_score.
    baseline_row = next(
        c for c in repo.list_contenders(orch.conn, cid) if c.is_calibration
    )
    assert result.baseline_score == pytest.approx(baseline_row.final_score)

    assert [c.score for c in result.contenders] == pytest.approx(
        sorted((c.final_score for c in repo.ranking(orch.conn, cid)), reverse=True)
    )

    # An unknown hotkey is refused; no sentinel can become payable.
    with pytest.raises(ResultNotReady, match="close-block uid"):
        build_competition_result(
            orch.conn,
            cid,
            applied_at=END + 2 * M,
            active_baseline=ACTIVE_BASELINE,
            uid_by_hotkey={},
        )


async def test_cycle_without_executable_baseline_cannot_be_anchored(control, tmp_path):
    """A cycle without its exact baseline can never reach earning evidence."""
    orch, _chain, _clock, _client = control
    manifest = build_manifest()  # no baseline
    cid = manifest.competition_id
    orch.create_competition(manifest, T0)
    with pytest.raises(ValueError, match="schema-v14 anchoring requires"):
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_PARAM_DIGEST,
            baseline_tree_digest=hashlib.sha256(b"none").hexdigest(),
            now=T0,
        )


def test_result_uids_come_from_the_chain_snapshot(orchestrator_factory, fixture_repos):
    """build_result resolves uids from the injected adapter, never invents them."""
    from vidaio.chain.adapter import ChainNeuron

    chain = RecordingChain()
    chain.set_neurons(
        [
            ChainNeuron(
                uid=3,
                hotkey="hk-a",
                coldkey="ck",
                ip="1.2.3.4",
                alpha_stake=1.0,
                emission=0.0,
            )
        ]
    )
    orch = orchestrator_factory(repos=fixture_repos, chain=chain)
    assert {n.hotkey: n.uid for n in orch.chain.neurons()} == {"hk-a": 3}
    with pytest.raises(ResultNotReady):
        orch.build_result("never-created")
