"""Fail-closed binding for competition emissions and the archived baseline."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.orchestrator.service import (
    EarningManifestError,
    reward_parameter_digest,
)
from vidaio.competition.states import Phase
from vidaio.tokenomics.config import TokenomicsConfig

from orchestrator_support import (
    BASELINE,
    CONTENDER_SHAS,
    FINALIZATION,
    M,
    START,
    T0,
    FakeRunner,
    RecordingChain,
    build_manifest,
    phase,
    repo_url,
    seed_items,
)


TOKEN = "earning-control-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _enable_emissions(orch) -> TokenomicsConfig:
    config = TokenomicsConfig(competition_emissions_enabled=True)
    orch.tokenomics = config
    return config


def _latest_halt_reason(orch, competition_id: str) -> str:
    event = next(
        event
        for event in reversed(repo.list_events(orch.conn, competition_id))
        if event["event_type"] == pers.EVENT_HALTED
    )
    return str(json.loads(event["payload_json"])["reason"])


def test_earning_create_requires_a_single_manifest_baseline(
    orchestrator_factory, fixture_repos
) -> None:
    orch = orchestrator_factory(repos=fixture_repos)
    _enable_emissions(orch)
    manifest = build_manifest("earning-needs-baseline")

    with pytest.raises(EarningManifestError, match="requires exactly one"):
        orch.create_competition(manifest, T0)
    assert repo.get_competition(orch.conn, manifest.competition_id) is None


async def test_control_refuses_a_no_baseline_earning_manifest(
    orchestrator_factory, fixture_repos
) -> None:
    orch = orchestrator_factory(
        repos=fixture_repos, control_token=TOKEN, clock=lambda: T0
    )
    _enable_emissions(orch)
    manifest = build_manifest("earning-control-needs-baseline")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=orch.control_app),
        base_url="http://control",
    ) as client:
        response = await client.post(
            "/competitions",
            headers=AUTH,
            json={"manifest": manifest.model_dump(mode="json")},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "earning_manifest_refused"
    assert repo.get_competition(orch.conn, manifest.competition_id) is None


async def test_earning_anchor_rejects_reward_and_baseline_tree_drift_before_chain(
    orchestrator_factory, fixture_repos
) -> None:
    chain = RecordingChain()
    orch = orchestrator_factory(repos=fixture_repos, chain=chain)
    config = _enable_emissions(orch)
    manifest = build_manifest("earning-anchor-bind", baseline=BASELINE)
    orch.create_competition(manifest, T0)
    baseline_image = FakeRunner.digest_for(BASELINE["tree_sha"])

    with pytest.raises(ValueError, match="reward_param_digest"):
        await orch.anchor_competition(
            manifest.competition_id,
            baseline_image_digest=baseline_image,
            reward_param_digest=hashlib.sha256(b"caller-policy").hexdigest(),
            now=T0,
        )
    with pytest.raises(ValueError, match="baseline_tree_digest"):
        await orch.anchor_competition(
            manifest.competition_id,
            baseline_image_digest=baseline_image,
            reward_param_digest=reward_parameter_digest(config),
            baseline_tree_digest=hashlib.sha256(b"different-tree").hexdigest(),
            now=T0,
        )
    assert chain.anchor_calls == []
    assert repo.get_competition(orch.conn, manifest.competition_id).commitment_root is None


async def test_built_baseline_must_match_anchored_image_before_any_contender_runs(
    orchestrator_factory, fixture_repos, tmp_path
) -> None:
    committed_image = BASELINE["image_digest"]
    rebuilt_image = hashlib.sha256(b"different-lifecycle-image").hexdigest()

    class DriftingBaselineRunner(FakeRunner):
        def build(self, contender):
            if contender.tree_sha == BASELINE["tree_sha"]:
                self.build_calls.append(contender.contender_id)
                return committed_image if len(self.build_calls) == 1 else rebuilt_image
            return super().build(contender)

    runner = DriftingBaselineRunner(tmp_path / "outputs")
    chain = RecordingChain()
    orch = orchestrator_factory(
        runner=runner, repos=fixture_repos, chain=chain
    )
    config = _enable_emissions(orch)
    manifest = build_manifest("earning-baseline-image-bind", baseline=BASELINE)
    cid = manifest.competition_id
    orch.create_competition(manifest, T0)
    await orch.anchor_competition(
        cid,
        baseline_image_digest=committed_image,
        reward_param_digest=reward_parameter_digest(config),
        now=T0,
    )
    await orch.step(START)
    commit_sha, tree_sha = CONTENDER_SHAS["hk-a"]
    orch.enroll_contender(
        cid,
        hotkey="hk-a",
        repo_url=repo_url("hk-a"),
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        stake=1000.0,
        now=START + M,
    )
    seed_items(orch, cid, tmp_path / "items")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    assert phase(orch, cid) is Phase.BUILDING
    baseline = next(
        contender
        for contender in repo.list_contenders(orch.conn, cid)
        if contender.is_calibration
    )

    await orch.step(FINALIZATION + 3 * M)

    assert pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.BUILDING
    assert runner.build_calls == [0, baseline.contender_id]
    assert "does not match the pre-enrollment commitment" in _latest_halt_reason(
        orch, cid
    )


async def test_active_reward_policy_drift_halts_before_any_build(
    orchestrator_factory, fixture_repos, tmp_path
) -> None:
    runner = FakeRunner(tmp_path / "outputs")
    chain = RecordingChain()
    orch = orchestrator_factory(
        runner=runner, repos=fixture_repos, chain=chain
    )
    config = _enable_emissions(orch)
    manifest = build_manifest("earning-policy-drift", baseline=BASELINE)
    cid = manifest.competition_id
    orch.create_competition(manifest, T0)
    await orch.anchor_competition(
        cid,
        baseline_image_digest=FakeRunner.digest_for(BASELINE["tree_sha"]),
        reward_param_digest=reward_parameter_digest(config),
        now=T0,
    )
    await orch.step(START)
    commit_sha, tree_sha = CONTENDER_SHAS["hk-a"]
    orch.enroll_contender(
        cid,
        hotkey="hk-a",
        repo_url=repo_url("hk-a"),
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        stake=1000.0,
        now=START + M,
    )
    seed_items(orch, cid, tmp_path / "items")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    assert phase(orch, cid) is Phase.BUILDING

    orch.tokenomics = TokenomicsConfig(
        competition_emissions_enabled=True,
        ewma_decay=0.5,
    )
    await orch.step(FINALIZATION + 3 * M)

    assert pers.is_halted(orch.conn, cid)
    assert runner.build_calls == [0]
    assert "active reward policy" in _latest_halt_reason(orch, cid)


async def _earning_competition_at_building(
    orch, runner, tmp_path, *, competition_id: str
) -> tuple[str, int]:
    config = _enable_emissions(orch)
    manifest = build_manifest(competition_id, baseline=BASELINE)
    cid = manifest.competition_id
    orch.create_competition(manifest, T0)
    await orch.anchor_competition(
        cid,
        baseline_image_digest=FakeRunner.digest_for(BASELINE["tree_sha"]),
        reward_param_digest=reward_parameter_digest(config),
        now=T0,
    )
    await orch.step(START)
    commit_sha, tree_sha = CONTENDER_SHAS["hk-a"]
    orch.enroll_contender(
        cid,
        hotkey="hk-a",
        repo_url=repo_url("hk-a"),
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        stake=1000.0,
        now=START + M,
    )
    seed_items(orch, cid, tmp_path / f"{competition_id}-items")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    baseline = next(
        contender
        for contender in repo.list_contenders(orch.conn, cid)
        if contender.is_calibration
    )
    assert phase(orch, cid) is Phase.BUILDING
    return cid, baseline.contender_id


async def test_baseline_build_fault_halts_instead_of_deadlocking_without_baseline(
    orchestrator_factory, fixture_repos, tmp_path
) -> None:
    runner = FakeRunner(tmp_path / "baseline-build-output")
    orch = orchestrator_factory(
        runner=runner,
        repos=fixture_repos,
        chain=RecordingChain(),
    )
    cid, baseline_id = await _earning_competition_at_building(
        orch, runner, tmp_path, competition_id="earning-baseline-build-fault"
    )
    runner.fail_build_for.add(BASELINE["repo_url"])

    await orch.step(FINALIZATION + 3 * M)

    assert pers.is_halted(orch.conn, cid)
    baseline = repo.get_contender(orch.conn, baseline_id)
    assert baseline is not None and baseline.status == "ACCEPTED"
    assert phase(orch, cid) is Phase.BUILDING
    assert "baseline image failed to build" in _latest_halt_reason(orch, cid)


async def test_baseline_probe_fault_halts_instead_of_deadlocking_without_baseline(
    orchestrator_factory, fixture_repos, tmp_path
) -> None:
    runner = FakeRunner(tmp_path / "baseline-probe-output")
    orch = orchestrator_factory(
        runner=runner,
        repos=fixture_repos,
        chain=RecordingChain(),
    )
    cid, baseline_id = await _earning_competition_at_building(
        orch, runner, tmp_path, competition_id="earning-baseline-probe-fault"
    )
    runner.fail_probe_for.add(BASELINE["tree_sha"])

    await orch.step(FINALIZATION + 3 * M)

    assert pers.is_halted(orch.conn, cid)
    baseline = repo.get_contender(orch.conn, baseline_id)
    assert baseline is not None and baseline.status == "ACCEPTED"
    assert phase(orch, cid) is Phase.BUILDING
    assert "baseline failed its isolation probe" in _latest_halt_reason(orch, cid)
