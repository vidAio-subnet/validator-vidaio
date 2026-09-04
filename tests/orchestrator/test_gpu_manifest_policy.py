"""The exact live Modal GPU must be committed by manifest.allowed_gpus."""

from __future__ import annotations

import pytest

from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.states import Phase

from orchestrator_support import (
    FINALIZATION,
    M,
    FakeRunner,
    build_manifest,
    phase,
    seed_items,
    start_and_enroll,
)


def _live_runner(tmp_path):
    runner = FakeRunner(tmp_path / "modal-output")
    runner.gpu = "L4"
    return runner


def test_live_modal_gpu_mismatch_is_rejected_before_manifest_persistence(
    orchestrator_factory, tmp_path
):
    runner = _live_runner(tmp_path)
    orch = orchestrator_factory(
        runner=runner, sandbox_backend="modal", modal_gpu="L4"
    )
    manifest = build_manifest(allowed_gpus=["A100"])

    with pytest.raises(ValueError, match="not committed in manifest.allowed_gpus"):
        orch.create_competition(manifest, FINALIZATION)
    assert repo.get_competition(orch.conn, manifest.competition_id) is None


async def test_persisted_manifest_is_rechecked_before_any_modal_build(
    orchestrator_factory, fixture_repos, tmp_path
):
    runner = _live_runner(tmp_path)
    orch = orchestrator_factory(
        runner=runner,
        repos=fixture_repos,
        sandbox_backend="modal",
        modal_gpu="L4",
    )
    manifest = build_manifest()
    cid = await start_and_enroll(orch, manifest, ["hk-a"])
    seed_items(orch, cid, tmp_path / "items-build")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    assert phase(orch, cid) is Phase.BUILDING

    runner.gpu = "A100"  # runtime/config drift after the persisted create gate
    await orch.step(FINALIZATION + 3 * M)
    assert pers.is_halted(orch.conn, cid)
    assert runner.build_calls == []


async def test_persisted_manifest_is_rechecked_before_any_modal_batch(
    orchestrator_factory, fixture_repos, tmp_path
):
    runner = _live_runner(tmp_path)
    orch = orchestrator_factory(
        runner=runner,
        repos=fixture_repos,
        sandbox_backend="modal",
        modal_gpu="L4",
    )
    manifest = build_manifest()
    cid = await start_and_enroll(orch, manifest, ["hk-a"])
    seed_items(orch, cid, tmp_path / "items-eval")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)
    assert phase(orch, cid) is Phase.EVALUATING
    assert runner.build_calls

    runner.gpu = "A100"
    await orch.step(FINALIZATION + 4 * M)
    assert pers.is_halted(orch.conn, cid)
    assert runner.batch_calls == []
