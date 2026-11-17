"""The Modal GPU of a competition comes from its anchored manifest.allowed_gpus.

Exactly one entry is that GPU; with several the process default
``orchestrator.modal_gpu`` is used when it is listed. The renewable runner is
configured per competition and must attest exactly that GPU before any build or
batch; a fixed runner cannot follow the manifest and must already match it."""

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


def test_fixed_runner_gpu_mismatch_is_rejected_before_manifest_persistence(
    orchestrator_factory, tmp_path
):
    runner = _live_runner(tmp_path)
    orch = orchestrator_factory(
        runner=runner, sandbox_backend="modal", modal_gpu="L4"
    )
    manifest = build_manifest(allowed_gpus=["A100"])

    with pytest.raises(ValueError, match="is not the competition GPU 'A100'"):
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


# ---- per-competition GPU -------------------------------------------------------

STREAMING = {"bits_per_pixel": 0.05, "max_keyframe_interval_seconds": 2.0,
             "peak_ratio": 1.5, "peak_window_seconds": 2.0}
SPEED = {"crown_margin": 0.05, "speed_weight": 0.3, "speed_fast_spf": 1 / 30,
         "speed_slow_spf": 2.0, "speed_inserted_frames": 1200}
RESOURCES = {"cpu": 8.0, "memory_mb": 32768, "batch_timeout_seconds": 1800}


class _GpuInner(FakeRunner):
    """What the Modal generation factory builds: the envelope's GPU is requested."""


def _renewable(tmp_path, *, honour_gpu=True):
    from vidaio.competition.runners.renewable import RenewableRunner

    made: list[_GpuInner] = []

    def factory(_sequence, resources=None):
        inner = _GpuInner(tmp_path / "work" / "outputs")
        inner.resources = resources
        inner.gpu = (resources or {}).get("gpu", "L4") if honour_gpu else "L4"
        made.append(inner)
        return inner

    runner = RenewableRunner(factory, static={"gpu": "L4"}, idle_grace_seconds=0.0)
    return runner, made


@pytest.mark.parametrize(
    ("allowed", "default", "expected"),
    [
        (["A100-80GB"], "L4", "A100-80GB"),
        (["L4"], "A100-80GB", "L4"),
        (["L4", "A100-80GB"], "A100-80GB", "A100-80GB"),
        (["A100-80GB", "L4"], "L4", "L4"),
    ],
)
def test_competition_gpu_comes_from_the_manifest(
    orchestrator_factory, tmp_path, allowed, default, expected
):
    orch = orchestrator_factory(
        runner=_live_runner(tmp_path), sandbox_backend="modal", modal_gpu=default
    )
    assert orch.competition_gpu(build_manifest(allowed_gpus=allowed)) == expected


def test_several_gpus_without_the_default_cannot_be_resolved(
    orchestrator_factory, tmp_path
):
    runner, _made = _renewable(tmp_path)
    orch = orchestrator_factory(runner=runner, sandbox_backend="modal", modal_gpu="L4")
    manifest = build_manifest(allowed_gpus=["A100-80GB", "H100"])
    with pytest.raises(ValueError, match="lists several GPUs"):
        orch.competition_gpu(manifest)
    with pytest.raises(ValueError, match="lists several GPUs"):
        orch.create_competition(manifest, FINALIZATION)
    assert repo.get_competition(orch.conn, manifest.competition_id) is None


@pytest.mark.parametrize("backend", ["modal", "docker"], ids=["modal", "local-sandbox"])
def test_speed_term_with_several_gpus_is_refused_at_creation(
    orchestrator_factory, tmp_path, backend
):
    runner, _made = _renewable(tmp_path)
    orch = orchestrator_factory(runner=runner, sandbox_backend=backend, modal_gpu="L4")
    manifest = build_manifest(
        compression_scoring="streaming", streaming_rules=STREAMING, result_rules=SPEED,
        sandbox_resources=RESOURCES, allowed_gpus=["L4", "A100-80GB"],
    )
    with pytest.raises(ValueError, match="must name exactly one"):
        orch.create_competition(manifest, FINALIZATION)
    assert repo.get_competition(orch.conn, manifest.competition_id) is None
    # one pinned GPU is fine, even when it is not the process default
    pinned = build_manifest(
        "comp-orch-pinned", compression_scoring="streaming", streaming_rules=STREAMING,
        result_rules=SPEED, sandbox_resources=RESOURCES, allowed_gpus=["A100-80GB"],
    )
    orch.create_competition(pinned, FINALIZATION)
    assert repo.get_competition(orch.conn, pinned.competition_id) is not None


async def test_competition_runs_on_its_manifest_gpu_not_the_process_default(
    orchestrator_factory, fixture_repos, tmp_path
):
    runner, made = _renewable(tmp_path)
    orch = orchestrator_factory(
        runner=runner, repos=fixture_repos, sandbox_backend="modal", modal_gpu="L4"
    )
    # Creation no longer fails because the process default differs.
    manifest = build_manifest(allowed_gpus=["A100-80GB"])
    cid = await start_and_enroll(orch, manifest, ["hk-a"])
    assert runner.gpu == "L4"  # idle: nothing configured yet
    assert orch.runner_envelope(cid)["gpu"] == "A100-80GB"
    seed_items(orch, cid, tmp_path / "items-a100")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)

    assert not pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.EVALUATING
    assert len(made) == 1 and made[0].resources["gpu"] == "A100-80GB"
    assert made[0].gpu == runner.gpu == "A100-80GB"
    assert made[0].build_calls


async def test_generation_that_ignores_the_competition_gpu_halts_before_building(
    orchestrator_factory, fixture_repos, tmp_path
):
    runner, made = _renewable(tmp_path, honour_gpu=False)
    orch = orchestrator_factory(
        runner=runner, repos=fixture_repos, sandbox_backend="modal", modal_gpu="L4"
    )
    cid = await start_and_enroll(orch, build_manifest(allowed_gpus=["A100-80GB"]), ["hk-a"])
    seed_items(orch, cid, tmp_path / "items-ignored")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)

    assert pers.is_halted(orch.conn, cid)
    assert len(made) == 1 and made[0].build_calls == []
    [halt] = [e for e in repo.list_events(orch.conn, cid) if e["event_type"] == pers.EVENT_HALTED]
    assert "is not the competition GPU 'A100-80GB'" in halt["payload_json"]


def test_renewable_runner_reports_the_configured_then_live_gpu(tmp_path):
    runner, made = _renewable(tmp_path)
    assert runner.gpu == "L4"  # static process default while unconfigured
    runner.configure({"gpu": "A100-80GB"})
    assert runner.gpu == "A100-80GB" and made == []  # what the next generation requests
    runner.acquire()
    assert made[0].gpu == runner.gpu == "A100-80GB"
    runner.configure({"gpu": "L4"})
    assert runner.gpu == "L4"  # stale generation is replaced on the next acquire
    runner.acquire()
    assert len(made) == 2 and made[1].gpu == "L4"
