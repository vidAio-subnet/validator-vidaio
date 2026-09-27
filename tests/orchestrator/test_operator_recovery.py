"""Operator recovery actions on a RUNNING competition (docs/COMPETITIONS.md,
"Saving a running competition"): each keeps the same competition going, is recorded
as a public append-only amendment, and never touches the anchored commitments."""

from __future__ import annotations

from datetime import timedelta

import pytest

from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.states import Phase

from orchestrator_support import (
    END,
    ENROLL_DEADLINE,
    FINALIZATION,
    M,
    START,
    FakeRunner,
    build_manifest,
    enroll,
    events_of,
    phase,
    seed_items,
    start_and_enroll,
)

RESOURCES = {"cpu": 4.0, "memory_mb": 8192, "batch_timeout_seconds": 900}


async def test_pause_stops_the_pipeline_until_cleared(orchestrator_factory, fixture_repos, tmp_path):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    assert phase(orch, cid) is Phase.BUILDING
    assert orch.pause(cid, "ops", FINALIZATION + 2 * M, reason="switching the sandbox backend")
    assert pers.is_halted(orch.conn, cid)
    await orch.step(FINALIZATION + 3 * M)
    assert phase(orch, cid) is Phase.BUILDING  # no build ran while paused
    assert orch.clear_halt(cid, "ops", FINALIZATION + 4 * M, reason="backend switched")
    await orch.step(FINALIZATION + 5 * M)
    assert phase(orch, cid) is Phase.EVALUATING
    kinds = [a["type"] for a in pers.operator_amendments(orch.conn, cid)]
    assert kinds == ["operator_paused", "orchestrator_halt_cleared"]


async def test_clear_halt_can_restore_the_requeue_budget(orchestrator_factory, fixture_repos, tmp_path):
    runner = FakeRunner(tmp_path / "work" / "outputs")
    runner.batch_fail_times = 10_000
    orch = orchestrator_factory(runner=runner, repos=fixture_repos, max_batch_requeues=1)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)
    await orch.step(FINALIZATION + 4 * M)
    await orch.step(FINALIZATION + 5 * M)
    assert pers.is_halted(orch.conn, cid)
    batch_ids = [r["batch_id"] for r in orch.conn.execute("SELECT batch_id FROM batches")]
    assert any(pers.requeue_count(orch.conn, cid, b) >= 1 for b in batch_ids)

    assert orch.clear_halt(
        cid, "ops", FINALIZATION + 6 * M, reason="runner fixed", reset_requeue_budget=True
    )
    assert all(pers.requeue_count(orch.conn, cid, b) == 0 for b in batch_ids)
    runner.batch_fail_times = 0
    await orch.step(FINALIZATION + 7 * M)
    assert phase(orch, cid) is Phase.SCORING


async def test_evaluation_rerun_resets_every_batch_behind_a_fence(
    orchestrator_factory, fixture_repos, tmp_path
):
    runner = FakeRunner(tmp_path / "work" / "outputs")
    runner.batch_fail_times = 2  # one batch burns its retries -> stays EVALUATING
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    with pytest.raises(ValueError, match="only in"):
        orch.rerun_evaluation(cid, "ops", FINALIZATION + 2 * M, reason="too early")
    await orch.step(FINALIZATION + 3 * M)
    await orch.step(FINALIZATION + 4 * M)
    assert phase(orch, cid) is Phase.EVALUATING
    done = sum(
        len(pers.outputs_for_contender(orch.conn, cid, c.contender_id))
        for c in repo.list_contenders(orch.conn, cid)
    )
    assert done > 0  # some batches already produced outputs

    orch.rerun_evaluation(cid, "ops", FINALIZATION + 4 * M, reason="GPU driver fault on the host")
    statuses = {r["status"] for r in orch.conn.execute("SELECT status FROM batches")}
    assert statuses == {"PENDING"}
    for contender in repo.list_contenders(orch.conn, cid):
        assert pers.outputs_for_contender(orch.conn, cid, contender.contender_id) == {}
    await orch.step(FINALIZATION + 5 * M)
    assert phase(orch, cid) is Phase.SCORING
    assert [a["type"] for a in pers.operator_amendments(orch.conn, cid)] == [
        "operator_evaluation_reset"
    ]


async def test_schedule_moves_later_only_and_is_public(orchestrator_factory, fixture_repos):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    now = START + 10 * M
    with pytest.raises(ValueError, match="later"):
        orch.amend_schedule(cid, "ops", now, reason="x", enrollment_deadline=ENROLL_DEADLINE - M)
    with pytest.raises(ValueError, match="enrollment_deadline <= finalization_time"):
        orch.amend_schedule(
            cid, "ops", now, reason="x", enrollment_deadline=FINALIZATION + timedelta(hours=1)
        )
    after = orch.amend_schedule(
        cid,
        "ops",
        now,
        reason="builder outage during enrollment",
        enrollment_deadline=ENROLL_DEADLINE + timedelta(hours=24),
        finalization_time=FINALIZATION + timedelta(hours=24),
        end_time=END + timedelta(hours=24),
    )
    comp = repo.get_competition(orch.conn, cid)
    assert comp.enrollment_deadline == ENROLL_DEADLINE + timedelta(hours=24)
    assert after["end_time"] == (END + timedelta(hours=24)).isoformat()
    # the old deadline no longer closes enrollment, the old finalization no longer fires
    enroll(orch, cid, "hk-b", now=ENROLL_DEADLINE + timedelta(hours=1))
    await orch.step(FINALIZATION + M)
    assert phase(orch, cid) is Phase.ENROLLING
    [event] = events_of(orch, cid, "schedule_amended")
    assert "builder outage" in event["payload_json"]


async def test_operator_can_reject_a_blocking_contender_before_it_is_built(
    orchestrator_factory, fixture_repos
):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    b = next(c for c in repo.list_contenders(orch.conn, cid) if c.hotkey == "hk-b")
    orch.reject_contender(cid, b.contender_id, "ops", START + 20 * M, reason="repository no longer readable")
    assert repo.get_contender(orch.conn, b.contender_id).status == "REJECTED"
    with pytest.raises(ValueError, match="only ENROLLED or ACCEPTED"):
        orch.reject_contender(cid, b.contender_id, "ops", START + 21 * M, reason="again")
    [amendment] = [a for a in pers.operator_amendments(orch.conn, cid) if a["type"] == "contender_rejected_by_operator"]
    assert amendment["details"]["reason"] == "repository no longer readable"


async def test_execution_envelope_can_only_be_raised_and_reruns_the_matrix(
    orchestrator_factory, fixture_repos, tmp_path
):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(sandbox_resources=RESOURCES), ["hk-a"])
    with pytest.raises(ValueError, match="only be raised"):
        orch.amend_execution(cid, "ops", START + M, reason="x", batch_timeout_seconds=600)
    effective = orch.amend_execution(
        cid, "ops", START + 2 * M, reason="clips heavier than expected", batch_timeout_seconds=1800
    )
    assert effective == {**RESOURCES, "batch_timeout_seconds": 1800}
    assert orch.effective_sandbox_resources(cid) == effective
    # the anchored manifest itself is untouched
    assert repo.get_manifest(orch.conn, cid).sandbox_resources.batch_timeout_seconds == 900


class _SlowRunner(FakeRunner):
    """Counts how many batches run at the same time."""

    def __init__(self, outputs_dir):
        import threading

        super().__init__(outputs_dir)
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.per_image: dict[str, int] = {}
        self.same_image_overlap = False

    def run_batch(self, image_digest, items, batch_index):
        import time

        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.per_image[image_digest] = self.per_image.get(image_digest, 0) + 1
            if self.per_image[image_digest] > 1:
                self.same_image_overlap = True
        try:
            time.sleep(0.2)
            return super().run_batch(image_digest, items, batch_index)
        finally:
            with self._lock:
                self.active -= 1
                self.per_image[image_digest] -= 1


@pytest.mark.parametrize("parallel,expect_overlap", [(1, False), (3, True)])
async def test_evaluation_batches_overlap_only_when_configured(
    orchestrator_factory, fixture_repos, tmp_path, parallel, expect_overlap
):
    runner = _SlowRunner(tmp_path / "work" / "outputs")
    orch = orchestrator_factory(
        runner=runner, repos=fixture_repos, evaluation_parallel_batches=parallel
    )
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)
    await orch.step(FINALIZATION + 4 * M)
    assert phase(orch, cid) is Phase.SCORING
    assert (runner.peak > 1) is expect_overlap
    assert runner.peak <= parallel
    # never two batches of one contender at once (runners retire an image's sandboxes)
    assert not runner.same_image_overlap


class _BlockingRunner(FakeRunner):
    """run_batch / build wait until the test releases them (mid-flight actions)."""

    def __init__(self, outputs_dir, *, block: str):
        import threading

        super().__init__(outputs_dir)
        self.block = block
        self.started = threading.Event()
        self.release = threading.Event()

    def run_batch(self, image_digest, items, batch_index):
        if self.block == "batch" and not self.release.is_set():
            self.started.set()
            self.release.wait(10)
        return super().run_batch(image_digest, items, batch_index)

    def build(self, contender):
        if self.block == "build" and contender.contender_id == self.block_id:
            self.started.set()
            self.release.wait(10)
        return super().build(contender)


async def _wait_started(runner) -> None:
    import asyncio

    for _ in range(200):
        if runner.started.is_set():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("runner never started")


async def test_a_rerun_during_a_running_batch_discards_its_result(
    orchestrator_factory, fixture_repos, tmp_path
):
    import asyncio

    runner = _BlockingRunner(tmp_path / "work" / "outputs", block="batch")
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)
    assert phase(orch, cid) is Phase.EVALUATING
    tick = asyncio.create_task(orch.step(FINALIZATION + 4 * M))
    await _wait_started(runner)
    orch.rerun_evaluation(cid, "ops", FINALIZATION + 4 * M, reason="driver fault")
    runner.release.set()
    await tick
    # the in-flight result was produced before the reset: it must not count
    assert {r["status"] for r in orch.conn.execute("SELECT status FROM batches")} == {"PENDING"}
    for contender in repo.list_contenders(orch.conn, cid):
        assert pers.outputs_for_contender(orch.conn, cid, contender.contender_id) == {}
    await orch.step(FINALIZATION + 5 * M)
    assert phase(orch, cid) is Phase.SCORING


async def test_a_contender_rejected_while_its_build_runs_is_never_built(
    orchestrator_factory, fixture_repos, tmp_path
):
    import asyncio

    runner = _BlockingRunner(tmp_path / "work" / "outputs", block="build")
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    b = next(c for c in repo.list_contenders(orch.conn, cid) if c.hotkey == "hk-b")
    runner.block_id = b.contender_id
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    assert phase(orch, cid) is Phase.BUILDING
    tick = asyncio.create_task(orch.step(FINALIZATION + 3 * M))
    await _wait_started(runner)
    with pytest.raises(ValueError, match="pause the competition"):
        orch.reject_contender(cid, b.contender_id, "ops", FINALIZATION + 3 * M, reason="x")
    orch.pause(cid, "ops", FINALIZATION + 3 * M, reason="reject a contender")
    orch.reject_contender(cid, b.contender_id, "ops", FINALIZATION + 3 * M, reason="repo gone")
    runner.release.set()
    await tick
    assert repo.get_contender(orch.conn, b.contender_id).status == "REJECTED"
    orch.clear_halt(cid, "ops", FINALIZATION + 4 * M, reason="resume")
    await orch.step(FINALIZATION + 5 * M)
    await orch.step(FINALIZATION + 6 * M)
    assert repo.get_contender(orch.conn, b.contender_id).status == "REJECTED"
    assert phase(orch, cid) is not Phase.BUILDING  # the others went on
