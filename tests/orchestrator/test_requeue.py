"""Retry / requeue / halt semantics (spec §14): transient failures retry with a
bounded budget; exhaustion HALTS the pipeline with a CRITICAL log — the
competition itself is never failed by infra."""

from __future__ import annotations

import logging

from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.states import Phase

from orchestrator_support import (
    END,
    FINALIZATION,
    M,
    AlwaysFailScoringClient,
    FakeRunner,
    FakeScoringClient,
    FlakyScoringClient,
    build_manifest,
    client_conn,
    events_of,
    phase,
    seed_items,
    start_and_enroll,
)


async def _drive_to_scoring(orch, tmp_path):
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)
    await orch.step(FINALIZATION + 10 * M)
    assert phase(orch, cid) is Phase.SCORING
    return cid


async def test_scoring_flaky_then_success_recorded_once(
    orchestrator_factory, fixture_repos, tmp_path
):
    client = FlakyScoringClient(fail_times=2)  # < scoring_retry_attempts (3)
    orch = orchestrator_factory(scoring_client=client, repos=fixture_repos)
    cid = await _drive_to_scoring(orch, tmp_path)
    await orch.step(FINALIZATION + 15 * M)
    assert phase(orch, cid) is Phase.AWAITING_END_TIME
    rows = orch.conn.execute(
        "SELECT contender_id, item_id, COUNT(*) AS n FROM performance_history"
        " WHERE competition_id = ? GROUP BY contender_id, item_id",
        (cid,),
    ).fetchall()
    assert len(rows) == 2 * 3 and all(r["n"] == 1 for r in rows)  # each exactly once
    # The first pair burned the two flaky attempts plus its success.
    assert len(client.calls) == 2 + 2 * 3
    assert not pers.is_halted(orch.conn, cid)


async def test_scoring_exhausted_halts_critical_not_failed(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    client = AlwaysFailScoringClient()
    orch = orchestrator_factory(scoring_client=client, repos=fixture_repos)
    cid = await _drive_to_scoring(orch, tmp_path)
    with caplog.at_level(logging.CRITICAL):
        await orch.step(FINALIZATION + 15 * M)
    assert any(r.levelno == logging.CRITICAL for r in caplog.records)

    # Halted, NOT failed: phase untouched, halt recorded in the event log.
    assert phase(orch, cid) is Phase.SCORING
    assert pers.is_halted(orch.conn, cid)
    assert len(events_of(orch, cid, "orchestrator_halted")) == 1

    # While halted, steps do no pipeline work (no further scoring attempts).
    calls_before = len(client.calls)
    await orch.step(FINALIZATION + 16 * M)
    assert len(client.calls) == calls_before
    assert phase(orch, cid) is Phase.SCORING

    # Operator fixes the worker and clears the halt -> pipeline resumes.
    fixed = FakeScoringClient()
    fixed.conn = client_conn(orch.core.db_path)
    orch.scoring_client = fixed
    assert orch.clear_halt(
        cid,
        operator="ops@vidaio",
        now=FINALIZATION + 17 * M,
        reason="runner capacity restored",
    )
    await orch.step(FINALIZATION + 18 * M)
    assert phase(orch, cid) is Phase.AWAITING_END_TIME
    await orch.step(END + 2 * M)
    assert phase(orch, cid) is Phase.COMPLETED


async def test_batch_infra_failure_requeues_then_succeeds(
    orchestrator_factory, fixture_repos, tmp_path
):
    runner = FakeRunner(tmp_path / "work" / "outputs")
    runner.batch_fail_times = 2  # burns the whole in-step retry budget (2) once
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)  # builds done -> EVALUATING
    await orch.step(FINALIZATION + 4 * M)  # first batch burns its budget -> requeued
    assert phase(orch, cid) is Phase.EVALUATING
    assert len(events_of(orch, cid, "batch_requeued")) == 1
    # The requeued batch blocks EVALUATING -> SCORING until it terminally completes.
    await orch.step(FINALIZATION + 5 * M)
    assert phase(orch, cid) is Phase.SCORING
    assert repo.count_non_terminal_batches(orch.conn, cid) == 0
    assert not pers.is_halted(orch.conn, cid)


async def test_batch_requeue_exhaustion_halts_competition_not_failed(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    runner = FakeRunner(tmp_path / "work" / "outputs")
    runner.batch_fail_times = 10_000
    orch = orchestrator_factory(
        runner=runner, repos=fixture_repos, max_batch_requeues=1
    )
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)  # builds done -> EVALUATING
    await orch.step(FINALIZATION + 4 * M)  # every batch fails -> requeue #1 each
    assert phase(orch, cid) is Phase.EVALUATING
    with caplog.at_level(logging.CRITICAL):
        await orch.step(FINALIZATION + 5 * M)  # budget exhausted -> halt
    assert pers.is_halted(orch.conn, cid)
    assert any(r.levelno == logging.CRITICAL for r in caplog.records)
    assert phase(orch, cid) is Phase.EVALUATING  # halted, never FAILED
    comp = repo.get_competition(orch.conn, cid)
    assert comp.failure_reason is None

    # Fix the infra, clear the halt: the requeued batch reruns and the
    # competition proceeds — nothing was corrupted by the halt.
    runner.batch_fail_times = 0
    orch.clear_halt(
        cid,
        operator="ops@vidaio",
        now=FINALIZATION + 6 * M,
        reason="runner capacity restored",
    )
    await orch.step(FINALIZATION + 7 * M)
    assert phase(orch, cid) is Phase.SCORING
