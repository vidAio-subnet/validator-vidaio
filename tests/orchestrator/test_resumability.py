"""Fault injection: a crash mid-stage must leave resumable DB state — a fresh
orchestrator over the same database finishes the competition without redoing
(or duplicating) committed work (spec §14 failure recovery)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from vidaio.competition import repository as repo
from vidaio.competition.states import Phase

from orchestrator_support import (
    END,
    FINALIZATION,
    M,
    CrashingRunner,
    CrashingScoringClient,
    SimulatedCrash,
    build_manifest,
    phase,
    seed_items,
    start_and_enroll,
)


async def test_crash_mid_building_resumes_without_duplicate_builds(
    orchestrator_factory, fixture_repos, tmp_path
):
    crasher = CrashingRunner(tmp_path / "work" / "outputs", crash_on_build_call=2)
    orch1 = orchestrator_factory(runner=crasher, repos=fixture_repos)
    cid = await start_and_enroll(orch1, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch1, cid, tmp_path / "item-src")
    await orch1.step(FINALIZATION)
    await orch1.step(FINALIZATION + 2 * M)
    with pytest.raises(SimulatedCrash):
        await orch1.step(FINALIZATION + 3 * M)  # dies after the first build committed

    # Crash left: one BUILT, one still ACCEPTED, phase unchanged.
    assert phase(orch1, cid) is Phase.BUILDING
    statuses = sorted(c.status for c in repo.list_contenders(orch1.conn, cid))
    assert statuses == ["ACCEPTED", "BUILT"]
    built_ids = [
        c.contender_id for c in repo.list_contenders(orch1.conn, cid) if c.status == "BUILT"
    ]

    # "Restart": a new orchestrator over the same DB/work dir, healthy runner.
    orch2 = orchestrator_factory(repos=fixture_repos)
    runner2 = orch2.runner
    await orch2.step(FINALIZATION + 4 * M)
    assert phase(orch2, cid) is Phase.EVALUATING
    # Exactly the ONE unfinished contender was built — no duplicate builds.
    assert len(runner2.build_calls) == 1
    assert runner2.build_calls[0] not in built_ids

    await orch2.step(FINALIZATION + 10 * M)
    await orch2.step(FINALIZATION + 15 * M)
    await orch2.step(END + M)
    assert phase(orch2, cid) is Phase.COMPLETED
    assert orch2.engine.audit_linkage_gaps(orch2.conn, cid) == []


async def test_crash_mid_scoring_resumes_without_duplicate_scores(
    orchestrator_factory, fixture_repos, tmp_path
):
    crasher = CrashingScoringClient(crash_on_call=3)
    orch1 = orchestrator_factory(scoring_client=crasher, repos=fixture_repos)
    cid = await start_and_enroll(orch1, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch1, cid, tmp_path / "item-src")
    await orch1.step(FINALIZATION)
    await orch1.step(FINALIZATION + 2 * M)
    await orch1.step(FINALIZATION + 3 * M)
    await orch1.step(FINALIZATION + 10 * M)
    assert phase(orch1, cid) is Phase.SCORING
    with pytest.raises(SimulatedCrash):
        await orch1.step(FINALIZATION + 15 * M)  # dies after 2 scores committed

    committed = orch1.conn.execute(
        "SELECT COUNT(*) AS n FROM performance_history WHERE competition_id = ?", (cid,)
    ).fetchone()["n"]
    assert committed == 2
    assert phase(orch1, cid) is Phase.SCORING
    # Atomicity: every committed score is already audit-linked (never half-recorded).
    assert orch1.engine.audit_linkage_gaps(orch1.conn, cid) == []

    orch2 = orchestrator_factory(repos=fixture_repos)
    await orch2.step(FINALIZATION + 20 * M)
    assert phase(orch2, cid) is Phase.AWAITING_END_TIME
    total = orch2.conn.execute(
        "SELECT COUNT(*) AS n FROM performance_history WHERE competition_id = ?", (cid,)
    ).fetchone()["n"]
    assert total == 2 * 3  # full matrix, no duplicates (UNIQUE(contender, item))
    # The restarted client only scored the pairs the crash left unscored.
    assert len(orch2.scoring_client.calls) == total - committed

    await orch2.step(END + timedelta(minutes=25))
    assert phase(orch2, cid) is Phase.COMPLETED
