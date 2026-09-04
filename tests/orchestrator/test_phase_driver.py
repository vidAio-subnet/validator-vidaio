"""Phase-driver logic with an all-fake runner/scorer (no docker involved)."""

from __future__ import annotations

import pytest

from vidaio.competition import repository as repo
from vidaio.competition.repository import EnrollmentError
from vidaio.competition.states import Phase

from orchestrator_support import (
    COMMITMENT_ROOT,
    CONTENDER_SHAS,
    FINALIZATION,
    M,
    START,
    T0,
    FakeRunner,
    build_manifest,
    drive_to_completion,
    phase,
    repo_url,
    events_of,
    seed_items,
    start_and_enroll,
)


async def test_full_lifecycle_completes(orchestrator_factory, fixture_repos, tmp_path):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    await drive_to_completion(orch, cid, tmp_path)
    assert phase(orch, cid) is Phase.COMPLETED

    contenders = repo.list_contenders(orch.conn, cid)
    assert all(c.status == "BUILT" for c in contenders)
    assert len(events_of(orch, cid, "contender_built")) == 2

    # Full score matrix, every row audit-linked in the same transaction.
    rows = orch.conn.execute(
        "SELECT * FROM performance_history WHERE competition_id = ?", (cid,)
    ).fetchall()
    assert len(rows) == 2 * 3
    assert orch.engine.audit_linkage_gaps(orch.conn, cid) == []
    assert len(events_of(orch, cid, "audit_bundle_built")) == 6

    ranking = repo.ranking(orch.conn, cid)
    assert [c.final_rank for c in ranking] == [1, 2]
    assert all(c.final_score is not None and c.final_score > 0 for c in ranking)

    # Batches all terminal-completed (3 items, batch max 2 -> 2 batches each).
    batches = orch.conn.execute(
        "SELECT status FROM batches WHERE competition_id = ?", (cid,)
    ).fetchall()
    assert len(batches) == 4 and all(b["status"] == "COMPLETED" for b in batches)


async def test_partial_build_failure_marks_contender_not_competition(
    orchestrator_factory, fixture_repos, tmp_path
):
    runner = FakeRunner(tmp_path / "work" / "outputs")
    runner.fail_build_for.add(repo_url("hk-b"))
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    await drive_to_completion(orch, cid, tmp_path)
    assert phase(orch, cid) is Phase.COMPLETED

    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    assert by_hotkey["hk-a"].status == "BUILT"
    assert by_hotkey["hk-b"].status == "BUILD_FAILED"
    assert len(events_of(orch, cid, "contender_build_failed")) == 1

    # The failed contender is still ranked (last, zero score) — never dropped.
    ranking = repo.ranking(orch.conn, cid)
    assert [c.hotkey for c in ranking] == ["hk-a", "hk-b"]
    assert ranking[1].final_score == 0.0


async def test_all_builds_failed_fails_competition(
    orchestrator_factory, fixture_repos, tmp_path
):
    runner = FakeRunner(tmp_path / "work" / "outputs")
    runner.fail_build_for.update({repo_url("hk-a"), repo_url("hk-b")})
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)  # all builds fail -> engine fails it
    comp = repo.get_competition(orch.conn, cid)
    assert comp.status is Phase.FAILED
    assert comp.failure_reason == "all builds failed"


async def test_probe_failure_disqualifies_with_recorded_evidence(
    orchestrator_factory, fixture_repos, tmp_path
):
    runner = FakeRunner(tmp_path / "work" / "outputs")
    runner.fail_probe_for.add(CONTENDER_SHAS["hk-b"][1])  # hk-b's tree sha
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    await drive_to_completion(orch, cid, tmp_path)
    assert phase(orch, cid) is Phase.COMPLETED

    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    assert by_hotkey["hk-b"].status == "BUILD_FAILED"
    assert by_hotkey["hk-b"].image_digest is None
    dq_events = events_of(orch, cid, "isolation_probe_failed")
    assert len(dq_events) == 1

    # Probe evidence persisted on a sandboxes row, status FAILED.
    sandbox = orch.conn.execute(
        "SELECT * FROM sandboxes WHERE competition_id = ? AND contender_id = ?",
        (cid, by_hotkey["hk-b"].contender_id),
    ).fetchone()
    assert sandbox["status"] == "FAILED"
    assert "network_blocked" in sandbox["isolation_probe_json"]


async def test_evaluating_waits_for_items(orchestrator_factory, fixture_repos, tmp_path):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)
    assert phase(orch, cid) is Phase.EVALUATING
    await orch.step(FINALIZATION + 10 * M)  # no items seeded: must wait, not advance
    assert phase(orch, cid) is Phase.EVALUATING
    assert repo.count_non_terminal_batches(orch.conn, cid) == 0


async def test_enrollment_passthrough_enforces_stake_gate(
    orchestrator_factory, fixture_repos
):
    orch = orchestrator_factory(repos=fixture_repos)
    manifest = build_manifest()
    cid = manifest.competition_id
    orch.create_competition(manifest, T0)
    orch.anchor_commitment(cid, COMMITMENT_ROOT, T0)
    await orch.step(START)
    with pytest.raises(EnrollmentError):
        orch.enroll_contender(
            cid,
            hotkey="hk-poor",
            repo_url=repo_url("hk-a"),
            commit_sha="9a" * 20,
            tree_sha="9b" * 20,
            stake=10.0,  # below minimum_alpha_stake 500
            now=START + M,
        )
