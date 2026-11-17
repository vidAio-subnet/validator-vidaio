"""Pre-deadline entry checks: the fetch check and the trial build WARN only. They record
events, never change a contender's status, never halt, and run one checkout per tick
off the event loop."""

from __future__ import annotations

import json
import threading
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import entries
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.orchestrator.public_api import _entry_view
from vidaio.competition.orchestrator.service import _precheck_checkout_problem
from vidaio.competition.runners import LocalRepoProvider
from vidaio.competition.runners.errors import CheckoutError, CheckoutRejectedError
from vidaio.competition.runners.renewable import RenewableRunner
from vidaio.competition.states import Phase

from orchestrator_support import (
    CONTENDER_SHAS,
    HONEST_RUN_SH,
    M,
    START,
    FakeRunner,
    build_manifest,
    drive_to_completion,
    phase,
    repo_url,
    start_and_enroll,
    write_solution,
)

LONG = {
    "enrollment_deadline": START + timedelta(days=3),
    "finalization_time": START + timedelta(days=4),
    "end_time": START + timedelta(days=6),
}


async def _tick(orch, at) -> None:
    """One orchestrator step, then wait for the background entry check it started."""
    await orch.step(at)
    task = orch._entry_check_task
    if task is not None:
        await task


def _events(orch, cid: str, event_type: str) -> list[dict]:
    return [
        {**json.loads(e["payload_json"]), "at": e["created_at"]}
        for e in repo.list_events(orch.conn, cid)
        if e["event_type"] == event_type
    ]


def _summary(orch, cid: str, *, trial: bool = False) -> dict[int, dict[str, str]]:
    contenders = [c for c in repo.list_contenders(orch.conn, cid) if not c.is_calibration]
    checks = entries.entry_checks(orch.conn, cid, contenders)
    return {
        c.contender_id: entries.checks_summary(checks[c.contender_id], trial_enabled=trial)
        for c in contenders
    }


def _replace(orch, cid: str, hotkey: str, commit: str, tree: str, now) -> None:
    orch.submit_entry(
        cid, hotkey=hotkey, repo_url=repo_url(hotkey), commit_sha=commit,
        tree_sha=tree, stake=1000.0, now=now, replace=True,
    )


# ---- fetch check -------------------------------------------------------------------


async def test_new_entries_are_checked_one_per_tick_and_only_warn(
    orchestrator_factory, fixture_repos
):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest("comp-precheck"), ["hk-a", "hk-link"])
    a, link = (c.contender_id for c in repo.list_contenders(orch.conn, cid))

    await _tick(orch, START + 6 * M)
    (first,) = _events(orch, cid, entries.EVENT_PRECHECKED)
    assert first["contender_id"] == a and first["ok"] is True and first["problems"] == []
    assert first["commit_sha"] == CONTENDER_SHAS["hk-a"][0]
    assert first["repo_url"] == repo_url("hk-a")

    await _tick(orch, START + 7 * M)
    second = _events(orch, cid, entries.EVENT_PRECHECKED)[1]
    assert second["contender_id"] == link and second["ok"] is False
    assert second["problems"] == ["unsafe_tree"] and "stolen-credentials" in second["detail"]

    await _tick(orch, START + 8 * M)  # nothing due
    assert len(_events(orch, cid, entries.EVENT_PRECHECKED)) == 2
    assert {c.status for c in repo.list_contenders(orch.conn, cid)} == {"ENROLLED"}
    assert not pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.ENROLLING
    assert _summary(orch, cid) == {
        a: {"fetch": "ok", "build": "disabled"},
        link: {"fetch": "unsafe_tree", "build": "disabled"},
    }


async def test_recheck_cadence_six_hours_then_hourly_before_the_deadline(
    orchestrator_factory, fixture_repos
):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest("comp-cadence", **LONG), ["hk-a"])
    t = START + 6 * M
    await _tick(orch, t)
    await _tick(orch, t + timedelta(hours=5, minutes=59))
    assert len(_events(orch, cid, entries.EVENT_PRECHECKED)) == 1
    await _tick(orch, t + timedelta(hours=6))
    assert len(_events(orch, cid, entries.EVENT_PRECHECKED)) == 2

    deadline = LONG["enrollment_deadline"]
    final = deadline - timedelta(hours=23)
    await _tick(orch, final)
    await _tick(orch, final + timedelta(minutes=59))
    assert len(_events(orch, cid, entries.EVENT_PRECHECKED)) == 3
    await _tick(orch, final + timedelta(hours=1))
    assert len(_events(orch, cid, entries.EVENT_PRECHECKED)) == 4
    # after the deadline (until finalization) an unreadable repo still gets noticed
    await _tick(orch, deadline + timedelta(hours=2))
    assert len(_events(orch, cid, entries.EVENT_PRECHECKED)) == 5


async def test_a_replaced_version_is_pending_until_checked_again(
    orchestrator_factory, fixture_repos
):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest("comp-replace", **LONG), ["hk-a", "hk-b"])
    a, b = (c.contender_id for c in repo.list_contenders(orch.conn, cid))
    await _tick(orch, START + 6 * M)
    await _tick(orch, START + 7 * M)
    assert _summary(orch, cid)[b]["fetch"] == "ok"

    _replace(orch, cid, "hk-b", "3b" * 20, "4b" * 20, START + 8 * M)
    assert _summary(orch, cid)[b]["fetch"] == "pending"
    await _tick(orch, START + 9 * M)
    latest = _events(orch, cid, entries.EVENT_PRECHECKED)[-1]
    assert latest["contender_id"] == b and latest["commit_sha"] == "3b" * 20
    assert _summary(orch, cid)[b]["fetch"] == "ok"


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (CheckoutError(
            "git fetch pinned commit exited 128: remote: Repository not found.\n"
            "fatal: repository 'https://github.com/x/y.git/' not found"), "repo_unreadable"),
        (CheckoutError(
            "git fetch pinned commit exited 128: fatal: remote error: upload-pack: "
            "not our ref " + "a" * 40), "commit_missing"),
        (CheckoutRejectedError(
            f"checked-out tree {'c' * 40!r} does not match enrolled {'d' * 40}"), "tree_mismatch"),
        (CheckoutRejectedError("unsafe checkout tree: entry 'x' is a symlink"), "unsafe_tree"),
        (CheckoutRejectedError(
            "submission contains .gitmodules; independently unpinned submodules are forbidden"),
         "unsafe_tree"),
        (CheckoutRejectedError("checkout is 900 bytes, over the cap of 512"), "over_size_cap"),
        (CheckoutRejectedError(
            "git checkout pinned commit crossed the checkout cap of 9 bytes for Git metadata"),
         "over_size_cap"),
        (CheckoutError(
            "git fetch pinned commit exited 128: fatal: Authentication failed for "
            "'https://github.com/x/y.git/'"), "check_unavailable"),
        (CheckoutError(
            "git fetch pinned commit exited 128: fatal: unable to access "
            "'https://github.com/ssl-tools/y.git/': Could not resolve host: github.com"),
         "check_unavailable"),
        (CheckoutError(
            "checkout exceeded the bounded 180.0s total Git timeout during git fetch"),
         "fetch_timeout"),
        (OSError("No space left on device"), "check_unavailable"),
    ],
)
def test_checkout_failures_map_to_problem_codes(exc: Exception, code: str) -> None:
    got, detail = _precheck_checkout_problem(exc)
    assert got == code
    assert got in entries.PROBLEM_CODES
    assert detail and "\n" not in detail and len(detail) <= 300


class _RecordingProvider:
    """LocalRepoProvider that records the calling thread, or raises."""

    def __init__(self, mapping, *, error: Exception | None = None) -> None:
        self._inner = LocalRepoProvider(mapping)
        self.error = error
        self.threads: list[int] = []

    def checkout(self, repo_url: str, commit_sha: str) -> Path:
        self.threads.append(threading.get_ident())
        if self.error is not None:
            raise self.error
        return self._inner.checkout(repo_url, commit_sha)

    def release(self, checkout) -> None:
        return None


async def test_the_checkout_runs_off_the_event_loop_and_failures_never_halt(
    orchestrator_factory, fixture_repos
):
    provider = _RecordingProvider(
        fixture_repos,
        error=CheckoutError("git fetch pinned commit exited 128: remote: Repository not found."),
    )
    orch = orchestrator_factory(repo_provider=provider)
    cid = await start_and_enroll(orch, build_manifest("comp-thread"), ["hk-a"])
    await _tick(orch, START + 6 * M)
    assert provider.threads and threading.get_ident() not in provider.threads
    (event,) = _events(orch, cid, entries.EVENT_PRECHECKED)
    assert event["problems"] == ["repo_unreadable"]
    assert not pers.is_halted(orch.conn, cid)
    assert repo.list_contenders(orch.conn, cid)[0].status == "ENROLLED"


async def test_tree_inspection_problems(orchestrator_factory, tmp_path: Path):
    good = write_solution(tmp_path / "good", HONEST_RUN_SH.format(head=1))
    nodocker = tmp_path / "nodocker"
    nodocker.mkdir()
    (nodocker / "run.sh").write_text("#!/bin/sh\n")
    big = write_solution(tmp_path / "big", HONEST_RUN_SH.format(head=1))
    (big / "weights.bin").write_bytes(b"w" * 4096)
    submodule = write_solution(tmp_path / "submodule", HONEST_RUN_SH.format(head=1))
    (submodule / ".gitmodules").write_text("[submodule 'x']\n")
    git_heavy = write_solution(tmp_path / "git-heavy", HONEST_RUN_SH.format(head=1))
    (git_heavy / ".git").mkdir()
    (git_heavy / ".git" / "pack").write_bytes(b"p" * 8192)  # never counted
    mapping = {
        "local://good": good, "local://nodocker": nodocker, "local://big": big,
        "local://submodule": submodule, "local://git-heavy": git_heavy,
    }
    orch = orchestrator_factory(repos=mapping, git_checkout_max_bytes=2048)
    cid = await start_and_enroll(orch, build_manifest("comp-inspect"), [])
    for index, name in enumerate(mapping):
        orch.enroll_contender(
            cid, hotkey=f"hk-{name.split('//')[1]}", repo_url=name,
            commit_sha=f"{index + 1:x}" * 40, tree_sha=f"{index + 6:x}" * 40,
            stake=1000.0, now=START + 5 * M,
        )
    for minute in range(len(mapping)):
        await _tick(orch, START + (6 + minute) * M)
    by_url = {e["repo_url"]: e for e in _events(orch, cid, entries.EVENT_PRECHECKED)}
    assert by_url["local://good"]["ok"] is True
    assert by_url["local://git-heavy"]["ok"] is True
    assert by_url["local://nodocker"]["problems"] == ["no_dockerfile"]
    assert by_url["local://big"]["problems"] == ["over_size_cap"]
    assert by_url["local://submodule"]["problems"] == ["unsafe_tree"]

    record = repo.find_contender_by_hotkey(orch.conn, cid, "hk-nodocker")
    ok, problems, detail = orch._precheck_entry(record, licenses_required=True)
    assert not ok
    assert problems == ["no_dockerfile", "no_license", "no_licenses_md"]
    assert "LICENSES.md" in detail
    (good / "LICENSE").write_text("MIT\n")
    (good / "LICENSES.md").write_text("none\n")
    record = repo.find_contender_by_hotkey(orch.conn, cid, "hk-good")
    assert orch._precheck_entry(record, licenses_required=True) == (True, [], "")


def test_licences_are_required_for_interpolation_and_upscaling() -> None:
    assert entries.licenses_required(SimpleNamespace(track="interpolation"))
    assert entries.licenses_required(SimpleNamespace(track="upscaling"))
    assert not entries.licenses_required(SimpleNamespace(track="compression"))
    assert not entries.licenses_required(SimpleNamespace(track="removal"))


async def test_fetch_check_can_be_turned_off(orchestrator_factory, fixture_repos):
    orch = orchestrator_factory(repos=fixture_repos, enrollment_precheck_enabled=False)
    cid = await start_and_enroll(orch, build_manifest("comp-off"), ["hk-a"])
    await _tick(orch, START + 6 * M)
    assert _events(orch, cid, entries.EVENT_PRECHECKED) == []


# ---- trial build -------------------------------------------------------------------


async def test_trial_build_is_off_by_default(orchestrator_factory, fixture_repos):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest("comp-trial-off"), ["hk-a"])
    for minute in (6, 7, 8):
        await _tick(orch, START + minute * M)
    assert orch.runner.build_calls == []
    assert _events(orch, cid, entries.EVENT_TRIAL_BUILT) == []


async def test_trial_build_uses_the_real_builder_once_and_keeps_nothing(
    orchestrator_factory, fixture_repos
):
    orch = orchestrator_factory(repos=fixture_repos, enrollment_trial_build_enabled=True)
    cid = await start_and_enroll(orch, build_manifest("comp-trial"), ["hk-a"])
    (a,) = (c.contender_id for c in repo.list_contenders(orch.conn, cid))
    await _tick(orch, START + 6 * M)  # fetch check
    assert orch.runner.build_calls == []
    await _tick(orch, START + 7 * M)  # trial build
    assert orch.runner.build_calls == [a]
    (event,) = _events(orch, cid, entries.EVENT_TRIAL_BUILT)
    assert event["ok"] is True and event["outcome"] == "ok" and event["error"] is None
    assert event["commit_sha"] == CONTENDER_SHAS["hk-a"][0]
    await _tick(orch, START + 8 * M)
    assert orch.runner.build_calls == [a]  # one build per version

    record = repo.get_contender(orch.conn, a)
    assert record.status == "ENROLLED" and record.image_digest is None
    kept = {e["event_type"] for e in repo.list_events(orch.conn, cid)}
    assert not kept & {
        pers.EVENT_MODAL_IMAGE_BOUND, pers.EVENT_MODAL_RUNTIME_BOUND,
        "contender_built", "contender_build_failed", "sandbox_probe",
    }
    assert orch.runner.probe_calls == []
    assert _summary(orch, cid, trial=True)[a] == {"fetch": "ok", "build": "ok"}


async def test_a_failed_trial_build_is_reported_in_full_to_the_entrant_only(
    orchestrator_factory, fixture_repos
):
    orch = orchestrator_factory(repos=fixture_repos, enrollment_trial_build_enabled=True)
    orch.runner.fail_build_for.add(repo_url("hk-a"))
    cid = await start_and_enroll(orch, build_manifest("comp-trial-fail"), ["hk-a"])
    (a,) = (c.contender_id for c in repo.list_contenders(orch.conn, cid))
    await _tick(orch, START + 6 * M)
    await _tick(orch, START + 7 * M)
    (event,) = _events(orch, cid, entries.EVENT_TRIAL_BUILT)
    assert event["outcome"] == "failed" and event["ok"] is False
    assert event["error"].startswith("build failed (BUILD_FAILED): forced build failure")
    assert _summary(orch, cid, trial=True)[a]["build"] == "failed"
    comp = repo.get_competition(orch.conn, cid)
    view = _entry_view(orch, comp, repo.get_contender(orch.conn, a))
    assert view["trial_build"]["outcome"] == "failed"
    assert "forced build failure" in view["trial_build"]["error"]
    assert repo.get_contender(orch.conn, a).status == "ENROLLED"
    await _tick(orch, START + 8 * M)
    assert orch.runner.build_calls == [a]  # the version has its answer


async def test_trial_build_infra_failure_is_ours_and_backs_off(
    orchestrator_factory, fixture_repos
):
    orch = orchestrator_factory(repos=fixture_repos, enrollment_trial_build_enabled=True)
    orch.runner.infra_fail_build_for.add(repo_url("hk-a"))
    cid = await start_and_enroll(orch, build_manifest("comp-trial-infra"), ["hk-a"])
    (a,) = (c.contender_id for c in repo.list_contenders(orch.conn, cid))
    await _tick(orch, START + 6 * M)
    await _tick(orch, START + 7 * M)
    (event,) = _events(orch, cid, entries.EVENT_TRIAL_BUILT)
    assert event["outcome"] == "unavailable" and event["error"] is None
    assert "docker daemon unusable" in event["detail"]
    assert _summary(orch, cid, trial=True)[a]["build"] == "pending"
    comp = repo.get_competition(orch.conn, cid)
    view = _entry_view(orch, comp, repo.get_contender(orch.conn, a))
    assert "our side" in view["trial_build"]["error"]
    assert not pers.is_halted(orch.conn, cid)

    await _tick(orch, START + 20 * M)  # inside the retry back-off
    assert orch.runner.build_calls == [a]
    orch.runner.infra_fail_build_for.clear()
    await _tick(orch, START + 38 * M)
    assert orch.runner.build_calls == [a, a]
    assert _events(orch, cid, entries.EVENT_TRIAL_BUILT)[-1]["outcome"] == "ok"


async def test_a_trial_build_that_cannot_fetch_the_source_never_blocks_other_entries(
    orchestrator_factory, fixture_repos
):
    """An entry that passed its fetch check and then removed its repository: the
    trial build's checkout fails (plain CheckoutError, classified as ours). That
    must not pause everyone's trial builds or drop the runtime: the entry's fetch
    check re-runs at once, it leaves the trial queue, and the next entry builds."""
    from orchestrator_support import FlakyRepoProvider

    provider = FlakyRepoProvider(fixture_repos)
    orch = orchestrator_factory(
        repos=fixture_repos, repo_provider=provider, enrollment_trial_build_enabled=True
    )
    cid = await start_and_enroll(orch, build_manifest("comp-trial-gone"), ["hk-a", "hk-b"])
    a, b = (c.contender_id for c in repo.list_contenders(orch.conn, cid))
    await _tick(orch, START + 6 * M)  # fetch check hk-a
    await _tick(orch, START + 7 * M)  # fetch check hk-b
    provider.fail_for[repo_url("hk-a")] = 1000  # the repository went away
    orch.runner.checkout_fail_build_for.add(repo_url("hk-a"))

    await _tick(orch, START + 8 * M)  # trial build hk-a: the checkout fails
    (event,) = _events(orch, cid, entries.EVENT_TRIAL_BUILT)
    assert event["outcome"] == "unavailable" and event["cause"] == entries.CAUSE_CHECKOUT
    assert entries.last_trial_unavailable_at(orch.conn, cid) is None  # no global pause
    checks = [e for e in _events(orch, cid, entries.EVENT_PRECHECKED) if e["contender_id"] == a]
    assert checks[-1]["ok"] is False  # the re-run fetch check failed: out of the queue
    assert _summary(orch, cid)[a]["fetch"] != "ok"

    await _tick(orch, START + 9 * M)  # the next entry builds at once
    assert orch.runner.build_calls == [a, b]
    await _tick(orch, START + 10 * M)
    assert orch.runner.build_calls == [a, b]  # hk-a waits for a passing fetch check


async def test_trial_builds_respect_the_daily_cap_per_entry(orchestrator_factory, fixture_repos):
    orch = orchestrator_factory(
        repos=fixture_repos, enrollment_trial_build_enabled=True,
        enrollment_trial_build_daily_cap=2,
    )
    cid = await start_and_enroll(orch, build_manifest("comp-cap", **LONG), ["hk-a"])
    t = START + 6 * M
    for version in range(3):
        if version:
            _replace(orch, cid, "hk-a", f"{version}c" * 20, f"{version}d" * 20, t)
        await _tick(orch, t + M)  # fetch check
        await _tick(orch, t + 2 * M)  # trial build (unless capped)
        t += 3 * M
    assert len(orch.runner.build_calls) == 2
    a = repo.list_contenders(orch.conn, cid)[0].contender_id
    assert _summary(orch, cid, trial=True)[a] == {"fetch": "ok", "build": "pending"}
    day_later = START + timedelta(days=1, minutes=12)
    await _tick(orch, day_later)  # the 6-hourly re-check comes first (one checkout/tick)
    await _tick(orch, day_later + M)
    assert len(orch.runner.build_calls) == 3  # a day later the cap has room again


async def test_no_trial_build_after_a_failed_fetch_check_or_after_the_deadline(
    orchestrator_factory, fixture_repos
):
    orch = orchestrator_factory(repos=fixture_repos, enrollment_trial_build_enabled=True)
    cid = await start_and_enroll(orch, build_manifest("comp-trial-skip"), ["hk-link", "hk-a"])
    await _tick(orch, START + 6 * M)  # hk-link: unsafe
    await _tick(orch, START + 7 * M)  # hk-a: ok
    deadline = repo.get_competition(orch.conn, cid).enrollment_deadline
    await _tick(orch, deadline + timedelta(seconds=1))
    await _tick(orch, deadline + 2 * M)
    assert orch.runner.build_calls == []


async def test_trial_build_waits_while_a_baseline_prebuild_holds_the_sandbox(
    orchestrator_factory, fixture_repos
):
    orch = orchestrator_factory(repos=fixture_repos, enrollment_trial_build_enabled=True)
    cid = await start_and_enroll(orch, build_manifest("comp-trial-lock"), ["hk-a"])
    await _tick(orch, START + 6 * M)
    await orch._sandbox_lock.acquire()
    try:
        await _tick(orch, START + 7 * M)
        assert orch.runner.build_calls == []
    finally:
        orch._sandbox_lock.release()
    await _tick(orch, START + 8 * M)
    assert len(orch.runner.build_calls) == 1
    assert len(_events(orch, cid, entries.EVENT_TRIAL_BUILT)) == 1


async def test_one_runtime_generation_serves_all_trial_builds_until_the_deadline(
    orchestrator_factory, fixture_repos, tmp_path: Path
):
    """Each renewable generation is a fresh Modal Environment + App: trial builds keep
    ONE generation alive before the deadline instead of minting one per build. An
    `unavailable` build drops it so the next attempt starts fresh; after the deadline
    the idle release applies, and BUILDING runs on a generation of its own."""
    made: list[FakeRunner] = []

    def factory(sequence, resources):
        runner = FakeRunner(tmp_path / "work" / "outputs")  # the orchestrator's pool
        if sequence == 1:
            runner.infra_fail_build_for.add(repo_url("hk-b"))
        made.append(runner)
        return runner

    renewable = RenewableRunner(factory, idle_grace_seconds=0.0)
    orch = orchestrator_factory(
        repos=fixture_repos, runner=renewable, enrollment_trial_build_enabled=True
    )
    cid = await start_and_enroll(
        orch, build_manifest("comp-trial-renew"), ["hk-a", "hk-lie", "hk-b"]
    )
    for minute in (6, 7, 8):  # three fetch checks
        await _tick(orch, START + minute * M)
    assert not renewable.active and made == []
    await _tick(orch, START + 9 * M)  # trial build hk-a
    await _tick(orch, START + 10 * M)  # trial build hk-lie
    assert len(made) == 1 and len(made[0].build_calls) == 2
    assert renewable.active  # kept between trial builds (no idle release)
    await _tick(orch, START + 11 * M)  # hk-b: our side fails
    outcomes = [e["outcome"] for e in _events(orch, cid, entries.EVENT_TRIAL_BUILT)]
    assert outcomes == ["ok", "ok", "unavailable"]
    assert not renewable.active  # discarded: the next attempt starts a fresh one
    await _tick(orch, START + 42 * M)  # after the retry back-off
    assert len(made) == 2 and made[1].build_calls
    assert _events(orch, cid, entries.EVENT_TRIAL_BUILT)[-1]["outcome"] == "ok"
    assert renewable.active

    deadline = repo.get_competition(orch.conn, cid).enrollment_deadline
    await _tick(orch, deadline + timedelta(seconds=1))
    assert not renewable.active  # released normally after the deadline
    await drive_to_completion(orch, cid, tmp_path)
    assert phase(orch, cid) is Phase.COMPLETED
    assert len(made) >= 3  # BUILDING/EVALUATING acquired a generation of their own
    assert made[2].build_calls


class _SlowRunner(FakeRunner):
    """A builder that blocks until released, like a long Modal image build."""

    def __init__(self, outputs_dir: Path) -> None:
        super().__init__(outputs_dir)
        self.release = threading.Event()
        self.started = threading.Event()
        self.entered = 0

    def build(self, contender) -> str:
        self.entered += 1
        self.started.set()
        self.release.wait(10)
        return super().build(contender)


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


async def test_a_slow_trial_build_never_blocks_the_tick_or_finalization_order(
    orchestrator_factory, fixture_repos, tmp_path: Path
):
    import time as _time

    from orchestrator_support import FINALIZATION

    runner = _SlowRunner(tmp_path / "slow-outputs")
    orch = orchestrator_factory(
        repos=fixture_repos, runner=runner, enrollment_trial_build_enabled=True
    )
    cid = await start_and_enroll(orch, build_manifest("comp-slow"), ["hk-a"])
    await _tick(orch, START + 6 * M)  # fetch check
    try:
        started = _time.monotonic()
        await orch.step(START + 7 * M)  # starts the trial build in the background
        task = orch._entry_check_task
        assert task is not None and not task.done()
        await _wait_until(runner.started.is_set)
        for minute in (8, 9, 10):
            await orch.step(START + minute * M)
            assert orch._check_tick_age()  # the tick stays fresh: health stays green
            assert orch._entry_check_task is task  # one task at a time
        assert _time.monotonic() - started < 5
        assert runner.entered == 1

        # finalization does not start while the check still runs
        await orch.step(FINALIZATION)
        assert phase(orch, cid) is Phase.FINALIZING_SUBMISSIONS
        assert pers.archived_submissions(orch.conn, cid) == {}
        assert _events(orch, cid, entries.EVENT_STAKE_RECHECKED) == []
    finally:
        runner.release.set()
    await task
    assert _events(orch, cid, entries.EVENT_TRIAL_BUILT)[0]["outcome"] == "ok"
    await orch.step(FINALIZATION + M)
    assert phase(orch, cid) is Phase.VALIDATING
    assert pers.archived_submissions(orch.conn, cid)


async def test_shutdown_cancels_a_running_entry_check(
    orchestrator_factory, fixture_repos, tmp_path: Path
):
    import asyncio

    runner = _SlowRunner(tmp_path / "slow-outputs")
    orch = orchestrator_factory(
        repos=fixture_repos, runner=runner, enrollment_trial_build_enabled=True,
        clock=lambda: START + 7 * M,
    )
    cid = await start_and_enroll(orch, build_manifest("comp-shutdown"), ["hk-a"])
    await _tick(orch, START + 6 * M)  # fetch check done; the trial build is next
    service = asyncio.create_task(orch.run())
    try:
        await _wait_until(runner.started.is_set)
        task = orch._entry_check_task
        assert task is not None and not task.done()
        orch.request_stop()
        await asyncio.wait_for(service, timeout=10)
        assert task.cancelled()
        assert orch._entry_check_task is None
        assert _events(orch, cid, entries.EVENT_TRIAL_BUILT) == []
    finally:
        runner.release.set()


async def test_a_competition_with_trial_builds_still_runs_to_completion(
    orchestrator_factory, fixture_repos, tmp_path: Path
):
    orch = orchestrator_factory(repos=fixture_repos, enrollment_trial_build_enabled=True)
    cid = await start_and_enroll(orch, build_manifest("comp-trial-e2e"), ["hk-a", "hk-b"])
    for minute in range(6, 12):
        await _tick(orch, START + minute * M)
    assert len(_events(orch, cid, entries.EVENT_TRIAL_BUILT)) == 2
    await drive_to_completion(orch, cid, tmp_path)
    assert phase(orch, cid) is Phase.COMPLETED
    assert {c.status for c in repo.list_contenders(orch.conn, cid)} == {"BUILT"}
    assert len(orch.runner.build_calls) == 4  # two trial builds + two real builds


# ---- alpha-stake re-check when enrollment closes -------------------------------------


class _StakeChain:
    """The chain seam behind the hotkey registry: hotkey -> alpha stake."""

    def __init__(self, stakes: dict[str, float]) -> None:
        self.stakes = stakes
        self.down = False

    def refresh(self) -> None:
        if self.down:
            raise RuntimeError("archive endpoint unreachable")

    def neurons(self):
        return [
            SimpleNamespace(hotkey=hk, is_validator=False, alpha_stake=stake)
            for hk, stake in self.stakes.items()
        ]


def _staked_orchestrator(orchestrator_factory, fixture_repos, chain, **overrides):
    orch = orchestrator_factory(repos=fixture_repos, chain=chain, **overrides)
    # report-mode stacks inject their signature verifier; the guard (and with it the
    # one shared registry) is then built exactly as for the public API
    orch._hotkey_verify_fn = lambda hotkey, payload, signature: True
    assert orch._stake_registry() is orch._hotkey_guard()._registry
    return orch


async def test_entries_below_the_stake_floor_are_warned_then_rejected_at_the_deadline(
    orchestrator_factory, fixture_repos, tmp_path: Path
):
    from orchestrator_support import FINALIZATION, seed_items

    chain = _StakeChain({"hk-a": 500.0, "hk-b": 120.0})  # manifest floor: 500
    orch = _staked_orchestrator(orchestrator_factory, fixture_repos, chain)
    cid = await start_and_enroll(orch, build_manifest("comp-stake"), ["hk-a", "hk-b"])
    a, b = (c.contender_id for c in repo.list_contenders(orch.conn, cid))
    await _tick(orch, START + 6 * M)
    await _tick(orch, START + 7 * M)
    checks = {e["contender_id"]: e for e in _events(orch, cid, entries.EVENT_PRECHECKED)}
    assert checks[a]["ok"] is True
    assert checks[b]["problems"] == ["stake_below_floor"]
    assert "120 is below the floor 500" in checks[b]["detail"]
    assert _summary(orch, cid)[b]["fetch"] == "stake_below_floor"
    assert {c.status for c in repo.list_contenders(orch.conn, cid)} == {"ENROLLED"}

    seed_items(orch, cid, tmp_path / "items")
    await _tick(orch, FINALIZATION)
    by_id = {c.contender_id: c for c in repo.list_contenders(orch.conn, cid)}
    assert by_id[b].status == "REJECTED"
    assert by_id[a].status != "REJECTED"  # exactly at the floor: untouched
    reasons = [
        e["reason"] for e in _events(orch, cid, "contender_validated")
        if e["contender_id"] == b
    ]
    assert reasons == ["alpha stake 120 below the floor 500 when entries were finalized"]
    (recheck,) = _events(orch, cid, entries.EVENT_STAKE_RECHECKED)
    assert recheck["floor"] == 500.0 and recheck["rejected"] == [b]
    assert not pers.is_halted(orch.conn, cid)
    archived = pers.archived_submissions(orch.conn, cid)
    assert a in archived and b not in archived


async def test_an_unreadable_registry_halts_finalization_and_rejects_nobody(
    orchestrator_factory, fixture_repos, tmp_path: Path
):
    from orchestrator_support import FINALIZATION, seed_items

    chain = _StakeChain({"hk-a": 900.0, "hk-b": 10.0})
    orch = _staked_orchestrator(orchestrator_factory, fixture_repos, chain)
    cid = await start_and_enroll(orch, build_manifest("comp-stake-down"), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "items")
    chain.down = True
    await _tick(orch, FINALIZATION)
    assert pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.FINALIZING_SUBMISSIONS
    assert {c.status for c in repo.list_contenders(orch.conn, cid)} == {"ENROLLED"}
    assert pers.archived_submissions(orch.conn, cid) == {}
    assert _events(orch, cid, entries.EVENT_STAKE_RECHECKED) == []

    chain.down = False
    orch.clear_halt(cid, "operator", FINALIZATION + M, reason="chain endpoint restored")
    await _tick(orch, FINALIZATION + 2 * M)
    statuses = {c.hotkey: c.status for c in repo.list_contenders(orch.conn, cid)}
    assert statuses["hk-b"] == "REJECTED" and statuses["hk-a"] != "REJECTED"
    assert len(_events(orch, cid, entries.EVENT_STAKE_RECHECKED)) == 1


class _SilentlyStaleChain(_StakeChain):
    """Like the production adapter: refresh() never raises and the cached neurons
    stay served; only has_fresh_snapshot() tells that no new read arrived."""

    def __init__(self, stakes: dict[str, float]) -> None:
        super().__init__(stakes)
        self.fresh = True

    def refresh(self) -> None:
        return None

    def has_fresh_snapshot(self, now: float, max_age_seconds: float) -> bool:
        return self.fresh


async def test_a_silently_stale_chain_halts_the_stake_recheck(
    orchestrator_factory, fixture_repos, tmp_path: Path
):
    from orchestrator_support import FINALIZATION, seed_items

    chain = _SilentlyStaleChain({"hk-a": 900.0, "hk-b": 10.0})
    orch = _staked_orchestrator(orchestrator_factory, fixture_repos, chain)
    cid = await start_and_enroll(orch, build_manifest("comp-stake-stale"), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "items")
    chain.fresh = False  # the endpoint went away; the old snapshot is still served
    await _tick(orch, FINALIZATION)
    assert pers.is_halted(orch.conn, cid)
    assert {c.status for c in repo.list_contenders(orch.conn, cid)} == {"ENROLLED"}
    assert _events(orch, cid, entries.EVENT_STAKE_RECHECKED) == []

    chain.fresh = True
    orch.clear_halt(cid, "operator", FINALIZATION + M, reason="chain endpoint restored")
    await _tick(orch, FINALIZATION + 2 * M)
    statuses = {c.hotkey: c.status for c in repo.list_contenders(orch.conn, cid)}
    assert statuses["hk-b"] == "REJECTED" and statuses["hk-a"] != "REJECTED"


async def test_stake_recheck_can_be_turned_off(
    orchestrator_factory, fixture_repos, tmp_path: Path
):
    from orchestrator_support import FINALIZATION, seed_items

    chain = _StakeChain({"hk-a": 1.0})
    orch = _staked_orchestrator(
        orchestrator_factory, fixture_repos, chain, enrollment_stake_recheck_enabled=False
    )
    cid = await start_and_enroll(orch, build_manifest("comp-stake-off"), ["hk-a"])
    await _tick(orch, START + 6 * M)
    assert _events(orch, cid, entries.EVENT_PRECHECKED)[0]["ok"] is True
    seed_items(orch, cid, tmp_path / "items")
    await _tick(orch, FINALIZATION)
    assert repo.list_contenders(orch.conn, cid)[0].status != "REJECTED"
    assert _events(orch, cid, entries.EVENT_STAKE_RECHECKED) == []


async def test_a_stake_warning_alone_does_not_block_the_trial_build(
    orchestrator_factory, fixture_repos
):
    chain = _StakeChain({"hk-a": 1.0})
    orch = _staked_orchestrator(
        orchestrator_factory, fixture_repos, chain, enrollment_trial_build_enabled=True
    )
    cid = await start_and_enroll(orch, build_manifest("comp-stake-trial"), ["hk-a"])
    await _tick(orch, START + 6 * M)
    await _tick(orch, START + 7 * M)
    assert _events(orch, cid, entries.EVENT_PRECHECKED)[0]["problems"] == ["stake_below_floor"]
    assert len(orch.runner.build_calls) == 1
