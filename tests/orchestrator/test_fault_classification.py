"""One untrusted contender must never halt the competition.

Two layers:
- the pure classifier (`classify_failure` / `fault_code`), one case per class;
- the orchestrator behaviour it drives: a contender that exits non-zero or
  produces nothing is zero-scored while everyone else finishes, and a genuine
  infra failure still requeues-then-halts exactly as before.
"""

from __future__ import annotations

import logging
import sqlite3

import pytest

from vidaio.core.resilience import RetriesExhausted
from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.orchestrator.failures import (
    Fault,
    classify_failure,
    fault_code,
    unwrap,
)
from vidaio.competition.orchestrator.scoring_client import ScoringClientError
from vidaio.competition.runners.errors import (
    BatchExecutionError,
    BatchTimeout,
    BuildError,
    BuildTimeout,
    CheckoutError,
    ContenderBuildError,
    InputStagingError,
    OutputRejectedError,
    OversizeOutputError,
    RunnerUnavailableError,
    SandboxIsolationError,
    SandboxProbeUnavailableError,
    SolutionExitError,
    UnknownImageError,
    UnsafePathError,
)
from vidaio.competition.states import Phase

from orchestrator_support import (
    CONTENDER_SHAS,
    END,
    FINALIZATION,
    M,
    ContenderFaultRunner,
    FakeRunner,
    build_manifest,
    events_of,
    phase,
    repo_url,
    seed_items,
    start_and_enroll,
)

# ---- the classifier -------------------------------------------------------------

CONTENDER_CASES = [
    SolutionExitError("exit 1"),
    BatchTimeout("blew its budget"),
    OutputRejectedError("output is a symlink"),
    OversizeOutputError("filled /output"),
    UnsafePathError("fifo in the tree"),
    ContenderBuildError("its Dockerfile does not build"),
    BuildTimeout("its build never finished"),
    ScoringClientError("undecodable media", status_code=400),
    ScoringClientError("unsupported media type", status_code=415),
    # 422s the worker attributed to the CONTENDER'S OWN OUTPUT.
    ScoringClientError(
        "the output is not a regular file",
        status_code=422,
        error_code="not_a_regular_file",
        error_field="output",
    ),
    ScoringClientError(
        "the output does not hash to its claimed digest",
        status_code=422,
        error_code="digest_mismatch",
        error_field="output",
    ),
]

INFRA_CASES = [
    BatchExecutionError("docker daemon said no"),
    UnknownImageError("image gone"),
    InputStagingError("sealed input missing"),
    RunnerUnavailableError("docker down"),
    SandboxIsolationError("flags did not take effect"),
    SandboxProbeUnavailableError("the probe could not be launched"),
    CheckoutError("the git host is unreachable"),
    BuildError("our docker CLI is unusable"),
    ScoringClientError("worker exploded", status_code=502),
    ScoringClientError("connect timeout"),  # transport, no status
    ScoringClientError("rate limited", status_code=429),
    # NOT a contender fault: a scorer-identity disagreement is OUR configuration.
    ScoringClientError("scorer_version_mismatch", status_code=409),
    ScoringClientError("bad control credentials", status_code=401),
    ScoringClientError("wrong route", status_code=404),
    # --- 422s that are OURS --------------------------------
    ScoringClientError(  # the sealed input WE named is gone
        "reference missing",
        status_code=422,
        error_code="file_missing",
        error_field="reference",
    ),
    ScoringClientError(  # ... or unreadable
        "miner input unreadable",
        status_code=422,
        error_code="unreadable_input",
        error_field="miner_input",
    ),
    ScoringClientError(  # OUR request params
        "vmaf_threshold is not a number",
        status_code=422,
        error_code="invalid_param",
    ),
    ScoringClientError(  # OUR manifest track
        "unsupported track", status_code=422, error_code="unsupported_track"
    ),
    ScoringClientError("worker rejected something", status_code=422),  # untypeable
    ConnectionError("transport"),
    sqlite3.OperationalError("database is locked"),
    RuntimeError("an orchestrator bug we did not anticipate"),
]


@pytest.mark.parametrize("exc", CONTENDER_CASES, ids=lambda e: type(e).__name__)
def test_contender_faults_are_classified_as_contender(exc):
    assert classify_failure(exc) is Fault.CONTENDER


@pytest.mark.parametrize("exc", INFRA_CASES, ids=lambda e: type(e).__name__)
def test_infra_faults_are_classified_as_infra(exc):
    assert classify_failure(exc) is Fault.INFRA


def test_unknown_failures_default_to_infra_not_contender():
    """Fail closed: OUR bug must halt loudly, never silently zero a contender."""
    assert classify_failure(Exception("mystery")) is Fault.INFRA


def test_classification_sees_through_retries_exhausted():
    try:
        try:
            raise SolutionExitError("exit 1")
        except SolutionExitError as inner:
            raise RetriesExhausted("failed after 2 attempts") from inner
    except RetriesExhausted as exhausted:
        assert isinstance(unwrap(exhausted), SolutionExitError)
        assert classify_failure(exhausted) is Fault.CONTENDER
        assert fault_code(exhausted) == "SOLUTION_EXIT_NONZERO"


def test_fault_codes_are_stable_strings():
    assert fault_code(SolutionExitError("x")) == "SOLUTION_EXIT_NONZERO"
    assert fault_code(BatchTimeout("x")) == "SOLUTION_TIMEOUT"
    assert fault_code(OutputRejectedError("x")) == "OUTPUT_REJECTED"
    assert fault_code(OversizeOutputError("x")) == "OUTPUT_OVERSIZE"
    assert fault_code(ContenderBuildError("x")) == "BUILD_FAILED"
    assert fault_code(BuildTimeout("x")) == "BUILD_TIMEOUT"
    assert fault_code(ScoringClientError("x", status_code=422)) == "SCORER_HTTP_422"
    # The worker's typed reason beats the bare status in the event log.
    assert (
        fault_code(
            ScoringClientError(
                "x", status_code=422, error_code="not_a_regular_file", error_field="output"
            )
        )
        == "SCORER_NOT_A_REGULAR_FILE"
    )


# ---- the 422 split, stated directly ------------------------


@pytest.mark.parametrize(
    "field, expected",
    [("output", True), ("reference", False), ("miner_input", False), (None, False)],
)
def test_a_422_is_contender_fault_only_for_the_output_field(field, expected):
    """A 422 about OUR half of the request must never zero a contender."""
    from vidaio.competition.orchestrator.failures import (
        scorer_rejection_is_contender_fault,
    )

    assert (
        scorer_rejection_is_contender_fault(422, "file_missing", field) is expected
    )


def test_an_untypeable_422_is_infra_not_contender():
    """Fail closed: a worker we cannot understand must not cost a contender."""
    from vidaio.competition.orchestrator.failures import (
        scorer_rejection_is_contender_fault,
    )

    assert scorer_rejection_is_contender_fault(422, None, None) is False
    assert scorer_rejection_is_contender_fault(422, "brand_new_reason", "output") is False


def test_the_http_client_carries_the_workers_typed_rejection():
    """The classifier can only narrow what the client actually reports."""
    import httpx

    from vidaio.competition.orchestrator.scoring_client import _typed_rejection

    response = httpx.Response(
        422,
        json={"detail": {"error": "not_a_regular_file", "field": "output", "path": "/x"}},
    )
    assert _typed_rejection(response) == ("not_a_regular_file", "output")
    assert _typed_rejection(httpx.Response(422, text="not json")) == (None, None)
    assert _typed_rejection(httpx.Response(422, json={"detail": "plain"})) == (None, None)


# ---- orchestrator behaviour -----------------------------------------------------


async def _drive(orch, cid, tmp_path):
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)  # builds -> EVALUATING
    await orch.step(FINALIZATION + 4 * M)  # batches
    await orch.step(FINALIZATION + 10 * M)  # -> SCORING
    await orch.step(FINALIZATION + 15 * M)  # scoring -> AWAITING_END_TIME


async def test_contender_exit_one_is_zeroed_and_competition_continues(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    runner = ContenderFaultRunner(tmp_path / "work" / "outputs")
    runner.fault_for[CONTENDER_SHAS["hk-exit"][1]] = SolutionExitError(
        "batch 0 solution exited 1: this solution refuses to work"
    )
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-exit"])
    with caplog.at_level(logging.CRITICAL):
        await _drive(orch, cid, tmp_path)

    # No halt, no CRITICAL, and the lifecycle ran to the review window.
    assert not pers.is_halted(orch.conn, cid)
    assert not [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert phase(orch, cid) is Phase.AWAITING_END_TIME

    # The fault is recorded with a machine-readable code on the batch + event log.
    faults = pers.contender_fault_events(orch.conn, cid)
    assert faults and "SOLUTION_EXIT_NONZERO" in faults[0]["payload_json"]
    failed = orch.conn.execute(
        "SELECT status, failure_code FROM batches WHERE competition_id = ?"
        " AND status = 'FAILED'",
        (cid,),
    ).fetchall()
    assert failed and "SOLUTION_EXIT_NONZERO" in failed[0]["failure_code"]

    # That contender is zeroed on every item; the honest one scored positively.
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    rows = orch.conn.execute(
        "SELECT contender_id, valid, item_score FROM performance_history"
        " WHERE competition_id = ?",
        (cid,),
    ).fetchall()
    untrusted = [r for r in rows if r["contender_id"] == by_hotkey["hk-exit"].contender_id]
    honest = [r for r in rows if r["contender_id"] == by_hotkey["hk-a"].contender_id]
    assert len(untrusted) == 3 and all(r["valid"] == 0 and r["item_score"] == 0 for r in untrusted)
    assert len(honest) == 3 and all(r["valid"] == 1 and r["item_score"] > 0 for r in honest)

    await orch.step(END + M)
    assert phase(orch, cid) is Phase.COMPLETED
    ranking = repo.ranking(orch.conn, cid)
    assert ranking[0].hotkey == "hk-a"  # the untrusted contender ranks last, not absent
    assert {c.hotkey for c in ranking} == {"hk-a", "hk-exit"}


async def test_missing_output_is_zeroed_without_ever_calling_the_scorer(
    orchestrator_factory, fixture_repos, tmp_path
):
    """A code-0 run that writes nothing must never reach ffmpeg (#14)."""
    runner = ContenderFaultRunner(tmp_path / "work" / "outputs")
    runner.silent_for.add(CONTENDER_SHAS["hk-silent"][1])
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-silent"])
    await _drive(orch, cid, tmp_path)

    assert not pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.AWAITING_END_TIME

    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    silent_id = by_hotkey["hk-silent"].contender_id
    # The scoring client was NEVER asked about the silent contender's items.
    assert all(cid_ != silent_id for cid_, _item in orch.scoring_client.calls)

    zeroed = events_of(orch, cid, pers.EVENT_ITEM_ZEROED)
    assert len(zeroed) == 3
    assert "METRIC_MISSING" in zeroed[0]["payload_json"]
    assert "no output produced" in zeroed[0]["payload_json"]

    rows = orch.conn.execute(
        "SELECT valid, item_score, output_bytes, audit_bundle_digest"
        " FROM performance_history WHERE contender_id = ?",
        (silent_id,),
    ).fetchall()
    assert len(rows) == 3
    assert all(r["valid"] == 0 and r["item_score"] == 0 for r in rows)
    # Zeroed rows are audit-linked exactly like scored ones (completion gate).
    assert all(r["audit_bundle_digest"] for r in rows)


async def test_scorer_4xx_zeroes_the_item_but_5xx_halts(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    from orchestrator_support import FakeScoringClient, client_conn

    class RejectingClient(FakeScoringClient):
        """Rejects one contender's bytes with 422 (contender fault)."""

        def __init__(self, reject_contender_hotkey: str) -> None:
            super().__init__()
            self.reject = reject_contender_hotkey

        def score_item(self, competition_id, contender_id, item, output):
            contender = repo.get_contender(self.conn, contender_id)
            if contender is not None and contender.hotkey == self.reject:
                self.calls.append((contender_id, item.item_id))
                raise ScoringClientError(
                    "unusable container",
                    status_code=422,
                    error_code="not_a_regular_file",
                    error_field="output",  # the CONTENDER's own bytes
                )
            return super().score_item(competition_id, contender_id, item, output)

    client = RejectingClient("hk-b")
    orch = orchestrator_factory(scoring_client=client, repos=fixture_repos)
    client.conn = client_conn(orch.core.db_path)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    await _drive(orch, cid, tmp_path)

    assert not pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.AWAITING_END_TIME
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    rejected_rows = orch.conn.execute(
        "SELECT valid, item_score FROM performance_history WHERE contender_id = ?",
        (by_hotkey["hk-b"].contender_id,),
    ).fetchall()
    assert len(rejected_rows) == 3
    assert all(r["valid"] == 0 for r in rejected_rows)
    # A deterministic 4xx must not have burned the transport retry budget.
    assert len([c for c in client.calls if c[0] == by_hotkey["hk-b"].contender_id]) == 3


async def test_a_422_about_our_own_input_halts_instead_of_zeroing_the_contender(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    """The round-2 bypass: every 422 used to be the contender's fault.

    A missing REFERENCE is our sealed input pool failing, not a bad submission.
    Zeroing for it would silently corrupt the result; it must halt loudly instead.
    """
    from orchestrator_support import FakeScoringClient, client_conn

    class OurFaultClient(FakeScoringClient):
        def score_item(self, competition_id, contender_id, item, output):
            self.calls.append((contender_id, item.item_id))
            raise ScoringClientError(
                "reference missing",
                status_code=422,
                error_code="file_missing",
                error_field="reference",
            )

    client = OurFaultClient()
    orch = orchestrator_factory(scoring_client=client, repos=fixture_repos)
    client.conn = client_conn(orch.core.db_path)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    with caplog.at_level(logging.CRITICAL):
        await _drive(orch, cid, tmp_path)

    assert pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.SCORING  # halted, never failed
    # NOTHING was zero-scored: no contender was blamed for our missing input.
    assert events_of(orch, cid, pers.EVENT_ITEM_ZEROED) == []
    assert (
        orch.conn.execute(
            "SELECT COUNT(*) AS n FROM performance_history WHERE competition_id = ?",
            (cid,),
        ).fetchone()["n"]
        == 0
    )


async def test_an_infra_build_failure_halts_and_never_marks_build_failed(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    """review #14 round 2: `except Exception -> BUILD_FAILED` blamed our outage.

    A contender eliminated for our broken docker never gets its run back, so an
    infra build failure must halt with the contender still ACCEPTED.
    """
    runner = FakeRunner(tmp_path / "work" / "outputs")
    runner.infra_fail_build_for.add(repo_url("hk-b"))
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    with caplog.at_level(logging.CRITICAL):
        await orch.step(FINALIZATION + 3 * M)

    assert pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.BUILDING  # halted, never FAILED
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    assert by_hotkey["hk-b"].status == "ACCEPTED"  # NOT judged
    assert events_of(orch, cid, "contender_build_failed") == []

    # Operator fixes docker and clears the halt: the contender still gets its run.
    runner.infra_fail_build_for.clear()
    assert orch.clear_halt(
        cid, "ops", FINALIZATION + 4 * M, reason="infrastructure restored"
    )
    await orch.step(FINALIZATION + 5 * M)
    assert phase(orch, cid) is Phase.EVALUATING
    assert {c.status for c in repo.list_contenders(orch.conn, cid)} == {"BUILT"}


async def test_a_contender_build_failure_still_marks_only_that_contender(
    orchestrator_factory, fixture_repos, tmp_path
):
    """The other half of the split: a Dockerfile that does not build IS the
    contender's fault, and must keep behaving exactly as before."""
    runner = FakeRunner(tmp_path / "work" / "outputs")
    runner.fail_build_for.add(repo_url("hk-b"))
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)

    assert not pers.is_halted(orch.conn, cid)
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    assert by_hotkey["hk-b"].status == "BUILD_FAILED"
    failed = events_of(orch, cid, "contender_build_failed")
    assert len(failed) == 1 and "BUILD_FAILED" in failed[0]["payload_json"]


async def test_a_probe_that_cannot_run_halts_and_disqualifies_nobody(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    """review #14 round 2: an unattestable boundary is OUR failure.

    The runner used to return an all-False report when the probe could not be
    launched or inspected — indistinguishable from an attested escape, and it
    disqualified an innocent contender.
    """
    runner = FakeRunner(tmp_path / "work" / "outputs")
    runner.probe_unavailable_for.add(CONTENDER_SHAS["hk-b"][1])
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    with caplog.at_level(logging.CRITICAL):
        await orch.step(FINALIZATION + 3 * M)

    assert pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.BUILDING
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    assert by_hotkey["hk-b"].status == "ACCEPTED"  # never disqualified
    assert events_of(orch, cid, "isolation_probe_failed") == []
    # No sandbox row claims a failed attestation that never happened.
    rows = orch.conn.execute(
        "SELECT status FROM sandboxes WHERE competition_id = ?", (cid,)
    ).fetchall()
    assert all(r["status"] == "CREATED" for r in rows)


async def test_a_probe_that_ran_and_failed_still_disqualifies_that_contender(
    orchestrator_factory, fixture_repos, tmp_path
):
    """The distinction is only useful if the real verdict still bites."""
    runner = FakeRunner(tmp_path / "work" / "outputs")
    runner.fail_probe_for.add(CONTENDER_SHAS["hk-b"][1])
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)

    assert not pers.is_halted(orch.conn, cid)
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    assert by_hotkey["hk-b"].status == "BUILD_FAILED"
    assert len(events_of(orch, cid, "isolation_probe_failed")) == 1


async def test_an_infra_checkout_failure_in_validation_halts_and_rejects_nobody(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    """review #14 round 2: `except Exception -> REJECTED` in validation.

    An unreachable git host is not a bad submission, and a REJECTED contender
    never comes back.
    """
    from orchestrator_support import FlakyRepoProvider

    provider = FlakyRepoProvider(fixture_repos, fail_for={repo_url("hk-b"): 99})
    orch = orchestrator_factory(repo_provider=provider)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")
    with caplog.at_level(logging.CRITICAL):
        await orch.step(FINALIZATION)  # backup halts first (see new-5)
    assert pers.is_halted(orch.conn, cid)

    # Even once the backup is out of the way, validation must not reject either.
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    assert by_hotkey["hk-b"].status == "ENROLLED"
    assert not any(
        "REJECTED" in (e["payload_json"] or "")
        for e in events_of(orch, cid, "contender_validated")
    )


async def test_infra_batch_failure_still_requeues_then_halts(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    """The classifier must not have weakened the INFRA path (spec §14)."""
    runner = FakeRunner(tmp_path / "work" / "outputs")
    runner.batch_fail_times = 10_000  # BatchExecutionError == infra
    orch = orchestrator_factory(
        runner=runner, repos=fixture_repos, max_batch_requeues=1
    )
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)
    await orch.step(FINALIZATION + 4 * M)  # requeue #1
    assert len(events_of(orch, cid, "batch_requeued")) >= 1
    assert not pers.contender_fault_events(orch.conn, cid)  # never mislabelled
    with caplog.at_level(logging.CRITICAL):
        await orch.step(FINALIZATION + 5 * M)  # budget exhausted -> halt
    assert pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.EVALUATING  # halted, never FAILED
