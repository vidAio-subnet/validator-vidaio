"""THE SUBMISSION ARCHIVE INVARIANT.

Every contender that can still win has an archived submission.

The bug: FINALIZING_SUBMISSIONS caught any checkout failure, logged "backup
skipped", and carried on to record the COMBINED backup_ref and advance the phase.
If the checkout then succeeded during validation — a transient failure is by
definition transient — that contender competed, could win, and had nothing in the
audit store to check its win against. The combined ref certified an archive set
that did not exist.

The fix splits by fault class:
  - CONTENDER fault (unsafe/oversize tree) -> that contender is REJECTED here and
    cannot advance, so the invariant holds and the others proceed;
  - INFRA fault (unreachable checkout, audit store down, anything unknown) ->
    finalization does NOT advance and the pipeline halts. We never certify a
    backup set we could not produce.
"""

from __future__ import annotations

import logging

from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.states import Phase

from orchestrator_support import (
    END,
    FINALIZATION,
    M,
    FlakyRepoProvider,
    build_manifest,
    drive_to_completion,
    enroll,
    events_of,
    phase,
    repo_url,
    seed_items,
    start_and_enroll,
)


async def test_a_transient_checkout_failure_cannot_produce_an_unarchived_winner(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    """The exact scenario: hk-b's checkout fails during backup, then recovers.

    Before the fix hk-b competed with no archived submission. Now finalization
    refuses to advance while its submission is unarchived; once the blip clears,
    hk-b runs WITH an archive. Nobody is silently disqualified either.
    """
    # 3 failures: the two in-step backup attempts plus one more, so the whole
    # stage genuinely fails before the provider recovers.
    provider = FlakyRepoProvider(fixture_repos, fail_for={repo_url("hk-b"): 3})
    orch = orchestrator_factory(repo_provider=provider)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")

    with caplog.at_level(logging.CRITICAL):
        await orch.step(FINALIZATION)

    # The phase did NOT advance and the combined ref was never recorded.
    assert pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.FINALIZING_SUBMISSIONS
    assert not [
        e
        for e in repo.list_events(orch.conn, cid)
        if e["event_type"] == "transition" and e["to_phase"] == "VALIDATING"
    ]
    # hk-a's archive is already recorded; hk-b's is not — and hk-b is untouched.
    archived = pers.archived_submissions(orch.conn, cid)
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    assert by_hotkey["hk-a"].contender_id in archived
    assert by_hotkey["hk-b"].contender_id not in archived
    assert by_hotkey["hk-b"].status == "ENROLLED"  # not rejected for our blip

    # Blip over: the operator clears the halt and finalization completes.
    assert orch.clear_halt(
        cid, "ops", FINALIZATION + M, reason="audit store restored"
    )
    await orch.step(FINALIZATION + 2 * M)
    assert phase(orch, cid) is Phase.VALIDATING
    archived = pers.archived_submissions(orch.conn, cid)
    assert {by_hotkey["hk-a"].contender_id, by_hotkey["hk-b"].contender_id} <= set(
        archived
    )
    # hk-a was NOT re-archived on re-entry (idempotent, one event per contender).
    events = events_of(orch, cid, pers.EVENT_SUBMISSION_ARCHIVED)
    assert len(events) == 2

    await drive_to_completion(orch, cid, tmp_path, seed=False)
    assert phase(orch, cid) is Phase.COMPLETED
    # Every contender that could win has an archived submission.
    ranked = repo.ranking(orch.conn, cid)
    assert {c.hotkey for c in ranked} == {"hk-a", "hk-b"}
    assert all(c.contender_id in archived for c in ranked)


async def test_archive_and_validation_release_every_consumed_fresh_checkout(
    orchestrator_factory, fixture_repos, tmp_path
):
    provider = FlakyRepoProvider(fixture_repos)
    orch = orchestrator_factory(repo_provider=provider)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")

    await orch.step(FINALIZATION)  # archive each tree, then enter VALIDATING
    assert len(provider.calls) == 2
    assert len(provider.releases) == 2

    await orch.step(FINALIZATION + 2 * M)  # inspect each tree, then enter BUILDING
    assert len(provider.calls) == 4
    assert len(provider.releases) == 4


async def test_an_unarchivable_submission_cannot_compete(
    orchestrator_factory, fixture_repos, tmp_path
):
    """A tree whose OWN content makes it unarchivable is the contender's fault:
    rejected here, unable to advance, everyone else proceeds."""
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    enroll(orch, cid, "hk-link")  # ships `stolen-credentials -> /etc/passwd`
    seed_items(orch, cid, tmp_path / "item-src")

    await orch.step(FINALIZATION)
    assert not pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.VALIDATING

    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    assert by_hotkey["hk-link"].status == "REJECTED"
    archived = pers.archived_submissions(orch.conn, cid)
    assert by_hotkey["hk-link"].contender_id not in archived
    assert by_hotkey["hk-a"].contender_id in archived

    await drive_to_completion(orch, cid, tmp_path, seed=False)
    assert phase(orch, cid) is Phase.COMPLETED
    # The rejected contender is still LISTED (never silently dropped) but it was
    # never built, never evaluated and scores zero — it cannot win, so the
    # invariant "everyone who can win has an archive" holds.
    ranking = repo.ranking(orch.conn, cid)
    assert [c.hotkey for c in ranking] == ["hk-a", "hk-link"]
    assert ranking[-1].hotkey == "hk-link" and ranking[-1].final_score == 0.0
    assert by_hotkey["hk-link"].contender_id not in pers.archived_submissions(
        orch.conn, cid
    )
    assert repo.get_contender(orch.conn, by_hotkey["hk-link"].contender_id).image_digest is None


async def test_the_combined_backup_ref_covers_exactly_the_archived_set(
    orchestrator_factory, fixture_repos, tmp_path
):
    """The ref is evidence, so it must be derived from real archived digests."""
    import hashlib

    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)

    archived = pers.archived_submissions(orch.conn, cid)
    expected = hashlib.sha256(
        "\n".join(sorted(f"{cid_}:{d}" for cid_, d in archived.items())).encode("utf-8")
    ).hexdigest()
    transition = [
        e
        for e in repo.list_events(orch.conn, cid)
        if e["event_type"] == "transition" and e["to_phase"] == "VALIDATING"
    ]
    assert transition
    assert f"audit://submissions/sha256:{expected}" in transition[0]["payload_json"]


async def test_a_crash_mid_backup_re_archives_only_what_is_missing(
    orchestrator_factory, fixture_repos, tmp_path
):
    """Idempotent re-entry: the evidence is the event log, not memory."""
    provider = FlakyRepoProvider(fixture_repos, fail_for={repo_url("hk-b"): 3})
    orch1 = orchestrator_factory(repo_provider=provider)
    cid = await start_and_enroll(orch1, build_manifest(), ["hk-a", "hk-b"])
    seed_items(orch1, cid, tmp_path / "item-src")
    await orch1.step(FINALIZATION)  # hk-a archived, then halt on hk-b
    assert pers.is_halted(orch1.conn, cid)
    first = dict(pers.archived_submissions(orch1.conn, cid))
    assert len(first) == 1

    # "Restart" over the same DB with a healthy provider.
    orch2 = orchestrator_factory(repos=fixture_repos)
    assert orch2.clear_halt(
        cid, "ops", FINALIZATION + M, reason="repository access restored"
    )
    await orch2.step(FINALIZATION + 2 * M)
    assert phase(orch2, cid) is Phase.VALIDATING
    second = pers.archived_submissions(orch2.conn, cid)
    assert len(second) == 2
    # The already-archived contender kept its ORIGINAL digest, unre-archived.
    for contender_id, digest in first.items():
        assert second[contender_id] == digest
    assert len(events_of(orch2, cid, pers.EVENT_SUBMISSION_ARCHIVED)) == 2


async def test_review_window_still_reaches_completion_after_a_backup_halt(
    orchestrator_factory, fixture_repos, tmp_path
):
    """A backup halt delays; it never fails the competition (spec §14)."""
    provider = FlakyRepoProvider(fixture_repos, fail_for={repo_url("hk-a"): 3})
    orch = orchestrator_factory(repo_provider=provider)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    comp = repo.get_competition(orch.conn, cid)
    assert comp.status is Phase.FINALIZING_SUBMISSIONS
    assert comp.failure_reason is None

    orch.clear_halt(cid, "ops", FINALIZATION + M, reason="repository access restored")
    for at in (
        FINALIZATION + 2 * M,
        FINALIZATION + 3 * M,
        FINALIZATION + 4 * M,
        FINALIZATION + 10 * M,
        FINALIZATION + 15 * M,
        END + M,
    ):
        await orch.step(at)
    assert phase(orch, cid) is Phase.COMPLETED
