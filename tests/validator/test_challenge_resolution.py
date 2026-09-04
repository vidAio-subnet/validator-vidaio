"""review #5: every challenge the validator FETCHES is resolved.

`/challenge/next` checks an asset out of the pool and leaves its commitment
unrevealed until `/challenge/{id}/resolve` arrives. The validator used to call
only the former, so after the initial assets were consumed every later round got
`pool_exhausted` forever. These tests pin the four resolution paths (success,
failure, timeout, shutdown), the startup recovery of stranded challenges, and the
retry discipline for a resolve that itself fails.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from vidaio.audit import ArtifactRef
from vidaio.validator import miner_manager
from vidaio.validator.evidence import ScorePacketEvidence

from validator_support import (
    OTHER_VALIDATOR_IDENTITY,
    VALIDATOR_IDENTITY,
    LegacyChallengeClient,
    mk_neuron,
)


def metric(validator, name: str, labels: dict[str, str] | None = None) -> float | None:
    return validator.health.registry.get_sample_value(name, labels or {})


async def test_successful_round_resolves_every_challenge_it_fetched(
    validator, chain, miner_client, challenge_client, conn
):
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "upscaling"}

    report = await validator.run_round()

    assert sorted(challenge_client.resolves) == [
        ("ch-compression", "resolved"),
        ("ch-upscaling", "resolved"),
    ]
    assert report.resolved_challenges == {
        "ch-compression": "resolved",
        "ch-upscaling": "resolved",
    }
    # nothing left checked out
    assert miner_manager.inflight_challenges(conn) == []
    assert metric(validator, "vidaio_validator_challenges_resolved_total", {"outcome": "resolved"}) == 2.0


async def test_successful_resolve_publishes_retired_holdout_for_public_audit(
    validator, chain, miner_client, challenge_client, conn, store
):
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    await validator.run_round()

    encoded = ScorePacketEvidence(conn).reference_original_refs("ch-compression")
    assert len(encoded) == 1
    ref = ArtifactRef.model_validate_json(encoded[0])
    assert store.is_released(ref)


async def test_miner_timeouts_still_resolve_the_challenge(
    validator, chain, miner_client, challenge_client, conn
):
    """A round where every miner timed out still owes the pool its asset back."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    miner_client.task_timeout_uids = {1}

    report = await validator.run_round()

    assert report.zeroed == {1: "availability:timeout"}
    assert report.non_punitive_skips == {}
    assert challenge_client.resolves == [("ch-compression", "resolved")]
    assert miner_manager.inflight_challenges(conn) == []


async def test_aborted_round_resolves_its_challenge_as_expired(
    validator, chain, miner_client, challenge_client, conn
):
    """A round that dies after fetching resolves with 'expired', not never."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    async def explode(*_args, **_kwargs):
        raise RuntimeError("round died after checking the asset out")

    validator._dispatch_all = explode

    with pytest.raises(RuntimeError):
        await validator.run_round()

    assert challenge_client.resolves == [("ch-compression", "expired")]
    assert miner_manager.inflight_challenges(conn) == []
    assert metric(validator, "vidaio_validator_challenges_resolved_total", {"outcome": "expired"}) == 1.0


async def test_challenge_fetch_failure_leaves_nothing_in_flight(
    validator, chain, miner_client, challenge_client, conn
):
    """A failed /challenge/next consumed nothing HERE — no resolve is owed."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    challenge_client.fail_tracks = {"compression"}

    await validator.run_round()

    assert challenge_client.resolves == []
    assert miner_manager.inflight_challenges(conn) == []


async def test_failed_resolve_is_retained_and_retried_next_round(
    validator, chain, miner_client, challenge_client, conn
):
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    challenge_client.fail_resolve_ids = {"ch-compression"}

    await validator.run_round()

    # the obligation survives so the pool cannot be permanently stranded
    rows = miner_manager.inflight_challenges(conn)
    assert [r["challenge_id"] for r in rows] == ["ch-compression"]
    assert rows[0]["outcome"] == "resolved"
    assert challenge_client.resolves == []
    assert metric(validator, "vidaio_validator_challenge_resolve_failures_total") == 1.0

    challenge_client.fail_resolve_ids.clear()
    await validator._drain_inflight_challenges()

    assert challenge_client.resolves == [("ch-compression", "resolved")]
    assert miner_manager.inflight_challenges(conn) == []


async def test_already_terminal_challenge_is_dropped_not_retried_forever(
    validator, challenge_client, conn
):
    """A 409 from the service means someone already resolved it — stop driving it."""
    miner_manager.record_inflight_challenge(
        conn,
        challenge_id="ch-gone",
        round_id="r0",
        track="compression",
        fetched_at=miner_manager.utc_now_iso(),
    )
    challenge_client.terminal.add("ch-gone")  # the service considers it terminal

    drained = await validator._drain_inflight_challenges()

    assert drained == 0
    assert challenge_client.resolves == []
    assert miner_manager.inflight_challenges(conn) == []  # row dropped, not retried


async def test_startup_recovery_resolves_challenges_stranded_by_a_crash(
    validator, challenge_client, conn
):
    """The recovery pass a crashed process's assets depend on."""
    # what a hard crash mid-round leaves behind: an open round ledger row and the
    # challenges that round had checked out
    miner_manager.begin_round(conn, "crashed-round", 42, miner_manager.utc_now_iso())
    for cid in ("ch-a", "ch-b"):
        miner_manager.record_inflight_challenge(
            conn,
            challenge_id=cid,
            round_id="crashed-round",
            track="compression",
            fetched_at=miner_manager.utc_now_iso(),
        )
    miner_manager.set_inflight_outcome(conn, "ch-a", "resolved")  # a had been scored

    drained = await validator.recover_inflight_challenges()

    assert drained == 2
    assert sorted(challenge_client.resolves) == [("ch-a", "resolved"), ("ch-b", "expired")]
    assert miner_manager.inflight_challenges(conn) == []
    # the partial round stays visible as a partial round — it is never completed
    assert [r["round_id"] for r in miner_manager.uncommitted_rounds(conn)] == [
        "crashed-round"
    ]


async def test_run_recovers_at_startup_and_drains_at_shutdown(
    validator, challenge_client, conn
):
    """Both ends of the process lifecycle drain the in-flight table."""
    miner_manager.record_inflight_challenge(
        conn,
        challenge_id="ch-late",
        round_id="r0",
        track="compression",
        fetched_at=miner_manager.utc_now_iso(),
    )
    # the challenge service is down at startup, so recovery cannot drain it
    challenge_client.fail_resolve_ids = {"ch-late"}

    async def one_round() -> None:
        challenge_client.fail_resolve_ids.clear()  # service comes back
        validator.request_stop()

    validator.run_round = one_round
    await asyncio.wait_for(validator.run(), timeout=5.0)

    # the SHUTDOWN drain caught what the startup pass could not
    assert challenge_client.resolves == [("ch-late", "expired")]
    assert miner_manager.inflight_challenges(conn) == []


# --- the LOST RESPONSE (the gap the in-flight table cannot see) ---------------
#
# `POST /challenge/next` is not idempotent: it checks an asset out and records a
# dispatched challenge BEFORE its response reaches us. If that response is lost —
# connection reset, process killed in the window between the service's commit and
# our `record_inflight_challenge` — the challenge exists, its asset is `in_use`
# and its commitment is unrevealed, and the only process that could resolve it
# never learned the id. Nothing to drain, nothing to retry, and the service's own
# `recover_orphans` will not touch it either: the media is present and the
# challenge is perfectly valid, merely abandoned. So the validator ASKS, via
# GET /challenges?status=dispatched&older_than_seconds=N.


async def test_startup_sweep_expires_a_challenge_whose_response_was_lost(
    validator, challenge_client, conn
):
    challenge_client.lose_response(
        "ch-orphan",
        track="compression",
        age_seconds=7200.0,
        owner=VALIDATOR_IDENTITY,  # the service says it is OURS
    )

    expired = await validator.sweep_orphaned_challenges()

    assert expired == ["ch-orphan"]
    assert challenge_client.resolves == [("ch-orphan", "expired")]
    # the service no longer holds it, so the asset is back in the pool
    assert challenge_client.dispatched == {}
    assert metric(
        validator, "vidaio_validator_orphaned_challenges_swept_total"
    ) == 1.0


async def test_sweep_never_touches_a_challenge_this_validator_is_driving(
    validator, challenge_client, conn
):
    """An in-flight row means the normal drain owns it — the sweep must not race it."""
    challenge_client.lose_response(
        "ch-mine", track="compression", age_seconds=9999.0, owner=VALIDATOR_IDENTITY
    )
    miner_manager.record_inflight_challenge(
        conn,
        challenge_id="ch-mine",
        round_id="r1",
        track="compression",
        fetched_at=miner_manager.utc_now_iso(),
    )

    assert await validator.sweep_orphaned_challenges() == []
    assert challenge_client.resolves == []


async def test_sweep_is_age_gated_so_a_live_rounds_challenge_is_never_expired(
    validator, chain, miner_client, challenge_client, conn
):
    """A round's own challenge is young; only abandoned ones are old enough."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    await validator.run_round()
    challenge_client.resolves.clear()
    # Re-dispatch it at age 0, as a live round would have it.
    challenge_client.terminal.clear()
    challenge_client.lose_response(
        "ch-compression",
        track="compression",
        age_seconds=0.0,
        owner=VALIDATOR_IDENTITY,
    )

    assert await validator.sweep_orphaned_challenges() == []
    assert challenge_client.resolves == []
    # and the sweep asked with the configured age gate AND our identity, not a guess
    assert challenge_client.list_calls[-1] == (
        validator.config.orphan_sweep_age_seconds,
        VALIDATOR_IDENTITY,
    )


async def test_startup_recovery_drains_first_then_sweeps(
    validator, challenge_client, conn
):
    """Both halves run, and the drain's rows are known to the sweep (no double-resolve)."""
    miner_manager.record_inflight_challenge(
        conn,
        challenge_id="ch-known",
        round_id="crashed",
        track="compression",
        fetched_at=miner_manager.utc_now_iso(),
    )
    challenge_client.lose_response(
        "ch-known", track="compression", age_seconds=7200.0, owner=VALIDATOR_IDENTITY
    )
    challenge_client.lose_response(
        "ch-orphan", track="compression", age_seconds=7200.0, owner=VALIDATOR_IDENTITY
    )

    drained = await validator.recover_inflight_challenges()

    assert drained == 1
    # ch-known resolved ONCE through the in-flight path, ch-orphan by the sweep
    assert challenge_client.resolves == [
        ("ch-known", "expired"),
        ("ch-orphan", "expired"),
    ]
    assert miner_manager.inflight_challenges(conn) == []


async def test_sweep_is_skipped_when_the_service_cannot_list(
    validator, challenge_client
):
    """An older challenge service (no /challenges) must not break startup."""
    challenge_client.fail_listing = True
    challenge_client.lose_response(
        "ch-orphan", track="compression", age_seconds=7200.0, owner=VALIDATOR_IDENTITY
    )

    assert await validator.sweep_orphaned_challenges() == []
    assert challenge_client.resolves == []


async def test_sweep_can_be_disabled(make_validator, challenge_client):
    validator = make_validator(config={"orphan_sweep_age_seconds": 0.0})
    challenge_client.lose_response(
        "ch-orphan", track="compression", age_seconds=7200.0, owner=VALIDATOR_IDENTITY
    )

    assert await validator.sweep_orphaned_challenges() == []
    assert challenge_client.list_calls == []  # not even asked


# --- the OWNERSHIP boundary -----------------------------
#
# `/challenges` used to answer with every dispatched challenge, and the validator
# expired anything absent from its OWN in-flight table. With two validators on the
# subnet that is a live-fire hazard: B's sweep cannot see A's in-flight table, so a
# challenge A is still working on looks exactly like an orphan. Scoring many miners
# sequentially can legitimately outlast the one-hour age gate, so this is not even
# an unlikely race. Ownership is therefore the boundary — and it must be PROVEN,
# never assumed from an unfiltered list.


async def test_the_sweep_never_touches_another_validators_challenge(
    validator, challenge_client, conn
):
    """The finding's exact scenario: B must not kill A's still-live challenge.

    The service here ACCEPTS `owner` but does not filter on it (an unknown query
    parameter is silently ignored by most HTTP frameworks), so the validator's
    own row-level check is the only thing standing between A's live challenge and
    an expiry — which is precisely why the check exists.
    """
    challenge_client.filters_by_owner = False
    challenge_client.lose_response(
        "ch-theirs",
        track="compression",
        age_seconds=7200.0,
        owner=OTHER_VALIDATOR_IDENTITY,
    )
    challenge_client.lose_response(
        "ch-ours", track="compression", age_seconds=7200.0, owner=VALIDATOR_IDENTITY
    )

    expired = await validator.sweep_orphaned_challenges()

    assert expired == ["ch-ours"]
    assert challenge_client.resolves == [("ch-ours", "expired")]
    # the other validator's challenge is untouched and still dispatched
    assert "ch-theirs" in challenge_client.dispatched
    assert metric(
        validator, "vidaio_validator_orphan_sweep_skipped_total", {"reason": "foreign_owner"}
    ) == 1.0


async def test_an_unattributed_challenge_is_never_expired(
    validator, challenge_client, conn
):
    """A service that cannot attribute ownership yields NO sweepable challenges.

    Most HTTP frameworks ignore an unknown query parameter, so `owner=` on the
    request proves nothing on its own: the answer could be the whole unfiltered
    list. Only a row that NAMES us is ours.
    """
    challenge_client.filters_by_owner = False
    challenge_client.lose_response("ch-nobody", track="compression", age_seconds=7200.0)

    assert await validator.sweep_orphaned_challenges() == []
    assert challenge_client.resolves == []
    assert metric(
        validator, "vidaio_validator_orphan_sweep_skipped_total", {"reason": "unattributed"}
    ) == 1.0


async def test_the_sweep_is_disabled_without_an_identity(
    make_validator, challenge_client
):
    validator = make_validator(config={"identity": ""})
    challenge_client.lose_response(
        "ch-orphan", track="compression", age_seconds=7200.0, owner=VALIDATOR_IDENTITY
    )

    assert await validator.sweep_orphaned_challenges() == []
    assert challenge_client.list_calls == []  # never even asked
    assert metric(
        validator, "vidaio_validator_orphan_sweep_skipped_total", {"reason": "no_identity"}
    ) == 1.0


async def test_every_fetch_carries_this_validators_identity(
    validator, chain, miner_client, challenge_client
):
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    await validator.run_round()

    assert challenge_client.fetch_owners == [VALIDATOR_IDENTITY]
    assert challenge_client.dispatched.get("ch-compression") is None  # resolved away


async def test_every_resolve_carries_the_identity_it_fetched_with(
    validator, chain, miner_client, challenge_client
):
    """The other half of ownership: the service 403s an owned challenge that is
    resolved anonymously, so fetching as X and resolving as nobody would strand
    every challenge this validator ever produced."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    await validator.run_round()

    assert challenge_client.resolve_owners == [VALIDATOR_IDENTITY]
    assert challenge_client.resolves == [("ch-compression", "resolved")]


async def test_the_sweep_expires_with_the_owner_too(
    validator, challenge_client
):
    """The expiring half of the same contract — the sweep resolves as us."""
    challenge_client.lose_response(
        "ch-orphan", track="compression", age_seconds=7200.0, owner=VALIDATOR_IDENTITY
    )

    assert await validator.sweep_orphaned_challenges() == ["ch-orphan"]

    assert challenge_client.resolve_owners == [VALIDATOR_IDENTITY]
    assert challenge_client.resolves == [("ch-orphan", "expired")]


async def test_a_resolve_refused_for_ownership_retains_the_inflight_row(
    validator, chain, miner_client, challenge_client
):
    """A 403 is NOT a drained challenge.

    Only ChallengeAlreadyTerminal drops the row. Anything else — including the
    ownership refusal — must leave it so a later round (or the operator) can
    still release the asset.
    """
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    challenge_client.fail_resolve_ids.add("ch-compression")

    await validator.run_round()

    assert [row["challenge_id"] for row in miner_manager.inflight_challenges(
        validator.conn
    )] == ["ch-compression"]
    assert metric(validator, "vidaio_validator_challenge_resolve_failures_total") == 1.0


async def test_a_client_without_the_owner_contract_fetches_and_sweeps_nothing(
    make_validator, tmp_path
):
    """Feature detection at BOTH ends: fetch the old way, expire nobody."""
    legacy = LegacyChallengeClient(tmp_path)
    validator = make_validator(challenge_client=legacy)
    legacy.lose_response("ch-orphan", track="compression", age_seconds=7200.0)

    # the fetch still works (the old signature is used)
    item = await validator._next_challenge("compression")
    assert item.track == "compression"

    assert await validator.sweep_orphaned_challenges() == []
    assert legacy.resolves == []
    assert metric(
        validator,
        "vidaio_validator_orphan_sweep_skipped_total",
        {"reason": "owner_unsupported"},
    ) == 1.0


async def test_a_client_without_the_owner_contract_still_resolves_the_old_way(
    make_validator, tmp_path
):
    """Feature detection on the RESOLVE signature: a pre-ownership client is
    called positionally rather than blowing up with a TypeError mid-drain."""
    legacy = LegacyChallengeClient(tmp_path)
    validator = make_validator(challenge_client=legacy)

    item = await validator._next_challenge("compression")
    miner_manager.record_inflight_challenge(
        validator.conn,
        challenge_id=item.resolve_id,
        round_id="r-1",
        track="compression",
        fetched_at=miner_manager.utc_now_iso(),
    )
    miner_manager.set_inflight_outcome(validator.conn, item.resolve_id, "resolved")

    assert await validator._drain_inflight_challenges() == 1
    assert legacy.resolves == [(item.resolve_id, "resolved")]


# --- the OWNER travels with the OBLIGATION -----------------
#
# `/challenge/{id}/resolve` is ownership-enforced (403 not_owner), and the
# in-flight row is the validator's promise to make that call. The row used to
# record the promise but not WHO could keep it, so recovery resolved with
# whatever `validator.identity` the restarted process happened to be configured
# with. Rotate the identity between the fetch and the crash — a key rotation, a
# copy-pasted config — and recovery is 403'd forever: the row survives every
# failed resolve by design, so it retried an impossible call every round while
# the service-side asset stayed `in_use` with its commitment unrevealed.


async def test_the_inflight_row_records_the_identity_that_fetched(
    validator, chain, miner_client, conn
):
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    validator.challenge_client.fail_resolve_ids.add("ch-compression")  # keep the row

    await validator.run_round()

    (row,) = miner_manager.inflight_challenges(conn)
    assert row["owner"] == VALIDATOR_IDENTITY


async def test_recovery_after_an_identity_rotation_resolves_as_the_FETCHER(
    make_validator, challenge_client, chain, miner_client, conn, caplog
):
    """Fetch as A, crash, restart as B: recovery must still resolve as A."""
    alice = make_validator(config={"identity": VALIDATOR_IDENTITY})
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    challenge_client.fail_resolve_ids.add("ch-compression")  # the round dies owing it
    await alice.run_round()
    assert [row["challenge_id"] for row in miner_manager.inflight_challenges(conn)] == [
        "ch-compression"
    ]

    # THE RESTART: same database, same challenge service, DIFFERENT identity.
    challenge_client.fail_resolve_ids.clear()
    challenge_client.resolve_owners.clear()  # only the post-restart calls matter
    bob = make_validator(config={"identity": OTHER_VALIDATOR_IDENTITY})
    with caplog.at_level(logging.WARNING):
        drained = await bob.recover_inflight_challenges()

    assert drained == 1
    # resolved AS ALICE — the only owner the service will accept for this row
    assert challenge_client.resolve_owners == [VALIDATOR_IDENTITY]
    assert challenge_client.resolves == [("ch-compression", "resolved")]
    assert miner_manager.inflight_challenges(conn) == []  # nothing stranded
    # ... and the rotation is not silent
    assert any(
        "fetched under a DIFFERENT identity" in r.getMessage() for r in caplog.records
    )


def park_a_foreign_challenge(challenge_client, conn, challenge_id: str = "ch-foreign"):
    """The unrecoverable shape: the service attributes the challenge to somebody
    else entirely, so even the RECORDED owner is refused and an operator has to
    look at it."""
    challenge_client.lose_response(
        challenge_id, track="compression", age_seconds=1.0, owner="5SomeThirdParty"
    )
    miner_manager.record_inflight_challenge(
        conn,
        challenge_id=challenge_id,
        round_id="r-1",
        track="compression",
        fetched_at=miner_manager.utc_now_iso(),
        owner=VALIDATOR_IDENTITY,
    )


async def test_a_403_parks_the_row_and_counts_it_instead_of_looping(
    validator, challenge_client, conn, caplog
):
    """A refusal the retry cannot fix is surfaced, not retried into the void."""
    park_a_foreign_challenge(challenge_client, conn)

    with caplog.at_level(logging.WARNING):
        drained = await validator._drain_inflight_challenges()

    assert drained == 0
    # the row STAYS — dropping it would erase the only record of a stranded
    # asset — but it is PARKED out of the drain selection
    assert miner_manager.inflight_challenges(conn) == []
    (parked,) = miner_manager.parked_challenges(conn)
    assert parked["challenge_id"] == "ch-foreign"
    assert parked["parked_at"] is not None
    assert "403 not_owner" in parked["park_reason"]
    assert metric(validator, "vidaio_validator_challenge_resolve_forbidden_total") == 1.0
    assert metric(validator, "vidaio_validator_parked_challenges") == 1.0
    # and it is NOT filed as a generic flaky-resolve failure that will be retried
    assert metric(validator, "vidaio_validator_challenge_resolve_failures_total") in (
        None,
        0.0,
    )
    assert any(
        "403 not_owner" in r.getMessage() and r.levelno >= logging.WARNING
        for r in caplog.records
    )


async def test_a_parked_row_is_never_retried_by_later_rounds_or_restarts(
    validator, make_validator, challenge_client, conn, caplog
):
    """The round-4 finding itself: one 403, then SILENCE until an operator acts.

    The retained row used to be re-selected by every round's drain and every
    startup recovery, producing an impossible resolve + metric bump + WARNING
    forever. Parked means parked: no further resolve attempts, across both.
    """
    park_a_foreign_challenge(challenge_client, conn)
    assert await validator._drain_inflight_challenges() == 0
    assert len(challenge_client.resolve_owners) == 1  # the one refused attempt

    # Later rounds' drains: no new attempt, no new metric bump.
    assert await validator._drain_inflight_challenges() == 0
    assert await validator._drain_inflight_challenges() == 0
    assert len(challenge_client.resolve_owners) == 1
    assert metric(validator, "vidaio_validator_challenge_resolve_forbidden_total") == 1.0

    # THE RESTART: a fresh process over the same database honors the park too,
    # and its startup pass surfaces the parked row instead of retrying it.
    restarted = make_validator()
    with caplog.at_level(logging.WARNING):
        assert await restarted.recover_inflight_challenges() == 0
    assert len(challenge_client.resolve_owners) == 1  # still just the original
    assert metric(restarted, "vidaio_validator_parked_challenges") == 1.0
    assert any(
        "PARKED challenge obligations exist" in r.getMessage() for r in caplog.records
    )
    # the record itself is intact for the operator
    (parked,) = miner_manager.parked_challenges(conn)
    assert parked["challenge_id"] == "ch-foreign"


async def test_the_unpark_config_flag_returns_parked_rows_to_the_startup_drain(
    validator, make_validator, challenge_client, conn
):
    """The operator's way out, config half: fix ownership, restart with
    `validator.unpark_challenges = true`, and the recovery pass drains the row."""
    park_a_foreign_challenge(challenge_client, conn)
    assert await validator._drain_inflight_challenges() == 0
    assert len(miner_manager.parked_challenges(conn)) == 1

    # The operator fixed the service-side ownership state...
    challenge_client.recorded_owners["ch-foreign"] = VALIDATOR_IDENTITY
    # ...and acknowledged the retry on the next start.
    restarted = make_validator(config={"unpark_challenges": True})
    drained = await restarted.recover_inflight_challenges()

    assert drained == 1
    assert challenge_client.resolves == [("ch-foreign", "expired")]
    assert miner_manager.inflight_challenges(conn) == []
    assert miner_manager.parked_challenges(conn) == []
    assert metric(restarted, "vidaio_validator_parked_challenges") == 0.0


async def test_the_unpark_admin_method_retries_and_a_standing_refusal_reparks(
    validator, challenge_client, conn
):
    """The runtime half — and unparking is never a bypass: a refusal that still
    stands earns exactly one new attempt and is parked again."""
    park_a_foreign_challenge(challenge_client, conn)
    assert await validator._drain_inflight_challenges() == 0

    assert validator.unpark_challenges() == ["ch-foreign"]
    assert metric(validator, "vidaio_validator_parked_challenges") == 0.0
    assert [r["challenge_id"] for r in miner_manager.inflight_challenges(conn)] == [
        "ch-foreign"
    ]

    # ownership was NOT fixed: one more refused attempt, then parked again
    assert await validator._drain_inflight_challenges() == 0
    assert len(challenge_client.resolve_owners) == 2
    assert [r["challenge_id"] for r in miner_manager.parked_challenges(conn)] == [
        "ch-foreign"
    ]
    assert metric(validator, "vidaio_validator_parked_challenges") == 1.0
    assert metric(validator, "vidaio_validator_challenge_resolve_forbidden_total") == 2.0


async def test_a_row_written_before_ownership_was_recorded_still_drains(
    validator, challenge_client, conn
):
    """Migration 0004 backfills owner='' — those rows keep the old behaviour."""
    challenge_client.lose_response(
        "ch-legacy", track="compression", age_seconds=1.0, owner=VALIDATOR_IDENTITY
    )
    conn.execute(
        "INSERT INTO inflight_challenges (challenge_id, round_id, track, outcome,"
        " fetched_at) VALUES ('ch-legacy', 'r-old', 'compression', 'expired', ?)",
        (miner_manager.utc_now_iso(),),
    )

    assert await validator._drain_inflight_challenges() == 1
    # resolved as the CURRENT identity, which is what an owner-less row meant
    assert challenge_client.resolve_owners == [VALIDATOR_IDENTITY]
    assert miner_manager.inflight_challenges(conn) == []
