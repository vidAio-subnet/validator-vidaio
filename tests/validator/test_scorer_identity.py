"""The validator's half of THE SCORER-IDENTITY CONTRACT (services.protocol).

The scoring worker mints ONE identity — `<name>+<identity digest[:12]>`, a
digest over every configured lever that can change a measured score — publishes
it on GET /healthz, and stamps it into every packet. The validator adopts it in
one of two ways:

  * PIN ON FIRST CONTACT (`validator.scorer_version` empty, the default):
    discover the identity from the worker, hold it for the process's life, omit
    the field from ScoreRequests, and bind every packet's scorer_version to it.
  * EXPLICIT OPERATOR PIN (`validator.scorer_version` set): assert it on the
    wire so the worker itself 409s a stranger, AND require the discovered
    identity to equal it — a disagreement is a CONFIG ERROR that fails loudly at
    startup instead of quietly scoring against a scorer nobody chose.

What must never happen: a validator accumulating EWMA from two different
scorers, or one that "recovers" from a mismatch by adopting whatever answered.
"""

from __future__ import annotations

import asyncio

import pytest

from vidaio.services.protocol import ScorerIdentityMismatch, ScorerRuntimeMismatch
from vidaio.validator import ScorePacketEvidence, miner_manager

from validator_support import FakeScoringClient, mk_neuron

PINNED = FakeScoringClient.EFFECTIVE_SCORER_VERSION
OTHER = "vidaio-scorer/1+ffffffffffff"


def metric(validator, name: str, labels: dict[str, str] | None = None) -> float | None:
    return validator.health.registry.get_sample_value(name, labels or {})


# --- pin on first contact ------------------------------------------------------


async def test_unpinned_validator_discovers_and_pins_the_workers_identity(
    validator, scoring_client
):
    assert validator.pinned_scorer_version is None  # nothing known before contact

    assert await validator.discover_scorer_identity() == PINNED

    assert validator.pinned_scorer_version == PINNED
    assert scoring_client.identity_calls == 1


async def test_the_pin_binds_every_packet_without_being_asserted_on_the_wire(
    validator, chain, miner_client, scoring_client, conn
):
    """Discovery pins; the request stays unasserted; the packet is still gated."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    report = await validator.run_round()

    assert validator.pinned_scorer_version == PINNED
    # Nothing asserted on the wire: echoing the identity back at the worker that
    # just published it would prove nothing (services.protocol).
    assert scoring_client.requests[0].scorer_version is None
    assert report.scored == {1: 0.8}
    assert ScorePacketEvidence(conn).packets()[0]["scorer_version"] == PINNED


async def test_a_packet_from_another_scorer_is_rejected_once_pinned(
    validator, chain, miner_client, scoring_client, conn
):
    """The pin is the whole point: a foreign packet must not be accumulated."""
    await validator.discover_scorer_identity()
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    # A compromised/MITM'd endpoint answering with a genuine packet from a
    # DIFFERENT scorer (self-consistent digest and all).
    scoring_client.spoof = {"hk1": {"scorer_version": OTHER}}

    report = await validator.run_round()

    assert report.rejected_packets == {1: "scorer_version"}
    assert report.scored == {}
    # Non-punitive: validator-side infra trouble never zeroes a miner.
    assert report.scoring_failed == [1]


async def test_a_worker_that_changes_identity_under_us_is_refused(validator, scoring_client):
    """One round's EWMA must never be split across two scorers."""
    await validator.discover_scorer_identity()
    scoring_client.effective_scorer_version = OTHER

    with pytest.raises(ScorerIdentityMismatch):
        await validator.discover_scorer_identity()

    assert validator.pinned_scorer_version == PINNED  # the pin does not move


# --- worker not up yet ---------------------------------------------------------


async def test_an_unreachable_worker_pins_nothing_and_is_retried(
    validator, scoring_client
):
    """A scoring worker that has not started must not make the validator unstartable."""
    scoring_client.identity_unavailable = True

    assert await validator.discover_scorer_identity() is None
    assert validator.pinned_scorer_version is None

    scoring_client.identity_unavailable = False
    assert await validator.discover_scorer_identity() == PINNED


async def test_a_runtime_contract_mismatch_latches_a_structured_scoring_refusal(
    validator, scoring_client
):
    async def _mismatched_runtime() -> str:
        raise ScorerRuntimeMismatch("remote payout backend map moved")

    scoring_client.scorer_identity = _mismatched_runtime
    with pytest.raises(ScorerRuntimeMismatch):
        await validator.discover_scorer_identity()

    assert "payout runtime" in (validator.scorer_pin_conflict or "")
    report = await validator.run_round()
    assert report.skipped_reason == "scorer_pin_conflict"


async def test_a_round_picks_up_the_identity_once_the_worker_appears(
    validator, chain, miner_client, scoring_client, conn
):
    """Reconnect: the pin is taken on the first round that reaches the worker."""
    scoring_client.identity_unavailable = True
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    report = await validator.run_round()
    assert validator.pinned_scorer_version is None  # nothing to pin yet
    # ... and nothing was scored UNBOUND while it was unknown (new-2)
    assert report.skipped_reason == "scorer_identity_unknown"
    assert report.scored == {}

    scoring_client.identity_unavailable = False
    report = await validator.run_round()
    assert validator.pinned_scorer_version == PINNED
    assert report.skipped_reason is None
    assert report.scored == {1: 0.8}


# --- the pin is REQUIRED, not optional -------------------


async def test_a_round_without_a_pin_is_skipped_not_scored_unbound(
    validator, chain, miner_client, scoring_client, conn
):
    """A blocked /healthz with a working /score must not produce unbound scores.

    Scoring here would accept ANY scorer's packet (there is nothing to compare
    scorer_version against) and fold it into the same EWMA a later, correctly
    pinned round contributes to.
    """
    scoring_client.identity_unavailable = True
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    report = await validator.run_round()

    assert report.skipped_reason == "scorer_identity_unknown"
    assert report.scored == {} and report.zeroed == {}
    assert scoring_client.requests == []  # /score was never called at all
    assert miner_manager.get_miner(conn, 1) is None  # no round, no registry write
    assert metric(
        validator,
        "vidaio_validator_rounds_skipped_total",
        {"reason": "scorer_identity_unknown"},
    ) == 1.0
    assert metric(validator, "vidaio_validator_scorer_pinned") == 0.0


async def test_an_operator_pin_alone_is_a_sufficient_binding(
    make_validator, chain, miner_client, scoring_client
):
    """With an explicit pin the binding exists even if /healthz never answers."""
    scoring_client.identity_unavailable = True
    validator = make_validator(config={"scorer_version": PINNED})
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    report = await validator.run_round()

    assert report.skipped_reason is None
    assert report.scored == {1: 0.8}
    assert scoring_client.requests[0].scorer_version == PINNED


# --- the pin SURVIVES a restart -------------------------


async def test_the_pin_is_persisted_and_reloaded_by_the_next_process(
    validator, make_validator, chain, miner_client, scoring_client, conn
):
    """A restart must not re-pin from scratch: the accumulators have a scorer."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    await validator.run_round()
    assert miner_manager.load_scorer_pin(conn)["scorer_version"] == PINNED

    # "restart": a brand new process over the SAME database, worker unreachable
    scoring_client.identity_unavailable = True
    restarted = make_validator()

    assert restarted.pinned_scorer_version == PINNED  # loaded from disk, not the wire
    assert scoring_client.identity_calls == 1  # the pin needed no round trip
    report = await restarted.run_round()
    assert report.scored == {1: 0.8}  # and it can keep scoring, bound as before


async def test_a_restart_onto_a_different_worker_refuses_to_score(
    validator, make_validator, chain, miner_client, scoring_client, conn, caplog
):
    """Two scorers must never merge into one accumulator — the whole finding."""
    import logging

    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    await validator.run_round()
    before = miner_manager.get_miner(conn, 1)["accumulate_score"]

    # the operator re-points the validator at a DIFFERENT scoring worker
    scoring_client.effective_scorer_version = OTHER
    restarted = make_validator()

    with caplog.at_level(logging.CRITICAL, logger="inference-validator"):
        report = await restarted.run_round()

    assert report.skipped_reason == "scorer_pin_conflict"
    assert report.scored == {}
    assert restarted.scorer_pin_conflict is not None
    # the accumulator built under the first scorer is untouched, not extended
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == before
    assert miner_manager.load_scorer_pin(conn)["scorer_version"] == PINNED
    assert any(
        "SCORER IDENTITY CHANGED" in r.message and r.levelno == logging.CRITICAL
        for r in caplog.records
    )
    # the conflict is a health condition: an operator has to look
    assert restarted.health.health_payload()[1]["checks"]["scorer_pin"] is False
    assert metric(restarted, "vidaio_validator_scorer_pinned") == 0.0


async def test_the_conflict_latches_until_an_operator_acknowledges_it(
    validator, make_validator, chain, miner_client, scoring_client, conn, caplog
):
    """Recovery is an OPERATOR action; a restart alone must not adopt the stranger."""
    import logging

    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    await validator.run_round()

    scoring_client.effective_scorer_version = OTHER
    conflicted = make_validator()
    for _ in range(3):  # it does not "settle down" on its own
        assert (await conflicted.run_round()).skipped_reason == "scorer_pin_conflict"

    # the acknowledgement: drop the pin deliberately
    with caplog.at_level(logging.CRITICAL, logger="inference-validator"):
        acknowledged = make_validator(config={"reset_scorer_pin": True})

    assert miner_manager.load_scorer_pin(conn) is None
    assert any("OPERATOR RESET" in r.message for r in caplog.records)

    report = await acknowledged.run_round()
    assert report.scored == {1: 0.8}
    assert acknowledged.pinned_scorer_version == OTHER
    assert miner_manager.load_scorer_pin(conn)["scorer_version"] == OTHER


async def test_startup_stays_up_but_refusing_when_the_worker_disagrees_with_the_pin(
    validator, make_validator, chain, miner_client, scoring_client, conn
):
    """A live deployment with a wrong scorer must stay OBSERVABLE, not crash-loop.

    The operator-pin config error still stops the process (see above) — that one
    is a typo and the deployment never started. This one is a running validator
    whose scoring half moved, and its health surface has to survive to say so.
    """
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    await validator.run_round()

    scoring_client.effective_scorer_version = OTHER
    restarted = make_validator()
    rounds: list[str | None] = []

    async def one_round():
        rounds.append((await type(restarted).run_round(restarted)).skipped_reason)
        restarted.request_stop()

    restarted.run_round = one_round
    await asyncio.wait_for(restarted.run(), timeout=5.0)  # does NOT raise

    assert rounds == ["scorer_pin_conflict"]
    assert restarted.health.health_payload()[1]["checks"]["scorer_pin"] is False


async def test_an_operator_pin_disagreeing_with_the_persisted_one_refuses_to_score(
    validator, make_validator, chain, miner_client, conn, caplog
):
    """Config says one scorer, the accumulators were built under another."""
    import logging

    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    await validator.run_round()

    with caplog.at_level(logging.CRITICAL, logger="inference-validator"):
        conflicted = make_validator(config={"scorer_version": OTHER})

    assert conflicted.scorer_pin_conflict is not None
    assert any("SCORER PIN CONFLICT" in r.message for r in caplog.records)
    assert (await conflicted.run_round()).skipped_reason == "scorer_pin_conflict"


# --- explicit operator pin -----------------------------------------------------


async def test_operator_pin_is_asserted_on_the_wire_and_verified_on_startup(
    make_validator, chain, miner_client, scoring_client
):
    validator = make_validator(config={"scorer_version": PINNED})

    assert await validator.discover_scorer_identity() == PINNED

    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    await validator.run_round()
    # Asserted, so the WORKER refuses a stranger too (409), not just us.
    assert scoring_client.requests[0].scorer_version == PINNED


async def test_operator_pin_that_no_live_worker_matches_fails_loudly(
    make_validator, scoring_client
):
    """A config error, not something to recover from by adopting the stranger."""
    validator = make_validator(config={"scorer_version": OTHER})

    with pytest.raises(ScorerIdentityMismatch) as excinfo:
        await validator.discover_scorer_identity()

    assert excinfo.value.expected == OTHER
    assert excinfo.value.discovered == PINNED
    assert validator.pinned_scorer_version is None


async def test_startup_refuses_to_run_against_the_wrong_scorer(
    make_validator, scoring_client
):
    """`run()` discovers BEFORE the first round, so nothing is ever scored."""
    validator = make_validator(config={"scorer_version": OTHER})

    with pytest.raises(ScorerIdentityMismatch):
        await asyncio.wait_for(validator.run(), timeout=5.0)

    assert scoring_client.requests == []


# --- clients without the discovery surface -------------------------------------


async def test_a_client_without_discovery_is_feature_detected(make_validator):
    """An injected double with no scorer_identity() leaves the validator unpinned."""

    class NoDiscovery:
        async def score(self, request):  # pragma: no cover - never called here
            raise AssertionError("not used")

    validator = make_validator(scoring_client=NoDiscovery())

    assert await validator.discover_scorer_identity() is None
    assert validator.pinned_scorer_version is None
