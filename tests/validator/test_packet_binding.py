"""review #6: a score packet is accepted only when it is BOUND to the request.

Verifying the packet's self-reported sha256 proves the bytes and the digest agree
— nothing more. A compromised or MITM'd scoring endpoint can return a genuine,
internally-consistent, HIGH-scoring packet that belongs to a different miner,
item, challenge, track or scorer, and the old code accumulated it for the current
uid. Every binding field is checked here, and the rejection is non-punitive: the
miner is skipped for the round (validator-infra trouble), never zeroed.
"""

from __future__ import annotations

import pytest

from vidaio.validator import ScorePacketEvidence, miner_manager

from validator_support import FakeScoringClient, mk_neuron

#: the validator PINS this scorer, so scorer_version is a binding field too
PINNED = FakeScoringClient.EFFECTIVE_SCORER_VERSION

#: field -> a value that is valid on its own but belongs to a DIFFERENT request
TAMPERED_FIELDS = {
    "challenge_id": "ch-some-other-challenge",
    "item_id": "ch-compression:999",
    "miner_hotkey": "hk-a-completely-different-miner",
    "track": "upscaling",
    "content_digest": "f" * 64,
    "scorer_version": "vidaio-scorer/999-not-ours",
}


def metric(validator, name: str, labels: dict[str, str] | None = None) -> float | None:
    return validator.health.registry.get_sample_value(name, labels or {})


@pytest.fixture
def validator(make_validator):
    """A validator that PINS its scorer — the strictest binding configuration."""
    return make_validator(config={"scorer_version": PINNED})


@pytest.mark.parametrize("field", sorted(TAMPERED_FIELDS))
async def test_unbound_packet_is_rejected_and_never_accumulated(
    validator, chain, miner_client, scoring_client, conn, field
):
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    scoring_client.scores = {"hk1": 0.99}  # a high score that must NOT land
    scoring_client.spoof = {"hk1": {field: TAMPERED_FIELDS[field]}}

    report = await validator.run_round()

    assert report.rejected_packets == {1: field}
    assert report.scored == {}
    assert report.scoring_failed == [1]
    # non-punitive: skipped, not zeroed — validator-side trouble never punishes miners
    assert 1 not in report.zeroed
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == 0.0
    assert (
        metric(validator, "vidaio_validator_score_packet_rejected_total", {"field": field})
        == 1.0
    )
    # and no evidence was persisted for a packet we refused
    assert ScorePacketEvidence(conn).recent_packet_digests() == []


async def test_rejection_logs_at_warning_with_the_mismatched_field(
    validator, chain, miner_client, scoring_client, caplog
):
    import logging

    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    scoring_client.spoof = {"hk1": {"miner_hotkey": "hk-someone-else"}}

    with caplog.at_level(logging.WARNING, logger="inference-validator"):
        await validator.run_round()

    hits = [
        r
        for r in caplog.records
        if r.name == "inference-validator"
        and getattr(r, "fields", {}).get("violation") == "SCORE_PACKET_NOT_BOUND"
    ]
    assert hits and hits[0].levelno == logging.WARNING
    assert hits[0].fields["field"] == "miner_hotkey"
    assert hits[0].fields["got"] == "hk-someone-else"
    assert hits[0].fields["expected"] == "hk1"


async def test_replayed_high_score_packet_from_another_miner_is_rejected(
    validator, chain, miner_client, scoring_client, conn
):
    """The exact case: uid 2's excellent packet replayed for uid 1."""
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "compression"}
    scoring_client.scores = {"hk1": 0.1, "hk2": 0.95}
    # the endpoint hands uid 1 a packet whose identity is entirely uid 2's
    scoring_client.spoof = {
        "hk1": {"miner_hotkey": "hk2", "item_id": "ch-compression:2"}
    }

    report = await validator.run_round()

    assert 1 not in report.scored  # the replay bought uid 1 nothing
    assert report.scored[2] == 0.95  # uid 2's own genuine packet still counts
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == 0.0
    digests = ScorePacketEvidence(conn).recent_packet_digests()
    assert len(digests) == 1  # only the honest packet became evidence


async def test_a_correctly_bound_packet_is_still_accepted(
    validator, chain, miner_client, scoring_client, conn
):
    """The guard must not reject honest packets."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    scoring_client.scores = {"hk1": 0.77}

    report = await validator.run_round()

    assert report.scored == {1: 0.77}
    assert report.rejected_packets == {}
    request = scoring_client.requests[0]
    row = ScorePacketEvidence(conn).packets()[0]
    assert row["challenge_id"] == request.challenge_id
    assert row["item_id"] == request.item_id
    assert row["miner_hotkey"] == "hk1"
    assert row["content_digest"] == request.output_digest
    assert row["scorer_version"] == validator.config.scorer_version


async def test_pinned_scorer_version_is_enforced_on_the_packet_itself(
    validator, chain, miner_client, scoring_client, conn
):
    """A compromised worker cannot slip a differently-scored packet past a pin.

    The scoring worker's own half of an internal review (stamping its OWN version and
    answering 409 on a mismatched assertion) is that service's fix; this is the
    validator's INDEPENDENT check, which holds even if the worker is untrusted
    and answers 200 with a foreign scorer's packet.
    """
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    scoring_client.spoof = {"hk1": {"scorer_version": "vidaio-scorer/0-legacy"}}

    report = await validator.run_round()

    assert report.rejected_packets == {1: "scorer_version"}
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == 0.0


# --- the CIRCULAR binding ----------------------------------
#
# item_id was the strongest-looking binding field and the weakest one: the
# validator copied `MinerTaskResponse.task_id` — a MINER-CONTROLLED value — into
# ScoreRequest.item_id and then "verified" the packet against it. The check could
# not fail, because both sides came from the miner. The dispatched id is now the
# only id that exists.


async def test_a_miner_echoing_a_different_task_id_is_rejected(
    validator, chain, miner_client, scoring_client, conn
):
    """The swap case: uid 1 answers under uid 2's task id."""
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "compression"}
    miner_client.swap_task_ids = {1: "ch-compression:2"}

    report = await validator.run_round()

    # The exact signed dispatch plus validator-signed availability observation
    # turns this miner-attributable protocol failure into an explicit zero fold.
    assert report.zeroed == {1: "availability:task_id_mismatch"}
    assert report.non_punitive_skips == {}
    assert 1 not in report.scored
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == 0.0
    assert metric(validator, "vidaio_validator_task_id_mismatch_total") == 1.0
    # the swapped response never reached the scoring worker at all
    assert [r.miner_hotkey for r in scoring_client.requests] == ["hk2"]
    # ... and uid 2's own honest round is unaffected
    assert report.scored[2] == 0.8


async def test_a_swapped_task_id_never_reaches_the_score_request(
    validator, chain, miner_client, scoring_client
):
    """Even a plausible id (a challenge we really did fetch) is refused."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    miner_client.swap_task_ids = {1: "ch-compression:1-replayed"}

    report = await validator.run_round()

    assert report.zeroed == {1: "availability:task_id_mismatch"}
    assert report.non_punitive_skips == {}
    assert scoring_client.requests == []


async def test_the_scored_item_id_is_the_validators_own_id(
    validator, chain, miner_client, scoring_client, conn
):
    """Not the miner's echo of it — the binding must not be circular."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    item = challenge_item_id = "ch-compression:1"

    report = await validator.run_round()

    assert report.scored == {1: 0.8}
    assert scoring_client.requests[0].item_id == challenge_item_id
    assert ScorePacketEvidence(conn).packets()[0]["item_id"] == item
    # and it is derived from the DISPATCH, reproducible without any miner input
    assert (
        validator.task_id_for(
            await validator.challenge_client.next_challenge("compression"), 1
        )
        == challenge_item_id
    )


async def test_mismatch_logs_the_dispatched_and_returned_ids(
    validator, chain, miner_client, caplog
):
    import logging

    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    miner_client.swap_task_ids = {1: "ch-compression:2"}

    with caplog.at_level(logging.WARNING, logger="inference-validator"):
        await validator.run_round()

    hits = [
        r
        for r in caplog.records
        if getattr(r, "fields", {}).get("violation") == "MINER_TASK_ID_MISMATCH"
    ]
    assert hits and hits[0].levelno == logging.WARNING
    assert hits[0].fields["dispatched_task_id"] == "ch-compression:1"
    assert hits[0].fields["returned_task_id"] == "ch-compression:2"


async def test_unpinned_validator_records_the_workers_own_version_as_evidence(
    make_validator, chain, miner_client, scoring_client, conn
):
    """Empty pin = accept whichever scorer answers, but PERSIST which one did."""
    validator = make_validator()  # scorer_version: "" (the default)
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    report = await validator.run_round()

    assert report.rejected_packets == {}
    assert report.scored == {1: 0.8}
    assert scoring_client.requests[0].scorer_version is None  # asserted nothing
    row = ScorePacketEvidence(conn).packets()[0]
    assert row["scorer_version"] == PINNED  # what actually ran is on record
