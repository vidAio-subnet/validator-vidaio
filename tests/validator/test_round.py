"""Full fake-round e2e for InferenceValidator: scoring, skips, dedup, failures."""

from __future__ import annotations

from datetime import datetime, timezone

from vidaio.tokenomics import accumulate
from vidaio.validator import miner_manager

from validator_support import mk_neuron

NOW = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)


def metric(validator, name: str, labels: dict[str, str] | None = None) -> float | None:
    return validator.health.registry.get_sample_value(name, labels or {})


async def test_full_round_scores_and_accumulates(
    validator, chain, miner_client, scoring_client, challenge_client, conn
):
    chain.set_neurons(
        [
            mk_neuron(1),
            mk_neuron(2),
            mk_neuron(3),
            mk_neuron(9, is_validator=True, axon_port=8300),
        ]
    )
    miner_client.tracks = {
        1: "compression",
        2: "compression",
        3: "upscaling",
        9: "compression",
    }
    scoring_client.scores = {"hk1": 0.8, "hk2": 0.4, "hk3": 0.6, "hk9": 0.7}

    report = await validator.run_round()

    decay = validator.tokenomics.ewma_decay
    assert report.scored == {1: 0.8, 2: 0.4, 3: 0.6, 9: 0.7}
    assert report.zeroed == {}
    assert report.skipped_unknown_track == []
    for uid, score in report.scored.items():
        row = miner_manager.get_miner(conn, uid)
        assert row["accumulate_score"] == accumulate(0.0, score, decay)
    # Validator permit is a capability, not an exclusive role. This serving
    # permit-holder is still probed, dispatched, scored, and accumulated.
    assert 9 in {uid for uid, _ in miner_client.task_calls}
    # one challenge per live track; every score request pinned the right artifacts
    assert sorted(challenge_client.fetches) == ["compression", "upscaling"]
    for request in scoring_client.requests:
        item = challenge_client.item_for(request.track)
        assert request.reference_digest == item.reference_digest
        assert request.miner_input_digest == item.miner_input_digest
        # unpinned validator: the assertion is OMITTED so the worker's own stamped
        # version is accepted (services.protocol scorer_version contract)
        assert validator.config.scorer_version == ""
        assert request.scorer_version is None
    assert metric(validator, "vidaio_validator_rounds_total") == 1.0
    assert (
        metric(validator, "vidaio_validator_scored_total", {"track": "compression"})
        == 3.0
    )
    assert (
        metric(validator, "vidaio_validator_scored_total", {"track": "upscaling"})
        == 1.0
    )

    # second round folds into the EWMA exactly like tokenomics.accumulate
    await validator.run_round()
    row = miner_manager.get_miner(conn, 1)
    assert row["accumulate_score"] == accumulate(
        accumulate(0.0, 0.8, decay), 0.8, decay
    )

    # the snapshot the weight-setter would consume reflects the accumulators
    snaps = {s.uid: s for s in miner_manager.snapshot(conn, chain.neurons(), NOW)}
    assert snaps[1].track == "compression"
    assert snaps[3].track == "upscaling"
    assert snaps[1].accumulate_score == row["accumulate_score"]


async def test_unknown_track_miner_skipped_never_defaulted(
    validator, chain, miner_client, scoring_client, conn
):
    """The fix of the old validator.py:844 bug: probe timeout -> SKIP, not upscaling."""
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression"}
    miner_client.warrant_fail_uids = {2}

    report = await validator.run_round()

    assert report.skipped_unknown_track == [2]
    assert 2 not in report.scored and 2 not in report.zeroed
    assert metric(validator, "vidaio_validator_skipped_unknown_track_total") == 1.0
    # never dispatched, never scored — under NO track
    assert 2 not in {uid for uid, _ in miner_client.task_calls}
    assert all(r.miner_hotkey != "hk2" for r in scoring_client.requests)
    upscaling_scored = metric(
        validator, "vidaio_validator_scored_total", {"track": "upscaling"}
    )
    assert upscaling_scored in (None, 0.0)
    # accumulator untouched (skip, not zero) and track still NULL in the DB
    row = miner_manager.get_miner(conn, 2)
    assert row["accumulate_score"] == 0.0
    assert row["track"] is None
    # and the miner is absent from the tokenomics snapshot entirely
    assert 2 not in {s.uid for s in miner_manager.snapshot(conn, chain.neurons(), NOW)}


async def test_garbage_warrant_answer_stays_unknown(validator, chain, miner_client):
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "upscalling-v2-turbo"}  # garbage, not a known track

    report = await validator.run_round()

    assert report.skipped_unknown_track == [1]
    assert metric(validator, "vidaio_validator_skipped_unknown_track_total") == 1.0


async def test_duplicate_response_zeroed(validator, chain, miner_client, conn):
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "compression"}
    miner_client.outputs = {1: b"same-bytes", 2: b"same-bytes"}

    report = await validator.run_round()

    assert report.scored == {1: 0.8}
    assert report.zeroed == {2: "duplicate"}
    decay = validator.tokenomics.ewma_decay
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == accumulate(
        0.0, 0.8, decay
    )
    assert miner_manager.get_miner(conn, 2)["accumulate_score"] == accumulate(
        0.0, 0.0, decay
    )


async def test_duplicate_winner_uses_anchor_salted_signed_identity(
    validator, chain, miner_client, scoring_client
):
    # For the fixture anchor block hash (ab*32), miner-z hashes before miner-a.
    # It wins despite its higher uid, later dispatch position, and
    # lexicographically larger raw hotkey.
    chain.set_neurons(
        [
            mk_neuron(1, hotkey="miner-a"),
            mk_neuron(2, hotkey="miner-z"),
        ]
    )
    miner_client.tracks = {1: "compression", 2: "compression"}
    miner_client.outputs = {1: b"same-bytes", 2: b"same-bytes"}

    report = await validator.run_round()

    assert report.scored == {2: 0.8}
    assert report.zeroed == {1: "duplicate"}
    assert [request.miner_hotkey for request in scoring_client.requests] == ["miner-z"]


async def test_distinct_encodings_are_never_economic_duplicates(
    validator, chain, miner_client, scoring_client
):
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "compression"}
    miner_client.outputs = {1: b"first-encoding", 2: b"second-encoding"}

    report = await validator.run_round()

    assert report.scored == {1: 0.8, 2: 0.8}
    assert report.zeroed == {}
    assert [request.miner_hotkey for request in scoring_client.requests] == [
        "hk1",
        "hk2",
    ]


async def test_miner_timeout_records_availability_zero(
    validator, chain, miner_client, conn
):
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "compression"}
    miner_client.task_timeout_uids = {2}

    report = await validator.run_round()

    assert report.scored == {1: 0.8}
    assert report.zeroed == {2: "availability:timeout"}
    assert report.non_punitive_skips == {}
    assert metric(validator, "vidaio_validator_miner_timeouts_total") == 1.0
    assert miner_manager.get_miner(conn, 2)["accumulate_score"] == 0.0
    assert metric(validator, "vidaio_validator_rounds_total") == 1.0


async def test_digest_mismatch_records_availability_zero(
    validator, chain, miner_client, scoring_client
):
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "compression"}
    miner_client.bad_digest_uids = {2}

    report = await validator.run_round()

    assert report.scored == {1: 0.8}
    assert report.zeroed == {2: "availability:output_digest_mismatch"}
    assert report.non_punitive_skips == {}
    assert metric(validator, "vidaio_validator_digest_mismatch_total") == 1.0
    # the tampered response never reached the scoring worker
    assert all(r.miner_hotkey != "hk2" for r in scoring_client.requests)


async def test_scoring_failure_skips_accumulation(
    validator, chain, miner_client, scoring_client, conn
):
    """Validator-infra trouble must not punish the miner with a zero."""
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "compression"}
    scoring_client.fail_hotkeys = {"hk2"}

    report = await validator.run_round()

    assert report.scored == {1: 0.8}
    assert report.scoring_failed == [2]
    assert 2 not in report.zeroed
    assert miner_manager.get_miner(conn, 2)["accumulate_score"] == 0.0  # untouched


async def test_tampered_score_packet_discarded(
    validator, chain, miner_client, scoring_client, conn
):
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    scoring_client.corrupt_digest_hotkeys = {"hk1"}

    report = await validator.run_round()

    assert report.scored == {}
    assert report.scoring_failed == [1]
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == 0.0


async def test_challenge_fetch_failure_skips_track(
    validator, chain, miner_client, challenge_client
):
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "upscaling"}
    challenge_client.fail_tracks = {"compression"}

    report = await validator.run_round()

    # the upscaling track still ran; the compression track was skipped, not crashed
    assert report.scored == {2: 0.8}
    assert 1 not in report.scored and 1 not in report.zeroed
    assert metric(validator, "vidaio_validator_rounds_total") == 1.0


async def test_ip_and_coldkey_dedup_first_uid_wins(validator, chain, miner_client):
    chain.set_neurons(
        [
            mk_neuron(1),
            mk_neuron(2, ip="10.0.0.1"),  # same IP as uid 1
            mk_neuron(3, coldkey="ck1"),  # same coldkey as uid 1
            mk_neuron(4),
        ]
    )
    miner_client.tracks = {1: "compression", 4: "compression"}

    await validator.run_round()

    dispatched = {uid for uid, _ in miner_client.task_calls}
    assert dispatched == {1, 4}
    assert set(miner_client.warrant_calls) == {1, 4}


async def test_unspecified_ips_do_not_collapse_non_serving_miners(
    validator, chain, miner_client
) -> None:
    chain.set_neurons(
        [
            mk_neuron(1, ip="0.0.0.0", coldkey="ck1"),
            mk_neuron(2, ip="0.0.0.0", coldkey="ck2"),
            mk_neuron(3, ip="::", coldkey="ck3"),
        ]
    )
    miner_client.tracks = {uid: "compression" for uid in (1, 2, 3)}

    await validator.run_round()

    assert {uid for uid, _ in miner_client.task_calls} == {1, 2, 3}


async def test_no_axon_control_cannot_shadow_serving_miner_dedup(
    validator, chain, miner_client
) -> None:
    """A lower-uid control identity stays census-only and consumes no dedup slot."""
    chain.set_neurons(
        [
            mk_neuron(
                1,
                is_validator=True,
                coldkey="shared-operator",
                ip="10.0.0.1",
                axon_port=None,
            ),
            mk_neuron(
                2,
                coldkey="shared-operator",
                ip="10.0.0.1",
                axon_port=8300,
            ),
        ]
    )
    miner_client.tracks = {2: "compression"}

    report = await validator.run_round()

    assert report.scored == {2: 0.8}
    assert set(miner_client.warrant_calls) == {2}
    assert {uid for uid, _ in miner_client.task_calls} == {2}


async def test_min_stake_floor_filters_miners(validator, chain, miner_client):
    validator.config = validator.config.model_copy(update={"min_stake": 5.0})
    chain.set_neurons([mk_neuron(1, alpha_stake=10.0), mk_neuron(2, alpha_stake=1.0)])
    miner_client.tracks = {1: "compression", 2: "compression"}

    await validator.run_round()

    assert {uid for uid, _ in miner_client.task_calls} == {1}


async def test_metagraph_refresh_throttled(
    raw_config, chain, challenge_client, miner_client, scoring_client, conn
):
    import random as _random

    from vidaio.validator import InferenceValidator

    calls = []
    chain.refresh = lambda: calls.append(1)  # type: ignore[method-assign]
    clock_value = [0.0]
    raw_config["validator"]["metagraph_refresh_seconds"] = 1800.0
    v = InferenceValidator(
        raw_config,
        chain=chain,
        challenge_client=challenge_client,
        miner_client=miner_client,
        scoring_client=scoring_client,
        conn=conn,
        rng=_random.Random(85),
        clock=lambda: clock_value[0],
    )
    await v.run_round()
    await v.run_round()
    assert len(calls) == 1  # second round inside the throttle window
    clock_value[0] = 1800.0
    await v.run_round()
    assert len(calls) == 2


async def test_health_checks(validator, chain):
    chain.set_neurons([])
    await validator.run_round()
    ok, payload = validator.health.health_payload()
    assert ok, payload
    assert payload["checks"] == {
        "db": True,
        "last_round_age": True,
        # an unresolved scorer-pin conflict is a not-scoring condition, so it is
        # part of the health surface
        "scorer_pin": True,
    }
