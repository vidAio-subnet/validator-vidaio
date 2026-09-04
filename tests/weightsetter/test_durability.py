"""review #10: a weight set can never succeed without a durable publication.

`set_weights` is a NON-IDEMPOTENT chain write behind a retry envelope. Three
failure modes are pinned here:

1. AMBIGUITY. The chain accepts the first request but its response is lost; the
   retry is tempo-rejected. The old code recorded the whole attempt as FAILED
   even though the chain weights had changed. It must be reconciled as a success
   and published — without ever writing the vector twice.
2. LOST PUBLICATION. Publication only began AFTER a successful submit, so a crash
   in between left an accepted vector permanently unaudited. Publication is now
   driven from an intent record written BEFORE the submit.
3. NO RE-DRIVE. "Pending anchors are re-drivable" — but nothing re-drove them.
   `reconcile()` runs at startup and before every attempt.

Also covers an internal review/#22 on the weight-setter side: a stale/unavailable chain
skips submission entirely, and health checks answer from their own connection.
"""

from __future__ import annotations

import logging
import threading

import pytest
from weightsetter_support import (
    ConfirmingChain,
    DelayedConfirmationChain,
    DenyingChain,
    FailingSnapshots,
    FixedSubmittedChain,
    HangingAnchorChain,
    HangingChain,
    LostResponseChain,
    OverwrittenVectorChain,
    RefreshCountingChain,
    RejectingRetryChain,
    ReportedVectorChain,
    StaleChain,
    StaleChainConfirms,
    VectorReadingChain,
)

from vidaio.audit import CommitmentStatus, sha256_hex
from vidaio.tokenomics.quantize import max_normalize_u16, quantize_u16
from vidaio.weightsetter import WeightSetter, intents

HOTKEY = "5Validator"


# --- 1. ambiguity reconciliation -----------------------------------------------


async def test_tempo_after_ambiguous_publishes_only_when_the_chain_shows_our_vector(
    make_setter, chain, ledger, conn, mk_miner, caplog
):
    """The an internal review scenario, with the round-3 evidence bar.

    The write landed, the response did not, and the retry was tempo-rejected. The
    tempo rejection alone only proves that SOMEBODY'S write occupies this window;
    the attempt becomes a success because the chain, asked afterwards, reports OUR
    vector back.
    """
    lossy = DelayedConfirmationChain(chain, readable_after=1)
    setter = make_setter([mk_miner(1)], chain_override=lossy, validator_hotkey=HOTKEY)

    with caplog.at_level(logging.WARNING, logger="weight-setter"):
        assert await setter.attempt_once() is True

    # the chain holds EXACTLY ONE vector — the retry was tempo-rejected, not a
    # second write, and we did not record a phantom failure
    assert len(chain.weight_calls) == 1
    assert lossy.calls == 2
    assert setter.metric_successes._value.get() == 1
    assert setter.metric_chain_failures._value.get() == 0

    row = intents.intents(conn)[0]
    assert row["state"] == intents.STATE_PUBLISHED
    assert row["resolution"] == "tempo_after_ambiguous"
    # ... and it IS audited: the vector was published and anchored
    assert ledger.current_status(int(row["commitment_id"])) == CommitmentStatus.ANCHORED
    assert len(chain.anchored) == 1
    assert any(
        "RECONCILED as a chain acceptance" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    )


async def test_ambiguous_then_tempo_with_an_unreadable_chain_is_NOT_published(
    make_setter, chain, ledger, conn, mk_miner, caplog
):
    """Round-3 an internal review, scenario 1: the intent that used to be published blind.

    An attempt times out before we learn anything, the retry hits the tempo gate,
    and the chain cannot be asked which vector it holds. The old code read the
    tempo rejection itself as proof that OUR write occupies the window and
    published a vector that may never have landed. It must now stay pending —
    publishable later, published never on this evidence.
    """
    lossy = LostResponseChain(chain)  # no submitted_weights: nothing readable
    setter = make_setter([mk_miner(1)], chain_override=lossy, validator_hotkey=HOTKEY)

    with caplog.at_level(logging.WARNING, logger="weight-setter"):
        assert await setter.attempt_once() is False

    assert lossy.calls == 2  # the retry happened and was tempo-rejected
    row = intents.intents(conn)[0]
    assert row["state"] == intents.STATE_PENDING  # NOT accepted, NOT abandoned
    assert row["last_check"] == "unknown"
    assert row["settled_at"] is None
    # nothing was published: no artifact commitment, no anchor, no ledger row
    assert chain.anchored == []
    assert setter.metric_publications._value.get() == 0
    assert setter.metric_successes._value.get() == 0
    assert setter.metric_abandoned._metrics == {}
    with pytest.raises(KeyError):
        ledger.get(1)
    # and it stays that way, however often we look
    for _ in range(5):
        assert await setter.reconcile() == 0
    assert intents.get_intent(conn, int(row["id"]))["state"] == intents.STATE_PENDING


async def test_a_later_intents_success_never_confirms_an_earlier_one(
    make_setter, chain, ledger, conn, mk_miner, clock
):
    """Round-3 an internal review, scenario 2.

    Intent A stays pending. Later, intent B lands — so the chain's `last_update`
    is now well past A's attempt block, which is all the old check looked at. A
    was "confirmed" and published although the chain holds B's vector, not A's.
    """
    overwritten = OverwrittenVectorChain(
        chain,
        weights={7: 1.0, 8: 0.5},
        block=500,  # B's vector, recorded later
    )
    setter = make_setter(
        [mk_miner(1)],
        chain_override=overwritten,
        validator_hotkey=HOTKEY,
        abandon_denied_intent_after_seconds=0.0,  # age would not save A either
    )
    intent_id = intents.record_intent(
        conn,
        created_at=clock().isoformat(),
        attempt_block=1,
        version_key=0,
        weights={1: 0.6, 2: 0.4},  # A's vector — nothing like what the chain holds
        packet_digests=[],
    )

    clock.advance(10 * 24 * 3600.0)
    for _ in range(3):
        assert await setter.reconcile() == 0

    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_PENDING  # neither confirmed nor buried
    assert row["last_check"] == "unknown"
    assert chain.anchored == []
    assert setter.metric_publications._value.get() == 0
    assert setter.metric_abandoned._metrics == {}
    with pytest.raises(KeyError):
        ledger.get(1)


async def test_a_rejected_retry_does_not_abandon_the_ambiguous_intent(
    make_setter, chain, ledger, conn, mk_miner, caplog
):
    """Round-3 an internal review, scenario 3: the credential rejection that buried a live
    intent.

    The first write LANDS and its response is lost; the retry comes back with a
    synchronous 401. That rejection describes the RETRY — the earlier attempt may
    be live on chain — so it may not settle anything. Once the chain can be read
    again, the vector matches and the intent is published.
    """
    rejecting = RejectingRetryChain(chain)
    setter = make_setter(
        [mk_miner(1)], chain_override=rejecting, validator_hotkey=HOTKEY
    )

    with caplog.at_level(logging.ERROR, logger="weight-setter"):
        assert await setter.attempt_once() is False

    assert rejecting.calls == 2
    row = intents.intents(conn)[0]
    assert row["state"] == intents.STATE_PENDING  # NOT abandoned on the retry's 401
    assert row["last_check"] == "unknown"
    assert setter.metric_abandoned._metrics == {}
    assert any("about the retry only" in r.message for r in caplog.records)

    # the chain becomes readable and shows OUR vector: now it may be published
    rejecting.weights_readable = True
    assert await setter.reconcile() == 1

    row = intents.get_intent(conn, int(row["id"]))
    assert row["state"] == intents.STATE_PUBLISHED
    assert row["resolution"] == "chain_confirmed_on_restart"
    assert ledger.current_status(int(row["commitment_id"])) == CommitmentStatus.ANCHORED
    assert len(chain.weight_calls) == 1  # still exactly one write ever reached it


async def test_identical_vectors_are_disambiguated_by_the_attempt_block(
    make_setter, chain, ledger, conn, mk_miner, clock
):
    """Round-3 an internal review, rule 3: two intents, one identical vector.

    The chain reports that vector at block 50. Only the attempt that sits closest
    below block 50 can claim it; the older one is not confirmed — not now, and not
    after its twin settles either.
    """
    vector = {1: 0.6, 2: 0.4}
    chain.advance_blocks(49)
    await chain.set_weights(vector, version_key=0)  # recorded at block 50
    setter = make_setter(
        [mk_miner(1)],
        chain_override=VectorReadingChain(chain),
        validator_hotkey=HOTKEY,
        abandon_denied_intent_after_seconds=0.0,
    )
    older = intents.record_intent(
        conn,
        created_at=clock().isoformat(),
        attempt_block=1,
        version_key=0,
        weights=vector,
        packet_digests=[],
    )
    newer = intents.record_intent(
        conn,
        created_at=clock().isoformat(),
        attempt_block=50,
        version_key=0,
        weights=vector,
        packet_digests=[],
    )

    assert await setter.reconcile() == 1  # exactly one of them is settleable

    assert intents.get_intent(conn, older)["state"] == intents.STATE_PENDING
    assert intents.get_intent(conn, older)["last_check"] == "unknown"
    newer_row = intents.get_intent(conn, newer)
    assert newer_row["state"] == intents.STATE_PUBLISHED
    assert newer_row["accepted_block"] == 50  # the CHAIN's block, not ours

    # the twin is settled now — that must not retroactively confirm the older one
    clock.advance(10 * 24 * 3600.0)
    for _ in range(3):
        assert await setter.reconcile() == 0
    assert intents.get_intent(conn, older)["state"] == intents.STATE_PENDING
    assert len(chain.anchored) == 1


async def test_codex_probe_a_tolerance_near_later_vector_never_confirms_the_earlier_intent(
    make_setter, chain, ledger, conn, mk_miner, clock
):
    """Round-4 an internal review, the executable probe, verbatim.

    Earlier intent {1: 65535, 2: 32767}; later intent {1: 65535, 2: 32768} — one
    u16 step apart, so `weights_match` ties the chain report to BOTH while their
    fingerprints differ. When the LATER vector lands, the old exact-fingerprint
    twin check saw no twin for the earlier intent and CONFIRMED (and published)
    a vector that never landed. With one equivalence relation everywhere, the
    chain report matching more than one candidate intent makes the earlier one
    UNKNOWN — pending forever, published never — while the later one, which
    block dating positively ties to the landing, settles normally.
    """
    earlier = {1: 65535.0, 2: 32767.0}
    later = {1: 65535.0, 2: 32768.0}
    # the raw facts of the probe: matches under tolerance, distinct fingerprints
    assert intents.weights_match(later, earlier)
    assert intents.vector_fingerprint(later) != intents.vector_fingerprint(earlier)

    chain.advance_blocks(49)
    await chain.set_weights(later, version_key=0)  # the LATER vector lands at block 50
    setter = make_setter(
        [mk_miner(1)],
        chain_override=VectorReadingChain(chain),
        validator_hotkey=HOTKEY,
        abandon_denied_intent_after_seconds=0.0,  # age must not bury the earlier one
    )
    earlier_id = intents.record_intent(
        conn,
        created_at=clock().isoformat(),
        attempt_block=1,
        version_key=0,
        weights=earlier,
        packet_digests=[],
    )
    later_id = intents.record_intent(
        conn,
        created_at=clock().isoformat(),
        attempt_block=50,
        version_key=0,
        weights=later,
        packet_digests=[],
    )

    assert await setter.reconcile() == 1  # ONLY the later intent is settleable

    early_row = intents.get_intent(conn, earlier_id)
    assert early_row["state"] == intents.STATE_PENDING  # NOT confirmed off B's landing
    assert early_row["last_check"] == "unknown"
    later_row = intents.get_intent(conn, later_id)
    assert later_row["state"] == intents.STATE_PUBLISHED
    assert later_row["accepted_block"] == 50

    # the later twin being settled must not hand the earlier one a confirmation,
    # however long we wait and however often we re-check
    clock.advance(10 * 24 * 3600.0)
    for _ in range(3):
        assert await setter.reconcile() == 0
    early_row = intents.get_intent(conn, earlier_id)
    assert early_row["state"] == intents.STATE_PENDING
    assert early_row["settled_at"] is None
    assert len(chain.anchored) == 1  # exactly ONE publication ever happened
    assert setter.metric_abandoned._metrics == {}  # ... and nothing was buried


async def test_near_identical_vectors_are_disambiguated_by_block_when_positively_decidable(
    make_setter, chain, ledger, conn, mk_miner, clock
):
    """The decidable half of an internal review: block dating CAN answer here.

    The chain recorded its (tolerance-near) vector at block 50. The other
    candidate intent did not even attempt until block 100 — it positively cannot
    be the author of a vector recorded 50 blocks before it tried — so the
    earlier intent is confirmed and published, and the later one is a positive
    denial (its own matching vector predates its attempt).
    """
    landed = {1: 65535.0, 2: 32768.0}
    ours = {1: 65535.0, 2: 32767.0}  # one u16 step away: matches under tolerance
    chain.advance_blocks(49)
    await chain.set_weights(landed, version_key=0)  # recorded at block 50
    chain.advance_blocks(100)
    setter = make_setter(
        [mk_miner(1)],
        chain_override=VectorReadingChain(chain),
        validator_hotkey=HOTKEY,
        abandon_denied_intent_after_seconds=3600.0,
    )
    author = intents.record_intent(
        conn,
        created_at=clock().isoformat(),
        attempt_block=45,
        version_key=0,
        weights=ours,
        packet_digests=[],
    )
    too_late = intents.record_intent(
        conn,
        created_at=clock().isoformat(),
        attempt_block=100,  # AFTER the set-block: cannot be the author
        version_key=0,
        weights=landed,
        packet_digests=[],
    )

    assert await setter.reconcile() == 1

    author_row = intents.get_intent(conn, author)
    assert author_row["state"] == intents.STATE_PUBLISHED  # decidable: confirmed
    assert author_row["accepted_block"] == 50
    late_row = intents.get_intent(conn, too_late)
    assert late_row["state"] == intents.STATE_PENDING
    assert late_row["last_check"] == "denied"  # predates its attempt: positive denial
    assert len(chain.anchored) == 1


async def test_an_adapter_that_cannot_read_vectors_never_confirms_or_denies(
    make_setter, chain, ledger, conn, mk_miner, clock, caplog
):
    """No vector read => UNKNOWN, forever, in both directions.

    Block bookkeeping ("our last_update advanced") is not accepted as a substitute
    — it is what published unlanded vectors — and its absence is not a denial
    either. Nothing is published and nothing is abandoned.
    """
    blind = HangingAnchorChain(chain)  # a plain proxy: no submitted_weights
    assert not hasattr(blind, "submitted_weights")
    setter = make_setter(
        [mk_miner(1)],
        chain_override=blind,
        validator_hotkey=HOTKEY,
        abandon_denied_intent_after_seconds=0.0,
    )
    landed = intents.record_intent(
        conn,
        created_at=clock().isoformat(),
        attempt_block=1,
        version_key=0,
        weights={1: 1.0},
        packet_digests=[],
    )

    clock.advance(10 * 24 * 3600.0)
    with caplog.at_level(logging.WARNING, logger="weight-setter"):
        for _ in range(5):
            assert await setter.reconcile() == 0

    row = intents.get_intent(conn, landed)
    assert row["state"] == intents.STATE_PENDING
    assert row["last_check"] == "unknown"
    assert chain.anchored == []
    assert chain.weight_calls == []  # nothing re-submitted either
    assert setter.metric_abandoned._metrics == {}
    assert setter.metric_publications._value.get() == 0
    with pytest.raises(KeyError):
        ledger.get(1)
    assert any("cannot report the weight vector" in r.message for r in caplog.records)


async def test_the_happy_path_publishes_exactly_once(
    make_setter, chain, ledger, conn, mk_miner
):
    """A directly-accepted set_weights is still the ordinary, one-shot path."""
    setter = make_setter([mk_miner(1)], validator_hotkey=HOTKEY)

    assert await setter.attempt_once() is True
    for _ in range(3):
        assert await setter.reconcile() == 0  # nothing owed, nothing re-published

    assert len(chain.weight_calls) == 1
    assert len(chain.anchored) == 1
    rows = intents.intents(conn)
    assert len(rows) == 1
    assert rows[0]["state"] == intents.STATE_PUBLISHED
    assert rows[0]["resolution"] == "chain_accepted"
    assert setter.metric_publications._value.get() == 1
    with pytest.raises(KeyError):
        ledger.get(2)  # exactly one commitment was ever minted


async def test_a_matching_vector_recorded_before_our_attempt_is_a_denial(
    make_setter, chain, conn, mk_miner, clock, caplog
):
    """Our own vector on chain, but recorded BEFORE we attempted: an earlier
    attempt's landing, not this one's — so this attempt positively did not land."""
    vector = {1: 1.0}
    await chain.set_weights(vector, version_key=0)  # recorded at block 1
    chain.advance_blocks(499)
    setter = make_setter(
        [mk_miner(1)],
        chain_override=VectorReadingChain(chain),
        validator_hotkey=HOTKEY,
        abandon_denied_intent_after_seconds=600.0,
    )
    intent_id = intents.record_intent(
        conn,
        created_at=clock().isoformat(),
        attempt_block=500,
        version_key=0,
        weights=vector,
        packet_digests=[],
    )

    assert await setter.reconcile() == 0  # denied, but too young to bury
    assert intents.get_intent(conn, intent_id)["last_check"] == "denied"

    clock.advance(601.0)
    with caplog.at_level(logging.CRITICAL, logger="weight-setter"):
        assert await setter.reconcile() == 1

    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_ABANDONED
    critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert critical and critical[0].fields["evidence"] == (
        "matching_vector_predates_this_attempt"
    )
    assert critical[0].fields["chain_set_block"] == 1


# --- the comparison itself: same vector, whatever scale the chain reports it in --


def test_a_chain_reported_vector_matches_ours_under_any_positive_rescaling():
    ours = {1: 0.6, 2: 0.3, 3: 0.1}
    u16 = {uid: value * 65535 / 0.6 for uid, value in ours.items()}  # max-normalized
    by_sum = {uid: value / 1.0 for uid, value in ours.items()}  # sum-normalized

    assert intents.weights_match(u16, ours)
    assert intents.weights_match(by_sum, ours)
    # a genuinely different vector does NOT match
    assert not intents.weights_match({1: 0.6, 2: 0.4}, ours)
    assert not intents.weights_match({1: 0.6, 2: 0.3}, ours)  # missing uid
    assert not intents.weights_match({}, ours)  # an empty report is not our vector
    # ... and the round trip through the chain's own grid is off by at most the
    # one u16 step the match tolerates — which is why the twin check uses this
    # SAME tolerance-based match against the chain report, never an exact
    # fingerprint comparison
    ours_u16 = intents.quantize_weights(ours)
    reported_u16 = intents.quantize_weights(u16)
    assert reported_u16.keys() == ours_u16.keys()
    assert max(abs(reported_u16[uid] - ours_u16[uid]) for uid in ours_u16) <= 1
    scaled = {uid: w * 3.0 for uid, w in ours.items()}
    # The pinned SDK casts each input to float32 before max-normalization. A
    # mathematically exact rescaling may therefore shift one emitted u16 step;
    # fingerprints name those exact bytes, while weights_match remains the one
    # tolerant semantic-equivalence relation.
    assert intents.quantize_weights(ours) == {1: 65535, 2: 32768, 3: 10922}
    assert intents.quantize_weights(scaled) == {1: 65535, 2: 32768, 3: 10923}
    assert intents.vector_fingerprint(ours) != intents.vector_fingerprint(scaled)
    assert intents.weights_match(scaled, ours)


async def test_a_landed_write_is_confirmed_against_a_REFRESHED_snapshot(
    make_setter, chain, ledger, conn, mk_miner, caplog
):
    """The round-2 regression, exactly: the first write landed, the response was
    lost, and the confirmation used to read the PRE-WRITE cached snapshot.

    That stale `False` said "the chain does not have our weights", so the tempo
    rejection that followed was recorded as an ordinary tempo gate and the intent
    — whose vector was LIVE ON CHAIN — was abandoned and never published. With a
    refreshed snapshot the write is CONFIRMED and the publication completes.
    """
    lossy = StaleChainConfirms(chain)
    setter = make_setter([mk_miner(1)], chain_override=lossy, validator_hotkey=HOTKEY)

    with caplog.at_level(logging.WARNING, logger="weight-setter"):
        assert await setter.attempt_once() is True

    assert len(chain.weight_calls) == 1  # exactly one write reached the chain
    row = intents.intents(conn)[0]
    assert row["state"] == intents.STATE_PUBLISHED  # NOT abandoned
    assert row["resolution"] == "chain_confirmed"
    assert ledger.current_status(int(row["commitment_id"])) == CommitmentStatus.ANCHORED
    assert len(chain.anchored) == 1
    assert setter.metric_chain_failures._value.get() == 0
    assert setter.metric_abandoned._metrics == {}


async def test_chain_confirmation_skips_the_resubmission_entirely(
    make_setter, chain, conn, mk_miner
):
    """With a readable last-weight block we never re-issue the write at all."""
    confirming = ConfirmingChain(chain)
    setter = make_setter(
        [mk_miner(1)], chain_override=confirming, validator_hotkey=HOTKEY
    )

    assert await setter.attempt_once() is True

    assert confirming.calls == 1  # the retry was answered by the CHAIN, not by a write
    assert len(chain.weight_calls) == 1
    row = intents.intents(conn)[0]
    assert row["resolution"] == "chain_confirmed"
    assert row["state"] == intents.STATE_PUBLISHED
    assert (
        setter.metric_reconciled.labels(resolution="chain_confirmed")._value.get() == 1
    )


async def test_a_plain_tempo_gate_is_not_reconciled_into_a_success(
    make_setter, chain, conn, mk_miner
):
    """No ambiguity, no reconciliation — an ordinary tempo gate is a reschedule."""
    await chain.set_weights({1: 1.0}, version_key=0)  # occupy the gate
    setter = make_setter([mk_miner(1)])

    assert await setter.attempt_once() is False

    assert len(chain.weight_calls) == 1  # only the priming call
    assert chain.anchored == []
    row = intents.intents(conn)[0]
    assert row["state"] == intents.STATE_ABANDONED
    assert row["resolution"] == "tempo_gated"
    assert setter.metric_reconciled._metrics == {}


async def test_exhausted_ambiguous_attempts_stay_pending_never_abandoned(
    make_setter, chain, conn, mk_miner
):
    """Round-2 an internal review: an UNKNOWN outcome must not be settled as a failure.

    Every attempt timed out and the chain cannot say whether any of them landed.
    Abandoning is terminal — the vector would never be published — so a vector
    that MIGHT be live on chain stays pending and gets re-checked instead.
    """
    setter = make_setter(
        [mk_miner(1)], chain_override=HangingChain(chain, hang_first=10)
    )

    assert await setter.attempt_once() is False

    assert chain.weight_calls == []
    row = intents.intents(conn)[0]
    assert row["state"] == intents.STATE_PENDING
    assert row["settled_at"] is None
    assert row["last_check"] == "unknown"  # and WHY it is still pending is recorded
    assert setter.metric_chain_failures._value.get() == 1
    assert setter.metric_unresolved_intents._value.get() == 1
    assert setter.metric_abandoned._metrics == {}
    # The gauge is synchronized at the durable mutation boundary. It must not
    # lie at zero until the next 72-minute reconciliation cadence.
    assert setter.metric_pending_intents._value.get() == 1


async def test_an_unknown_intent_survives_many_reconciles_untouched(
    make_setter, chain, conn, mk_miner
):
    """Unknown-forever: pending forever, never abandoned, never re-submitted."""
    setter = make_setter(
        [mk_miner(1)], chain_override=HangingChain(chain, hang_first=10)
    )
    await setter.attempt_once()
    intent_id = int(intents.intents(conn)[0]["id"])

    for _ in range(25):
        assert await setter.reconcile() == 0

    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_PENDING
    assert chain.weight_calls == []  # nothing was ever re-submitted either
    assert setter.metric_abandoned._metrics == {}
    assert setter.metric_pending_intents._value.get() == 1


# --- 2. the intent record is written BEFORE the chain write --------------------


async def test_intent_records_the_exact_vector_before_submitting(
    make_setter, chain, conn, mk_miner
):
    setter = make_setter([mk_miner(1, score=0.9), mk_miner(2, score=0.3)])

    assert await setter.attempt_once() is True

    row = intents.intents(conn)[0]
    _block, submitted = chain.weight_calls[0]
    # After acceptance the durable row is reconciled to the EXACT submitted u16
    # (round-4 #3) — what actually landed on the chain's grid — not the
    # pre-quantization float intent recorded before the write. Publication reads
    # THIS row, so it must carry chain state byte-for-byte.
    expected = max_normalize_u16(quantize_u16(submitted))
    assert intents.load_vector(row) == {uid: float(q) for uid, q in expected.items()}
    assert row["vector_digest"] == sha256_hex(row["vector_json"].encode("utf-8"))
    assert row["attempt_block"] == 1
    assert row["accepted_block"] == chain.weight_calls[0][0]


# --- 3. the re-drive loop ------------------------------------------------------


async def test_crash_after_acceptance_is_published_on_restart(
    make_setter, chain, ledger, store, conn, mk_miner, raw_config
):
    """A process that died between acceptance and publication leaves an intent."""
    setter = make_setter([mk_miner(1)])
    # what the crashed process had done: intent recorded, chain accepted, nothing
    # published yet
    intent_id = intents.record_intent(
        conn,
        created_at="2026-08-20T12:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 0.6, 2: 0.4},
        packet_digests=[],
    )
    intents.mark_accepted(
        conn, intent_id, accepted_block=1, resolution="chain_accepted"
    )

    settled = await setter.reconcile()

    assert settled == 1
    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_PUBLISHED
    assert ledger.current_status(int(row["commitment_id"])) == CommitmentStatus.ANCHORED
    assert len(chain.anchored) == 1
    assert setter.metric_redriven._value.get() == 1


async def test_pending_anchor_is_re_driven_on_the_same_commitment(
    make_setter, chain, ledger, conn, mk_miner
):
    """An anchor failure must be retried, not duplicated into a second commitment."""
    hanging = HangingAnchorChain(chain)
    setter = make_setter([mk_miner(1)], chain_override=hanging)

    assert await setter.attempt_once() is True  # weights set, anchor fails

    row = intents.intents(conn)[0]
    commitment_id = int(row["commitment_id"])
    assert row["state"] == intents.STATE_ACCEPTED  # publication still owed
    assert ledger.current_status(commitment_id) == CommitmentStatus.PENDING_CHAIN
    assert chain.anchored == []
    assert setter.metric_publications._value.get() == 0

    hanging.anchor_ok = True
    assert await setter.reconcile() == 1

    row = intents.get_intent(conn, int(row["id"]))
    assert row["state"] == intents.STATE_PUBLISHED
    assert int(row["commitment_id"]) == commitment_id  # the SAME commitment
    assert ledger.current_status(commitment_id) == CommitmentStatus.ANCHORED
    assert len(chain.anchored) == 1
    with pytest.raises(KeyError):
        ledger.current_status(commitment_id + 1)  # no duplicate was minted


async def test_startup_reconciliation_never_publishes_before_the_first_attempt(
    make_setter, chain, ledger, conn, mk_miner
):
    setter = make_setter([mk_miner(1)], attempt_interval_seconds=3600.0)
    intent_id = intents.record_intent(
        conn,
        created_at="2026-08-20T12:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 1.0},
        packet_digests=[],
    )
    intents.mark_accepted(
        conn, intent_id, accepted_block=1, resolution="chain_accepted"
    )

    setter.request_stop()
    await setter.run()  # the read-only recovery pass runs even though the loop exits

    # Storage/anchor publication stays behind the scheduled weight attempt. A
    # wedged evidence backend therefore cannot delay the first emission write;
    # the accepted row remains durable for the post-attempt publication drain.
    assert intents.get_intent(conn, intent_id)["state"] == intents.STATE_ACCEPTED
    assert chain.anchored == []


async def test_unconfirmable_pending_intent_stays_pending_and_is_not_resubmitted(
    make_setter, chain, conn, mk_miner, caplog
):
    """A crash before the chain outcome was known: neither re-issued nor buried.

    Round-2 an internal review: this used to be abandoned on `unconfirmable_after_crash`,
    which is a POSITIVE claim ("this vector is not on chain") made from the
    absence of information. If the write had landed, the audit trail was lost
    forever. Now it just stays pending.
    """
    setter = make_setter([mk_miner(1)])
    intent_id = intents.record_intent(
        conn,
        created_at="2026-08-20T12:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 1.0},
        packet_digests=[],
    )

    with caplog.at_level(logging.WARNING, logger="weight-setter"):
        assert await setter.reconcile() == 0  # settled NOTHING

    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_PENDING
    assert row["last_check"] == "unknown"
    assert chain.weight_calls == []  # nothing was blindly re-submitted
    assert any("still UNKNOWN" in r.message for r in caplog.records)
    assert setter.metric_abandoned._metrics == {}


async def test_a_denied_intent_is_abandoned_only_when_old_enough(
    make_setter, chain, conn, mk_miner, clock, caplog
):
    """The ONLY path to `abandoned`: a positive denial plus a bounded age."""
    denying = DenyingChain(chain)
    setter = make_setter(
        [mk_miner(1)],
        chain_override=denying,
        validator_hotkey=HOTKEY,
        abandon_denied_intent_after_seconds=600.0,
    )
    intent_id = intents.record_intent(
        conn,
        created_at=clock().isoformat(),
        attempt_block=1,
        version_key=0,
        weights={1: 1.0},
        packet_digests=[],
    )

    # denied, but too young to bury: the chain may still be catching up
    assert await setter.reconcile() == 0
    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_PENDING
    assert row["last_check"] == "denied"

    clock.advance(601.0)
    with caplog.at_level(logging.CRITICAL, logger="weight-setter"):
        assert await setter.reconcile() == 1

    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_ABANDONED
    assert row["resolution"] == "chain_denied_after_crash"
    assert chain.weight_calls == []  # abandoning is not re-submitting
    assert (
        setter.metric_abandoned.labels(
            resolution="chain_denied_after_crash"
        )._value.get()
        == 1
    )
    # CRITICAL, with the evidence that justified a terminal decision
    critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert critical and "ABANDONING a weight intent" in critical[0].message
    assert critical[0].fields["chain_check"] == "denied"
    assert critical[0].fields["attempt_block"] == 1
    assert critical[0].fields["vector_digest"]


async def test_a_stale_snapshot_is_never_read_as_a_denial(
    make_setter, chain, conn, mk_miner, clock
):
    """The exact false negative behind the finding: stale must not mean absent."""
    stale = StaleChain(chain, fresh=False)
    setter = make_setter(
        [mk_miner(1)],
        chain_override=stale,
        validator_hotkey=HOTKEY,
        abandon_denied_intent_after_seconds=0.0,  # age would not save it
    )
    intent_id = intents.record_intent(
        conn,
        created_at=clock().isoformat(),
        attempt_block=1,
        version_key=0,
        weights={1: 1.0},
        packet_digests=[],
    )

    clock.advance(10 * 24 * 3600.0)  # however long we wait
    for _ in range(5):
        assert await setter.reconcile() == 0

    assert intents.get_intent(conn, intent_id)["state"] == intents.STATE_PENDING


async def test_a_confirmation_reads_a_refreshed_snapshot_not_the_cached_one(
    make_setter, chain, conn, mk_miner
):
    """The mechanical half of the fix: refresh() before answering."""
    counting = RefreshCountingChain(chain)
    setter = make_setter(
        [mk_miner(1)], chain_override=counting, validator_hotkey=HOTKEY
    )
    intents.record_intent(
        conn,
        created_at="2026-08-20T12:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 1.0},
        packet_digests=[],
    )

    before = counting.refreshes
    await setter.reconcile()

    assert counting.refreshes > before  # the post-write world was actually read


async def test_pending_intent_is_deferred_while_the_chain_is_unusable(
    make_setter, chain, conn, mk_miner
):
    """A stale read must not become a verdict — the intent waits for a real one."""
    stale = StaleChain(chain, fresh=False)
    setter = make_setter([mk_miner(1)], chain_override=stale, validator_hotkey=HOTKEY)
    intent_id = intents.record_intent(
        conn,
        created_at="2026-08-20T12:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 1.0},
        packet_digests=[],
    )

    assert await setter.reconcile() == 0

    assert intents.get_intent(conn, intent_id)["state"] == intents.STATE_PENDING
    assert chain.weight_calls == []


async def test_confirmed_pending_intent_is_published_on_restart(
    make_setter, chain, ledger, conn, mk_miner
):
    """The chain says our weights landed — complete the audit trail, don't lose it."""
    await chain.set_weights({1: 1.0}, version_key=0)  # the crashed process's write
    confirming = ConfirmingChain(chain)
    setter = make_setter(
        [mk_miner(1)], chain_override=confirming, validator_hotkey=HOTKEY
    )
    intent_id = intents.record_intent(
        conn,
        created_at="2026-08-20T12:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 1.0},
        packet_digests=[],
    )

    assert await setter.reconcile() == 1

    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_PUBLISHED
    assert row["resolution"] == "chain_confirmed_on_restart"
    assert ledger.current_status(int(row["commitment_id"])) == CommitmentStatus.ANCHORED
    assert len(chain.weight_calls) == 1  # still exactly one write


#


def test_accept_with_vector_is_atomic_when_the_rewrite_crashes(conn, monkeypatch):
    """The DIRECT path never leaves an ACCEPTED intent carrying its FLOAT vector.

    The connection is autocommit, so a separate mark_accepted + reconcile_vector was
    TWO commits: a crash between them left an ACCEPTED intent still carrying its
    pre-quantization float, which startup reconciliation then anchored verbatim — a
    vector the chain never held. accept_with_vector wraps both in one
    BEGIN IMMEDIATE ... COMMIT, so if the rewrite raises, the acceptance rolls back
    with it. The intent is therefore EITHER fully reconciled OR still pending — never
    accepted-with-float.
    """
    intent_id = intents.record_intent(
        conn,
        created_at="2026-08-20T12:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 0.6, 2: 0.4},
        packet_digests=[],
    )

    def _boom(*_a, **_k):
        raise RuntimeError("crash between acceptance and vector rewrite")

    monkeypatch.setattr(intents, "reconcile_vector", _boom)

    with pytest.raises(RuntimeError):
        intents.accept_with_vector(
            conn,
            intent_id,
            accepted_block=7,
            resolution="chain_accepted",
            weights={1: 39321, 2: 26214},  # the u16 that would have been written
        )

    row = intents.get_intent(conn, intent_id)
    # NEITHER write survived: not ACCEPTED, and the stored vector is still the float
    assert row["state"] == intents.STATE_PENDING
    assert row["accepted_block"] is None
    assert intents.load_vector(row) == {1: 0.6, 2: 0.4}


def test_accept_with_vector_accepts_and_rewrites_together(conn):
    """The happy path: one commit sets state AND the submitted vector."""
    intent_id = intents.record_intent(
        conn,
        created_at="2026-08-20T12:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 0.6, 2: 0.4},
        packet_digests=[],
    )

    intents.accept_with_vector(
        conn,
        intent_id,
        accepted_block=7,
        resolution="chain_accepted",
        weights={1: 39321, 2: 26214},
    )

    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_ACCEPTED
    assert row["accepted_block"] == 7
    assert row["resolution"] == "chain_accepted"
    assert intents.load_vector(row) == {1: 65535.0, 2: 43690.0}
    assert row["vector_digest"] == sha256_hex(row["vector_json"].encode("utf-8"))


def test_accept_with_vector_without_a_reported_vector_keeps_the_stored_vector(conn):
    """No reported vector (block-bookkeeping recovery) -> plain accept, no rewrite."""
    intent_id = intents.record_intent(
        conn,
        created_at="2026-08-20T12:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 1.0},
        packet_digests=[],
    )

    intents.accept_with_vector(
        conn,
        intent_id,
        accepted_block=3,
        resolution="chain_accepted",
        weights=None,
    )

    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_ACCEPTED
    assert row["accepted_block"] == 3
    assert intents.load_vector(row) == {1: 1.0}  # untouched


async def test_recovery_confirmed_intent_stores_the_exact_reported_vector(
    make_setter, chain, ledger, conn, mk_miner
):
    """A recovery-confirmed pending intent publishes the chain's EXACT u16.

    Confirmation is scale-invariant (weights_match), so the stored FLOAT need only
    be scale-equivalent to what the chain holds. Before this fix the ORIGINAL float
    was published verbatim — a vector the chain never held. The recovery evidence now
    carries the exact reported vector, and reconcile() rewrites the stored row to it,
    atomically with the state change, BEFORE _publish_intent reads it.
    """
    reported = {1: 39321, 2: 26214}  # the chain's u16, scale-equivalent to the float
    reader = ReportedVectorChain(chain, weights=reported, block=5)
    setter = make_setter([mk_miner(1)], chain_override=reader, validator_hotkey=HOTKEY)
    intent_id = intents.record_intent(
        conn,
        created_at="2026-08-20T12:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 0.6, 2: 0.4},  # the same vector, DIFFERENT bytes than `reported`
        packet_digests=[],
    )
    # the two are one vector under the match, but are not byte-equal
    assert intents.weights_match(reported, {1: 0.6, 2: 0.4})
    assert {1: 0.6, 2: 0.4} != {uid: float(w) for uid, w in reported.items()}

    assert await setter.reconcile() == 1

    row = intents.get_intent(conn, intent_id)
    assert row["state"] == intents.STATE_PUBLISHED
    assert row["resolution"] == "chain_confirmed_on_restart"
    # The durable row carries the canonical runtime max-grid, not the float intent
    # or a sum-grid phrasing returned by a non-live adapter.
    assert intents.load_vector(row) == {1: 65535.0, 2: 43690.0}
    assert row["vector_digest"] == sha256_hex(row["vector_json"].encode("utf-8"))
    assert row["accepted_block"] == 5  # the chain's own block, carried through
    assert ledger.current_status(int(row["commitment_id"])) == CommitmentStatus.ANCHORED


#


async def test_direct_and_recovery_paths_persist_byte_identical_canonical_bytes(
    make_setter, chain, ledger, conn, mk_miner
):
    """Round-6 an internal review: one accepted write, two paths, byte-identical durable bytes.

    The direct-success path persists `SetWeightsResult.submitted` (service.py ~1130)
    and the recovery path persists the `submitted_weights()` readback (service.py
    ~1811). `SubmittedWeights.weights` explicitly permits raw u16, sum-normalized, or
    untouched-float phrasings, and the two surfaces disagree: here the direct submit
    reports the raw u16 grid while the recovery readback reports the SAME vector as
    scale-equivalent floats. Before this fix each path stored its representation
    verbatim, so the SAME accepted vector produced DIFFERENT anchored bytes depending
    on whether recovery ran — breaking "anchored == chain state, deterministically".
    Both now canonicalize through `max_normalize_u16` at the reconcile choke point,
    so the stored vector_json/vector_digest are byte-identical regardless of path.
    """
    # RECOVERY first — its intent is alone in the DB, so there is no identical-twin
    # ambiguity. The readback reports the accepted vector as scale-equivalent FLOATS.
    recovery_chain = ReportedVectorChain(chain, weights={1: 0.6, 2: 0.4}, block=5)
    recovery_setter = make_setter(
        [mk_miner(1)], chain_override=recovery_chain, validator_hotkey=HOTKEY
    )
    recovery_id = intents.record_intent(
        conn,
        created_at="2026-08-20T12:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 0.6, 2: 0.4},
        packet_digests=[],
    )
    assert await recovery_setter.reconcile() == 1
    recovery_row = intents.get_intent(conn, recovery_id)

    # DIRECT — set_weights reports the SAME accepted vector as the raw u16 GRID. The
    # direct-success accept path is not twin-gated, so the already-published recovery
    # intent (same vector) does not interfere.
    direct_chain = FixedSubmittedChain(chain, submitted={1: 39321, 2: 26214})
    direct_setter = make_setter(
        [mk_miner(1), mk_miner(2)], chain_override=direct_chain, validator_hotkey=HOTKEY
    )
    assert await direct_setter.attempt_once() is True
    direct_row = intents.get_intent(conn, int(intents.intents(conn)[-1]["id"]))

    # the two paths saw the vector in DIFFERENT, non-byte-equal representations ...
    assert dict(direct_chain.submitted) != dict(recovery_chain.reported)
    # ... yet persisted BYTE-IDENTICAL canonical vector_json + digest.
    assert direct_row["vector_json"] == recovery_row["vector_json"]
    assert direct_row["vector_digest"] == recovery_row["vector_digest"]
    assert direct_row["vector_digest"] == sha256_hex(
        direct_row["vector_json"].encode("utf-8")
    )
    assert intents.load_vector(direct_row) == {1: 65535.0, 2: 43690.0}


def test_reconciled_vector_is_always_the_max_u16_grid_regardless_of_representation(
    conn,
):
    """The canonical stored vector is the runtime max-grid for ANY phrasing.

    ``max_normalize_u16`` is proportional and idempotent, so every
    permitted representation of one vector — untouched floats, raw u16, any positive
    rescaling — reconciles to the IDENTICAL grid vector and the IDENTICAL durable
    bytes.
    """
    representations = [
        {1: 0.6, 2: 0.4},  # untouched submission floats
        {1: 39321, 2: 26214},  # raw u16 the chain stores
        {1: 60.0, 2: 40.0},  # an arbitrary positive rescaling
        {1: 6, 2: 4},  # small integers, same ratio
    ]
    seen: set[tuple[str, str]] = set()
    for rep in representations:
        intent_id = intents.record_intent(
            conn,
            created_at="2026-08-20T12:00:00+00:00",
            attempt_block=1,
            version_key=0,
            weights={1: 0.6, 2: 0.4},
            packet_digests=[],
        )
        intents.accept_with_vector(
            conn,
            intent_id,
            accepted_block=7,
            resolution="chain_accepted",
            weights=rep,
        )
        row = intents.get_intent(conn, intent_id)
        vector = intents.load_vector(row)
        assert max(vector.values()) == 65535  # on the runtime max-grid
        assert vector == {1: 65535.0, 2: 43690.0}
        assert row["vector_digest"] == sha256_hex(row["vector_json"].encode("utf-8"))
        seen.add((row["vector_json"], row["vector_digest"]))
    # every representation persisted the SAME canonical bytes
    assert len(seen) == 1


#


async def test_stale_chain_snapshot_skips_submission(
    make_setter, chain, conn, mk_miner
):
    stale = StaleChain(chain, fresh=False)
    setter = make_setter([mk_miner(1)], chain_override=stale)

    assert await setter.attempt_once() is False

    assert chain.weight_calls == []  # NOT an empty/partial vector — nothing at all
    assert chain.anchored == []
    assert intents.intents(conn) == []
    assert stale.freshness_calls
    assert (
        setter.metric_chain_state_skips.labels(
            reason="chain_snapshot_stale"
        )._value.get()
        == 1
    )


async def test_unavailable_chain_state_skips_submission(make_setter, chain, conn):
    setter = make_setter([], snapshots_override=FailingSnapshots())

    assert await setter.attempt_once() is False

    assert chain.weight_calls == []
    assert intents.intents(conn) == []
    assert (
        setter.metric_chain_state_skips.labels(
            reason="snapshots_unavailable"
        )._value.get()
        == 1
    )


async def test_failed_refresh_is_not_recorded_as_fresh(make_setter, chain, mk_miner):
    class BrokenRefresh(StaleChain):
        def refresh(self) -> None:
            raise OSError("chainsim unreachable")

    broken = BrokenRefresh(chain)
    broken.has_fresh_snapshot = None  # adapter without the new surface yet
    setter = make_setter([mk_miner(1)], chain_override=broken)

    assert await setter.attempt_once() is False

    assert setter._last_refresh_at is None
    assert chain.weight_calls == []
    assert (
        setter.metric_chain_state_skips.labels(
            reason="chain_snapshot_never_refreshed"
        )._value.get()
        == 1
    )


#


async def test_health_check_from_another_thread_reports_a_healthy_db(
    make_setter, mk_miner
):
    setter = make_setter([mk_miner(1)])
    results: list[tuple[bool, dict]] = []
    thread = threading.Thread(
        target=lambda: results.append(setter.health.health_payload())
    )
    thread.start()
    thread.join(timeout=5.0)

    assert results, "health check thread did not finish"
    ok, payload = results[0]
    assert ok is True, payload
    assert payload["checks"]["db"] is True


def test_in_memory_db_registers_no_lying_health_check(raw_config, chain):
    from vidaio.core.db import connect
    from vidaio.weightsetter import migrate

    class NoMiners:
        def miner_snapshots(self):
            return []

    conn = connect(":memory:")
    migrate(conn)
    setter = WeightSetter(
        {
            **raw_config,
            "weightsetter": {
                **raw_config["weightsetter"],
                "publication_enabled": False,
            },
        },
        chain=chain,
        snapshots=NoMiners(),
        conn=conn,
    )
    assert setter._conn_factory is None
    assert "db" not in setter.health.health_payload()[1]["checks"]
