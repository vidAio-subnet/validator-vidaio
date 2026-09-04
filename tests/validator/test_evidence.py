"""review #7 (score evidence is durable) and #9 (a round is all-or-nothing).

#7: only numeric EWMAs used to survive a round, so a published weight vector
could not be reproduced by a third party and every real inference publication
anchored the "no score packets" sentinel. The exact packet bytes are now pinned
per (round, uid, item) and archived as SCORE_PACKET artifacts, and
`ScorePacketEvidence` serves their digests to the weight-setter.

#9: EWMAs used to be written one row at a time under autocommit while the
weight-setter read the same database, so a crash or a concurrent read could see a
mixed-round vector with no way to detect or repair it. One round is now ONE
BEGIN IMMEDIATE transaction stamped into a round ledger.
"""

from __future__ import annotations

import json

import pytest

from vidaio.audit import ArtifactKind, ArtifactRef, sha256_hex
from vidaio.audit.store import backend_key
from vidaio.scoring import ItemScore
from vidaio.tokenomics import accumulate
from vidaio.validator import AvailabilityFoldEvidence, ScorePacketEvidence, miner_manager

from validator_support import mk_neuron

NOW_ISO = "2026-08-20T12:00:00+00:00"
DECAY = 0.75


# --- #7 evidence ---------------------------------------------------------------


async def test_round_persists_packet_bytes_digest_and_bindings(
    validator, chain, miner_client, scoring_client, conn, store
):
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "compression"}
    scoring_client.scores = {"hk1": 0.8, "hk2": 0.4}

    report = await validator.run_round()

    rows = ScorePacketEvidence(conn).packets()
    assert {r["uid"] for r in rows} == {1, 2}
    for row in rows:
        # the digest is the sha256 of the persisted BYTES, recomputable by anyone
        assert row["packet_digest"] == sha256_hex(row["packet_json"].encode("utf-8"))
        packet = ItemScore.from_json(row["packet_json"])
        assert packet.miner_hotkey == row["miner_hotkey"]
        assert packet.score == pytest.approx(report.scored[row["uid"]])
        assert row["round_id"] == report.round_id
        # ... and the same bytes are in the audit store as a SCORE_PACKET artifact
        ref = ArtifactRef(
            digest=row["packet_digest"],
            kind=ArtifactKind.SCORE_PACKET,
            byte_size=len(row["packet_json"].encode("utf-8")),
            backend_key=row["audit_ref"],
        )
        assert row["audit_ref"] == backend_key(
            ArtifactKind.SCORE_PACKET, row["packet_digest"]
        )
        assert json.loads(store.get(ref)) == json.loads(row["packet_json"])


async def test_timeout_creates_request_bound_availability_fold(
    validator, chain, miner_client, conn, store
):
    """A signed anchored dispatch makes a timeout an explicit zero fold."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    miner_client.task_timeout_uids = {1}

    report = await validator.run_round()

    assert report.zeroed == {1: "availability:timeout"}
    assert report.non_punitive_skips == {}
    assert ScorePacketEvidence(conn).packets() == []
    observations = AvailabilityFoldEvidence(conn).observations()
    assert len(observations) == 1
    assert observations[0]["reason"] == "timeout"
    assert observations[0]["score"] == 0.0
    assert sha256_hex(observations[0]["observation_json"].encode()) == observations[0][
        "observation_digest"
    ]
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == 0.0


async def test_positive_then_timeout_decays_earning_state(
    validator, chain, miner_client, conn
):
    """A miner cannot freeze a good EWMA by withholding later responses."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    first = await validator.run_round()
    first_value = miner_manager.get_miner(conn, 1)["accumulate_score"]

    miner_client.task_timeout_uids = {1}
    second = await validator.run_round()

    assert first.scored == {1: 0.8}
    assert second.zeroed == {1: "availability:timeout"}
    assert second.non_punitive_skips == {}
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == pytest.approx(
        DECAY * first_value
    )
    rows = ScorePacketEvidence(conn).packets()
    assert [float(row["score"]) for row in rows] == [0.8]
    assert all(row["packet_digest"] for row in rows)


@pytest.mark.parametrize(
    ("failure_set", "reason"),
    (
        ("task_cold_start_uids", "restart_fence_exhausted"),
        ("task_unreachable_endpoint_uids", "unreachable_endpoint"),
    ),
)
async def test_selective_restart_or_undialable_endpoint_cannot_freeze_ewma(
    validator,
    chain,
    miner_client,
    conn,
    failure_set: str,
    reason: str,
):
    """Every miner-attributable refusal decays a previously winning EWMA."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    first = await validator.run_round()
    first_value = miner_manager.get_miner(conn, 1)["accumulate_score"]

    getattr(miner_client, failure_set).add(1)
    second = await validator.run_round()

    assert first.scored == {1: 0.8}
    assert second.zeroed == {1: f"availability:{reason}"}
    assert second.non_punitive_skips == {}
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == pytest.approx(
        DECAY * first_value
    )
    observations = AvailabilityFoldEvidence(conn).observations()
    assert [row["reason"] for row in observations] == [reason]


async def test_recent_packet_digests_is_the_real_merkle_set(
    validator, chain, miner_client, conn
):
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "compression"}

    await validator.run_round()

    evidence = ScorePacketEvidence(conn)
    digests = evidence.recent_packet_digests()
    assert len(digests) == 2
    assert digests == sorted(digests)  # deterministic and reproducible from SQL
    # the PublicationInputs surface the weight-setter consumes returns the same set
    assert list(evidence.score_packet_digests()) == digests


async def test_evidence_without_an_audit_store_is_db_only_and_warned(
    make_validator, chain, miner_client, conn, caplog
):
    import logging

    with caplog.at_level(logging.WARNING, logger="inference-validator"):
        validator = make_validator(store=None)
    assert any("no audit store configured" in r.message for r in caplog.records)

    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    await validator.run_round()

    rows = ScorePacketEvidence(conn).packets()
    assert len(rows) == 1
    assert rows[0]["audit_ref"] is None  # DB-only, explicitly
    assert rows[0]["packet_json"]  # the bytes are still preserved


async def test_audit_store_failure_fails_the_item_closed(
    make_validator, chain, miner_client, conn
):
    """Round-2 an internal review: a CONFIGURED store that fails must not be fail-open.

    Committing the score with audit_ref NULL published a weight nobody could
    reproduce from the audit store — the exact claim the store exists to back.
    The item is dropped instead (non-punitively: the miner is not zeroed).
    """

    class BrokenStore:
        def put(self, data, kind):
            raise OSError("audit backend unreachable")

    validator = make_validator(store=BrokenStore())
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "compression"}

    report = await validator.run_round()

    assert report.scored == {}
    assert report.audit_store_failed == [1, 2]
    assert report.scoring_failed == [1, 2]
    assert report.zeroed == {}  # validator-side trouble never punishes a miner
    # nothing was accumulated and NO evidence row claims an archive that is absent
    assert ScorePacketEvidence(conn).packets() == []
    for uid in (1, 2):
        assert miner_manager.get_miner(conn, uid)["accumulate_score"] == 0.0
    assert (
        validator.health.registry.get_sample_value(
            "vidaio_validator_audit_store_failures_total"
        )
        == 2.0
    )


async def test_timeout_fold_is_durable_while_audit_store_is_down(
    make_validator, chain, miner_client, conn
):
    class BrokenStore:
        def put(self, data, kind):
            raise OSError("audit backend unreachable")

    validator = make_validator(store=BrokenStore())
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    miner_client.task_timeout_uids = {1}

    report = await validator.run_round()

    assert report.zeroed == {1: "availability:timeout"}
    assert report.non_punitive_skips == {}
    assert report.scoring_failed == []
    assert report.audit_store_failed == []
    assert ScorePacketEvidence(conn).packets() == []
    assert len(AvailabilityFoldEvidence(conn).observations()) == 1
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == 0.0


async def test_db_only_mode_is_allowed_but_visibly_flagged(
    make_validator, chain, miner_client, conn
):
    """No store configured is a legitimate mode — it just must never be silent."""
    validator = make_validator(store=None)
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    report = await validator.run_round()

    assert report.scored == {1: 0.8}
    assert report.audit_store_failed == []
    assert ScorePacketEvidence(conn).packets()[0]["audit_ref"] is None
    assert (
        validator.health.registry.get_sample_value(
            "vidaio_validator_audit_store_configured"
        )
        == 0.0
    )


# --- #9 atomicity --------------------------------------------------------------


def test_commit_round_is_all_or_nothing(conn):
    """A failure part-way through leaves NO partial EWMA and no ledger stamp."""
    miner_manager.sync_neurons(conn, [mk_neuron(1), mk_neuron(2)], block=1)
    miner_manager.begin_round(conn, "r1", 1, NOW_ISO)
    broken_packet = {  # missing 'score' -> KeyError after the EWMAs are written
        "uid": 1,
        "item_id": "i1",
        "challenge_id": "c1",
        "track": "compression",
        "miner_hotkey": "hk1",
        "content_digest": "a" * 64,
        "packet_digest": "b" * 64,
        "packet_json": "{}",
        "scorer_version": "v1",
    }

    with pytest.raises(KeyError):
        miner_manager.commit_round(
            conn,
            "r1",
            scores={1: 0.8, 2: 0.4},
            decay=DECAY,
            packets=[broken_packet],
            committed_at=NOW_ISO,
        )

    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == 0.0
    assert miner_manager.get_miner(conn, 2)["accumulate_score"] == 0.0
    assert conn.execute("SELECT COUNT(*) c FROM score_packets").fetchone()["c"] == 0
    # the partial round is DETECTABLE afterwards
    assert [r["round_id"] for r in miner_manager.uncommitted_rounds(conn)] == ["r1"]


def test_commit_round_refuses_an_unopened_or_recommitted_round(conn):
    miner_manager.sync_neurons(conn, [mk_neuron(1)], block=1)
    with pytest.raises(miner_manager.RoundLedgerError):
        miner_manager.commit_round(
            conn, "never-begun", scores={1: 0.8}, decay=DECAY, committed_at=NOW_ISO
        )
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == 0.0

    miner_manager.begin_round(conn, "r1", 1, NOW_ISO)
    miner_manager.commit_round(
        conn, "r1", scores={1: 0.8}, decay=DECAY, committed_at=NOW_ISO
    )
    with pytest.raises(miner_manager.RoundLedgerError):
        miner_manager.commit_round(
            conn, "r1", scores={1: 0.4}, decay=DECAY, committed_at=NOW_ISO
        )
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] == accumulate(
        0.0, 0.8, DECAY
    )


def test_packets_until_excludes_future_packet_and_round(conn):
    early = "2026-08-20T12:00:00+00:00"
    late = "2026-08-20T13:00:00+00:00"
    for round_id, created_at, uid in (("r1", early, 1), ("r2", late, 2)):
        miner_manager.begin_round(conn, round_id, uid, created_at)
        miner_manager.commit_round(
            conn,
            round_id,
            scores={},
            decay=DECAY,
            committed_at=created_at,
            packets=(
                {
                    "uid": uid,
                    "item_id": f"i{uid}",
                    "challenge_id": f"c{uid}",
                    "track": "compression",
                    "miner_hotkey": f"hk{uid}",
                    "content_digest": "a" * 64,
                    "packet_digest": str(uid) * 64,
                    "packet_json": "{}",
                    "scorer_version": "v1",
                    "score": 0.5,
                },
            ),
        )

    rows = ScorePacketEvidence(conn).packets(until="2026-08-20T12:30:00+00:00")
    assert [row["round_id"] for row in rows] == ["r1"]


def test_packets_through_block_and_open_round_probe(conn):
    miner_manager.begin_round(conn, "old", 10, NOW_ISO)
    miner_manager.commit_round(
        conn, "old", scores={}, decay=DECAY, committed_at=NOW_ISO
    )
    miner_manager.begin_round(conn, "future", 20, NOW_ISO)

    evidence = ScorePacketEvidence(conn)
    assert evidence.packets(through_block=10) == []
    assert not evidence.has_uncommitted_round_through(10)
    assert evidence.has_uncommitted_round_through(20)


async def test_crash_mid_round_leaves_no_partial_ewma_visible(
    validator, chain, miner_client, challenge_client, conn
):
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "upscaling"}
    original = validator._run_track
    seen: list[str] = []

    async def flaky(track, neurons, report, *, round_id, evidence, availability):
        result = await original(
            track,
            neurons,
            report,
            round_id=round_id,
            evidence=evidence,
            availability=availability,
        )
        seen.append(track)
        if len(seen) == 2:
            raise RuntimeError("process died between tracks")
        return result

    validator._run_track = flaky

    with pytest.raises(RuntimeError):
        await validator.run_round()

    # NOTHING of the round is observable: not the EWMAs, not the evidence, and
    # (round 2) not the registry sync either — the miners were never even
    # registered, so a reader sees exactly the state that preceded the round.
    assert miner_manager.get_miner(conn, 1) is None
    assert miner_manager.get_miner(conn, 2) is None
    assert len(miner_manager.uncommitted_rounds(conn)) == 1
    assert ScorePacketEvidence(conn).recent_packet_digests() == []
    # ... and the challenges it fetched were STILL resolved
    assert len(challenge_client.resolves) == 2


# --- #9 round 2: the round's WHOLE observable state is one transaction ---------
#
# The EWMA fold was already atomic, but the registry half was not: `begin_round`
# autocommitted, then the miner sync + retention fold committed in their own
# transaction, and only later did the EWMAs and evidence land. A weight-setter
# reading the same file between those commits saw hotkey resets, purged
# accumulators and freshly-registered miners belonging to a round that might never
# finish. These tests crash the round at each intermediate point and require the
# reader's view to be EXACTLY the previous committed one.


def reader_view(conn) -> dict:
    """Everything a weight-setter can observe in the validator's database."""
    return {
        "miners": [
            dict(r) for r in conn.execute("SELECT * FROM miners ORDER BY uid")
        ],
        # (The retention_windows table was REMOVED with the retention multiplier for v1 —
        # retention removed — owner decision; an internal review.)
        "digests": ScorePacketEvidence(conn).recent_packet_digests(),
    }


@pytest.mark.parametrize(
    "crash_at", ["warrant_probe", "before_dispatch", "mid_scoring", "before_commit"]
)
async def test_a_crash_at_any_point_leaves_the_previous_state_visible(
    validator, chain, miner_client, conn, monkeypatch, crash_at
):
    """Prove it at every intermediate point of the round."""
    # --- a first round establishes the PREVIOUS consistent state ---------------
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression", 2: "compression"}
    await validator.run_round()
    before = reader_view(conn)
    assert before["miners"] and before["digests"]  # there IS something to protect

    # --- a second round that will die: a new miner AND a hotkey rotation, so it
    # carries every kind of registry effect (insert, purge, retention, track) ---
    chain.advance_blocks(150)
    chain.set_neurons([mk_neuron(1, hotkey="hk1-rotated"), mk_neuron(2)])

    boom = RuntimeError(f"process died at {crash_at}")
    if crash_at == "warrant_probe":

        async def die(neuron):
            raise boom

        monkeypatch.setattr(validator, "_probe_warrant", die)
    elif crash_at == "before_dispatch":

        async def die(*args, **kwargs):
            raise boom

        monkeypatch.setattr(validator, "_run_track", die)
    elif crash_at == "mid_scoring":

        async def die(*args, **kwargs):
            raise boom

        monkeypatch.setattr(validator, "_score_one", die)
    else:

        def die(*args, **kwargs):
            raise boom

        monkeypatch.setattr(miner_manager, "commit_round", die)

    with pytest.raises(RuntimeError):
        await validator.run_round()

    # NOTHING of the dead round is observable: not the rotated hotkey, not the
    # purged accumulator, not the new miner, not the retention fold, not evidence.
    assert reader_view(conn) == before
    assert len(miner_manager.uncommitted_rounds(conn)) == 1  # but it IS detectable


async def test_the_registry_lands_only_when_the_round_commits(
    validator, chain, miner_client, conn
):
    """The happy path of the same rule: commit publishes everything at once."""
    chain.set_neurons([mk_neuron(1), mk_neuron(2)])
    miner_client.tracks = {1: "compression", 2: "upscaling"}

    assert reader_view(conn) == {"miners": [], "digests": []}

    report = await validator.run_round()

    view = reader_view(conn)
    assert [r["uid"] for r in view["miners"]] == [1, 2]
    assert [r["track"] for r in view["miners"]] == ["compression", "upscaling"]
    assert len(view["digests"]) == 2
    assert miner_manager.uncommitted_rounds(conn) == []
    assert report.round_id is not None


async def test_a_hotkey_purge_and_its_re_probe_land_together(
    validator, chain, miner_client, conn
):
    """A reader never sees the purge without the round that caused it."""
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}
    await validator.run_round()
    assert miner_manager.get_miner(conn, 1)["accumulate_score"] > 0.0

    chain.advance_blocks(150)
    chain.set_neurons([mk_neuron(1, hotkey="hk1-rotated")])
    miner_client.tracks = {1: "upscaling"}  # the NEW miner serves another track

    await validator.run_round()

    row = miner_manager.get_miner(conn, 1)
    assert row["hotkey"] == "hk1-rotated"
    assert row["track"] == "upscaling"  # re-probed, never carried over
    # purged to 0.0 and then folded with THIS round's score, in one transaction
    assert row["accumulate_score"] == accumulate(0.0, 0.8, validator.tokenomics.ewma_decay)


def test_readers_ignore_uncommitted_rounds(conn):
    """The weight-setter must never publish evidence from a partial round."""
    miner_manager.begin_round(conn, "committed", 1, NOW_ISO)
    miner_manager.commit_round(
        conn,
        "committed",
        scores={},
        decay=DECAY,
        packets=[
            {
                "uid": 1,
                "item_id": "i1",
                "challenge_id": "c1",
                "track": "compression",
                "miner_hotkey": "hk1",
                "content_digest": "a" * 64,
                "packet_digest": "1" * 64,
                "packet_json": "{}",
                "scorer_version": "v1",
                "score": 0.8,
            }
        ],
        committed_at=NOW_ISO,
    )
    # a round left open by a crash, with evidence rows written by direct SQL
    miner_manager.begin_round(conn, "partial", 2, NOW_ISO)
    conn.execute(
        "INSERT INTO score_packets (round_id, uid, item_id, challenge_id, track,"
        " miner_hotkey, content_digest, packet_digest, packet_json, scorer_version,"
        " score, created_at) VALUES ('partial', 2, 'i2', 'c2', 'compression', 'hk2',"
        " ?, ?, '{}', 'v1', 0.9, ?)",
        ("b" * 64, "2" * 64, NOW_ISO),
    )

    evidence = ScorePacketEvidence(conn)
    assert evidence.recent_packet_digests() == ["1" * 64]
    assert evidence.has_uncommitted_round() is True


def test_recent_packet_digests_respects_the_since_cutoff(conn):
    for index, stamp in enumerate(
        ["2026-08-19T00:00:00+00:00", "2026-08-20T00:00:00+00:00"], start=1
    ):
        round_id = f"r{index}"
        miner_manager.begin_round(conn, round_id, index, stamp)
        miner_manager.commit_round(
            conn,
            round_id,
            scores={},
            decay=DECAY,
            packets=[
                {
                    "uid": index,
                    "item_id": f"i{index}",
                    "challenge_id": f"c{index}",
                    "track": "compression",
                    "miner_hotkey": f"hk{index}",
                    "content_digest": "a" * 64,
                    "packet_digest": str(index) * 64,
                    "packet_json": "{}",
                    "scorer_version": "v1",
                    "score": 0.5,
                }
            ],
            committed_at=stamp,
        )

    evidence = ScorePacketEvidence(conn)
    assert evidence.recent_packet_digests() == ["1" * 64, "2" * 64]
    assert evidence.recent_packet_digests("2026-08-19T12:00:00+00:00") == ["2" * 64]
    # a naive cutoff is interpreted as UTC rather than rejected outright
    assert evidence.recent_packet_digests("2026-08-19T12:00:00") == ["2" * 64]
