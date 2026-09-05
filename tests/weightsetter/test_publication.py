"""Publication conventions: the empty-packet merkle sentinel and real packet roots.

The sentinel is a documented convention for "no packets back this publication" —
NOT a default. review #7: the validator persists real score-packet evidence and
serves it through `ScorePacketEvidence`, so a real inference publication must
carry the real merkle set. Those tests live at the bottom of this module.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from weightsetter_support import FakePublicationInputs
from weightsetter_support import (
    AuthorityHarness,
    FakeScoringAuthorityClient,
    make_item,
    make_miner,
)

from vidaio.audit import canonical_json_bytes, merkle_root, sha256_hex
from vidaio.audit.store import set_member_key
from vidaio.authority import EPOCH_LOG_MEMBER, epoch_prefix
from vidaio.chain import ChainNeuron
from vidaio.core import apply_migrations, connect
from vidaio.epoch import EpochLog, EpochLogInvalid
from vidaio.validator import MIGRATIONS_DIR, ScorePacketEvidence, miner_manager
from vidaio.weightsetter import (
    EMPTY_SCORE_PACKET_MARKER,
    EMPTY_SCORE_PACKET_SET_ROOT,
    intents,
)
from vidaio.weightsetter.shared_snapshot import SharedSnapshotProvider


def test_empty_sentinel_is_the_documented_convention():
    # audit.merkle_root requires >= 1 leaf — that gap is exactly why the sentinel exists
    with pytest.raises(ValueError):
        merkle_root([])
    # convention: the root over the SINGLE leaf sha256(EMPTY_SCORE_PACKET_MARKER)
    assert EMPTY_SCORE_PACKET_SET_ROOT == merkle_root(
        [sha256_hex(EMPTY_SCORE_PACKET_MARKER)]
    )


async def test_no_publication_inputs_uses_sentinel_root(make_setter, ledger, mk_miner):
    setter = make_setter([mk_miner(1)])  # publication_inputs=None

    assert await setter.attempt_once() is True

    record = json.loads(ledger.get(1)["canonical_json"])
    assert record["score_packet_merkle_root"] == EMPTY_SCORE_PACKET_SET_ROOT


async def test_empty_packet_list_uses_sentinel_root(make_setter, ledger, mk_miner):
    setter = make_setter([mk_miner(1)], publication_inputs=FakePublicationInputs([]))

    assert await setter.attempt_once() is True

    record = json.loads(ledger.get(1)["canonical_json"])
    assert record["score_packet_merkle_root"] == EMPTY_SCORE_PACKET_SET_ROOT


async def test_real_score_packets_produce_their_merkle_root(
    make_setter, ledger, mk_miner
):
    digests = [
        sha256_hex(b"packet-1"),
        sha256_hex(b"packet-2"),
        sha256_hex(b"packet-3"),
    ]
    setter = make_setter(
        [mk_miner(1)], publication_inputs=FakePublicationInputs(digests)
    )

    assert await setter.attempt_once() is True

    record = json.loads(ledger.get(1)["canonical_json"])
    assert record["score_packet_merkle_root"] == merkle_root(digests)
    assert record["score_packet_merkle_root"] != EMPTY_SCORE_PACKET_SET_ROOT


async def test_shared_publication_binds_exact_epoch_packets_and_snapshot(
    make_setter, ledger, conn, tmp_path
):
    """A thin validator publishes from its verified log, not its empty local DB."""
    authority = AuthorityHarness(tmp_path / "authority")
    try:
        # The withheld upscaling pool targets the subnet owner. Production binds
        # that positive uid to the refreshed metagraph identity immediately before
        # submission; model the same live owner instead of trusting a numeric uid.
        authority.chain._neurons.append(  # noqa: SLF001 - explicit chain fixture
            ChainNeuron(
                uid=0,
                hotkey="owner-hk",
                coldkey="owner-ck",
                ip="0.0.0.0",
                alpha_stake=0.0,
                emission=0.0,
                is_validator=True,
            )
        )
        finalized = await authority.finalize(
            epoch_id=55,
            close_block=19999,
            miners=[make_miner(1)],
            items=[make_item(1, authority.store)],
        )
        provider = authority.provider(epoch_id=55)
        setter = make_setter(
            [],
            chain_override=authority.chain,
            snapshots_override=provider,
            publication_inputs=provider,
            # The fixed compression/upscaling split withholds the empty
            # upscaling pool to the authority's canonical test sink (uid 0).
            burn_uid=0,
        )

        assert await setter.attempt_once() is True

        record = json.loads(ledger.get(1)["canonical_json"])
        packet_digests = list(provider.score_packet_digests())
        assert record["score_packet_merkle_root"] == merkle_root(packet_digests)
        assert record["snapshot_digest"] == finalized.log_digest
        assert intents.intents(conn)[0]["snapshot_digest"] == finalized.log_digest
    finally:
        authority.close()


async def test_shared_evidence_failure_never_blocks_authority_vector_and_recovers_exact_epoch(
    make_setter, ledger, conn, tmp_path, clock
):
    """Publication/audit evidence is post-submit and report-only in production.

    Even when both the cheap pre-submit ref copy and the first post-submit exact
    resolution fail, the authenticated authority vector lands. The accepted intent
    keeps ``null`` (never ``[]``) plus the exact epoch identity, and a later retry
    resolves that old epoch by id even when ``latest_pointer`` is unavailable.
    """
    authority = AuthorityHarness(tmp_path / "authority-no-gate")
    try:
        authority.chain._neurons.append(  # noqa: SLF001 - explicit chain fixture
            ChainNeuron(
                uid=0,
                hotkey="owner-hk",
                coldkey="owner-ck",
                ip="0.0.0.0",
                alpha_stake=0.0,
                emission=0.0,
                is_validator=True,
            )
        )
        finalized = await authority.finalize(
            epoch_id=55,
            close_block=19999,
            miners=[make_miner(1)],
            items=[make_item(1, authority.store)],
        )
        provider = authority.provider(epoch_id=55)
        exact_resolver = provider.score_packet_digests_for_epoch

        def capture_failed():
            raise OSError("durable ref copy failed")

        def exact_resolution_failed(epoch_id, *, expected_snapshot_digest):
            raise OSError("public epoch evidence temporarily unavailable")

        provider.committed_packet_digests = capture_failed  # type: ignore[method-assign]
        provider.score_packet_digests_for_epoch = (  # type: ignore[method-assign]
            exact_resolution_failed
        )
        anchors_before = len(authority.chain.anchored)
        setter = make_setter(
            [],
            chain_override=authority.chain,
            snapshots_override=provider,
            publication_inputs=provider,
            burn_uid=0,
        )
        setter._monotonic_clock = lambda: clock().timestamp()

        assert await setter.attempt_once() is True

        assert len(authority.chain.weight_calls) == 1
        row = intents.intents(conn)[0]
        assert row["state"] == intents.STATE_ACCEPTED
        assert row["packet_digests_json"] == "null"
        assert row["snapshot_epoch_id"] == 55
        assert row["snapshot_digest"] == finalized.log_digest
        assert len(authority.chain.anchored) == anchors_before
        with pytest.raises(KeyError):
            ledger.get(1)
        assert setter.metric_publication_input_failures._value.get() == 1

        # Simulate a restart after the provider's latest-epoch cache disappeared.
        # Recovery must use pointer_for(55), never latest_pointer().
        provider._resolved = None  # noqa: SLF001 - crash/restart fixture
        provider._client.set_latest(None)  # noqa: SLF001 - exact-history fixture
        calls = []
        recovered_digests = []

        def recover(epoch_id, *, expected_snapshot_digest):
            calls.append((epoch_id, expected_snapshot_digest))
            resolved = list(
                exact_resolver(
                    epoch_id, expected_snapshot_digest=expected_snapshot_digest
                )
            )
            recovered_digests.extend(resolved)
            return resolved

        provider.score_packet_digests_for_epoch = recover  # type: ignore[method-assign]

        clock.advance(setter.config.reconciliation_interval_seconds)
        assert await setter.reconcile() == 1
        row = intents.intents(conn)[0]
        assert row["state"] == intents.STATE_PUBLISHED
        assert row["packet_digests_json"] != "null"
        assert calls == [(55, finalized.log_digest)]
        record = json.loads(ledger.get(1)["canonical_json"])
        assert record["score_packet_merkle_root"] == merkle_root(recovered_digests)
        assert record["snapshot_digest"] == finalized.log_digest
    finally:
        authority.close()


async def test_strict_economic_audit_failure_never_blocks_authenticated_authority_vector(
    make_setter, conn, tmp_path
):
    """Decision 24: economic re-derivation reports, but exact safe u16 still lands.

    The stored bytes remain canonical, content-addressed and independently anchored;
    their pointer binds the exact u16 grid and close-block census.  Only the declared
    floating economic shares disagree with that u16.  ``EpochLog.from_json`` therefore
    rejects the full audit model, while the narrow submission view must still deliver
    the authority vector unchanged to ``set_weights``.
    """
    authority = AuthorityHarness(tmp_path / "authority-economic-disagreement")
    try:
        authority.chain._neurons.append(  # noqa: SLF001 - explicit chain fixture
            ChainNeuron(
                uid=0,
                hotkey="owner-hk",
                coldkey="owner-ck",
                ip="0.0.0.0",
                alpha_stake=0.0,
                emission=0.0,
                is_validator=True,
            )
        )
        finalized = await authority.finalize(
            epoch_id=55,
            close_block=19999,
            miners=[make_miner(1)],
            items=[make_item(1, authority.store)],
        )
        original_u16 = dict(finalized.log.weight_u16)
        assert set(original_u16) == {0, 1}

        raw = json.loads(finalized.log.to_json())
        raw["weight_shares"] = {"0": 0.99, "1": 0.01}
        bad_bytes = canonical_json_bytes(raw)
        with pytest.raises(EpochLogInvalid, match="weight_u16"):
            EpochLog.from_json(bad_bytes)

        authority.store._raw_put(  # noqa: SLF001 - authenticated-byte fixture
            set_member_key(epoch_prefix(55), EPOCH_LOG_MEMBER), bad_bytes
        )
        digest = sha256_hex(bad_bytes)
        old_pointer = authority.pointer(55)
        pointer = old_pointer.model_copy(
            update={
                "snapshot_digest": digest,
                "anchor": old_pointer.anchor.model_copy(update={"digest": digest}),
            }
        )

        class ExactAnchor:
            def read_epoch_anchor(self, *, netuid, epoch_id):
                assert (netuid, epoch_id) == (85, 55)
                return digest

        provider = SharedSnapshotProvider(
            client=FakeScoringAuthorityClient(latest=pointer),
            store=authority.store,
            netuid=85,
            anchor_reader=ExactAnchor(),
        )
        setter = make_setter(
            [],
            chain_override=authority.chain,
            snapshots_override=provider,
            publication_inputs=provider,
            burn_uid=0,
        )

        assert await setter.attempt_once() is True

        assert provider.resolved_log() is None
        assert provider.epoch_inputs().weight_u16 == original_u16
        assert len(authority.chain.weight_calls) == 1
        assert authority.chain.weight_calls[0][1] == {
            uid: float(value) for uid, value in original_u16.items()
        }
        row = intents.intents(conn)[0]
        assert row["state"] == intents.STATE_PUBLISHED
        assert row["snapshot_digest"] == digest
    finally:
        authority.close()


def test_unresolved_intent_never_loads_as_empty_and_resolves_exactly_once(conn):
    intent_id = intents.record_intent(
        conn,
        created_at="2026-08-20T11:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 1.0},
        packet_digests=None,
        snapshot_digest="a" * 64,
        snapshot_epoch_id=55,
    )
    row = intents.get_intent(conn, intent_id)
    assert intents.packet_digests_resolved(row) is False
    with pytest.raises(ValueError, match="unresolved"):
        intents.load_packet_digests(row)

    resolved = [sha256_hex(b"packet-b"), sha256_hex(b"packet-a")]
    intents.attach_packet_digests(conn, intent_id, packet_digests=resolved)
    intents.attach_packet_digests(conn, intent_id, packet_digests=reversed(resolved))
    assert intents.load_packet_digests(intents.get_intent(conn, intent_id)) == sorted(
        resolved
    )
    with pytest.raises(ValueError, match="conflicts"):
        intents.attach_packet_digests(
            conn, intent_id, packet_digests=[sha256_hex(b"different")]
        )


#


def _validator_db(path: Path) -> sqlite3.Connection:
    conn = connect(path)
    apply_migrations(conn, MIGRATIONS_DIR)
    return conn


def _commit_round(conn: sqlite3.Connection, round_id: str, digests, *, at: str) -> None:
    """Write a committed round of score-packet evidence exactly as the validator does."""
    miner_manager.begin_round(conn, round_id, 1, at)
    miner_manager.commit_round(
        conn,
        round_id,
        scores={},
        decay=0.75,
        packets=[
            {
                "uid": index,
                "item_id": f"{round_id}:{index}",
                "challenge_id": round_id,
                "track": "compression",
                "miner_hotkey": f"hk{index}",
                "content_digest": "a" * 64,
                "packet_digest": digest,
                "packet_json": "{}",
                "scorer_version": "vidaio-scorer/1",
                "score": 0.5,
            }
            for index, digest in enumerate(digests, start=1)
        ],
        committed_at=at,
    )


async def test_validator_evidence_publishes_the_real_merkle_set(
    make_setter, ledger, mk_miner, tmp_path
):
    """The end of an internal review: real inference weights carry real packet digests."""
    vconn = _validator_db(tmp_path / "validator.db")
    digests = [sha256_hex(b"real-1"), sha256_hex(b"real-2")]
    _commit_round(vconn, "r1", digests, at="2026-08-20T11:00:00+00:00")

    evidence = ScorePacketEvidence(vconn)
    setter = make_setter([mk_miner(1)], publication_inputs=evidence)

    assert await setter.attempt_once() is True

    record = json.loads(ledger.get(1)["canonical_json"])
    assert record["score_packet_merkle_root"] == merkle_root(digests)
    assert record["score_packet_merkle_root"] != EMPTY_SCORE_PACKET_SET_ROOT


async def test_evidence_from_an_uncommitted_round_is_never_published(
    make_setter, ledger, mk_miner, tmp_path
):
    """The weight-setter reads only COMMITTED rounds."""
    vconn = _validator_db(tmp_path / "validator.db")
    miner_manager.begin_round(vconn, "partial", 1, "2026-08-20T11:00:00+00:00")
    vconn.execute(
        "INSERT INTO score_packets (round_id, uid, item_id, challenge_id, track,"
        " miner_hotkey, content_digest, packet_digest, packet_json, scorer_version,"
        " score, created_at) VALUES ('partial', 1, 'i1', 'c1', 'compression', 'hk1',"
        " ?, ?, '{}', 'v1', 0.9, '2026-08-20T11:00:00+00:00')",
        ("a" * 64, sha256_hex(b"partial-round-packet")),
    )
    setter = make_setter([mk_miner(1)], publication_inputs=ScorePacketEvidence(vconn))

    assert await setter.attempt_once() is True

    record = json.loads(ledger.get(1)["canonical_json"])
    assert record["score_packet_merkle_root"] == EMPTY_SCORE_PACKET_SET_ROOT


async def test_consecutive_publications_partition_the_evidence(
    make_setter, chain, ledger, conn, mk_miner, tmp_path
):
    """The FREEZE watermark of the last published intent is the next `since`."""
    vconn = _validator_db(tmp_path / "validator.db")
    first = [sha256_hex(b"round-1")]
    _commit_round(vconn, "r1", first, at="2026-08-20T11:00:00+00:00")
    setter = make_setter([mk_miner(1)], publication_inputs=ScorePacketEvidence(vconn))

    assert await setter.attempt_once() is True
    assert json.loads(ledger.get(1)["canonical_json"])["score_packet_merkle_root"] == (
        merkle_root(first)
    )

    # a later round's evidence, published after the tempo gate reopens
    watermark = intents.publication_watermark(conn)
    assert watermark is not None
    second = [sha256_hex(b"round-2")]
    _commit_round(vconn, "r2", second, at="2026-08-21T11:00:00+00:00")
    chain.advance_blocks(chain.tempo + 1)

    assert await setter.attempt_once() is True

    record = json.loads(ledger.get(2)["canonical_json"])
    # ONLY the new round's packets — the first publication already committed to r1
    assert record["score_packet_merkle_root"] == merkle_root(second)


# --- new-6: a delayed publication must not open an evidence GAP ------------------
#
# The evidence window's lower bound used to be the previous intent's `settled_at`
# — the moment its anchor finally succeeded. Its packet list, though, was frozen
# when the intent was RECORDED. Whenever an anchor hung or failed and was re-driven
# a cycle later, every packet produced in between fell between the two windows:
# past the first publication's frozen list, before the second's cutoff. Nothing
# ever committed to it, and nothing could detect that.


async def test_a_failed_anchor_leaves_no_evidence_gap(
    make_setter, chain, ledger, conn, clock, mk_miner, tmp_path
):
    """The finding's scenario: publish, anchor fails, packets arrive, re-drive."""
    from weightsetter_support import HangingAnchorChain

    vconn = _validator_db(tmp_path / "validator.db")
    first = [sha256_hex(b"before-freeze")]
    _commit_round(vconn, "r1", first, at="2026-08-20T11:00:00+00:00")

    hanging = HangingAnchorChain(chain)
    setter = make_setter(
        [mk_miner(1)],
        chain_override=hanging,
        publication_inputs=ScorePacketEvidence(vconn),
    )
    setter._monotonic_clock = lambda: clock().timestamp()

    # the weights land, the anchor does not: the intent is accepted, not
    # published. Its packet list froze HERE, at T0 = 12:00.
    assert await setter.attempt_once() is True
    row = intents.intents(conn)[0]
    assert row["state"] == intents.STATE_ACCEPTED
    assert row["packets_frozen_at"].startswith("2026-08-20T12:00:00")

    # packets produced WHILE the anchor is owed — the ones that used to vanish:
    # after the first list was frozen, before the first intent settled.
    during = [sha256_hex(b"during-the-failed-anchor")]
    _commit_round(vconn, "r2", during, at="2026-08-20T12:10:00+00:00")

    # the anchor finally succeeds twenty minutes later, so settled_at (12:20) is
    # WELL past those packets — the old cutoff would have skipped straight over them
    clock.advance(20 * 60)
    hanging.anchor_ok = True
    assert await setter.reconcile() == 1
    row = intents.intents(conn)[0]
    assert row["state"] == intents.STATE_PUBLISHED
    assert row["settled_at"].startswith("2026-08-20T12:20:00")
    assert intents.publication_watermark(conn) == row["packets_frozen_at"]

    after = [sha256_hex(b"after")]
    _commit_round(vconn, "r3", after, at="2026-08-20T12:25:00+00:00")
    clock.advance(10 * 60)
    chain.advance_blocks(chain.tempo + 1)
    assert await setter.attempt_once() is True

    published = json.loads(ledger.get(2)["canonical_json"])
    # every packet belongs to SOME publication: no digest fell between the two
    assert published["score_packet_merkle_root"] == merkle_root(during + after)


def test_the_watermark_is_the_freeze_instant_not_the_settlement(conn):
    """Unit-level: the two timestamps differ, and the earlier one is the bound."""
    intent_id = intents.record_intent(
        conn,
        created_at="2026-08-20T11:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 1.0},
        packet_digests=[],
        packets_frozen_at="2026-08-20T11:00:00+00:00",
    )
    intents.mark_accepted(
        conn, intent_id, accepted_block=1, resolution="chain_accepted"
    )
    # settled an hour after the list was frozen (a re-driven anchor)
    intents.mark_published(conn, intent_id, at="2026-08-20T12:00:00+00:00")

    assert intents.publication_watermark(conn) == "2026-08-20T11:00:00+00:00"


def test_a_legacy_row_without_a_freeze_stamp_falls_back_to_settled_at(conn):
    """Rows written before the column existed keep their old (later) bound."""
    intent_id = intents.record_intent(
        conn,
        created_at="2026-08-20T11:00:00+00:00",
        attempt_block=1,
        version_key=0,
        weights={1: 1.0},
        packet_digests=[],
    )
    intents.mark_accepted(
        conn, intent_id, accepted_block=1, resolution="chain_accepted"
    )
    intents.mark_published(conn, intent_id, at="2026-08-20T12:00:00+00:00")
    conn.execute(
        "UPDATE weight_intents SET packets_frozen_at = NULL WHERE id = ?", (intent_id,)
    )

    assert intents.publication_watermark(conn) == "2026-08-20T12:00:00+00:00"


#


class BrokenPublicationInputs:
    """A provider whose evidence read fails (a corrupt/locked validator DB)."""

    def __init__(self) -> None:
        self.calls = 0

    def recent_packet_digests(self, since=None):
        self.calls += 1
        raise sqlite3.OperationalError("database disk image is malformed")

    def score_packet_digests(self):
        raise sqlite3.OperationalError("database disk image is malformed")


async def test_unreadable_evidence_skips_the_attempt_instead_of_anchoring_the_sentinel(
    make_setter, chain, ledger, conn, mk_miner
):
    """Round-2 an internal review: 'could not read' must never publish as 'there were none'.

    The sentinel is a PUBLIC claim that this publication had no score packets.
    Anchoring it because the evidence query raised would put a false statement on
    chain — and it is unfalsifiable afterwards, since the real packets exist.
    """
    provider = BrokenPublicationInputs()
    setter = make_setter([mk_miner(1)], publication_inputs=provider)

    assert await setter.attempt_once() is False

    assert provider.calls == 1
    assert chain.weight_calls == []  # nothing submitted either — no intent, no write
    assert chain.anchored == []
    assert intents.intents(conn) == []
    with pytest.raises(KeyError):
        ledger.get(1)
    assert setter.metric_publication_input_failures._value.get() == 1


async def test_a_genuinely_empty_packet_set_still_publishes_the_sentinel(
    make_setter, ledger, mk_miner, tmp_path
):
    """The distinction is the point: an EMPTY read is honest, a FAILED one is not."""
    vconn = _validator_db(tmp_path / "validator.db")  # no rounds at all
    setter = make_setter([mk_miner(1)], publication_inputs=ScorePacketEvidence(vconn))

    assert await setter.attempt_once() is True

    record = json.loads(ledger.get(1)["canonical_json"])
    assert record["score_packet_merkle_root"] == EMPTY_SCORE_PACKET_SET_ROOT


async def test_a_provider_raising_typeerror_is_not_mistaken_for_a_narrower_surface(
    make_setter, chain, conn, mk_miner
):
    """A TypeError from INSIDE the provider used to be read as 'takes no cutoff'.

    That silently fell back to the unwindowed call and widened the evidence
    window. Feature detection is by signature; a raised TypeError is a failure.
    """

    class ExplodingProvider:
        def recent_packet_digests(self, since=None):
            raise TypeError("unsupported operand type(s) inside the provider")

        def score_packet_digests(self):  # pragma: no cover - must not be reached
            raise AssertionError("the fallback must not be taken")

    setter = make_setter([mk_miner(1)], publication_inputs=ExplodingProvider())

    assert await setter.attempt_once() is False

    assert chain.weight_calls == []
    assert intents.intents(conn) == []
    assert setter.metric_publication_input_failures._value.get() == 1
