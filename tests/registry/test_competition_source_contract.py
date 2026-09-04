"""The registry <-> competition READ contract, pinned from both sides.

`vidaio.registry.competition_source` reads the competition database with plain
SELECTs and spells the orchestrator's event names as LITERALS on purpose — the
registry must not import the orchestrator's runtime (a promotion has to work when
that service is not even running). The cost of that decision is that the two
sides can drift silently: rename the event on the write side and the read side
keeps returning None, which now means "this competition archived nothing" and
refuses every promotion.

So the names and the payload SHAPE are pinned here, and the archival record is
round-tripped through both sides' real code: written by the orchestrator's own
`record_submission_archived`, read back by `SqliteCompetitionSource.facts()`.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.states import TRANSITIONS, Phase
from vidaio.registry.competition_source import (
    SUBMISSION_ARCHIVED_EVENT,
    SUBMISSION_BACKUP_GUARD,
    SqliteCompetitionSource,
)

from registry_support import COMPETITION_ID, seed_competition

AT = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)


def test_the_event_and_guard_names_match_the_write_side() -> None:
    assert SUBMISSION_ARCHIVED_EVENT == pers.EVENT_SUBMISSION_ARCHIVED
    assert SUBMISSION_BACKUP_GUARD == TRANSITIONS[
        (Phase.FINALIZING_SUBMISSIONS, Phase.VALIDATING)
    ]


def test_the_archival_record_round_trips_through_both_sides(
    comp_conn: sqlite3.Connection,
) -> None:
    """Written by the orchestrator's writer, read by the registry's reader."""
    contender_id = seed_competition(
        comp_conn, score_packet_digest="ab" * 32, archive_winner=False
    )
    source = SqliteCompetitionSource(comp_conn)
    winner = source.facts(COMPETITION_ID).winner
    assert winner is not None
    assert winner.archived_artifact_digest is None  # nothing archived yet

    pers.record_submission_archived(
        comp_conn, COMPETITION_ID, contender_id, "5c" * 32, 4096, AT
    )

    winner = source.facts(COMPETITION_ID).winner
    assert winner is not None
    assert winner.archived_artifact_digest == "5c" * 32
    assert winner.archived_artifact_bytes == 4096


def test_another_contenders_archive_is_never_read_as_the_winners(
    comp_conn: sqlite3.Connection,
) -> None:
    """The event log is per-COMPETITION; the payload's contender_id is the filter.

    Reading the latest archival event without checking whose it is would let the
    calibration entry's (or a loser's) tarball be promoted as the champion.
    """
    contender_id = seed_competition(
        comp_conn, score_packet_digest="ab" * 32, archive_winner=False
    )
    pers.record_submission_archived(
        comp_conn, COMPETITION_ID, contender_id, "5c" * 32, 4096, AT
    )
    pers.record_submission_archived(
        comp_conn, COMPETITION_ID, contender_id + 1, "9e" * 32, 8192, AT
    )

    winner = SqliteCompetitionSource(comp_conn).facts(COMPETITION_ID).winner
    assert winner is not None
    assert winner.archived_artifact_digest == "5c" * 32
