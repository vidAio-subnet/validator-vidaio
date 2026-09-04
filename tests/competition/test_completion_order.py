"""Global competition result order follows terminal events, not scheduling order."""

from datetime import timedelta

from vidaio.competition import Phase
from vidaio.competition import repository as repo
from vidaio.competition.epoch_evidence import latest_completed_competition_id
from vidaio.competition.orchestrator.results import competition_cycle, completed_at

from support import END, T0, build_manifest


def _record_completion(conn, competition_id: str, when) -> None:
    repo.set_status(conn, competition_id, Phase.COMPLETED, when)
    repo.record_event(
        conn,
        competition_id,
        "transition",
        when,
        from_phase=Phase.AWAITING_END_TIME,
        to_phase=Phase.COMPLETED,
        guard="test-terminal-completion",
    )


def test_later_completion_replaces_even_when_competition_was_created_first(
    engine, conn
) -> None:
    older_created = build_manifest("created-first")
    newer_created = build_manifest("created-second")
    engine.create_competition(conn, older_created, T0)
    engine.create_competition(conn, newer_created, T0 + timedelta(seconds=1))

    second_completed_at = END + timedelta(minutes=1)
    first_completed_at = END + timedelta(minutes=2)
    _record_completion(conn, newer_created.competition_id, second_completed_at)
    _record_completion(conn, older_created.competition_id, first_completed_at)

    assert competition_cycle(conn, newer_created.competition_id) == 1
    assert competition_cycle(conn, older_created.competition_id) == 2
    assert completed_at(conn, older_created.competition_id) == first_completed_at
    assert (
        latest_completed_competition_id(conn, through_time=second_completed_at)
        == newer_created.competition_id
    )
    assert (
        latest_completed_competition_id(conn, through_time=first_completed_at)
        == older_created.competition_id
    )


def test_equal_completion_timestamps_use_append_order_as_stable_tie_break(
    engine, conn
) -> None:
    first = build_manifest("tie-first")
    second = build_manifest("tie-second")
    engine.create_competition(conn, first, T0)
    engine.create_competition(conn, second, T0 + timedelta(seconds=1))

    when = END + timedelta(minutes=1)
    _record_completion(conn, first.competition_id, when)
    _record_completion(conn, second.competition_id, when)

    assert competition_cycle(conn, first.competition_id) == 1
    assert competition_cycle(conn, second.competition_id) == 2
    assert latest_completed_competition_id(conn, through_time=when) == second.competition_id
