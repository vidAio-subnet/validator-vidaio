"""A completed competition that cannot be applied must never HOLD the epoch."""

from __future__ import annotations

import logging

import pytest

from vidaio.competition.epoch_evidence import (
    CompetitionEvidenceError,
    build_competition_epoch_evidence,
)
from vidaio.competition.evidence_guard import (
    guarded_competition_evidence,
    precheck_applicable,
)
from vidaio.epoch import MinerCensusEntry
from vidaio.tokenomics.breakthrough import resolve_reward_window
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import RewardWindowState

from integration_support import COMPLETED_AT, GoldenWorld

CONFIG = TokenomicsConfig(competition_emissions_enabled=True)


def _census(*hotkeys: str) -> dict[str, MinerCensusEntry]:
    return {
        hotkey: MinerCensusEntry(
            uid=index + 1, hotkey=hotkey, coldkey=f"ck-{hotkey}", ip=f"10.0.0.{index + 1}"
        )
        for index, hotkey in enumerate(hotkeys)
    }


def _builder(world: GoldenWorld, *hotkeys: str):
    def build():
        return build_competition_epoch_evidence(
            world.comp_conn,
            census_by_hotkey=_census(*hotkeys),
            store=world.store,
            through_time=COMPLETED_AT,
        )

    return build


def test_complete_evidence_passes_through_unchanged(fresh_world: GoldenWorld) -> None:
    guarded = guarded_competition_evidence(
        _builder(fresh_world, "hk-a", "hk-b"),
        policy="defer", config=CONFIG, prior=None, now=COMPLETED_AT,
    )
    assert guarded.deferred_error is None and not guarded.quarantined
    assert guarded.evidence is not None
    assert guarded.evidence.result.contenders[0].hotkey in {"hk-a", "hk-b"}


def test_missing_census_contender_defers_instead_of_holding(
    fresh_world: GoldenWorld, caplog: pytest.LogCaptureFixture
) -> None:
    deferred: list[str] = []
    with caplog.at_level(logging.CRITICAL):
        guarded = guarded_competition_evidence(
            _builder(fresh_world, "hk-a"),  # hk-b is BUILT but left the census
            policy="defer", config=CONFIG, prior=None, now=COMPLETED_AT,
            on_defer=deferred.append,
        )
    assert guarded.evidence is None
    assert guarded.deferred_error is not None
    assert "absent from the close-block census" in guarded.deferred_error
    assert len(deferred) == 1
    assert any("DEFERRED" in record.getMessage() for record in caplog.records)


def test_hold_policy_keeps_the_fail_closed_behaviour(fresh_world: GoldenWorld) -> None:
    with pytest.raises(CompetitionEvidenceError, match="absent from the close-block census"):
        guarded_competition_evidence(
            _builder(fresh_world, "hk-a"),
            policy="hold", config=CONFIG, prior=None, now=COMPLETED_AT,
        )


def test_unexpected_store_failure_is_also_deferred() -> None:
    def build():
        raise OSError("object store unreachable")

    calls: list[str] = []
    guarded = guarded_competition_evidence(
        build, policy="defer", config=CONFIG, prior=None, now=COMPLETED_AT,
        on_defer=calls.append,
    )
    assert guarded.evidence is None
    assert guarded.deferred_error == "OSError: object store unreachable"
    assert calls == ["OSError: object store unreachable"]


def test_quarantined_competition_is_skipped_without_an_error(
    fresh_world: GoldenWorld,
) -> None:
    evidence = _builder(fresh_world, "hk-a", "hk-b")()
    assert evidence is not None
    calls: list[str] = []
    guarded = guarded_competition_evidence(
        _builder(fresh_world, "hk-a", "hk-b"),
        policy="defer", config=CONFIG, prior=None, now=COMPLETED_AT,
        quarantined=frozenset({evidence.competition_input.competition_id}),
        on_defer=calls.append,
    )
    assert guarded.evidence is None and guarded.quarantined
    assert guarded.deferred_error is None and calls == []


def test_already_applied_cycle_is_deferred_by_the_precheck(
    fresh_world: GoldenWorld,
) -> None:
    evidence = _builder(fresh_world, "hk-a", "hk-b")()
    assert evidence is not None
    applied = resolve_reward_window(CONFIG, RewardWindowState(), evidence.result)
    with pytest.raises(CompetitionEvidenceError, match="not newer than the last applied"):
        precheck_applicable(CONFIG, applied, evidence, now=COMPLETED_AT)
    guarded = guarded_competition_evidence(
        lambda: evidence, policy="defer", config=CONFIG, prior=applied, now=COMPLETED_AT,
    )
    assert guarded.evidence is None and guarded.deferred_error is not None


def test_applied_at_must_equal_the_epoch_close_time(fresh_world: GoldenWorld) -> None:
    from datetime import timedelta

    evidence = _builder(fresh_world, "hk-a", "hk-b")()
    assert evidence is not None
    with pytest.raises(CompetitionEvidenceError, match="applied_at"):
        precheck_applicable(
            CONFIG, None, evidence, now=COMPLETED_AT + timedelta(seconds=1)
        )


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="defer"):
        guarded_competition_evidence(
            lambda: None, policy="ignore", config=CONFIG, prior=None, now=COMPLETED_AT,  # type: ignore[arg-type]
        )
