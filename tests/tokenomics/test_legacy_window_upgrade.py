"""Tokenomics v3: a window folded before schema v18 is re-resolved once from its source result."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from vidaio.authority.finalizer import EpochFinalizer, EpochLogInvalid
from vidaio.epoch.window_source import WindowSourceUnavailable, find_window_source_result
from vidaio.tokenomics import (
    EmissionState,
    MinerSnapshot,
    RewardWindowState,
    TokenomicsConfig,
    build_weight_vector,
    is_legacy_window,
    resolve_reward_window,
    upgrade_legacy_window,
)
from vidaio.tokenomics.breakthrough import emission_shares, podium_hotkey_shares
from vidaio.tokenomics.state import CompetitionResult, CompetitionRules, ContenderResult

T0 = datetime(2026, 9, 27, 12, 33, tzinfo=UTC)
DIGEST = "ab" * 32
CONFIG = TokenomicsConfig(competition_emissions_enabled=True)

#: sn85-compression-001c, official packet means (all ten clips valid), 2026-09-25.
RESULT_001C = (
    ("c9", 205, 0.86989713), ("c12", 209, 0.86904473), ("c13", 139, 0.86902210),
    ("c14", 129, 0.86876852), ("c6", 8, 0.86876727), ("c10", 137, 0.86870710),
    ("c5", 165, 0.86823017), ("c7", 144, 0.86669866), ("c1", 236, 0.86507131),
    ("c8", 4, 0.86471659), ("c4", 92, 0.86460087), ("c2", 10, 0.86383178),
    ("c3", 189, 0.81060471),
)
RULES_001C = CompetitionRules(
    crown_margin=0.05, crown_min_score=0.869, podium_min_margin=0.0, podium_min_score=0.86
)


def _result(cycle: int = 7, applied_at: datetime = T0) -> CompetitionResult:
    return CompetitionResult(
        competition_id="sn85-compression-001c", track="compression", cycle=cycle,
        applied_at=applied_at,
        contenders=tuple(ContenderResult(h, u, s) for h, u, s in RESULT_001C),
        baseline_score=0.25595025, baseline_version=0, baseline_artifact_digest=DIGEST,
        rules=RULES_001C,
    )


def _legacy_window(result: CompetitionResult) -> RewardWindowState:
    """What the pre-v3 fold wrote: three hotkeys, no payout policy."""
    fresh = resolve_reward_window(CONFIG, RewardWindowState(), result)
    return replace(fresh, podium_hotkeys=fresh.podium_hotkeys[:3], competition_share=None, place_shares=())


def test_001c_replay_pays_the_top_five_with_the_crown_split() -> None:
    result = _result()
    legacy = _legacy_window(result)
    assert is_legacy_window(legacy) and legacy.kind is EmissionState.CROWN
    upgraded = upgrade_legacy_window(CONFIG, legacy, result)
    assert upgraded.podium_hotkeys == ("c9", "c12", "c13", "c14", "c6")
    assert upgraded.place_shares == pytest.approx((0.90, 0.04, 0.03, 0.02, 0.01))
    assert upgraded.competition_share == 1.0
    assert not is_legacy_window(upgraded)
    # Only the payout changes: kind, interval, winner and provenance are kept.
    for field in ("kind", "starts_at", "ends_at", "winner_hotkey", "winner_uid", "winner_score",
                  "winner_margin", "baseline_score", "source_competition_id", "source_cycle",
                  "last_applied_cycle"):
        assert getattr(upgraded, field) == getattr(legacy, field), field
    # The whole emission goes to the five places; inference earns nothing.
    shares = emission_shares(CONFIG, upgraded, T0 + timedelta(hours=2))
    assert shares.inference == 0.0 and shares.competition == 1.0 and shares.burn == 0.0
    assert podium_hotkey_shares(upgraded, CONFIG) == pytest.approx(
        {"c9": 0.9, "c12": 0.04, "c13": 0.03, "c14": 0.02, "c6": 0.01}
    )
    miners = [
        MinerSnapshot(uid=u, hotkey=h, coldkey=f"ck{u}", ip=f"10.0.{u // 256}.{u % 256}",
                      track="compression", accumulate_score=0.0)
        for h, u, _ in RESULT_001C[:5]
    ]
    vector = build_weight_vector(CONFIG, miners, burn_uid=0, reward_state=upgraded, now=T0 + timedelta(hours=2))
    assert vector == pytest.approx({205: 0.9, 209: 0.04, 139: 0.03, 129: 0.02, 8: 0.01})


def test_upgrade_is_one_shot_and_leaves_v3_and_idle_windows_alone() -> None:
    result = _result()
    upgraded = upgrade_legacy_window(CONFIG, _legacy_window(result), result)
    assert upgrade_legacy_window(CONFIG, upgraded, result) is upgraded
    fresh = resolve_reward_window(CONFIG, RewardWindowState(), result)
    assert not is_legacy_window(fresh) and upgrade_legacy_window(CONFIG, fresh, result) is fresh
    idle = RewardWindowState()
    assert not is_legacy_window(idle) and upgrade_legacy_window(CONFIG, idle, result) is idle


def test_upgrade_refuses_a_source_that_is_not_the_windows_own_result() -> None:
    result = _result()
    legacy = _legacy_window(result)
    for other in (
        replace(result, cycle=8),
        replace(result, competition_id="sn85-compression-002"),
        replace(result, track="upscaling"),
        replace(result, applied_at=T0 + timedelta(minutes=72)),
        replace(result, contenders=result.contenders[1:]),  # another winner
    ):
        with pytest.raises(ValueError):
            upgrade_legacy_window(CONFIG, legacy, other)


def test_per_competition_split_and_empty_places_follow_the_anchored_rules() -> None:
    rules = replace(RULES_001C, crown_split=(0.7, 0.2, 0.1), crown_competition_share=0.95)
    three = _result()
    three = replace(three, rules=rules)
    upgraded = upgrade_legacy_window(CONFIG, _legacy_window(three), three)
    assert upgraded.podium_hotkeys == ("c9", "c12", "c13")
    assert upgraded.place_shares == pytest.approx((0.7, 0.2, 0.1))
    assert upgraded.competition_share == 0.95
    # Only two contenders clear the podium bar: their shares absorb the empty places.
    short = replace(_result(), contenders=_result().contenders[:2] + (ContenderResult("x", 1, 0.5),))
    lifted = upgrade_legacy_window(CONFIG, _legacy_window(short), short)
    assert lifted.podium_hotkeys == ("c9", "c12")
    assert lifted.place_shares == pytest.approx((0.9 / 0.94, 0.04 / 0.94))


def _chain(result: CompetitionResult, depth: int) -> SimpleNamespace:
    """A hash chain of `depth` carry logs on top of the log that applied `result`."""
    window = _legacy_window(result)
    logs = {}
    applying = SimpleNamespace(epoch_id=100, prior_epoch_id=99, prior_log_digest="d99",
                               reward_window_state=window, competition_result=result)
    logs[(100, "d100")] = applying
    for i in range(1, depth + 1):
        logs[(100 + i, f"d{100 + i}")] = SimpleNamespace(
            epoch_id=100 + i, prior_epoch_id=100 + i - 1, prior_log_digest=f"d{100 + i - 1}",
            reward_window_state=window, competition_result=None)
    logs[(99, "d99")] = SimpleNamespace(epoch_id=99, prior_epoch_id=98, prior_log_digest="d98",
                                        reward_window_state=RewardWindowState(), competition_result=None)
    reads: list[int] = []

    def read(epoch_id: int, digest: str) -> SimpleNamespace:
        reads.append(epoch_id)
        return logs[(epoch_id, digest)]

    return SimpleNamespace(read=read, reads=reads, top=logs[(100 + depth, f"d{100 + depth}")], window=window)


def test_the_source_is_found_by_walking_the_hash_chain() -> None:
    result = _result()
    chain = _chain(result, depth=5)
    assert find_window_source_result(chain.read, chain.top, chain.window) == result
    assert chain.reads == [104, 103, 102, 101, 100]
    # the prior log itself applied the result: no read at all
    direct = _chain(result, depth=0)
    assert find_window_source_result(direct.read, direct.top, direct.window) == result
    assert direct.reads == []


def test_the_walk_is_bounded_and_never_leaves_the_window() -> None:
    result = _result()
    chain = _chain(result, depth=5)
    with pytest.raises(WindowSourceUnavailable):
        find_window_source_result(chain.read, chain.top, chain.window, max_steps=3)
    other = replace(chain.window, source_cycle=6, last_applied_cycle=6)
    with pytest.raises(WindowSourceUnavailable):
        find_window_source_result(chain.read, chain.top, other)
    # a chain whose applying log lacks the result stops at the window's start
    applied_elsewhere = _chain(replace(result, competition_id="other"), depth=2)
    target = replace(applied_elsewhere.window, source_competition_id="sn85-compression-001c")
    with pytest.raises(WindowSourceUnavailable):
        find_window_source_result(applied_elsewhere.read, applied_elsewhere.top, target)


def test_the_finalizer_upgrades_an_active_legacy_window_only_under_schema_18() -> None:
    result = _result()
    legacy = _legacy_window(result)
    now = T0 + timedelta(hours=3)
    v3 = EpochFinalizer(CONFIG, scorer_version="vidaio-scorer/1+test", schema_version=18)
    assert v3.upgrades_legacy_windows
    with pytest.raises(EpochLogInvalid):
        v3._upgrade_carried_window(legacy, None, now)
    with pytest.raises(EpochLogInvalid):
        v3._upgrade_carried_window(legacy, replace(result, cycle=9), now)
    upgraded = v3._upgrade_carried_window(legacy, result, now)
    assert upgraded.podium_hotkeys == ("c9", "c12", "c13", "c14", "c6")
    # expired windows pay nothing and are left as they are, with or without a source
    expired = legacy.ends_at + timedelta(minutes=1)
    assert v3._upgrade_carried_window(legacy, None, expired) is legacy
    v2 = EpochFinalizer(CONFIG, scorer_version="vidaio-scorer/1+test", schema_version=17)
    assert not v2.upgrades_legacy_windows
    assert v2._upgrade_carried_window(legacy, None, now) is legacy
