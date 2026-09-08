"""The optional alpha floor gates inference before dedup, never reward size."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from vidaio.tokenomics import (
    MinerSnapshot, RewardWindowState, TokenomicsConfig, build_weight_vector,
    ensure_locked_levers, resolve_reward_window,
)
from vidaio.tokenomics.rank_curve import dedup_excluded, eligible_for_ranking


def miner(uid, alpha, *, ip=None, coldkey=None):
    return MinerSnapshot(
        uid=uid, hotkey=f"hk{uid}", coldkey=coldkey or f"ck{uid}",
        ip=ip or f"10.0.0.{uid}", track="compression",
        accumulate_score=0.9, alpha_stake=alpha,
    )


@pytest.mark.parametrize("value", [-1, float("inf"), float("-inf"), float("nan")])
def test_floor_and_snapshot_reject_invalid_alpha(value):
    with pytest.raises(ValueError):
        TokenomicsConfig(payout_min_alpha_stake=value)
    with pytest.raises(ValueError):
        miner(1, value)


def test_default_zero_is_identical_and_stake_is_never_a_multiplier():
    config = TokenomicsConfig()
    ensure_locked_levers(config)
    assert config.payout_min_alpha_stake == 0.0
    zero = [miner(1, 0), miner(2, 0)]
    assert build_weight_vector(config, zero, burn_uid=99) == build_weight_vector(
        config, [miner(1, 1_000_000), miner(2, 2)], burn_uid=99,
    )


@pytest.mark.parametrize("identity", ["ip", "coldkey"])
def test_below_floor_low_uid_cannot_shadow_eligible_miner(identity):
    below = miner(1, 4.999)
    funded = replace(miner(2, 5), **{identity: getattr(below, identity)})
    candidates = [funded, below]
    assert eligible_for_ranking(candidates, payout_min_alpha_stake=5) == [funded]
    assert dedup_excluded(candidates, payout_min_alpha_stake=5) == set()
    config = TokenomicsConfig(payout_min_alpha_stake=5)
    ensure_locked_levers(config)
    assert build_weight_vector(config, candidates, burn_uid=99) == pytest.approx(
        {1: 0, 2: 0.64, 99: 0.36}
    )


def test_empty_stake_eligible_pool_is_withheld():
    assert build_weight_vector(
        TokenomicsConfig(payout_min_alpha_stake=5), [miner(1, 4)], burn_uid=99,
    ) == {1: 0, 99: 1}


def test_floor_leaves_competition_podium_unchanged(live_cfg, mk_result, mk_podium_miners):
    config = live_cfg.model_copy(update={"payout_min_alpha_stake": 5})
    state = resolve_reward_window(
        config, RewardWindowState(), mk_result(scores=(0.51, 0.50, 0.49)),
    )
    candidates = [miner(1, 10)] + mk_podium_miners(100, 101, 102)
    expected = build_weight_vector(
        live_cfg, candidates, burn_uid=99, reward_state=state,
        now=datetime(2026, 8, 20, 12, tzinfo=UTC),
    )
    actual = build_weight_vector(
        config, candidates, burn_uid=99, reward_state=state,
        now=datetime(2026, 8, 20, 12, tzinfo=UTC),
    )
    assert actual == expected
    assert [actual[uid] for uid in (100, 101, 102)] == pytest.approx([0.28, 0.08, 0.04])
