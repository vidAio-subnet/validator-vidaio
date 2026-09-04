from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from vidaio.tokenomics import (
    MinerSnapshot,
    RewardWindowState,
    TokenomicsConfig,
    build_weight_vector,
    ensure_locked_levers,
    quantize_u16,
    resolve_reward_window,
)
from vidaio.tokenomics.quantize import U16_MAX

BURN_UID = 94
T0 = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)


def _two_track_inference(mk_miner):
    return [
        mk_miner(1, score=0.9, track="compression"),
        mk_miner(2, score=0.9, track="upscaling"),
    ]


def test_idle_is_80_inference_20_canonical_sink(cfg, mk_miner) -> None:
    vector = build_weight_vector(cfg, _two_track_inference(mk_miner), burn_uid=BURN_UID)
    assert vector == pytest.approx({1: 0.64, 2: 0.16, BURN_UID: 0.20})
    assert sum(vector.values()) == pytest.approx(1.0)


def test_podium_is_60_inference_40_ranked_70_20_10(
    live_cfg, mk_miner, mk_result, mk_podium_miners
) -> None:
    state = resolve_reward_window(
        live_cfg, RewardWindowState(), mk_result(scores=(0.51, 0.50, 0.49))
    )
    miners = _two_track_inference(mk_miner) + mk_podium_miners(100, 101, 102)
    vector = build_weight_vector(
        live_cfg, miners, burn_uid=BURN_UID, reward_state=state, now=T0
    )
    assert vector == pytest.approx({1: 0.48, 2: 0.12, 100: 0.28, 101: 0.08, 102: 0.04})
    assert BURN_UID not in vector


def test_crown_is_10_inference_90_ranked_70_20_10(
    live_cfg, mk_miner, mk_result, mk_podium_miners
) -> None:
    state = resolve_reward_window(
        live_cfg, RewardWindowState(), mk_result(scores=(0.8, 0.7, 0.6))
    )
    miners = _two_track_inference(mk_miner) + mk_podium_miners(100, 101, 102)
    vector = build_weight_vector(
        live_cfg, miners, burn_uid=BURN_UID, reward_state=state, now=T0
    )
    assert vector == pytest.approx({1: 0.08, 2: 0.02, 100: 0.63, 101: 0.18, 102: 0.09})
    assert BURN_UID not in vector


@pytest.mark.parametrize(
    ("winner_score", "winner_share", "burn_share"),
    [(0.51, 0.28, 0.12), (0.80, 0.63, 0.27)],
)
def test_missing_podium_ranks_go_to_sink(
    live_cfg,
    mk_miner,
    mk_result,
    mk_podium_miners,
    winner_score,
    winner_share,
    burn_share,
) -> None:
    state = resolve_reward_window(
        live_cfg, RewardWindowState(), mk_result(scores=(winner_score,))
    )
    miners = _two_track_inference(mk_miner) + mk_podium_miners(100)
    vector = build_weight_vector(
        live_cfg, miners, burn_uid=BURN_UID, reward_state=state, now=T0
    )
    assert vector[100] == pytest.approx(winner_share)
    assert vector[BURN_UID] == pytest.approx(burn_share)
    assert sum(vector.values()) == pytest.approx(1.0)


def test_deregistered_podium_rank_goes_to_sink(
    live_cfg, mk_miner, mk_result, mk_podium_miners
) -> None:
    state = resolve_reward_window(
        live_cfg, RewardWindowState(), mk_result(scores=(0.51, 0.50, 0.49))
    )
    miners = _two_track_inference(mk_miner) + mk_podium_miners(100, 102)
    vector = build_weight_vector(
        live_cfg, miners, burn_uid=BURN_UID, reward_state=state, now=T0
    )
    assert vector[100] == pytest.approx(0.28)
    assert vector[102] == pytest.approx(0.04)
    assert vector[BURN_UID] == pytest.approx(0.08)


def test_exact_window_end_reverts_to_idle(live_cfg, mk_miner, mk_result) -> None:
    state = resolve_reward_window(
        live_cfg, RewardWindowState(), mk_result(scores=(0.8,))
    )
    vector = build_weight_vector(
        live_cfg,
        _two_track_inference(mk_miner),
        burn_uid=BURN_UID,
        reward_state=state,
        now=T0 + timedelta(hours=168),
    )
    assert vector == pytest.approx({1: 0.64, 2: 0.16, BURN_UID: 0.20})


def test_disabled_competition_forces_idle_even_with_crown(
    cfg, mk_miner, mk_result
) -> None:
    state = resolve_reward_window(cfg, RewardWindowState(), mk_result(scores=(0.8,)))
    vector = build_weight_vector(
        cfg, _two_track_inference(mk_miner), burn_uid=BURN_UID, reward_state=state
    )
    assert vector == pytest.approx({1: 0.64, 2: 0.16, BURN_UID: 0.20})


def test_no_eligible_miners_burns_everything(cfg, mk_miner) -> None:
    miners = [mk_miner(1, excluded=True), mk_miner(2, score=-1.0)]
    assert build_weight_vector(cfg, miners, burn_uid=BURN_UID) == {
        1: 0.0,
        2: 0.0,
        BURN_UID: 1.0,
    }


def test_without_sink_unavailable_shares_fail_closed(cfg, mk_miner) -> None:
    with pytest.raises(ValueError, match="canonical burn_uid is required"):
        build_weight_vector(cfg, _two_track_inference(mk_miner))


def test_burn_uid_cannot_overlap_snapshot(cfg, mk_miner) -> None:
    with pytest.raises(ValueError, match="overlaps"):
        build_weight_vector(cfg, [mk_miner(1)], burn_uid=1)


def test_empty_burn_quantizes_to_full_grid(cfg) -> None:
    vector = build_weight_vector(cfg, [], burn_uid=BURN_UID)
    assert vector == {BURN_UID: 1.0}
    assert quantize_u16(vector) == {BURN_UID: U16_MAX}


@pytest.mark.parametrize(
    "bad",
    [
        TokenomicsConfig(burn_proportion=0.05),
        TokenomicsConfig(alpha_stake_weigh_factor=0.5),
        TokenomicsConfig(retention_full_window_required=False),
        TokenomicsConfig(top_n_per_track=4),
        TokenomicsConfig(minimum_payout_score=0.2),
        TokenomicsConfig(track_weights={"compression": 0.5, "upscaling": 0.5}),
        TokenomicsConfig(empty_pool_policy="redistribute"),
    ],
)
def test_locked_levers_fail_closed(bad) -> None:
    with pytest.raises(ValueError):
        ensure_locked_levers(bad)


def test_input_order_does_not_change_output(
    live_cfg, mk_miner, mk_result, mk_podium_miners
) -> None:
    state = resolve_reward_window(
        live_cfg, RewardWindowState(), mk_result(scores=(0.8, 0.7, 0.6))
    )
    rng = random.Random(85)
    miners: list[MinerSnapshot] = [
        mk_miner(
            uid,
            score=0.2 + uid / 100,
            track=("compression" if uid % 2 else "upscaling"),
        )
        for uid in range(1, 12)
    ] + mk_podium_miners(100, 101, 102)
    expected = build_weight_vector(
        live_cfg, miners, burn_uid=BURN_UID, reward_state=state, now=T0
    )
    for _ in range(20):
        rng.shuffle(miners)
        assert (
            build_weight_vector(
                live_cfg, miners, burn_uid=BURN_UID, reward_state=state, now=T0
            )
            == expected
        )


def test_live_windows_require_explicit_chain_time(live_cfg, mk_result) -> None:
    state = resolve_reward_window(
        live_cfg, RewardWindowState(), mk_result(scores=(0.8,))
    )
    with pytest.raises(ValueError, match="now is required"):
        build_weight_vector(live_cfg, [], reward_state=state)
