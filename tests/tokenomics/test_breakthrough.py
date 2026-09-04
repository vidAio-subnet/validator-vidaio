from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from vidaio.tokenomics import (
    ContenderResult,
    EmissionShares,
    EmissionState,
    RewardWindowState,
    TokenomicsConfig,
    active_emission_state,
    contender_margin,
    emission_shares,
    podium_hotkey_shares,
    qualifies_for_crown,
    resolve_reward_window,
    window_active,
    winner,
)

T0 = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)
BASELINE_DIGEST = "ab" * 32


def test_score_relative_margin_and_undefined_baseline() -> None:
    assert contender_margin(0.5, 0.525) == pytest.approx(0.05)
    assert contender_margin(0.5, 0.45) == pytest.approx(-0.10)
    assert contender_margin(None, 0.9) is None
    assert contender_margin(0.0, 0.9) is None


def test_crown_floor_is_inclusive_and_decimal_stable(cfg) -> None:
    assert not qualifies_for_crown(cfg, 0.5, 0.524999999999)
    assert qualifies_for_crown(cfg, 0.5, 0.525)
    assert qualifies_for_crown(cfg, 0.5, 0.525000000001)
    # Avoid the classic binary intermediate where 0.1 * 1.05 is not exact.
    assert qualifies_for_crown(cfg, 0.1, 0.105)


def test_winner_is_recomputed_from_scores_not_caller_order(mk_result) -> None:
    result = mk_result(
        contenders=(
            ContenderResult("first", 1, 0.6),
            ContenderResult("second", 2, 0.9),
        )
    )
    assert winner(result).hotkey == "second"
    assert winner(mk_result(contenders=())) is None


def test_podium_result_creates_seven_day_half_open_window(cfg, mk_result) -> None:
    state = resolve_reward_window(cfg, RewardWindowState(), mk_result(scores=(0.524,)))
    assert state.kind is EmissionState.PODIUM
    assert state.starts_at == T0
    assert state.ends_at == T0 + timedelta(hours=168)
    assert state.winner_margin == pytest.approx(0.048)
    assert window_active(state, T0)
    assert window_active(state, state.ends_at - timedelta(microseconds=1))
    assert not window_active(state, state.ends_at)
    assert active_emission_state(state, state.ends_at) is EmissionState.IDLE


def test_exact_floor_creates_crown_with_full_provenance(cfg, mk_result) -> None:
    state = resolve_reward_window(
        cfg,
        RewardWindowState(),
        mk_result(scores=(0.525, 0.52, 0.51, 0.50), track="upscaling"),
    )
    assert state.kind is EmissionState.CROWN
    assert state.podium_hotkeys == ("comp100", "comp101", "comp102")
    assert state.winner_hotkey == "comp100"
    assert state.winner_uid == 100
    assert state.baseline_version == 0
    assert state.baseline_artifact_digest == BASELINE_DIGEST
    assert state.source_competition_id == "competition-1"
    assert state.source_track == "upscaling"
    assert state.source_cycle == state.last_applied_cycle == 1


def test_mid_window_latest_result_globally_replaces_and_can_downgrade(
    cfg, mk_result
) -> None:
    crown = resolve_reward_window(
        cfg, RewardWindowState(), mk_result(cycle=1, scores=(0.8,))
    )
    replacement_at = T0 + timedelta(hours=12)
    podium = resolve_reward_window(
        cfg,
        crown,
        mk_result(
            cycle=2,
            applied_at=replacement_at,
            scores=(0.51,),
            start_uid=200,
            track="upscaling",
        ),
    )
    assert podium.kind is EmissionState.PODIUM
    assert podium.starts_at == replacement_at
    assert podium.ends_at == replacement_at + timedelta(hours=168)
    assert podium.winner_hotkey == "comp200"
    assert podium.source_track == "upscaling"


def test_mid_window_podium_can_upgrade_to_crown(cfg, mk_result) -> None:
    podium = resolve_reward_window(
        cfg, RewardWindowState(), mk_result(cycle=1, scores=(0.51,))
    )
    crown = resolve_reward_window(
        cfg,
        podium,
        mk_result(
            cycle=2, applied_at=T0 + timedelta(hours=1), scores=(0.8,), start_uid=200
        ),
    )
    assert crown.kind is EmissionState.CROWN
    assert crown.last_applied_cycle == 2


def test_successful_result_is_idempotent_and_older_cycle_cannot_replay(
    cfg, mk_result
) -> None:
    result = mk_result(cycle=3, scores=(0.8,))
    state = resolve_reward_window(cfg, RewardWindowState(), result)
    assert resolve_reward_window(cfg, state, result) == state
    assert (
        resolve_reward_window(cfg, state, mk_result(cycle=2, scores=(0.51,))) == state
    )


@pytest.mark.parametrize("baseline", [None, 0.0])
def test_baseline_failure_preserves_window_and_same_cycle_can_retry(
    cfg, mk_result, baseline
) -> None:
    prior = resolve_reward_window(
        cfg, RewardWindowState(), mk_result(cycle=1, scores=(0.51,))
    )
    attempted_at = T0 + timedelta(hours=4)
    failed = mk_result(
        cycle=2, applied_at=attempted_at, scores=(0.9,), baseline_score=baseline
    )
    assert resolve_reward_window(cfg, prior, failed) == prior
    assert prior.last_applied_cycle == 1
    repaired = mk_result(
        cycle=2, applied_at=attempted_at, scores=(0.9,), baseline_score=0.5
    )
    applied = resolve_reward_window(cfg, prior, repaired)
    assert applied.kind is EmissionState.CROWN
    assert applied.last_applied_cycle == 2


def test_no_eligible_winner_is_retryable_and_does_not_reset(cfg, mk_result) -> None:
    prior = resolve_reward_window(
        cfg, RewardWindowState(), mk_result(cycle=1, scores=(0.51,))
    )
    assert resolve_reward_window(cfg, prior, mk_result(cycle=2, contenders=())) == prior


def test_newer_cycle_cannot_backdate_application(cfg, mk_result) -> None:
    prior = resolve_reward_window(cfg, RewardWindowState(), mk_result(cycle=1))
    with pytest.raises(ValueError, match="regress applied_at"):
        resolve_reward_window(
            cfg,
            prior,
            mk_result(cycle=2, applied_at=T0 - timedelta(seconds=1), scores=(0.8,)),
        )


def test_testnet_window_override_is_part_of_pure_fold(mk_result) -> None:
    cfg = TokenomicsConfig(result_window_hours=0.25)
    state = resolve_reward_window(cfg, RewardWindowState(), mk_result(scores=(0.8,)))
    assert state.ends_at == T0 + timedelta(minutes=15)


def test_emission_share_table_and_disabled_flag(live_cfg, mk_result) -> None:
    idle = RewardWindowState()
    assert emission_shares(live_cfg, idle, T0) == EmissionShares(0.8, 0.0, 0.2)
    podium = resolve_reward_window(live_cfg, idle, mk_result(scores=(0.51,)))
    assert emission_shares(live_cfg, podium, T0) == EmissionShares(0.6, 0.4, 0.0)
    crown = resolve_reward_window(live_cfg, idle, mk_result(scores=(0.8,)))
    assert emission_shares(live_cfg, crown, T0) == EmissionShares(0.1, 0.9, 0.0)
    assert emission_shares(TokenomicsConfig(), crown, T0) == EmissionShares(
        0.8, 0.0, 0.2
    )


def test_podium_shares_leave_missing_ranks_unallocated(cfg, mk_result) -> None:
    one = resolve_reward_window(cfg, RewardWindowState(), mk_result(scores=(0.51,)))
    assert podium_hotkey_shares(one) == {"comp100": 0.70}
