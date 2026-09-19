"""Pure schema-v15 competition-window state machine.

Auditors verify packets under the protocol's boundary hysteresis, then authority and
auditor both derive economics from the same committed score representation here. Local
CPU float drift therefore cannot select a different side of the crown boundary.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from decimal import Decimal

from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import (
    CompetitionResult,
    CompetitionRules,
    ContenderResult,
    EmissionShares,
    EmissionState,
    RewardWindowState,
)

PODIUM_SPLIT = (0.70, 0.20, 0.10)


def contender_margin(
    baseline_score: float | None, contender_score: float | None
) -> float | None:
    """Score-relative improvement over the archived executable baseline."""
    if (
        baseline_score is None
        or contender_score is None
        or not math.isfinite(baseline_score)
        or not math.isfinite(contender_score)
        or baseline_score <= 0.0
    ):
        return None
    baseline = Decimal(str(baseline_score))
    score = Decimal(str(contender_score))
    return float((score - baseline) / baseline)


def qualifies_for_crown(
    config: TokenomicsConfig,
    baseline_score: float | None,
    contender_score: float | None,
    rules: CompetitionRules | None = None,
) -> bool:
    """Inclusive crown test using canonical decimal spellings, with no threshold drift.

    Without ``rules`` the protocol default margin applies. With manifest-anchored
    ``rules`` the competition's own margin replaces it and the optional absolute
    ``crown_min_score`` must also be reached.
    """
    if (
        baseline_score is None
        or contender_score is None
        or not math.isfinite(baseline_score)
        or not math.isfinite(contender_score)
        or baseline_score <= 0.0
    ):
        return False
    baseline = Decimal(str(baseline_score))
    score = Decimal(str(contender_score))
    floor = Decimal(
        str(config.breakthrough_margin_floor if rules is None else rules.crown_margin)
    )
    if rules is not None and rules.crown_min_score is not None:
        if score < Decimal(str(rules.crown_min_score)):
            return False
    return score >= baseline * (Decimal(1) + floor)


def qualifies_for_podium(
    rules: CompetitionRules | None,
    baseline_score: float | None,
    contender_score: float | None,
) -> bool:
    """Inclusive paid-rank test. No rules (or no podium conditions) = every contender."""
    if rules is None or (
        rules.podium_min_margin is None and rules.podium_min_score is None
    ):
        return True
    if contender_score is None or not math.isfinite(contender_score):
        return False
    score = Decimal(str(contender_score))
    if rules.podium_min_score is not None and score < Decimal(
        str(rules.podium_min_score)
    ):
        return False
    if rules.podium_min_margin is not None:
        if (
            baseline_score is None
            or not math.isfinite(baseline_score)
            or baseline_score <= 0.0
        ):
            return False
        baseline = Decimal(str(baseline_score))
        if score < baseline * (Decimal(1) + Decimal(str(rules.podium_min_margin))):
            return False
    return True


def podium_contenders(result: CompetitionResult) -> tuple[ContenderResult, ...]:
    """Ranked contenders that meet the competition's anchored podium conditions.

    Both conditions are monotone in the score, so the qualifying set is always a prefix
    of the score-ordered result: a non-qualifying contender can never sit above a
    qualifying one, and an unfilled place is never handed to somebody below the bar.
    """
    return tuple(
        contender
        for contender in result.contenders
        if qualifies_for_podium(result.rules, result.baseline_score, contender.score)
    )


def winner(result: CompetitionResult) -> ContenderResult | None:
    qualified = podium_contenders(result)
    return qualified[0] if qualified else None


def resolve_reward_window(
    config: TokenomicsConfig,
    prior: RewardWindowState,
    result: CompetitionResult | None,
) -> RewardWindowState:
    """Fold a valid result; newer results globally replace and restart the window.

    A failed/non-positive baseline or absent winner is a retryable no-op. It neither
    erases the prior window nor consumes the cycle, so completed audit evidence for the
    same cycle may apply later. Successfully applied cycles are replay-safe.
    """
    if result is None:
        return prior
    if (
        prior.last_applied_cycle is not None
        and result.cycle <= prior.last_applied_cycle
    ):
        return prior
    best = winner(result)
    if result.baseline_score is None or result.baseline_score <= 0.0 or best is None:
        return prior
    if prior.starts_at is not None and result.applied_at < prior.starts_at:
        raise ValueError("a newer competition cycle cannot regress applied_at")

    margin = contender_margin(result.baseline_score, best.score)
    if margin is None:  # fail-closed defensive seam
        return prior
    kind = (
        EmissionState.CROWN
        if qualifies_for_crown(
            config, result.baseline_score, best.score, result.rules
        )
        else EmissionState.PODIUM
    )
    return RewardWindowState(
        kind=kind,
        starts_at=result.applied_at,
        ends_at=result.applied_at + timedelta(hours=config.result_window_hours),
        podium_hotkeys=tuple(
            c.hotkey for c in podium_contenders(result)[: len(PODIUM_SPLIT)]
        ),
        winner_hotkey=best.hotkey,
        winner_uid=best.uid,
        winner_score=best.score,
        winner_margin=margin,
        baseline_score=result.baseline_score,
        baseline_version=result.baseline_version,
        baseline_artifact_digest=result.baseline_artifact_digest,
        source_competition_id=result.competition_id,
        source_track=result.track,
        source_cycle=result.cycle,
        last_applied_cycle=result.cycle,
    )


def window_active(state: RewardWindowState, now: datetime) -> bool:
    """True exactly inside the chain-time interval [starts_at, ends_at)."""
    if state.kind is EmissionState.IDLE:
        return False
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("composition time must be timezone-aware")
    assert state.starts_at is not None and state.ends_at is not None
    return state.starts_at <= now < state.ends_at


def active_emission_state(state: RewardWindowState, now: datetime) -> EmissionState:
    return state.kind if window_active(state, now) else EmissionState.IDLE


def emission_shares(
    config: TokenomicsConfig,
    state: RewardWindowState,
    now: datetime,
) -> EmissionShares:
    active = (
        active_emission_state(state, now)
        if config.competition_emissions_enabled
        else EmissionState.IDLE
    )
    if active is EmissionState.CROWN:
        return EmissionShares(
            config.crown_inference_share, config.crown_competition_share, 0.0
        )
    if active is EmissionState.PODIUM:
        return EmissionShares(
            config.podium_inference_share, config.podium_competition_share, 0.0
        )
    return EmissionShares(config.idle_inference_share, 0.0, config.idle_burn_share)


def podium_hotkey_shares(state: RewardWindowState) -> dict[str, float]:
    """Payable fractions; absent ranks deliberately remain unallocated."""
    return {hotkey: share for hotkey, share in zip(state.podium_hotkeys, PODIUM_SPLIT)}
