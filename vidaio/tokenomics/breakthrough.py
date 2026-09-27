"""Pure schema-v15 competition-window state machine.

Auditors verify packets under the protocol's boundary hysteresis, then authority and
auditor both derive economics from the same committed score representation here. Local
CPU float drift therefore cannot select a different side of the crown boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal

from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import (
    MAX_PODIUM_PLACES,
    CompetitionResult,
    CompetitionRules,
    ContenderResult,
    EmissionShares,
    EmissionState,
    RewardWindowState,
)

#: The pre-v3 fixed split (three places, 70/20/10). Kept for windows folded before
#: schema v18, which carry no payout policy of their own.
LEGACY_PODIUM_SPLIT = (0.70, 0.20, 0.10)
PODIUM_SPLIT = LEGACY_PODIUM_SPLIT


@dataclass(frozen=True)
class PayoutPolicy:
    """The payout policy in force for one window: per-competition rules override the
    protocol defaults field by field, and the result is what the window carries."""

    competition_share: float
    split: tuple[float, ...]
    redistribute_empty_places: bool
    no_qualifier_closes_window: bool


def effective_payout_policy(
    config: TokenomicsConfig, rules: CompetitionRules | None, kind: EmissionState
) -> PayoutPolicy:
    crown = kind is EmissionState.CROWN
    share = config.crown_competition_share if crown else config.podium_competition_share
    split = tuple(config.crown_split if crown else config.podium_split)
    redistribute = config.redistribute_empty_places
    closes = config.no_qualifier_closes_window
    if rules is not None:
        override_share = rules.crown_competition_share if crown else rules.podium_competition_share
        if override_share is not None:
            share = float(override_share)
        override_split = rules.crown_split if crown else rules.podium_split
        if override_split is not None:
            split = tuple(override_split)
        if rules.redistribute_empty_places is not None:
            redistribute = rules.redistribute_empty_places
        if rules.no_qualifier_closes_window is not None:
            closes = rules.no_qualifier_closes_window
    return PayoutPolicy(
        competition_share=share, split=split[:MAX_PODIUM_PLACES],
        redistribute_empty_places=redistribute, no_qualifier_closes_window=closes,
    )


def place_shares(split: tuple[float, ...], filled: int, redistribute: bool) -> tuple[float, ...]:
    """Fractions of the competition pot for ``filled`` paid places (Decimal-exact)."""
    if filled <= 0:
        return ()
    head = [Decimal(str(x)) for x in split[:filled]]
    if redistribute:
        total = sum(head)
        if total > 0:
            head = [x / total for x in head]
    return tuple(float(x) for x in head)


def contender_margin(
    baseline_score: float | None, contender_score: float | None
) -> float | None:
    """Score-relative improvement over the archived executable baseline.

    A baseline that was MEASURED at exactly zero (every one of its outputs failed the
    quality gate) has no relative improvement: the recorded margin is then the
    contender's absolute score, a finite and re-derivable number. Such a result is only
    ever payable under anchored absolute score bars (see :func:`zero_baseline_payable`).
    """
    if (
        baseline_score is None
        or contender_score is None
        or not math.isfinite(baseline_score)
        or not math.isfinite(contender_score)
        or baseline_score < 0.0
    ):
        return None
    if baseline_score == 0.0:
        return float(Decimal(str(contender_score)))
    baseline = Decimal(str(baseline_score))
    score = Decimal(str(contender_score))
    return float((score - baseline) / baseline)


def zero_baseline_payable(rules: CompetitionRules | None) -> bool:
    """True when a competition's anchored rules carry an absolute score bar.

    The relative margins are meaningless against a baseline that scored zero, so a
    zero-baseline result stays the historical retryable no-op UNLESS the manifest
    anchored ``crown_min_score`` and/or ``podium_min_score`` before enrollment. Those
    absolute bars then decide on their own; nothing is ever granted by default.
    """
    return rules is not None and (
        rules.crown_min_score is not None or rules.podium_min_score is not None
    )


def qualifies_for_crown(
    config: TokenomicsConfig,
    baseline_score: float | None,
    contender_score: float | None,
    rules: CompetitionRules | None = None,
) -> bool:
    """Inclusive crown test using canonical decimal spellings, with no threshold drift.

    Without ``rules`` the protocol default margin applies. With manifest-anchored
    ``rules`` the competition's own margin replaces it and the optional absolute
    ``crown_min_score`` must also be reached. Against a baseline measured at zero a
    relative margin cannot discriminate, so a crown then REQUIRES an anchored
    ``crown_min_score`` and that bar decides alone.
    """
    if (
        baseline_score is None
        or contender_score is None
        or not math.isfinite(baseline_score)
        or not math.isfinite(contender_score)
        or baseline_score < 0.0
    ):
        return False
    score = Decimal(str(contender_score))
    if baseline_score == 0.0:
        if rules is None or rules.crown_min_score is None:
            return False
        return score >= Decimal(str(rules.crown_min_score))
    baseline = Decimal(str(baseline_score))
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
    """Inclusive paid-rank test. No rules (or no podium conditions) = every contender.

    Against a baseline measured at zero the relative ``podium_min_margin`` cannot
    discriminate; a paid rank then REQUIRES an anchored ``podium_min_score``.
    """
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
            or baseline_score < 0.0
        ):
            return False
        if baseline_score == 0.0:
            return rules.podium_min_score is not None
        baseline = Decimal(str(baseline_score))
        if score < baseline * (Decimal(1) + Decimal(str(rules.podium_min_margin))):
            return False
    return True


def decision_baseline(result: CompetitionResult) -> float | None:
    """The baseline score the crown/podium rules compare against.

    A competition anchored without an executable baseline is decided exactly like a
    baseline measured at zero: the relative margins cannot discriminate and only the
    anchored absolute bars (``crown_min_score`` / ``podium_min_score``) decide.
    """
    return 0.0 if not result.has_baseline else result.baseline_score


def result_kind(
    config: TokenomicsConfig, result: CompetitionResult
) -> EmissionState | None:
    """CROWN or PODIUM for a result that opens a window, else None (same rules as
    :func:`resolve_reward_window`, without the cycle/prior bookkeeping)."""
    baseline = decision_baseline(result)
    best = winner(result)
    if baseline is None or baseline < 0.0 or best is None:
        return None
    if baseline == 0.0 and not zero_baseline_payable(result.rules):
        return None
    if qualifies_for_crown(config, baseline, best.score, result.rules):
        return EmissionState.CROWN
    return EmissionState.PODIUM


def podium_contenders(result: CompetitionResult) -> tuple[ContenderResult, ...]:
    """Ranked contenders that meet the competition's anchored podium conditions.

    Both conditions are monotone in the score, so the qualifying set is always a prefix
    of the score-ordered result: a non-qualifying contender can never sit above a
    qualifying one, and an unfilled place is never handed to somebody below the bar.
    """
    return tuple(
        contender
        for contender in result.contenders
        if qualifies_for_podium(result.rules, decision_baseline(result), contender.score)
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

    A failed baseline or absent winner is a retryable no-op. It neither erases the prior
    window nor consumes the cycle, so completed audit evidence for the same cycle may
    apply later. A baseline MEASURED at zero is the same no-op unless the competition
    anchored absolute score bars (:func:`zero_baseline_payable`); with them the result
    applies and those bars decide. Successfully applied cycles are replay-safe.
    """
    if result is None:
        return prior
    if (
        prior.last_applied_cycle is not None
        and result.cycle <= prior.last_applied_cycle
    ):
        return prior
    best = winner(result)
    baseline = decision_baseline(result)
    if baseline is None or baseline < 0.0:
        return prior
    if best is None:
        # Nobody met the anchored podium conditions. Tokenomics v3: the result
        # CLOSES the running window (the subnet burns until the next result);
        # the pre-v3 behaviour left the previous window untouched.
        closes = effective_payout_policy(
            config, result.rules, EmissionState.PODIUM
        ).no_qualifier_closes_window
        return RewardWindowState() if closes else prior
    if baseline == 0.0 and not zero_baseline_payable(result.rules):
        return prior
    if prior.starts_at is not None and result.applied_at < prior.starts_at:
        raise ValueError("a newer competition cycle cannot regress applied_at")

    margin = contender_margin(baseline, best.score)
    if margin is None:  # fail-closed defensive seam
        return prior
    kind = (
        EmissionState.CROWN
        if qualifies_for_crown(config, baseline, best.score, result.rules)
        else EmissionState.PODIUM
    )
    policy = effective_payout_policy(config, result.rules, kind)
    paid = tuple(c.hotkey for c in podium_contenders(result)[: len(policy.split)])
    return RewardWindowState(
        kind=kind,
        starts_at=result.applied_at,
        ends_at=result.applied_at + timedelta(hours=config.result_window_hours),
        podium_hotkeys=paid,
        competition_share=policy.competition_share,
        place_shares=place_shares(policy.split, len(paid), policy.redistribute_empty_places),
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


def is_legacy_window(state: RewardWindowState) -> bool:
    """A CROWN/PODIUM window folded before schema v18: it carries no payout policy.

    Such a window names at most the three legacy podium hotkeys and no per-place
    shares, so it cannot express the policy the tokenomics-v3 protocol promises for it.
    """
    return (
        state.kind in (EmissionState.CROWN, EmissionState.PODIUM)
        and state.competition_share is None
        and not state.place_shares
    )


def upgrade_legacy_window(
    config: TokenomicsConfig,
    state: RewardWindowState,
    source: CompetitionResult,
) -> RewardWindowState:
    """Re-resolve a legacy window's paid places and policy from its source result.

    The first schema-v18 fold of a still-active legacy window replaces its three
    legacy podium hotkeys with the places the v3 payout policy assigns from the SAME
    committed result, and records that policy. Everything else stays: the kind (a
    CROWN stays a CROWN), the chain-time interval, the winner and the provenance.
    ``source`` is the ``competition_result`` committed in the epoch log that applied
    the window; any mismatch with the window's provenance is refused, never guessed.
    A window that is not legacy is returned unchanged, so the upgrade happens once.
    """
    if not is_legacy_window(state):
        return state
    if (
        source.competition_id != state.source_competition_id
        or source.cycle != state.source_cycle
        or source.track != state.source_track
        or source.applied_at != state.starts_at
    ):
        raise ValueError(
            "legacy reward window source result does not match the window provenance "
            f"({source.competition_id!r}, cycle {source.cycle}) vs "
            f"({state.source_competition_id!r}, cycle {state.source_cycle})"
        )
    best = winner(source)
    if best is None or best.hotkey != state.winner_hotkey or best.uid != state.winner_uid:
        raise ValueError(
            "legacy reward window winner differs from the winner of its source result"
        )
    policy = effective_payout_policy(config, source.rules, state.kind)
    paid = tuple(c.hotkey for c in podium_contenders(source)[: len(policy.split)])
    return replace(
        state,
        podium_hotkeys=paid,
        competition_share=policy.competition_share,
        place_shares=place_shares(policy.split, len(paid), policy.redistribute_empty_places),
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
    if active in (EmissionState.CROWN, EmissionState.PODIUM):
        crown = active is EmissionState.CROWN
        competition = (
            config.crown_competition_share if crown else config.podium_competition_share
        )
        if state.competition_share is not None:
            competition = float(state.competition_share)
        inference = config.crown_inference_share if crown else config.podium_inference_share
        inference = min(inference, max(0.0, 1.0 - competition))
        burn = float(Decimal(1) - Decimal(str(inference)) - Decimal(str(competition)))
        return EmissionShares(inference, competition, max(0.0, burn))
    return EmissionShares(config.idle_inference_share, 0.0, config.idle_burn_share)


def podium_hotkey_shares(
    state: RewardWindowState, config: TokenomicsConfig | None = None
) -> dict[str, float]:
    """Payable fractions of the competition pot per hotkey.

    A window folded under tokenomics v3 carries its own ``place_shares``. An older
    window (no policy) is paid with the live protocol defaults, redistributed over
    its places, or with the legacy 70/20/10 split when no config is given.
    """
    if state.place_shares:
        return dict(zip(state.podium_hotkeys, state.place_shares))
    if config is None:
        return {h: s for h, s in zip(state.podium_hotkeys, LEGACY_PODIUM_SPLIT)}
    policy = effective_payout_policy(config, None, state.kind)
    shares = place_shares(policy.split, len(state.podium_hotkeys), policy.redistribute_empty_places)
    return dict(zip(state.podium_hotkeys, shares))
