"""Epoch-safe application of competition evidence.

A competition result is an *optional* input of an epoch.  The inference economy must
keep finalizing when a completed competition cannot (yet) produce complete, auditable
evidence: one missing bundle or an unregistered contender must never stop every weight
update on the subnet.

The policy is deliberately narrow:

* ``defer`` (shipping default) — the failing result is not applied in this epoch.  The
  predecessor reward window simply continues, the epoch finalizes, and the very same
  result is retried at the next epoch close because the applied-cycle cursor did not
  move.  Auditors re-derive exactly that: a log without a competition input carries its
  predecessor's window forward.  Nothing is invented and nothing is lost.
* ``hold`` — the historical fail-closed behaviour: re-raise, so the epoch HOLDs.

Deferral is loud: a CRITICAL structured log line and a counter on every occurrence.
``quarantined`` ids are operator-declared results that must never be applied (for
example a competition whose hidden media leaked); they are skipped without an error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Literal

from vidaio.competition.economic_result import derive_competition_result
from vidaio.competition.epoch_evidence import (
    CompetitionEpochEvidence,
    CompetitionEvidenceError,
)
from vidaio.tokenomics.breakthrough import resolve_reward_window
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import RewardWindowState

EvidenceFailurePolicy = Literal["defer", "hold"]

_LOG = logging.getLogger("vidaio.competition.evidence_guard")


@dataclass(frozen=True, slots=True)
class GuardedEvidence:
    """Outcome of one guarded evidence build."""

    evidence: CompetitionEpochEvidence | None
    #: ``None`` when nothing went wrong; otherwise ``"<Type>: <message>"``.
    deferred_error: str | None = None
    #: True when the selected result was skipped because the operator quarantined it.
    quarantined: bool = False


def precheck_applicable(
    config: TokenomicsConfig,
    prior: RewardWindowState | None,
    evidence: CompetitionEpochEvidence,
    *,
    now: datetime,
) -> None:
    """Run the finalizer's own competition checks BEFORE anything is published.

    The finalizer refuses a log whose competition result is inconsistent.  Running the
    identical pure checks here turns those refusals into a deferral of the result
    instead of a HOLD of the whole epoch.
    """
    result = evidence.result
    if evidence.competition_input.applied_at != now:
        raise CompetitionEvidenceError(
            "competition input applied_at must equal the epoch close-block time"
        )
    derived = derive_competition_result(
        evidence.competition_input, evidence.packet_scores
    )
    if derived != result:
        raise CompetitionEvidenceError(
            "competition result does not equal the deterministic committed-packet "
            "score derivation"
        )
    prior_state = prior or RewardWindowState()
    if (
        prior_state.last_applied_cycle is not None
        and result.cycle <= prior_state.last_applied_cycle
    ):
        raise CompetitionEvidenceError(
            f"competition cycle {result.cycle} is not newer than the last applied "
            f"reward cycle {prior_state.last_applied_cycle}"
        )
    try:
        resolve_reward_window(config, prior_state, result)
    except ValueError as exc:
        raise CompetitionEvidenceError(
            f"competition result cannot open a reward window: {exc}"
        ) from exc


def guarded_competition_evidence(
    build: Callable[[], CompetitionEpochEvidence | None],
    *,
    policy: EvidenceFailurePolicy,
    config: TokenomicsConfig,
    prior: RewardWindowState | None,
    now: datetime,
    quarantined: frozenset[str] = frozenset(),
    on_defer: Callable[[str], None] | None = None,
    logger: logging.Logger | None = None,
) -> GuardedEvidence:
    """Build + pre-validate competition evidence under the configured failure policy."""
    if policy not in ("defer", "hold"):
        raise ValueError("competition evidence failure policy must be 'defer' or 'hold'")
    log = logger or _LOG
    try:
        evidence = build()
        if evidence is None:
            return GuardedEvidence(None)
        competition_id = evidence.competition_input.competition_id
        if competition_id in quarantined:
            log.warning(
                "completed competition is operator-quarantined; its result is not applied",
                extra={"fields": {"competition_id": competition_id}},
            )
            return GuardedEvidence(None, quarantined=True)
        precheck_applicable(config, prior, evidence, now=now)
        return GuardedEvidence(evidence)
    except Exception as exc:  # noqa: BLE001 - the policy decides, never a bare pass
        if policy == "hold":
            raise
        error = f"{type(exc).__name__}: {exc}"
        log.critical(
            "competition evidence could not be applied; result DEFERRED, the epoch "
            "finalizes on the predecessor reward window and the result is retried at "
            "the next epoch close — operator investigation required",
            extra={"fields": {"error": error, "policy": policy}},
        )
        if on_defer is not None:
            on_defer(error)
        return GuardedEvidence(None, deferred_error=error)


__all__ = [
    "EvidenceFailurePolicy",
    "GuardedEvidence",
    "guarded_competition_evidence",
    "precheck_applicable",
]
