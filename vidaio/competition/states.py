"""Competition phase state machine — spec: the design spec §04.

One competition runs at a time (enforced by a partial unique index in SQL, see
migrations/0001_schema.sql). Time-based ticks drive the outer transitions;
pipeline-completion calls drive the inner ones. Compression and upscaling use
the same lifecycle; their item/scoring contracts differ behind the phase guards.

The transition table below is the single source of truth for allowed edges and
their guard names; the engine refuses anything not listed here.
"""

from __future__ import annotations

from enum import StrEnum


class Phase(StrEnum):
    SCHEDULED = "SCHEDULED"
    ENROLLING = "ENROLLING"
    FINALIZING_SUBMISSIONS = "FINALIZING_SUBMISSIONS"
    VALIDATING = "VALIDATING"
    BUILDING = "BUILDING"
    EVALUATING = "EVALUATING"
    SCORING = "SCORING"
    AWAITING_END_TIME = "AWAITING_END_TIME"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


#: Phases that count as "running" for the single-running-competition invariant.
#: SCHEDULED is not running (many competitions may be queued); terminal phases free the slot.
RUNNING_PHASES: frozenset[Phase] = frozenset(
    {
        Phase.ENROLLING,
        Phase.FINALIZING_SUBMISSIONS,
        Phase.VALIDATING,
        Phase.BUILDING,
        Phase.EVALUATING,
        Phase.SCORING,
        Phase.AWAITING_END_TIME,
    }
)

TERMINAL_PHASES: frozenset[Phase] = frozenset({Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED})

#: Allowed edges, exactly per the spec §04 state diagram: (from, to) -> guard name.
TRANSITIONS: dict[tuple[Phase, Phase], str] = {
    (Phase.SCHEDULED, Phase.ENROLLING): "start_time_commitment_anchored_and_no_other_running",
    (Phase.ENROLLING, Phase.FINALIZING_SUBMISSIONS): "finalization_time_reached",
    (Phase.FINALIZING_SUBMISSIONS, Phase.VALIDATING): "submission_backup_completed",
    (Phase.VALIDATING, Phase.BUILDING): "accepted_contender_and_no_pending_review",
    (Phase.BUILDING, Phase.EVALUATING): "at_least_one_image_built",
    (Phase.EVALUATING, Phase.SCORING): "evaluation_complete",
    (Phase.SCORING, Phase.AWAITING_END_TIME): "scores_persisted",
    (Phase.AWAITING_END_TIME, Phase.COMPLETED): "end_time_reached",
    # Failure / cancellation edges (spec diagram).
    (Phase.SCHEDULED, Phase.FAILED): "scheduling_failure",
    (Phase.VALIDATING, Phase.FAILED): "no_accepted_contender",
    (Phase.BUILDING, Phase.FAILED): "all_builds_failed",
    (Phase.ENROLLING, Phase.CANCELLED): "cancelled_during_enrollment",
}


def is_allowed(from_phase: Phase, to_phase: Phase) -> bool:
    return (from_phase, to_phase) in TRANSITIONS


def guard_name(from_phase: Phase, to_phase: Phase) -> str | None:
    return TRANSITIONS.get((from_phase, to_phase))
