"""Pure derivation of auditable competition economics from committed packet scores.

This module deliberately performs no I/O and trusts no stored aggregate.  Its only
score input is the top-level score independently recovered for every packet digest
committed by :class:`vidaio.epoch.log.CompetitionInput`.  Exact digest coverage and
the full subject-by-item matrix are prerequisites; malformed or partial evidence
cannot produce an earning result.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from vidaio.epoch.log import CompetitionInput
from vidaio.tokenomics.rank_curve import dedup_ip_key
from vidaio.tokenomics.state import CompetitionResult, ContenderResult


class CompetitionEconomicResultError(ValueError):
    """Committed competition scores cannot derive one canonical economic result."""


@dataclass(frozen=True, slots=True)
class SubjectAggregate:
    """Arithmetic mean independently derived for one committed audit subject."""

    subject_id: str
    role: str
    uid: int | None
    hotkey: str | None
    score: float


@dataclass(frozen=True, slots=True)
class CompetitionEconomicDerivation:
    """Canonical tokenomics result plus the means used to construct it."""

    result: CompetitionResult
    subject_aggregates: tuple[SubjectAggregate, ...]
    baseline_score: float

    def aggregate_by_subject_id(self) -> dict[str, float]:
        """Return a fresh subject-id -> mean view for logs/tests/diagnostics."""
        return {
            aggregate.subject_id: aggregate.score
            for aggregate in self.subject_aggregates
        }


@dataclass(frozen=True, slots=True)
class CompetitionDedupCandidate:
    """Close-census and exact-output identity for one payable contender."""

    subject_id: str
    uid: int
    coldkey: str
    ip: str
    output_digests: tuple[str, ...]


def competition_dedup_losers(
    candidates: tuple[CompetitionDedupCandidate, ...],
) -> frozenset[str]:
    """Lowest uid wins each advertised-IP, coldkey, and exact-output slot.

    The scan deliberately mirrors inference dedup. Unspecified addresses are not a
    shared identity, while a byte-identical full output matrix is: later contenders
    cannot occupy a second podium/crown slot with the same solution output.
    """
    seen_ips: set[str] = set()
    seen_coldkeys: set[str] = set()
    seen_outputs: set[tuple[str, ...]] = set()
    losers: set[str] = set()
    for candidate in sorted(candidates, key=lambda value: value.uid):
        if not candidate.output_digests:
            raise CompetitionEconomicResultError(
                f"competition contender {candidate.subject_id!r} has no output matrix"
            )
        ip_key = dedup_ip_key(candidate.ip)
        if (
            (ip_key is not None and ip_key in seen_ips)
            or candidate.coldkey in seen_coldkeys
            or candidate.output_digests in seen_outputs
        ):
            losers.add(candidate.subject_id)
            continue
        if ip_key is not None:
            seen_ips.add(ip_key)
        seen_coldkeys.add(candidate.coldkey)
        seen_outputs.add(candidate.output_digests)
    return frozenset(losers)


def _required_packet_digests(competition: CompetitionInput) -> tuple[str, ...]:
    expected_items = len(competition.items)
    digests: list[str] = []
    for subject in competition.subjects:
        observed = len(subject.packet_digests)
        if observed != expected_items:
            raise CompetitionEconomicResultError(
                f"competition subject {subject.subject_id!r} has {observed} score "
                f"packet(s), expected exactly {expected_items} (one per audit item)"
            )
        digests.extend(subject.packet_digests)
    if len(set(digests)) != len(digests):
        raise CompetitionEconomicResultError(
            "competition packet digests must be unique across audit subjects"
        )
    return tuple(digests)


def _validated_score(digest: str, value: object) -> float:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise CompetitionEconomicResultError(
            f"packet {digest!r} has a non-numeric top-level score {value!r}"
        )
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise CompetitionEconomicResultError(
            f"packet {digest!r} score must be finite and within [0, 1], got {value!r}"
        )
    return score


def aggregate_subject_scores(
    competition: CompetitionInput,
    packet_scores: Mapping[str, float],
) -> tuple[SubjectAggregate, ...]:
    """Validate exact evidence coverage and derive every subject's arithmetic mean.

    Mapping insertion order is irrelevant.  Subject order follows the committed
    ``CompetitionInput`` only for diagnostic stability; economic contender order is
    derived separately from score/hotkey/uid.
    """
    if not isinstance(competition, CompetitionInput):
        raise CompetitionEconomicResultError(
            "competition must be a validated CompetitionInput"
        )
    if not isinstance(packet_scores, Mapping):
        raise CompetitionEconomicResultError("packet_scores must be a digest mapping")

    required = _required_packet_digests(competition)
    expected = set(required)
    supplied = set(packet_scores)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(repr(value) for value in supplied - expected)
        raise CompetitionEconomicResultError(
            f"competition packet score coverage mismatch: missing={missing}, extra={extra}"
        )

    validated = {
        digest: _validated_score(digest, packet_scores[digest]) for digest in required
    }
    aggregates: list[SubjectAggregate] = []
    for subject in competition.subjects:
        values = [validated[digest] for digest in subject.packet_digests]
        aggregates.append(
            SubjectAggregate(
                subject_id=subject.subject_id,
                role=subject.role,
                uid=subject.uid,
                hotkey=subject.hotkey,
                score=math.fsum(values) / len(values),
            )
        )
    return tuple(aggregates)


def _single_role(
    subjects: tuple[SubjectAggregate, ...],
    role: str,
    *,
    required: bool,
) -> SubjectAggregate | None:
    matches = [subject for subject in subjects if subject.role == role]
    expected = "exactly one" if required else "at most one"
    if (required and len(matches) != 1) or (not required and len(matches) > 1):
        raise CompetitionEconomicResultError(
            f"an economic competition requires {expected} {role!r} subject; "
            f"found {len(matches)}"
        )
    return matches[0] if matches else None


def _contender_result(subject: SubjectAggregate) -> ContenderResult:
    if subject.uid is None or not subject.hotkey:
        raise CompetitionEconomicResultError(
            f"contender subject {subject.subject_id!r} has no payable uid/hotkey identity"
        )
    return ContenderResult(
        hotkey=subject.hotkey,
        uid=subject.uid,
        score=subject.score,
    )


def _v14_result_provenance(
    competition: CompetitionInput,
) -> tuple[datetime, int, str]:
    """Read the v14 fields while epoch/log.py is updated by its owning agent.

    This deliberately fails clearly against an older CompetitionInput instead of
    falling back to its database completion timestamp.
    """
    missing = [
        name
        for name in ("applied_at", "baseline_version", "baseline_artifact_digest")
        if not hasattr(competition, name)
    ]
    if missing:
        raise CompetitionEconomicResultError(
            "schema-v14 CompetitionInput lacks required economic provenance: "
            + ", ".join(missing)
        )
    return (
        getattr(competition, "applied_at"),
        getattr(competition, "baseline_version"),
        getattr(competition, "baseline_artifact_digest"),
    )


def derive_competition_economics(
    competition: CompetitionInput,
    packet_scores: Mapping[str, float],
) -> CompetitionEconomicDerivation:
    """Derive the unique ranked ``CompetitionResult`` from recomputed packet scores."""
    aggregates = aggregate_subject_scores(competition, packet_scores)
    baseline = _single_role(aggregates, "baseline", required=True)
    if baseline is None:  # narrowed by the required=True check above
        raise CompetitionEconomicResultError("competition has no archived baseline")
    unsupported = sorted(
        subject.subject_id
        for subject in aggregates
        if subject.role not in {"baseline", "contender"}
    )
    if unsupported:
        raise CompetitionEconomicResultError(
            "schema-v14 economic subjects may only be baseline or contender; "
            f"unsupported={unsupported}"
        )

    dedup_excluded_subjects = {
        subject.subject_id
        for subject in competition.subjects
        if subject.role == "contender" and subject.dedup_excluded
    }
    contenders = [
        _contender_result(subject)
        for subject in aggregates
        if subject.role == "contender"
        and subject.subject_id not in dedup_excluded_subjects
    ]
    if not contenders:
        raise CompetitionEconomicResultError(
            "an economic competition requires at least one contender"
        )
    contenders.sort(key=lambda value: (-value.score, value.hotkey, value.uid))
    applied_at, baseline_version, baseline_artifact_digest = _v14_result_provenance(
        competition
    )
    result = CompetitionResult(
        competition_id=competition.competition_id,
        track=competition.track,
        cycle=competition.cycle,
        applied_at=applied_at,
        contenders=tuple(contenders),
        baseline_score=baseline.score,
        baseline_version=baseline_version,
        baseline_artifact_digest=baseline_artifact_digest,
    )
    return CompetitionEconomicDerivation(
        result=result,
        subject_aggregates=aggregates,
        baseline_score=baseline.score,
    )


def derive_competition_result(
    competition: CompetitionInput,
    packet_scores: Mapping[str, float],
) -> CompetitionResult:
    """Convenience view returning only the tokenomics result."""
    return derive_competition_economics(competition, packet_scores).result
