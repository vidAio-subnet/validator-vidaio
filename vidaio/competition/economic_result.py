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
from urllib.parse import urlsplit

from vidaio.epoch.log import CompetitionInput
from vidaio.tokenomics.rank_curve import dedup_ip_key
from vidaio.tokenomics.state import CompetitionResult, CompetitionRules, ContenderResult


class CompetitionEconomicResultError(ValueError):
    """Committed competition scores cannot derive one canonical economic result."""


@dataclass(frozen=True, slots=True)
class SubjectAggregate:
    """Score independently derived for one committed audit subject.

    ``score`` is what ranks: the arithmetic mean of the packet scores (``quality``),
    times the speed multiplier when the anchored rules carry a speed term.
    """

    subject_id: str
    role: str
    uid: int | None
    hotkey: str | None
    score: float
    quality: float | None = None
    run_seconds: float | None = None
    speed_score: float | None = None


@dataclass(frozen=True, slots=True)
class CompetitionEconomicDerivation:
    """Canonical tokenomics result plus the means used to construct it."""

    result: CompetitionResult
    subject_aggregates: tuple[SubjectAggregate, ...]
    #: None for a competition anchored without an executable baseline.
    baseline_score: float | None

    def aggregate_by_subject_id(self) -> dict[str, float]:
        """Return a fresh subject-id -> mean view for logs/tests/diagnostics."""
        return {
            aggregate.subject_id: aggregate.score
            for aggregate in self.subject_aggregates
        }


@dataclass(frozen=True, slots=True)
class CompetitionDedupCandidate:
    """Close-census, repository-account and exact-output identity for one payable
    contender."""

    subject_id: str
    uid: int
    coldkey: str
    ip: str
    output_digests: tuple[str, ...]
    #: :func:`repo_account_key` of the enrolled repository (None = no account).
    repo_account: str | None = None
    #: Git tree SHA of the enrolled commit (None = unknown).
    tree_sha: str | None = None


def repo_account_key(repo_url: str | None) -> str | None:
    """The hosting account a repository URL belongs to, as ``host/owner`` (lowercase),
    or None when the URL names no owner and repository.

    Entries enrolled from repositories of the same account count as one operator in
    :func:`competition_dedup_losers`, whatever coldkeys and hotkeys they use.
    """
    if not repo_url:
        return None
    try:
        parts = urlsplit(repo_url.strip())
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    segments = [segment for segment in parts.path.split("/") if segment]
    if not host or len(segments) < 2:
        return None
    return f"{host}/{segments[0].lower()}"


def competition_dedup_losers(
    candidates: tuple[CompetitionDedupCandidate, ...],
) -> frozenset[str]:
    """Lowest uid wins each advertised-IP, coldkey, repository-account, source-tree
    and exact-output slot.

    The scan deliberately mirrors inference dedup. Unspecified addresses are not a
    shared identity, while a byte-identical full output matrix is: later contenders
    cannot occupy a second podium/crown slot with the same solution output. One
    operator enrolling several repositories of one account (any coldkeys) likewise
    holds a single slot, and so does one source tree enrolled from several accounts
    (forks of one solution whose outputs were made to differ by a byte).
    """
    seen_ips: set[str] = set()
    seen_coldkeys: set[str] = set()
    seen_accounts: set[str] = set()
    seen_trees: set[str] = set()
    seen_outputs: set[tuple[str, ...]] = set()
    losers: set[str] = set()
    for candidate in sorted(candidates, key=lambda value: value.uid):
        if not candidate.output_digests:
            raise CompetitionEconomicResultError(
                f"competition contender {candidate.subject_id!r} has no output matrix"
            )
        ip_key = dedup_ip_key(candidate.ip)
        account = candidate.repo_account
        tree = None if not candidate.tree_sha else candidate.tree_sha.lower()
        if (
            (ip_key is not None and ip_key in seen_ips)
            or candidate.coldkey in seen_coldkeys
            or (account is not None and account in seen_accounts)
            or (tree is not None and tree in seen_trees)
            or candidate.output_digests in seen_outputs
        ):
            losers.add(candidate.subject_id)
            continue
        if ip_key is not None:
            seen_ips.add(ip_key)
        seen_coldkeys.add(candidate.coldkey)
        if account is not None:
            seen_accounts.add(account)
        if tree is not None:
            seen_trees.add(tree)
        seen_outputs.add(candidate.output_digests)
    return frozenset(losers)


def speed_score(
    run_seconds: float, *, inserted_frames: int, fast_spf: float, slow_spf: float
) -> float:
    """Speed credit in [0, 1] from a measured run time (log scale: every 2x faster is
    worth the same). ``spf`` = seconds per inserted frame; 1 at or under ``fast_spf``,
    0 at or over ``slow_spf``."""
    for name, value in (("run_seconds", run_seconds), ("fast_spf", fast_spf), ("slow_spf", slow_spf)):
        if not math.isfinite(value) or value < 0:
            raise CompetitionEconomicResultError(f"{name} must be finite and >= 0, got {value!r}")
    if inserted_frames < 1 or not 0 < fast_spf < slow_spf:
        raise CompetitionEconomicResultError("speed term needs inserted_frames >= 1 and 0 < fast_spf < slow_spf")
    spf = run_seconds / inserted_frames
    if spf <= fast_spf:
        return 1.0
    if spf >= slow_spf:
        return 0.0
    return math.log(slow_spf / spf) / math.log(slow_spf / fast_spf)


def speed_adjusted_score(quality: float, speed: float, *, weight: float) -> float:
    """``quality * (1 - weight + weight * speed)``, clamped to [0, 1]."""
    return min(1.0, max(0.0, quality * (1.0 - weight + weight * speed)))


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
    rules = getattr(competition, "rules", None)
    timed = rules is not None and rules.has_speed_term
    aggregates: list[SubjectAggregate] = []
    for subject in competition.subjects:
        values = [validated[digest] for digest in subject.packet_digests]
        quality = math.fsum(values) / len(values)
        score, speed = quality, None
        if timed:
            if subject.run_seconds is None:
                raise CompetitionEconomicResultError(
                    f"competition subject {subject.subject_id!r} has no measured run time"
                )
            speed = speed_score(
                subject.run_seconds,
                inserted_frames=rules.speed_inserted_frames,
                fast_spf=rules.speed_fast_spf,
                slow_spf=rules.speed_slow_spf,
            )
            score = speed_adjusted_score(quality, speed, weight=rules.speed_weight)
        aggregates.append(
            SubjectAggregate(
                subject_id=subject.subject_id,
                role=subject.role,
                uid=subject.uid,
                hotkey=subject.hotkey,
                score=score,
                quality=quality,
                run_seconds=subject.run_seconds,
                speed_score=speed,
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


def _competition_rules(competition: CompetitionInput) -> CompetitionRules | None:
    committed = getattr(competition, "rules", None)
    if committed is None:
        return None
    try:
        return CompetitionRules(
            crown_margin=committed.crown_margin,
            crown_min_score=committed.crown_min_score,
            podium_min_margin=committed.podium_min_margin,
            podium_min_score=committed.podium_min_score,
            # the anchored payout policy (tokenomics v3) travels with the rules
            crown_competition_share=getattr(committed, "crown_competition_share", None),
            podium_competition_share=getattr(committed, "podium_competition_share", None),
            crown_split=getattr(committed, "crown_split", None),
            podium_split=getattr(committed, "podium_split", None),
            redistribute_empty_places=getattr(committed, "redistribute_empty_places", None),
            no_qualifier_closes_window=getattr(committed, "no_qualifier_closes_window", None),
        )
    except ValueError as exc:
        raise CompetitionEconomicResultError(
            f"committed competition rules are invalid: {exc}"
        ) from exc


def derive_competition_economics(
    competition: CompetitionInput,
    packet_scores: Mapping[str, float],
) -> CompetitionEconomicDerivation:
    """Derive the unique ranked ``CompetitionResult`` from recomputed packet scores."""
    aggregates = aggregate_subject_scores(competition, packet_scores)
    has_baseline = getattr(competition, "baseline_version", None) is not None
    baseline = _single_role(aggregates, "baseline", required=has_baseline)
    if has_baseline and baseline is None:  # narrowed by required=True above
        raise CompetitionEconomicResultError("competition has no archived baseline")
    if not has_baseline and baseline is not None:
        raise CompetitionEconomicResultError(
            "a competition committed without a baseline carries a baseline subject"
        )
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
    # Every committed, non-deduplicated contender stays in the result (the epoch log
    # binds the result to exactly that identity set). The anchored podium conditions
    # select the PAID ranks later, in the reward-window fold.
    rules = _competition_rules(competition)
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
        baseline_score=None if baseline is None else baseline.score,
        baseline_version=baseline_version,
        baseline_artifact_digest=baseline_artifact_digest,
        rules=rules,
    )
    return CompetitionEconomicDerivation(
        result=result,
        subject_aggregates=aggregates,
        baseline_score=None if baseline is None else baseline.score,
    )


def derive_competition_result(
    competition: CompetitionInput,
    packet_scores: Mapping[str, float],
) -> CompetitionResult:
    """Convenience view returning only the tokenomics result."""
    return derive_competition_economics(competition, packet_scores).result
