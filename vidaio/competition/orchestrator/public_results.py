"""The public results table of a COMPLETED competition.

Hidden inputs become public at the result, and so do the scores measured on them:
this table exists ONLY once a competition is COMPLETED (the caller enforces it and
:func:`collect` refuses anything else).

``final_score`` is the economic score the result ranks and pays with: the
arithmetic mean of the subject's packet scores (``quality``), times the speed
multiplier when the anchored rules carry a speed term, and 0 when the subject
failed more items than the anchored ``max_failed_item_fraction``. It is computed by
the very functions the epoch evidence uses: ``economic_result.aggregate_subject_scores``
over the packet-bound per-item scores (``performance_history.item_score`` is the
packet's own top-level score; the evidence path re-reads every packet and refuses
any disagreement), with ``epoch_evidence._subject_run_seconds`` as the run time. It
is NOT ``contenders.final_score``, which is the review score without speed.

Rank follows the economic order ``(-score, hotkey)``. What only the published epoch
result applies is said in the note: the close-block registration census and the
duplicate-operator rule (same coldkey, IP, repository account, source tree or
output) can still remove an entry from the paid podium.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Sequence

from vidaio.audit.store import ArtifactKind, AuditStore
from vidaio.competition import repository as repo
from vidaio.competition.economic_result import aggregate_subject_scores
from vidaio.competition.epoch_evidence import (
    _effective_batch_timeout,
    _performance_rows,
    _subject_run_seconds,
)
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.orchestrator.results import ResultNotReady
from vidaio.competition.states import Phase
from vidaio.epoch.log import (
    CompetitionAuditItem,
    CompetitionAuditSubject,
    CompetitionInput,
    CompetitionRulesInput,
)

_MAX_PACKET_BYTES = 4 * 1024 * 1024

RESULTS_NOTE = (
    "final_score = quality x (1 - speed_weight + speed_weight x speed) when the "
    "anchored rules carry a speed term (else quality), and 0 above the anchored "
    "failed-item limit; quality is the mean item score. Rank is the economic order. "
    "The published result also applies the close-block registration census and the "
    "duplicate-operator rule, and pays places by the anchored result rules."
)


@dataclass
class _Subject:
    record: repo.ContenderRecord
    rows: list[sqlite3.Row]
    run_seconds: float | None
    complete: bool


@dataclass
class ResultsPlan:
    """Everything read from the database; packets to open are listed separately."""

    competition_id: str
    rules: CompetitionRulesInput | None
    item_count: int
    subjects: list[_Subject]
    #: (contender_id, item_id) -> reason code of an orchestrator-minted zero
    zero_codes: dict[tuple[int, int], str]
    #: packet digests of scorer gate failures whose reason must be read from the packet
    packets_to_read: list[str] = field(default_factory=list)


def collect(conn: sqlite3.Connection, competition_id: str) -> ResultsPlan:
    """Read one COMPLETED competition's scored matrix (database only)."""
    comp = repo.get_competition(conn, competition_id)
    if comp is None or comp.status is not Phase.COMPLETED:
        raise ResultNotReady(f"competition {competition_id} is not COMPLETED")
    manifest = repo.get_manifest(conn, competition_id)
    rules = (
        None
        if manifest.result_rules is None
        else CompetitionRulesInput(**manifest.result_rules.model_dump())
    )
    timed = rules is not None and rules.has_speed_term
    items = pers.list_items(conn, competition_id)
    subjects: list[_Subject] = []
    for record in repo.list_contenders(conn, competition_id):
        if record.status != "BUILT":
            continue  # only machine-accepted entries are scored into the result
        rows = list(_performance_rows(conn, competition_id, record.contender_id))
        complete = bool(items) and [int(r["item_id"]) for r in rows] == [
            int(i["item_id"]) for i in items
        ]
        run_seconds = None
        if complete and timed:
            assert manifest.sandbox_resources is not None  # manifest-validated
            run_seconds = _subject_run_seconds(
                conn,
                competition_id,
                record.contender_id,
                batch_timeout_seconds=_effective_batch_timeout(
                    conn,
                    competition_id,
                    manifest.sandbox_resources.batch_timeout_seconds,
                ),
                subject_id=f"contender:{record.contender_id}",
            )
        subjects.append(_Subject(record, rows, run_seconds, complete))
    zero_codes: dict[tuple[int, int], str] = {}
    for row in conn.execute(
        "SELECT payload_json FROM events WHERE competition_id = ? AND event_type = ?"
        " ORDER BY event_id",
        (competition_id, pers.EVENT_ITEM_ZEROED),
    ).fetchall():
        try:
            payload = json.loads(row["payload_json"] or "{}")
            zero_codes[(int(payload["contender_id"]), int(payload["item_id"]))] = str(
                payload["code"]
            )
        except (KeyError, TypeError, ValueError):
            continue
    plan = ResultsPlan(
        competition_id=competition_id,
        rules=rules,
        item_count=len(items),
        subjects=subjects,
        zero_codes=zero_codes,
    )
    for subject in subjects:
        for row in subject.rows:
            key = (subject.record.contender_id, int(row["item_id"]))
            if not row["valid"] and key not in zero_codes:
                plan.packets_to_read.append(str(row["score_packet_digest"]))
    return plan


def read_failure_codes(store: AuditStore, digests: Sequence[str]) -> dict[str, str]:
    """Blocking: the first violation code of each gate-failed scorer packet."""
    codes: dict[str, str] = {}
    for digest in digests:
        try:
            raw = store.get_digest_limited(
                ArtifactKind.SCORE_PACKET, digest, max_bytes=_MAX_PACKET_BYTES
            )
            violations = json.loads(raw).get("violations") or []
            code = violations[0].get("code") if violations else None
        except Exception:  # noqa: BLE001 - a display detail, never an error
            continue
        if isinstance(code, str) and code:
            codes[digest] = code
    return codes


def assemble(plan: ResultsPlan, failure_codes: dict[str, str]) -> dict[str, Any]:
    """The public table: ranked contender rows, the baseline line and the note."""
    complete = [s for s in plan.subjects if s.complete]
    aggregates: dict[int, Any] = {}
    if complete:
        subjects = tuple(
            CompetitionAuditSubject.model_construct(
                subject_id=f"subject:{s.record.contender_id}",
                role="baseline" if s.record.is_calibration else "contender",
                uid=None,
                hotkey=s.record.hotkey,
                dedup_excluded=False,
                packet_digests=tuple(str(r["score_packet_digest"]) for r in s.rows),
                run_seconds=s.run_seconds,
            )
            for s in complete
        )
        competition = CompetitionInput.model_construct(
            competition_id=plan.competition_id,
            rules=plan.rules,
            items=tuple(
                CompetitionAuditItem.model_construct(item_index=index)
                for index in range(plan.item_count)
            ),
            subjects=subjects,
        )
        packet_scores = {
            str(r["score_packet_digest"]): float(r["item_score"])
            for s in complete
            for r in s.rows
        }
        for subject, aggregate in zip(
            complete, aggregate_subject_scores(competition, packet_scores)
        ):
            aggregates[subject.record.contender_id] = aggregate

    def failure_code_counts(subject: _Subject) -> dict[str, int]:
        """How many items failed with each code: counts only, never which item.
        Per-item scores would tell entrants which kinds of clips were hard, and the
        subnet does not share anything about a hidden set beyond the manifest."""
        counts: dict[str, int] = {}
        for row in subject.rows:
            if bool(row["valid"]):
                continue
            code = plan.zero_codes.get(
                (subject.record.contender_id, int(row["item_id"]))
            ) or failure_codes.get(str(row["score_packet_digest"])) or "unknown"
            counts[code] = counts.get(code, 0) + 1
        return dict(sorted(counts.items()))

    def line(subject: _Subject) -> dict[str, Any]:
        aggregate = aggregates.get(subject.record.contender_id)
        entry: dict[str, Any] = {
            "final_score": None if aggregate is None else _finite(aggregate.score),
            "quality": None if aggregate is None else _finite(aggregate.quality),
            "speed": None if aggregate is None else _finite(aggregate.speed_score),
            "run_seconds": subject.run_seconds,
            "failed_items": sum(1 for r in subject.rows if not r["valid"]),
        }
        if aggregate is not None and aggregate.failed_fraction is not None:
            entry["failed_fraction"] = aggregate.failed_fraction
        if not subject.complete:
            entry["incomplete"] = True
        entry["failure_codes"] = failure_code_counts(subject)
        return entry

    contenders = [s for s in plan.subjects if not s.record.is_calibration]
    contenders.sort(
        key=lambda s: (
            s.record.contender_id not in aggregates,
            -(aggregates[s.record.contender_id].score)
            if s.record.contender_id in aggregates
            else 0.0,
            s.record.hotkey or "",
            s.record.contender_id,
        )
    )
    rows = []
    for rank, subject in enumerate(contenders, start=1):
        rows.append(
            {
                "rank": rank if subject.record.contender_id in aggregates else None,
                "hotkey": subject.record.hotkey,
                "contender_id": subject.record.contender_id,
                **line(subject),
            }
        )
    baseline = next((s for s in plan.subjects if s.record.is_calibration), None)
    rules = plan.rules
    return {
        "results": rows,
        "results_baseline": None if baseline is None else line(baseline),
        "results_speed_term": (
            None
            if rules is None or not rules.has_speed_term
            else {
                "weight": rules.speed_weight,
                "fast_spf": rules.speed_fast_spf,
                "slow_spf": rules.speed_slow_spf,
                "inserted_frames": rules.speed_inserted_frames,
            }
        ),
        "results_note": RESULTS_NOTE,
    }


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(value) else None


__all__ = ["RESULTS_NOTE", "ResultsPlan", "assemble", "collect", "read_failure_codes"]
