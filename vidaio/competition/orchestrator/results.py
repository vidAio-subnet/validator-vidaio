"""Diagnostic completed-row adapter for the schema-v14 tokenomics result.

The database completion event is useful for visibility/cutoff selection only. It is
never an economic clock: callers must supply the finalized epoch close-block
``applied_at`` plus the active executable-baseline registry record. Stored human
``final_rank`` and review flags are ignored; earning order is always
``(-score, hotkey, uid)``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Mapping, Protocol

from vidaio.competition import repository as repo
from vidaio.competition.states import Phase
from vidaio.tokenomics.state import CompetitionResult, ContenderResult

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ActiveBaselineProvenance(Protocol):
    """Structural subset supplied by the executable baseline registry."""

    version: int
    artifact_digest: str


class ResultNotReady(Exception):
    """The database cannot yet produce a complete diagnostic result."""


def competition_cycle(conn: sqlite3.Connection, competition_id: str) -> int:
    """Stable 1-based global ordinal of the terminal COMPLETED transition.

    A competition's creation/start order is not its result order: a scheduled row
    may wait behind another competition and complete later.  The append-only event
    id is allocated in the same transaction as the terminal status transition, so
    counting terminal events through that id gives every later completion a strictly
    higher, replay-safe cycle while retaining compact sequential cycle numbers.
    """
    mine = _completion_event(conn, competition_id)
    row = conn.execute(
        """SELECT COUNT(*) AS n FROM events
           WHERE from_phase = ?
             AND to_phase = ?
             AND event_id <= ?""",
        (
            Phase.AWAITING_END_TIME.value,
            Phase.COMPLETED.value,
            mine["event_id"],
        ),
    ).fetchone()
    return int(row["n"])


def _completion_event(
    conn: sqlite3.Connection, competition_id: str
) -> sqlite3.Row:
    """Return the one immutable terminal transition for ``competition_id``."""
    competition = repo.get_competition(conn, competition_id)
    if competition is None:
        raise ResultNotReady(f"unknown competition {competition_id}")
    rows = conn.execute(
        """SELECT event_id, created_at FROM events
           WHERE competition_id = ?
             AND from_phase = ?
             AND to_phase = ?
           ORDER BY event_id""",
        (
            competition_id,
            Phase.AWAITING_END_TIME.value,
            Phase.COMPLETED.value,
        ),
    ).fetchall()
    if len(rows) != 1:
        raise ResultNotReady(
            f"competition {competition_id} requires exactly one terminal COMPLETED "
            f"event; found {len(rows)}"
        )
    return rows[0]


def completed_at(conn: sqlite3.Connection, competition_id: str) -> datetime:
    """Return the DB completion time for diagnostics/cutoff selection only."""
    row = _completion_event(conn, competition_id)
    parsed = repo.parse_ts(row["created_at"])
    if parsed is None:
        raise ResultNotReady(
            f"competition {competition_id} has a malformed completion timestamp"
        )
    return parsed.astimezone(timezone.utc)


def _is_baseline(record: object) -> bool:
    """Migration bridge until the repository record rename lands atomically."""
    if hasattr(record, "is_baseline"):
        return bool(getattr(record, "is_baseline"))
    return bool(getattr(record, "is_calibration", False))


def _validated_baseline(
    baseline: ActiveBaselineProvenance,
) -> tuple[int, str]:
    version = getattr(baseline, "version", None)
    digest = getattr(baseline, "artifact_digest", None)
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("active baseline version must be a non-negative integer")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("active baseline artifact_digest must be lowercase sha256 hex")
    return version, digest


def build_competition_result(
    conn: sqlite3.Connection,
    competition_id: str,
    *,
    applied_at: datetime,
    active_baseline: ActiveBaselineProvenance,
    uid_by_hotkey: Mapping[str, int],
) -> CompetitionResult:
    """Build a score-derived diagnostic result for one completed competition.

    This adapter does not claim its DB aggregates are authoritative epoch evidence;
    production constructs the same result from committed packet matrices. It exists
    for previews and consistency checks and deliberately refuses unresolved identities.
    """
    comp = repo.get_competition(conn, competition_id)
    if comp is None:
        raise ResultNotReady(f"unknown competition {competition_id}")
    if comp.status is not Phase.COMPLETED:
        raise ResultNotReady(
            f"competition {competition_id} is {comp.status.value}; a result exists only once COMPLETED"
        )
    if applied_at.tzinfo is None or applied_at.utcoffset() is None:
        raise ValueError(
            "applied_at must be the timezone-aware finalized epoch close time"
        )
    baseline_version, baseline_digest = _validated_baseline(active_baseline)

    records = repo.list_contenders(conn, competition_id)
    baselines = [record for record in records if _is_baseline(record)]
    if len(baselines) != 1:
        raise ResultNotReady(
            f"competition {competition_id} requires exactly one executable baseline row; found {len(baselines)}"
        )
    baseline_value = baselines[0].final_score
    contenders: list[ContenderResult] = []
    for record in records:
        if _is_baseline(record) or record.status != "BUILT":
            continue
        if not record.hotkey or record.hotkey not in uid_by_hotkey:
            raise ResultNotReady(
                f"competition contender {record.hotkey!r} has no close-block uid"
            )
        if record.final_score is None:
            raise ResultNotReady(
                f"competition contender {record.hotkey!r} has no committed aggregate score"
            )
        contenders.append(
            ContenderResult(
                hotkey=record.hotkey,
                uid=uid_by_hotkey[record.hotkey],
                score=float(record.final_score),
            )
        )
    if not contenders:
        raise ResultNotReady(
            f"competition {competition_id} has no payable built contender"
        )
    contenders.sort(key=lambda value: (-value.score, value.hotkey, value.uid))
    manifest = repo.get_manifest(conn, competition_id)
    return CompetitionResult(
        competition_id=competition_id,
        track=manifest.track,
        cycle=competition_cycle(conn, competition_id),
        applied_at=applied_at,
        contenders=tuple(contenders),
        baseline_score=None if baseline_value is None else float(baseline_value),
        baseline_version=baseline_version,
        baseline_artifact_digest=baseline_digest,
    )


def result_payload(result: CompetitionResult) -> dict[str, object]:
    return {
        "competition_id": result.competition_id,
        "track": result.track,
        "cycle": result.cycle,
        "applied_at": result.applied_at.isoformat(),
        "baseline_score": result.baseline_score,
        "baseline_version": result.baseline_version,
        "baseline_artifact_digest": result.baseline_artifact_digest,
        "contenders": [
            {"hotkey": contender.hotkey, "uid": contender.uid, "score": contender.score}
            for contender in result.contenders
        ],
    }


def result_json(result: CompetitionResult) -> str:
    return json.dumps(result_payload(result), sort_keys=True)
