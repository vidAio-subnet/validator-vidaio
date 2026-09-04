"""Typed data-access layer over the competition schema (spec: design spec §06).

All functions take an open sqlite3 connection (vidaio.core.db.connect) and, where a
timestamp is written, an explicit timezone-aware `now` — no wall-clock reads happen
in this module. Phase changes go through the engine, never through raw set_status
calls from outside this package.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vidaio.core.db import apply_migrations
from vidaio.competition.manifest import ArchivedBaseline, CompetitionManifest
from vidaio.competition.item_commitment import evaluation_item_commitment
from vidaio.competition.states import Phase, RUNNING_PHASES

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class EnrollmentError(Exception):
    """Enrollment rejected: wrong phase, past deadline, or stake below the gate."""


class ScorePacketError(Exception):
    """Score packet rejected: unparseable, violates the gates-first invariant, or its
    identity (miner hotkey / challenge id / item id) does not match the row being
    recorded."""


class EvaluationItemBindingError(ValueError):
    """Persisted evaluation rows disagree with their anchored manifest bindings."""


class EvaluationItemReuseError(EvaluationItemBindingError):
    """Evaluation media bytes were already exposed by another competition."""


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Apply this module's migrations. Safe to call at every startup."""
    return apply_migrations(conn, MIGRATIONS_DIR)


# ---- time helpers -------------------------------------------------------------

def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("naive datetime passed to competition persistence")
    return dt.astimezone(timezone.utc).isoformat()


def parse_ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


# ---- records ------------------------------------------------------------------

@dataclass(frozen=True)
class CompetitionRecord:
    competition_id: str
    track: str
    status: Phase
    manifest_digest: str
    commitment_root: str | None
    start_time: datetime
    enrollment_deadline: datetime
    finalization_time: datetime
    end_time: datetime
    human_review_deadline: datetime | None
    failure_reason: str | None

    @staticmethod
    def from_row(row: sqlite3.Row) -> "CompetitionRecord":
        return CompetitionRecord(
            competition_id=row["competition_id"],
            track=row["track"],
            status=Phase(row["status"]),
            manifest_digest=row["manifest_digest"],
            commitment_root=row["commitment_root"],
            start_time=parse_ts(row["start_time"]),  # type: ignore[arg-type]
            enrollment_deadline=parse_ts(row["enrollment_deadline"]),  # type: ignore[arg-type]
            finalization_time=parse_ts(row["finalization_time"]),  # type: ignore[arg-type]
            end_time=parse_ts(row["end_time"]),  # type: ignore[arg-type]
            human_review_deadline=parse_ts(row["human_review_deadline"]),
            failure_reason=row["failure_reason"],
        )


@dataclass(frozen=True)
class ContenderRecord:
    contender_id: int
    competition_id: str
    hotkey: str | None
    is_calibration: bool
    repo_url: str
    commit_sha: str
    tree_sha: str
    image_digest: str | None
    status: str
    enrollment_stake: float
    eligible: bool
    manual_disqualified: bool
    final_score: float | None
    final_rank: int | None
    media_score_aggregate: float | None
    worst_decile_aggregate: float | None
    cost_efficiency_aggregate: float | None
    length_coverage: float | None
    average_vmaf: float | None
    average_compression_rate: float | None

    @staticmethod
    def from_row(row: sqlite3.Row) -> "ContenderRecord":
        return ContenderRecord(
            contender_id=row["contender_id"],
            competition_id=row["competition_id"],
            hotkey=row["hotkey"],
            is_calibration=bool(row["is_calibration"]),
            repo_url=row["repo_url"],
            commit_sha=row["commit_sha"],
            tree_sha=row["tree_sha"],
            image_digest=row["image_digest"],
            status=row["status"],
            enrollment_stake=row["enrollment_stake"],
            eligible=bool(row["eligible"]),
            manual_disqualified=bool(row["manual_disqualified"]),
            final_score=row["final_score"],
            final_rank=row["final_rank"],
            media_score_aggregate=row["media_score_aggregate"],
            worst_decile_aggregate=row["worst_decile_aggregate"],
            cost_efficiency_aggregate=row["cost_efficiency_aggregate"],
            length_coverage=row["length_coverage"],
            average_vmaf=row["average_vmaf"],
            average_compression_rate=row["average_compression_rate"],
        )


# ---- competitions -------------------------------------------------------------

def insert_competition(
    conn: sqlite3.Connection, manifest: CompetitionManifest, now: datetime
) -> None:
    ts = iso(now)
    conn.execute(
        """INSERT INTO competitions
           (competition_id, track, status, manifest_json, manifest_digest,
            start_time, enrollment_deadline, finalization_time, end_time,
            created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            manifest.competition_id,
            manifest.track,
            Phase.SCHEDULED.value,
            manifest.canonical_json(),
            manifest.manifest_digest(),
            iso(manifest.start_time),
            iso(manifest.enrollment_deadline),
            iso(manifest.finalization_time),
            iso(manifest.end_time),
            ts,
            ts,
        ),
    )


def get_competition_row(conn: sqlite3.Connection, competition_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM competitions WHERE competition_id = ?", (competition_id,)
    ).fetchone()


def get_competition(conn: sqlite3.Connection, competition_id: str) -> CompetitionRecord | None:
    row = get_competition_row(conn, competition_id)
    return CompetitionRecord.from_row(row) if row is not None else None


def get_manifest(conn: sqlite3.Connection, competition_id: str) -> CompetitionManifest:
    row = get_competition_row(conn, competition_id)
    if row is None:
        raise KeyError(f"unknown competition {competition_id}")
    return CompetitionManifest.model_validate_json(row["manifest_json"])


def list_competitions_in(
    conn: sqlite3.Connection, phases: Iterable[Phase]
) -> list[CompetitionRecord]:
    names = [p.value for p in phases]
    marks = ",".join("?" for _ in names)
    rows = conn.execute(
        f"SELECT * FROM competitions WHERE status IN ({marks})"
        " ORDER BY start_time, competition_id",
        names,
    ).fetchall()
    return [CompetitionRecord.from_row(r) for r in rows]


def running_competition_id(conn: sqlite3.Connection) -> str | None:
    names = [p.value for p in RUNNING_PHASES]
    marks = ",".join("?" for _ in names)
    row = conn.execute(
        f"SELECT competition_id FROM competitions WHERE status IN ({marks})", names
    ).fetchone()
    return row["competition_id"] if row is not None else None


def set_status(
    conn: sqlite3.Connection,
    competition_id: str,
    status: Phase,
    now: datetime,
    *,
    failure_reason: str | None = None,
) -> None:
    """Engine-internal: persist a phase change. Callers must hold transition guards."""
    conn.execute(
        "UPDATE competitions SET status = ?, failure_reason = COALESCE(?, failure_reason),"
        " updated_at = ? WHERE competition_id = ?",
        (status.value, failure_reason, iso(now), competition_id),
    )


def set_human_review_deadline(
    conn: sqlite3.Connection, competition_id: str, deadline: datetime, now: datetime
) -> None:
    conn.execute(
        "UPDATE competitions SET human_review_deadline = ?, updated_at = ?"
        " WHERE competition_id = ?",
        (iso(deadline), iso(now), competition_id),
    )


def set_commitment_root(
    conn: sqlite3.Connection, competition_id: str, root: str, now: datetime
) -> None:
    """Engine-internal: persist the anchored pre-commitment root. Callers
    (engine.mark_commitment_anchored) hold the phase/idempotency guards."""
    conn.execute(
        "UPDATE competitions SET commitment_root = ?, updated_at = ? WHERE competition_id = ?",
        (root, iso(now), competition_id),
    )


# ---- contenders ---------------------------------------------------------------

def enroll_contender(
    conn: sqlite3.Connection,
    competition_id: str,
    *,
    hotkey: str,
    repo_url: str,
    commit_sha: str,
    tree_sha: str,
    stake: float,
    now: datetime,
) -> int:
    """Enroll a real (earning-eligible) contender during ENROLLING.

    Enforces the phase, the enrollment deadline, and the alpha-stake gate from the
    manifest (spec §04: minimum_alpha_stake controls enrollment eligibility).
    """
    comp = get_competition(conn, competition_id)
    if comp is None:
        raise EnrollmentError(f"unknown competition {competition_id}")
    if comp.status is not Phase.ENROLLING:
        raise EnrollmentError(f"competition {competition_id} is {comp.status}, not ENROLLING")
    if now > comp.enrollment_deadline:
        raise EnrollmentError("enrollment deadline has passed")
    manifest = get_manifest(conn, competition_id)
    if stake < manifest.minimum_alpha_stake:
        raise EnrollmentError(
            f"stake {stake} below minimum_alpha_stake {manifest.minimum_alpha_stake}"
        )
    ts = iso(now)
    cur = conn.execute(
        """INSERT INTO contenders
           (competition_id, hotkey, is_calibration, repo_url, commit_sha, tree_sha,
            status, enrollment_stake, created_at, updated_at)
           VALUES (?, ?, 0, ?, ?, ?, 'ENROLLED', ?, ?, ?)""",
        (competition_id, hotkey, repo_url, commit_sha, tree_sha, stake, ts, ts),
    )
    record_event(
        conn,
        competition_id,
        "contender_enrolled",
        now,
        payload={"contender_id": cur.lastrowid, "hotkey": hotkey},
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def insert_calibration_contender(
    conn: sqlite3.Connection, competition_id: str, baseline: ArchivedBaseline, now: datetime
) -> int:
    """Insert the baseline calibration contender (idempotent: returns the existing row's id).

    hotkey is NULL and is_calibration=1 — the schema forbids a final_rank on this row,
    so it can never reach ranking/podium/payout (the project design record #1).
    """
    existing = conn.execute(
        "SELECT contender_id FROM contenders WHERE competition_id = ? AND is_calibration = 1",
        (competition_id,),
    ).fetchone()
    if existing is not None:
        return int(existing["contender_id"])
    ts = iso(now)
    cur = conn.execute(
        """INSERT INTO contenders
           (competition_id, hotkey, is_calibration, repo_url, commit_sha, tree_sha,
            status, enrollment_stake, created_at, updated_at)
           VALUES (?, NULL, 1, ?, ?, ?, 'ENROLLED', 0, ?, ?)""",
        (competition_id, baseline.repo_url, baseline.commit_sha, baseline.tree_sha, ts, ts),
    )
    record_event(
        conn,
        competition_id,
        "calibration_injected",
        now,
        payload={"contender_id": cur.lastrowid, "tree_sha": baseline.tree_sha},
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def get_contender(conn: sqlite3.Connection, contender_id: int) -> ContenderRecord | None:
    row = conn.execute(
        "SELECT * FROM contenders WHERE contender_id = ?", (contender_id,)
    ).fetchone()
    return ContenderRecord.from_row(row) if row is not None else None


def list_contenders(conn: sqlite3.Connection, competition_id: str) -> list[ContenderRecord]:
    rows = conn.execute(
        "SELECT * FROM contenders WHERE competition_id = ? ORDER BY contender_id",
        (competition_id,),
    ).fetchall()
    return [ContenderRecord.from_row(r) for r in rows]


def set_contender_status(
    conn: sqlite3.Connection, contender_id: int, status: str, now: datetime
) -> None:
    conn.execute(
        "UPDATE contenders SET status = ?, updated_at = ? WHERE contender_id = ?",
        (status, iso(now), contender_id),
    )


def set_contender_image_digest(
    conn: sqlite3.Connection, contender_id: int, image_digest: str, now: datetime
) -> None:
    conn.execute(
        "UPDATE contenders SET image_digest = ?, status = 'BUILT', updated_at = ?"
        " WHERE contender_id = ?",
        (image_digest, iso(now), contender_id),
    )


def count_contenders_by_status(conn: sqlite3.Connection, competition_id: str) -> dict[str, int]:
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM contenders WHERE competition_id = ?"
        " GROUP BY status",
        (competition_id,),
    ).fetchall()
    return {r["status"]: r["n"] for r in rows}


def count_accepted_real_contenders(conn: sqlite3.Connection, competition_id: str) -> int:
    """ACCEPTED contenders excluding the calibration baseline — a competition with only
    the baseline accepted has no real contender and must fail VALIDATING (spec §04)."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM contenders WHERE competition_id = ?"
        " AND status = 'ACCEPTED' AND is_calibration = 0",
        (competition_id,),
    ).fetchone()
    return int(row["n"])


def count_pending_validation(conn: sqlite3.Connection, competition_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM contenders WHERE competition_id = ? AND status = 'ENROLLED'",
        (competition_id,),
    ).fetchone()
    return int(row["n"])


# ---- evaluation items & per-item scores --------------------------------------

def add_evaluation_item(
    conn: sqlite3.Connection,
    competition_id: str,
    *,
    item_index: int,
    input_sha256: str,
    input_bytes: int,
    threshold_commitment: str,
    challenge_id: str,
    now: datetime,
    length_seconds: float | None = None,
    scoring_item_id: str | None = None,
    reference_sha256: str | None = None,
    reference_bytes: int | None = None,
    upscale_factor: int | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    item_commitment: str | None = None,
) -> int:
    """Add one sealed evaluation input, with a compatibility-safe track binding.

    (challenge_id, scoring_item_id) is the identity a score packet must carry to be
    recorded against this item (record_item_score). scoring_item_id defaults to the
    sealed input's sha256 — the content-addressed item identity.

    Legacy/compression callers may omit every new keyword; the pristine reference is
    normalized to the same bytes as ``input_sha256``.  Upscaling callers must provide
    a distinct pristine reference plus factor.  Their canonical item commitment is
    derived here and must equal the entry at ``item_index`` in the already-persisted,
    pre-enrollment manifest.  ``item_commitment`` is accepted only as an optional
    caller assertion: it never overrides the derived value.
    """
    if _SHA256_HEX.fullmatch(input_sha256) is None:
        raise ValueError("input_sha256 must be lowercase sha256 hex")
    if input_bytes < 0:
        raise ValueError("input_bytes must be non-negative")
    manifest = get_manifest(conn, competition_id)
    if manifest.track == "compression":
        normalized_reference = reference_sha256 or input_sha256
        normalized_reference_bytes = (
            input_bytes if reference_bytes is None else reference_bytes
        )
        if normalized_reference != input_sha256:
            raise ValueError(
                "compression evaluation reference must equal the miner input"
            )
        if normalized_reference_bytes != input_bytes:
            raise ValueError(
                "compression evaluation reference size must equal the miner input size"
            )
        if (
            upscale_factor is not None
            or target_width is not None
            or target_height is not None
            or item_commitment is not None
        ):
            raise ValueError(
                "compression evaluation items cannot carry upscaling "
                "factor/geometry/commitment"
            )
        derived_commitment = None
    else:
        if reference_sha256 is None or reference_bytes is None:
            raise ValueError(
                "upscaling evaluation item requires reference_sha256/reference_bytes"
            )
        if upscale_factor is None:
            raise ValueError("upscaling evaluation item requires upscale_factor")
        if target_width is None or target_height is None:
            raise ValueError(
                "upscaling evaluation item requires target_width/target_height"
            )
        if (
            type(target_width) is not int
            or type(target_height) is not int
            or target_width <= 0
            or target_height <= 0
        ):
            raise ValueError("upscaling target dimensions must be positive integers")
        if reference_bytes <= 0 or input_bytes <= 0:
            raise ValueError("upscaling reference/input artifacts must be non-empty")
        allowed = manifest.allowed_upscale_factors or []
        if upscale_factor not in allowed:
            raise ValueError(
                f"upscale_factor {upscale_factor} is not allowed by the manifest"
            )
        commitments = manifest.evaluation_item_commitments or []
        if item_index >= len(commitments):
            raise ValueError(
                f"item_index {item_index} has no precommitted manifest entry"
            )
        derived_commitment = evaluation_item_commitment(
            competition_id=competition_id,
            item_index=item_index,
            reference_sha256=reference_sha256,
            input_sha256=input_sha256,
            upscale_factor=upscale_factor,
            target_width=target_width,
            target_height=target_height,
        )
        if derived_commitment != commitments[item_index]:
            raise ValueError(
                "upscaling evaluation item bytes/factor/geometry do not match the "
                "pre-enrollment manifest commitment"
            )
        if item_commitment is not None and item_commitment != derived_commitment:
            raise ValueError(
                "caller-supplied item_commitment disagrees with the canonical preimage"
            )
        normalized_reference = reference_sha256
        normalized_reference_bytes = reference_bytes
    # A completed competition publishes pristine upscaling references.  Reusing
    # either side of any older item (including cross-kind ref<->input reuse) would
    # let a contender know future hidden bytes.  Check here for a readable failure;
    # migration triggers repeat it authoritatively under concurrent/direct SQL.
    candidate_digests = {input_sha256, normalized_reference}
    marks = ",".join("?" for _ in candidate_digests)
    reused = conn.execute(
        f"SELECT competition_id, input_sha256, reference_sha256 FROM evaluation_items "
        f"WHERE competition_id != ? AND (input_sha256 IN ({marks}) OR "
        f"reference_sha256 IN ({marks})) LIMIT 1",
        (
            competition_id,
            *candidate_digests,
            *candidate_digests,
        ),
    ).fetchone()
    if reused is not None:
        raise EvaluationItemReuseError(
            "evaluation media digest was already used by competition "
            f"{reused['competition_id']!r}; hidden inputs/references are single-use"
        )

    cur = conn.execute(
        """INSERT INTO evaluation_items
           (competition_id, item_index, input_sha256, input_bytes, length_seconds,
            threshold_commitment, challenge_id, scoring_item_id, created_at,
            reference_sha256, reference_bytes, upscale_factor, target_width,
            target_height, item_commitment)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            competition_id,
            item_index,
            input_sha256,
            input_bytes,
            length_seconds,
            threshold_commitment,
            challenge_id,
            scoring_item_id if scoring_item_id is not None else input_sha256,
            iso(now),
            normalized_reference,
            normalized_reference_bytes,
            upscale_factor,
            target_width,
            target_height,
            derived_commitment,
        ),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def validate_evaluation_item_bindings(
    conn: sqlite3.Connection, competition_id: str
) -> list[sqlite3.Row]:
    """Return ordered item rows only when their full track binding is valid.

    This check is intentionally repeated at trusted consumption boundaries.  Direct
    SQL, a buggy migration, or mutable DB metadata cannot change the reference/input
    pairing or factor committed inside the canonical manifest.
    """
    manifest = get_manifest(conn, competition_id)
    rows = conn.execute(
        "SELECT * FROM evaluation_items WHERE competition_id = ? ORDER BY item_index",
        (competition_id,),
    ).fetchall()
    for row in rows:
        digests = {str(row["input_sha256"])}
        if row["reference_sha256"] is not None:
            digests.add(str(row["reference_sha256"]))
        marks = ",".join("?" for _ in digests)
        reused = conn.execute(
            f"SELECT competition_id FROM evaluation_items WHERE competition_id != ? "
            f"AND (input_sha256 IN ({marks}) OR reference_sha256 IN ({marks})) LIMIT 1",
            (competition_id, *digests, *digests),
        ).fetchone()
        if reused is not None:
            raise EvaluationItemReuseError(
                f"evaluation media for {competition_id!r} is reused by competition "
                f"{reused['competition_id']!r}; hidden inputs/references are single-use"
            )
    if manifest.track == "upscaling":
        commitments = manifest.evaluation_item_commitments or []
        if len(rows) != len(commitments):
            raise EvaluationItemBindingError(
                f"upscaling item matrix has {len(rows)} row(s), but the manifest "
                f"commits {len(commitments)}"
            )
        allowed = set(manifest.allowed_upscale_factors or [])
        for expected_index, (row, committed) in enumerate(zip(rows, commitments)):
            item_index = int(row["item_index"])
            reference_sha256 = row["reference_sha256"]
            input_sha256 = str(row["input_sha256"])
            factor = row["upscale_factor"]
            target_width = row["target_width"]
            target_height = row["target_height"]
            if item_index != expected_index:
                raise EvaluationItemBindingError(
                    f"upscaling item order skips/reorders index {expected_index}"
                )
            if not isinstance(reference_sha256, str):
                raise EvaluationItemBindingError(
                    f"upscaling item {item_index} has no pristine reference digest"
                )
            if (
                _SHA256_HEX.fullmatch(reference_sha256) is None
                or _SHA256_HEX.fullmatch(input_sha256) is None
            ):
                raise EvaluationItemBindingError(
                    f"upscaling item {item_index} has a malformed media digest"
                )
            if reference_sha256 == input_sha256:
                raise EvaluationItemBindingError(
                    f"upscaling item {item_index} aliases reference and miner input"
                )
            if (
                not isinstance(row["reference_bytes"], int)
                or int(row["reference_bytes"]) <= 0
                or int(row["input_bytes"]) <= 0
            ):
                raise EvaluationItemBindingError(
                    f"upscaling item {item_index} has an invalid artifact size"
                )
            if not isinstance(factor, int) or factor not in allowed:
                raise EvaluationItemBindingError(
                    f"upscaling item {item_index} factor is not manifest-allowed"
                )
            if (target_width is None) != (target_height is None):
                raise EvaluationItemBindingError(
                    f"upscaling item {item_index} has incomplete target geometry"
                )
            if target_width is not None and (
                not isinstance(target_width, int)
                or not isinstance(target_height, int)
                or target_width <= 0
                or target_height <= 0
            ):
                raise EvaluationItemBindingError(
                    f"upscaling item {item_index} has invalid target geometry"
                )
            try:
                derived = evaluation_item_commitment(
                    competition_id=competition_id,
                    item_index=item_index,
                    reference_sha256=reference_sha256,
                    input_sha256=input_sha256,
                    upscale_factor=factor,
                    target_width=target_width,
                    target_height=target_height,
                )
            except ValueError as exc:
                raise EvaluationItemBindingError(
                    f"upscaling item {item_index} binding is invalid: {exc}"
                ) from exc
            if row["item_commitment"] != derived or committed != derived:
                raise EvaluationItemBindingError(
                    f"upscaling item {item_index} does not match its manifest commitment"
                )
        return rows

    for row in rows:
        if (
            row["reference_sha256"] != row["input_sha256"]
            or row["reference_bytes"] != row["input_bytes"]
            or row["upscale_factor"] is not None
            or row["target_width"] is not None
            or row["target_height"] is not None
            or row["item_commitment"] is not None
        ):
            raise EvaluationItemBindingError(
                f"compression item {row['item_index']} is not normalized to reference=input"
            )
    return rows


class ScorePacketPayload(BaseModel):
    """Local shape of the scoring module's ItemScore JSON — only the fields this
    module binds persistence to. Deliberately NOT imported from vidaio.scoring:
    persistence must not depend on scoring internals. Extra packet fields are
    allowed; they still count toward the digest (which hashes the exact bytes).

    `score` must be a FINITE float in [0, 1]: Infinity/NaN or out-of-range values
    make the whole packet unparseable (ScorePacketError) — a non-finite score can
    never be persisted, so it can never reach aggregation or ranking."""

    model_config = ConfigDict(extra="allow", frozen=True)

    item_id: str
    challenge_id: str
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    gate_passed: bool
    miner_hotkey: str | None = None
    content_digest: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


def _packet_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def record_item_score(
    conn: sqlite3.Connection,
    competition_id: str,
    *,
    contender_id: int,
    item_id: int,
    packet_bytes: bytes,
    now: datetime,
    output_bytes: int | None = None,
    batch_id: int | None = None,
) -> int:
    """Persist one per-item score, packet-bound.

    The ONLY score source is `packet_bytes` — the scorer's ItemScore JSON, verbatim:
    item_score is the packet's top-level `score`, valid is its `gate_passed`, and
    score_packet_digest is sha256 over the exact bytes handed in. Callers cannot
    supply score/valid/digest independently. Raises ScorePacketError when the packet
    is unparseable (including a non-finite or out-of-[0,1] `score` — see
    ScorePacketPayload), when a gate-failed packet carries a non-zero score (the
    packet's own invariant), or when the packet's identity (miner_hotkey,
    challenge_id, item_id) does not match the contender / evaluation item being
    recorded.
    """
    try:
        packet = ScorePacketPayload.model_validate_json(packet_bytes)
    except ValidationError as exc:
        raise ScorePacketError(f"unparseable score packet: {exc}") from exc
    if not packet.gate_passed and packet.score != 0.0:
        raise ScorePacketError(
            f"packet violates the gates-first invariant: gate_passed is False but "
            f"score is {packet.score} (must be 0.0)"
        )

    contender = get_contender(conn, contender_id)
    if contender is None or contender.competition_id != competition_id:
        raise ScorePacketError(f"contender {contender_id} not part of {competition_id}")
    if packet.miner_hotkey != contender.hotkey:
        raise ScorePacketError(
            f"packet miner_hotkey {packet.miner_hotkey!r} does not match contender "
            f"{contender_id} hotkey {contender.hotkey!r}"
        )

    item = conn.execute(
        "SELECT * FROM evaluation_items WHERE item_id = ?", (item_id,)
    ).fetchone()
    if item is None or item["competition_id"] != competition_id:
        raise ScorePacketError(f"evaluation item {item_id} not part of {competition_id}")
    if packet.challenge_id != item["challenge_id"] or packet.item_id != item["scoring_item_id"]:
        raise ScorePacketError(
            f"packet identity ({packet.challenge_id!r}, {packet.item_id!r}) does not "
            f"match evaluation item {item_id} "
            f"({item['challenge_id']!r}, {item['scoring_item_id']!r})"
        )

    item_score = packet.score if packet.gate_passed else 0.0
    cur = conn.execute(
        """INSERT INTO performance_history
           (competition_id, contender_id, item_id, batch_id, vmaf, compression_rate,
            cost, length_seconds, valid, item_score, output_sha256, output_bytes,
            score_packet_digest, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            competition_id,
            contender_id,
            item_id,
            batch_id,
            _packet_metric(packet.metrics, "vmaf"),
            _packet_metric(packet.metrics, "compression_rate"),
            _packet_metric(packet.metrics, "cost"),
            _packet_metric(packet.metrics, "length_seconds"),
            1 if packet.gate_passed else 0,
            item_score,
            packet.content_digest,
            output_bytes,
            hashlib.sha256(packet_bytes).hexdigest(),
            iso(now),
        ),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def set_audit_bundle_digest(
    conn: sqlite3.Connection, performance_id: int, audit_bundle_digest: str
) -> None:
    """Link one persisted (contender, item) score to its audit-store bundle (§08).

    The digest must be a 64-char lowercase sha256 hex string (an empty or malformed
    value can never unlock the completion gate), and linkage is WRITE-ONCE: once a
    row is linked, overwriting its digest with a DIFFERENT value raises; re-linking
    the SAME value is an idempotent no-op. Both invariants are also enforced at the
    SQL level (CHECK constraint + write-once BEFORE UPDATE trigger)."""
    if not isinstance(audit_bundle_digest, str) or not _SHA256_HEX.fullmatch(
        audit_bundle_digest
    ):
        raise ValueError(
            f"audit_bundle_digest must be a 64-char lowercase sha256 hex digest, "
            f"got {audit_bundle_digest!r}"
        )
    row = conn.execute(
        "SELECT audit_bundle_digest FROM performance_history WHERE performance_id = ?",
        (performance_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown performance row {performance_id}")
    existing = row["audit_bundle_digest"]
    if existing is not None:
        if existing == audit_bundle_digest:
            return  # idempotent re-link of the same bundle
        raise ValueError(
            f"performance row {performance_id} is already audit-linked to {existing}; "
            f"refusing to overwrite with {audit_bundle_digest} (linkage is write-once)"
        )
    conn.execute(
        "UPDATE performance_history SET audit_bundle_digest = ? WHERE performance_id = ?",
        (audit_bundle_digest, performance_id),
    )


# ---- pipeline-completion guard queries (engine.mark_* verification) -----------

def count_non_terminal_batches(conn: sqlite3.Connection, competition_id: str) -> int:
    """Batches not yet terminal (COMPLETED/FAILED) — evaluation cannot complete
    while any exist."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM batches WHERE competition_id = ?"
        " AND status NOT IN ('COMPLETED', 'FAILED')",
        (competition_id,),
    ).fetchone()
    return int(row["n"])


def count_missing_item_scores(conn: sqlite3.Connection, competition_id: str) -> int:
    """(accepted real contender, evaluation item) pairs with no performance_history
    row — scores cannot be marked persisted while any are missing."""
    row = conn.execute(
        """SELECT COUNT(*) AS n
           FROM contenders c
           JOIN evaluation_items ei ON ei.competition_id = c.competition_id
           WHERE c.competition_id = ? AND c.is_calibration = 0
             AND c.status IN ('ACCEPTED', 'BUILT')
             AND NOT EXISTS (
                 SELECT 1 FROM performance_history ph
                 WHERE ph.contender_id = c.contender_id AND ph.item_id = ei.item_id)""",
        (competition_id,),
    ).fetchone()
    return int(row["n"])


def count_missing_calibration_rows(conn: sqlite3.Connection, competition_id: str) -> int:
    """(calibration contender, evaluation item) pairs with no performance_history row.

    The baseline calibration contender must hold a score row for EVERY evaluation item
    exactly like a real contender — its score drives the ratchet/crown, so a baseline
    with missing rows must stall completion (audit_linkage_gaps only sees rows that
    EXIST; this closes the zero-row bypass). No status filter: a configured baseline is
    accountable for the full matrix regardless of its build/acceptance state."""
    row = conn.execute(
        """SELECT COUNT(*) AS n
           FROM contenders c
           JOIN evaluation_items ei ON ei.competition_id = c.competition_id
           WHERE c.competition_id = ? AND c.is_calibration = 1
             AND NOT EXISTS (
                 SELECT 1 FROM performance_history ph
                 WHERE ph.contender_id = c.contender_id AND ph.item_id = ei.item_id)""",
        (competition_id,),
    ).fetchone()
    return int(row["n"])


# ---- ranking ------------------------------------------------------------------

def ranking(conn: sqlite3.Connection, competition_id: str) -> list[ContenderRecord]:
    """Final ranking: calibration and disqualified rows are excluded by construction —
    only rows holding a final_rank appear, and the schema forbids final_rank on
    calibration rows; recalculate_ranks never assigns one to disqualified rows."""
    rows = conn.execute(
        """SELECT * FROM contenders
           WHERE competition_id = ? AND final_rank IS NOT NULL
             AND is_calibration = 0 AND manual_disqualified = 0 AND eligible = 1
           ORDER BY final_rank""",
        (competition_id,),
    ).fetchall()
    return [ContenderRecord.from_row(r) for r in rows]


def podium(conn: sqlite3.Connection, competition_id: str, n: int = 3) -> list[ContenderRecord]:
    return ranking(conn, competition_id)[:n]


# ---- event log ----------------------------------------------------------------

def record_event(
    conn: sqlite3.Connection,
    competition_id: str,
    event_type: str,
    now: datetime,
    *,
    from_phase: Phase | None = None,
    to_phase: Phase | None = None,
    guard: str | None = None,
    payload: dict[str, Any] | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO events
           (competition_id, event_type, from_phase, to_phase, guard, payload_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            competition_id,
            event_type,
            from_phase.value if from_phase else None,
            to_phase.value if to_phase else None,
            guard,
            json.dumps(payload, sort_keys=True) if payload is not None else None,
            iso(now),
        ),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def list_events(conn: sqlite3.Connection, competition_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM events WHERE competition_id = ? ORDER BY event_id",
        (competition_id,),
    ).fetchall()


# ---- human reviews (hash-chained, append-only) --------------------------------

def _chain_genesis(competition_id: str) -> str:
    return hashlib.sha256(f"vidaio-review-chain:{competition_id}".encode("utf-8")).hexdigest()


def _review_content_json(
    competition_id: str,
    contender_id: int,
    action: str,
    reviewer: str,
    reason: str,
    detail_json: str | None,
    supersedes_review_id: int | None,
    created_at: str,
) -> str:
    return json.dumps(
        {
            "competition_id": competition_id,
            "contender_id": contender_id,
            "action": action,
            "reviewer": reviewer,
            "reason": reason,
            "detail_json": detail_json,
            "supersedes_review_id": supersedes_review_id,
            "created_at": created_at,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _chain_hash(prev_hash: str, content_json: str) -> str:
    # integrity_hash = sha256(prev_row_hash || canonical row json)
    return hashlib.sha256((prev_hash + content_json).encode("utf-8")).hexdigest()


def insert_human_review(
    conn: sqlite3.Connection,
    competition_id: str,
    *,
    contender_id: int,
    action: str,
    reviewer: str,
    reason: str,
    now: datetime,
    detail: dict[str, Any] | None = None,
    supersedes_review_id: int | None = None,
) -> int:
    """Append a review row, extending the competition's hash chain."""
    prev = conn.execute(
        "SELECT integrity_hash FROM human_reviews WHERE competition_id = ?"
        " ORDER BY review_id DESC LIMIT 1",
        (competition_id,),
    ).fetchone()
    prev_hash = prev["integrity_hash"] if prev is not None else _chain_genesis(competition_id)
    created_at = iso(now)
    detail_json = (
        json.dumps(detail, sort_keys=True, separators=(",", ":")) if detail is not None else None
    )
    content = _review_content_json(
        competition_id, contender_id, action, reviewer, reason,
        detail_json, supersedes_review_id, created_at,
    )
    cur = conn.execute(
        """INSERT INTO human_reviews
           (competition_id, contender_id, action, reviewer, reason, detail_json,
            supersedes_review_id, prev_hash, integrity_hash, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            competition_id,
            contender_id,
            action,
            reviewer,
            reason,
            detail_json,
            supersedes_review_id,
            prev_hash,
            _chain_hash(prev_hash, content),
            created_at,
        ),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def verify_review_chain(conn: sqlite3.Connection, competition_id: str) -> bool:
    """Recompute the whole hash chain; True iff every row links and hashes correctly."""
    rows = conn.execute(
        "SELECT * FROM human_reviews WHERE competition_id = ? ORDER BY review_id",
        (competition_id,),
    ).fetchall()
    expected_prev = _chain_genesis(competition_id)
    for row in rows:
        if row["prev_hash"] != expected_prev:
            return False
        content = _review_content_json(
            row["competition_id"],
            row["contender_id"],
            row["action"],
            row["reviewer"],
            row["reason"],
            row["detail_json"],
            row["supersedes_review_id"],
            row["created_at"],
        )
        if row["integrity_hash"] != _chain_hash(expected_prev, content):
            return False
        expected_prev = row["integrity_hash"]
    return True


def list_reviews(conn: sqlite3.Connection, competition_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM human_reviews WHERE competition_id = ? ORDER BY review_id",
        (competition_id,),
    ).fetchall()


def effective_reviews(conn: sqlite3.Connection, competition_id: str) -> list[sqlite3.Row]:
    """Reviews not superseded by any later review of the SAME competition (append-only
    supersedes resolution). The schema's composite FK already forbids cross-competition
    supersession; the same-competition predicate here keeps the query correct even
    against a database that lost that constraint."""
    return conn.execute(
        """SELECT r.* FROM human_reviews r
           WHERE r.competition_id = ?
             AND NOT EXISTS (
                 SELECT 1 FROM human_reviews s
                 WHERE s.supersedes_review_id = r.review_id
                   AND s.competition_id = r.competition_id)
           ORDER BY r.review_id""",
        (competition_id,),
    ).fetchall()
