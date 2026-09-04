"""Schema-v14 executable baseline ledger.

The reward window and the serving baseline are intentionally different state
machines.  A reward window may end after seven days; the executable selected by
this ledger remains active until a verified replacement is activated or an
operator appends an explicit rollback version.

There are exactly two protocol tracks and both start at version zero.  Genesis is
not an empty/implicit baseline: :func:`seed_genesis_baselines` verifies and
publishes an archived reference implementation and its provenance for each track
before inserting either row.  Later CROWN activation is private to the verified
promotion pipeline; there is no public "promote this candidate" primitive.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidaio.audit.canonical import SHA256_HEX_PATTERN
from vidaio.audit.store import (
    ArtifactKind,
    ArtifactRef,
    AuditStore,
    IntegrityError,
    SEALED_KINDS,
    backend_key,
)
from vidaio.competition.interfaces import logical_build_identity
from vidaio.registry.registry import RegistryError, iso, transaction

SUPPORTED_TRACKS: tuple[str, str] = ("compression", "upscaling")
_SHA1_HEX_PATTERN = r"^[0-9a-f]{40}$"


class BaselineRegistryError(RegistryError):
    """Base class for schema-v14 baseline state failures."""


class GenesisBaselineError(BaselineRegistryError):
    """Genesis seeds are incomplete, conflicting, missing, or corrupt."""


class BaselineRollbackError(BaselineRegistryError):
    """An explicit baseline rollback is invalid."""


class PendingPromotionError(BaselineRegistryError):
    """A track has an unresolved CROWN promotion and must not start a new cycle."""


class GenesisBaseline(BaseModel):
    """Archived public reference implementation used as version zero."""

    model_config = ConfigDict(frozen=True)

    track: str
    artifact: ArtifactRef
    image_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    provenance: ArtifactRef
    repo_url: str = Field(min_length=1)
    commit_sha: str = Field(pattern=_SHA1_HEX_PATTERN)
    tree_sha: str = Field(pattern=_SHA1_HEX_PATTERN)

    @model_validator(mode="after")
    def _valid_seed(self) -> "GenesisBaseline":
        if self.track not in SUPPORTED_TRACKS:
            raise ValueError(
                f"unsupported baseline track {self.track!r}; expected {SUPPORTED_TRACKS}"
            )
        if self.artifact.byte_size <= 0:
            raise ValueError("genesis executable must be non-empty")
        if self.provenance.byte_size <= 0:
            raise ValueError("genesis provenance must be non-empty")
        expected_image = logical_build_identity(
            repo_url=self.repo_url,
            commit_sha=self.commit_sha,
            tree_sha=self.tree_sha,
        )
        if self.image_digest != expected_image:
            raise ValueError(
                "genesis image_digest must use the stable logical build identity "
                "for its exact repository/commit/tree"
            )
        return self


@dataclass(frozen=True)
class BaselineRecord:
    baseline_id: int
    track: str
    version: int
    artifact_digest: str
    artifact_kind: str
    artifact_bytes: int
    image_digest: str
    provenance_digest: str
    provenance_kind: str
    provenance_bytes: int
    repo_url: str
    commit_sha: str
    tree_sha: str
    source_kind: str
    source_epoch_id: str | None
    source_snapshot_digest: str | None
    source_anchor_block: int | None
    source_anchor_digest: str | None
    source_competition_id: str | None
    source_cycle: int | None
    winner_uid: int | None
    winner_hotkey: str | None
    winner_score: float | None
    winner_margin: float | None
    compared_baseline_version: int | None
    compared_baseline_score: float | None
    compared_baseline_digest: str | None
    status: str
    reinstated_version: int | None
    rollback_reason: str | None
    activated_at: datetime

    @staticmethod
    def from_row(row: sqlite3.Row) -> "BaselineRecord":
        return BaselineRecord(
            baseline_id=int(row["baseline_id"]),
            track=str(row["track"]),
            version=int(row["version"]),
            artifact_digest=str(row["artifact_digest"]),
            artifact_kind=str(row["artifact_kind"]),
            artifact_bytes=int(row["artifact_bytes"]),
            image_digest=str(row["image_digest"]),
            provenance_digest=str(row["provenance_digest"]),
            provenance_kind=str(row["provenance_kind"]),
            provenance_bytes=int(row["provenance_bytes"]),
            repo_url=str(row["repo_url"]),
            commit_sha=str(row["commit_sha"]),
            tree_sha=str(row["tree_sha"]),
            source_kind=str(row["source_kind"]),
            source_epoch_id=row["source_epoch_id"],
            source_snapshot_digest=row["source_snapshot_digest"],
            source_anchor_block=(
                None
                if row["source_anchor_block"] is None
                else int(row["source_anchor_block"])
            ),
            source_anchor_digest=row["source_anchor_digest"],
            source_competition_id=row["source_competition_id"],
            source_cycle=(
                None if row["source_cycle"] is None else int(row["source_cycle"])
            ),
            winner_uid=None if row["winner_uid"] is None else int(row["winner_uid"]),
            winner_hotkey=row["winner_hotkey"],
            winner_score=(
                None if row["winner_score"] is None else float(row["winner_score"])
            ),
            winner_margin=(
                None if row["winner_margin"] is None else float(row["winner_margin"])
            ),
            compared_baseline_version=(
                None
                if row["compared_baseline_version"] is None
                else int(row["compared_baseline_version"])
            ),
            compared_baseline_score=(
                None
                if row["compared_baseline_score"] is None
                else float(row["compared_baseline_score"])
            ),
            compared_baseline_digest=row["compared_baseline_digest"],
            status=str(row["status"]),
            reinstated_version=(
                None
                if row["reinstated_version"] is None
                else int(row["reinstated_version"])
            ),
            rollback_reason=row["rollback_reason"],
            activated_at=datetime.fromisoformat(str(row["activated_at"])),
        )

    def artifact_ref(self) -> ArtifactRef:
        kind = ArtifactKind(self.artifact_kind)
        return ArtifactRef(
            digest=self.artifact_digest,
            kind=kind,
            byte_size=self.artifact_bytes,
            backend_key=backend_key(kind, self.artifact_digest),
        )

    def provenance_ref(self) -> ArtifactRef:
        kind = ArtifactKind(self.provenance_kind)
        return ArtifactRef(
            digest=self.provenance_digest,
            kind=kind,
            byte_size=self.provenance_bytes,
            backend_key=backend_key(kind, self.provenance_digest),
        )


@dataclass(frozen=True)
class PromotionLatch:
    latch_id: int
    track: str
    snapshot_digest: str
    competition_id: str
    epoch_id: str
    cycle: int
    anchor_block: int
    anchor_digest: str
    winner_uid: int
    winner_hotkey: str
    compared_baseline_version: int
    compared_baseline_digest: str
    status: str
    promoted_baseline_id: int | None
    latched_at: datetime
    resolved_at: datetime | None

    @staticmethod
    def from_row(row: sqlite3.Row) -> "PromotionLatch":
        return PromotionLatch(
            latch_id=int(row["latch_id"]),
            track=str(row["track"]),
            snapshot_digest=str(row["snapshot_digest"]),
            competition_id=str(row["competition_id"]),
            epoch_id=str(row["epoch_id"]),
            cycle=int(row["cycle"]),
            anchor_block=int(row["anchor_block"]),
            anchor_digest=str(row["anchor_digest"]),
            winner_uid=int(row["winner_uid"]),
            winner_hotkey=str(row["winner_hotkey"]),
            compared_baseline_version=int(row["compared_baseline_version"]),
            compared_baseline_digest=str(row["compared_baseline_digest"]),
            status=str(row["status"]),
            promoted_baseline_id=(
                None
                if row["promoted_baseline_id"] is None
                else int(row["promoted_baseline_id"])
            ),
            latched_at=datetime.fromisoformat(str(row["latched_at"])),
            resolved_at=(
                None
                if row["resolved_at"] is None
                else datetime.fromisoformat(str(row["resolved_at"]))
            ),
        )


def current_baseline(conn: sqlite3.Connection, track: str) -> BaselineRecord | None:
    _require_track(track)
    row = conn.execute(
        "SELECT * FROM baselines WHERE track = ? AND status = 'active'", (track,)
    ).fetchone()
    return BaselineRecord.from_row(row) if row is not None else None


def baseline_version(
    conn: sqlite3.Connection, track: str, version: int
) -> BaselineRecord | None:
    _require_track(track)
    row = conn.execute(
        "SELECT * FROM baselines WHERE track = ? AND version = ?", (track, version)
    ).fetchone()
    return BaselineRecord.from_row(row) if row is not None else None


def baseline_history(conn: sqlite3.Connection, track: str) -> list[BaselineRecord]:
    _require_track(track)
    rows = conn.execute(
        "SELECT * FROM baselines WHERE track = ? ORDER BY version", (track,)
    ).fetchall()
    return [BaselineRecord.from_row(row) for row in rows]


def baseline_events(conn: sqlite3.Connection, track: str) -> list[sqlite3.Row]:
    _require_track(track)
    return conn.execute(
        "SELECT * FROM baseline_events WHERE track = ? ORDER BY event_id", (track,)
    ).fetchall()


def pending_promotion(conn: sqlite3.Connection, track: str) -> PromotionLatch | None:
    _require_track(track)
    row = conn.execute(
        "SELECT * FROM baseline_promotion_latches "
        "WHERE track = ? AND status = 'pending'",
        (track,),
    ).fetchone()
    return PromotionLatch.from_row(row) if row is not None else None


def promotion_latch_by_key(
    conn: sqlite3.Connection,
    *,
    snapshot_digest: str,
    competition_id: str,
    track: str,
) -> PromotionLatch | None:
    _require_track(track)
    row = conn.execute(
        "SELECT * FROM baseline_promotion_latches "
        "WHERE snapshot_digest = ? AND competition_id = ? AND track = ?",
        (snapshot_digest, competition_id, track),
    ).fetchone()
    return PromotionLatch.from_row(row) if row is not None else None


def require_no_pending_promotion(conn: sqlite3.Connection, track: str) -> None:
    """Orchestrator interlock called before scheduling the next competition."""
    latch = pending_promotion(conn, track)
    if latch is not None:
        raise PendingPromotionError(
            f"{track} has unresolved CROWN promotion from competition "
            f"{latch.competition_id!r} / snapshot {latch.snapshot_digest}; "
            "the next competition must not start until that executable is promoted"
        )


def seed_genesis_baselines(
    conn: sqlite3.Connection,
    store: AuditStore,
    seeds: Sequence[GenesisBaseline],
    now: datetime,
) -> Mapping[str, BaselineRecord]:
    """Install both archived public v0 implementations, atomically and idempotently.

    Verification and publication happen before the database transaction.  If any
    artifact is absent/corrupt or cannot be published, neither track is inserted.
    A retry with byte-identical seeds is a no-op; a retry with different provenance
    is a hard conflict rather than a silent genesis rewrite.
    """
    by_track = {seed.track: seed for seed in seeds}
    if len(by_track) != len(seeds):
        raise GenesisBaselineError("genesis includes a duplicate track")
    if set(by_track) != set(SUPPORTED_TRACKS):
        raise GenesisBaselineError(
            "genesis must seed exactly compression and upscaling; got "
            f"{sorted(by_track)}"
        )

    existing = {track: baseline_version(conn, track, 0) for track in SUPPORTED_TRACKS}
    if any(record is not None for record in existing.values()):
        if not all(record is not None for record in existing.values()):
            raise GenesisBaselineError(
                "partial genesis detected: both track-v0 rows must exist together"
            )
        for track, record in existing.items():
            assert record is not None
            seed = by_track[track]
            if not _seed_matches(record, seed):
                raise GenesisBaselineError(
                    f"{track} v0 already exists with different archived identity"
                )
        return {
            track: record for track, record in existing.items() if record is not None
        }

    for seed in by_track.values():
        _verify_ref(store, seed.artifact, what=f"{seed.track} genesis executable")
        _verify_ref(store, seed.provenance, what=f"{seed.track} genesis provenance")
    for seed in by_track.values():
        _publish(store, seed.artifact, what=f"{seed.track} genesis executable")
        _publish(store, seed.provenance, what=f"{seed.track} genesis provenance")

    ts = iso(now)
    with transaction(conn):
        # Re-read under the write lock so concurrent identical initialization is
        # idempotent and a conflicting one cannot interleave.
        concurrent = {
            track: baseline_version(conn, track, 0) for track in SUPPORTED_TRACKS
        }
        if any(record is not None for record in concurrent.values()):
            if not all(record is not None for record in concurrent.values()):
                raise GenesisBaselineError(
                    "concurrent initialization left partial genesis"
                )
            for track, record in concurrent.items():
                assert record is not None
                if not _seed_matches(record, by_track[track]):
                    raise GenesisBaselineError(
                        f"concurrent {track} v0 uses a different archived identity"
                    )
            return {
                track: record
                for track, record in concurrent.items()
                if record is not None
            }
        for track in SUPPORTED_TRACKS:
            seed = by_track[track]
            cur = conn.execute(
                """INSERT INTO baselines
                   (track, version, artifact_digest, artifact_kind, artifact_bytes,
                    image_digest, provenance_digest, provenance_kind,
                    provenance_bytes, repo_url, commit_sha, tree_sha, source_kind,
                    status, activated_at, updated_at)
                   VALUES (?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'genesis',
                           'active', ?, ?)""",
                (
                    track,
                    seed.artifact.digest,
                    seed.artifact.kind.value,
                    seed.artifact.byte_size,
                    seed.image_digest,
                    seed.provenance.digest,
                    seed.provenance.kind.value,
                    seed.provenance.byte_size,
                    seed.repo_url,
                    seed.commit_sha,
                    seed.tree_sha,
                    ts,
                    ts,
                ),
            )
            _record_event(
                conn,
                track,
                "baseline_genesis_seeded",
                0,
                now,
                payload={
                    "artifact_digest": seed.artifact.digest,
                    "image_digest": seed.image_digest,
                    "provenance_digest": seed.provenance.digest,
                    "commit_sha": seed.commit_sha,
                    "tree_sha": seed.tree_sha,
                    "baseline_id": int(cur.lastrowid),
                },
            )
    result = {track: baseline_version(conn, track, 0) for track in SUPPORTED_TRACKS}
    assert all(record is not None for record in result.values())
    return {track: record for track, record in result.items() if record is not None}


def rollback_baseline(
    conn: sqlite3.Connection,
    track: str,
    to_version: int,
    reason: str,
    now: datetime,
) -> BaselineRecord:
    """Append a serving rollback; never rewrite/reactivate an historical row."""
    _require_track(track)
    if not reason.strip():
        raise BaselineRollbackError("baseline rollback requires a non-empty reason")
    with transaction(conn):
        target = baseline_version(conn, track, to_version)
        if target is None:
            raise BaselineRollbackError(
                f"unknown {track} baseline version {to_version}"
            )
        active = current_baseline(conn, track)
        if active is None:
            raise BaselineRollbackError(f"{track} has no active baseline")
        if active.version == to_version:
            raise BaselineRollbackError(
                f"{track} baseline v{to_version} is already active"
            )
        version = _next_version(conn, track)
        ts = iso(now)
        conn.execute(
            "UPDATE baselines SET status = 'rolled_back', updated_at = ? "
            "WHERE baseline_id = ?",
            (ts, active.baseline_id),
        )
        cur = conn.execute(
            """INSERT INTO baselines
               (track, version, artifact_digest, artifact_kind, artifact_bytes,
                image_digest, provenance_digest, provenance_kind, provenance_bytes,
                repo_url, commit_sha, tree_sha, source_kind, source_epoch_id,
                source_snapshot_digest,
                source_anchor_block, source_anchor_digest, source_competition_id,
                source_cycle, winner_uid, winner_hotkey, winner_score, winner_margin,
                compared_baseline_version, compared_baseline_score,
                compared_baseline_digest, status,
                reinstated_version, rollback_reason, activated_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'rollback', ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)""",
            (
                track,
                version,
                target.artifact_digest,
                target.artifact_kind,
                target.artifact_bytes,
                target.image_digest,
                target.provenance_digest,
                target.provenance_kind,
                target.provenance_bytes,
                target.repo_url,
                target.commit_sha,
                target.tree_sha,
                target.source_epoch_id,
                target.source_snapshot_digest,
                target.source_anchor_block,
                target.source_anchor_digest,
                target.source_competition_id,
                target.source_cycle,
                target.winner_uid,
                target.winner_hotkey,
                target.winner_score,
                target.winner_margin,
                target.compared_baseline_version,
                target.compared_baseline_score,
                target.compared_baseline_digest,
                to_version,
                reason.strip(),
                ts,
                ts,
            ),
        )
        _record_event(
            conn,
            track,
            "baseline_rolled_back",
            version,
            now,
            snapshot_digest=target.source_snapshot_digest,
            payload={
                "baseline_id": int(cur.lastrowid),
                "reinstated_version": to_version,
                "replaced_version": active.version,
                "reason": reason.strip(),
            },
        )
        result = baseline_version(conn, track, version)
    assert result is not None
    return result


def baseline_invariant_violations(conn: sqlite3.Connection) -> list[str]:
    """Return missing/multiple-active defects, including an unseeded track."""
    rows = {
        str(row["track"]): int(row["actives"])
        for row in conn.execute(
            "SELECT track, SUM(status = 'active') AS actives "
            "FROM baselines GROUP BY track"
        ).fetchall()
    }
    violations: list[str] = []
    for track in SUPPORTED_TRACKS:
        count = rows.get(track, 0)
        if count != 1:
            violations.append(
                f"track {track!r} has {count} active baselines, expected exactly 1"
            )
    unknown = sorted(set(rows) - set(SUPPORTED_TRACKS))
    violations.extend(
        f"unsupported baseline track {track!r} exists" for track in unknown
    )
    return violations


def verify_baseline_invariants(
    conn: sqlite3.Connection, log: logging.Logger | None = None
) -> list[str]:
    violations = baseline_invariant_violations(conn)
    logger = log or logging.getLogger("vidaio.registry.baseline")
    for violation in violations:
        logger.critical("baseline registry invariant violated: %s", violation)
    return violations


# ---- package-private write helpers used by the CROWN pipeline -----------------


def _next_version(conn: sqlite3.Connection, track: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(version), -1) AS version FROM baselines WHERE track = ?",
        (track,),
    ).fetchone()
    return int(row["version"]) + 1


def _record_event(
    conn: sqlite3.Connection,
    track: str,
    event_type: str,
    version: int | None,
    now: datetime,
    *,
    payload: Mapping[str, object],
    snapshot_digest: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO baseline_events
           (track, event_type, version, snapshot_digest, payload_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            track,
            event_type,
            version,
            snapshot_digest,
            json.dumps(dict(payload), sort_keys=True, separators=(",", ":")),
            iso(now),
        ),
    )


def _verify_ref(store: AuditStore, ref: ArtifactRef, *, what: str) -> None:
    if ref.byte_size <= 0:
        raise GenesisBaselineError(f"{what} is empty")
    digest = hashlib.sha256()
    size = 0
    try:
        with contextlib.closing(store.open_stream(ref)) as stream:
            while chunk := stream.read(1 << 20):
                size += len(chunk)
                if size > ref.byte_size:
                    raise IntegrityError(
                        f"{what} exceeds committed size {ref.byte_size}"
                    )
                digest.update(chunk)
    except (FileNotFoundError, OSError, IntegrityError) as exc:
        raise GenesisBaselineError(f"{what} is not verifiably archived: {exc}") from exc
    if size != ref.byte_size or digest.hexdigest() != ref.digest:
        raise GenesisBaselineError(
            f"{what} bytes do not match its content-addressed reference"
        )


def _publish(store: AuditStore, ref: ArtifactRef, *, what: str) -> None:
    # Non-sealed kinds already live at their public content address. ``release``
    # is intentionally defined only for encrypted holdouts.
    if ref.kind not in SEALED_KINDS:
        _verify_ref(store, ref, what=what)
        return
    try:
        store.release(ref)
        if not store.is_released(ref):
            raise IntegrityError("release marker/public copy is absent")
    except (FileNotFoundError, OSError, IntegrityError) as exc:
        raise GenesisBaselineError(f"{what} could not be published: {exc}") from exc


def _seed_matches(record: BaselineRecord, seed: GenesisBaseline) -> bool:
    return (
        record.source_kind == "genesis"
        and record.version == 0
        and record.artifact_digest == seed.artifact.digest
        and record.artifact_kind == seed.artifact.kind.value
        and record.artifact_bytes == seed.artifact.byte_size
        and record.image_digest == seed.image_digest
        and record.provenance_digest == seed.provenance.digest
        and record.provenance_kind == seed.provenance.kind.value
        and record.provenance_bytes == seed.provenance.byte_size
        and record.repo_url == seed.repo_url
        and record.commit_sha == seed.commit_sha
        and record.tree_sha == seed.tree_sha
    )


def _require_track(track: str) -> None:
    if track not in SUPPORTED_TRACKS:
        raise BaselineRegistryError(
            f"unsupported baseline track {track!r}; expected {SUPPORTED_TRACKS}"
        )


__all__ = [
    "SUPPORTED_TRACKS",
    "BaselineRecord",
    "BaselineRegistryError",
    "BaselineRollbackError",
    "GenesisBaseline",
    "GenesisBaselineError",
    "PendingPromotionError",
    "PromotionLatch",
    "baseline_events",
    "baseline_history",
    "baseline_invariant_violations",
    "baseline_version",
    "current_baseline",
    "pending_promotion",
    "promotion_latch_by_key",
    "require_no_pending_promotion",
    "rollback_baseline",
    "seed_genesis_baselines",
    "verify_baseline_invariants",
]
