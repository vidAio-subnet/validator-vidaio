"""Model registry — archived champions per track + promotion bookkeeping (design spec §20).

The registry is the ratchet's memory: one active champion per track, every
past champion kept forever (superseded, never deleted), versions strictly
monotonic per track. The champion executable itself lives in the audit store
(LocalFsStore / Hippius) — the registry row holds the content-addressed
ArtifactRef fields (digest, kind, byte size), so anyone with store access can
fetch the exact archived bytes back out.

Promotion guard: a candidate must strictly beat the current champion's
RECORDED holdout score (or be the track's first champion). Equal is not
better — the champion is the quality floor and the floor only ever rises.

ATOMICITY. A promotion is three writes — supersede the reigning row, insert the
replacement, append the event — and a rollback the same shape. Run as separate
autocommits, a crash between the first two leaves the track with NO active
champion, and the retry then reads "no reigning champion" and skips the quality
floor entirely: the ratchet silently unlatches. Every promote/rollback therefore
runs inside ONE `BEGIN IMMEDIATE` transaction (see `transaction`), so either the
whole handover lands or none of it does.

The schema carries the matching invariant as a partial unique index
(`ux_champions_one_active`), and `active_invariant_violations` re-checks it —
as a startup verification that logs CRITICAL, and as a test assertion.

All writes take an explicit timezone-aware `now` (injected clock discipline,
matching vidaio.competition.repository).
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field

from vidaio.audit.canonical import SHA256_HEX_PATTERN
from vidaio.audit.store import ArtifactKind, ArtifactRef, backend_key
from vidaio.core.db import apply_migrations

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class RegistryError(Exception):
    """Base class for registry/promotion failures."""


class LegacyRegistryWriteDisabledError(RegistryError):
    """Schema-v13 candidate writes are disabled for the schema-v14 release."""


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """One `BEGIN IMMEDIATE` unit of work; re-entrant (nests as a no-op).

    Connections come from vidaio.core.db.connect with isolation_level=None, so
    without this each execute() is separately durable — see module docstring.
    """
    if conn.in_transaction:
        yield
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


class ChampionNotBeatenError(RegistryError):
    """Candidate's holdout score does not strictly beat the reigning champion's."""


class RollbackError(RegistryError):
    """Rollback target is unknown, or is already the serving champion."""


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Apply this module's migrations. Safe to call at every startup."""
    return apply_migrations(conn, MIGRATIONS_DIR)


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("naive datetime passed to registry persistence")
    return dt.astimezone(timezone.utc).isoformat()


class ChampionCandidate(BaseModel):
    """A holdout winner proposed for promotion (built by PromotionPipeline)."""

    model_config = ConfigDict(frozen=True)

    track: str
    artifact_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    artifact_kind: ArtifactKind
    artifact_bytes: int = Field(ge=0)
    source_competition_id: str
    contender_hotkey: str
    #: Verified hidden-holdout score in [0, 1] (finite by pydantic).
    holdout_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    audit_bundle_digest: str = Field(pattern=SHA256_HEX_PATTERN)


@dataclass(frozen=True)
class ChampionRecord:
    champion_id: int
    track: str
    version: int
    artifact_digest: str
    artifact_kind: ArtifactKind
    artifact_bytes: int
    source_competition_id: str
    contender_hotkey: str
    holdout_score: float
    audit_bundle_digest: str
    status: str
    reinstated_version: int | None
    rollback_reason: str | None
    promoted_at: datetime

    @staticmethod
    def from_row(row: sqlite3.Row) -> "ChampionRecord":
        return ChampionRecord(
            champion_id=row["champion_id"],
            track=row["track"],
            version=row["version"],
            artifact_digest=row["artifact_digest"],
            artifact_kind=ArtifactKind(row["artifact_kind"]),
            artifact_bytes=row["artifact_bytes"],
            source_competition_id=row["source_competition_id"],
            contender_hotkey=row["contender_hotkey"],
            holdout_score=row["holdout_score"],
            audit_bundle_digest=row["audit_bundle_digest"],
            status=row["status"],
            reinstated_version=row["reinstated_version"],
            rollback_reason=row["rollback_reason"],
            promoted_at=datetime.fromisoformat(row["promoted_at"]),
        )

    def artifact_ref(self) -> ArtifactRef:
        """Reconstruct the audit-store handle to the archived executable."""
        return ArtifactRef(
            digest=self.artifact_digest,
            kind=self.artifact_kind,
            byte_size=self.artifact_bytes,
            backend_key=backend_key(self.artifact_kind, self.artifact_digest),
        )


# ---- queries -------------------------------------------------------------------

def current(conn: sqlite3.Connection, track: str) -> ChampionRecord | None:
    """The serving champion for a track (None before the first promotion)."""
    row = conn.execute(
        "SELECT * FROM champions WHERE track = ? AND status = 'active'", (track,)
    ).fetchone()
    return ChampionRecord.from_row(row) if row is not None else None


def get_version(conn: sqlite3.Connection, track: str, version: int) -> ChampionRecord | None:
    row = conn.execute(
        "SELECT * FROM champions WHERE track = ? AND version = ?", (track, version)
    ).fetchone()
    return ChampionRecord.from_row(row) if row is not None else None


def history(conn: sqlite3.Connection, track: str) -> list[ChampionRecord]:
    rows = conn.execute(
        "SELECT * FROM champions WHERE track = ? ORDER BY version", (track,)
    ).fetchall()
    return [ChampionRecord.from_row(r) for r in rows]


# ---- writes --------------------------------------------------------------------

def promote(
    conn: sqlite3.Connection, track: str, candidate: ChampionCandidate, now: datetime
) -> ChampionRecord:
    """Disabled schema-v13 write entrypoint retained only for import compatibility.

    Executable state can now change only through
    :class:`vidaio.registry.baseline_promotion.BaselinePromotionPipeline`, which
    derives the machine winner from anchored current-schema CROWN evidence.
    """
    raise LegacyRegistryWriteDisabledError(
        "direct schema-v13 candidate promotion is disabled; use the verified "
        "current-schema CROWN baseline pipeline"
    )


def rollback(
    conn: sqlite3.Connection, track: str, to_version: int, reason: str, now: datetime
) -> ChampionRecord:
    """Disabled legacy-table rollback retained only for import compatibility.

    Schema-v14 rollbacks use :func:`vidaio.registry.baseline.rollback_baseline`.
    """
    raise LegacyRegistryWriteDisabledError(
        "schema-v13 registry rollback is disabled; use rollback_baseline"
    )


def list_events(conn: sqlite3.Connection, track: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM registry_events WHERE track = ? ORDER BY event_id", (track,)
    ).fetchall()


# ---- invariant -----------------------------------------------------------------

def active_invariant_violations(conn: sqlite3.Connection) -> list[str]:
    """Tracks that do not have EXACTLY ONE active champion. Empty = healthy.

    Every track that has ever had a champion must have exactly one serving now:
    zero means a promotion/rollback tore mid-handover and the quality floor is
    unguarded (a retry would see "no reigning champion" and skip the score
    guard); more than one means two backends both claim to be the champion.
    """
    rows = conn.execute(
        "SELECT track, SUM(status = 'active') AS actives"
        " FROM champions GROUP BY track ORDER BY track"
    ).fetchall()
    return [
        f"track {row['track']!r} has {int(row['actives'])} active champions, expected 1"
        for row in rows
        if int(row["actives"]) != 1
    ]


def verify_startup_invariants(
    conn: sqlite3.Connection, log: logging.Logger | None = None
) -> list[str]:
    """Startup check: log CRITICAL for every invariant violation. Returns them.

    Deliberately non-fatal — a registry that lost its champion still has to come
    up so an operator can roll forward — but it must SAY SO, loudly, once per
    boot rather than silently promoting the next candidate past a floor that is
    no longer there.
    """
    violations = active_invariant_violations(conn)
    logger = log or logging.getLogger("vidaio.registry")
    for violation in violations:
        logger.critical(
            "registry invariant violated: %s — the champion handover did not "
            "complete; the quality floor for this track is UNGUARDED until an "
            "operator promotes or rolls back explicitly",
            violation,
        )
    return violations
