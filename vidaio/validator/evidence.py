"""Score-packet evidence — the validator's half of the auditable inference path.

review service-review #7: the validator persisted only numeric EWMAs, so the
packet bytes behind a published weight vector were discarded and every real
inference publication anchored the "no score packets" sentinel. That made the
published weights unreproducible by a third party — a direct violation of the
the project design record integrity invariant "every scored metric must be independently
recomputable from the audit store".

`InferenceValidator` now writes the exact packet bytes + digest per
(round, uid, item) into the validator DB (inside the round's single atomic
transaction) and archives them as SCORE_PACKET artifacts when an
audit store is configured. This module is the READER over that evidence:

    ScorePacketEvidence(conn).recent_packet_digests(since)

It is also a structural `vidaio.weightsetter.PublicationInputs`
(`score_packet_digests()`), so the weight-setter consumes it with no wiring
change and real publications carry the real merkle set.

Only COMMITTED rounds are ever returned: a round whose transaction never
finished has committed_at NULL in the round ledger and must not contribute
evidence to a publication.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

#: Default lookback for the PublicationInputs adapter when the caller supplies no
#: cutoff: one day of rounds (validator cadence is 1–2 h, so this spans a handful).
DEFAULT_LOOKBACK_SECONDS = 24 * 3600.0


def _as_utc_iso(value: datetime | str | None) -> str | None:
    """Normalize a cutoff to the canonical UTC ISO form rows are stored in.

    Rows are written with `miner_manager.utc_now_iso()` (always '+00:00'), so
    normalized ISO strings compare chronologically as plain strings.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


class ScorePacketEvidence:
    """Read-only view of persisted score packets (see module docstring).

    The weight-setter runs as its OWN process (spec §13: the DB is the only
    shared state), so it constructs this over its own connection to the
    validator's database file — never over the validator's connection.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        lookback_seconds: float = DEFAULT_LOOKBACK_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._conn = conn
        self._lookback_seconds = lookback_seconds
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))

    # -- PublicationInputs surface --------------------------------------------

    def recent_packet_digests(self, since: datetime | str | None = None) -> list[str]:
        """Distinct packet digests from COMMITTED rounds at/after `since`.

        `since=None` means "every committed round on record". Returned sorted so
        the caller's merkle root is reproducible from this query alone.
        """
        sql = (
            "SELECT DISTINCT p.packet_digest AS d FROM score_packets p"
            " JOIN rounds r ON r.round_id = p.round_id"
            " WHERE r.committed_at IS NOT NULL"
        )
        params: list[object] = []
        cutoff = _as_utc_iso(since)
        if cutoff is not None:
            sql += " AND p.created_at >= ?"
            params.append(cutoff)
        return sorted(str(row["d"]) for row in self._conn.execute(sql, params))

    def score_packet_digests(self) -> Sequence[str]:
        """PublicationInputs conformance: the digests inside the lookback window."""
        since = self._clock() - timedelta(seconds=self._lookback_seconds)
        return self.recent_packet_digests(since)

    # -- inspection ------------------------------------------------------------

    def packets(
        self,
        since: datetime | str | None = None,
        *,
        until: datetime | str | None = None,
        through_block: int | None = None,
    ) -> list[sqlite3.Row]:
        """Full committed evidence rows inside an optional inclusive time window.

        ``until`` prevents packets committed after an epoch close timestamp from
        leaking into a catch-up finalization. Both packet and round timestamps are
        bounded, and partial rounds remain excluded.
        """
        sql = (
            "SELECT p.* FROM score_packets p JOIN rounds r ON r.round_id = p.round_id"
            " WHERE r.committed_at IS NOT NULL"
        )
        params: list[object] = []
        cutoff = _as_utc_iso(since)
        if cutoff is not None:
            sql += " AND p.created_at >= ?"
            params.append(cutoff)
        upper = _as_utc_iso(until)
        if upper is not None:
            sql += " AND p.created_at <= ? AND r.committed_at <= ?"
            params.extend((upper, upper))
        if through_block is not None:
            if through_block < 0:
                raise ValueError(
                    f"through_block must be non-negative, got {through_block}"
                )
            sql += " AND r.block <= ?"
            params.append(through_block)
        return self._conn.execute(sql + " ORDER BY p.created_at, p.uid", params).fetchall()

    def has_uncommitted_round(self) -> bool:
        """True when a partial round is detectable — its evidence is excluded."""
        row = self._conn.execute(
            "SELECT 1 FROM rounds WHERE committed_at IS NULL LIMIT 1"
        ).fetchone()
        return row is not None

    def has_uncommitted_round_through(self, block: int) -> bool:
        """Whether a round assigned to ``block`` or earlier is still incomplete."""
        if block < 0:
            raise ValueError(f"block must be non-negative, got {block}")
        row = self._conn.execute(
            "SELECT 1 FROM rounds WHERE committed_at IS NULL AND block <= ? LIMIT 1",
            (block,),
        ).fetchone()
        return row is not None

    def reference_original_refs(self, challenge_id: str) -> list[str]:
        """Distinct archived holdout refs for a terminal challenge.

        Only committed score rows count.  The validator calls this after the
        challenge service confirms retirement, then publishes those exact refs
        through the audit store's post-retirement release namespace.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT p.reference_original_ref AS ref"
            " FROM score_packets p JOIN rounds r ON r.round_id = p.round_id"
            " WHERE r.committed_at IS NOT NULL AND p.challenge_id = ?"
            " AND p.reference_original_ref IS NOT NULL"
            " ORDER BY p.reference_original_ref",
            (challenge_id,),
        ).fetchall()
        return [str(row["ref"]) for row in rows]


class AvailabilityFoldEvidence:
    """Read-only committed availability observations for v12 epoch assembly."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def observations(
        self,
        since: datetime | str | None = None,
        *,
        until: datetime | str | None = None,
        through_block: int | None = None,
    ) -> list[sqlite3.Row]:
        """Committed observations in deterministic fold order.

        The authority resolves each row's actual ordering key from the immutable
        pre-dispatch challenge commitment, just as it does for media score packets.
        """
        sql = (
            "SELECT a.* FROM availability_folds a"
            " JOIN rounds r ON r.round_id = a.round_id"
            " WHERE r.committed_at IS NOT NULL"
        )
        params: list[object] = []
        cutoff = _as_utc_iso(since)
        if cutoff is not None:
            sql += " AND a.created_at >= ?"
            params.append(cutoff)
        upper = _as_utc_iso(until)
        if upper is not None:
            sql += " AND a.created_at <= ? AND r.committed_at <= ?"
            params.extend((upper, upper))
        if through_block is not None:
            if through_block < 0:
                raise ValueError(
                    f"through_block must be non-negative, got {through_block}"
                )
            sql += " AND r.block <= ?"
            params.append(through_block)
        return self._conn.execute(
            sql + " ORDER BY a.created_at, a.uid, a.item_id", params
        ).fetchall()
