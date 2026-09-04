"""Authoritative, READ-ONLY view of the competition database.

Promotion used to be handed a `CompetitionEvidence` object by its caller and
verify only that the object was self-consistent with the blobs it named. That
proves nothing: a caller who substitutes a winner, a bundle, a packet and an
artifact that all agree with each other passes every self-consistency check and
installs an arbitrary backend as the quality floor. The evidence has to come
from the competition's OWN tables, not from the promoter's word.

This module is that read. It queries the shipped competition schema
(`vidaio/competition/migrations/0001_schema.sql`) directly with plain SELECTs
and nothing else — no writes, no transitions, no imports from
`vidaio.competition.orchestrator` (the orchestrator is a separate service that
owns the WRITE side; coupling the registry to its runtime would make promotion
depend on a process that need not even be running).

WHAT IT READS, AND FROM WHERE — every field is a persisted column:

  competitions          competition_id, track, status, manifest_digest
                        (only a COMPLETED competition can promote)
  contenders            contender_id, hotkey, final_rank, final_score, status,
                        image_digest — the winner is the BEST-RANKED
                        non-calibration, non-disqualified, eligible row, exactly
                        the rule `competition.repository.ranking()` uses. The
                        schema forbids a final_rank on a calibration row, so a
                        calibration entry can never be selected.
  evaluation_items      item_id, item_index, challenge_id, scoring_item_id,
                        input_sha256, reference_sha256 — the packet identity and
                        exact evaluation-media identity of each holdout item.
  performance_history   score_packet_digest, audit_bundle_digest, item_score,
                        valid — the winner's per-item audit linkage. The
                        score_packet_digest column is NOT NULL by schema and
                        audit_bundle_digest is write-once, so these are the
                        competition's immutable record of which packet and which
                        bundle produced each score.
  events                the FINALIZING_SUBMISSIONS -> VALIDATING transition
                        (guard `submission_backup_completed`) carries the
                        audit-store backup reference; its absence means the
                        competition never archived any submission, so there is
                        no archived executable to promote.
  events                `contender_submission_archived`, PER CONTENDER: the
                        audit-store content address (and byte size) of that
                        contender's archived submission tarball. This is what
                        lets promotion DERIVE the winner's artifact instead of
                        taking the promoter's word for it.

THE ARTIFACT DIGEST IS DERIVED, NOT ASSERTED. There is still no per-contender
COLUMN for it — the competition schema records the winner's executable as a
stable logical BUILD identity (the versioned digest of exact repository, commit,
and tree source coordinates) and the
transition event carries only one COMBINED backup reference. But the write side
appends one `contender_submission_archived` event per archived contender
 whose payload IS the store
reference the tarball was `put` under, and finalization refuses to certify the
combined backup until every contender that can still win has one. So the
winner's artifact digest is readable from the competition's own append-only
event log, and `PromotionPipeline` requires the offered artifact to BE it
(:class:`~vidaio.registry.promotion.ArtifactLinkageError` otherwise) rather than
merely to verify in the store. A competition whose log predates the per-contender
event has `archived_artifact_digest is None`; promotion refuses that outright
rather than falling back to the weaker check, because a certified backup set with
no archived winner is a contradiction, not a legacy shape.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

#: The shipped competition schema this module reads. Exposed so a caller (or a
#: test) can migrate a competition database without importing the competition
#: package's runtime.
COMPETITION_MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[1] / "competition" / "migrations"
)

#: Terminal phase in which — and only in which — a competition has a winner.
COMPLETED = "COMPLETED"

#: `events.guard` of the FINALIZING_SUBMISSIONS -> VALIDATING transition, whose
#: payload carries the audit-store submission backup reference.
SUBMISSION_BACKUP_GUARD = "submission_backup_completed"

#: `events.event_type` of the PER-CONTENDER archival record. Its payload is
#: {"contender_id", "digest", "byte_size"} — the audit-store reference the
#: contender's submission tarball was stored under. Spelled as a literal (not
#: imported from `vidaio.competition.orchestrator.persistence`) on purpose: this
#: module reads the competition's PERSISTED schema and must not couple the
#: registry to the orchestrator's runtime. It is the same wire name, pinned by
#: tests on both sides.
SUBMISSION_ARCHIVED_EVENT = "contender_submission_archived"

#: `contenders.status` a contender must hold for its executable to exist.
BUILT = "BUILT"


@dataclass(frozen=True)
class ItemLinkage:
    """One (winner, holdout item) row: the competition's audit linkage for it.

    One instance exists for EVERY evaluation item of the competition, whether or
    not the winner has a `performance_history` row for it — `scored=False` marks
    the rowless shape (an internal review: an inner join used to silently OMIT such
    an item, so a holdout item the winner was never scored on simply vanished
    from verification while the promoted aggregate claimed to cover the holdout).
    """

    item_id: int
    item_index: int
    #: Challenge the item was minted from; the audit bundle must name it.
    challenge_id: str
    #: The packet-level item id (`evaluation_items.scoring_item_id`); the audit
    #: bundle's `item_id` and the packet's `item_id` must both equal this.
    scoring_item_id: str
    #: Exact miner-visible evaluation input. The audit bundle's challenge-input
    #: artifact must carry this digest.
    input_sha256: str
    #: Exact pristine/reference media. Compression normalizes this to
    #: ``input_sha256``; upscaling binds it to ``reference_original``.
    reference_sha256: str | None
    #: Sealed score-policy commitment copied into the audit bundle.
    threshold_commitment: str
    #: Upscaling-only manifest preimage fields; both are NULL for compression.
    upscale_factor: int | None
    item_commitment: str | None
    #: sha256 of the EXACT ItemScore packet bytes this score came from
    #: (None only when `scored` is False — there is no row to read it from).
    score_packet_digest: str | None
    #: sha256 of the audit bundle linked to this row (None = linkage gap).
    audit_bundle_digest: str | None
    #: The persisted per-item score the packet must reproduce (None = no row).
    item_score: float | None
    valid: bool
    #: False = the competition holds NO `performance_history` row for the winner
    #: on this evaluation item. Promotion treats that as an audit-linkage gap and
    #: refuses: an aggregate that claims the holdout cannot be verified against a
    #: holdout item nobody ever scored.
    scored: bool = True


@dataclass(frozen=True)
class WinnerFacts:
    """The competition's recorded holdout winner."""

    contender_id: int
    hotkey: str
    final_rank: int
    final_score: float
    status: str
    image_digest: str | None
    #: Audit-store content address of THIS contender's archived submission
    #: tarball, from the `contender_submission_archived` event. None = the
    #: competition never archived one for the winner, which promotion refuses.
    archived_artifact_digest: str | None = None
    #: Byte size recorded alongside it (0 when unknown).
    archived_artifact_bytes: int = 0


@dataclass(frozen=True)
class CompetitionFacts:
    """Everything promotion is allowed to believe about a competition."""

    competition_id: str
    track: str
    status: str
    #: Pre-enrollment competition policy manifest; every item bundle must name it.
    manifest_digest: str
    winner: WinnerFacts | None
    #: Every challenge id the competition's holdout was built from.
    holdout_challenge_ids: frozenset[str]
    #: The WINNER's per-item rows, in item order. Empty when the winner has no
    #: persisted scores at all.
    items: tuple[ItemLinkage, ...]
    #: Audit-store reference of the completed submission backup (None = the
    #: competition never recorded one).
    submission_backup_ref: str | None

    @property
    def completed(self) -> bool:
        return self.status == COMPLETED

    def linkage_for_bundle(self, digest: str) -> ItemLinkage | None:
        """The winner's item whose recorded bundle digest is `digest`."""
        for item in self.items:
            if item.audit_bundle_digest == digest:
                return item
        return None

    def audit_linkage_gaps(self) -> list[str]:
        """Winner items whose audit linkage is incomplete. Empty = fully linked.

        BOTH digests are required, because both are load-bearing: the bundle is
        the evidence object and the packet digest is what pins WHICH packet the
        persisted item score came from. A row carrying one but not the other can
        never be verified end to end, so it is a gap exactly like a null bundle
        (the aggregate holdout score includes that item's score either way —
        round-3 finding #6).
        """
        gaps: list[str] = []
        for item in self.items:
            if not item.scored:
                # Round-4 an internal review: an evaluation item with NO winner row at all.
                # It used to disappear through an inner join; it is a gap, and the
                # loudest kind — nothing about this item can be verified.
                gaps.append(
                    f"item {item.item_index} (item_id={item.item_id}): no "
                    "performance_history row for the winner — the item was never "
                    "scored, so the holdout aggregate cannot be audited over it"
                )
                continue
            missing = [
                name
                for name, digest in (
                    ("audit bundle", item.audit_bundle_digest),
                    ("score packet", item.score_packet_digest),
                )
                if not (digest or "").strip()
            ]
            if missing:
                gaps.append(
                    f"item {item.item_index} (item_id={item.item_id}): "
                    f"no linked {' and no linked '.join(missing)}"
                )
        return gaps


@runtime_checkable
class CompetitionSource(Protocol):
    """Where promotion gets its ground truth. Read-only, by construction."""

    def facts(self, competition_id: str) -> CompetitionFacts | None:
        """The competition's recorded outcome, or None if it does not exist."""
        ...


class SqliteCompetitionSource:
    """`CompetitionSource` over a competition SQLite database.

    Every statement is a SELECT; the connection is never written through. Use
    :meth:`open_read_only` in production so the operating system enforces that
    as well — the registry has no business mutating competition history.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row

    @classmethod
    def open_read_only(cls, path: str | Path) -> "SqliteCompetitionSource":
        """Open `path` in SQLite's read-only mode (URI `mode=ro`)."""
        conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True, timeout=30)
        conn.row_factory = sqlite3.Row
        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    # -- the read ------------------------------------------------------------

    def facts(self, competition_id: str) -> CompetitionFacts | None:
        row = self._conn.execute(
            "SELECT competition_id, track, status, manifest_digest "
            "FROM competitions WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        if row is None:
            return None
        winner = self._winner(competition_id)
        return CompetitionFacts(
            competition_id=row["competition_id"],
            track=row["track"],
            status=row["status"],
            manifest_digest=str(row["manifest_digest"]),
            winner=winner,
            holdout_challenge_ids=self._holdout_challenge_ids(competition_id),
            items=(
                self._winner_items(competition_id, winner.contender_id)
                if winner is not None
                else ()
            ),
            submission_backup_ref=self._submission_backup_ref(competition_id),
        )

    def _winner(self, competition_id: str) -> WinnerFacts | None:
        """Best-ranked payable contender — `repository.ranking()`'s first row.

        Calibration rows can never carry a final_rank (schema CHECK), so the
        non-earning calibration entry is excluded by construction rather than by a
        filter someone can forget.
        """
        row = self._conn.execute(
            """SELECT contender_id, hotkey, final_rank, final_score, status, image_digest
                 FROM contenders
                WHERE competition_id = ?
                  AND final_rank IS NOT NULL
                  AND is_calibration = 0
                  AND manual_disqualified = 0
                  AND eligible = 1
             ORDER BY final_rank, contender_id
                LIMIT 1""",
            (competition_id,),
        ).fetchone()
        if row is None or row["hotkey"] is None or row["final_score"] is None:
            return None
        contender_id = int(row["contender_id"])
        digest, byte_size = self._archived_submission(competition_id, contender_id)
        return WinnerFacts(
            contender_id=contender_id,
            hotkey=row["hotkey"],
            final_rank=int(row["final_rank"]),
            final_score=float(row["final_score"]),
            status=row["status"],
            image_digest=row["image_digest"],
            archived_artifact_digest=digest,
            archived_artifact_bytes=byte_size,
        )

    def _archived_submission(
        self, competition_id: str, contender_id: int
    ) -> tuple[str | None, int]:
        """(digest, byte_size) of this contender's archived submission tarball.

        Read from the append-only event log rather than a column because that is
        where the write side records it (there is no column — see the module
        docstring). The LAST event wins: archival is idempotent on re-entry, so a
        repeat would carry the same digest, and taking the latest is the same rule
        the orchestrator's own `archived_submissions` uses.
        """
        rows = self._conn.execute(
            """SELECT payload_json FROM events
                WHERE competition_id = ? AND event_type = ?
             ORDER BY event_id""",
            (competition_id, SUBMISSION_ARCHIVED_EVENT),
        ).fetchall()
        found: tuple[str | None, int] = (None, 0)
        for row in rows:
            if not row["payload_json"]:
                continue
            try:
                payload = json.loads(row["payload_json"])
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("contender_id") != contender_id:
                continue
            digest = payload.get("digest")
            if not isinstance(digest, str) or not digest.strip():
                continue
            size = payload.get("byte_size")
            found = (digest.strip(), int(size) if isinstance(size, int) else 0)
        return found

    def _holdout_challenge_ids(self, competition_id: str) -> frozenset[str]:
        return frozenset(
            row["challenge_id"]
            for row in self._conn.execute(
                "SELECT DISTINCT challenge_id FROM evaluation_items WHERE competition_id = ?",
                (competition_id,),
            )
        )

    def _winner_items(
        self, competition_id: str, contender_id: int
    ) -> tuple[ItemLinkage, ...]:
        """EVERY evaluation item, LEFT-joined to the winner's row for it.

        Round-4 an internal review: this used to be an inner join, so an evaluation item
        the winner had NO `performance_history` row for was silently omitted —
        a two-holdout competition with one row verified only that one item and
        promoted. The join now starts from `evaluation_items` (the holdout's
        full item set) and a missing winner row surfaces as `scored=False`,
        which `audit_linkage_gaps()` turns into a refusal naming the item. The
        contender filter lives in the ON clause on purpose: in a WHERE clause it
        would turn the LEFT join back into an inner one.
        """
        rows = self._conn.execute(
            """SELECT ei.item_id            AS item_id,
                      ei.item_index         AS item_index,
                      ei.challenge_id       AS challenge_id,
                      ei.scoring_item_id    AS scoring_item_id,
                      ei.input_sha256       AS input_sha256,
                      ei.reference_sha256   AS reference_sha256,
                      ei.threshold_commitment AS threshold_commitment,
                      ei.upscale_factor     AS upscale_factor,
                      ei.item_commitment    AS item_commitment,
                      ph.contender_id       AS scored_contender_id,
                      ph.score_packet_digest AS score_packet_digest,
                      ph.audit_bundle_digest AS audit_bundle_digest,
                      ph.item_score         AS item_score,
                      ph.valid              AS valid
                 FROM evaluation_items ei
                 LEFT JOIN performance_history ph
                   ON ph.competition_id = ei.competition_id
                  AND ph.item_id = ei.item_id
                  AND ph.contender_id = ?
                WHERE ei.competition_id = ?
             ORDER BY ei.item_index""",
            (contender_id, competition_id),
        ).fetchall()
        items: list[ItemLinkage] = []
        for r in rows:
            scored = r["scored_contender_id"] is not None
            items.append(
                ItemLinkage(
                    item_id=int(r["item_id"]),
                    item_index=int(r["item_index"]),
                    challenge_id=r["challenge_id"],
                    scoring_item_id=r["scoring_item_id"],
                    input_sha256=str(r["input_sha256"]),
                    reference_sha256=(
                        None
                        if r["reference_sha256"] is None
                        else str(r["reference_sha256"])
                    ),
                    threshold_commitment=str(r["threshold_commitment"]),
                    upscale_factor=(
                        None
                        if r["upscale_factor"] is None
                        else int(r["upscale_factor"])
                    ),
                    item_commitment=(
                        None
                        if r["item_commitment"] is None
                        else str(r["item_commitment"])
                    ),
                    score_packet_digest=r["score_packet_digest"] if scored else None,
                    audit_bundle_digest=r["audit_bundle_digest"] if scored else None,
                    item_score=float(r["item_score"]) if scored else None,
                    valid=bool(r["valid"]) if scored else False,
                    scored=scored,
                )
            )
        return tuple(items)

    def _submission_backup_ref(self, competition_id: str) -> str | None:
        row = self._conn.execute(
            """SELECT payload_json FROM events
                WHERE competition_id = ? AND guard = ?
             ORDER BY event_id DESC LIMIT 1""",
            (competition_id, SUBMISSION_BACKUP_GUARD),
        ).fetchone()
        if row is None or not row["payload_json"]:
            return None
        try:
            payload = json.loads(row["payload_json"])
        except ValueError:
            return None
        ref = payload.get("backup_ref") if isinstance(payload, dict) else None
        return ref if isinstance(ref, str) and ref.strip() else None
