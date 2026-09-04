"""Shared helpers for registry tests.

Two things live here:

`archive_win` builds a genuinely self-consistent evidence chain in the AUDIT
STORE: a real ItemScore packet, a real AuditBundle referencing it, a champion
executable blob, all content-addressed. On its own that chain proves nothing
chain. So it is paired with `seed_competition`, which writes the matching rows
into a REAL competition database (the shipped
`vidaio/competition/migrations/0001_schema.sql`, applied verbatim).

The pipeline reads that database through `SqliteCompetitionSource`. A test that
wants a substituted win simply skips `seed_competition`, or seeds a competition
that says something DIFFERENT from what the store's chain claims; the promotion
must then refuse with the error that link owns.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from vidaio.audit.bundle import CompetitionItemBinding, LifecycleStage, build_bundle
from vidaio.audit.canonical import canonical_json_bytes
from vidaio.audit.store import ArtifactKind, ArtifactRef, LocalFsStore
from vidaio.core.db import apply_migrations
from vidaio.registry.competition_source import (
    COMPETITION_MIGRATIONS_DIR,
    SUBMISSION_ARCHIVED_EVENT,
)
from vidaio.registry.registry import ChampionCandidate
from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.result import ItemScore, config_digest

NOW = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
TS = NOW.isoformat()

COMPETITION_ID = "comp-042"
WINNER = "5HoldoutWinner"
CHALLENGE_ID = "chal-holdout-1"
ITEM_ID = "item-1"
TRACK = "compression"
CHAMPION_BLOB = b"champion-executable-bytes v1"
IMAGE_DIGEST = "1a" * 32
SCORER = "scorer-v1+0123456789ab"
THRESHOLD_COMMITMENT = "ab" * 32
BACKUP_REF = "audit://submissions/sha256:" + "9f" * 32
#: The miner output every honest bundle archives — and, since promotion binds
#: `packet.content_digest` to `bundle.miner_output.digest`,
#: the content digest every honest packet must carry.
MINER_OUTPUT_BYTES = b"miner output bytes"
CONTENT_DIGEST = hashlib.sha256(MINER_OUTPUT_BYTES).hexdigest()


@dataclass(frozen=True)
class ArchivedWin:
    """The audit-store side of a win: everything a promotion is handed."""

    artifact_ref: ArtifactRef
    bundle_ref: ArtifactRef
    packet_ref: ArtifactRef
    packet: ItemScore
    bundle_challenge_id: str
    bundle_item_id: str
    challenge_input_ref: ArtifactRef
    manifest_ref: ArtifactRef
    reference_original_ref: ArtifactRef | None = None
    competition_item: CompetitionItemBinding | None = None
    threshold_commitment: str = THRESHOLD_COMMITMENT


@dataclass(frozen=True)
class SeededItem:
    """One (evaluation_items + performance_history) row pair for the winner.

    A winner's holdout score is the AGGREGATE over all of these, which is why
    promotion has to verify every one of them. Each
    field is independently bendable so a test can corrupt exactly ONE item and
    assert the item-naming error it earns.
    """

    challenge_id: str
    scoring_item_id: str
    score_packet_digest: str | None
    audit_bundle_digest: str | None
    item_score: float
    input_sha256: str | None = None
    reference_sha256: str | None = None
    threshold_commitment: str = THRESHOLD_COMMITMENT
    upscale_factor: int | None = None
    item_commitment: str | None = None


def item_of(win: ArchivedWin, *, item_score: float | None = None) -> SeededItem:
    """The `SeededItem` that records exactly what `win` archived."""
    return SeededItem(
        challenge_id=win.bundle_challenge_id,
        scoring_item_id=win.bundle_item_id,
        score_packet_digest=win.packet_ref.digest,
        audit_bundle_digest=win.bundle_ref.digest,
        item_score=win.packet.score if item_score is None else item_score,
        input_sha256=win.challenge_input_ref.digest,
        reference_sha256=(
            win.reference_original_ref.digest
            if win.reference_original_ref is not None
            else win.challenge_input_ref.digest
        ),
        threshold_commitment=win.threshold_commitment,
        upscale_factor=(
            None
            if win.competition_item is None
            else win.competition_item.upscale_factor
        ),
        item_commitment=(
            None
            if win.competition_item is None
            else win.competition_item.item_commitment
        ),
    )


def score_packet(
    *,
    score: float,
    hotkey: str | None = WINNER,
    challenge_id: str = CHALLENGE_ID,
    item_id: str = ITEM_ID,
    track: str = TRACK,
    scorer_version: str | None = SCORER,
    content_digest: str | None = CONTENT_DIGEST,
) -> ItemScore:
    """A packet honestly bound to the fixture bundle's archived output by
    default; pass `content_digest` to bend that binding (None = a packet that
    names no output at all)."""
    return ItemScore(
        item_id=item_id,
        challenge_id=challenge_id,
        track=track,
        miner_hotkey=hotkey,
        content_digest=content_digest,
        score=score,
        gate_passed=True,
        scorer_version=scorer_version,
        scoring_config_digest=config_digest(ScoringConfig()),
    )


def archive_win(
    store: LocalFsStore,
    *,
    score: float = 0.7,
    artifact_bytes: bytes = CHAMPION_BLOB,
    packet: ItemScore | None = None,
    bundle_hotkey: str | None = WINNER,
    bundle_challenge_id: str = CHALLENGE_ID,
    bundle_item_id: str = ITEM_ID,
    bundle_scorer_version: str = SCORER,
    track: str = TRACK,
    challenge_input_bytes: bytes | None = None,
    reference_original_bytes: bytes | None = None,
    release_upscaling_reference: bool = True,
    compression_reference_original_bytes: bytes | None = None,
    threshold_commitment: str = THRESHOLD_COMMITMENT,
    execution_image_digest: str | None = IMAGE_DIGEST,
) -> ArchivedWin:
    """Archive a winner's packet, bundle and executable, mutually consistent.

    Every keyword lets ONE link be bent independently so a test can assert the
    typed error that link is supposed to raise.
    """
    packet = packet if packet is not None else score_packet(score=score, track=track)
    packet_ref = store.put(packet.to_json().encode("utf-8"), ArtifactKind.SCORE_PACKET)
    if challenge_input_bytes is None:
        # Different holdout items are different media by default. This also
        # respects migration 0002's cross-competition single-use invariant when
        # a test promotes a later competition with another item.
        challenge_input_bytes = (
            f"sealed challenge input:{bundle_challenge_id}:{bundle_item_id}"
        ).encode("utf-8")
    input_ref = store.put(challenge_input_bytes, ArtifactKind.CHALLENGE_INPUT)
    output_ref = store.put(MINER_OUTPUT_BYTES, ArtifactKind.MINER_OUTPUT)
    manifest_ref = store.put(b'{"manifest": true}', ArtifactKind.MANIFEST)
    stage = LifecycleStage.PRE_REVEAL
    reference_ref: ArtifactRef | None = None
    dag_ref: ArtifactRef | None = None
    competition_item: CompetitionItemBinding | None = None
    if track == "upscaling":
        if reference_original_bytes is None:
            reference_original_bytes = (
                f"pristine reference:{bundle_challenge_id}:{bundle_item_id}"
            ).encode("utf-8")
        reference_ref = store.put(
            reference_original_bytes, ArtifactKind.REFERENCE_ORIGINAL
        )
        if release_upscaling_reference:
            # A promotable competition is COMPLETED; production completion has
            # already published this plaintext ref for keyless CPU auditors.
            store.release(reference_ref)
        competition_item = CompetitionItemBinding(
            item_index=0,
            input_sha256=input_ref.digest,
            reference_sha256=reference_ref.digest,
            upscale_factor=2,
            item_commitment="ce" * 32,
        )
        stage = LifecycleStage.COMPETITION_SEALED
    elif compression_reference_original_bytes is not None:
        reference_ref = store.put(
            compression_reference_original_bytes,
            ArtifactKind.REFERENCE_ORIGINAL,
        )
        dag_ref = store.put(b'{"competition-fixture": true}', ArtifactKind.DAG_REVEAL)
        stage = LifecycleStage.POST_RETIREMENT
    bundle = build_bundle(
        challenge_id=bundle_challenge_id,
        item_id=bundle_item_id,
        miner_hotkey=bundle_hotkey,
        commitment_hash=threshold_commitment,
        stage=stage,
        challenge_input=input_ref,
        miner_output=output_ref,
        manifest=manifest_ref,
        score_packet=packet_ref,
        reference_original=reference_ref,
        dag_reveal=dag_ref,
        competition_item=competition_item,
        execution_image_digest=execution_image_digest,
        scorer_version=bundle_scorer_version,
        created_at=TS,
    )
    bundle_ref = store.put(
        canonical_json_bytes(bundle.model_dump(mode="json")), ArtifactKind.AUDIT_BUNDLE
    )
    artifact_ref = store.put(artifact_bytes, ArtifactKind.MINER_OUTPUT)
    return ArchivedWin(
        artifact_ref=artifact_ref,
        bundle_ref=bundle_ref,
        packet_ref=packet_ref,
        packet=packet,
        bundle_challenge_id=bundle_challenge_id,
        bundle_item_id=bundle_item_id,
        challenge_input_ref=input_ref,
        manifest_ref=manifest_ref,
        reference_original_ref=reference_ref,
        competition_item=competition_item,
        threshold_commitment=threshold_commitment,
    )


# ---- the authoritative side: a real competition database -----------------------


def competition_db(path: Path | str = ":memory:") -> sqlite3.Connection:
    """A migrated, EMPTY competition database (the shipped schema, verbatim)."""
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    apply_migrations(conn, COMPETITION_MIGRATIONS_DIR)
    return conn


def seed_competition(
    conn: sqlite3.Connection,
    win: ArchivedWin | None = None,
    *,
    competition_id: str = COMPETITION_ID,
    track: str = TRACK,
    status: str = "COMPLETED",
    winner_hotkey: str = WINNER,
    final_score: float = 0.7,
    item_score: float | None = None,
    challenge_id: str = CHALLENGE_ID,
    scoring_item_id: str = ITEM_ID,
    input_sha256: str | None = None,
    reference_sha256: str | None = None,
    threshold_commitment: str = THRESHOLD_COMMITMENT,
    upscale_factor: int | None = None,
    item_commitment: str | None = None,
    manifest_digest: str | None = None,
    score_packet_digest: str | None = None,
    audit_bundle_digest: str | None = None,
    contender_status: str = "BUILT",
    image_digest: str | None = IMAGE_DIGEST,
    backup_ref: str | None = BACKUP_REF,
    archived_digest: str | None = None,
    archived_bytes: int | None = None,
    archive_winner: bool = True,
    extra_items: int = 0,
    link_extra_items: bool = True,
    more_items: Sequence[SeededItem] = (),
) -> int:
    """Write a competition whose rows say what `win` claims. Returns contender_id.

    `win` supplies the recorded digests unless they are overridden explicitly —
    that override is how a test makes the DATABASE and the STORE disagree.

    ARCHIVAL: the write side appends one `contender_submission_archived` event per
    archived contender, and that event is what promotion DERIVES the champion
    artifact from. It is written here for the winner by default (digest/size taken
    from `win`); `archived_digest` bends it, `archive_winner=False` removes it.

    ITEMS: item 0 is described by the top-level arguments. `extra_items=N` adds N
    filler items whose evidence is a ghost (`link_extra_items=False` makes them
    unlinked instead), and `more_items` adds fully-specified `SeededItem`s — the
    way to build a MULTI-ITEM winner whose every item is archived for real, or
    one where exactly one item is bent (round-3 finding #6).
    """
    if win is not None:
        score_packet_digest = score_packet_digest or win.packet_ref.digest
        audit_bundle_digest = (
            audit_bundle_digest
            if audit_bundle_digest is not None
            else win.bundle_ref.digest
        )
        item_score = item_score if item_score is not None else win.packet.score
        archived_digest = (
            archived_digest if archived_digest is not None else win.artifact_ref.digest
        )
        archived_bytes = (
            archived_bytes if archived_bytes is not None else win.artifact_ref.byte_size
        )
        input_sha256 = (
            input_sha256 if input_sha256 is not None else win.challenge_input_ref.digest
        )
        reference_sha256 = (
            reference_sha256
            if reference_sha256 is not None
            else (
                win.reference_original_ref.digest
                if win.reference_original_ref is not None
                else win.challenge_input_ref.digest
            )
        )
        if win.competition_item is not None:
            upscale_factor = (
                upscale_factor
                if upscale_factor is not None
                else win.competition_item.upscale_factor
            )
            item_commitment = (
                item_commitment
                if item_commitment is not None
                else win.competition_item.item_commitment
            )
        manifest_digest = (
            manifest_digest if manifest_digest is not None else win.manifest_ref.digest
        )
    assert score_packet_digest is not None, "seed_competition needs a packet digest"
    item_score = 0.0 if item_score is None else item_score
    input_sha256 = (
        input_sha256
        or hashlib.sha256(f"{competition_id}:item:0:input".encode("utf-8")).hexdigest()
    )
    reference_sha256 = reference_sha256 or input_sha256
    manifest_digest = manifest_digest or ("cd" * 32)

    conn.execute(
        """INSERT INTO competitions
           (competition_id, track, status, manifest_json, manifest_digest,
            start_time, enrollment_deadline, finalization_time, end_time,
            created_at, updated_at)
           VALUES (?, ?, ?, '{}', ?, ?, ?, ?, ?, ?, ?)""",
        (competition_id, track, status, manifest_digest, TS, TS, TS, TS, TS, TS),
    )
    cur = conn.execute(
        """INSERT INTO contenders
           (competition_id, hotkey, is_calibration, repo_url, commit_sha, tree_sha,
            image_digest, status, final_score, final_rank, created_at, updated_at)
           VALUES (?, ?, 0, 'https://example.invalid/repo', 'abc123', 'def456',
                   ?, ?, ?, 1, ?, ?)""",
        (
            competition_id,
            winner_hotkey,
            image_digest,
            contender_status,
            final_score,
            TS,
            TS,
        ),
    )
    contender_id = int(cur.lastrowid)
    # A calibration baseline that must never be selectable as the winner.
    conn.execute(
        """INSERT INTO contenders
           (competition_id, hotkey, is_calibration, repo_url, commit_sha, tree_sha,
            status, final_score, created_at, updated_at)
           VALUES (?, NULL, 1, 'https://example.invalid/calibration', 'b0', 'b1',
                   'BUILT', 0.99, ?, ?)""",
        (competition_id, TS, TS),
    )

    seeded: list[SeededItem] = [
        SeededItem(
            challenge_id=challenge_id,
            scoring_item_id=scoring_item_id,
            score_packet_digest=score_packet_digest,
            audit_bundle_digest=audit_bundle_digest,
            item_score=item_score,
            input_sha256=input_sha256,
            reference_sha256=reference_sha256,
            threshold_commitment=threshold_commitment,
            upscale_factor=upscale_factor,
            item_commitment=item_commitment,
        )
    ]
    for index in range(1, 1 + extra_items):
        seeded.append(
            SeededItem(
                challenge_id=f"{challenge_id}-{index}",
                scoring_item_id=f"{scoring_item_id}-{index}",
                score_packet_digest="5e" * 32,
                audit_bundle_digest=("7c" * 32) if link_extra_items else None,
                item_score=0.5,
                input_sha256=hashlib.sha256(
                    f"{competition_id}:item:{index}:input".encode("utf-8")
                ).hexdigest(),
                reference_sha256=hashlib.sha256(
                    f"{competition_id}:item:{index}:input".encode("utf-8")
                ).hexdigest(),
                threshold_commitment=threshold_commitment,
            )
        )
    seeded.extend(more_items)

    for index, item in enumerate(seeded):
        item_cur = conn.execute(
            """INSERT INTO evaluation_items
               (competition_id, item_index, input_sha256, input_bytes, length_seconds,
                threshold_commitment, challenge_id, scoring_item_id, created_at,
                reference_sha256, reference_bytes, upscale_factor, item_commitment)
               VALUES (?, ?, ?, 1024, 4.0, ?, ?, ?, ?, ?, 1024, ?, ?)""",
            (
                competition_id,
                index,
                item.input_sha256
                or hashlib.sha256(
                    f"{competition_id}:item:{index}:input".encode("utf-8")
                ).hexdigest(),
                item.threshold_commitment,
                item.challenge_id,
                item.scoring_item_id,
                TS,
                item.reference_sha256
                or item.input_sha256
                or hashlib.sha256(
                    f"{competition_id}:item:{index}:input".encode("utf-8")
                ).hexdigest(),
                item.upscale_factor,
                item.item_commitment,
            ),
        )
        item_id = int(item_cur.lastrowid)
        conn.execute(
            """INSERT INTO performance_history
               (competition_id, contender_id, item_id, item_score, valid,
                score_packet_digest, audit_bundle_digest, created_at)
               VALUES (?, ?, ?, ?, 1, ?, ?, ?)""",
            (
                competition_id,
                contender_id,
                item_id,
                item.item_score,
                item.score_packet_digest,
                item.audit_bundle_digest,
                TS,
            ),
        )

    if archive_winner and archived_digest:
        # The per-contender archival record the orchestrator appends
        # (persistence.record_submission_archived) — promotion reads the winner's
        # artifact address out of exactly this payload.
        conn.execute(
            """INSERT INTO events
               (competition_id, event_type, payload_json, created_at)
               VALUES (?, ?, ?, ?)""",
            (
                competition_id,
                SUBMISSION_ARCHIVED_EVENT,
                json.dumps(
                    {
                        "contender_id": contender_id,
                        "digest": archived_digest,
                        "byte_size": (
                            archived_bytes
                            if archived_bytes is not None
                            else len(CHAMPION_BLOB)
                        ),
                    }
                ),
                TS,
            ),
        )
    if backup_ref is not None:
        conn.execute(
            """INSERT INTO events
               (competition_id, event_type, from_phase, to_phase, guard,
                payload_json, created_at)
               VALUES (?, 'transition', 'FINALIZING_SUBMISSIONS', 'VALIDATING',
                       'submission_backup_completed', ?, ?)""",
            (competition_id, '{"backup_ref": "%s"}' % backup_ref, TS),
        )
    return contender_id


def candidate(
    *,
    track: str = "compression",
    score: float = 0.5,
    artifact_digest: str = "ab" * 32,
    hotkey: str = "5Winner",
    competition_id: str = "comp-001",
    bundle_digest: str = "cd" * 32,
) -> ChampionCandidate:
    return ChampionCandidate(
        track=track,
        artifact_digest=artifact_digest,
        artifact_kind=ArtifactKind.MINER_OUTPUT,
        artifact_bytes=64,
        source_competition_id=competition_id,
        contender_hotkey=hotkey,
        holdout_score=score,
        audit_bundle_digest=bundle_digest,
    )
