"""Orchestrator-owned persistence over the review-shipped competition schema.

The repository module (vidaio.competition.repository) owns contenders, items,
scores and events; this module adds the orchestrator's OWN bookkeeping on the
same schema — batch rows, batch-output records, requeue counting and the
halt/clear ledger — WITHOUT modifying the shipped module (compose, don't edit).

Everything here is derived from (and re-derivable from) the database. Modal's
executable Image handles remain process-local, while append-only owned-Image ids
allow a replacement process to rehydrate exactly what this competition created.
The runtime binding and evaluation-reset events force reprobe plus a whole-matrix
rerun (spec §14 failure recovery); Sandboxes/instances are never restored.

Batch membership is deterministic, not stored: evaluation items ordered by
item_index are partitioned into consecutive slices of the manifest's
evaluation_batch_size.max — batch_index k covers items[k*size:(k+1)*size]. The
same (items, size) always derives the same membership, so a restarted
orchestrator re-derives exactly what the crashed one ran.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, Sequence

from vidaio.competition import repository as repo
from vidaio.competition.interfaces import (
    BUILD_IDENTITY_SCHEME,
    BatchItem,
    BatchOutput,
    logical_build_identity,
)

#: Event types this module appends to the shipped append-only event log.
EVENT_BATCH_OUTPUTS = "batch_outputs"
EVENT_BATCH_REQUEUED = "batch_requeued"
EVENT_BATCH_CONTENDER_FAILED = "batch_contender_failed"
EVENT_ITEM_ZEROED = "item_zero_scored"
EVENT_COMMITMENT_ANCHORED = "commitment_anchored_onchain"
EVENT_HALTED = "orchestrator_halted"
EVENT_HALT_CLEARED = "orchestrator_halt_cleared"
EVENT_COMPETITION_REFERENCES_RELEASED = "competition_references_released"
#: One contender's submission tarball archived.
EVENT_SUBMISSION_ARCHIVED = "contender_submission_archived"
#: Anchor claim ledger. See "anchor claims" below.
EVENT_ANCHOR_CLAIMED = "commitment_anchor_claimed"
EVENT_ANCHOR_FAILED = "commitment_anchor_failed"
EVENT_ANCHOR_RELEASED = "commitment_anchor_released"
#: Fresh-only Modal runtime provenance.  Image digests in ``contenders`` are
#: evidence identifiers, not executable SDK handles; these events fence a live
#: in-process handle set and make every post-restart restore/reset auditable.
EVENT_MODAL_RUNTIME_BOUND = "modal_runtime_bound"
EVENT_MODAL_EVALUATION_RESET = "modal_evaluation_reset"
#: Exact immutable provider Image ids created by this competition.  These are
#: append-only ownership bindings, never provider-discovery results; restart may
#: rehydrate only a matching id/spec/digest recorded here.
EVENT_MODAL_IMAGE_BOUND = "modal_image_bound"


@contextmanager
def txn(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


# ---- evaluation items ----------------------------------------------------------


def list_items(conn: sqlite3.Connection, competition_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM evaluation_items WHERE competition_id = ? ORDER BY item_index",
        (competition_id,),
    ).fetchall()


def batch_items_for(
    items: Sequence[sqlite3.Row], batch_index: int, batch_size: int
) -> list[BatchItem]:
    """Deterministic batch membership (see module docstring)."""
    window = items[batch_index * batch_size : (batch_index + 1) * batch_size]
    return [
        BatchItem(
            item_id=row["item_id"],
            item_index=row["item_index"],
            input_sha256=row["input_sha256"],
            input_bytes=row["input_bytes"],
            length_seconds=row["length_seconds"],
            upscale_factor=row["upscale_factor"],
            target_width=row["target_width"],
            target_height=row["target_height"],
        )
        for row in window
    ]


def batch_count(n_items: int, batch_size: int) -> int:
    return (n_items + batch_size - 1) // batch_size if n_items else 0


# ---- batches -------------------------------------------------------------------


def ensure_batches(
    conn: sqlite3.Connection,
    competition_id: str,
    contender_id: int,
    *,
    n_items: int,
    batch_size: int,
    now: datetime,
) -> int:
    """Idempotently create the contender's batch rows (PENDING). Returns how many
    NEW rows were inserted; existing (contender_id, batch_index) rows are kept."""
    created = 0
    ts = repo.iso(now)
    for batch_index in range(batch_count(n_items, batch_size)):
        cur = conn.execute(
            """INSERT INTO batches (competition_id, contender_id, batch_index, status, created_at)
               VALUES (?, ?, ?, 'PENDING', ?)
               ON CONFLICT (contender_id, batch_index) DO NOTHING""",
            (competition_id, contender_id, batch_index, ts),
        )
        created += cur.rowcount if cur.rowcount > 0 else 0
    return created


def runnable_batches(
    conn: sqlite3.Connection, competition_id: str
) -> list[sqlite3.Row]:
    """Batches the orchestrator should (re)run: PENDING, REQUEUED — and RUNNING,
    which after a crash is a stale claim (status COMPLETED is written atomically
    with the batch_outputs event, so a RUNNING row can never have outputs)."""
    return conn.execute(
        """SELECT * FROM batches WHERE competition_id = ?
           AND status IN ('PENDING', 'REQUEUED', 'RUNNING')
           ORDER BY contender_id, batch_index""",
        (competition_id,),
    ).fetchall()


def _effective_batch_event_floor(conn: sqlite3.Connection, competition_id: str) -> int:
    """Last Modal evaluation-reset event id, or zero.

    Resetting never deletes prior batch evidence.  Instead, readers select only
    effective output/requeue events after this append-only fence, while the old
    run remains inspectable in the event log.
    """
    row = conn.execute(
        "SELECT COALESCE(MAX(event_id), 0) AS event_id FROM events"
        " WHERE competition_id = ? AND event_type = ?",
        (competition_id, EVENT_MODAL_EVALUATION_RESET),
    ).fetchone()
    return int(row["event_id"])


def set_batch_status(
    conn: sqlite3.Connection,
    batch_id: int,
    status: str,
    now: datetime,
    *,
    failure_code: str | None = None,
    started: bool = False,
    finished: bool = False,
) -> None:
    ts = repo.iso(now)
    conn.execute(
        """UPDATE batches SET status = ?, failure_code = ?,
               started_at = CASE WHEN ? THEN ? ELSE started_at END,
               finished_at = CASE WHEN ? THEN ? ELSE finished_at END
           WHERE batch_id = ?""",
        (
            status,
            failure_code,
            1 if started else 0,
            ts,
            1 if finished else 0,
            ts,
            batch_id,
        ),
    )


def complete_batch(
    conn: sqlite3.Connection,
    competition_id: str,
    batch_id: int,
    contender_id: int,
    outputs: Sequence[BatchOutput],
    now: datetime,
) -> None:
    """Atomically record the batch's outputs (append-only event) and mark it
    COMPLETED — a crash between the two can never happen (single transaction),
    so COMPLETED always implies the outputs are recorded."""
    with txn(conn):
        repo.record_event(
            conn,
            competition_id,
            EVENT_BATCH_OUTPUTS,
            now,
            payload={
                "batch_id": batch_id,
                "contender_id": contender_id,
                "outputs": [
                    [o.item_id, o.output_sha256, o.output_bytes] for o in outputs
                ],
            },
        )
        set_batch_status(conn, batch_id, "COMPLETED", now, finished=True)


def requeue_batch(
    conn: sqlite3.Connection,
    competition_id: str,
    batch_id: int,
    reason: str,
    now: datetime,
) -> None:
    with txn(conn):
        repo.record_event(
            conn,
            competition_id,
            EVENT_BATCH_REQUEUED,
            now,
            payload={"batch_id": batch_id, "reason": reason[:500]},
        )
        set_batch_status(conn, batch_id, "REQUEUED", now, failure_code=reason[:200])


def fail_batch_contender_fault(
    conn: sqlite3.Connection,
    competition_id: str,
    batch_id: int,
    contender_id: int,
    code: str,
    reason: str,
    now: datetime,
) -> None:
    """Terminally FAIL one batch because the CONTENDER's own submission failed.

    review service-review #14: a solution's `exit 1`, timeout, unsafe/oversize
    output is that contender's problem, not a systemic infra blocker. The batch
    becomes terminal (so evaluation can still complete), the reason code is in the
    append-only event log, and the competition keeps running. Items covered by
    this batch simply have no outputs — they are zero-scored downstream with a
    reason code, never substituted and never sent to ffmpeg.
    """
    with txn(conn):
        repo.record_event(
            conn,
            competition_id,
            EVENT_BATCH_CONTENDER_FAILED,
            now,
            payload={
                "batch_id": batch_id,
                "contender_id": contender_id,
                "code": code,
                "reason": reason[:500],
            },
        )
        set_batch_status(
            conn,
            batch_id,
            "FAILED",
            now,
            failure_code=f"{code}: {reason}"[:200],
            finished=True,
        )


def contender_fault_events(
    conn: sqlite3.Connection, competition_id: str
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM events WHERE competition_id = ? AND event_type = ? ORDER BY event_id",
        (competition_id, EVENT_BATCH_CONTENDER_FAILED),
    ).fetchall()


def requeue_count(conn: sqlite3.Connection, competition_id: str, batch_id: int) -> int:
    """How many times this batch has been requeued (derived from the event log —
    survives ordinary restarts; a full Modal runtime reset starts a new effective
    attempt window because every batch is rerun)."""
    floor = _effective_batch_event_floor(conn, competition_id)
    rows = conn.execute(
        "SELECT payload_json FROM events WHERE competition_id = ? AND event_type = ?"
        " AND event_id > ?",
        (competition_id, EVENT_BATCH_REQUEUED, floor),
    ).fetchall()
    count = 0
    for row in rows:
        payload = json.loads(row["payload_json"] or "{}")
        if payload.get("batch_id") == batch_id:
            count += 1
    return count


def outputs_for_contender(
    conn: sqlite3.Connection, competition_id: str, contender_id: int
) -> dict[int, tuple[str, int]]:
    """item_id -> (output_sha256, output_bytes) from recorded batch_outputs events.
    Events below the latest Modal reset remain evidence but are ineffective;
    within the active window, later events win."""
    floor = _effective_batch_event_floor(conn, competition_id)
    rows = conn.execute(
        "SELECT payload_json FROM events WHERE competition_id = ? AND event_type = ?"
        " AND event_id > ? ORDER BY event_id",
        (competition_id, EVENT_BATCH_OUTPUTS, floor),
    ).fetchall()
    outputs: dict[int, tuple[str, int]] = {}
    for row in rows:
        payload = json.loads(row["payload_json"] or "{}")
        if payload.get("contender_id") != contender_id:
            continue
        for item_id, digest, nbytes in payload.get("outputs", []):
            outputs[int(item_id)] = (str(digest), int(nbytes))
    return outputs


def batch_id_for_item(
    conn: sqlite3.Connection,
    competition_id: str,
    contender_id: int,
    item_index: int,
    batch_size: int,
) -> int | None:
    row = conn.execute(
        "SELECT batch_id FROM batches WHERE competition_id = ? AND contender_id = ?"
        " AND batch_index = ?",
        (competition_id, contender_id, item_index // batch_size),
    ).fetchone()
    return row["batch_id"] if row is not None else None


# ---- score-row lookups ---------------------------------------------------------


def has_score_row(conn: sqlite3.Connection, contender_id: int, item_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM performance_history WHERE contender_id = ? AND item_id = ?",
        (contender_id, item_id),
    ).fetchone()
    return row is not None


def unlinked_performance_rows(
    conn: sqlite3.Connection, competition_id: str
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM performance_history WHERE competition_id = ?"
        " AND audit_bundle_digest IS NULL ORDER BY performance_id",
        (competition_id,),
    ).fetchall()


# ---- fresh Modal runtime restart fence ----------------------------------------


def record_modal_image_binding(
    conn: sqlite3.Connection,
    competition_id: str,
    *,
    contender_id: int | None,
    is_calibration: bool,
    repo_url: str,
    commit_sha: str,
    tree_sha: str,
    image_digest: str,
    image_object_id: str,
    runtime_session_id: str,
    runtime_label: str,
    now: datetime,
) -> None:
    """Bind one exact competition-created immutable Modal Image to its source.

    ``image_digest`` is the stable logical identity of the exact pinned source;
    ``image_object_id`` is the opaque provider handle for this particular build.
    Both land in the typed append-only ledger and the chronological event in the
    caller's transaction.  A restarted process may rehydrate the exact owned
    object without listing or selecting any external resource. GPU Sandboxes are
    deliberately absent: they remain fresh per batch and are never restored.
    """
    expected = logical_build_identity(
        repo_url=repo_url,
        commit_sha=commit_sha,
        tree_sha=tree_sha,
    )
    if image_digest != expected:
        raise ValueError(
            "Modal image binding logical digest does not match its pinned source"
        )
    conn.execute(
        """INSERT INTO modal_image_bindings
           (competition_id, contender_id, is_calibration, repo_url, commit_sha,
            tree_sha, build_identity_scheme, image_digest, provider,
            image_object_id, runtime_session_id, runtime_label, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'modal', ?, ?, ?, ?)""",
        (
            competition_id,
            contender_id,
            1 if is_calibration else 0,
            repo_url,
            commit_sha,
            tree_sha,
            BUILD_IDENTITY_SCHEME,
            image_digest,
            image_object_id,
            runtime_session_id,
            runtime_label,
            repo.iso(now),
        ),
    )
    repo.record_event(
        conn,
        competition_id,
        EVENT_MODAL_IMAGE_BOUND,
        now,
        payload={
            "contender_id": contender_id,
            "is_calibration": is_calibration,
            "repo_url": repo_url,
            "commit_sha": commit_sha,
            "tree_sha": tree_sha,
            "build_identity_scheme": BUILD_IDENTITY_SCHEME,
            "image_digest": image_digest,
            "provider": "modal",
            "image_object_id": image_object_id,
            "runtime_session_id": runtime_session_id,
            "runtime_label": runtime_label,
        },
    )


def latest_modal_image_binding(
    conn: sqlite3.Connection,
    competition_id: str,
    image_digest: str,
    *,
    is_calibration: bool | None = None,
) -> dict[str, object] | None:
    """Return the newest typed ownership binding for one logical identity.

    Calibration and earning contenders may intentionally submit identical pinned
    source, so callers can select the role instead of accepting whichever event
    happened to be newest for the shared logical digest.
    """
    sql = (
        "SELECT * FROM modal_image_bindings"
        " WHERE competition_id = ? AND image_digest = ?"
    )
    params: list[object] = [competition_id, image_digest]
    if is_calibration is not None:
        sql += " AND is_calibration = ?"
        params.append(1 if is_calibration else 0)
    sql += " ORDER BY binding_id DESC LIMIT 1"
    row = conn.execute(sql, tuple(params)).fetchone()
    if row is None:
        return None
    payload = dict(row)
    payload["is_calibration"] = bool(payload["is_calibration"])
    return payload


def latest_modal_calibration_binding(
    conn: sqlite3.Connection,
    competition_id: str,
) -> dict[str, object] | None:
    """Return the newest typed pre-anchor/baseline ownership binding, if any."""
    row = conn.execute(
        "SELECT * FROM modal_image_bindings"
        " WHERE competition_id = ? AND is_calibration = 1"
        " ORDER BY binding_id DESC LIMIT 1",
        (competition_id,),
    ).fetchone()
    if row is None:
        return None
    payload = dict(row)
    payload["is_calibration"] = True
    return payload


def latest_modal_runtime_binding(
    conn: sqlite3.Connection, competition_id: str
) -> dict[str, object] | None:
    """Return the most recent append-only fresh-runtime binding."""
    row = conn.execute(
        "SELECT event_id, payload_json, created_at FROM events"
        " WHERE competition_id = ? AND event_type = ?"
        " ORDER BY event_id DESC LIMIT 1",
        (competition_id, EVENT_MODAL_RUNTIME_BOUND),
    ).fetchone()
    if row is None:
        return None
    payload = json.loads(row["payload_json"] or "{}")
    payload["event_id"] = int(row["event_id"])
    payload["created_at"] = row["created_at"]
    return payload


def record_modal_runtime_binding(
    conn: sqlite3.Connection,
    competition_id: str,
    *,
    runtime_session_id: str,
    runtime_label: str,
    phase: str,
    now: datetime,
    reason: str,
    previous_runtime_session_id: str | None = None,
    rebound_images: Sequence[tuple[int, str]] = (),
) -> None:
    """Append the runtime that owns all currently executable image handles.

    Callers either hold ``txn`` or are at a point where this single event is the
    complete operation (the initial, pre-build binding).
    """
    repo.record_event(
        conn,
        competition_id,
        EVENT_MODAL_RUNTIME_BOUND,
        now,
        payload={
            "runtime_session_id": runtime_session_id,
            "runtime_label": runtime_label,
            "phase": phase,
            "reason": reason,
            "previous_runtime_session_id": previous_runtime_session_id,
            "rebound_images": [
                {"contender_id": contender_id, "image_digest": image_digest}
                for contender_id, image_digest in rebound_images
            ],
        },
    )


def reset_evaluation_for_modal_runtime(
    conn: sqlite3.Connection,
    competition_id: str,
    *,
    previous_runtime_session_id: str | None,
    runtime_session_id: str,
    runtime_label: str,
    phase: str,
    rebound_images: Sequence[tuple[int, str]],
    probe_records: Sequence[tuple[int, str, str]],
    now: datetime,
) -> None:
    """Atomically invalidate the old runtime's batch results and bind the new one.

    No row or event is deleted.  Every batch is returned to PENDING and the reset
    event becomes the read fence used by ``outputs_for_contender`` and
    ``requeue_count``.  Therefore the effective evaluation after a restart is
    produced wholly by one fresh runtime, never a mixture of old and new Modal
    handles/outputs.
    """
    with txn(conn):
        for contender_id, image_digest, probe_json in probe_records:
            record_sandbox_probe(
                conn,
                competition_id,
                contender_id,
                image_digest,
                probe_json,
                passed=True,
                now=now,
            )
        repo.record_event(
            conn,
            competition_id,
            EVENT_MODAL_EVALUATION_RESET,
            now,
            payload={
                "previous_runtime_session_id": previous_runtime_session_id,
                "runtime_session_id": runtime_session_id,
                "runtime_label": runtime_label,
                "phase": phase,
                "rebound_images": [
                    {"contender_id": contender_id, "image_digest": image_digest}
                    for contender_id, image_digest in rebound_images
                ],
                "policy": "discard_prior_effective_batches_and_rerun_full_matrix",
            },
        )
        conn.execute(
            """UPDATE batches
               SET status = 'PENDING', failure_code = NULL,
                   started_at = NULL, finished_at = NULL
               WHERE competition_id = ?""",
            (competition_id,),
        )
        record_modal_runtime_binding(
            conn,
            competition_id,
            runtime_session_id=runtime_session_id,
            runtime_label=runtime_label,
            phase=phase,
            now=now,
            reason="fresh_runtime_full_evaluation_reset",
            previous_runtime_session_id=previous_runtime_session_id,
            rebound_images=rebound_images,
        )


# ---- halt ledger ----------------------------------------------------------------


def is_halted(conn: sqlite3.Connection, competition_id: str) -> bool:
    """True when the most recent halt-ledger event is a halt (derived purely from
    the DB — a restarted orchestrator stays halted until an operator clears it)."""
    row = conn.execute(
        "SELECT event_type FROM events WHERE competition_id = ? AND event_type IN (?, ?)"
        " ORDER BY event_id DESC LIMIT 1",
        (competition_id, EVENT_HALTED, EVENT_HALT_CLEARED),
    ).fetchone()
    return row is not None and row["event_type"] == EVENT_HALTED


def record_halt(
    conn: sqlite3.Connection, competition_id: str, reason: str, now: datetime
) -> bool:
    """Record a pipeline halt (idempotent: a second halt while halted is a no-op).
    The competition's PHASE is untouched — a systemic infra blocker halts the
    pipeline, it never fails the competition (spec §14)."""
    if is_halted(conn, competition_id):
        return False
    repo.record_event(
        conn, competition_id, EVENT_HALTED, now, payload={"reason": reason[:1000]}
    )
    return True


def clear_halt(
    conn: sqlite3.Connection,
    competition_id: str,
    operator: str,
    now: datetime,
    *,
    reason: str,
) -> bool:
    """Operator action: resume after a fixed blocker, with an auditable reason."""
    operator = operator.strip()
    reason = reason.strip()
    if not operator:
        raise ValueError("clear-halt operator must be non-empty")
    if not reason:
        raise ValueError("clear-halt reason must be non-empty")
    if len(operator) > 256:
        raise ValueError("clear-halt operator must be at most 256 characters")
    if len(reason) > 1000:
        raise ValueError("clear-halt reason must be at most 1000 characters")
    if not is_halted(conn, competition_id):
        return False
    repo.record_event(
        conn,
        competition_id,
        EVENT_HALT_CLEARED,
        now,
        payload={"operator": operator, "reason": reason},
    )
    return True


# ---- submission backups ----------------------------------


def record_submission_archived(
    conn: sqlite3.Connection,
    competition_id: str,
    contender_id: int,
    digest: str,
    byte_size: int,
    now: datetime,
) -> None:
    """Evidence that THIS contender's pinned tree is in the audit store.

    The finalization phase may only advance once every contender that can still
    win has one of these — the combined backup_ref used to be recorded even when a
    checkout had been skipped, certifying an archive that did not exist.
    """
    repo.record_event(
        conn,
        competition_id,
        EVENT_SUBMISSION_ARCHIVED,
        now,
        payload={
            "contender_id": contender_id,
            "digest": digest,
            "byte_size": byte_size,
        },
    )


def archived_submissions(
    conn: sqlite3.Connection, competition_id: str
) -> dict[int, str]:
    """contender_id -> archived tarball digest, from the append-only event log.

    Derived, never cached: a restarted orchestrator re-enters finalization and
    archives exactly the contenders still missing (spec §14 idempotent re-entry).
    """
    rows = conn.execute(
        "SELECT payload_json FROM events WHERE competition_id = ? AND event_type = ?"
        " ORDER BY event_id",
        (competition_id, EVENT_SUBMISSION_ARCHIVED),
    ).fetchall()
    archived: dict[int, str] = {}
    for row in rows:
        payload = json.loads(row["payload_json"] or "{}")
        contender_id = payload.get("contender_id")
        digest = payload.get("digest")
        if contender_id is not None and digest:
            archived[int(contender_id)] = str(digest)
    return archived


# ---- anchor claims ---------------------------------------
#
# The chain write used to happen BEFORE the guarded DB transition, so two
# concurrent anchor requests carrying different payloads could BOTH reach the
# chain while only the first to return was ever recorded — a second valid,
# untracked commitment for the same competition.
#
# The right to anchor is now CLAIMED IN THE DB FIRST, inside a BEGIN IMMEDIATE
# transaction that also verifies the competition is still SCHEDULED and unanchored.
# A claim carries the EXACT payload digest that is about to be written, so a crash
# mid-anchor leaves a re-checkable record rather than an unknown.
#
# Ledger semantics (all in the append-only event log, no schema change):
#   EVENT_ANCHOR_CLAIMED   the right is taken; the payload digest is on record
#   EVENT_COMMITMENT_ANCHORED   exact finalized/archive receipt persisted in the
#                               same transaction as the lifecycle root -> RESOLVES
#   EVENT_ANCHOR_RELEASED  an operator resolved an ambiguous claim -> RESOLVES it
#   EVENT_ANCHOR_FAILED    the write failed; AMBIGUOUS (a timeout may still have
#                          landed), so it annotates and does NOT resolve.

_ANCHOR_RESOLVING = (EVENT_COMMITMENT_ANCHORED, EVENT_ANCHOR_RELEASED)


def latest_verified_anchor_receipt(
    conn: sqlite3.Connection, competition_id: str
) -> dict[str, object] | None:
    """Latest independently finalized/archive-verified anchor evidence, if any."""

    row = conn.execute(
        "SELECT payload_json, created_at FROM events"
        " WHERE competition_id = ? AND event_type = ?"
        " ORDER BY event_id DESC LIMIT 1",
        (competition_id, EVENT_COMMITMENT_ANCHORED),
    ).fetchone()
    if row is None:
        return None
    payload = json.loads(row["payload_json"] or "{}")
    if payload.get("archive_verified") is not True:
        return None
    payload.setdefault("verified_at", row["created_at"])
    return payload


def open_anchor_claim(
    conn: sqlite3.Connection, competition_id: str
) -> dict[str, object] | None:
    """The unresolved anchor claim for this competition, or None.

    A claim is open while no resolving event follows it in the event log.
    """
    row = conn.execute(
        "SELECT event_type, payload_json, created_at FROM events"
        " WHERE competition_id = ? AND event_type IN (?, ?, ?)"
        " ORDER BY event_id DESC LIMIT 1",
        (competition_id, EVENT_ANCHOR_CLAIMED, *_ANCHOR_RESOLVING),
    ).fetchone()
    if row is None or row["event_type"] != EVENT_ANCHOR_CLAIMED:
        return None
    payload = json.loads(row["payload_json"] or "{}")
    payload.setdefault("claimed_at", row["created_at"])
    return payload


def record_anchor_claim(
    conn: sqlite3.Connection,
    competition_id: str,
    *,
    payload_digest: str,
    root: str,
    now: datetime,
) -> None:
    """Take the exclusive right to write EXACTLY this payload on chain."""
    repo.record_event(
        conn,
        competition_id,
        EVENT_ANCHOR_CLAIMED,
        now,
        payload={
            "payload_digest": payload_digest,
            "root": root,
            "claimed_at": repo.iso(now),
        },
    )


def record_anchor_failure(
    conn: sqlite3.Connection,
    competition_id: str,
    *,
    payload_digest: str,
    reason: str,
    now: datetime,
) -> None:
    """Annotate a failed write/receipt attempt WITHOUT resolving the claim.

    A failed anchor is AMBIGUOUS by nature (a timeout or a dropped connection may
    still have landed the extrinsic), so the claim deliberately stays open. Once
    stale, the identical payload may be checked again in read-only mode; another
    write requires explicit operator release after proving nothing landed.
    """
    repo.record_event(
        conn,
        competition_id,
        EVENT_ANCHOR_FAILED,
        now,
        payload={"payload_digest": payload_digest, "reason": reason[:500]},
    )


def release_anchor_claim(
    conn: sqlite3.Connection,
    competition_id: str,
    *,
    operator: str,
    reason: str,
    now: datetime,
) -> bool:
    """Operator resolution of an ambiguous claim. True when one was open."""
    claim = open_anchor_claim(conn, competition_id)
    if claim is None:
        return False
    repo.record_event(
        conn,
        competition_id,
        EVENT_ANCHOR_RELEASED,
        now,
        payload={
            "operator": operator,
            "reason": reason[:500],
            "released_payload_digest": claim.get("payload_digest"),
        },
    )
    return True


# ---- sandboxes (probe evidence) -------------------------------------------------


def record_sandbox_probe(
    conn: sqlite3.Connection,
    competition_id: str,
    contender_id: int,
    image_digest: str,
    probe_json: str,
    passed: bool,
    now: datetime,
) -> int:
    """Persist the isolation-probe evidence on a sandboxes row (spec §05: the probe
    report is part of the audit trail, not just a boolean)."""
    ts = repo.iso(now)
    cur = conn.execute(
        """INSERT INTO sandboxes
           (competition_id, contender_id, image_digest, status, isolation_probe_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            competition_id,
            contender_id,
            image_digest,
            "CREATED" if passed else "FAILED",
            probe_json,
            ts,
        ),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]
