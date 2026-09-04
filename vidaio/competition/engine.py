"""Competition lifecycle engine — spec: design spec §04.

Time-based ticks drive the outer transitions (SCHEDULED->ENROLLING,
ENROLLING->FINALIZING_SUBMISSIONS, AWAITING_END_TIME->COMPLETED); explicit
pipeline-completion calls drive the inner ones. Every transition:

- is guarded by the table in states.TRANSITIONS (anything else raises IllegalTransition),
- is idempotent (re-applying an already-applied transition is a no-op, not an error),
- appends to the append-only event log,
- emits one structured JSON log line with competition_id / from / to / guard fields.

The single-running-competition invariant is enforced twice: as a pre-check here and,
authoritatively, by a partial unique index in SQL (migrations/0001_schema.sql).

No wall-clock reads: every entry point takes `now` explicitly (timezone-aware).
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Callable, Iterator, Mapping

from vidaio.core.logging import get_logger, log_fields
from vidaio.competition import repository as repo
from vidaio.competition.config import CompetitionConfig
from vidaio.competition.manifest import CompetitionManifest, validate_against_config
from vidaio.competition.review import recalculate_ranks
from vidaio.competition.states import TRANSITIONS, Phase

logger = get_logger("vidaio.competition.engine")

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class IllegalTransition(Exception):
    def __init__(
        self,
        competition_id: str,
        from_phase: Phase | None,
        to_phase: Phase,
        guard: str | None,
        detail: str = "",
    ) -> None:
        self.competition_id = competition_id
        self.from_phase = from_phase
        self.to_phase = to_phase
        self.guard = guard
        self.detail = detail
        frm = from_phase.value if from_phase else "<missing>"
        msg = f"{competition_id}: {frm} -> {to_phase.value} is not allowed"
        if guard:
            msg += f" (guard: {guard})"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


class _CompletionGateOpen(Exception):
    """Internal: the in-transaction completion re-check found the gate still open —
    unlinked/invalid score rows and/or missing baseline calibration rows.

    Caught by tick() — the competition stays in AWAITING_END_TIME (with a structured
    log line) instead of failing the whole tick."""

    def __init__(
        self, missing_calibration_rows: int, gaps: list[tuple[int, int]]
    ) -> None:
        self.missing_calibration_rows = missing_calibration_rows
        self.gaps = gaps
        super().__init__(
            f"{missing_calibration_rows} baseline calibration row(s) missing; "
            f"{len(gaps)} performance row(s) lack an audit_bundle_digest"
        )


def _require_aware(now: datetime) -> None:
    if now.tzinfo is None:
        raise ValueError("engine requires a timezone-aware `now`")


@contextmanager
def _txn(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


class LifecycleEngine:
    def __init__(self, config: CompetitionConfig | None = None) -> None:
        self.config = config or CompetitionConfig()

    # ---- creation -------------------------------------------------------------

    def create_competition(
        self, conn: sqlite3.Connection, manifest: CompetitionManifest, now: datetime
    ) -> None:
        """Create a competition in SCHEDULED. The manifest must clear the configured
        validation bounds; its canonical JSON and digest are persisted verbatim."""
        _require_aware(now)
        validate_against_config(manifest, self.config)
        with _txn(conn):
            repo.insert_competition(conn, manifest, now)
            repo.record_event(
                conn,
                manifest.competition_id,
                "created",
                now,
                to_phase=Phase.SCHEDULED,
                payload={"manifest_digest": manifest.manifest_digest()},
            )
        logger.info(
            "competition created",
            extra=log_fields(
                competition_id=manifest.competition_id,
                phase=Phase.SCHEDULED.value,
                manifest_digest=manifest.manifest_digest(),
            ),
        )

    # ---- core transition ------------------------------------------------------

    def _apply(
        self,
        conn: sqlite3.Connection,
        competition_id: str,
        to_phase: Phase,
        now: datetime,
        *,
        reason: str | None = None,
        payload: dict | None = None,
        in_txn_check: Callable[[], None] | None = None,
    ) -> bool:
        """Apply one guarded transition. Returns True if applied, False if it was an
        idempotent no-op (competition already in to_phase). Raises IllegalTransition
        for any edge not in the transition table.

        in_txn_check, when given, runs INSIDE the BEGIN IMMEDIATE transaction —
        after the write lock is held, before any row changes: a
        guard whose pre-check passed is re-verified once no concurrent writer can
        change its inputs; raising rolls the transition back entirely."""
        _require_aware(now)
        comp = repo.get_competition(conn, competition_id)
        if comp is None:
            raise IllegalTransition(competition_id, None, to_phase, None, "unknown competition")
        if comp.status is to_phase:
            return False  # idempotent re-apply
        guard = TRANSITIONS.get((comp.status, to_phase))
        if (
            comp.status is Phase.SCHEDULED
            and to_phase is Phase.ENROLLING
            and comp.commitment_root is None
        ):
            # Structural backstop for the tick guard: enrollment can never
            # open before the pre-commitment is anchored — even via a direct _apply.
            raise IllegalTransition(
                competition_id,
                comp.status,
                to_phase,
                guard,
                "pre-commitment not anchored (mark_commitment_anchored must precede enrollment)",
            )
        if guard is None:
            # Re-applying a transition the event log already holds (e.g. a pipeline
            # retry of mark_evaluation_complete after the phase moved on) is a no-op,
            # not an error. Anything else is an illegal edge.
            already = conn.execute(
                "SELECT 1 FROM events WHERE competition_id = ? AND event_type = 'transition'"
                " AND to_phase = ? LIMIT 1",
                (competition_id, to_phase.value),
            ).fetchone()
            if already is not None:
                return False
            raise IllegalTransition(
                competition_id,
                comp.status,
                to_phase,
                None,
                "edge not in transition table",
            )
        event_payload = dict(payload or {})
        if reason is not None:
            event_payload["reason"] = reason
        try:
            with _txn(conn):
                if in_txn_check is not None:
                    in_txn_check()
                repo.set_status(conn, competition_id, to_phase, now, failure_reason=reason)
                repo.record_event(
                    conn,
                    competition_id,
                    "transition",
                    now,
                    from_phase=comp.status,
                    to_phase=to_phase,
                    guard=guard,
                    payload=event_payload or None,
                )
        except sqlite3.IntegrityError as exc:
            # The partial unique index rejected a second running competition.
            raise IllegalTransition(
                competition_id,
                comp.status,
                to_phase,
                guard,
                f"single-running-competition invariant violated: {exc}",
            ) from exc
        logger.info(
            "phase transition",
            extra=log_fields(
                competition_id=competition_id,
                phase=to_phase.value,
                from_phase=comp.status.value,
                to_phase=to_phase.value,
                guard=guard,
                **({"reason": reason} if reason else {}),
            ),
        )
        return True

    # ---- time-based transitions (tick) ---------------------------------------

    def tick(self, conn: sqlite3.Connection, now: datetime) -> list[tuple[str, Phase, Phase]]:
        """Drive all due time-based transitions. Safe to call repeatedly (idempotent):
        a tick that finds nothing due applies nothing. Returns (id, from, to) applied."""
        _require_aware(now)
        applied: list[tuple[str, Phase, Phase]] = []

        # AWAITING_END_TIME -> COMPLETED at max(end_time, human_review_deadline)
        # (first: completing frees the running slot for a queued SCHEDULED competition
        # within the same tick). The review window is never truncated by end_time
        #. Completion is additionally gated on full audit linkage
        # (config.require_audit_linkage): every performance row — the baseline
        # calibration rows included — must carry its audit_bundle_digest.
        for comp in repo.list_competitions_in(conn, [Phase.AWAITING_END_TIME]):
            completion_due = comp.end_time
            if comp.human_review_deadline is not None:
                completion_due = max(completion_due, comp.human_review_deadline)
            if now >= completion_due:
                if self._complete(conn, comp, now):
                    applied.append((comp.competition_id, Phase.AWAITING_END_TIME, Phase.COMPLETED))

        # ENROLLING -> FINALIZING_SUBMISSIONS at finalization_time; the archived baseline
        # is injected during FINALIZING as a calibration contender (non-earning).
        for comp in repo.list_competitions_in(conn, [Phase.ENROLLING]):
            if now >= comp.finalization_time:
                if self._apply(conn, comp.competition_id, Phase.FINALIZING_SUBMISSIONS, now):
                    applied.append(
                        (comp.competition_id, Phase.ENROLLING, Phase.FINALIZING_SUBMISSIONS)
                    )
                    self._inject_baseline(conn, comp.competition_id, now)

        # SCHEDULED -> ENROLLING at start_time, gated on no other running competition
        # AND an anchored pre-commitment (an internal review: enrollment can never open before
        # the commitment root — which covers the manifest digest — is anchored).
        for comp in repo.list_competitions_in(conn, [Phase.SCHEDULED]):
            if now < comp.start_time:
                continue
            if comp.commitment_root is None:
                logger.info(
                    "start deferred: pre-commitment not anchored",
                    extra=log_fields(
                        competition_id=comp.competition_id,
                        phase=comp.status.value,
                        guard="start_time_commitment_anchored_and_no_other_running",
                        reason="commitment_root missing",
                    ),
                )
                continue
            running = repo.running_competition_id(conn)
            if running is not None:
                logger.info(
                    "start deferred: another competition is running",
                    extra=log_fields(
                        competition_id=comp.competition_id,
                        phase=comp.status.value,
                        blocking_competition_id=running,
                        guard="start_time_commitment_anchored_and_no_other_running",
                    ),
                )
                continue
            if self._apply(conn, comp.competition_id, Phase.ENROLLING, now):
                applied.append((comp.competition_id, Phase.SCHEDULED, Phase.ENROLLING))
        return applied

    def _complete(
        self, conn: sqlite3.Connection, comp: repo.CompetitionRecord, now: datetime
    ) -> bool:
        """AWAITING_END_TIME -> COMPLETED, gated on full audit linkage AND a
        complete baseline score matrix.

        With require_audit_linkage (the default), completion requires BOTH:
        - every performance_history row of the competition — including the baseline
          calibration rows — carries its audit_bundle_digest
          (audit_linkage_gaps() must be empty), and
        - when the competition has a calibration contender, it holds a performance
          row for EVERY evaluation item (count_missing_calibration_rows == 0) —
          audit_linkage_gaps only sees rows that EXIST, so a baseline with zero rows
          would otherwise bypass the gate entirely.
        The checks run twice: a cheap pre-check that logs a structured reason and
        leaves the competition in AWAITING_END_TIME, and a re-check INSIDE the
        transition's BEGIN IMMEDIATE transaction so a score row
        landing between check and commit can't slip an unlinked competition through.

        require_audit_linkage=False (tests/dev ONLY) bypasses the gate; every
        bypassed completion emits a warning log line.
        """
        competition_id = comp.competition_id
        try:
            repo.validate_evaluation_item_bindings(conn, competition_id)
        except repo.EvaluationItemBindingError as exc:
            self._log_item_binding_blocker(comp, str(exc))
            return False
        if not self.config.require_audit_linkage:
            logger.warning(
                "audit linkage completion gate BYPASSED (require_audit_linkage=False; "
                "tests/dev only — production must gate completion on full linkage)",
                extra=log_fields(
                    competition_id=competition_id,
                    phase=comp.status.value,
                    guard=TRANSITIONS[(Phase.AWAITING_END_TIME, Phase.COMPLETED)],
                    require_audit_linkage=False,
                ),
            )
            return self._apply(conn, competition_id, Phase.COMPLETED, now)

        def _blockers() -> tuple[int, list[tuple[int, int]]]:
            return (
                repo.count_missing_calibration_rows(conn, competition_id),
                self.audit_linkage_gaps(conn, competition_id),
            )

        missing_baseline, gaps = _blockers()
        if missing_baseline or gaps:
            self._log_completion_blockers(comp, missing_baseline, gaps)
            return False

        def _recheck() -> None:
            repo.validate_evaluation_item_bindings(conn, competition_id)
            open_missing, open_gaps = _blockers()
            if open_missing or open_gaps:
                raise _CompletionGateOpen(open_missing, open_gaps)

        try:
            return self._apply(
                conn, competition_id, Phase.COMPLETED, now, in_txn_check=_recheck
            )
        except _CompletionGateOpen as exc:
            # A row landed (or vanished) between the pre-check and the write lock:
            # the transition rolled back; stay in AWAITING_END_TIME until the
            # pipeline/audit runner closes the gate.
            self._log_completion_blockers(comp, exc.missing_calibration_rows, exc.gaps)
            return False
        except repo.EvaluationItemBindingError as exc:
            # Direct SQL cannot race a changed factor/reference through completion:
            # the second check runs under the same write lock as the transition.
            self._log_item_binding_blocker(comp, str(exc))
            return False

    def _log_item_binding_blocker(
        self, comp: repo.CompetitionRecord, reason: str
    ) -> None:
        logger.info(
            "completion deferred: evaluation item binding invalid",
            extra=log_fields(
                competition_id=comp.competition_id,
                phase=comp.status.value,
                guard=TRANSITIONS[(Phase.AWAITING_END_TIME, Phase.COMPLETED)],
                reason="evaluation_item_binding_invalid",
                detail=reason,
            ),
        )

    def _log_completion_blockers(
        self,
        comp: repo.CompetitionRecord,
        missing_calibration_rows: int,
        gaps: list[tuple[int, int]],
    ) -> None:
        guard = TRANSITIONS[(Phase.AWAITING_END_TIME, Phase.COMPLETED)]
        if missing_calibration_rows:
            logger.info(
                "completion deferred: baseline calibration score rows missing",
                extra=log_fields(
                    competition_id=comp.competition_id,
                    phase=comp.status.value,
                    guard=guard,
                    reason="calibration_rows_missing",
                    missing_calibration_rows=missing_calibration_rows,
                ),
            )
        if gaps:
            logger.info(
                "completion deferred: scores not fully audit-linked",
                extra=log_fields(
                    competition_id=comp.competition_id,
                    phase=comp.status.value,
                    guard=guard,
                    reason="audit_linkage_gaps",
                    gap_count=len(gaps),
                    gaps=[list(pair) for pair in gaps[:20]],
                ),
            )

    def _inject_baseline(self, conn: sqlite3.Connection, competition_id: str, now: datetime) -> None:
        manifest = repo.get_manifest(conn, competition_id)
        if manifest.baseline is None:
            return
        with _txn(conn):
            contender_id = repo.insert_calibration_contender(
                conn, competition_id, manifest.baseline, now
            )
        logger.info(
            "baseline calibration contender injected (non-earning; excluded from ranking)",
            extra=log_fields(
                competition_id=competition_id,
                phase=Phase.FINALIZING_SUBMISSIONS.value,
                contender_id=contender_id,
                is_calibration=1,
            ),
        )

    # ---- pre-commitment anchoring ---------------------------------------------

    def mark_commitment_anchored(
        self,
        conn: sqlite3.Connection,
        competition_id: str,
        commitment_root: str,
        now: datetime,
        *,
        onchain_evidence: Mapping[str, object] | None = None,
    ) -> bool:
        """Record the anchored pre-commitment root (sha256 hex; the manifest digest is
        part of that commitment upstream — the audit module anchors it on chain).

        Must happen while SCHEDULED: together with the SCHEDULED -> ENROLLING guard
        this makes the commitment_anchored event always precede the enrolling
        transition in the event log. Idempotent for the same root; a
        different root, a malformed root, or a non-SCHEDULED phase raises.

        When ``onchain_evidence`` is supplied, the verified receipt event is
        appended in the SAME SQLite transaction as the lifecycle root.  A crash
        can therefore leave neither visible or both visible, never an earning
        competition root whose inclusion/archive proof was lost between commits.
        """
        _require_aware(now)
        if not _SHA256_HEX.fullmatch(commitment_root):
            raise ValueError(
                f"commitment_root must be a 64-char lowercase sha256 hex digest, "
                f"got {commitment_root!r}"
            )
        comp = repo.get_competition(conn, competition_id)
        if comp is None:
            raise IllegalTransition(
                competition_id, None, Phase.ENROLLING, "commitment_anchored", "unknown competition"
            )
        if comp.commitment_root is not None:
            if comp.commitment_root == commitment_root:
                return False  # idempotent re-anchor
            raise ValueError(
                f"competition {competition_id} is already anchored to "
                f"{comp.commitment_root}; refusing to re-anchor to {commitment_root}"
            )
        if comp.status is not Phase.SCHEDULED:
            raise IllegalTransition(
                competition_id,
                comp.status,
                Phase.ENROLLING,
                "commitment_anchored",
                "the pre-commitment can only be anchored while SCHEDULED",
            )
        with _txn(conn):
            repo.set_commitment_root(conn, competition_id, commitment_root, now)
            repo.record_event(
                conn,
                competition_id,
                "commitment_anchored",
                now,
                payload={"commitment_root": commitment_root},
            )
            if onchain_evidence is not None:
                repo.record_event(
                    conn,
                    competition_id,
                    "commitment_anchored_onchain",
                    now,
                    payload=dict(onchain_evidence),
                )
        logger.info(
            "pre-commitment anchored",
            extra=log_fields(
                competition_id=competition_id,
                phase=comp.status.value,
                commitment_root=commitment_root,
            ),
        )
        return True

    # ---- pipeline-completion transitions --------------------------------------

    def mark_submissions_backed_up(
        self, conn: sqlite3.Connection, competition_id: str, backup_ref: str, now: datetime
    ) -> bool:
        """FINALIZING_SUBMISSIONS -> VALIDATING once the audit-store submission
        backup is COMPLETED (spec §04). backup_ref is the audit-store reference of
        the completed backup (artifact digest/location) — required and persisted on
        the transition event."""
        if not backup_ref or not backup_ref.strip():
            raise ValueError(
                "backup_ref must be a non-empty audit-store backup reference "
                "(artifact digest/location)"
            )
        return self._apply(
            conn,
            competition_id,
            Phase.VALIDATING,
            now,
            payload={"backup_ref": backup_ref},
        )

    def mark_validation_complete(
        self, conn: sqlite3.Connection, competition_id: str, now: datetime
    ) -> Phase:
        """VALIDATING -> BUILDING when >=1 real contender is ACCEPTED and nothing is
        pending review; VALIDATING -> FAILED when validation left no accepted real
        contender (the calibration baseline alone cannot carry a competition)."""
        _require_aware(now)
        comp = repo.get_competition(conn, competition_id)
        if comp is None:
            raise IllegalTransition(competition_id, None, Phase.BUILDING, None, "unknown competition")
        if comp.status in (Phase.BUILDING, Phase.FAILED):
            return comp.status  # idempotent re-apply
        if comp.status is not Phase.VALIDATING:
            raise IllegalTransition(
                competition_id,
                comp.status,
                Phase.BUILDING,
                "accepted_contender_and_no_pending_review",
                "validation can only complete from VALIDATING",
            )
        pending = repo.count_pending_validation(conn, competition_id)
        if pending:
            raise IllegalTransition(
                competition_id,
                comp.status,
                Phase.BUILDING,
                "accepted_contender_and_no_pending_review",
                f"{pending} contender(s) still pending validation review",
            )
        accepted = repo.count_accepted_real_contenders(conn, competition_id)
        if accepted == 0:
            self._apply(
                conn,
                competition_id,
                Phase.FAILED,
                now,
                reason="no accepted contender after validation",
            )
            return Phase.FAILED
        self._apply(
            conn,
            competition_id,
            Phase.BUILDING,
            now,
            payload={"accepted_contenders": accepted},
        )
        return Phase.BUILDING

    def mark_builds_complete(
        self, conn: sqlite3.Connection, competition_id: str, n_built: int, now: datetime
    ) -> Phase:
        """BUILDING -> EVALUATING when >=1 image built; BUILDING -> FAILED when all
        builds failed (spec §04)."""
        _require_aware(now)
        if n_built < 0:
            raise ValueError("n_built must be >= 0")
        comp = repo.get_competition(conn, competition_id)
        if comp is None:
            raise IllegalTransition(
                competition_id, None, Phase.EVALUATING, None, "unknown competition"
            )
        if comp.status in (Phase.EVALUATING, Phase.FAILED):
            return comp.status  # idempotent re-apply
        if comp.status is not Phase.BUILDING:
            raise IllegalTransition(
                competition_id,
                comp.status,
                Phase.EVALUATING,
                "at_least_one_image_built",
                "builds can only complete from BUILDING",
            )
        if n_built == 0:
            self._apply(conn, competition_id, Phase.FAILED, now, reason="all builds failed")
            return Phase.FAILED
        self._apply(
            conn, competition_id, Phase.EVALUATING, now, payload={"images_built": n_built}
        )
        return Phase.EVALUATING

    def mark_evaluation_complete(
        self, conn: sqlite3.Connection, competition_id: str, now: datetime
    ) -> bool:
        """EVALUATING -> SCORING once every batch has completed or been terminally
        failed/zero-scored (spec §04). Verified against the DB: any
        batch not yet terminal (COMPLETED/FAILED) blocks the transition. The check
        runs both before the transaction (cheap failure) and again INSIDE the
        transition's BEGIN IMMEDIATE transaction: a batch
        inserted between check and commit still blocks — and rolls back — the
        transition."""
        _require_aware(now)
        comp = repo.get_competition(conn, competition_id)
        in_txn_check: Callable[[], None] | None = None
        if comp is not None and comp.status is Phase.EVALUATING:

            def _no_pending_batches() -> None:
                pending = repo.count_non_terminal_batches(conn, competition_id)
                if pending:
                    raise IllegalTransition(
                        competition_id,
                        Phase.EVALUATING,
                        Phase.SCORING,
                        TRANSITIONS[(Phase.EVALUATING, Phase.SCORING)],
                        f"{pending} batch(es) not yet terminal (COMPLETED/FAILED)",
                    )

            _no_pending_batches()  # cheap pre-check outside the write lock
            in_txn_check = _no_pending_batches  # authoritative re-check inside it
        return self._apply(conn, competition_id, Phase.SCORING, now, in_txn_check=in_txn_check)

    def mark_scores_persisted(
        self, conn: sqlite3.Connection, competition_id: str, now: datetime
    ) -> bool:
        """SCORING -> AWAITING_END_TIME once per-item scores are persisted.

        Verified against the DB: every accepted real contender must hold
        a performance_history row for every evaluation item. The phase transition,
        the human-review deadline (now + human_review_window_hours) and the initial
        ranking commit in ONE transaction — a crash mid-way leaves the competition
        in SCORING with nothing applied. The complete-matrix check runs both before
        the transaction (cheap failure) and again INSIDE the BEGIN IMMEDIATE
        transaction: an evaluation item inserted between check
        and commit still blocks — and rolls back — the transition.
        """
        _require_aware(now)
        to_phase = Phase.AWAITING_END_TIME
        comp = repo.get_competition(conn, competition_id)
        if comp is None:
            raise IllegalTransition(competition_id, None, to_phase, None, "unknown competition")
        if comp.status is to_phase:
            return False  # idempotent re-apply
        guard = TRANSITIONS.get((comp.status, to_phase))
        if guard is None:
            # Mirror _apply: a pipeline retry after the phase moved on is a no-op.
            already = conn.execute(
                "SELECT 1 FROM events WHERE competition_id = ? AND event_type = 'transition'"
                " AND to_phase = ? LIMIT 1",
                (competition_id, to_phase.value),
            ).fetchone()
            if already is not None:
                return False
            raise IllegalTransition(
                competition_id, comp.status, to_phase, None, "edge not in transition table"
            )
        def _matrix_complete() -> None:
            missing = repo.count_missing_item_scores(conn, competition_id)
            if missing:
                raise IllegalTransition(
                    competition_id,
                    comp.status,
                    to_phase,
                    guard,
                    f"{missing} (contender, item) pair(s) have no persisted score row",
                )

        _matrix_complete()  # cheap pre-check outside the write lock
        deadline = now + timedelta(hours=self.config.human_review_window_hours)
        try:
            with _txn(conn):
                _matrix_complete()  # authoritative re-check under BEGIN IMMEDIATE
                repo.set_status(conn, competition_id, to_phase, now)
                repo.record_event(
                    conn,
                    competition_id,
                    "transition",
                    now,
                    from_phase=comp.status,
                    to_phase=to_phase,
                    guard=guard,
                )
                repo.set_human_review_deadline(conn, competition_id, deadline, now)
                recalculate_ranks(conn, competition_id, now, manage_txn=False)
        except sqlite3.IntegrityError as exc:
            raise IllegalTransition(
                competition_id,
                comp.status,
                to_phase,
                guard,
                f"single-running-competition invariant violated: {exc}",
            ) from exc
        logger.info(
            "phase transition",
            extra=log_fields(
                competition_id=competition_id,
                phase=to_phase.value,
                from_phase=comp.status.value,
                to_phase=to_phase.value,
                guard=guard,
            ),
        )
        return True

    # ---- audit linkage check hook ---------------------------------------------

    def audit_linkage_gaps(
        self, conn: sqlite3.Connection, competition_id: str
    ) -> list[tuple[int, int]]:
        """(contender_id, item_id) score rows still lacking their per-(contender, item)
        audit_bundle_digest. The audit runner links each bundle via
        repository.set_audit_bundle_digest; a non-empty result means the competition's
        scores are not yet fully audit-linked.

        Deliberately covers EVERY performance_history row of the competition — the
        baseline calibration rows included: the baseline score drives the ratchet/crown, so
        its output must be exactly as recomputable from the audit store as any
        contender's. With config.require_audit_linkage this gates the
        AWAITING_END_TIME -> COMPLETED transition (see tick/_complete).

        A row whose digest is present but INVALID (empty / not 64 chars) counts as
        a gap too — the schema CHECK makes such a value unstorable, but the gate
        stays robust against a database that lost the constraint: presence of a
        string is never enough, only a real digest closes a gap."""
        rows = conn.execute(
            "SELECT contender_id, item_id FROM performance_history"
            " WHERE competition_id = ?"
            " AND (audit_bundle_digest IS NULL"
            "      OR typeof(audit_bundle_digest) != 'text'"
            "      OR length(audit_bundle_digest) != 64)"
            " ORDER BY contender_id, item_id",
            (competition_id,),
        ).fetchall()
        return [(row["contender_id"], row["item_id"]) for row in rows]

    # ---- failure / cancellation ----------------------------------------------

    def fail(
        self, conn: sqlite3.Connection, competition_id: str, now: datetime, reason: str
    ) -> bool:
        """Apply a FAILED edge from the current phase (only edges in the transition
        table are allowed: SCHEDULED, VALIDATING, BUILDING)."""
        return self._apply(conn, competition_id, Phase.FAILED, now, reason=reason)

    def cancel(
        self, conn: sqlite3.Connection, competition_id: str, now: datetime, reason: str
    ) -> bool:
        """ENROLLING -> CANCELLED (the only cancellation edge in the spec diagram)."""
        return self._apply(conn, competition_id, Phase.CANCELLED, now, reason=reason)
