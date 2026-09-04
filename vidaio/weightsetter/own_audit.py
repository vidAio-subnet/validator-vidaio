"""Legacy own-audit classifier retained for focused compatibility tests.

Production weight-setting no longer imports or constructs this helper. The live
``own-auditor`` uses the ``vidaio.auditor`` spine from a separate OS
process/container, CPU-recomputes every committed item, and owns its own durable
``AuditCursor`` plus pending-report outbox. Findings are signed, centrally reported,
and manually remediated; they cannot enter the current ``WeightSetter`` because its
constructor has no audit seam.

This module preserves the earlier single-process classifier, optional contiguous CLEAN
ledger/cursor, and backfill behavior for regression and backward-compatibility tests. Its
reports remain report-only, but it is not production runtime wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from vidaio.audit.store import AuditStore
from vidaio.auditor import AuditReport, AuditStatus, SamplePolicy
from vidaio.auditor.sampling import NO_BEACON
from vidaio.auditor.service import Auditor, _fold_matches
from vidaio.epoch import EpochLog
from vidaio.tokenomics.ewma import is_excluded
from vidaio.weightsetter.own_audit_cursor import OwnAuditCursor
from vidaio.weightsetter.own_audit_ledger import OwnAuditLedger


class _NullRecomputer:
    """A recomputer that refuses to run — used when the gate samples NO media.

    The cheap earning-state re-fold + weight re-derivation cover the full vector
    without any media recompute; when `SamplePolicy(sample_rate=0.0)` is used no
    media item is sampled, so this is never invoked. If a caller sets a positive
    sample rate they MUST inject a real recomputer instead (else this raises loudly
    rather than silently PASSing an unrecomputed item).
    """

    def recompute(self, bundle: object, artifacts: object):  # pragma: no cover
        raise RuntimeError(
            "own-audit gate has no recomputer wired but a media item was sampled —"
            " inject a RealScoreRecomputer or keep the audit media rate at 0.0"
        )


@dataclass(frozen=True, slots=True)
class OwnAuditVerdict:
    """The gate's decision on one epoch log."""

    ok: bool                    # True only when the own audit is CLEAN
    status: AuditStatus         # CLEAN / DISPUTED / INCONCLUSIVE
    report: AuditReport         # the full report (verdicts + detail) for logging


class OwnAuditGate:
    """Run the legacy own-audit classifier and return its report-only verdict.

    Wraps an `Auditor` over the shared object store. `review(log)` returns an
    `OwnAuditVerdict`; `ok` says only whether the report is CLEAN. Current production
    `WeightSetter` does not accept this object. Optional legacy/test seams:

    - `beacon_for(log) -> str` supplies the post-finalization anchor beacon so the
      media sampling (if any) is un-steerable (#10); best-effort — a failure falls
      back to `NO_BEACON` (irrelevant at sample_rate 0).
    - `prior_log_for(log) -> EpochLog | None` supplies the previous epoch's log so the
      earning carry-in is chained back (a missing prior only downgrades a nonzero
      carry-in to an earning SKIP, never a false CLEAN nor a false DISPUTED).
    - `is_genesis_for(log) -> bool` supplies the INDEPENDENT determination of whether the
      log is the genuine genesis epoch. It is the only authorisation
      for a MISSING `prior_log_digest`: a NON-genesis epoch that omits it is a broken
      chain ⇒ DISPUTED (an authority resetting chained earning/window/crown state at an
      arbitrary epoch). When not wired the gate defaults to treating the epoch as genesis
      (preserving prior behaviour); callers that need this historical check must wire an
      independent genesis determination.
    - `report_only=True` returns the signed Auditor report directly and bypasses all
      legacy ledger/cursor/backfill policy. Test/legacy callers may use this mode: a prior finding
      cannot suppress later reports, and bookkeeping cannot manufacture a second status.
    - `ledger` is the DURABLE own-audited-CLEAN store. When wired, an
      epoch with a NONZERO carry-in for any uid is cleared ONLY if the predecessor
      (epoch_id-1, prior_log_digest) is a RECORDED CLEAN digest — otherwise the carry-in is
      UNVERIFIED ⇒ INCONCLUSIVE. Each CLEAN clear is RECORDED so the chain extends.
      Genesis (authenticated) and a zero carry-in need no predecessor. When NOT wired the
      legacy classifier cannot vouch for a nonzero carry-in and returns INCONCLUSIVE.
    - `cursor` + `fetch_log_for` turn `review` into a CONTIGUOUS
      backfilling walk, mirroring the public auditor loop's durable-contiguous-cursor design.
      `cursor` is the durable highest-contiguously-CLEAN epoch; `fetch_log_for(epoch_id)`
      fetches+verifies a HISTORICAL epoch's log (wired to the shared provider's
      `resolve_epoch`, itself using `pointer_for`). When BOTH are wired, `review(latest_log)`
      first BACKFILLS every epoch in `cursor+1 .. latest-1` (own-audit each, record CLEAN,
      advance the cursor) so `latest`'s predecessor is present, then reviews `latest`. A
      SKIPPED epoch is thus filled in instead of stranding the chain, and a restart resumes
      from the durable cursor. A withheld/unavailable/non-clearing predecessor stops the
      legacy backfill and makes `latest` INCONCLUSIVE, but an honest contiguous
      chain never wedges. On a fresh cursor the walk starts from `audit_floor` (the
      authenticated genesis floor, consistent with the auditor loop). When either seam is
      unwired the gate stays the pure SINGLE-epoch gate (report/dryrun and the unit tests).
    """

    def __init__(
        self,
        *,
        auditor: Auditor,
        store: AuditStore,
        policy: SamplePolicy | None = None,
        recomputer: object | None = None,
        beacon_for: Callable[[EpochLog], str | None] | None = None,
        prior_log_for: Callable[[EpochLog], EpochLog | None] | None = None,
        is_genesis_for: Callable[[EpochLog], bool] | None = None,
        ledger: OwnAuditLedger | None = None,
        cursor: OwnAuditCursor | None = None,
        fetch_log_for: Callable[[int], EpochLog | None] | None = None,
        audit_floor: int | None = None,
        report_only: bool = False,
    ) -> None:
        self._auditor = auditor
        self._store = store
        self._is_genesis_for = is_genesis_for
        self._ledger = ledger
        self._cursor = cursor
        self._fetch_log_for = fetch_log_for
        self._audit_floor = audit_floor
        self._report_only = report_only
        #: default: earning-state re-fold + weight re-derivation over the FULL vector,
        #: NO media sampled (min_samples=0, so sample_rate 0 truly samples nothing —
        #: media coverage is the separate auditor loop's job).
        self._policy = policy or SamplePolicy(sample_rate=0.0, min_samples=0)
        self._recomputer = recomputer or _NullRecomputer()
        self._beacon_for = beacon_for
        self._prior_log_for = prior_log_for

    def review(self, log: EpochLog, *, now: datetime | None = None) -> OwnAuditVerdict:
        """Own-audit ``log`` and return a verdict, optionally backfilling legacy state.

        When a durable `cursor` AND a `fetch_log_for` seam are wired, this walks the
        own-audit contiguously from `cursor+1` (or `audit_floor` on a fresh cursor) up to
        `log.epoch_id`, BACKFILLING every skipped predecessor (fetch+verify+own-audit each,
        record CLEAN, advance the cursor) so `log`'s carry-in predecessor is present on an
        honest chain. The walk STOPS at the first predecessor that does not clear or cannot
        be fetched, leaving `log` INCONCLUSIVE on its now-missing predecessor — never a
        false CLEAN, never a permanent wedge on an honest chain. `report_only=True` skips
        this policy entirely and returns the current signed Auditor report.
        """
        now = now or datetime.now(timezone.utc)
        if self._report_only:
            # Report-only compatibility mode inspects the current epoch independently. It
            # returns exactly the signed Auditor verdict and never consults/mutates the
            # legacy submit-gate ledger/cursor, so a disputed/missing predecessor cannot
            # suppress reports for all later epochs.
            report = self._audit_report(log, now)
            return OwnAuditVerdict(
                ok=report.overall is AuditStatus.CLEAN,
                status=report.overall,
                report=report,
            )
        if self._cursor is not None and self._fetch_log_for is not None:
            # an internal review: the backfill REPORTS whether it reached the target's
            # IMMEDIATE predecessor CONTIGUOUSLY. If it STOPPED EARLY (a predecessor in the
            # range was unavailable/non-clearing), the own-audited-CLEAN chain has a GAP
            # before the target, so the contiguous invariant would break if we recorded the
            # target or advanced the cursor past the gap — even for a ZERO-carry / burn target
            # (whose self-consistency alone never proves the skipped predecessor honest). So we
            # own-audit the target for a report but do NOT record it and mark INCONCLUSIVE
            # (INCONCLUSIVE) without advancing; the gap epoch is retried next pass and, once
            # fillable, the target proceeds.
            reached = self._backfill_predecessors(log, now)
            verdict = self._review_one(log, now, record=reached)
            if not reached:
                if verdict.ok:
                    # Target is self-consistent CLEAN but sits PAST a gap — INCONCLUSIVE
                    # (do not record, do not advance the cursor past the unfilled predecessor).
                    return OwnAuditVerdict(
                        ok=False, status=AuditStatus.INCONCLUSIVE, report=verdict.report
                    )
                return verdict
            if verdict.ok:
                # The target own-audited CLEAN (its predecessor is recorded, on an honest
                # contiguous chain) AND was durably recorded (record=True succeeded, else
                # _review_one would be INCONCLUSIVE) — advance the durable cursor
                # past it too, so the next attempt resumes at target+1. Guarded to stay
                # monotonic (the authority may not have advanced, or a concurrent pass already
                # recorded it).
                last = self._cursor.last_clean()
                if last is None or log.epoch_id > last:
                    try:
                        self._cursor.advance_to(log.epoch_id)
                    except Exception:
                        pass
            return verdict
        return self._review_one(log, now)

    def _backfill_predecessors(self, log: EpochLog, now: datetime) -> bool:
        """Contiguously own-audit + record every epoch in `cursor+1 .. log.epoch_id-1`.

        Mirrors the public auditor loop's `_audit_once`: walk in ascending order, advancing
        the durable cursor ONLY past an epoch that own-audits CLEAN; stop at the first epoch
        that is non-clearing or cannot be fetched, so no later epoch is recorded ahead of it
        and the target `log` becomes INCONCLUSIVE on its missing predecessor. A fetch failure
        (transient or 404) stops the legacy walk and is retried next attempt.

        Returns True iff the walk reached the target's IMMEDIATE predecessor CONTIGUOUSLY
        (every epoch in `start .. log.epoch_id-1` own-audited CLEAN and recorded); False if it
        STOPPED EARLY. The caller marks the target INCONCLUSIVE and refuses
        to advance the cursor past the gap even for a zero-carry / burn target — the contiguous
        own-audited-CLEAN invariant must hold regardless of the target's carry-in.
        """
        assert self._cursor is not None and self._fetch_log_for is not None
        last = self._cursor.last_clean()
        if last is None:
            # Fresh cursor: start from the AUTHENTICATED floor (the operator-configured
            # genesis floor, consistent with the auditor loop), else the target itself (no
            # backfill — the pre-cursor behaviour) when no floor is wired.
            start = self._audit_floor if self._audit_floor is not None else log.epoch_id
        else:
            start = last + 1
        for epoch_id in range(start, log.epoch_id):
            try:
                pred_log = self._fetch_log_for(epoch_id)
            except Exception:
                # A withheld / unavailable / tampered predecessor cannot be verified. Stop
                # the walk; the target is then INCONCLUSIVE on its missing
                # predecessor and this epoch is retried next attempt (no permanent wedge on
                # an honest chain, where the fetch succeeds).
                return False
            if pred_log is None:
                return False
            verdict = self._review_one(pred_log, now)
            if not verdict.ok:
                # This predecessor did not clear (DISPUTED / INCONCLUSIVE, or a ledger write
                # failure per an internal review) — stop the walk so nothing is recorded ahead
                # of it and the target is INCONCLUSIVE on the broken chain.
                return False
            # A CLEAN predecessor was recorded in the ledger by _review_one; advance the
            # durable contiguous cursor past it (ascending, so strictly increasing).
            try:
                self._cursor.advance_to(epoch_id)
            except Exception:
                # A cursor write race/failure leaves the legacy classifier recoverable; the ledger
                # already recorded the CLEAN digest, so the target's predecessor check still
                # sees it. The cursor re-advances next attempt.
                pass
        return True

    def _review_one(
        self, log: EpochLog, now: datetime, *, record: bool = True
    ) -> OwnAuditVerdict:
        report = self._audit_report(log, now)
        if report.overall is not AuditStatus.CLEAN:
            return OwnAuditVerdict(ok=False, status=report.overall, report=report)

        # Legacy contiguous classification below is retained for explicit consumers/tests.
        # Production WeightSetter does not construct this class.
        is_genesis = True
        if self._is_genesis_for is not None:
            try:
                is_genesis = bool(self._is_genesis_for(log))
            except Exception:
                is_genesis = False

        # an internal review (CRITICAL): the auditor cleared this epoch as self-consistent —
        # its earning re-fold verified the nonzero carry-in equals the PREDECESSOR's stated
        # accumulate_score. But that only chains the NUMBER back one epoch; it does NOT prove
        # the predecessor was itself own-audited CLEAN. An untrusted authority could publish a
        # structurally-invalid predecessor carrying an INJECTED accumulator, chain it into
        # this self-consistent epoch, and pass here. So a NONZERO carry-in is trusted ONLY
        # when the predecessor (epoch_id-1, prior_log_digest) is a RECORDED own-audited-CLEAN
        # digest in the durable ledger. If it is not (fresh gate with no history, or an
        # unaudited/injected predecessor, or no ledger wired), the carry-in is UNVERIFIED ⇒
        # INCONCLUSIVE. Genuine genesis (is_genesis, prior_log_digest None) and a zero
        # carry-in are exempt.
        if self._has_nonzero_carry_in(log) and not (is_genesis and log.prior_log_digest is None):
            if not self._predecessor_recorded_clean(log):
                return OwnAuditVerdict(
                    ok=False, status=AuditStatus.INCONCLUSIVE, report=report
                )

        # A genuine CLEAN clear: RECORD this epoch's (epoch_id, log_digest) so the durable,
        # contiguous own-audited-CLEAN chain extends and a LATER epoch carrying this one in
        # can be trusted.
        #
        # `record` is False when the legacy contiguous caller detected a GAP before the
        # target: do not record it ahead of an unfilled predecessor. A ledger write failure
        # becomes INCONCLUSIVE, so the cursor never advances past an absent ledger entry.
        if self._ledger is not None and record:
            try:
                self._ledger.record_clean(log.epoch_id, log.log_digest())
            except Exception:
                return OwnAuditVerdict(
                    ok=False, status=AuditStatus.INCONCLUSIVE, report=report
                )
        return OwnAuditVerdict(ok=True, status=AuditStatus.CLEAN, report=report)

    def _audit_report(self, log: EpochLog, now: datetime) -> AuditReport:
        """Build the Auditor's signed report without legacy ledger/cursor policy."""
        beacon = NO_BEACON
        if self._beacon_for is not None:
            try:
                beacon = self._beacon_for(log) or NO_BEACON
            except Exception:
                beacon = NO_BEACON
        prior: EpochLog | None = None
        if self._prior_log_for is not None:
            try:
                prior = self._prior_log_for(log)
            except Exception:
                prior = None
        # an internal review: independently decide whether this log is the true genesis. A
        # failed determination fails CLOSED to NON-genesis (is_genesis False): a missing
        # prior_log_digest is then treated as a broken chain, never washed as genesis.
        is_genesis = True
        if self._is_genesis_for is not None:
            try:
                is_genesis = bool(self._is_genesis_for(log))
            except Exception:
                is_genesis = False
        report = self._auditor.audit_epoch(
            log,
            self._store,
            self._policy,
            self._recomputer,
            now,
            beacon=beacon,
            prior_log=prior,
            is_genesis=is_genesis,
        )
        return report

    @staticmethod
    def _has_nonzero_carry_in(log: EpochLog) -> bool:
        """True iff any uid's earning state DEPENDS on the PREDECESSOR epoch's log.

        an internal review (CRITICAL): the round-12 gate looked ONLY at `earning_inputs`
        (the EXPLICIT carry-in `prior_accumulate_score`), but the auditor's earning re-fold
        ALSO trusts an IMPLICIT carry-forward: a ZERO-weight uid with a POSITIVE
        `accumulate_score` and NO `EarningInput` this epoch is a pure carry-forward whose
        value `_carry_forward_verdict` chains to the predecessor's stated accumulator
        (service.py `_earning_verdict_for_uid` / `_carry_forward_verdict`). An unaudited
        predecessor could inject an accumulator, the next zero-weight epoch carries it
        IMPLICITLY (no EarningInput), becomes recorded CLEAN, and a later earning epoch
        consumes it — bypassing the predecessor-ledger requirement. So the gate's
        "depends on predecessor" predicate must cover BOTH:

        - EXPLICIT: an `EarningInput` with a NONZERO `prior_accumulate_score` (a zero carry-in
          folds from genesis and depends on no predecessor — same tolerance as the auditor's
          `_carry_in_check` zero check, so a carry-in the auditor treats as 0.0 is not
          spuriously flagged here);
        - IMPLICIT: a uid with NO `EarningInput` this epoch whose stated `accumulate_score`
          is positive (and not the exclusion sentinel) — the exact predicate the auditor uses
          to route a uid through `_carry_forward_verdict` (`_earning_verdicts`: `not
          is_excluded(m.accumulate_score) and m.accumulate_score > 0.0`), minus the burn uid
          (which the auditor discards).

        In the legacy contiguous classifier, no accumulator — explicit or implicit — is
        declared CLEAN without a recorded-CLEAN predecessor. Report-only mode relies on
        the Auditor report directly and never turns this bookkeeping into enforcement.
        """
        manifest = log.audit_manifest
        # EXPLICIT carry-in: an EarningInput chained against the prior epoch by _carry_in_check.
        if any(
            not _fold_matches(ei.prior_accumulate_score, 0.0)
            for ei in manifest.earning_inputs.values()
        ):
            return True
        # IMPLICIT carry-forward: a positive stated accumulator with NO EarningInput this
        # epoch — routed through the auditor's _carry_forward_verdict, which chains it to the
        # predecessor's value. Mirror that verdict's audited-set predicate EXACTLY.
        for m in log.miners:
            if m.uid == log.burn_uid:
                continue
            if m.uid in manifest.earning_inputs:
                continue
            if not is_excluded(m.accumulate_score) and m.accumulate_score > 0.0:
                return True
        return False

    def _predecessor_recorded_clean(self, log: EpochLog) -> bool:
        """True iff the chained predecessor is a recorded CLEAN entry (gap-aware, v16)."""
        if self._ledger is None or log.prior_log_digest is None:
            return False
        prior_epoch_id = log.prior_epoch_id
        if prior_epoch_id is None:
            return False
        return self._ledger.is_clean(prior_epoch_id, log.prior_log_digest)


__all__ = ["OwnAuditGate", "OwnAuditVerdict"]
