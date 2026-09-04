"""AuditResultsService — the honesty surface (build-wave 7, the project design record §3.2).

A ``BaseService`` (FastAPI) that RECEIVES the auditors' signed ``AuditReport``s,
verifies each signature, persists it append-only, and publishes the AGGREGATE across
auditors so any misreporting by the central Scoring Authority becomes publicly visible.
It is the server side of the auditor's ``AuditResultsClient`` seam: report JSON in,
``{report_id, accepted}`` out (``report_id`` = the report digest, matching the auditor's
``SubmitAck`` contract).

Boundary (the project design record §3.2): POST /audit/report is bearer-gated (validators
carry the token) AND every report is hotkey-signed — a report is rejected before it is
persisted if it is unsigned (401) or badly signed (403). READS are OPEN: the honesty
surface is meant to be public (the dashboard and anyone else read it without a token).

Modes (the project design record rule 8): the same service code runs everywhere; only the
injected ``ReportVerifier`` differs — a real ``HotkeySignatureVerifier`` in production.
The default is FAIL-CLOSED: with no verifier injected and ``dev_insecure_verifier``
unset, the service uses ``RejectingVerifier`` and refuses every report, so a
misconfigured deployment is loud rather than silently spoofable. The ``Sha256Verifier``
double is used only when explicitly opted into (injected in tests, or
``audit_api.dev_insecure_verifier: true`` for chainless runs).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from datetime import datetime, timezone
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query
from prometheus_client import Counter, Gauge

from vidaio.auditor.report import AuditReport
from vidaio.core import log_fields, section
from vidaio.services.base import BaseService

from vidaio.audit_api.aggregate import epoch_rollup, epoch_status, feed_entry
from vidaio.audit_api.config import AuditResultsConfig
from vidaio.audit_api.store import AuditResultsStore, RecordOutcome
from vidaio.audit_api.verify import RejectingVerifier, ReportVerifier, Sha256Verifier

#: Typed authorization / rejection failures (the `error` field of every 4xx detail).
AUTH_MISSING = "auth_token_missing"
AUTH_INVALID = "auth_token_invalid"
REPORT_UNSIGNED = "report_unsigned"
REPORT_SIGNATURE_INVALID = "report_signature_invalid"
REPORT_CONFLICT = "report_conflict"


def _bearer(authorization: str | None) -> str | None:
    """Extract `Authorization: Bearer <token>`; None if absent or malformed."""
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


def _tokens_match(presented: str, configured: str) -> bool:
    """Constant-time compare (over digests, so length leaks nothing either)."""
    return hmac.compare_digest(
        hashlib.sha256(presented.encode("utf-8")).digest(),
        hashlib.sha256(configured.encode("utf-8")).digest(),
    )


class AuditResultsService(BaseService):
    name = "audit-results"

    def __init__(
        self,
        raw_config: dict[str, Any],
        *,
        metrics_port: int | None = None,
        store: AuditResultsStore | None = None,
        verifier: ReportVerifier | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        cfg = section(raw_config, "audit_api", AuditResultsConfig)
        super().__init__(
            raw_config,
            metrics_port=metrics_port if metrics_port is not None else cfg.metrics_port,
        )
        self.config = cfg
        self._now = now or (lambda: datetime.now(timezone.utc))

        # Store + signature verifier. The store is injected in tests (in-memory);
        # built from config in production (own SQLite).
        self._store: AuditResultsStore = (
            store if store is not None else AuditResultsStore.open(cfg.db_path)
        )
        # Verifier resolution is FAIL-CLOSED. An explicitly injected verifier wins
        # (a real HotkeySignatureVerifier in production, a Sha256Verifier double in
        # tests). Otherwise the ONLY way to get the insecure Sha256 double is to opt
        # in via `dev_insecure_verifier` (chainless runs); with neither, the default
        # is RejectingVerifier — a misconfigured deployment refuses every report
        # rather than silently accepting tampered reports.
        if verifier is not None:
            self._verifier: ReportVerifier = verifier
        elif cfg.dev_insecure_verifier:
            self._verifier = Sha256Verifier(cfg.verifier_secret)
        else:
            self._verifier = RejectingVerifier()

        self._http_api_ok = True
        self.health.register_check("http_api", lambda: self._http_api_ok)

        reg = self.health.registry
        self._m_received = Counter(
            "vidaio_audit_reports_received_total",
            "AuditReports persisted, by their overall verdict",
            ["verdict"],
            registry=reg,
        )
        self._m_received_by_mode = Counter(
            "vidaio_audit_reports_received_by_mode_total",
            "AuditReports persisted, by origin mode and overall verdict",
            ["audit_mode", "verdict"],
            registry=reg,
        )
        self._m_rejected = Counter(
            "vidaio_audit_reports_rejected_total",
            "AuditReports refused before persistence, by typed reason",
            ["reason"],
            registry=reg,
        )
        self._m_conflicts = Counter(
            "vidaio_audit_report_conflicts_total",
            "Divergent resubmissions for an already-reported (auditor, epoch, mode)",
            registry=reg,
        )
        self._m_conflicts_by_mode = Counter(
            "vidaio_audit_report_conflicts_by_mode_total",
            "Divergent resubmissions, by audit report origin mode",
            ["audit_mode"],
            registry=reg,
        )
        self._m_auth_failures = Counter(
            "vidaio_audit_auth_failures_total",
            "Rejected report submissions by typed authorization error",
            ["error"],
            registry=reg,
        )
        self._m_disputed_epochs = Gauge(
            "vidaio_audit_disputed_epochs",
            "Epochs with at least one DISPUTED auditor report",
            registry=reg,
        )
        self._refresh_disputed_gauge()
        self.app = self._build_app()

    # -- authorization + verification ------------------------------------------

    def _authorize(self, authorization: str | None) -> None:
        """Gate POST /audit/report on `audit_api.api_token` when configured.

        401 on a missing/malformed bearer, 403 on a present-but-wrong token. Open
        (no gate) when `api_token` is unset — a loopback/dev posture; production
        sets it (validators carry it). Reads are never gated here.
        """
        configured = (self.config.api_token or "").strip()
        if not configured:
            return
        presented = _bearer(authorization)
        if presented is None:
            self._m_auth_failures.labels(AUTH_MISSING).inc()
            raise HTTPException(
                status_code=401,
                detail={
                    "error": AUTH_MISSING,
                    "message": "POST /audit/report requires 'Authorization: Bearer "
                    "<audit_api.api_token>'",
                },
            )
        if not _tokens_match(presented, configured):
            self._m_auth_failures.labels(AUTH_INVALID).inc()
            raise HTTPException(
                status_code=403,
                detail={
                    "error": AUTH_INVALID,
                    "message": "the presented bearer token is not valid for the "
                    "audit results API",
                },
            )

    def _verify_signature(self, report: AuditReport) -> None:
        """Reject an unsigned (401) or badly-signed (403) report before persisting.

        The signature is over ``canonical_bytes()`` (the report WITHOUT its
        signature), re-derived from the parsed report and checked against the CLAIMED
        ``auditor_hotkey`` — so a mutated field breaks the signature and a report
        cannot be attributed to an auditor who did not sign it.
        """
        if not report.auditor_signature:
            self._m_rejected.labels(REPORT_UNSIGNED).inc()
            raise HTTPException(
                status_code=401,
                detail={
                    "error": REPORT_UNSIGNED,
                    "message": "the AuditReport carries no auditor_signature; a report "
                    "must be hotkey-signed over canonical_bytes()",
                },
            )
        if not self._verifier.verify(
            report.canonical_bytes(),
            report.auditor_signature,
            auditor_hotkey=report.auditor_hotkey,
        ):
            self._m_rejected.labels(REPORT_SIGNATURE_INVALID).inc()
            raise HTTPException(
                status_code=403,
                detail={
                    "error": REPORT_SIGNATURE_INVALID,
                    "message": "the auditor_signature does not verify over the report's "
                    "canonical bytes",
                },
            )

    def _refresh_disputed_gauge(self) -> None:
        self._m_disputed_epochs.set(self._store.disputed_epoch_count())

    # -- the API ---------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="vidaio audit results", docs_url=None, redoc_url=None)

        @app.get("/healthz")
        async def healthz() -> dict[str, Any]:
            return {"service": self.name, "status": "ok"}

        @app.post("/audit/report")
        async def submit_report(
            report: AuditReport, authorization: str | None = Header(default=None)
        ) -> Any:
            from fastapi.responses import JSONResponse

            self._authorize(authorization)
            self._verify_signature(report)
            result = self._store.record(report, received_at=self._now().isoformat())

            if result.outcome is RecordOutcome.CONFLICT:
                self._m_conflicts.inc()
                self._m_conflicts_by_mode.labels(report.audit_mode.value).inc()
                self._refresh_disputed_gauge()
                conflict_fields = log_fields(
                    epoch_id=report.epoch_id,
                    auditor_hotkey=report.auditor_hotkey,
                    audit_mode=report.audit_mode.value,
                    rejected_overall=report.overall.value,
                    kept_report_id=result.report_id,
                    remediation="manual",
                    weight_action="unchanged",
                )
                if report.overall.value == "DISPUTED":
                    self.log.critical(
                        "conflicting DISPUTED audit report recorded; operator"
                        " investigation is required and weight-setting remains unchanged",
                        extra=conflict_fields,
                    )
                else:
                    self.log.warning(
                        "conflicting audit report recorded for operator investigation;"
                        " weight-setting remains unchanged",
                        extra=conflict_fields,
                    )
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": REPORT_CONFLICT,
                        "report_id": result.report_id,
                        "accepted": False,
                        "message": "an earlier, different report is already on record "
                        f"for auditor {report.auditor_hotkey!r} epoch {report.epoch_id} "
                        f"mode {report.audit_mode.value!r}; "
                        "the first is kept and this divergence was recorded as a signal",
                    },
                )

            if result.outcome is RecordOutcome.NEW:
                self._m_received.labels(result.kept.overall).inc()
                self._m_received_by_mode.labels(
                    report.audit_mode.value, result.kept.overall
                ).inc()
                self._refresh_disputed_gauge()
                finding_fields = log_fields(
                    epoch_id=report.epoch_id,
                    auditor_hotkey=report.auditor_hotkey,
                    audit_mode=report.audit_mode.value,
                    verdict=result.kept.overall,
                    report_id=result.report_id,
                    remediation="manual",
                    weight_action="unchanged",
                )
                if result.kept.overall == "DISPUTED":
                    self.log.critical(
                        "central Audit Results API accepted a DISPUTED finding;"
                        " operator investigation is required and weight-setting"
                        " remains unchanged",
                        extra=finding_fields,
                    )
                elif result.kept.overall == "INCONCLUSIVE":
                    self.log.warning(
                        "central Audit Results API accepted an INCONCLUSIVE finding;"
                        " operator investigation is required and weight-setting"
                        " remains unchanged",
                        extra=finding_fields,
                    )
                status_code = 201
            else:  # DUPLICATE — idempotent re-post of the identical report
                status_code = 200
            return JSONResponse(
                status_code=status_code,
                content={"report_id": result.report_id, "accepted": True},
            )

        @app.get("/audit/status")
        async def audit_status(epoch_id: int = Query(...)) -> dict[str, Any]:
            reports = self._store.for_epoch(epoch_id)
            conflicts = self._store.conflicts_for_epoch(epoch_id)
            disputed_conflicts = self._store.disputed_conflicts_for_epoch(epoch_id)
            return epoch_status(
                epoch_id,
                reports,
                conflicts=conflicts,
                disputed_conflicts=disputed_conflicts,
            )

        @app.get("/audit/feed")
        async def audit_feed(limit: int | None = Query(default=None)) -> dict[str, Any]:
            n = self.config.feed_default_limit if limit is None else limit
            n = max(1, min(n, self.config.feed_max_limit))
            reports = self._store.recent(n)
            return {
                "limit": n,
                "disputed_epochs": self._store.disputed_epoch_count(),
                "total_conflicts": self._store.total_conflicts(),
                "reports": [feed_entry(s) for s in reports],
            }

        @app.get("/audit/epochs")
        async def audit_epochs() -> dict[str, Any]:
            rollups = [
                epoch_rollup(
                    eid,
                    self._store.for_epoch(eid),
                    conflicts=self._store.conflicts_for_epoch(eid),
                    disputed_conflicts=self._store.disputed_conflicts_for_epoch(eid),
                )
                for eid in self._store.epoch_ids()
            ]
            return {"epochs": rollups}

        return app

    # -- lifecycle -------------------------------------------------------------

    def _on_api_exit(self, api: asyncio.Task[Any]) -> None:
        """The uvicorn task ended without a stop being requested — fatal exit so a
        supervisor restarts an audit-results API whose port nobody answers (the
        exit-code contract, vidaio.services.base)."""
        self._http_api_ok = False
        error: BaseException | None = None
        try:
            error = api.exception()
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            error = None
        detail = "" if error is None else f"{type(error).__name__}: {error}"
        self.fail_fatal(
            "audit-results HTTP API exited unexpectedly — no honesty surface is serving"
            f" (port={self.config.http_port} error={detail})"
        )

    async def run(self) -> None:
        server = uvicorn.Server(
            uvicorn.Config(
                self.app,
                host=self.config.http_host,
                port=self.config.http_port,
                log_level="warning",
            )
        )

        # SystemExit (uvicorn's bind-failure exit) is a BaseException: awaited bare
        # it would tear the loop down instead of being reported.
        async def _serve() -> None:
            try:
                await server.serve()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                raise RuntimeError(
                    f"uvicorn exited: {type(exc).__name__}: {exc}"
                ) from exc

        api = asyncio.create_task(_serve(), name="audit-results-http")
        stop = asyncio.create_task(self.stopping.wait(), name="audit-results-stop")
        try:
            await asyncio.wait({api, stop}, return_when=asyncio.FIRST_COMPLETED)
            if api.done() and not self.stopping.is_set():
                self._on_api_exit(api)
        finally:
            server.should_exit = True
            stop.cancel()
            await asyncio.gather(api, return_exceptions=True)
            self.close()

    def close(self) -> None:
        self._store.close()
