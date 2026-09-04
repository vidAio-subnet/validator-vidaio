"""Audit Results API configuration (config section: ``audit_api``).

The Audit Results API is the SECOND central surface (the project design record rule 10,
the project design record §3.2, build-wave 7). It RECEIVES the auditors' signed
``AuditReport``s, persists them append-only, and is the honesty surface a dashboard
reads: any misreporting by the central Scoring Authority becomes publicly visible as a
DISPUTED epoch. This config carries the HTTP/metrics ports, the SQLite report-store
path, the bearer token gating report submission, and the signature-verify seam's key.

Trust model (the project design record §3.2, §8 [PENDING]): every report is
hotkey-signed by its auditor over ``canonical_bytes()``, so the owner cannot substitute
or silently drop a report without that auditor noticing its own missing/altered entry.
The POST is additionally bearer-gated (validators carry the token); READS are open —
the honesty surface is meant to be public (the dashboard and anyone else can read it).

Credentials are never stored here. Verifier selection is FAIL-CLOSED: in production a
REAL ``HotkeySignatureVerifier`` is injected at construction (the deploy-time impl);
with nothing injected the service defaults to ``RejectingVerifier`` (refuse every
report) UNLESS ``dev_insecure_verifier`` is explicitly set, which opts into the
insecure ``Sha256Verifier`` double (keyed by ``verifier_secret``) for chainless runs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class AuditResultsConfig(BaseModel):
    """Schema for the ``audit_api:`` section of config."""

    model_config = ConfigDict(extra="forbid")

    # -- HTTP API + metrics (see vidaio.services.protocol port table) ------------
    #: The report-ingest + honesty-read API bind host/port (audit-results http 8710).
    http_host: str = "0.0.0.0"
    http_port: int = 8710
    #: Prometheus metrics port (audit-results 9112).
    metrics_port: int = 9112

    # -- report store (this service's own SQLite, own migrations) ----------------
    #: Append-only store of received AuditReports (one per auditor+epoch+audit mode) + the
    #: conflict ledger (a divergent resubmission is itself a signal).
    db_path: Path = Path("./data/audit_results.db")

    # -- auth ---------------------------------------------------------------------
    #: Bearer token gating POST /audit/report (validators carry it). None = OPEN —
    #: a loopback/dev posture; PRODUCTION DEPLOYMENTS MUST SET IT. READS are always
    #: open (the honesty surface is public); /healthz is always open (liveness).
    api_token: str | None = None

    #: FAIL-CLOSED verifier selection. False (default) => with no verifier injected the
    #: service uses ``RejectingVerifier`` and refuses every report — a misconfigured
    #: deployment is loud, never spoofable. Set true ONLY for chainless/dev runs to opt
    #: into the insecure ``Sha256Verifier`` double (keyed by ``verifier_secret``).
    #: PRODUCTION never sets this: it injects a real ``HotkeySignatureVerifier``.
    dev_insecure_verifier: bool = False

    #: The shared secret the insecure ``Sha256Verifier`` double checks signatures
    #: against (mirrors the auditor's ``Sha256Signer`` secret) — used ONLY when
    #: ``dev_insecure_verifier`` is set. Ignored in production (a real hotkey verifier
    #: is injected). NOT a real signature scheme; never trusted by default.
    verifier_secret: str = ""

    # -- read sizing --------------------------------------------------------------
    #: Default / hard-max number of reports GET /audit/feed returns (newest first).
    feed_default_limit: int = Field(default=50, ge=1)
    feed_max_limit: int = Field(default=500, ge=1)
