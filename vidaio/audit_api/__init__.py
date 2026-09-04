"""vidaio.audit_api — the Audit Results API (build-wave 7, docs/DECISIONS rule 10).

The SECOND central surface of the honest rebuild (the project design record §3.2):
where the Scoring Authority API (``vidaio.authority``) publishes the epoch log, THIS
API RECEIVES the auditors' independent verdicts on it. Each validator, wearing its
auditor hat, recomputes a sample of an epoch's scores and POSTs a hotkey-signed
``AuditReport`` here; the service verifies the signature, persists it append-only
(one per auditor+epoch+audit mode; a divergent same-mode resubmission is a logged
conflict), and publishes the AGGREGATE across auditors — the investigation and
operator-alerting surface that makes any misreporting by the central Scoring Authority
publicly visible as a DISPUTED epoch. Verdicts never control weight submission.

It is the server side of the auditor's ``AuditResultsClient`` seam: report JSON in,
``{report_id, accepted}`` out (``report_id`` = the report digest). ``vidaio.dashboard``
reads it (the honesty panel); the auditor never imports it (it holds only the seam).
"""

from vidaio.audit_api.aggregate import (
    EPOCH_CLEAN,
    EPOCH_DISPUTED,
    EPOCH_INCONCLUSIVE,
    EPOCH_UNAUDITED,
    epoch_rollup,
    epoch_status,
    feed_entry,
)
from vidaio.audit_api.client import (
    AuditResultsConflict,
    AuditResultsUnavailable,
    HttpAuditResultsClient,
)
from vidaio.audit_api.config import AuditResultsConfig
from vidaio.audit_api.service import AuditResultsService
from vidaio.audit_api.store import (
    AuditResultsStore,
    RecordOutcome,
    RecordResult,
    StoredReport,
)
from vidaio.audit_api.verify import (
    FrozenRegisteredHotkeys,
    HotkeySignatureVerifier,
    NoRegisteredHotkeys,
    RegisteredHotkeys,
    RejectingVerifier,
    ReportVerifier,
    Sha256Verifier,
)

__all__ = [
    # config
    "AuditResultsConfig",
    # service
    "AuditResultsService",
    # store
    "AuditResultsStore",
    "StoredReport",
    "RecordResult",
    "RecordOutcome",
    # aggregate
    "epoch_status",
    "epoch_rollup",
    "feed_entry",
    "EPOCH_CLEAN",
    "EPOCH_DISPUTED",
    "EPOCH_INCONCLUSIVE",
    "EPOCH_UNAUDITED",
    # verify seam
    "ReportVerifier",
    "HotkeySignatureVerifier",
    "Sha256Verifier",
    "RejectingVerifier",
    # registration seam
    "RegisteredHotkeys",
    "FrozenRegisteredHotkeys",
    "NoRegisteredHotkeys",
    # client
    "HttpAuditResultsClient",
    "AuditResultsConflict",
    "AuditResultsUnavailable",
]
