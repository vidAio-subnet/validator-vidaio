"""The seam the auditor POSTs its AuditReport through.

The real Audit Results API is wave 7 (the project design record §3.2, §7). This
wave defines the CONTRACT — :class:`AuditResultsClient` (``submit(report) -> ack``)
— plus a recording fake for tests. The auditor produces the report and submits it
through this seam; swapping in the real HTTP client later changes nothing above it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from vidaio.auditor.report import AuditReport


class SubmitAck(BaseModel):
    """The Audit Results API's acknowledgement of a persisted report (§3.2: 201)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str
    accepted: bool


class AuditResultsClient:
    """Protocol: submit a signed AuditReport, get an ack.

    Structural (duck-typed) so the real HTTP client, the recording fake, and any
    on-chain/IPFS mirror all satisfy it without importing this base.
    """

    def submit(self, report: AuditReport) -> SubmitAck:  # pragma: no cover - contract
        raise NotImplementedError


class RecordingAuditResultsClient:
    """Test double: records every submitted report and acks it.

    ``submitted`` is the reports in submission order; ``report_id`` is the report's
    own digest so the ack is deterministic and traceable back to the exact bytes.
    """

    def __init__(self) -> None:
        self.submitted: list[AuditReport] = []

    def submit(self, report: AuditReport) -> SubmitAck:
        self.submitted.append(report)
        return SubmitAck(report_id=report.report_digest(), accepted=True)

    @property
    def last(self) -> AuditReport:
        """The most recently submitted report (raises if none)."""
        return self.submitted[-1]
