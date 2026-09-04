"""The HTTP ``AuditResultsClient`` — the auditor's real POST path to this API.

The auditor produces a signed ``AuditReport`` and submits it through the
``AuditResultsClient`` seam (``vidaio.auditor.client``: ``submit(report) -> SubmitAck``).
:class:`HttpAuditResultsClient` is the production impl of that seam — it POSTs the
report JSON to ``/audit/report`` and returns the server's ``{report_id, accepted}``
ack. It is duck-typed to the auditor's ``AuditResultsClient`` (returns the auditor's
own ``SubmitAck``), so the auditor's ``audit_and_submit`` drives it unchanged.

The seam is synchronous (the auditor runs on its own thread), so this client is a
synchronous ``httpx`` caller. A test may inject a transport (an ``httpx.MockTransport``
that drives the ASGI app) so the whole round-trip runs without binding a port.
"""

from __future__ import annotations

import httpx

from vidaio.auditor.client import SubmitAck
from vidaio.auditor.report import AuditReport

#: The route the auditor POSTs a report to (matches the service).
REPORT_ROUTE = "/audit/report"


class AuditResultsUnavailable(RuntimeError):
    """The Audit Results API could not be reached or answered unusably.

    Transport failure, a non-2xx / non-409 status, or a malformed/mismatched
    success acknowledgement. The auditor treats a failed submit as "not yet
    submitted" and retries on its own cadence — a dropped report is never
    mistaken for an accepted one.
    """


class AuditResultsConflict(RuntimeError):
    """The API kept an EARLIER, different report for this (auditor, epoch, mode).

    A 409: the auditor already committed to a verdict for this epoch and this
    submission diverges from it. ``report_id`` is the KEPT (first) report's id; the
    divergence was recorded server-side as a signal. Not retryable as-is.
    """

    def __init__(self, report_id: str, message: str) -> None:
        super().__init__(message)
        self.report_id = report_id


class HttpAuditResultsClient:
    """POST a signed AuditReport to the Audit Results API; return its ack.

    ``base_url`` is the API root; ``token`` (when set) is sent as the bearer the
    POST is gated on. ``client`` lets a caller supply its own transport (tests wire
    one onto the ASGI app); ``base_url`` then only supplies the path prefix.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._client = client

    def submit(self, report: AuditReport) -> SubmitAck:
        """Submit ``report``; return the ``{report_id, accepted}`` ack.

        Raises :class:`AuditResultsConflict` on a 409 (an earlier, different report
        is on record) and :class:`AuditResultsUnavailable` on transport failure or
        any other non-success status or unusable success acknowledgement.
        """
        url = self._base_url + REPORT_ROUTE
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        body = report.model_dump(mode="json")
        try:
            if self._client is not None:
                response = self._client.post(
                    url, json=body, headers=headers, timeout=self._timeout
                )
            else:
                response = httpx.post(
                    url, json=body, headers=headers, timeout=self._timeout
                )
        except httpx.HTTPError as exc:
            raise AuditResultsUnavailable(
                f"could not reach the audit results API at {url}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code == 409:
            data = self._json(response)
            raise AuditResultsConflict(
                str(data.get("report_id", "")),
                str(data.get("message", "audit report conflict")),
            )
        if response.status_code not in (200, 201):
            raise AuditResultsUnavailable(
                f"audit results API rejected the report: HTTP {response.status_code} "
                f"{response.text}"
            )
        data = self._json(response)
        # Treat the acknowledgement as a receipt for THESE exact signed report
        # bytes, not merely as a truthy response from the endpoint.  Coercing
        # values here (``bool("false")`` or ``str(...)``) can otherwise advance
        # the auditor's durable outbox/cursor after a malformed proxy response or
        # after the API acknowledges a different report.  Only the two literal
        # protocol fields below constitute acceptance.
        expected_report_id = report.report_digest()
        if data.get("accepted") is not True:
            raise AuditResultsUnavailable(
                "audit results API returned an acknowledgement without "
                "accepted=true"
            )
        if data.get("report_id") != expected_report_id:
            raise AuditResultsUnavailable(
                "audit results API acknowledgement report_id does not match the "
                "submitted report"
            )
        return SubmitAck(report_id=expected_report_id, accepted=True)

    @staticmethod
    def _json(response: httpx.Response) -> dict:
        try:
            payload = response.json()
        except ValueError as exc:
            raise AuditResultsUnavailable(
                f"audit results API returned non-JSON (HTTP {response.status_code})"
            ) from exc
        if not isinstance(payload, dict):
            raise AuditResultsUnavailable("audit results API returned a non-object body")
        return payload
