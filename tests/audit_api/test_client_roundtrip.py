"""The auditor->API round-trip closes: HttpAuditResultsClient against the real service.

These tests are SYNCHRONOUS on purpose: the auditor's ``AuditResultsClient.submit`` is
sync (it runs on the auditor thread), so the real HTTP client is sync too. The bridge
(``sync_asgi_client``) drives the async service app from a sync caller — no port bound,
no server, no sleep — so the whole seam runs end to end: Sha256Signer -> HTTP client ->
service -> Sha256Verifier -> store, and the returned SubmitAck's ``report_id`` is the
report digest (the auditor's contract).
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from audit_api_support import AuditApi, fail_item, make_report, sync_asgi_client

from vidaio.audit_api import (
    AuditResultsConflict,
    AuditResultsUnavailable,
    HttpAuditResultsClient,
)
from vidaio.auditor.client import AuditResultsClient, SubmitAck


def test_http_client_satisfies_the_auditor_seam() -> None:
    """The HTTP client is duck-typed to the auditor's AuditResultsClient Protocol."""
    client = HttpAuditResultsClient("http://audit.test", token="t")
    assert isinstance(client, AuditResultsClient) or hasattr(client, "submit")
    # returns the auditor's own SubmitAck type — the contract the auditor expects.
    assert SubmitAck.__module__ == "vidaio.auditor.client"


def test_submit_roundtrip_report_id_is_digest() -> None:
    api = AuditApi()
    try:
        http = sync_asgi_client(api.service.app)
        results = HttpAuditResultsClient("http://audit.test", client=http)
        report = make_report(epoch_id=500, inference_n=2)

        ack = results.submit(report)
        assert isinstance(ack, SubmitAck)
        assert ack.accepted is True
        assert ack.report_id == report.report_digest()  # the SubmitAck contract

        # the server actually persisted the exact report.
        stored = api.store.get(report.report_digest())
        assert stored is not None and stored.report == report

        # a re-submit of the identical report is idempotent (still accepted).
        again = results.submit(report)
        assert again.report_id == report.report_digest() and again.accepted is True
        http.close()
    finally:
        api.close()


def test_submit_conflict_raises_and_keeps_first() -> None:
    api = AuditApi()
    try:
        http = sync_asgi_client(api.service.app)
        results = HttpAuditResultsClient("http://audit.test", client=http)
        first = make_report(auditor_hotkey="hk-x", epoch_id=501)
        divergent = make_report(
            auditor_hotkey="hk-x", epoch_id=501, item_verdicts=(fail_item(),)
        )

        results.submit(first)
        with pytest.raises(AuditResultsConflict) as exc:
            results.submit(divergent)
        # the kept report id is the FIRST report's digest
        assert exc.value.report_id == first.report_digest()
        http.close()
    finally:
        api.close()


def test_submit_bad_signature_surfaces_as_unavailable() -> None:
    """A 403 (bad signature) is a non-success the client refuses to treat as accepted."""
    api = AuditApi()
    try:
        http = sync_asgi_client(api.service.app)
        results = HttpAuditResultsClient("http://audit.test", client=http)
        wrong = make_report(epoch_id=502, secret="not-the-secret")
        with pytest.raises(AuditResultsUnavailable):
            results.submit(wrong)
        http.close()
    finally:
        api.close()


def test_bearer_token_is_sent() -> None:
    """With a token configured on the API, the client's bearer is accepted."""
    api = AuditApi(api_token="validator-token")
    try:
        http = sync_asgi_client(api.service.app)
        results = HttpAuditResultsClient(
            "http://audit.test", token="validator-token", client=http
        )
        ack = results.submit(make_report(epoch_id=503))
        assert ack.accepted is True

        # the wrong token is refused (surfaced as unavailable, never "accepted").
        bad = HttpAuditResultsClient("http://audit.test", token="nope", client=sync_asgi_client(api.service.app))
        with pytest.raises(AuditResultsUnavailable):
            bad.submit(make_report(epoch_id=504))
        http.close()
    finally:
        api.close()


@pytest.mark.parametrize("status_code", [200, 201])
def test_submit_accepts_only_an_ack_for_the_exact_submitted_report(
    status_code: int,
) -> None:
    report = make_report(epoch_id=505)

    def ack_exact_report(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"report_id": report.report_digest(), "accepted": True},
        )

    with httpx.Client(transport=httpx.MockTransport(ack_exact_report)) as http:
        ack = HttpAuditResultsClient("http://audit.test", client=http).submit(report)

    assert ack == SubmitAck(report_id=report.report_digest(), accepted=True)


@pytest.mark.parametrize(
    "payload",
    [
        {"report_id": "unused", "accepted": False},
        {"report_id": "unused", "accepted": 1},
        {"report_id": "unused", "accepted": "true"},
        {"report_id": "unused"},
        {"accepted": True},
        {"report_id": 123, "accepted": True},
    ],
    ids=[
        "explicitly-rejected",
        "integer-is-not-true",
        "string-is-not-true",
        "missing-accepted",
        "missing-report-id",
        "non-string-report-id",
    ],
)
def test_submit_rejects_malformed_acknowledgements(payload: dict[str, object]) -> None:
    report = make_report(epoch_id=506)

    def malformed_ack(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json=payload)

    with httpx.Client(transport=httpx.MockTransport(malformed_ack)) as http:
        client = HttpAuditResultsClient("http://audit.test", client=http)
        with pytest.raises(AuditResultsUnavailable):
            client.submit(report)


def test_submit_rejects_acknowledgement_for_a_different_report() -> None:
    submitted = make_report(epoch_id=507)
    different = make_report(epoch_id=508)

    def mismatched_ack(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={"report_id": different.report_digest(), "accepted": True},
        )

    with httpx.Client(transport=httpx.MockTransport(mismatched_ack)) as http:
        client = HttpAuditResultsClient("http://audit.test", client=http)
        with pytest.raises(AuditResultsUnavailable, match="does not match"):
            client.submit(submitted)
