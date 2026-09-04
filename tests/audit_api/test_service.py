"""POST /audit/report — signature-gated persistence + the conflict rule.

Every test drives the REAL service (Sha256Verifier keyed to the auditor's signer,
append-only store) over an ASGI client. The central assertions: a signed report is
persisted and its ``report_id`` is the report digest (the auditor's SubmitAck
contract); an unsigned/badly-signed report is refused before persistence; a divergent
resubmission keeps the FIRST report and records the conflict as a signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from audit_api_support import AuditApi, fail_item, make_report, skip_item

POST = "/audit/report"


def _body(report) -> dict:
    return report.model_dump(mode="json")


# -- happy path: a signed report is persisted; report_id == report digest ----------


async def test_signed_report_persisted_and_returns_report_id(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    report = make_report(epoch_id=100, inference_n=3)
    resp = await client.post(POST, json=_body(report))
    assert resp.status_code == 201
    body = resp.json()
    assert body == {"report_id": report.report_digest(), "accepted": True}

    # It is persisted and reconstructable to the exact report.
    stored = api.store.get(report.report_digest())
    assert stored is not None
    assert stored.report == report
    assert stored.overall == "CLEAN"
    # metric counted by verdict
    assert api.metric("vidaio_audit_reports_received_total", verdict="CLEAN") == 1.0


async def test_inconclusive_report_persisted_not_integrity_error(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    """An all-SKIP (INCONCLUSIVE) report persists as 201 — not a 500 IntegrityError.

    The auditor emits INCONCLUSIVE when every sampled media item SKIPped (nothing could
    be recomputed). The 0003 migration widened the overall CHECK to admit INCONCLUSIVE,
    so the row stores instead of raising sqlite3.IntegrityError at POST (#8).
    """
    report = make_report(
        epoch_id=105, inference_n=1, item_verdicts=(skip_item(),)
    )
    resp = await client.post(POST, json=_body(report))
    assert resp.status_code == 201  # persisted, NOT a 500 / IntegrityError
    assert resp.json() == {"report_id": report.report_digest(), "accepted": True}

    stored = api.store.get(report.report_digest())
    assert stored is not None
    assert stored.report == report
    # recomputed overall is INCONCLUSIVE — its own column value, not CLEAN/DISPUTED
    assert stored.overall == "INCONCLUSIVE"
    # counted under its own verdict label
    assert api.metric("vidaio_audit_reports_received_total", verdict="INCONCLUSIVE") == 1.0
    # INCONCLUSIVE is not a dispute: the disputed-epochs gauge stays 0
    assert api.metric("vidaio_audit_disputed_epochs") == 0.0


async def test_resubmitting_the_identical_report_is_idempotent(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    report = make_report(epoch_id=101)
    first = await client.post(POST, json=_body(report))
    assert first.status_code == 201
    again = await client.post(POST, json=_body(report))
    assert again.status_code == 200  # idempotent, not a conflict
    assert again.json() == {"report_id": report.report_digest(), "accepted": True}
    # persisted exactly once
    assert len(api.store.for_epoch(101)) == 1


# -- signature gate: unsigned (401) / bad signature (403) --------------------------


async def test_unsigned_report_rejected_401(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    report = make_report(epoch_id=102, sign=False)
    resp = await client.post(POST, json=_body(report))
    assert resp.status_code == 401
    assert resp.json()["detail"]["error"] == "report_unsigned"
    assert api.store.get(report.report_digest()) is None
    assert api.metric("vidaio_audit_reports_rejected_total", reason="report_unsigned") == 1.0


async def test_badly_signed_report_rejected_403(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    # Signed with the WRONG secret — verifies false against the service's verifier.
    report = make_report(epoch_id=103, secret="not-the-secret")
    resp = await client.post(POST, json=_body(report))
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == "report_signature_invalid"
    assert api.store.get(report.report_digest()) is None
    assert (
        api.metric("vidaio_audit_reports_rejected_total", reason="report_signature_invalid")
        == 1.0
    )


async def test_tampered_report_breaks_its_signature(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    """A field mutated after signing no longer verifies over canonical_bytes()."""
    report = make_report(epoch_id=104, inference_n=1)
    body = _body(report)
    body["inference_n"] = 999  # tamper: the signature was over the original bytes
    resp = await client.post(POST, json=body)
    assert resp.status_code == 403


# -- the conflict rule: keep the first, record the divergence ----------------------


async def test_divergent_resubmission_is_a_conflict_first_kept(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    first = make_report(auditor_hotkey="hk-auditor-1", epoch_id=200, snapshot_digest="a" * 64)
    # SAME auditor + epoch, DIFFERENT content (a FAIL item) => different digest.
    divergent = make_report(
        auditor_hotkey="hk-auditor-1",
        epoch_id=200,
        snapshot_digest="c" * 64,
        item_verdicts=(fail_item(),),
    )
    assert first.report_digest() != divergent.report_digest()

    r1 = await client.post(POST, json=_body(first))
    assert r1.status_code == 201
    r2 = await client.post(POST, json=_body(divergent))
    assert r2.status_code == 409
    conflict_body = r2.json()
    assert conflict_body["error"] == "report_conflict"
    assert conflict_body["accepted"] is False
    # the KEPT report is the first one
    assert conflict_body["report_id"] == first.report_digest()

    # The store kept the first and never persisted the divergent one.
    stored = api.store.for_epoch(200)
    assert len(stored) == 1
    assert stored[0].report_id == first.report_digest()
    # the conflict itself is recorded as a signal (per-epoch + total).
    assert api.store.conflicts_for_epoch(200) == 1
    assert api.metric("vidaio_audit_report_conflicts_total") == 1.0
    # and it surfaces on /audit/status.
    status = (await client.get("/audit/status", params={"epoch_id": 200})).json()
    assert status["conflicts"] == 1


async def test_disputed_conflicting_report_still_marks_epoch_disputed(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    """A CLEAN first report cannot bury a later DISPUTED divergent one.

    The same auditor's second, divergent report is a conflict (first kept, not
    persisted) — but if that rejected report is DISPUTED, the epoch is still DISPUTED
    and the dispute surfaces on /audit/status.
    """
    clean_first = make_report(auditor_hotkey="hk-a", epoch_id=210, snapshot_digest="a" * 64)
    disputed_divergent = make_report(
        auditor_hotkey="hk-a",
        epoch_id=210,
        snapshot_digest="c" * 64,
        item_verdicts=(fail_item(),),
    )
    assert (await client.post(POST, json=_body(clean_first))).status_code == 201
    assert (await client.post(POST, json=_body(disputed_divergent))).status_code == 409

    status = (await client.get("/audit/status", params={"epoch_id": 210})).json()
    # only the CLEAN first report is persisted...
    assert status["clean"] == 1 and status["disputed"] == 0
    # ...but the DISPUTED divergent report flips the epoch and is surfaced as a signal
    assert status["verdict"] == "DISPUTED"
    assert status["conflicts"] == 1
    assert status["disputed_conflicts"] == 1
    # and the disputed-epochs gauge agrees with the surface
    assert api.metric("vidaio_audit_disputed_epochs") == 1.0


async def test_inconclusive_conflicting_report_is_recorded_not_500(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    """A divergent INCONCLUSIVE resubmission is a RECORDED conflict, not a 500 (#5).

    The 0002 conflict-ledger CHECK only admitted CLEAN/DISPUTED, so a divergent
    report whose recomputed verdict is INCONCLUSIVE (an all-SKIP media sample) raised
    sqlite3.IntegrityError at the conflict INSERT -> an UNRECORDED 500. 0004 widens
    the CHECK to INCONCLUSIVE (as 0003 did for the reports table): the conflict is
    now stored and surfaced. INCONCLUSIVE is NOT a dispute, so the epoch verdict is
    unaffected — disputed_conflicts stays 0 (only a real DISPUTED divergence flips it).
    """
    clean_first = make_report(auditor_hotkey="hk-a", epoch_id=220, snapshot_digest="a" * 64)
    inconclusive_divergent = make_report(
        auditor_hotkey="hk-a",
        epoch_id=220,
        snapshot_digest="c" * 64,
        inference_n=1,
        item_verdicts=(skip_item(),),  # all-SKIP -> recomputed overall INCONCLUSIVE
    )
    assert clean_first.report_digest() != inconclusive_divergent.report_digest()

    assert (await client.post(POST, json=_body(clean_first))).status_code == 201
    # Before 0004 this POST 500'd (IntegrityError) and recorded nothing.
    r2 = await client.post(POST, json=_body(inconclusive_divergent))
    assert r2.status_code == 409
    assert r2.json()["error"] == "report_conflict"

    # The first is kept; the conflict is recorded as a signal.
    stored = api.store.for_epoch(220)
    assert len(stored) == 1 and stored[0].report_id == clean_first.report_digest()
    assert api.store.conflicts_for_epoch(220) == 1
    assert api.metric("vidaio_audit_report_conflicts_total") >= 1.0

    status = (await client.get("/audit/status", params={"epoch_id": 220})).json()
    assert status["conflicts"] == 1
    # INCONCLUSIVE divergence is not a dispute: the epoch verdict is unaffected.
    assert status["disputed_conflicts"] == 0
    assert status["verdict"] != "DISPUTED"


# -- bearer gating on POST (401 missing / 403 wrong) — reads stay open -------------


async def test_post_auth_gating_and_reads_open() -> None:
    a = AuditApi(api_token="s3cr3t-validator-token")
    try:
        report = make_report(epoch_id=300)
        transport = httpx.ASGITransport(app=a.service.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://a.test") as c:
            # missing bearer -> 401
            assert (await c.post(POST, json=_body(report))).status_code == 401
            # wrong token -> 403
            wrong = {"Authorization": "Bearer nope"}
            assert (await c.post(POST, json=_body(report), headers=wrong)).status_code == 403
            # right token -> 201
            ok = {"Authorization": "Bearer s3cr3t-validator-token"}
            assert (await c.post(POST, json=_body(report), headers=ok)).status_code == 201
            # READS are never gated (the honesty surface is public).
            assert (await c.get("/audit/status", params={"epoch_id": 300})).status_code == 200
            assert (await c.get("/audit/feed")).status_code == 200
            assert (await c.get("/audit/epochs")).status_code == 200
            assert (await c.get("/healthz")).status_code == 200
    finally:
        a.close()


async def test_malformed_report_body_is_422(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    resp = await client.post(POST, json={"not": "an audit report"})
    assert resp.status_code == 422
