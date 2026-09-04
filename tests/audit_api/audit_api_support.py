"""Builders for the Audit Results API suite — real store, real signer/verifier.

An :class:`AuditApi` wires an ``AuditResultsService`` over an in-memory
``AuditResultsStore`` and a ``Sha256Verifier`` keyed to the same secret the tests
sign reports with — the exact sign→verify seam production runs, only with the
deterministic double in place of a hotkey keypair. No boto3, no bittensor, no ports.

The report builders produce the SAME ``AuditReport`` shape the wave-6 auditor emits
(``vidaio.auditor.report``), signed with :class:`Sha256Signer`, so every test drives
reports only the real auditor could have produced.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from vidaio.audit_api import AuditResultsService, AuditResultsStore, Sha256Verifier
from vidaio.auditor.report import (
    WEIGHT_DERIVATION_MISMATCH,
    AuditMode,
    AuditReport,
    AuditStatus,
    ItemVerdict,
    ItemVerdictKind,
    Sha256Signer,
    WeightVerdict,
    overall_status,
)

NOW = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)

#: The shared secret the auditor's Sha256Signer signs with and the service's
#: Sha256Verifier checks against — the report-mode stand-in for a hotkey keypair.
VERIFIER_SECRET = "sn85-audit-secret"

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _hex(seed: str) -> str:
    return (seed * 64)[:64]


def pass_item(item_id: str = "i1", *, source: str = "inference") -> ItemVerdict:
    return ItemVerdict(
        source=source,
        challenge_id="c1",
        item_id=item_id,
        miner_hotkey="hk-miner",
        uid=1,
        bundle_digest=_hex("1"),
        packet_digest=_hex("2"),
        verdict=ItemVerdictKind.PASS,
    )


def fail_item(
    item_id: str = "i9", *, code: str = "SCORE_MISMATCH", source: str = "inference"
) -> ItemVerdict:
    return ItemVerdict(
        source=source,
        challenge_id="c9",
        item_id=item_id,
        miner_hotkey="hk-misreport",
        uid=9,
        bundle_digest=_hex("3"),
        packet_digest=_hex("4"),
        verdict=ItemVerdictKind.FAIL,
        code=code,
        detail="recomputed score does not match the published packet",
    )


def skip_item(item_id: str = "i5", *, source: str = "inference") -> ItemVerdict:
    """A SKIPped media item — the auditor could not recompute it (unreachable/opaque).

    A report whose media items ALL SKIP rolls up to INCONCLUSIVE (nothing recomputed —
    not clean, a needs-attention state, #8), never washed to CLEAN.
    """
    return ItemVerdict(
        source=source,
        challenge_id="c5",
        item_id=item_id,
        miner_hotkey="hk-miner",
        uid=5,
        bundle_digest=_hex("6"),
        packet_digest=_hex("7"),
        verdict=ItemVerdictKind.SKIP,
    )


def earning_skip_item(uid: int = 3) -> ItemVerdict:
    """An UNVERIFIABLE earning re-derivation (source="earning") — a SKIP, not a PASS.

    A nonzero-weight uid whose earning state the auditor could not re-derive (e.g. a
    nonzero carry-in with no prior log) rolls the epoch up to INCONCLUSIVE, never CLEAN
    (#2/D). Exists to prove the store/aggregate pass ``earning_verdicts`` as the third
    channel so the persisted verdict matches the report's own derived ``overall``.
    """
    return ItemVerdict(
        source="earning",
        challenge_id="",
        item_id=f"uid:{uid}",
        miner_hotkey=f"hk{uid}",
        uid=uid,
        bundle_digest="",
        packet_digest="",
        verdict=ItemVerdictKind.SKIP,
        code="EARNING_STATE_UNVERIFIED",
        detail="nonzero carry-in cannot be verified without the prior epoch's log",
    )


def _weight_verdict(fail: bool) -> WeightVerdict:
    return WeightVerdict(
        recomputed_weight_vector_digest=_hex("5") if fail else DIGEST_A,
        published_weight_vector_digest=DIGEST_A,
        verdict=ItemVerdictKind.FAIL if fail else ItemVerdictKind.PASS,
        code=WEIGHT_DERIVATION_MISMATCH if fail else "",
    )


def make_report(
    *,
    auditor_hotkey: str = "hk-auditor-1",
    epoch_id: int = 100,
    snapshot_digest: str = DIGEST_B,
    item_verdicts: tuple[ItemVerdict, ...] = (),
    earning_verdicts: tuple[ItemVerdict, ...] = (),
    weight_fail: bool = False,
    competition_n: int = 0,
    inference_n: int = 0,
    sign: bool = True,
    secret: str = VERIFIER_SECRET,
    overall: AuditStatus | None = None,
    audit_mode: AuditMode | str = AuditMode.BEACON,
) -> AuditReport:
    """A signed AuditReport in the auditor's exact shape (unsigned when sign=False).

    ``overall`` defaults to the honest roll-up; pass it explicitly to spoof a report
    whose self-reported verdict DISAGREES with its item verdicts (e.g. overall=CLEAN
    with a FAIL item) — used to prove the store/aggregate recompute rather than trust.
    """
    weight_verdict = _weight_verdict(weight_fail)
    report = AuditReport(
        auditor_hotkey=auditor_hotkey,
        epoch_id=epoch_id,
        audit_mode=audit_mode,
        snapshot_digest=snapshot_digest,
        pipeline_version="vidaio-scorer/1+0123456789ab",
        sampled_at=NOW,
        competition_n=competition_n,
        inference_n=inference_n,
        item_verdicts=item_verdicts,
        earning_verdicts=earning_verdicts,
        weight_verdict=weight_verdict,
        overall=overall
        if overall is not None
        else overall_status(item_verdicts, weight_verdict, earning_verdicts),
    )
    if sign:
        report = report.signed(Sha256Signer(secret))
    return report


class AuditApi:
    """The service under test + the concrete backends it was wired with.

    By default injects a ``Sha256Verifier`` (the exact sign→verify seam). Pass
    ``inject_verifier=False`` to let the service resolve its verifier from config —
    fail-closed ``RejectingVerifier`` unless ``dev_insecure_verifier`` opts into the
    Sha256 double — so tests can exercise the production default construction.
    """

    def __init__(
        self,
        *,
        api_token: str | None = None,
        secret: str = VERIFIER_SECRET,
        inject_verifier: bool = True,
        dev_insecure_verifier: bool = False,
    ) -> None:
        self.store = AuditResultsStore.open(":memory:")
        self.service = AuditResultsService(
            {
                "core": {"metrics_port": 0},
                "audit_api": {
                    "http_host": "127.0.0.1",
                    "http_port": 0,
                    "metrics_port": 0,
                    "api_token": api_token,
                    "dev_insecure_verifier": dev_insecure_verifier,
                    "verifier_secret": secret,
                },
            },
            metrics_port=0,
            store=self.store,
            verifier=Sha256Verifier(secret) if inject_verifier else None,
            now=lambda: NOW,
        )

    def metric(self, name: str, **labels: str) -> float:
        value = self.service.health.registry.get_sample_value(name, labels or None)
        return 0.0 if value is None else float(value)

    def close(self) -> None:
        self.store.close()


AUDIT_BASE_URL = "http://audit.test"


def sync_asgi_client(app: object, *, base_url: str = AUDIT_BASE_URL) -> httpx.Client:
    """A SYNC httpx.Client that drives an async ASGI ``app`` — no port, no server.

    The auditor's ``AuditResultsClient.submit`` is synchronous, so the real HTTP
    client is too; this bridge lets that sync client reach the async service app by
    running each request on a fresh loop (usable only from a sync test — there is no
    already-running loop to collide with).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        async def call() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url=base_url
            ) as client:
                return await client.request(
                    request.method,
                    str(request.url),
                    content=request.content,
                    headers=request.headers,
                )

        upstream = asyncio.run(call())
        return httpx.Response(
            upstream.status_code,
            content=upstream.content,
            headers={"content-type": "application/json"},
        )

    return httpx.Client(transport=httpx.MockTransport(handler), base_url=base_url)
