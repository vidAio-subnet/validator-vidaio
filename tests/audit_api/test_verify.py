"""The signature-verify seam: fail-closed default + the hotkey-signature contract.

Two layers are covered:

- unit: :class:`HotkeySignatureVerifier` requires BOTH subnet REGISTRATION (the claimed
  ``auditor_hotkey`` is a registered neuron) AND a valid signature over the claimed
  ``auditor_hotkey`` — a valid signature from an UNREGISTERED key is rejected, so a
  non-auditor cannot spoof a report by holding any keypair. The fail-closed
  :class:`RejectingVerifier` refuses everything, and an un-provisioned registration seam
  (:class:`NoRegisteredHotkeys`) rejects everyone.
- integration: the service's DEFAULT construction (no verifier injected) is fail-closed
  — a correctly-signed report is REJECTED unless the insecure Sha256 double is
  explicitly opted into via ``dev_insecure_verifier``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from audit_api_support import NOW, AUDIT_BASE_URL, AuditApi, make_report

from vidaio.audit_api.verify import (
    FrozenRegisteredHotkeys,
    HotkeySignatureVerifier,
    NoRegisteredHotkeys,
    RejectingVerifier,
    Sha256Verifier,
)

POST = "/audit/report"
VALID_HOTKEY_SIGNATURE = "ab" * 64


def _body(report) -> dict:
    return report.model_dump(mode="json")


# -- HotkeySignatureVerifier: authenticity = REGISTRATION + a valid signature ------


def test_hotkey_verifier_binds_signature_to_claimed_hotkey() -> None:
    """A signature valid for the CLAIMED, REGISTERED ss58 verifies; the crypto is a seam."""
    seen: dict[str, object] = {}

    def fake_verify(auditor_hotkey: str, payload: bytes, signature: str) -> bool:
        seen["args"] = (auditor_hotkey, payload, signature)
        # the keypair holder can sign for its own hotkey; anyone else cannot.
        return auditor_hotkey == "hk-registered" and signature == VALID_HOTKEY_SIGNATURE

    v = HotkeySignatureVerifier(
        registered=FrozenRegisteredHotkeys({"hk-registered"}),
        verify_fn=fake_verify,
    )

    # a real auditor's signature over its own REGISTERED hotkey verifies
    assert (
        v.verify(
            b"canonical-bytes",
            VALID_HOTKEY_SIGNATURE,
            auditor_hotkey="hk-registered",
        )
        is True
    )
    assert seen["args"] == (
        "hk-registered",
        b"canonical-bytes",
        VALID_HOTKEY_SIGNATURE,
    )

    # an adversary presenting the same signature under a different hotkey cannot spoof
    assert (
        v.verify(
            b"canonical-bytes",
            VALID_HOTKEY_SIGNATURE,
            auditor_hotkey="hk-adversary",
        )
        is False
    )


def test_hotkey_verifier_rejects_valid_signature_from_unregistered_hotkey() -> None:
    """A VALID signature from an UNREGISTERED hotkey is rejected: registration is required.

    Anyone can mint a keypair and produce a signature that cryptographically verifies;
    only a subnet-registered validator/auditor may submit, so an unregistered signer is
    refused even though its signature is genuine.
    """
    signed: dict[str, bool] = {}

    def always_valid(auditor_hotkey: str, payload: bytes, signature: str) -> bool:
        signed["called"] = True
        return True  # the signature itself is cryptographically valid

    v = HotkeySignatureVerifier(
        registered=FrozenRegisteredHotkeys({"hk-registered"}),
        verify_fn=always_valid,
    )

    # registered signer: accepted
    assert (
        v.verify(b"bytes", VALID_HOTKEY_SIGNATURE, auditor_hotkey="hk-registered")
        is True
    )
    # unregistered signer with an equally-genuine signature: rejected, and the crypto
    # seam is not even consulted (registration is checked first, fail-closed).
    signed.clear()
    assert (
        v.verify(b"bytes", VALID_HOTKEY_SIGNATURE, auditor_hotkey="hk-unregistered")
        is False
    )
    assert "called" not in signed


def test_hotkey_verifier_no_registration_provider_is_fail_closed() -> None:
    """With no registration seam injected (NoRegisteredHotkeys), everyone is rejected."""
    v = HotkeySignatureVerifier(verify_fn=lambda hk, p, s: True)
    assert v.verify(b"x", VALID_HOTKEY_SIGNATURE, auditor_hotkey="hk-anything") is False
    assert NoRegisteredHotkeys().is_registered("hk") is False


def test_hotkey_verifier_rejects_empty_and_never_500s() -> None:
    reg = FrozenRegisteredHotkeys({"hk"})

    def fake_verify(auditor_hotkey: str, payload: bytes, signature: str) -> bool:
        return True  # would accept — but empties must short-circuit before calling

    v = HotkeySignatureVerifier(registered=reg, verify_fn=fake_verify)
    assert v.verify(b"x", "", auditor_hotkey="hk") is False
    assert v.verify(b"x", VALID_HOTKEY_SIGNATURE, auditor_hotkey="") is False

    # a malformed ss58 / signature (backend raises) is a failed verification, not a 500
    def boom(auditor_hotkey: str, payload: bytes, signature: str) -> bool:
        raise ValueError("bad ss58")

    assert (
        HotkeySignatureVerifier(registered=reg, verify_fn=boom).verify(
            b"x", VALID_HOTKEY_SIGNATURE, auditor_hotkey="hk"
        )
        is False
    )

    # an unreadable metagraph (registration lookup raises) is fail-closed, not a 500
    class Boom:
        def is_registered(self, hotkey: str) -> bool:
            raise RuntimeError("metagraph unreachable")

    assert (
        HotkeySignatureVerifier(
            registered=Boom(), verify_fn=lambda hk, p, s: True
        ).verify(b"x", VALID_HOTKEY_SIGNATURE, auditor_hotkey="hk")
        is False
    )


@pytest.mark.parametrize(
    "signature",
    (
        "ab" * 63,
        "ab" * 65,
        "AB" * 64,
        "gg" * 64,
    ),
)
def test_hotkey_verifier_rejects_noncanonical_signature_shape_before_crypto(
    signature: str,
) -> None:
    called = False

    def crypto(_hotkey: str, _payload: bytes, _signature: str) -> bool:
        nonlocal called
        called = True
        return True

    verifier = HotkeySignatureVerifier(
        registered=FrozenRegisteredHotkeys({"hk"}), verify_fn=crypto
    )
    assert verifier.verify(b"payload", signature, auditor_hotkey="hk") is False
    assert called is False


def test_rejecting_verifier_refuses_everything() -> None:
    v = RejectingVerifier()
    assert v.verify(b"x", "any-signature", auditor_hotkey="hk") is False
    assert v.verify(b"", "", auditor_hotkey="") is False


def test_sha256_double_still_verifies_over_secret() -> None:
    from vidaio.auditor.report import Sha256Signer

    payload = b"the canonical report bytes"
    good = Sha256Signer("s").sign(payload)
    v = Sha256Verifier("s")
    assert v.verify(payload, good, auditor_hotkey="ignored") is True
    assert v.verify(payload, "deadbeef", auditor_hotkey="ignored") is False
    assert v.verify(payload, "", auditor_hotkey="ignored") is False


# -- integration: default construction is FAIL-CLOSED ------------------------------


async def test_default_construction_is_fail_closed() -> None:
    """No verifier injected + dev flag off => RejectingVerifier: a valid report is refused."""
    a = AuditApi(inject_verifier=False)  # production default construction
    try:
        report = make_report(epoch_id=400)  # correctly signed with the shared secret
        transport = httpx.ASGITransport(app=a.service.app)
        async with httpx.AsyncClient(transport=transport, base_url=AUDIT_BASE_URL) as c:
            resp = await c.post(POST, json=_body(report))
            # rejected (not silently accepted) — the fail-closed posture
            assert resp.status_code == 403
            assert resp.json()["detail"]["error"] == "report_signature_invalid"
        assert a.store.get(report.report_digest()) is None
    finally:
        a.close()


async def test_dev_insecure_opt_in_accepts_signed_reports() -> None:
    """dev_insecure_verifier explicitly opts into the Sha256 double for chainless runs."""
    a = AuditApi(inject_verifier=False, dev_insecure_verifier=True)
    try:
        report = make_report(epoch_id=401)
        transport = httpx.ASGITransport(app=a.service.app)
        async with httpx.AsyncClient(transport=transport, base_url=AUDIT_BASE_URL) as c:
            resp = await c.post(POST, json=_body(report))
            assert resp.status_code == 201
        assert a.store.get(report.report_digest()) is not None
    finally:
        a.close()


# -- integration: the HotkeySignatureVerifier's REGISTRATION gate over the service --


def _hotkey_service(registered: set[str]):
    """A service wired with the production HotkeySignatureVerifier over a fixed
    registration set. ``verify_fn`` accepts any non-empty signature (the crypto seam is
    exercised elsewhere) so the test isolates the REGISTRATION check."""
    from vidaio.audit_api import (
        AuditResultsService,
        AuditResultsStore,
        FrozenRegisteredHotkeys,
        HotkeySignatureVerifier,
    )

    store = AuditResultsStore.open(":memory:")
    verifier = HotkeySignatureVerifier(
        registered=FrozenRegisteredHotkeys(registered),
        verify_fn=lambda hk, payload, sig: bool(sig),
    )
    service = AuditResultsService(
        {"core": {"metrics_port": 0}, "audit_api": {"http_port": 0, "metrics_port": 0}},
        metrics_port=0,
        store=store,
        verifier=verifier,
        now=lambda: NOW,
    )
    return service, store


async def test_registered_auditor_accepted_unregistered_rejected() -> None:
    """Valid signature + REGISTERED hotkey => 201; valid signature + UNREGISTERED => 403.

    Proves the production verifier's registration gate over the full POST path: only a
    subnet-registered auditor can submit; an unregistered key cannot spoof a report even
    though its signature verifies.
    """
    service, store = _hotkey_service({"hk-registered-auditor"})
    try:
        transport = httpx.ASGITransport(app=service.app)
        async with httpx.AsyncClient(transport=transport, base_url=AUDIT_BASE_URL) as c:
            registered = make_report(
                auditor_hotkey="hk-registered-auditor", epoch_id=410
            ).model_copy(update={"auditor_signature": VALID_HOTKEY_SIGNATURE})
            resp = await c.post(POST, json=_body(registered))
            assert resp.status_code == 201
            assert store.get(registered.report_digest()) is not None

            tampered = make_report(
                auditor_hotkey="hk-unregistered", epoch_id=411
            ).model_copy(update={"auditor_signature": VALID_HOTKEY_SIGNATURE})
            resp = await c.post(POST, json=_body(tampered))
            assert resp.status_code == 403
            assert resp.json()["detail"]["error"] == "report_signature_invalid"
            assert store.get(tampered.report_digest()) is None
    finally:
        store.close()
