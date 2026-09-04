"""The signature-verify seam — the server mirror of the auditor's ``ReportSigner``.

Every ``AuditReport`` arrives hotkey-signed over ``canonical_bytes()`` (the report
WITHOUT its signature). The Audit Results API re-derives those bytes from the parsed
report and asks a :class:`ReportVerifier` whether the presented ``auditor_signature``
is valid over them FOR THE CLAIMED ``auditor_hotkey`` — so it can reject an unsigned,
badly-signed, or misattributed report before it is ever persisted
(the project design record §3.2).

Three implementations satisfy the seam:

- :class:`HotkeySignatureVerifier` — the PRODUCTION contract. Auditor authenticity is
  TWO facts, both required: (1) the ``auditor_hotkey`` is a REGISTERED neuron on the
  subnet metagraph, and (2) an sr25519/ed25519 signature verifies against that hotkey's
  ss58. A valid signature alone is not enough — ANYONE can mint a keypair and sign, so
  without (1) an unregistered key could sign a report claiming to be an auditor. Only a
  subnet-registered validator/auditor may submit. Registration is answered by an
  injected :class:`RegisteredHotkeys` seam (production: the chain adapter's metagraph;
  tests: a fixed set); the bittensor signature call lives behind an injectable
  ``verify_fn`` seam (as the chain adapter isolates its transport) so the class imports
  and unit-tests without the SDK.
- :class:`RejectingVerifier` — the FAIL-CLOSED default. Where no verifier is configured
  or injected, every report is rejected. A misconfigured deployment refuses reports
  (loud) instead of trusting tampered reports (silent).
- :class:`Sha256Verifier` — an explicit test/dev DOUBLE, the counterpart of
  ``vidaio.auditor.report.Sha256Signer`` (same ``sha256(secret \\x00 payload)``
  construction). NOT a real signature scheme and NEVER a default: it must be opted into
  (inject it, or set ``audit_api.dev_insecure_verifier``) for chainless runs.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Callable, Iterable, Protocol, runtime_checkable


_SR25519_SIGNATURE = re.compile(r"^[0-9a-f]{128}$")


@runtime_checkable
class RegisteredHotkeys(Protocol):
    """Answers whether an ss58 is a REGISTERED neuron on the subnet metagraph.

    The registration seam of :class:`HotkeySignatureVerifier`: a valid signature only
    proves the holder of SOME keypair signed — not that the signer is an auditor. This
    provider closes that gap by asserting the ``auditor_hotkey`` is a hotkey the subnet
    actually registered. Production backs it with the chain adapter's metagraph;
    :class:`FrozenRegisteredHotkeys` is the deterministic test/static double. Structural,
    so any metagraph-backed adapter satisfies it without importing this base.
    """

    def is_registered(self, hotkey: str) -> bool: ...


class FrozenRegisteredHotkeys:
    """A FIXED registration set — the deterministic double for the metagraph.

    Holds a snapshot of registered ss58s; ``is_registered`` is exact membership. Used in
    tests and static/dev configs in place of a live metagraph read. Production injects a
    metagraph-backed :class:`RegisteredHotkeys` instead so registration tracks the chain.
    """

    def __init__(self, hotkeys: Iterable[str]) -> None:
        self._hotkeys = frozenset(hotkeys)

    def is_registered(self, hotkey: str) -> bool:
        return hotkey in self._hotkeys


class NoRegisteredHotkeys:
    """Fail-closed registration: nobody is registered, so every report is rejected.

    The default when :class:`HotkeySignatureVerifier` is constructed without a
    registration provider — a misconfigured verifier refuses everyone rather than
    silently accepting a valid signature from an unregistered (non-auditor) key.
    """

    def is_registered(self, hotkey: str) -> bool:  # noqa: ARG002
        return False


@runtime_checkable
class ReportVerifier(Protocol):
    """Verifies an auditor's hex signature over its canonical report bytes.

    ``auditor_hotkey`` is the ss58 the report CLAIMS to come from: the production
    hotkey verifier checks the signature against that on-chain identity, so a report
    cannot be attributed to an auditor who did not sign it. Structural, so the real
    hotkey verifier, the Sha256 double, and any future on-chain/IPFS-anchored check
    all satisfy it without importing this base.
    """

    def verify(
        self, payload: bytes, signature: str, *, auditor_hotkey: str
    ) -> bool: ...


def _bittensor_hotkey_verify(
    auditor_hotkey: str, payload: bytes, signature: str
) -> bool:
    """Verify an sr25519/ed25519 signature against a bittensor hotkey ss58.

    The DEPLOY-TIME implementation (needs the bittensor SDK); imported lazily so the
    module — and the whole audit_api package — loads without bittensor present. Wired
    as the default ``verify_fn`` of :class:`HotkeySignatureVerifier`; tests inject a
    fake in its place.
    """
    from bittensor import Keypair  # deploy-time dependency, imported lazily

    keypair = Keypair(ss58_address=auditor_hotkey)
    return bool(keypair.verify(payload, bytes.fromhex(signature)))


class HotkeySignatureVerifier:
    """PRODUCTION verifier: subnet REGISTRATION plus a hotkey signature.

    Authenticity requires BOTH, in this order:

    1. the claimed ``auditor_hotkey`` is a REGISTERED neuron on the subnet metagraph
       (asked of the injected :class:`RegisteredHotkeys`), and
    2. an sr25519/ed25519 signature that verifies against that ``auditor_hotkey`` ss58.

    A valid signature ALONE is insufficient: anyone can mint a keypair and sign, so a
    signature from an unregistered hotkey is rejected even though the crypto checks out —
    only a subnet-registered validator/auditor can submit. Registration defaults to the
    fail-closed :class:`NoRegisteredHotkeys` (reject everyone) when no provider is
    injected, so a misconfigured verifier never accepts a non-auditor. The bittensor
    verification call is isolated behind ``verify_fn`` (default:
    :func:`_bittensor_hotkey_verify`), the same seam pattern the chain adapter uses for
    its transport, so this class is importable and unit-testable without the SDK and a
    fake signer + a fixed registration set can drive it in tests.
    """

    def __init__(
        self,
        registered: RegisteredHotkeys | None = None,
        verify_fn: Callable[[str, bytes, str], bool] | None = None,
    ) -> None:
        self._registered: RegisteredHotkeys = registered or NoRegisteredHotkeys()
        self._verify_fn = verify_fn or _bittensor_hotkey_verify

    def verify(self, payload: bytes, signature: str, *, auditor_hotkey: str) -> bool:
        # sr25519/ed25519 signatures are exactly 64 bytes.  Reject malformed,
        # uppercase, truncated, and padded representations before either the
        # registration lookup or the SDK crypto backend sees them.
        if (
            not isinstance(signature, str)
            or not _SR25519_SIGNATURE.fullmatch(signature)
            or not auditor_hotkey
        ):
            return False
        try:
            # Registration first: an unregistered hotkey is not an auditor, so a valid
            # signature from it must still be rejected (a non-auditor cannot spoof a
            # report by holding any keypair).
            if not self._registered.is_registered(auditor_hotkey):
                return False
            return bool(self._verify_fn(auditor_hotkey, payload, signature))
        except Exception:
            # A malformed signature / ss58, or an unreadable metagraph, is a failed
            # verification, never a 500 (and fail-closed: rejected, not accepted).
            return False


class Sha256Verifier:
    """Deterministic verifier DOUBLE — the mirror of the auditor's ``Sha256Signer``.

    NOT a real signature scheme and NEVER a default (fail-closed uses
    :class:`RejectingVerifier`): an explicit test/dev stand-in that lets the
    sign→verify seam be exercised without crypto. ``verify`` is true iff ``signature``
    equals ``sha256(secret \\x00 payload)`` for this verifier's secret, which is
    exactly what ``Sha256Signer(secret).sign(payload)`` produces — so a report signed
    by an auditor holding the same secret verifies, and any other (or empty) signature
    does not. The claimed ``auditor_hotkey`` is not part of the check here: identity is
    bound only in the real :class:`HotkeySignatureVerifier`.
    """

    def __init__(self, secret: str) -> None:
        self._secret = secret.encode("utf-8")

    def _expected(self, payload: bytes) -> str:
        return hashlib.sha256(self._secret + b"\x00" + payload).hexdigest()

    def verify(
        self,
        payload: bytes,
        signature: str,
        *,
        auditor_hotkey: str = "",  # noqa: ARG002
    ) -> bool:
        if not signature:
            return False
        return hmac.compare_digest(self._expected(payload), signature)


class RejectingVerifier:
    """A verifier that rejects everything — the fail-closed default.

    Used where no verifier is configured or injected: an unverifiable report is
    never accepted (mirrors the auditor's "unknown is never assume-fine"). Making
    the absence of a verifier reject rather than accept keeps a misconfigured
    deployment from silently trusting spoofable reports.
    """

    def verify(
        self,
        payload: bytes,
        signature: str,
        *,
        auditor_hotkey: str = "",  # noqa: ARG002
    ) -> bool:
        return False
