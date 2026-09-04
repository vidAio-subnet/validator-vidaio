"""Registered-hotkey authentication for miner/validator-facing surfaces (P2).

Every API a miner or validator talks to must verify the caller is a REGISTERED
hotkey on this subnet (and, where the route is validator-only, that it carries a
validator permit) — and refuse otherwise. The design generalizes the two
primitives the tree already trusts (`vidaio/services/artifact_auth.py`):
domain-separated canonical-JSON signing and a chain-registry seam — into three
pieces:

* ``RegisteredHotkeyRegistry`` — ONE cached metagraph view per process:
  ``is_registered`` / ``has_validator_permit`` / ``alpha_stake``, refreshed at
  most every ``ttl_seconds`` and NEVER per request (the per-request
  ``chain.refresh()`` of the older ``ChainValidatorRegistry`` was an RPC-flood
  vector). Fail-closed: an unavailable registry raises
  ``HotkeyRegistryUnavailable`` (HTTP 503) — never allow-through.
* **Scheme A — signed request** (low-rate, high-value calls): headers
  ``X-Vidaio-Hotkey/-Timestamp/-Nonce/-Signature`` where the signature covers a
  domain-separated canonical digest of ``(method, path, sha256(body),
  timestamp, nonce)``. Replay is refused by the same bounded-nonce-cache
  semantics as artifact-auth.
* **Scheme B — challenge-minted session token** (high-rate polling): the client
  requests a server nonce, returns it signed (Scheme A envelope), and receives
  an opaque short-lived HMAC token (self-issued — no JWT/JWKS service) binding
  ``{hotkey, scope, expiry}``. Every bearer use re-checks ``is_registered``
  against the cached registry, so DEREGISTRATION REVOKES within the TTL.

Rollout: ``mode: log`` verifies and LOGS every refusal without refusing (so a
soak observes zero false refusals first); ``mode: enforce`` refuses. ``off``
disables the layer entirely (dev/report). Production preflight requires
``enforce`` on mainnet (`production_hotkey_auth_problems`).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from vidaio.services.artifact_auth import (
    BoundedReplayCache,
    bittensor_hotkey_verify,
)

log = logging.getLogger("hotkey-auth")

HOTKEY_AUTH_VERSION = 1
REQUEST_DOMAIN = b"vidaio-hotkey-auth-request-v1:"
CHALLENGE_DOMAIN = b"vidaio-hotkey-auth-challenge-v1:"

HEADER_HOTKEY = "x-vidaio-hotkey"
HEADER_TIMESTAMP = "x-vidaio-timestamp"
HEADER_NONCE = "x-vidaio-nonce"
HEADER_SIGNATURE = "x-vidaio-signature"

_NONCE = re.compile(r"[0-9a-f]{32}")
_SS58 = re.compile(r"[1-9A-HJ-NP-Za-km-z]{40,64}")


class HotkeyAuthError(Exception):
    """Base class for a refused hotkey-auth fact; carries the HTTP status."""

    status_code = 401
    code = "hotkey_auth_failed"


class HotkeyAuthMissing(HotkeyAuthError):
    code = "hotkey_auth_missing"


class HotkeyAuthInvalid(HotkeyAuthError):
    code = "hotkey_auth_invalid"


class HotkeyReplay(HotkeyAuthError):
    code = "hotkey_auth_replay"


class HotkeyTokenInvalid(HotkeyAuthError):
    code = "hotkey_token_invalid"


class HotkeyNotRegistered(HotkeyAuthError):
    status_code = 403
    code = "hotkey_not_registered"


class HotkeyNoValidatorPermit(HotkeyAuthError):
    status_code = 403
    code = "hotkey_no_validator_permit"


class HotkeyBelowStakeFloor(HotkeyAuthError):
    status_code = 403
    code = "hotkey_below_stake_floor"


class HotkeyRegistryUnavailable(HotkeyAuthError):
    status_code = 503
    code = "hotkey_registry_unavailable"


class HotkeyAuthConfig(BaseModel):
    """Schema for the shared ``hotkey_auth:`` config section."""

    model_config = ConfigDict(extra="forbid")

    #: ``off`` disables the layer (dev/report); ``log`` verifies + logs refusals
    #: without refusing (the soak posture); ``enforce`` refuses. Production
    #: preflight requires ``enforce`` outside the explicit testnet overlay.
    mode: Literal["off", "log", "enforce"] = "log"
    #: Cached-registry refresh interval. Deregistration revokes within this TTL.
    registry_ttl_seconds: float = Field(default=45.0, gt=0)
    #: How long a stale snapshot may keep serving when refresh FAILS before the
    #: registry fails closed (503). Bounds the availability/security trade.
    registry_max_stale_seconds: float = Field(default=300.0, gt=0)
    #: Scheme-A signed-request timestamp window (same 120 s as artifact-auth).
    replay_window_seconds: float = Field(default=120.0, gt=0)
    #: Scheme-B session-token lifetime.
    token_ttl_seconds: float = Field(default=3600.0, gt=0)
    #: Bounded replay-cache sizing (never evicts a live nonce; fails closed).
    max_replay_entries: int = Field(default=65536, ge=1)
    max_replay_entries_per_hotkey: int = Field(default=1024, ge=1)
    #: Minimum alpha stake to enroll in competitions: registered + signed +
    #: staked, or no sandbox build. None until the exact
    #: floor is set at mainnet-config sign-off; preflight then requires it.
    min_enroll_alpha_stake: float | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class _RegistryEntry:
    is_validator: bool
    alpha_stake: float


class RegisteredHotkeyRegistry:
    """One cached, fail-closed view of the subnet's registered hotkeys.

    ``chain`` is any adapter exposing ``refresh()`` and ``neurons()`` (the same
    seams every service already holds). The snapshot refreshes lazily, at most
    once per ``ttl_seconds`` — NEVER once per request. A failing refresh keeps
    serving the stale snapshot up to ``max_stale_seconds``, then every lookup
    raises ``HotkeyRegistryUnavailable`` (503) — an unavailable registry never
    silently allows.
    """

    def __init__(
        self,
        chain: object,
        *,
        ttl_seconds: float = 45.0,
        max_stale_seconds: float = 300.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0 or max_stale_seconds < ttl_seconds:
            raise ValueError(
                "registry ttl must be positive and max_stale must be >= ttl"
            )
        self._chain = chain
        self._ttl = ttl_seconds
        self._max_stale = max_stale_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshot: dict[str, _RegistryEntry] | None = None
        self._refreshed_at: float = 0.0

    def _entries(self) -> dict[str, _RegistryEntry]:
        now = self._clock()
        with self._lock:
            fresh = self._snapshot is not None and now - self._refreshed_at < self._ttl
            if not fresh:
                try:
                    self._chain.refresh()  # type: ignore[attr-defined]
                    neurons = list(self._chain.neurons())  # type: ignore[attr-defined]
                    self._snapshot = {
                        str(n.hotkey): _RegistryEntry(
                            is_validator=bool(getattr(n, "is_validator", False)),
                            alpha_stake=float(getattr(n, "alpha_stake", 0.0)),
                        )
                        for n in neurons
                    }
                    self._refreshed_at = now
                except Exception as exc:  # noqa: BLE001 - stale-serve then fail closed
                    if (
                        self._snapshot is None
                        or now - self._refreshed_at >= self._max_stale
                    ):
                        raise HotkeyRegistryUnavailable(
                            f"hotkey registry refresh failed and no snapshot newer "
                            f"than {self._max_stale}s exists: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    log.warning(
                        "hotkey registry refresh failed; serving stale snapshot",
                        extra={
                            "fields": {
                                "age_seconds": round(now - self._refreshed_at, 1),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        },
                    )
            assert self._snapshot is not None
            return self._snapshot

    def is_registered(self, hotkey: str) -> bool:
        return hotkey in self._entries()

    def has_validator_permit(self, hotkey: str) -> bool:
        entry = self._entries().get(hotkey)
        return entry is not None and entry.is_validator

    def alpha_stake(self, hotkey: str) -> float | None:
        entry = self._entries().get(hotkey)
        return entry.alpha_stake if entry is not None else None


def _canonical_request_digest(
    *, method: str, path: str, body: bytes, timestamp: int, nonce: str, hotkey: str
) -> bytes:
    """Domain-separated canonical bytes both signer and verifier compute."""
    body_sha = hashlib.sha256(body or b"").hexdigest()
    payload = (
        f'{{"body_sha256":"{body_sha}","hotkey":"{hotkey}","method":"{method.upper()}",'
        f'"nonce":"{nonce}","path":"{path}","timestamp":{timestamp},'
        f'"version":{HOTKEY_AUTH_VERSION}}}'
    ).encode("ascii")
    return REQUEST_DOMAIN + payload


def sign_request_headers(
    signer: object, *, method: str, path: str, body: bytes = b"", now: float | None = None
) -> dict[str, str]:
    """Client half of Scheme A: the four signed headers for one request.

    ``signer`` exposes ``hotkey`` (property or attr) and ``sign(bytes) -> str``
    (hex) — the same shape as `artifact_auth.CallableHotkeySigner`.
    """
    timestamp = int(now if now is not None else time.time())
    nonce = secrets.token_hex(16)
    hotkey = str(getattr(signer, "hotkey"))
    digest = _canonical_request_digest(
        method=method, path=path, body=body, timestamp=timestamp, nonce=nonce, hotkey=hotkey
    )
    signature = signer.sign(digest)  # type: ignore[attr-defined]
    return {
        HEADER_HOTKEY: hotkey,
        HEADER_TIMESTAMP: str(timestamp),
        HEADER_NONCE: nonce,
        HEADER_SIGNATURE: signature,
    }


@dataclass(frozen=True)
class AuthenticatedHotkey:
    """A verified caller: the registered hotkey and how it authenticated."""

    hotkey: str
    scheme: Literal["signed", "token"]
    is_validator: bool
    alpha_stake: float


class HotkeyAuthGuard:
    """Registry + Scheme A + Scheme B + rollout mode, bundled per service.

    ``require(...)`` is the one entry point: it authenticates the request
    (signed headers or bearer token), checks registration (and permit / stake
    floor where asked), and raises the typed ``HotkeyAuthError`` on refusal.
    In ``log`` mode refusals are logged (with the would-be status) and a
    sentinel ``None`` is returned instead of raising, so the soak observes the
    exact refusals enforcement would produce without breaking anything.
    """

    def __init__(
        self,
        registry: RegisteredHotkeyRegistry,
        config: HotkeyAuthConfig,
        *,
        verify_fn: Callable[[str, bytes, str], bool] = bittensor_hotkey_verify,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._registry = registry
        self._config = config
        self._verify = verify_fn
        self._clock = clock
        self._replay = BoundedReplayCache(
            config.max_replay_entries,
            max_entries_per_validator=config.max_replay_entries_per_hotkey,
        )
        #: Per-process token secret: self-issued, restart re-auths (deliberate).
        self._token_secret = secrets.token_bytes(32)
        self._challenges: dict[str, tuple[str, float]] = {}
        self._challenge_lock = threading.Lock()

    @property
    def mode(self) -> str:
        return self._config.mode

    # -- Scheme A -------------------------------------------------------------

    def _verify_signed(
        self, headers: Mapping[str, str], *, method: str, path: str, body: bytes
    ) -> str:
        lowered = {k.lower(): v for k, v in headers.items()}
        missing = [
            name
            for name in (HEADER_HOTKEY, HEADER_TIMESTAMP, HEADER_NONCE, HEADER_SIGNATURE)
            if not lowered.get(name)
        ]
        if missing:
            raise HotkeyAuthMissing(f"missing signed-request header(s): {missing}")
        hotkey = lowered[HEADER_HOTKEY].strip()
        if not _SS58.fullmatch(hotkey):
            raise HotkeyAuthInvalid("hotkey header is not a plausible ss58 address")
        nonce = lowered[HEADER_NONCE].strip()
        if not _NONCE.fullmatch(nonce):
            raise HotkeyAuthInvalid("nonce must be 128-bit lowercase hex")
        try:
            timestamp = int(lowered[HEADER_TIMESTAMP].strip())
        except ValueError as exc:
            raise HotkeyAuthInvalid("timestamp header is not an integer") from exc
        now = self._clock()
        window = self._config.replay_window_seconds
        if abs(now - timestamp) > window:
            raise HotkeyAuthInvalid(
                f"signed request timestamp outside the ±{int(window)}s window"
            )
        digest = _canonical_request_digest(
            method=method,
            path=path,
            body=body,
            timestamp=timestamp,
            nonce=nonce,
            hotkey=hotkey,
        )
        if not self._verify(hotkey, digest, lowered[HEADER_SIGNATURE].strip()):
            raise HotkeyAuthInvalid("request signature does not verify")
        try:
            self._replay.claim(hotkey, nonce, expires_at=timestamp + window, now=now)
        except Exception as exc:
            raise HotkeyReplay(str(exc)) from exc
        return hotkey

    # -- Scheme B -------------------------------------------------------------

    def mint_challenge(self) -> str:
        """A server nonce the client must sign to redeem a session token."""
        nonce = secrets.token_hex(16)
        now = self._clock()
        with self._challenge_lock:
            # bounded: drop expired, refuse growth beyond the replay-cache bound
            expired = [
                n for n, (_, exp) in self._challenges.items() if exp < now
            ]
            for n in expired:
                del self._challenges[n]
            if len(self._challenges) >= self._config.max_replay_entries:
                raise HotkeyRegistryUnavailable("challenge store is full")
            self._challenges[nonce] = ("", now + self._config.replay_window_seconds)
        return nonce

    def redeem_challenge(
        self, headers: Mapping[str, str], *, challenge: str, path: str
    ) -> str:
        """Verify the signed challenge redemption; return a session token."""
        with self._challenge_lock:
            entry = self._challenges.pop(challenge, None)
        if entry is None or entry[1] < self._clock():
            raise HotkeyAuthInvalid("unknown or expired auth challenge")
        hotkey = self._verify_signed(
            headers, method="POST", path=path, body=challenge.encode("ascii")
        )
        self._require_registered(hotkey)
        return self._mint_token(hotkey)

    def _mint_token(self, hotkey: str) -> str:
        expires = int(self._clock() + self._config.token_ttl_seconds)
        payload = f"vk{HOTKEY_AUTH_VERSION}.{hotkey}.{expires}"
        mac = hmac.new(
            self._token_secret, payload.encode("ascii"), hashlib.sha256
        ).hexdigest()
        return f"{payload}.{mac}"

    def _verify_token(self, token: str) -> str:
        parts = token.split(".")
        if len(parts) != 4 or parts[0] != f"vk{HOTKEY_AUTH_VERSION}":
            raise HotkeyTokenInvalid("malformed session token")
        version, hotkey, expires_raw, mac = parts
        payload = f"{version}.{hotkey}.{expires_raw}"
        expected = hmac.new(
            self._token_secret, payload.encode("ascii"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(mac, expected):
            raise HotkeyTokenInvalid("session token signature does not verify")
        try:
            expires = int(expires_raw)
        except ValueError as exc:
            raise HotkeyTokenInvalid("session token expiry is not an integer") from exc
        if self._clock() >= expires:
            raise HotkeyTokenInvalid("session token has expired")
        return hotkey

    # -- registration / permit / stake ---------------------------------------

    def _require_registered(self, hotkey: str) -> AuthenticatedHotkey:
        if not self._registry.is_registered(hotkey):
            raise HotkeyNotRegistered(
                f"hotkey {hotkey} is not registered on this subnet"
            )
        return AuthenticatedHotkey(
            hotkey=hotkey,
            scheme="signed",
            is_validator=self._registry.has_validator_permit(hotkey),
            alpha_stake=self._registry.alpha_stake(hotkey) or 0.0,
        )

    # -- the one entry point ---------------------------------------------------

    def require(
        self,
        headers: Mapping[str, str],
        *,
        method: str,
        path: str,
        body: bytes = b"",
        require_validator_permit: bool = False,
        min_alpha_stake: float | None = None,
    ) -> AuthenticatedHotkey | None:
        """Authenticate + authorize one request; the mode decides refusal.

        Returns the verified caller, or ``None`` when ``mode`` is ``off`` (layer
        disabled) or ``log`` (refusal logged, request allowed through so a soak
        can observe enforcement before it bites). ``enforce`` raises the typed
        ``HotkeyAuthError`` — FastAPI wrappers map it to its ``status_code``.
        """
        if self._config.mode == "off":
            return None
        try:
            lowered = {k.lower(): v for k, v in headers.items()}
            bearer = lowered.get("authorization", "")
            if bearer.lower().startswith(f"bearer vk{HOTKEY_AUTH_VERSION}."):
                hotkey = self._verify_token(bearer[7:].strip())
                verified = self._require_registered(hotkey)
                verified = AuthenticatedHotkey(
                    hotkey=verified.hotkey,
                    scheme="token",
                    is_validator=verified.is_validator,
                    alpha_stake=verified.alpha_stake,
                )
            else:
                hotkey = self._verify_signed(
                    lowered, method=method, path=path, body=body
                )
                verified = self._require_registered(hotkey)
            if require_validator_permit and not verified.is_validator:
                raise HotkeyNoValidatorPermit(
                    f"hotkey {verified.hotkey} is registered but carries no "
                    "validator permit"
                )
            if (
                min_alpha_stake is not None
                and verified.alpha_stake < min_alpha_stake
            ):
                raise HotkeyBelowStakeFloor(
                    f"hotkey {verified.hotkey} has alpha stake "
                    f"{verified.alpha_stake} below the required floor "
                    f"{min_alpha_stake}"
                )
            return verified
        except HotkeyAuthError as exc:
            if self._config.mode == "log":
                log.warning(
                    "hotkey-auth would refuse this request (log-only mode)",
                    extra={
                        "fields": {
                            "code": exc.code,
                            "status": exc.status_code,
                            "method": method,
                            "path": path,
                            "error": str(exc),
                        }
                    },
                )
                return None
            raise


__all__ = [
    "AuthenticatedHotkey",
    "HotkeyAuthConfig",
    "HotkeyAuthError",
    "HotkeyAuthGuard",
    "HotkeyAuthInvalid",
    "HotkeyAuthMissing",
    "HotkeyBelowStakeFloor",
    "HotkeyNoValidatorPermit",
    "HotkeyNotRegistered",
    "HotkeyRegistryUnavailable",
    "HotkeyReplay",
    "HotkeyTokenInvalid",
    "RegisteredHotkeyRegistry",
    "sign_request_headers",
    "HEADER_HOTKEY",
    "HEADER_TIMESTAMP",
    "HEADER_NONCE",
    "HEADER_SIGNATURE",
]
