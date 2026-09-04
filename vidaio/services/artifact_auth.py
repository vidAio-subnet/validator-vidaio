"""Hotkey identity binding for miner artifact protocol v2.

V2 authenticates both directions without assuming a shared bearer token:

* a currently registered validator signs canonical request facts before the
  miner reads a body byte;
* the intended, chain-attributed miner hotkey signs canonical response facts;
* a fresh timestamp plus a per-request 128-bit nonce is claimed in a bounded
  miner-side replay cache only after registration and signature verification.

The cryptographic and registration operations are narrow injected seams. The
production verifier lazily uses ``bittensor.Keypair``; report/tests inject a
deterministic signer/verifier and a fixed or chainsim-backed registry.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from vidaio.services.protocol import (
    MINER_ARTIFACT_AUTH_VERSION,
    MINER_ARTIFACT_VERSION_HEADER,
    MINER_HOTKEY_HEADER,
    MINER_INPUT_SIZE_HEADER,
    MINER_OUTPUT_SIZE_HEADER,
    MINER_REQUEST_NONCE_HEADER,
    MINER_REQUEST_SIGNATURE_HEADER,
    MINER_REQUEST_TIMESTAMP_HEADER,
    MINER_RESPONSE_SIGNATURE_HEADER,
    MINER_VALIDATOR_HOTKEY_HEADER,
    MinerArtifactTaskRequest,
)

REQUEST_SIGNATURE_DOMAIN = b"vidaio:miner-artifact:request:v2\x00"
RESPONSE_SIGNATURE_DOMAIN = b"vidaio:miner-artifact:response:v2\x00"

_NONCE = re.compile(r"^[0-9a-f]{32}$")
# Bittensor hotkeys sign with sr25519/ed25519, whose signature is exactly 64
# bytes.  Accepting merely "some even-length hex" lets truncated or padded
# values reach different crypto backends with backend-specific behaviour.
_HEX_SIGNATURE = re.compile(r"^[0-9a-f]{128}$")


class ArtifactAuthError(ValueError):
    """Base class for a refused artifact-v2 authentication fact."""


class ArtifactAuthMissing(ArtifactAuthError):
    """A required v2 authentication header or configured seam is absent."""


class ArtifactAuthInvalid(ArtifactAuthError):
    """A signature or signed field is malformed, inconsistent, or invalid."""


class ArtifactAuthExpired(ArtifactAuthError):
    """The signed request timestamp is outside the server freshness window."""


class ArtifactAuthColdStart(ArtifactAuthError):
    """The request could predate this replay cache and must be retried fresh."""


class ArtifactWrongMiner(ArtifactAuthError):
    """A request or response is bound to a different miner hotkey."""


class ArtifactUnregisteredValidator(ArtifactAuthError):
    """The request signer is not a current validator in the subnet snapshot."""


class ArtifactReplay(ArtifactAuthError):
    """The validator hotkey already used this nonce inside its valid window."""


class ArtifactReplayCacheFull(ArtifactAuthError):
    """The bounded replay cache has no safe capacity for another live nonce."""


@runtime_checkable
class ArtifactHotkeySigner(Protocol):
    """Signs bytes under one explicitly named hotkey identity."""

    @property
    def hotkey(self) -> str: ...

    def sign(self, payload: bytes) -> str: ...


@dataclass(frozen=True)
class CallableHotkeySigner:
    """Identity-bound adapter around an existing wallet-backed ``sign`` seam."""

    hotkey: str
    sign_fn: Callable[[bytes], str]

    def __post_init__(self) -> None:
        _validate_hotkey(self.hotkey, field="signer hotkey")

    def sign(self, payload: bytes) -> str:
        signature = self.sign_fn(payload)
        _validate_signature(signature)
        return signature


@runtime_checkable
class CurrentValidatorRegistry(Protocol):
    """Answers from current/fresh subnet state, never from caller claims."""

    def is_current_validator(self, hotkey: str) -> bool: ...


class FrozenValidatorRegistry:
    """Deterministic report/test double for current validator membership."""

    def __init__(self, hotkeys: set[str] | frozenset[str] | tuple[str, ...]) -> None:
        self._hotkeys = frozenset(hotkeys)

    def is_current_validator(self, hotkey: str) -> bool:
        return hotkey in self._hotkeys


class ChainValidatorRegistry:
    """Cached metagraph-backed validator-permit check (P2 refactor).

    ``refresh`` is fail-closed by the chain adapter. Where it exposes freshness,
    a snapshot older than ``max_snapshot_age_seconds`` is rejected as unknown.
    Exactly one matching neuron must exist and carry validator permit.

    The permit view refreshes AT MOST once per ``ttl_seconds`` (default 45 s) —
    never once per request. The previous per-request ``chain.refresh()`` was an
    RPC-flood vector on the miner ingress (the design notes §3.2): request
    rate translated 1:1 into chain RPC load. Deregistration/permit-loss still
    revokes within the TTL, matching the shared ``RegisteredHotkeyRegistry``
    semantics. A failed refresh serves the stale view up to
    ``max_snapshot_age_seconds``, then fails closed (unknown ⇒ refuse).
    """

    def __init__(
        self,
        chain: object,
        *,
        max_snapshot_age_seconds: float = 300.0,
        ttl_seconds: float = 45.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_snapshot_age_seconds <= 0:
            raise ValueError("validator-registry snapshot age must be positive")
        if not 0 < ttl_seconds <= max_snapshot_age_seconds:
            raise ValueError(
                "validator-registry ttl must be positive and no larger than "
                "max_snapshot_age_seconds"
            )
        self._chain = chain
        self._max_age = max_snapshot_age_seconds
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._permits: frozenset[str] | None = None
        self._counts: dict[str, int] = {}
        self._refreshed_at: float = 0.0

    def _view(self) -> tuple[frozenset[str], dict[str, int]] | None:
        now = self._clock()
        with self._lock:
            if self._permits is not None and now - self._refreshed_at < self._ttl:
                return self._permits, self._counts
            try:
                refresh = getattr(self._chain, "refresh")
                neurons = getattr(self._chain, "neurons")
                refresh()
                freshness = getattr(self._chain, "has_fresh_snapshot", None)
                if callable(freshness) and not freshness(now, self._max_age):
                    raise RuntimeError("chain snapshot is stale")
                counts: dict[str, int] = {}
                permits: set[str] = set()
                for n in neurons():
                    hk = str(n.hotkey)
                    counts[hk] = counts.get(hk, 0) + 1
                    if bool(getattr(n, "is_validator", False)):
                        permits.add(hk)
                self._permits = frozenset(permits)
                self._counts = counts
                self._refreshed_at = now
            except Exception:
                if self._permits is None or now - self._refreshed_at >= self._max_age:
                    return None  # fail closed: no acceptable view exists
            return self._permits, self._counts

    def is_current_validator(self, hotkey: str) -> bool:
        view = self._view()
        if view is None:
            return False
        permits, counts = view
        return counts.get(hotkey, 0) == 1 and hotkey in permits


def bittensor_hotkey_verify(hotkey: str, payload: bytes, signature: str) -> bool:
    """Verify an sr25519/ed25519 hex signature against an ss58 hotkey."""
    try:
        _validate_signature(signature)
        from bittensor import Keypair  # lazy release dependency

        return bool(
            Keypair(ss58_address=hotkey).verify(payload, bytes.fromhex(signature))
        )
    except Exception:
        return False


@dataclass(frozen=True)
class ArtifactRequestClaims:
    version: str
    validator_hotkey: str
    miner_hotkey: str
    timestamp: int
    nonce: str
    input_size: int


class ValidatorArtifactRequestReceipt(BaseModel):
    """Persistable proof of the exact validator-signed artifact-v2 request.

    Unlike :class:`MinerArtifactReceipt`, this exists before a miner responds. It is
    therefore the chronology/input proof used by an availability fold when the miner
    times out or fails transport/protocol validation. It proves what was offered and
    to which intended hotkey; it does not by itself prove the negative fact that no
    response existed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    validator_hotkey: str
    miner_hotkey: str
    timestamp: int = Field(ge=0)
    nonce: str
    input_size: int = Field(ge=1)
    metadata: MinerArtifactTaskRequest
    request_signature: str

    def claims(self) -> ArtifactRequestClaims:
        return ArtifactRequestClaims(
            version=self.version,
            validator_hotkey=self.validator_hotkey,
            miner_hotkey=self.miner_hotkey,
            timestamp=self.timestamp,
            nonce=self.nonce,
            input_size=self.input_size,
        )

    def signed_bytes(self) -> bytes:
        return canonical_request_bytes(self.claims(), self.metadata)


def verify_validator_artifact_request_receipt(
    receipt: ValidatorArtifactRequestReceipt,
    *,
    verify_fn: Callable[[str, bytes, str], bool] = bittensor_hotkey_verify,
) -> bool:
    """Cryptographically verify a persisted validator request receipt."""
    try:
        _validate_hotkey(receipt.validator_hotkey, field="validator hotkey")
        _validate_hotkey(receipt.miner_hotkey, field="intended miner hotkey")
        _validate_signature(receipt.request_signature)
        return bool(
            verify_fn(
                receipt.validator_hotkey,
                receipt.signed_bytes(),
                receipt.request_signature,
            )
        )
    except Exception:
        return False


class MinerArtifactReceipt(BaseModel):
    """Persistable proof of the miner-signed artifact-v2 response.

    It contains every fact needed to reconstruct ``canonical_response_bytes``.
    In particular, ``metadata.commitment_anchor`` is inside the request digest,
    closing the chronology proof: the miner signed its output only after receiving
    a receipt that an auditor independently verifies at finalized archive state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    validator_hotkey: str
    miner_hotkey: str
    timestamp: int = Field(ge=0)
    nonce: str
    input_size: int = Field(ge=1)
    metadata: MinerArtifactTaskRequest
    #: The validator's original request signature. New v2 clients persist it so
    #: a later protocol/digest failure can still be converted into a request-bound
    #: availability observation. Optional only for pre-availability legacy fixtures.
    request_signature: str = ""
    output_digest: str
    output_size: int = Field(ge=1)
    processing_seconds: str = ""
    response_signature: str

    def claims(self) -> ArtifactRequestClaims:
        return ArtifactRequestClaims(
            version=self.version,
            validator_hotkey=self.validator_hotkey,
            miner_hotkey=self.miner_hotkey,
            timestamp=self.timestamp,
            nonce=self.nonce,
            input_size=self.input_size,
        )

    def signed_bytes(self) -> bytes:
        return canonical_response_bytes(
            self.claims(),
            self.metadata,
            output_digest=self.output_digest,
            output_size=self.output_size,
            processing_seconds=self.processing_seconds,
        )

    def request_receipt(self) -> ValidatorArtifactRequestReceipt | None:
        if not self.request_signature:
            return None
        return ValidatorArtifactRequestReceipt(
            version=self.version,
            validator_hotkey=self.validator_hotkey,
            miner_hotkey=self.miner_hotkey,
            timestamp=self.timestamp,
            nonce=self.nonce,
            input_size=self.input_size,
            metadata=self.metadata,
            request_signature=self.request_signature,
        )


def verify_miner_artifact_receipt(
    receipt: MinerArtifactReceipt,
    *,
    verify_fn: Callable[[str, bytes, str], bool] = bittensor_hotkey_verify,
) -> bool:
    """Cryptographically verify a persisted miner response receipt."""
    _validate_hotkey(receipt.miner_hotkey, field="miner hotkey")
    _validate_signature(receipt.response_signature)
    try:
        return bool(
            verify_fn(
                receipt.miner_hotkey,
                receipt.signed_bytes(),
                receipt.response_signature,
            )
        )
    except Exception:
        return False


def _validate_hotkey(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ArtifactAuthInvalid(f"{field} must be 1..128 characters")
    if any(ord(ch) < 0x21 or ord(ch) > 0x7E for ch in value):
        raise ArtifactAuthInvalid(f"{field} must contain printable non-space ASCII")
    return value


def _validate_signature(value: str) -> str:
    if not isinstance(value, str) or not _HEX_SIGNATURE.fullmatch(value):
        raise ArtifactAuthInvalid(
            "hotkey signature must be exactly 64-byte lowercase hex"
        )
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ArtifactAuthInvalid(
            f"artifact signature payload is not canonical: {exc}"
        ) from exc


def canonical_request_bytes(
    claims: ArtifactRequestClaims, metadata: MinerArtifactTaskRequest
) -> bytes:
    """Domain-separated canonical request bytes shared by signer and verifier."""
    if claims.version != MINER_ARTIFACT_AUTH_VERSION:
        raise ArtifactAuthInvalid(
            f"unsupported signed artifact version {claims.version!r}"
        )
    _validate_hotkey(claims.validator_hotkey, field="validator hotkey")
    _validate_hotkey(claims.miner_hotkey, field="intended miner hotkey")
    if not isinstance(claims.nonce, str) or not _NONCE.fullmatch(claims.nonce):
        raise ArtifactAuthInvalid("request nonce must be 128-bit lowercase hex")
    if claims.timestamp < 0:
        raise ArtifactAuthInvalid("request timestamp must be non-negative")
    if claims.input_size < 1:
        raise ArtifactAuthInvalid("signed input size must be positive")
    if not math.isfinite(metadata.deadline_seconds) or metadata.deadline_seconds <= 0:
        raise ArtifactAuthInvalid("task deadline must be finite and positive")
    payload = {
        "commitment_anchor": (
            None
            if metadata.commitment_anchor is None
            else metadata.commitment_anchor.model_dump(mode="json")
        ),
        "deadline_seconds": metadata.deadline_seconds,
        "input_digest": metadata.input_digest,
        "input_size": claims.input_size,
        "intended_miner_hotkey": claims.miner_hotkey,
        "nonce": claims.nonce,
        "params": metadata.params,
        "signer_hotkey": claims.validator_hotkey,
        "task_id": metadata.task_id,
        "timestamp": claims.timestamp,
        "track": metadata.track,
        "version": claims.version,
    }
    return REQUEST_SIGNATURE_DOMAIN + _canonical_json(payload)


def canonical_response_bytes(
    claims: ArtifactRequestClaims,
    metadata: MinerArtifactTaskRequest,
    *,
    output_digest: str,
    output_size: int,
    processing_seconds: str,
) -> bytes:
    """Domain-separated response bytes binding the complete signed request."""
    request_digest = hashlib.sha256(
        canonical_request_bytes(claims, metadata)
    ).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", output_digest):
        raise ArtifactAuthInvalid("response output digest must be lowercase sha256 hex")
    if output_size < 1:
        raise ArtifactAuthInvalid("signed output size must be positive")
    if len(processing_seconds) > 64:
        raise ArtifactAuthInvalid("processing-seconds header is too long")
    if processing_seconds:
        try:
            parsed = float(processing_seconds)
        except ValueError as exc:
            raise ArtifactAuthInvalid("processing seconds is not numeric") from exc
        if not math.isfinite(parsed) or parsed < 0:
            raise ArtifactAuthInvalid(
                "processing seconds must be finite and non-negative"
            )
    payload = {
        "input_digest": metadata.input_digest,
        "input_size": claims.input_size,
        "miner_hotkey": claims.miner_hotkey,
        "nonce": claims.nonce,
        "output_digest": output_digest,
        "output_size": output_size,
        "processing_seconds": processing_seconds,
        "request_digest": request_digest,
        "signer_hotkey": claims.validator_hotkey,
        "task_id": metadata.task_id,
        "track": metadata.track,
        "version": claims.version,
    }
    return RESPONSE_SIGNATURE_DOMAIN + _canonical_json(payload)


class BoundedReplayCache:
    """Thread-safe cache that never evicts a still-replayable nonce.

    When all bounded slots hold live nonces, new work fails closed until expiry;
    evicting an unexpired entry would make that signed request replayable again.
    A second per-validator limit prevents one registered but malicious validator
    from monopolizing the global cache and denying every honest signer.
    """

    def __init__(self, max_entries: int, *, max_entries_per_validator: int) -> None:
        if max_entries < 1:
            raise ValueError("replay cache max_entries must be >= 1")
        if not 1 <= max_entries_per_validator <= max_entries:
            raise ValueError(
                "replay cache max_entries_per_validator must be between 1 and "
                "max_entries"
            )
        self._max_entries = max_entries
        self._max_entries_per_validator = max_entries_per_validator
        self._entries: dict[tuple[str, str], float] = {}
        self._validator_counts: dict[str, int] = {}
        self._expiry_heap: list[tuple[float, tuple[str, str]]] = []
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def claim(self, hotkey: str, nonce: str, *, expires_at: float, now: float) -> None:
        key = (hotkey, nonce)
        with self._lock:
            while self._expiry_heap and self._expiry_heap[0][0] < now:
                expiry, expired_key = heapq.heappop(self._expiry_heap)
                if self._entries.get(expired_key) == expiry:
                    del self._entries[expired_key]
                    expired_hotkey = expired_key[0]
                    remaining = self._validator_counts[expired_hotkey] - 1
                    if remaining:
                        self._validator_counts[expired_hotkey] = remaining
                    else:
                        del self._validator_counts[expired_hotkey]
            if key in self._entries:
                raise ArtifactReplay("signed request nonce was already used")
            if self._validator_counts.get(hotkey, 0) >= self._max_entries_per_validator:
                raise ArtifactReplayCacheFull(
                    "artifact replay cache validator quota is full of unexpired nonces"
                )
            if len(self._entries) >= self._max_entries:
                raise ArtifactReplayCacheFull(
                    "artifact replay cache is full of unexpired nonces"
                )
            self._entries[key] = expires_at
            self._validator_counts[hotkey] = self._validator_counts.get(hotkey, 0) + 1
            heapq.heappush(self._expiry_heap, (expires_at, key))


def _required(headers: Mapping[str, str], name: str) -> str:
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower())
    if not isinstance(value, str) or not value:
        raise ArtifactAuthMissing(f"missing {name} header")
    return value


class ArtifactClientAuth:
    """Validator/gateway side request signer and miner-response verifier."""

    def __init__(
        self,
        signer: ArtifactHotkeySigner,
        *,
        verify_fn: Callable[[str, bytes, str], bool] = bittensor_hotkey_verify,
        clock: Callable[[], float] = time.time,
        nonce_factory: Callable[[], str] = lambda: secrets.token_hex(16),
    ) -> None:
        _validate_hotkey(signer.hotkey, field="validator signer hotkey")
        self.signer = signer
        self._verify_fn = verify_fn
        self._clock = clock
        self._nonce_factory = nonce_factory

    def sign_request(
        self,
        metadata: MinerArtifactTaskRequest,
        *,
        input_size: int,
        intended_miner_hotkey: str,
    ) -> tuple[ArtifactRequestClaims, dict[str, str]]:
        claims = ArtifactRequestClaims(
            version=MINER_ARTIFACT_AUTH_VERSION,
            validator_hotkey=self.signer.hotkey,
            miner_hotkey=_validate_hotkey(
                intended_miner_hotkey, field="intended miner hotkey"
            ),
            timestamp=int(self._clock()),
            nonce=self._nonce_factory(),
            input_size=input_size,
        )
        payload = canonical_request_bytes(claims, metadata)
        signature = self.signer.sign(payload)
        _validate_signature(signature)
        return claims, {
            MINER_ARTIFACT_VERSION_HEADER: claims.version,
            MINER_VALIDATOR_HOTKEY_HEADER: claims.validator_hotkey,
            MINER_HOTKEY_HEADER: claims.miner_hotkey,
            MINER_REQUEST_TIMESTAMP_HEADER: str(claims.timestamp),
            MINER_REQUEST_NONCE_HEADER: claims.nonce,
            MINER_INPUT_SIZE_HEADER: str(claims.input_size),
            MINER_REQUEST_SIGNATURE_HEADER: signature,
        }

    def verify_response(
        self,
        headers: Mapping[str, str],
        claims: ArtifactRequestClaims,
        metadata: MinerArtifactTaskRequest,
        *,
        output_digest: str,
        output_size: int,
        processing_seconds: str,
    ) -> None:
        version = _required(headers, MINER_ARTIFACT_VERSION_HEADER)
        if version != MINER_ARTIFACT_AUTH_VERSION:
            raise ArtifactAuthInvalid(
                f"artifact protocol downgrade: response version {version!r} is not "
                f"{MINER_ARTIFACT_AUTH_VERSION!r}"
            )
        miner_hotkey = _required(headers, MINER_HOTKEY_HEADER)
        if miner_hotkey != claims.miner_hotkey:
            raise ArtifactWrongMiner(
                f"response signer {miner_hotkey!r} is not intended miner "
                f"{claims.miner_hotkey!r}"
            )
        try:
            signed_size = int(_required(headers, MINER_OUTPUT_SIZE_HEADER))
        except ValueError as exc:
            raise ArtifactAuthInvalid("signed output size is not an integer") from exc
        if signed_size != output_size:
            raise ArtifactAuthInvalid(
                f"signed output size {signed_size} != received {output_size}"
            )
        signature = _required(headers, MINER_RESPONSE_SIGNATURE_HEADER)
        _validate_signature(signature)
        payload = canonical_response_bytes(
            claims,
            metadata,
            output_digest=output_digest,
            output_size=output_size,
            processing_seconds=processing_seconds,
        )
        try:
            verified = bool(self._verify_fn(miner_hotkey, payload, signature))
        except Exception as exc:
            raise ArtifactAuthInvalid(
                f"miner response signature verifier failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not verified:
            raise ArtifactAuthInvalid("miner response hotkey signature is invalid")


class ArtifactServerAuth:
    """Miner side validator verifier, replay guard, and response signer."""

    def __init__(
        self,
        signer: ArtifactHotkeySigner,
        validators: CurrentValidatorRegistry,
        *,
        request_max_age_seconds: float = 120.0,
        request_future_skew_seconds: float = 5.0,
        replay_cache_entries: int = 10_000,
        replay_cache_entries_per_validator: int | None = None,
        verify_fn: Callable[[str, bytes, str], bool] = bittensor_hotkey_verify,
        clock: Callable[[], float] = time.time,
        replay_cache: BoundedReplayCache | None = None,
    ) -> None:
        if request_max_age_seconds <= 0:
            raise ValueError("artifact request max age must be positive")
        if request_future_skew_seconds < 0:
            raise ValueError("artifact request future skew must be non-negative")
        _validate_hotkey(signer.hotkey, field="miner signer hotkey")
        self.signer = signer
        self.validators = validators
        self.request_max_age_seconds = request_max_age_seconds
        self.request_future_skew_seconds = request_future_skew_seconds
        self._verify_fn = verify_fn
        self._clock = clock
        # Restarting an in-memory nonce cache must not make a still-fresh signed
        # request replayable. Include the configured positive-skew allowance:
        # immediately before this process started, a peer could legitimately
        # have signed as far ahead as start+skew. The short startup blackout is
        # bounded by skew+one integer-second tick; callers retry with a new nonce.
        self.request_timestamp_floor = int(
            self._clock() + self.request_future_skew_seconds
        )
        per_validator = (
            min(256, replay_cache_entries)
            if replay_cache_entries_per_validator is None
            else replay_cache_entries_per_validator
        )
        self.replay_cache = replay_cache or BoundedReplayCache(
            replay_cache_entries,
            max_entries_per_validator=per_validator,
        )

    def verify_request(
        self,
        headers: Mapping[str, str],
        metadata: MinerArtifactTaskRequest,
        *,
        content_length: int,
    ) -> ArtifactRequestClaims:
        version = _required(headers, MINER_ARTIFACT_VERSION_HEADER)
        if version != MINER_ARTIFACT_AUTH_VERSION:
            raise ArtifactAuthInvalid(f"signed artifact version {version!r} is not v2")
        validator_hotkey = _required(headers, MINER_VALIDATOR_HOTKEY_HEADER)
        miner_hotkey = _required(headers, MINER_HOTKEY_HEADER)
        _validate_hotkey(validator_hotkey, field="validator hotkey")
        _validate_hotkey(miner_hotkey, field="intended miner hotkey")
        if miner_hotkey != self.signer.hotkey:
            raise ArtifactWrongMiner(
                f"request intended for miner {miner_hotkey!r}, not {self.signer.hotkey!r}"
            )
        raw_timestamp = _required(headers, MINER_REQUEST_TIMESTAMP_HEADER)
        raw_size = _required(headers, MINER_INPUT_SIZE_HEADER)
        try:
            timestamp = int(raw_timestamp)
            input_size = int(raw_size)
        except ValueError as exc:
            raise ArtifactAuthInvalid(
                "request timestamp/input size is not an integer"
            ) from exc
        if str(timestamp) != raw_timestamp or str(input_size) != raw_size:
            raise ArtifactAuthInvalid("request timestamp/input size is not canonical")
        if input_size != content_length:
            raise ArtifactAuthInvalid(
                f"signed input size {input_size} != Content-Length {content_length}"
            )
        nonce = _required(headers, MINER_REQUEST_NONCE_HEADER)
        claims = ArtifactRequestClaims(
            version=version,
            validator_hotkey=validator_hotkey,
            miner_hotkey=miner_hotkey,
            timestamp=timestamp,
            nonce=nonce,
            input_size=input_size,
        )
        payload = canonical_request_bytes(claims, metadata)
        signature = _required(headers, MINER_REQUEST_SIGNATURE_HEADER)
        _validate_signature(signature)
        try:
            verified = bool(self._verify_fn(validator_hotkey, payload, signature))
        except Exception as exc:
            raise ArtifactAuthInvalid(
                f"validator request signature verifier failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not verified:
            raise ArtifactAuthInvalid("validator request hotkey signature is invalid")

        # Only an authenticated hotkey may trigger a live metagraph refresh.
        # Signature verification is local and bounded; doing the registry lookup
        # first lets unauthenticated traffic amplify into chain RPC work.
        now = self._clock()
        if timestamp < now - self.request_max_age_seconds:
            raise ArtifactAuthExpired("signed artifact request has expired")
        if timestamp > now + self.request_future_skew_seconds:
            raise ArtifactAuthExpired(
                "signed artifact request timestamp is in the future"
            )
        if timestamp <= self.request_timestamp_floor:
            raise ArtifactAuthColdStart(
                "signed artifact request is not newer than this miner's replay-cache "
                "startup fence; retry with a fresh timestamp and nonce"
            )
        if not self.validators.is_current_validator(validator_hotkey):
            raise ArtifactUnregisteredValidator(
                f"request signer {validator_hotkey!r} is not a current validator"
            )
        self.replay_cache.claim(
            validator_hotkey,
            nonce,
            expires_at=timestamp + self.request_max_age_seconds,
            now=now,
        )
        return claims

    def response_headers(
        self,
        claims: ArtifactRequestClaims,
        metadata: MinerArtifactTaskRequest,
        *,
        output_digest: str,
        output_size: int,
        processing_seconds: str,
    ) -> dict[str, str]:
        if claims.miner_hotkey != self.signer.hotkey:
            raise ArtifactWrongMiner("cannot sign a response for another miner hotkey")
        payload = canonical_response_bytes(
            claims,
            metadata,
            output_digest=output_digest,
            output_size=output_size,
            processing_seconds=processing_seconds,
        )
        signature = self.signer.sign(payload)
        _validate_signature(signature)
        return {
            MINER_ARTIFACT_VERSION_HEADER: MINER_ARTIFACT_AUTH_VERSION,
            MINER_HOTKEY_HEADER: self.signer.hotkey,
            MINER_OUTPUT_SIZE_HEADER: str(output_size),
            MINER_RESPONSE_SIGNATURE_HEADER: signature,
        }
