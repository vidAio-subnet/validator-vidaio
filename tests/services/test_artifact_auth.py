"""Artifact-v2 hotkey identity, canonical binding, and replay tests."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from vidaio.services.artifact_auth import (
    ArtifactAuthColdStart,
    ArtifactAuthExpired,
    ArtifactAuthInvalid,
    ArtifactClientAuth,
    ArtifactReplay,
    ArtifactReplayCacheFull,
    ArtifactServerAuth,
    ArtifactUnregisteredValidator,
    ArtifactWrongMiner,
    ChainValidatorRegistry,
    CallableHotkeySigner,
    FrozenValidatorRegistry,
)
from vidaio.services.protocol import (
    MINER_ARTIFACT_VERSION,
    MINER_ARTIFACT_VERSION_HEADER,
    MINER_HOTKEY_HEADER,
    MINER_INPUT_SIZE_HEADER,
    MINER_OUTPUT_SIZE_HEADER,
    MINER_REQUEST_NONCE_HEADER,
    MINER_REQUEST_SIGNATURE_HEADER,
    MINER_REQUEST_TIMESTAMP_HEADER,
    MinerArtifactTaskRequest,
)

KEYS = {
    "validator-a": b"validator-a-secret",
    "validator-b": b"validator-b-secret",
    "miner-a": b"miner-a-secret",
    "miner-b": b"miner-b-secret",
}


def _signature(hotkey: str, payload: bytes) -> str:
    # 64-byte deterministic test double matching the wire shape of sr25519.
    return hashlib.sha512(KEYS[hotkey] + b"\x00" + payload).hexdigest()


def _verify(hotkey: str, payload: bytes, signature: str) -> bool:
    try:
        expected = _signature(hotkey, payload)
    except KeyError:
        return False
    return signature == expected


def _signer(hotkey: str) -> CallableHotkeySigner:
    return CallableHotkeySigner(hotkey, lambda payload: _signature(hotkey, payload))


def _metadata(**updates: object) -> MinerArtifactTaskRequest:
    values: dict[str, object] = {
        "task_id": "task-1",
        "track": "compression",
        "input_digest": hashlib.sha256(b"input").hexdigest(),
        "params": {"quality": 7, "codec": "h265"},
        "deadline_seconds": 30.0,
    }
    values.update(updates)
    return MinerArtifactTaskRequest.model_validate(values)


def _pair(
    *,
    now: list[float] | None = None,
    validator: str = "validator-a",
    miner: str = "miner-a",
    registered: tuple[str, ...] = ("validator-a",),
    nonce: str = "01" * 16,
    cache_entries: int = 10,
    cache_entries_per_validator: int | None = None,
) -> tuple[ArtifactClientAuth, ArtifactServerAuth]:
    current = now or [1_000.0]

    def clock() -> float:
        return current[0]

    server_clock_calls = 0

    def server_clock() -> float:
        nonlocal server_clock_calls
        server_clock_calls += 1
        # Most tests exercise steady state. Construct the server far enough in
        # the past that its cold-start floor is already behind `current`.
        return current[0] - 6 if server_clock_calls == 1 else current[0]

    client = ArtifactClientAuth(
        _signer(validator),
        verify_fn=_verify,
        clock=clock,
        nonce_factory=lambda: nonce,
    )
    server = ArtifactServerAuth(
        _signer(miner),
        FrozenValidatorRegistry(registered),
        verify_fn=_verify,
        clock=server_clock,
        request_max_age_seconds=120,
        request_future_skew_seconds=5,
        replay_cache_entries=cache_entries,
        replay_cache_entries_per_validator=cache_entries_per_validator,
    )
    return client, server


def _signed_request(
    client: ArtifactClientAuth,
    metadata: MinerArtifactTaskRequest,
    *,
    size: int = 5,
    miner: str = "miner-a",
):  # type: ignore[no-untyped-def]
    return client.sign_request(
        metadata,
        input_size=size,
        intended_miner_hotkey=miner,
    )


def test_signed_request_and_response_round_trip() -> None:
    client, server = _pair()
    metadata = _metadata()
    claims, headers = _signed_request(client, metadata)

    accepted = server.verify_request(headers, metadata, content_length=5)
    assert accepted == claims

    output = b"result"
    digest = hashlib.sha256(output).hexdigest()
    response_headers = server.response_headers(
        accepted,
        metadata,
        output_digest=digest,
        output_size=len(output),
        processing_seconds="0.125",
    )
    client.verify_response(
        response_headers,
        claims,
        metadata,
        output_digest=digest,
        output_size=len(output),
        processing_seconds="0.125",
    )


@pytest.mark.parametrize(
    ("metadata_update", "header_update", "content_length"),
    [
        ({"task_id": "task-2"}, {}, 5),
        ({"track": "upscaling"}, {}, 5),
        ({"input_digest": "f" * 64}, {}, 5),
        ({"params": {"quality": 8}}, {}, 5),
        ({"deadline_seconds": 31.0}, {}, 5),
        ({}, {MINER_REQUEST_TIMESTAMP_HEADER: "1001"}, 5),
        ({}, {MINER_REQUEST_NONCE_HEADER: "02" * 16}, 5),
        ({}, {MINER_INPUT_SIZE_HEADER: "6"}, 6),
    ],
)
def test_request_signature_binds_all_task_and_transport_facts(
    metadata_update: dict[str, object],
    header_update: dict[str, str],
    content_length: int,
) -> None:
    client, server = _pair()
    original = _metadata()
    _, headers = _signed_request(client, original)
    headers.update(header_update)
    observed = original.model_copy(update=metadata_update)

    with pytest.raises(ArtifactAuthInvalid, match="signature"):
        server.verify_request(headers, observed, content_length=content_length)


def test_invalid_signature_never_queries_validator_registry() -> None:
    class CountingRegistry:
        calls = 0

        def is_current_validator(self, hotkey: str) -> bool:
            self.calls += 1
            return hotkey == "validator-a"

    client, server = _pair()
    registry = CountingRegistry()
    server.validators = registry
    metadata = _metadata()
    _, headers = _signed_request(client, metadata)
    headers[MINER_REQUEST_SIGNATURE_HEADER] = "00" * 64

    with pytest.raises(ArtifactAuthInvalid, match="signature"):
        server.verify_request(headers, metadata, content_length=5)

    assert registry.calls == 0


def test_request_for_another_miner_is_refused() -> None:
    client, server = _pair()
    metadata = _metadata()
    _, headers = _signed_request(client, metadata, miner="miner-b")
    with pytest.raises(ArtifactWrongMiner):
        server.verify_request(headers, metadata, content_length=5)


def test_unregistered_or_foreign_validator_is_refused() -> None:
    client, server = _pair(validator="validator-b")
    metadata = _metadata()
    _, headers = _signed_request(client, metadata)
    with pytest.raises(ArtifactUnregisteredValidator):
        server.verify_request(headers, metadata, content_length=5)


def test_chain_validator_registry_requires_one_fresh_permitted_identity() -> None:
    class Chain:
        fresh = True
        rows = [SimpleNamespace(hotkey="validator-a", is_validator=True)]
        refreshes = 0

        def refresh(self) -> None:
            self.refreshes += 1

        def neurons(self):  # type: ignore[no-untyped-def]
            return list(self.rows)

        def has_fresh_snapshot(self, _now: float, _max_age: float) -> bool:
            return self.fresh

    chain = Chain()
    now = [1_000.0]
    registry = ChainValidatorRegistry(chain, ttl_seconds=45, clock=lambda: now[0])
    assert registry.is_current_validator("validator-a") is True
    assert chain.refreshes == 1

    # P2: the permit view is CACHED — a burst of requests inside the TTL costs
    # exactly one chain refresh (the per-request refresh was an RPC-flood vector).
    for _ in range(25):
        assert registry.is_current_validator("validator-a") is True
    assert chain.refreshes == 1

    # Permit loss revokes within one TTL.
    chain.rows = [SimpleNamespace(hotkey="validator-a", is_validator=False)]
    now[0] += 46
    assert registry.is_current_validator("validator-a") is False

    # Duplicate identities are never a validator.
    chain.rows = [
        SimpleNamespace(hotkey="validator-a", is_validator=True),
        SimpleNamespace(hotkey="validator-a", is_validator=True),
    ]
    now[0] += 46
    assert registry.is_current_validator("validator-a") is False

    # A stale chain snapshot serves the last good view only up to max-stale,
    # then fails closed (unknown => refuse).
    chain.rows = [SimpleNamespace(hotkey="validator-a", is_validator=True)]
    now[0] += 46
    assert registry.is_current_validator("validator-a") is True
    chain.fresh = False
    now[0] += 301  # beyond max_snapshot_age_seconds (300)
    assert registry.is_current_validator("validator-a") is False


def test_valid_request_nonce_can_be_claimed_only_once() -> None:
    client, server = _pair()
    metadata = _metadata()
    _, headers = _signed_request(client, metadata)
    server.verify_request(headers, metadata, content_length=5)
    with pytest.raises(ArtifactReplay):
        server.verify_request(headers, metadata, content_length=5)


def test_restart_fence_rejects_even_a_pre_restart_future_skew_signature() -> None:
    now = [1_000.25]

    def clock() -> float:
        return now[0]

    server = ArtifactServerAuth(
        _signer("miner-a"),
        FrozenValidatorRegistry(("validator-a",)),
        verify_fn=_verify,
        clock=clock,
        request_future_skew_seconds=5,
    )
    assert server.request_timestamp_floor == 1_005
    client = ArtifactClientAuth(
        _signer("validator-a"),
        verify_fn=_verify,
        clock=lambda: 1_005.0,
        nonce_factory=lambda: "01" * 16,
    )
    metadata = _metadata()
    _, captured = _signed_request(client, metadata)
    now[0] = 1_005.0
    with pytest.raises(ArtifactAuthColdStart, match="startup fence"):
        server.verify_request(captured, metadata, content_length=5)

    now[0] = 1_006.0
    fresh_client = ArtifactClientAuth(
        _signer("validator-a"),
        verify_fn=_verify,
        clock=clock,
        nonce_factory=lambda: "02" * 16,
    )
    _, fresh = _signed_request(fresh_client, metadata)
    server.verify_request(fresh, metadata, content_length=5)


@pytest.mark.parametrize("signed_time", [879.0, 1_006.0])
def test_expired_and_excessively_future_requests_are_refused(
    signed_time: float,
) -> None:
    client_clock = [signed_time]
    client, _ = _pair(now=client_clock)
    metadata = _metadata()
    _, headers = _signed_request(client, metadata)
    server_clock = [1_000.0]
    server = ArtifactServerAuth(
        _signer("miner-a"),
        FrozenValidatorRegistry(("validator-a",)),
        verify_fn=_verify,
        clock=lambda: server_clock[0],
        request_max_age_seconds=120,
        request_future_skew_seconds=5,
    )
    with pytest.raises(ArtifactAuthExpired):
        server.verify_request(headers, metadata, content_length=5)


def test_replay_cache_never_evicts_a_live_nonce_to_make_room() -> None:
    now = [1_000.0]
    first_client, server = _pair(now=now, nonce="01" * 16, cache_entries=1)
    metadata = _metadata()
    _, first = _signed_request(first_client, metadata)
    server.verify_request(first, metadata, content_length=5)

    second_client, _ = _pair(now=now, nonce="02" * 16)
    _, second = _signed_request(second_client, metadata)
    with pytest.raises(ArtifactReplayCacheFull):
        server.verify_request(second, metadata, content_length=5)

    now[0] = 1_121.0
    # The original server owns the cache whose expired entry must now be
    # reclaimed; sign a new request at the advanced wall clock.
    second_client = ArtifactClientAuth(
        _signer("validator-a"),
        verify_fn=_verify,
        clock=lambda: now[0],
        nonce_factory=lambda: "02" * 16,
    )
    _, second = _signed_request(second_client, metadata)
    server.verify_request(second, metadata, content_length=5)


def test_one_registered_validator_cannot_exhaust_every_validators_cache() -> None:
    now = [1_000.0]
    first_a, server = _pair(
        now=now,
        registered=("validator-a", "validator-b"),
        nonce="01" * 16,
        cache_entries=3,
        cache_entries_per_validator=1,
    )
    metadata = _metadata()
    _, headers_a1 = _signed_request(first_a, metadata)
    server.verify_request(headers_a1, metadata, content_length=5)

    second_a, _ = _pair(now=now, nonce="02" * 16)
    _, headers_a2 = _signed_request(second_a, metadata)
    with pytest.raises(ArtifactReplayCacheFull, match="validator quota"):
        server.verify_request(headers_a2, metadata, content_length=5)

    validator_b, _ = _pair(now=now, validator="validator-b", nonce="01" * 16)
    _, headers_b = _signed_request(validator_b, metadata)
    accepted = server.verify_request(headers_b, metadata, content_length=5)
    assert accepted.validator_hotkey == "validator-b"


@pytest.mark.parametrize(
    "mutation",
    ("downgrade", "foreign_miner", "digest", "size", "processing", "task"),
)
def test_response_signature_rejects_downgrade_foreign_identity_and_tamper(
    mutation: str,
) -> None:
    client, server = _pair()
    metadata = _metadata()
    claims, request_headers = _signed_request(client, metadata)
    server.verify_request(request_headers, metadata, content_length=5)
    digest = hashlib.sha256(b"result").hexdigest()
    headers = server.response_headers(
        claims,
        metadata,
        output_digest=digest,
        output_size=6,
        processing_seconds="0.125",
    )
    observed_metadata = metadata
    observed_digest = digest
    observed_size = 6
    observed_processing = "0.125"
    if mutation == "downgrade":
        headers[MINER_ARTIFACT_VERSION_HEADER] = MINER_ARTIFACT_VERSION
    elif mutation == "foreign_miner":
        headers[MINER_HOTKEY_HEADER] = "miner-b"
    elif mutation == "digest":
        observed_digest = "e" * 64
    elif mutation == "size":
        headers[MINER_OUTPUT_SIZE_HEADER] = "7"
    elif mutation == "processing":
        observed_processing = "0.126"
    elif mutation == "task":
        observed_metadata = metadata.model_copy(update={"task_id": "task-2"})

    with pytest.raises((ArtifactAuthInvalid, ArtifactWrongMiner)):
        client.verify_response(
            headers,
            claims,
            observed_metadata,
            output_digest=observed_digest,
            output_size=observed_size,
            processing_seconds=observed_processing,
        )


def test_odd_length_signature_is_never_passed_to_crypto_verifier() -> None:
    client, server = _pair()
    metadata = _metadata()
    _, headers = _signed_request(client, metadata)
    headers[MINER_REQUEST_SIGNATURE_HEADER] = "abc"
    with pytest.raises(ArtifactAuthInvalid, match="hex"):
        server.verify_request(headers, metadata, content_length=5)


@pytest.mark.parametrize("signature", ("ab" * 63, "ab" * 65, "AB" * 64))
def test_non_exact_sr25519_signature_shape_is_rejected(signature: str) -> None:
    client, server = _pair()
    metadata = _metadata()
    _, headers = _signed_request(client, metadata)
    headers[MINER_REQUEST_SIGNATURE_HEADER] = signature
    with pytest.raises(ArtifactAuthInvalid, match="exactly 64-byte lowercase hex"):
        server.verify_request(headers, metadata, content_length=5)
