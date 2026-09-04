"""Registered-hotkey auth (P2): cached registry, Scheme A/B, modes, fail-closed."""

from __future__ import annotations

import hashlib

import pytest

from vidaio.services.hotkey_auth import (
    HEADER_HOTKEY,
    HEADER_NONCE,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    AuthenticatedHotkey,
    HotkeyAuthConfig,
    HotkeyAuthGuard,
    HotkeyAuthInvalid,
    HotkeyAuthMissing,
    HotkeyBelowStakeFloor,
    HotkeyNoValidatorPermit,
    HotkeyNotRegistered,
    HotkeyRegistryUnavailable,
    HotkeyReplay,
    RegisteredHotkeyRegistry,
    sign_request_headers,
)

VALI = "5" + "F" * 47
MINER = "5" + "E" * 47
STRANGER = "5" + "D" * 47


class _Neuron:
    def __init__(self, hotkey: str, *, permit: bool = False, stake: float = 0.0):
        self.hotkey = hotkey
        self.is_validator = permit
        self.alpha_stake = stake


class FakeChain:
    def __init__(self, neurons):
        self._neurons = list(neurons)
        self.refreshes = 0
        self.fail = False

    def refresh(self):
        if self.fail:
            raise RuntimeError("chain down")
        self.refreshes += 1

    def neurons(self):
        if self.fail:
            raise RuntimeError("chain down")
        return list(self._neurons)


class FakeClock:
    def __init__(self, now: float = 1_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class _Signer:
    """Deterministic fake signer paired with the fake verify below."""

    def __init__(self, hotkey: str):
        self.hotkey = hotkey

    def sign(self, payload: bytes) -> str:
        return hashlib.sha256(self.hotkey.encode() + payload).hexdigest()


def _fake_verify(hotkey: str, payload: bytes, signature: str) -> bool:
    return signature == hashlib.sha256(hotkey.encode() + payload).hexdigest()


def _guard(
    chain=None, *, mode="enforce", clock=None, **cfg
) -> tuple[HotkeyAuthGuard, FakeChain, FakeClock]:
    chain = chain or FakeChain(
        [_Neuron(VALI, permit=True, stake=1000.0), _Neuron(MINER, stake=5.0)]
    )
    clock = clock or FakeClock()
    registry = RegisteredHotkeyRegistry(chain, ttl_seconds=45, clock=clock)
    config = HotkeyAuthConfig(mode=mode, **cfg)
    guard = HotkeyAuthGuard(registry, config, verify_fn=_fake_verify, clock=clock)
    return guard, chain, clock


def _signed(hotkey: str, clock: FakeClock, *, method="GET", path="/epoch/latest", body=b""):
    return sign_request_headers(
        _Signer(hotkey), method=method, path=path, body=body, now=clock.now
    )


# -- registry ---------------------------------------------------------------------------


def test_registry_caches_and_never_refreshes_per_request() -> None:
    chain = FakeChain([_Neuron(VALI, permit=True)])
    clock = FakeClock()
    registry = RegisteredHotkeyRegistry(chain, ttl_seconds=45, clock=clock)
    for _ in range(50):
        assert registry.is_registered(VALI)
    assert chain.refreshes == 1
    clock.now += 46
    assert registry.is_registered(VALI)
    assert chain.refreshes == 2


def test_registry_serves_stale_then_fails_closed() -> None:
    chain = FakeChain([_Neuron(VALI)])
    clock = FakeClock()
    registry = RegisteredHotkeyRegistry(
        chain, ttl_seconds=10, max_stale_seconds=60, clock=clock
    )
    assert registry.is_registered(VALI)
    chain.fail = True
    clock.now += 30  # stale but under max_stale: still serving
    assert registry.is_registered(VALI)
    clock.now += 60  # beyond max_stale: fail closed
    with pytest.raises(HotkeyRegistryUnavailable):
        registry.is_registered(VALI)


def test_deregistration_revokes_within_ttl() -> None:
    chain = FakeChain([_Neuron(VALI, permit=True)])
    clock = FakeClock()
    registry = RegisteredHotkeyRegistry(chain, ttl_seconds=10, clock=clock)
    assert registry.has_validator_permit(VALI)
    chain._neurons = []  # deregistered on chain
    clock.now += 11
    assert not registry.is_registered(VALI)


# -- Scheme A ---------------------------------------------------------------------------


def test_signed_request_roundtrip_enforce() -> None:
    guard, _, clock = _guard()
    verified = guard.require(
        _signed(VALI, clock), method="GET", path="/epoch/latest"
    )
    assert isinstance(verified, AuthenticatedHotkey)
    assert verified.hotkey == VALI and verified.is_validator


def test_missing_headers_refused() -> None:
    guard, _, _ = _guard()
    with pytest.raises(HotkeyAuthMissing):
        guard.require({}, method="GET", path="/epoch/latest")


def test_tampered_body_refused() -> None:
    guard, _, clock = _guard()
    headers = _signed(VALI, clock, method="POST", path="/x", body=b"payload")
    with pytest.raises(HotkeyAuthInvalid):
        guard.require(headers, method="POST", path="/x", body=b"tampered")


def test_wrong_path_refused() -> None:
    guard, _, clock = _guard()
    headers = _signed(VALI, clock, path="/epoch/latest")
    with pytest.raises(HotkeyAuthInvalid):
        guard.require(headers, method="GET", path="/epoch/1")


def test_replayed_nonce_refused() -> None:
    guard, _, clock = _guard()
    headers = _signed(VALI, clock)
    assert guard.require(headers, method="GET", path="/epoch/latest") is not None
    with pytest.raises(HotkeyReplay):
        guard.require(headers, method="GET", path="/epoch/latest")


def test_stale_timestamp_refused() -> None:
    guard, _, clock = _guard()
    headers = _signed(VALI, clock)
    clock.now += 300
    with pytest.raises(HotkeyAuthInvalid):
        guard.require(headers, method="GET", path="/epoch/latest")


def test_unregistered_hotkey_is_403() -> None:
    guard, _, clock = _guard()
    headers = _signed(STRANGER, clock)
    with pytest.raises(HotkeyNotRegistered) as exc:
        guard.require(headers, method="GET", path="/epoch/latest")
    assert exc.value.status_code == 403


def test_validator_permit_gate() -> None:
    guard, _, clock = _guard()
    headers = _signed(MINER, clock)
    with pytest.raises(HotkeyNoValidatorPermit):
        guard.require(
            headers,
            method="GET",
            path="/epoch/latest",
            require_validator_permit=True,
        )


def test_min_stake_gate() -> None:
    guard, _, clock = _guard()
    headers = _signed(MINER, clock)
    with pytest.raises(HotkeyBelowStakeFloor):
        guard.require(
            headers, method="GET", path="/epoch/latest", min_alpha_stake=100.0
        )


# -- Scheme B ---------------------------------------------------------------------------


def test_challenge_token_flow_and_revocation() -> None:
    guard, chain, clock = _guard()
    challenge = guard.mint_challenge()
    headers = sign_request_headers(
        _Signer(VALI),
        method="POST",
        path="/auth/token",
        body=challenge.encode(),
        now=clock.now,
    )
    token = guard.redeem_challenge(headers, challenge=challenge, path="/auth/token")
    verified = guard.require(
        {"authorization": f"Bearer {token}"}, method="GET", path="/epoch/latest"
    )
    assert verified is not None and verified.scheme == "token"
    # deregistration revokes the token within the registry TTL
    chain._neurons = []
    clock.now += 46
    with pytest.raises(HotkeyNotRegistered):
        guard.require(
            {"authorization": f"Bearer {token}"}, method="GET", path="/epoch/latest"
        )


def test_expired_token_refused() -> None:
    guard, _, clock = _guard(token_ttl_seconds=60)
    challenge = guard.mint_challenge()
    headers = sign_request_headers(
        _Signer(VALI), method="POST", path="/auth/token",
        body=challenge.encode(), now=clock.now,
    )
    token = guard.redeem_challenge(headers, challenge=challenge, path="/auth/token")
    clock.now += 61
    with pytest.raises(Exception):
        guard.require(
            {"authorization": f"Bearer {token}"}, method="GET", path="/epoch/latest"
        )


def test_challenge_is_single_use() -> None:
    guard, _, clock = _guard()
    challenge = guard.mint_challenge()
    headers = sign_request_headers(
        _Signer(VALI), method="POST", path="/auth/token",
        body=challenge.encode(), now=clock.now,
    )
    guard.redeem_challenge(headers, challenge=challenge, path="/auth/token")
    headers2 = sign_request_headers(
        _Signer(VALI), method="POST", path="/auth/token",
        body=challenge.encode(), now=clock.now,
    )
    with pytest.raises(HotkeyAuthInvalid):
        guard.redeem_challenge(headers2, challenge=challenge, path="/auth/token")


def test_forged_token_refused() -> None:
    guard, _, _ = _guard()
    forged = f"1.{VALI}.99999999999.{'ab' * 32}"
    with pytest.raises(Exception):
        guard.require(
            {"authorization": f"Bearer {forged}"}, method="GET", path="/x"
        )


# -- modes ------------------------------------------------------------------------------


def test_log_mode_never_refuses_but_returns_none() -> None:
    guard, _, _ = _guard(mode="log")
    assert guard.require({}, method="GET", path="/epoch/latest") is None


def test_off_mode_is_a_noop() -> None:
    guard, _, _ = _guard(mode="off")
    assert guard.require({}, method="GET", path="/epoch/latest") is None
