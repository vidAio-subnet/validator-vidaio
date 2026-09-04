from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from advertise_miner import (  # noqa: E402
    advertise_miner,
    parse_public_miner_url,
    verify_miner_advertisement,
)


HOTKEY = "5MinerHotkey"
COLDKEY = "5MinerColdkey"
PUBLIC_IP = "8.8.8.8"


@pytest.mark.parametrize(
    "url",
    (
        "http://127.0.0.1:8300",
        "http://169.254.169.254:8300",
        "http://10.0.0.1:8300",
        "http://[::1]:8300",
        "http://[::ffff:127.0.0.1]:8300",
        "http://[2001:4860:4860::8888%25eth0]:8300",
        "http://miner.example:8300",
        "ftp://8.8.8.8:8300",
        "http://8.8.8.8",
        "http://user:secret@8.8.8.8:8300",
        "http://8.8.8.8:8300/task",
    ),
)
def test_public_miner_url_rejects_non_dialable_or_ssrf_shapes(url: str) -> None:
    with pytest.raises(ValueError):
        parse_public_miner_url(url)


def test_advertisement_readback_requires_exact_hotkey_ip_and_port() -> None:
    neuron = SimpleNamespace(
        hotkey=HOTKEY,
        coldkey=COLDKEY,
        ip=PUBLIC_IP,
        axon_port=8300,
        is_validator=False,
    )
    got = verify_miner_advertisement(
        [neuron], hotkey=HOTKEY, miner_url=f"http://{PUBLIC_IP}:8300"
    )
    assert (got.hotkey, got.ip, got.port) == (HOTKEY, PUBLIC_IP, 8300)

    with pytest.raises(RuntimeError, match="expected"):
        verify_miner_advertisement(
            [neuron], hotkey=HOTKEY, miner_url=f"http://{PUBLIC_IP}:8301"
        )
    with pytest.raises(RuntimeError, match="not registered"):
        verify_miner_advertisement(
            [neuron], hotkey="5SomeoneElse", miner_url=f"http://{PUBLIC_IP}:8300"
        )


@pytest.mark.parametrize("scheme", ("http", "https"))
def test_public_miner_url_supports_explicit_http_or_https(scheme: str) -> None:
    assert parse_public_miner_url(f"{scheme}://{PUBLIC_IP}:8300") == (
        PUBLIC_IP,
        8300,
    )
    assert (
        verify_miner_advertisement(
            [
                SimpleNamespace(
                    hotkey=HOTKEY,
                    coldkey=COLDKEY,
                    ip=PUBLIC_IP,
                    axon_port=8300,
                    is_validator=False,
                )
            ],
            hotkey=HOTKEY,
            miner_url=f"{scheme}://{PUBLIC_IP}:8300",
            required_scheme=scheme,
        ).hotkey
        == HOTKEY
    )


@pytest.mark.parametrize("permit", (False, True))
def test_advertisement_preserves_permit_without_excluding_miner(
    permit: bool,
) -> None:
    got = verify_miner_advertisement(
        [
            SimpleNamespace(
                hotkey=HOTKEY,
                coldkey=COLDKEY,
                axon_info=SimpleNamespace(ip=PUBLIC_IP, port=8300),
                validator_permit=permit,
            )
        ],
        hotkey=HOTKEY,
        miner_url=f"https://{PUBLIC_IP}:8300",
    )
    assert got.validator_permit is permit


@pytest.mark.parametrize(
    "neuron, message",
    (
        (
            SimpleNamespace(
                hotkey=HOTKEY, coldkey=COLDKEY, ip=PUBLIC_IP, axon_port=8300
            ),
            "no available validator status",
        ),
        (
            SimpleNamespace(
                hotkey=HOTKEY,
                ip=PUBLIC_IP,
                axon_port=8300,
                is_validator=False,
            ),
            "no coldkey identity",
        ),
        (
            SimpleNamespace(
                hotkey=HOTKEY,
                coldkey=COLDKEY,
                ip=PUBLIC_IP,
                axon_port=8300,
                is_validator=False,
                validator_permit=True,
            ),
            "contradictory validator status",
        ),
    ),
)
def test_advertisement_rejects_malformed_or_ambiguous_status(
    neuron: object, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        verify_miner_advertisement(
            [neuron],
            hotkey=HOTKEY,
            miner_url=f"http://{PUBLIC_IP}:8300",
        )


def test_public_miner_url_rejects_fleet_scheme_drift() -> None:
    with pytest.raises(ValueError, match="fleet-wide miner_url_scheme"):
        parse_public_miner_url(f"http://{PUBLIC_IP}:8300", required_scheme="https")


class _Keypair:
    def __init__(self, ss58_address: str) -> None:
        self.ss58_address = ss58_address

    @classmethod
    def create_from_seed(cls, _seed: str) -> "_Keypair":
        return cls(HOTKEY)

    @classmethod
    def create_from_mnemonic(cls, _mnemonic: str) -> "_Keypair":
        return cls(HOTKEY)


class _Axon:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _Subtensor:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.neuron = SimpleNamespace(
            hotkey=HOTKEY,
            coldkey=COLDKEY,
            is_null=False,
            validator_permit=False,
            axon_info=SimpleNamespace(ip="0.0.0.0", port=0),
        )

    def get_neuron_for_pubkey_and_subnet(self, hotkey: str, *, netuid: int):
        assert hotkey == HOTKEY
        assert netuid == 85
        return self.neuron

    def serve_axon(self, **kwargs):
        self.calls.append(kwargs)
        axon = kwargs["axon"]
        self.neuron.axon_info = SimpleNamespace(
            ip=axon.external_ip, port=axon.external_port
        )
        return SimpleNamespace(success=True, message="included", error=None)


def test_advertise_signs_waits_for_finalization_and_checks_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MINER_SEED", "0x" + "11" * 32)
    config = SimpleNamespace(
        wallet_name="",
        wallet_hotkey="",
        wallet_path="",
        hotkey_seed_env="TEST_MINER_SEED",
        validator_hotkey=HOTKEY,
        netuid=85,
        endpoint="wss://chain.example",
        network="test",
    )
    bt = SimpleNamespace(Keypair=_Keypair, Axon=_Axon)
    subtensor = _Subtensor()
    # A miner may gain a validator permit as it earns. The advertisement remains
    # valid and the observed capability is preserved in the receipt.
    subtensor.neuron.validator_permit = True

    got = advertise_miner(
        config,
        external_ip=PUBLIC_IP,
        external_port=8300,
        bt_module=bt,
        subtensor=subtensor,
    )

    assert got.hotkey == HOTKEY
    assert got.coldkey == COLDKEY
    assert got.ip == PUBLIC_IP
    assert got.port == 8300
    assert got.validator_permit is True
    assert len(subtensor.calls) == 1
    assert subtensor.calls[0]["netuid"] == 85
    assert subtensor.calls[0]["mev_protection"] is False
    assert subtensor.calls[0]["wait_for_inclusion"] is True
    assert subtensor.calls[0]["wait_for_finalization"] is True
    assert subtensor.calls[0]["wait_for_revealed_execution"] is True
    assert subtensor.calls[0]["axon"].wallet.coldkeypub.ss58_address == COLDKEY


def test_advertise_rejects_unavailable_status_before_chain_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MINER_SEED", "0x" + "11" * 32)
    config = SimpleNamespace(
        wallet_name="",
        wallet_hotkey="",
        wallet_path="",
        hotkey_seed_env="TEST_MINER_SEED",
        validator_hotkey=HOTKEY,
        netuid=85,
        endpoint="wss://chain.example",
        network="test",
    )
    subtensor = _Subtensor()
    subtensor.neuron = SimpleNamespace(
        hotkey=HOTKEY,
        coldkey=COLDKEY,
        is_null=False,
        axon_info=SimpleNamespace(ip="0.0.0.0", port=0),
    )

    with pytest.raises(RuntimeError, match="no available validator status"):
        advertise_miner(
            config,
            external_ip=PUBLIC_IP,
            external_port=8300,
            bt_module=SimpleNamespace(Keypair=_Keypair, Axon=_Axon),
            subtensor=subtensor,
        )
    assert subtensor.calls == []


def test_advertise_surfaces_a_structured_extrinsic_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MINER_SEED", "0x" + "11" * 32)
    config = SimpleNamespace(
        wallet_name="",
        wallet_hotkey="",
        wallet_path="",
        hotkey_seed_env="TEST_MINER_SEED",
        validator_hotkey=HOTKEY,
        netuid=85,
        endpoint="wss://chain.example",
        network="test",
    )
    subtensor = _Subtensor()

    def rejected(**kwargs):
        subtensor.calls.append(kwargs)
        return SimpleNamespace(
            success=False,
            message="rate limited",
            error=RuntimeError("TooManyRegistrationsThisBlock"),
        )

    subtensor.serve_axon = rejected
    with pytest.raises(
        RuntimeError, match="rate limited.*TooManyRegistrationsThisBlock"
    ):
        advertise_miner(
            config,
            external_ip=PUBLIC_IP,
            external_port=8300,
            bt_module=SimpleNamespace(Keypair=_Keypair, Axon=_Axon),
            subtensor=subtensor,
        )


def test_advertise_refuses_loaded_hotkey_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MINER_SEED", "mnemonic words")
    config = SimpleNamespace(
        wallet_name="",
        wallet_hotkey="",
        wallet_path="",
        hotkey_seed_env="TEST_MINER_SEED",
        validator_hotkey="5DifferentHotkey",
        netuid=85,
        endpoint="",
        network="test",
    )
    with pytest.raises(RuntimeError, match="does not match"):
        advertise_miner(
            config,
            external_ip=PUBLIC_IP,
            external_port=8300,
            bt_module=SimpleNamespace(Keypair=_Keypair, Axon=_Axon),
            subtensor=_Subtensor(),
        )
