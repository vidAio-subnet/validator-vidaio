#!/usr/bin/env python3
"""Advertise the reference miner's HTTP(S) endpoint in the Bittensor metagraph.

The vidaio wire protocol is HTTP(S), not a Bittensor Synapse, but validators
discover peers from ``metagraph.axons[uid].{ip,port}``.  This one-shot helper uses
the pinned SDK's ``serve_axon`` extrinsic only to publish that address.  It does
not start an Axon server and never needs a coldkey private key: the registered
coldkey *public* address is read from chain while the hotkey signs the extrinsic.

Run from the release image after registering the miner hotkey and starting the
HTTP(S) miner edge (one identity/endpoint per track):

    python scripts/advertise_miner.py --config config/default.yaml \
      --external-ip PUBLIC_IP --external-port 8300 --external-scheme https

``VIDAIO__CHAIN__*`` overrides select the testnet endpoint/netuid and miner
hotkey wallet exactly as they do for the chain adapter.  A non-global address,
an identity mismatch, a rejected extrinsic, or a mismatching metagraph readback
is a hard failure. Bittensor stores only IP/port, so ``--external-scheme`` must
match the fleet-wide ``validator.miner_url_scheme`` used by every validator.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True, slots=True)
class AdvertisedMiner:
    hotkey: str
    coldkey: str
    ip: str
    port: int
    validator_permit: bool


def neuron_validator_permit(neuron: object) -> bool:
    """Normalize validator-permit capability from adapter and SDK neuron shapes.

    ``ChainNeuron`` exposes ``is_validator`` while the SDK ``NeuronInfo`` returned
    by ``get_neuron_for_pubkey_and_subnet`` exposes ``validator_permit``. A permit
    is not an exclusive role: a serving miner can acquire one as stake changes and
    must remain advertiseable/earnable. Advertisement is still a chain write, so
    an absent, malformed, or contradictory status is never guessed.
    """
    hotkey = str(getattr(neuron, "hotkey", ""))
    observed: list[tuple[str, bool]] = []
    for field in ("is_validator", "validator_permit"):
        if not hasattr(neuron, field):
            continue
        value = getattr(neuron, field)
        if not isinstance(value, bool):
            raise RuntimeError(
                f"metagraph entry for {hotkey!r} has unavailable/malformed "
                f"validator status in {field}: {value!r}"
            )
        observed.append((field, value))
    if not observed:
        raise RuntimeError(
            f"metagraph entry for {hotkey!r} has no available validator status"
        )
    if any(value != observed[0][1] for _, value in observed[1:]):
        rendered = ", ".join(f"{field}={value}" for field, value in observed)
        raise RuntimeError(
            f"metagraph entry for {hotkey!r} has contradictory validator status: "
            f"{rendered}"
        )
    return observed[0][1]


def require_validator_status(neuron: object) -> bool:
    """Return the observed permit capability; never use it as a miner exclusion."""
    return neuron_validator_permit(neuron)


def _public_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Return a canonical globally-routable literal, rejecting SSRF spellings."""
    if "%" in value:
        # A zone id is an interface-local routing instruction, not a portable
        # chain address, and can make equivalent endpoints compare unequal.
        raise ValueError("miner endpoint IP must not contain an IPv6 zone identifier")
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError as exc:
        raise ValueError("miner endpoint must use a literal IP address") from exc
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if not address.is_global or address.is_multicast or address.is_unspecified:
        raise ValueError(f"miner endpoint IP {value!r} is not globally routable")
    return address


def parse_public_miner_url(
    url: str, *, required_scheme: str | None = None
) -> tuple[str, int]:
    """Parse the exact URL shape the shipped ``HttpMinerClient`` can dial."""
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("miner URL scheme must be exactly http:// or https://")
    if required_scheme is not None and scheme != required_scheme:
        raise ValueError(
            f"miner URL scheme {scheme!r} does not match the validator's "
            f"fleet-wide miner_url_scheme {required_scheme!r}"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("miner URL must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("miner URL must contain only a host and explicit port")
    if parsed.hostname is None:
        raise ValueError("miner URL has no host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"miner URL has an invalid port: {exc}") from exc
    if port is None:
        raise ValueError("miner URL must include the advertised port explicitly")
    address = _public_ip(parsed.hostname)
    return address.compressed, port


def _peer(neuron: object) -> AdvertisedMiner:
    """Normalize an adapter ``ChainNeuron`` or SDK ``NeuronInfo`` readback."""
    hotkey = str(getattr(neuron, "hotkey", ""))
    coldkey = str(getattr(neuron, "coldkey", ""))
    if not coldkey:
        raise RuntimeError(
            f"metagraph entry for {hotkey!r} has no coldkey identity for dedup proof"
        )
    validator_permit = require_validator_status(neuron)
    axon = getattr(neuron, "axon_info", None)
    if axon is None:
        ip = str(getattr(neuron, "ip", ""))
        port = getattr(neuron, "axon_port", None)
    else:
        ip = str(getattr(axon, "ip", ""))
        port = getattr(axon, "port", None)
    if not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65536:
        raise RuntimeError(
            f"metagraph entry for {hotkey!r} has no valid axon port: {port!r}"
        )
    try:
        normalized_ip = _public_ip(ip).compressed
    except ValueError as exc:
        raise RuntimeError(
            f"metagraph entry for {hotkey!r} has no public axon IP: {ip!r}"
        ) from exc
    return AdvertisedMiner(
        hotkey=hotkey,
        coldkey=coldkey,
        ip=normalized_ip,
        port=port,
        validator_permit=validator_permit,
    )


def verify_miner_advertisement(
    neurons: Iterable[object],
    *,
    hotkey: str,
    miner_url: str,
    required_scheme: str | None = None,
) -> AdvertisedMiner:
    """Require one metagraph entry whose IP/port exactly match ``miner_url``."""
    expected_ip, expected_port = parse_public_miner_url(
        miner_url, required_scheme=required_scheme
    )
    matches = [
        neuron for neuron in neurons if str(getattr(neuron, "hotkey", "")) == hotkey
    ]
    if not matches:
        raise RuntimeError(f"miner hotkey {hotkey!r} is not registered on the subnet")
    if len(matches) != 1:
        raise RuntimeError(
            f"metagraph contains {len(matches)} entries for hotkey {hotkey!r}"
        )
    actual = _peer(matches[0])
    if (actual.ip, actual.port) != (expected_ip, expected_port):
        raise RuntimeError(
            f"miner {hotkey!r} advertises {actual.ip}:{actual.port}, expected "
            f"{expected_ip}:{expected_port} from {miner_url!r}"
        )
    return actual


class _HotkeyWallet:
    """Hotkey signer + public coldkey shape required by ``serve_axon``."""

    def __init__(self, hotkey: object, coldkeypub: object) -> None:
        self.hotkey = hotkey
        self.coldkeypub = coldkeypub

    def unlock_hotkey(self) -> object:
        return self.hotkey


def _load_wallet(bt: Any, subtensor: Any, config: Any) -> tuple[object, object]:
    """Load the signing hotkey and bind it to its chain-registered coldkey pubkey."""
    if config.wallet_name and config.wallet_hotkey:
        disk_wallet = bt.Wallet(
            name=config.wallet_name,
            hotkey=config.wallet_hotkey,
            path=config.wallet_path or None,
        )
        hotkey = disk_wallet.hotkey
    else:
        encoded = os.environ.get(config.hotkey_seed_env, "").strip()
        if not encoded:
            raise RuntimeError(
                "no miner hotkey configured: set chain.wallet_name/wallet_hotkey or "
                f"${config.hotkey_seed_env}"
            )
        hotkey = (
            bt.Keypair.create_from_seed(encoded)
            if encoded.startswith("0x")
            else bt.Keypair.create_from_mnemonic(encoded)
        )
    hotkey_ss58 = str(getattr(hotkey, "ss58_address", ""))
    if not hotkey_ss58:
        raise RuntimeError("loaded miner hotkey exposes no ss58 address")
    if config.validator_hotkey.strip() and hotkey_ss58 != config.validator_hotkey:
        raise RuntimeError(
            f"loaded hotkey {hotkey_ss58} does not match chain.validator_hotkey "
            f"{config.validator_hotkey}"
        )
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(
        hotkey_ss58, netuid=config.netuid
    )
    if bool(getattr(neuron, "is_null", False)):
        raise RuntimeError(
            f"miner hotkey {hotkey_ss58} is not registered on subnet {config.netuid}"
        )
    # Refuse malformed/ambiguous SDK state before ``serve_axon``. A true validator
    # permit is recorded but does not disqualify a serving miner: permits can change
    # as stake changes and are a capability, not an exclusive chain role.
    require_validator_status(neuron)
    coldkey_ss58 = str(getattr(neuron, "coldkey", ""))
    if not coldkey_ss58:
        raise RuntimeError("registered neuron exposes no coldkey public address")
    coldkeypub = bt.Keypair(ss58_address=coldkey_ss58)
    return _HotkeyWallet(hotkey, coldkeypub), neuron


def advertise_miner(
    config: Any,
    *,
    external_ip: str,
    external_port: int,
    external_scheme: str = "http",
    bt_module: Any | None = None,
    subtensor: Any | None = None,
) -> AdvertisedMiner:
    """Submit the serve extrinsic, wait for finalization, and verify readback."""
    address = _public_ip(external_ip).compressed
    scheme = external_scheme.strip().lower()
    if scheme not in {"http", "https"}:
        raise ValueError("external_scheme must be exactly 'http' or 'https'")
    if not isinstance(external_port, int) or isinstance(external_port, bool):
        raise ValueError("external_port must be an integer")
    if not 0 < external_port < 65536:
        raise ValueError("external_port must be in [1, 65535]")
    if bt_module is None:
        try:
            import bittensor as bt_module
        except ImportError as exc:
            raise RuntimeError(
                "advertising a miner requires the release 'chain' extra"
            ) from exc
    owns_subtensor = subtensor is None
    if subtensor is None:
        subtensor = bt_module.Subtensor(
            network=config.endpoint or config.network,
            fallback_endpoints=list(getattr(config, "fallback_endpoints", ()) or ())
            or None,
        )
    try:
        wallet, _ = _load_wallet(bt_module, subtensor, config)
        hotkey = str(wallet.hotkey.ss58_address)
        axon = bt_module.Axon(
            wallet=wallet,
            ip="0.0.0.0",
            port=external_port,
            external_ip=address,
            external_port=external_port,
        )
        response = subtensor.serve_axon(
            netuid=config.netuid,
            axon=axon,
            mev_protection=False,
            raise_error=False,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            wait_for_revealed_execution=True,
        )
        success = getattr(response, "success", False)
        if callable(success):
            success = success()
        if not bool(success):
            message = str(getattr(response, "message", "") or "serve_axon rejected")
            error = getattr(response, "error", None)
            raise RuntimeError(
                f"serve_axon did not finalize successfully: {message}"
                + (f" ({error})" if error else "")
            )
        readback = subtensor.get_neuron_for_pubkey_and_subnet(
            hotkey, netuid=config.netuid
        )
        return verify_miner_advertisement(
            [readback],
            hotkey=hotkey,
            miner_url=(
                f"{scheme}://[{address}]:{external_port}"
                if ":" in address
                else f"{scheme}://{address}:{external_port}"
            ),
            required_scheme=scheme,
        )
    finally:
        if owns_subtensor:
            close = getattr(subtensor, "close", None)
            if callable(close):
                close()


def main() -> int:
    from vidaio.chain.factory import ChainConfig
    from vidaio.core import load_raw_config, section
    from vidaio.validator.config import ValidatorConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    parser.add_argument("--external-ip", required=True, help="public literal IPv4/IPv6")
    parser.add_argument("--external-port", required=True, type=int)
    parser.add_argument(
        "--external-scheme",
        choices=("http", "https"),
        help="must match validator.miner_url_scheme (defaults to that config value)",
    )
    args = parser.parse_args()
    raw = load_raw_config(args.config)
    config = section(raw, "chain", ChainConfig)
    validator = section(raw, "validator", ValidatorConfig)
    if config.mode != "bittensor":
        raise SystemExit("miner advertisement requires chain.mode=bittensor")
    result = advertise_miner(
        config,
        external_ip=args.external_ip,
        external_port=args.external_port,
        external_scheme=args.external_scheme or validator.miner_url_scheme,
    )
    print(json.dumps({"status": "ok", **asdict(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
