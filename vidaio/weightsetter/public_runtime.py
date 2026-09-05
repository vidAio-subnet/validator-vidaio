"""Standalone thin-validator wiring and controlled mainnet readiness checks."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import secrets
import sys
from typing import Any
from urllib.parse import urlsplit

from vidaio.audit import AuditConfig, CommitmentLedger, make_public_store, make_unsealed_writer_store
from vidaio.audit.store import ArtifactKind
from vidaio.chain.factory import ChainConfig, make_chain_adapter, make_read_only_chain_adapter
from vidaio.core import CoreConfig, connect, section, setup_logging
from vidaio.epoch import EPOCH_LOG_SCHEMA_VERSION
from vidaio.services import run_service
from vidaio.services.artifact_auth import CallableHotkeySigner
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.weights import ensure_locked_levers
from vidaio.weightsetter import WeightSetter
from vidaio.weightsetter.config import WeightSetterConfig
from vidaio.weightsetter.shared_snapshot import ChainAdapterAnchorReader, make_snapshot_provider


def validate_configuration(raw: dict[str, Any]) -> dict[str, Any]:
    chain = section(raw, "chain", ChainConfig)
    config = section(raw, "weightsetter", WeightSetterConfig)
    audit = section(raw, "audit", AuditConfig)
    core = section(raw, "core", CoreConfig)
    tokenomics = section(raw, "tokenomics", TokenomicsConfig)
    problems = []
    if chain.mode != "bittensor" or chain.network != "finney" or chain.netuid != 85 or core.netuid != 85:
        problems.append("mainnet thin validation requires bittensor/finney/netuid85")
    if not chain.endpoint or any(urlsplit(endpoint).scheme != "wss" or not urlsplit(endpoint).hostname
                                 for endpoint in [chain.endpoint, *chain.fallback_endpoints]):
        problems.append("explicit secure archive chain endpoints are required")
    if not chain.validator_hotkey.strip() or chain.validator_hotkey != config.validator_hotkey:
        problems.append("chain and weightsetter validator hotkeys must match")
    if not chain.anchor_hotkey.strip():
        problems.append("chain.anchor_hotkey must identify the authority anchor signer")
    if chain.anchor_writer_lock_path is None or not chain.anchor_writer_lock_path.is_absolute():
        problems.append("an absolute persistent anchor writer lock path is required")
    if config.provider != "shared" or not config.verify_anchor:
        problems.append("shared snapshots with mandatory independent anchor verification are required")
    authority = urlsplit(config.authority_url)
    if authority.scheme != "https" or not authority.hostname or authority.username or authority.password:
        problems.append("an HTTPS authority URL without embedded credentials is required")
    if config.authority_netuid != 85 or config.version_key != EPOCH_LOG_SCHEMA_VERSION or chain.version_key != EPOCH_LOG_SCHEMA_VERSION:
        problems.append("authority subnet85 and current epoch version_key must match")
    if not config.publication_enabled or config.chain_timeout_seconds < 180:
        problems.append("publication must be enabled and chain_timeout_seconds must be at least180")
    if config.max_chain_snapshot_age_seconds <= 0:
        problems.append("chain snapshot freshness checking must remain enabled")
    if (raw.get("hotkey_auth") or {}).get("mode") != "enforce":
        problems.append("hotkey authentication must be enforce")
    if tokenomics.result_window_hours != 168:
        problems.append("mainnet tokenomics result_window_hours must be168")
    ensure_locked_levers(tokenomics)
    if audit.backend != "s3" or not audit.s3_bucket or not audit.s3_region or audit.allow_plaintext_holdout:
        problems.append("the mainnet public-evidence S3 store must be configured without plaintext holdouts")
    if os.environ.get(audit.holdout_key_env, "").strip():
        problems.append("thin validators must not receive the authority holdout key")
    if problems:
        raise ValueError("; ".join(problems))
    return {"role": "thin-validator-node", "network": chain.network, "netuid": chain.netuid,
            "version_key": config.version_key, "validator_hotkey": config.validator_hotkey,
            "anchor_hotkey": chain.anchor_hotkey, "authority_url": config.authority_url,
            "provider": config.provider, "verify_anchor": True, "publication_enabled": True}


def verify_runtime() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "verify_release_dependencies.py"
    spec = importlib.util.spec_from_file_location("public_release_dependencies", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("release dependency verifier is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.verify(preload_media=True, require_runtime_manifest=True, require_canonical_runtime=True)


def static_preflight(raw: dict[str, Any]) -> dict[str, Any]:
    result = validate_configuration(raw)
    result["dependencies"] = verify_runtime()
    return result


def snapshot_provider(raw: dict[str, Any], chain: Any, store: Any) -> Any:
    config = section(raw, "weightsetter", WeightSetterConfig)
    return make_snapshot_provider(config, store=store, local_provider=None,
                                  anchor_reader=ChainAdapterAnchorReader(chain),
                                  signer=CallableHotkeySigner(config.validator_hotkey, chain.sign))


def fresh_floor(raw: dict[str, Any], latest_epoch: int) -> int:
    value = (raw.get("local_stack") or {}).get("auditor_cursor_floor")
    if isinstance(value, bool) or not isinstance(value, int) or value != latest_epoch + 1:
        raise ValueError(f"fresh deployment requires local_stack.auditor_cursor_floor={latest_epoch + 1}; configured={value!r}")
    return value


def suggested_floor(raw: dict[str, Any]) -> dict[str, int]:
    chain_config = section(raw, "chain", ChainConfig)
    if chain_config.mode != "bittensor" or chain_config.network != "finney" or chain_config.netuid != 85:
        raise ValueError("floor discovery requires finney subnet85")
    chain = make_read_only_chain_adapter(raw)
    try:
        boundary = chain.latest_closed_epoch(netuid=85)
        if boundary is None:
            raise RuntimeError("no archive-proven finalized epoch")
        return {"latest_closed_epoch": boundary.epoch_id, "close_block": boundary.close_block,
                "auditor_cursor_floor": boundary.epoch_id + 1}
    finally:
        chain.close()


def live_preflight(raw: dict[str, Any]) -> dict[str, Any]:
    from bittensor_wallet import Keypair

    validate_configuration(raw)
    chain_config = section(raw, "chain", ChainConfig)
    audit_config = section(raw, "audit", AuditConfig)
    chain = make_chain_adapter(raw)
    try:
        finalized = chain.finalized_block()
        chain.refresh()
        if finalized < 7201 or chain.current_block() < finalized:
            raise RuntimeError("invalid or inconsistent finalized chain head")
        historical = list(chain.neurons_at(finalized - 7200))
        if not historical:
            raise RuntimeError("7200-block archive metagraph unavailable")
        neurons = list(chain.neurons_at(finalized))
        matches = [neuron for neuron in neurons if neuron.hotkey == chain_config.validator_hotkey]
        if len(matches) != 1 or not matches[0].is_validator:
            raise RuntimeError("configured hotkey lacks a finalized SN85 registration/validator permit")
        boundary = chain.latest_closed_epoch(netuid=85)
        if boundary is None or boundary.close_block > chain.finalized_block():
            raise RuntimeError("latest closed epoch is unavailable or unfinalized")
        floor = fresh_floor(raw, boundary.epoch_id)
        message = b"vidaio.public-validator.preflight/1:" + secrets.token_bytes(32)
        signature = chain.sign(message)
        if not Keypair(ss58_address=chain_config.validator_hotkey).verify(message, bytes.fromhex(signature)):
            raise RuntimeError("local hotkey signature verification failed")
        capacity = chain.commitment_capacity(85, chain_config.validator_hotkey)
        if capacity.remaining_space < capacity.required_space(128):
            raise RuntimeError("insufficient commitment publication capacity")
        public_store = make_public_store(audit_config)
        provider = snapshot_provider(raw, chain, public_store)
        provider.miner_snapshots()
        inputs = provider.epoch_inputs()
        resolved_boundary = provider.resolved_latest_boundary()
        if resolved_boundary != (boundary.epoch_id, boundary.close_block):
            raise RuntimeError("epoch changed during preflight; refresh the floor and retry")
        if inputs.epoch_id != boundary.epoch_id or sum(inputs.weight_u16.values()) != 65535:
            raise RuntimeError("authenticated authority vector/epoch is invalid")
        writer = make_unsealed_writer_store(audit_config)
        probe = b"vidaio.public-validator.storage-preflight/1:" + secrets.token_bytes(32)
        reference = writer.put(probe, ArtifactKind.WEIGHT_VECTOR)
        if public_store.get(reference) != probe:
            raise RuntimeError("public publication storage round trip failed")
        return {"status": "MAINNET_THIN_VALIDATOR_PREFLIGHT_PASS", "finalized_block": finalized,
                "archive_probe_block": finalized - 7200, "uid": matches[0].uid,
                "validator_hotkey": chain_config.validator_hotkey, "local_signature_verified": True,
                "latest_closed_epoch": inputs.epoch_id, "close_block": inputs.close_block,
                "auditor_cursor_floor": floor, "snapshot_digest": provider.resolved_snapshot_digest(),
                "three_way_digest_verified": True, "weight_u16": inputs.weight_u16,
                "public_storage_probe_digest": reference.digest,
                "commit_reveal_enabled": bool(chain.commit_reveal_enabled()),
                "chain_write_performed": False, "scoring_or_auditor_qualification_claimed": False}
    finally:
        chain.close()


def run_thin_validator(raw: dict[str, Any]) -> None:
    static_preflight(raw)
    core = section(raw, "core", CoreConfig)
    core.data_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(core.log_level)
    chain = make_chain_adapter(raw)
    connection = connect(core.db_path)
    store = make_unsealed_writer_store(section(raw, "audit", AuditConfig))
    ledger = CommitmentLedger.open(core.data_dir / "ledger.db")
    provider = snapshot_provider(raw, chain, store)
    service = WeightSetter(raw, chain=chain, snapshots=provider, conn=connection,
                           store=store, ledger=ledger, publication_inputs=provider)
    run_service(service)
