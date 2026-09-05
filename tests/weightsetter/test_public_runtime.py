import sys
from types import SimpleNamespace

import pytest

from vidaio.audit.store import ArtifactKind
from vidaio.chain.adapter import ChainNeuron, CommitmentCapacity, EpochBoundary
from vidaio.weightsetter import public_runtime


@pytest.fixture
def public_config(tmp_path):
    return {
        "core": {"data_dir": str(tmp_path), "netuid": 85},
        "chain": {"mode": "bittensor", "network": "finney", "netuid": 85,
                  "endpoint": "wss://archive.example:443", "validator_hotkey": "validator",
                  "anchor_hotkey": "authority", "anchor_writer_lock_path": str(tmp_path / "writer.lock")},
        "weightsetter": {"provider": "shared", "verify_anchor": True,
                         "validator_hotkey": "validator", "authority_url": "https://authority.example",
                         "publication_enabled": True},
        "hotkey_auth": {"mode": "enforce"},
        "tokenomics": {"result_window_hours": 168},
        "audit": {"backend": "s3", "s3_bucket": "evidence", "s3_region": "eu-central-1"},
        "local_stack": {"auditor_cursor_floor": 25},
    }


@pytest.mark.parametrize("section,key,value", [
    ("chain", "network", "test"), ("chain", "netuid", 1),
    ("chain", "endpoint", "ws://archive.example"),
    ("chain", "fallback_endpoints", ["ws://archive.example"]),
    ("chain", "anchor_hotkey", ""), ("chain", "anchor_writer_lock_path", "relative.lock"),
    ("weightsetter", "validator_hotkey", "wrong"), ("weightsetter", "provider", "local"),
    ("weightsetter", "verify_anchor", False), ("weightsetter", "publication_enabled", False),
    ("weightsetter", "authority_url", "http://authority.example"),
    ("weightsetter", "authority_url", "https://secret:password@authority.example"),
    ("weightsetter", "version_key", 15), ("weightsetter", "chain_timeout_seconds", 10),
    ("weightsetter", "max_chain_snapshot_age_seconds", 0),
    ("hotkey_auth", "mode", "observe"), ("tokenomics", "result_window_hours", 24),
    ("tokenomics", "burn_proportion", 1.0), ("audit", "backend", "local"),
    ("audit", "allow_plaintext_holdout", True),
])
def test_public_config_fails_closed(public_config, section, key, value):
    public_config[section][key] = value
    with pytest.raises(ValueError):
        public_runtime.validate_configuration(public_config)


def test_no_authority_holdout_key(public_config, monkeypatch):
    monkeypatch.setenv("VIDAIO_AUDIT_HOLDOUT_KEY", "not-for-a-validator")
    with pytest.raises(ValueError, match="holdout key"):
        public_runtime.validate_configuration(public_config)


def test_static_invokes_release_qualification(public_config, monkeypatch):
    monkeypatch.setattr(public_runtime, "verify_runtime", lambda: {"runtime": "verified"})
    result = public_runtime.static_preflight(public_config)
    assert result["dependencies"] == {"runtime": "verified"}
    assert result["provider"] == "shared"


@pytest.mark.parametrize("floor", [None, 0, 24, 26, True, "25"])
def test_fresh_floor_is_exact(public_config, floor):
    public_config["local_stack"]["auditor_cursor_floor"] = floor
    with pytest.raises(ValueError, match="auditor_cursor_floor=25"):
        public_runtime.fresh_floor(public_config, 24)


def test_floor_discovery_does_not_construct_a_signer(public_config, monkeypatch):
    events = []
    chain = SimpleNamespace(latest_closed_epoch=lambda **kwargs: EpochBoundary(24, 9000),
                            close=lambda: events.append("closed"))
    monkeypatch.setattr(public_runtime, "make_read_only_chain_adapter", lambda raw: chain)
    monkeypatch.setattr(public_runtime, "make_chain_adapter", lambda raw: pytest.fail("signer loaded"))
    assert public_runtime.suggested_floor(public_config)["auditor_cursor_floor"] == 25
    assert events == ["closed"]


@pytest.fixture
def live_dependencies(monkeypatch):
    events = []
    neuron = ChainNeuron(7, "validator", "coldkey", "1.2.3.4", 100, 0, is_validator=True)
    def read_neurons(block):
        events.append(("neurons", block))
        return [neuron]
    chain = SimpleNamespace(finalized_block=lambda: 10000, current_block=lambda: 10001,
                            refresh=lambda: None, neurons_at=read_neurons,
                            latest_closed_epoch=lambda **kwargs: EpochBoundary(24, 9000),
                            sign=lambda message: "ab" * 64, close=lambda: events.append("closed"),
                            commitment_capacity=lambda *args: CommitmentCapacity(85, "validator", 10000, 24, 24, 65536, 0, 0),
                            commit_reveal_enabled=lambda: True)
    inputs = SimpleNamespace(epoch_id=24, close_block=9000, weight_u16={7: 65535})
    provider = SimpleNamespace(miner_snapshots=lambda: events.append("verified-snapshot"),
                               epoch_inputs=lambda: inputs,
                               resolved_latest_boundary=lambda: (24, 9000),
                               resolved_snapshot_digest=lambda: "a" * 64)
    def put(payload, kind):
        assert kind == ArtifactKind.WEIGHT_VECTOR
        events.append(("public-probe", payload))
        return SimpleNamespace(digest="b" * 64)
    writer = SimpleNamespace(put=put)
    reader = SimpleNamespace(get=lambda reference: events[-1][1])
    keypair = SimpleNamespace(verify=lambda message, signature: signature == bytes.fromhex("ab" * 64))
    monkeypatch.setitem(sys.modules, "bittensor_wallet", SimpleNamespace(Keypair=lambda **kwargs: keypair))
    monkeypatch.setattr(public_runtime, "make_chain_adapter", lambda raw: chain)
    monkeypatch.setattr(public_runtime, "snapshot_provider", lambda *args: provider)
    monkeypatch.setattr(public_runtime, "make_public_store", lambda config: reader)
    monkeypatch.setattr(public_runtime, "make_unsealed_writer_store", lambda config: writer)
    return chain, provider, events


def test_live_preflight_is_not_a_signing_role(public_config, live_dependencies):
    chain, provider, events = live_dependencies
    result = public_runtime.live_preflight(public_config)
    assert result["uid"] == 7
    assert result["three_way_digest_verified"] is True
    assert result["chain_write_performed"] is False
    assert result["scoring_or_auditor_qualification_claimed"] is False
    assert ("neurons", 2800) in events
    assert events[-1] == "closed"


@pytest.mark.parametrize("failure", ["permit", "signature", "capacity", "boundary", "vector", "anchor"])
def test_live_failure_closes_chain_before_storage_write(public_config, live_dependencies, monkeypatch, failure):
    chain, provider, events = live_dependencies
    if failure == "permit":
        chain.neurons_at = lambda block: [ChainNeuron(7, "validator", "cold", "", 0, 0)]
    elif failure == "signature":
        chain.sign = lambda message: "cd" * 64
    elif failure == "capacity":
        chain.commitment_capacity = lambda *args: CommitmentCapacity(85, "validator", 10000, 24, 24, 1024, 1024, 1024)
    elif failure == "boundary":
        provider.resolved_latest_boundary = lambda: (23, 8640)
    elif failure == "vector":
        provider.epoch_inputs = lambda: SimpleNamespace(epoch_id=24, weight_u16={7: 123})
    else:
        provider.miner_snapshots = lambda: (_ for _ in ()).throw(RuntimeError("anchor mismatch"))
    with pytest.raises(RuntimeError):
        public_runtime.live_preflight(public_config)
    assert not any(isinstance(event, tuple) and event[0] == "public-probe" for event in events)
    assert events[-1] == "closed"


def test_runner_wires_shared_provider_and_publication_inputs(public_config, monkeypatch):
    chain = object()
    provider = object()
    captured = {}
    monkeypatch.setattr(public_runtime, "static_preflight", lambda raw: {})
    monkeypatch.setattr(public_runtime, "make_chain_adapter", lambda raw: chain)
    monkeypatch.setattr(public_runtime, "snapshot_provider", lambda raw, adapter, store: provider)
    monkeypatch.setattr(public_runtime, "make_unsealed_writer_store", lambda config: object())
    monkeypatch.setattr(public_runtime, "WeightSetter", lambda raw, **kwargs: captured.update(kwargs) or "service")
    monkeypatch.setattr(public_runtime, "run_service", lambda service: captured.update(service=service))
    public_runtime.run_thin_validator(public_config)
    assert captured["snapshots"] is provider
    assert captured["publication_inputs"] is provider
    assert captured["chain"] is chain
    assert captured["service"] == "service"
    captured["conn"].close()
