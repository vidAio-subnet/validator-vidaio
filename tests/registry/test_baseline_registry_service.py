"""Production boundary for the persistent schema-v14 baseline registry."""

from __future__ import annotations

import copy
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from vidaio.audit.store import ArtifactKind, LocalFsStore
from vidaio.competition.interfaces import logical_build_identity
from vidaio.registry import (
    REGISTRY_API_KIND,
    BaselineRegistryService,
    GenesisBaselineError,
    RegistryConfig,
    RegistryStartupError,
    current_baseline,
    production_registry_problems,
)

NOW = datetime(2026, 8, 27, 2, 0, tzinfo=timezone.utc)


def _configured_world(tmp_path: Path):
    store = LocalFsStore(tmp_path / "audit")
    seeds: dict[str, dict[str, object]] = {}
    for index, track in enumerate(("compression", "upscaling"), start=1):
        executable = store.put(
            f"{track}-reference-v0".encode(), ArtifactKind.SUBMISSION_ARCHIVE
        )
        provenance = store.put(
            f"{track}-reference-v0-provenance".encode(), ArtifactKind.MANIFEST
        )
        repo_url = f"https://example.invalid/vidaio-{track}-v0.git"
        commit_sha = f"{index + 16:02x}" * 20
        tree_sha = f"{index + 32:02x}" * 20
        seeds[track] = {
            "artifact_digest": executable.digest,
            "artifact_bytes": executable.byte_size,
            "image_digest": logical_build_identity(
                repo_url=repo_url,
                commit_sha=commit_sha,
                tree_sha=tree_sha,
            ),
            "provenance_digest": provenance.digest,
            "provenance_bytes": provenance.byte_size,
            "repo_url": repo_url,
            "commit_sha": commit_sha,
            "tree_sha": tree_sha,
        }
    raw = {
        "core": {"metrics_port": 0},
        "registry": {
            "db_path": str(tmp_path / "registry-state" / "registry.db"),
            "http_host": "127.0.0.1",
            "http_port": 8720,
            "metrics_port": 9123,
            "automatic_promotion_enabled": False,
            "allow_disabled_automatic_promotion_for_testnet": True,
            "genesis_baselines": seeds,
        },
    }
    return raw, store


def _service(raw, store) -> BaselineRegistryService:
    return BaselineRegistryService(raw, metrics_port=0, store=store, now=lambda: NOW)


def test_startup_migrates_persistent_db_and_seeds_exact_v0_pair(tmp_path) -> None:
    raw, store = _configured_world(tmp_path)
    service = _service(raw, store)
    try:
        assert Path(raw["registry"]["db_path"]).is_file()
        assert service.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert {
            track: current_baseline(service.conn, track).version
            for track in ("compression", "upscaling")
        } == {"compression": 0, "upscaling": 0}
        assert all(
            store.is_released(current_baseline(service.conn, track).artifact_ref())
            for track in ("compression", "upscaling")
        )
    finally:
        service.close()


def test_metrics_health_reads_registry_from_its_request_thread(tmp_path) -> None:
    """The threaded /health server must not reuse the loop-owned SQLite handle."""
    raw, store = _configured_world(tmp_path)
    service = _service(raw, store)
    results: list[tuple[bool, dict]] = []
    try:
        thread = threading.Thread(
            target=lambda: results.append(service.health.health_payload())
        )
        thread.start()
        thread.join(timeout=5.0)

        assert results, "registry health check thread did not finish"
        ok, payload = results[0]
        assert ok is True, payload
        assert payload["checks"]["invariants"] is True
    finally:
        service.close()


async def test_http_surface_is_read_only_and_exposes_verified_active_state(
    tmp_path,
) -> None:
    raw, store = _configured_world(tmp_path)
    service = _service(raw, store)
    try:
        routes = {
            (route.path, method)
            for route in service.app.routes
            for method in (route.methods or set())
        }
        assert routes == {
            ("/healthz", "GET"),
            ("/v1/baselines", "GET"),
        }

        transport = httpx.ASGITransport(app=service.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://registry"
        ) as client:
            health = await client.get("/healthz")
            assert health.status_code == 200
            assert health.json() == {
                "service": "baseline-registry",
                "status": "ok",
                "schema_version": 14,
                "active_versions": {"compression": 0, "upscaling": 0},
                "archives_verified": True,
                "automatic_promotion": {
                    "enabled": False,
                    "adapter_wired": False,
                    "testnet_exception": True,
                    "status": "disabled_testnet_exception",
                },
            }

            response = await client.get("/v1/baselines")
            assert response.status_code == 200
            body = response.json()
            assert body["kind"] == REGISTRY_API_KIND
            assert body["schema_version"] == 14
            assert body["automatic_promotion"] == {
                "enabled": False,
                "adapter_wired": False,
                "testnet_exception": True,
                "status": "disabled_testnet_exception",
            }
            assert set(body["baselines"]) == {"compression", "upscaling"}
            assert body["pending_promotions"] == {}
            assert all(row["version"] == 0 for row in body["baselines"].values())

            assert (await client.post("/v1/baselines", json={})).status_code == 405
            assert (await client.get("/openapi.json")).status_code == 404
    finally:
        service.close()


def test_identical_restart_is_idempotent_but_conflicting_v0_is_refused(
    tmp_path,
) -> None:
    raw, store = _configured_world(tmp_path)
    first = _service(raw, store)
    identities = {
        track: current_baseline(first.conn, track).baseline_id
        for track in ("compression", "upscaling")
    }
    first.close()

    second = _service(raw, store)
    try:
        assert {
            track: current_baseline(second.conn, track).baseline_id
            for track in ("compression", "upscaling")
        } == identities
        assert second.conn.execute("SELECT COUNT(*) FROM baselines").fetchone()[0] == 2
    finally:
        second.close()

    conflicting = copy.deepcopy(raw)
    replacement = store.put(
        b"different-compression-v0", ArtifactKind.SUBMISSION_ARCHIVE
    )
    compression = conflicting["registry"]["genesis_baselines"]["compression"]
    compression["artifact_digest"] = replacement.digest
    compression["artifact_bytes"] = replacement.byte_size
    with pytest.raises(GenesisBaselineError, match="different archived identity"):
        _service(conflicting, store)


def test_absent_archive_fails_before_either_v0_row_is_inserted(tmp_path) -> None:
    raw, store = _configured_world(tmp_path)
    raw["registry"]["genesis_baselines"]["compression"]["artifact_digest"] = "f" * 64
    db_path = Path(raw["registry"]["db_path"])

    with pytest.raises(GenesisBaselineError, match="not verifiably archived"):
        _service(raw, store)

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM baselines").fetchone()[0] == 0
    finally:
        conn.close()


def test_incomplete_or_malformed_seed_configuration_is_fail_closed(tmp_path) -> None:
    raw, store = _configured_world(tmp_path)
    del raw["registry"]["genesis_baselines"]["upscaling"]
    with pytest.raises(ValueError, match="exactly compression and upscaling"):
        _service(raw, store)

    raw, _store = _configured_world(tmp_path / "second")
    raw["registry"]["genesis_baselines"]["compression"]["repo_url"] = (
        "https://token@example.invalid/repo.git"
    )
    cfg = RegistryConfig.model_validate(raw["registry"])
    assert any(
        "credential-free https git URL" in item for item in cfg.genesis_problems()
    )

    raw, _store = _configured_world(tmp_path / "third")
    raw["registry"]["genesis_baselines"]["compression"]["image_digest"] = "f" * 64
    cfg = RegistryConfig.model_validate(raw["registry"])
    assert any(
        "stable logical build identity" in item for item in cfg.genesis_problems()
    )


async def test_invariant_break_is_visible_and_never_returns_stale_200(tmp_path) -> None:
    raw, store = _configured_world(tmp_path)
    service = _service(raw, store)
    try:
        service.conn.execute(
            "UPDATE baselines SET status = 'superseded' "
            "WHERE track = 'compression' AND status = 'active'"
        )
        transport = httpx.ASGITransport(app=service.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://registry"
        ) as client:
            health = await client.get("/healthz")
            listing = await client.get("/v1/baselines")
        assert health.status_code == 503
        assert "0 active baselines" in health.json()["violations"][0]
        assert listing.status_code == 503
        assert listing.json()["error"] == "registry_invariant_violation"
    finally:
        service.close()


def test_production_config_requires_absolute_state_and_canonical_ports(
    tmp_path,
) -> None:
    raw, _store = _configured_world(tmp_path)
    cfg = RegistryConfig.model_validate(raw["registry"])
    assert production_registry_problems(cfg) == []

    bad = cfg.model_copy(
        update={"db_path": Path("relative.db"), "http_port": 1, "metrics_port": 2}
    )
    assert production_registry_problems(bad) == [
        "registry.db_path must be an absolute writable production path",
        "registry.http_port must be 8720",
        "registry.metrics_port must be 9123",
    ]


def test_startup_refuses_an_existing_broken_ledger(tmp_path) -> None:
    raw, store = _configured_world(tmp_path)
    first = _service(raw, store)
    first.conn.execute(
        "UPDATE baselines SET status = 'superseded' WHERE track = 'upscaling'"
    )
    first.close()

    with pytest.raises(RegistryStartupError, match="0 active baselines"):
        _service(raw, store)


def test_automatic_promotion_posture_is_explicit_and_fail_closed(tmp_path) -> None:
    raw, store = _configured_world(tmp_path)
    raw["registry"]["automatic_promotion_enabled"] = True
    raw["registry"]["allow_disabled_automatic_promotion_for_testnet"] = False
    with pytest.raises(RegistryStartupError, match="requires an injected"):
        _service(raw, store)

    class Watcher:
        async def run(self, *, db_path, store, stopping) -> None:
            await stopping.wait()

    raw["registry"]["automatic_promotion_enabled"] = False
    with pytest.raises(RegistryStartupError, match="injected while"):
        BaselineRegistryService(
            raw,
            metrics_port=0,
            store=store,
            watcher=Watcher(),
            now=lambda: NOW,
        )

    raw["registry"]["automatic_promotion_enabled"] = True
    active = BaselineRegistryService(
        raw,
        metrics_port=0,
        store=store,
        watcher=Watcher(),
        now=lambda: NOW,
    )
    try:
        assert active.public_state()["automatic_promotion"] == {
            "enabled": True,
            "adapter_wired": True,
            "testnet_exception": False,
            "status": "active",
        }
    finally:
        active.close()
