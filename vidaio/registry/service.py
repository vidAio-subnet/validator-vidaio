"""Production service owning the schema-v14 executable-baseline registry.

The HTTP surface is intentionally read-only. Database mutation occurs only at
startup (atomic, content-verified v0 seeding) or through an injected internal
verified-CROWN watcher. No request can promote or roll back an executable.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import Gauge

from vidaio.audit.config import AuditConfig
from vidaio.audit.store import (
    ArtifactRef,
    AuditStore,
    IntegrityError,
    SEALED_KINDS,
    make_store,
)
from vidaio.core import connect, connect_read_only, section
from vidaio.registry.baseline import (
    BaselineRecord,
    GenesisBaseline,
    SUPPORTED_TRACKS,
    baseline_invariant_violations,
    current_baseline,
    pending_promotion,
    seed_genesis_baselines,
)
from vidaio.registry.config import RegistryConfig
from vidaio.registry.registry import migrate
from vidaio.services.base import BaseService

REGISTRY_API_KIND = "vidaio.baseline-registry.v1"
REGISTRY_SCHEMA_VERSION = 14


class RegistryStartupError(RuntimeError):
    """The service cannot establish an audit-complete baseline ledger."""


class BaselinePromotionWatcher(Protocol):
    """Internal chain-proof watcher seam; deliberately absent from the HTTP API.

    A future concrete watcher opens its own database connections from ``db_path``
    and applies only the existing verified-CROWN promotion pipeline. Returning
    before ``stopping`` is set is a fatal service failure.
    """

    async def run(
        self,
        *,
        db_path: Path,
        store: AuditStore,
        stopping: asyncio.Event,
    ) -> None: ...


def _verify_ref(store: AuditStore, ref: ArtifactRef, *, label: str) -> None:
    digest = hashlib.sha256()
    size = 0
    try:
        with contextlib.closing(store.open_stream(ref)) as stream:
            while chunk := stream.read(1 << 20):
                size += len(chunk)
                if size > ref.byte_size:
                    raise IntegrityError(
                        f"{label} exceeds configured size {ref.byte_size}"
                    )
                digest.update(chunk)
    except (FileNotFoundError, OSError, IntegrityError) as exc:
        raise RegistryStartupError(
            f"{label} is absent or corrupt in the configured audit store: {exc}"
        ) from exc
    if size != ref.byte_size or digest.hexdigest() != ref.digest:
        raise RegistryStartupError(
            f"{label} does not match its configured content address/size"
        )


def verify_genesis_archives(
    store: AuditStore, seeds: Sequence[GenesisBaseline]
) -> dict[str, dict[str, Any]]:
    """Read and hash every configured v0 archive without mutating storage."""

    verified: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        _verify_ref(store, seed.artifact, label=f"{seed.track} v0 executable")
        _verify_ref(store, seed.provenance, label=f"{seed.track} v0 provenance")
        verified[seed.track] = {
            "artifact_digest": seed.artifact.digest,
            "artifact_bytes": seed.artifact.byte_size,
            "provenance_digest": seed.provenance.digest,
            "provenance_bytes": seed.provenance.byte_size,
            "image_digest": seed.image_digest,
        }
    return verified


def verify_active_archives(
    store: AuditStore, records: Sequence[BaselineRecord]
) -> None:
    """Verify every active executable/provenance ref and sealed release marker."""

    for record in records:
        artifact = record.artifact_ref()
        provenance = record.provenance_ref()
        _verify_ref(
            store, artifact, label=f"{record.track} v{record.version} executable"
        )
        _verify_ref(
            store,
            provenance,
            label=f"{record.track} v{record.version} provenance",
        )
        if artifact.kind in SEALED_KINDS and not store.is_released(artifact):
            raise RegistryStartupError(
                f"{record.track} v{record.version} executable has no public release"
            )


def _record_json(record: BaselineRecord) -> dict[str, Any]:
    return {
        "track": record.track,
        "version": record.version,
        "artifact_digest": record.artifact_digest,
        "artifact_kind": record.artifact_kind,
        "artifact_bytes": record.artifact_bytes,
        "image_digest": record.image_digest,
        "provenance_digest": record.provenance_digest,
        "provenance_kind": record.provenance_kind,
        "provenance_bytes": record.provenance_bytes,
        "repo_url": record.repo_url,
        "commit_sha": record.commit_sha,
        "tree_sha": record.tree_sha,
        "source_kind": record.source_kind,
        "source_epoch_id": record.source_epoch_id,
        "source_snapshot_digest": record.source_snapshot_digest,
        "source_anchor_block": record.source_anchor_block,
        "source_anchor_digest": record.source_anchor_digest,
        "source_competition_id": record.source_competition_id,
        "source_cycle": record.source_cycle,
        "winner_uid": record.winner_uid,
        "winner_hotkey": record.winner_hotkey,
        "winner_score": record.winner_score,
        "winner_margin": record.winner_margin,
        "compared_baseline_version": record.compared_baseline_version,
        "compared_baseline_score": record.compared_baseline_score,
        "compared_baseline_digest": record.compared_baseline_digest,
        "reinstated_version": record.reinstated_version,
        "activated_at": record.activated_at.isoformat(),
    }


class BaselineRegistryService(BaseService):
    """Persistent baseline-ledger owner with a public read-only API."""

    name = "baseline-registry"

    def __init__(
        self,
        raw_config: dict[str, Any],
        *,
        metrics_port: int | None = None,
        store: AuditStore | None = None,
        watcher: BaselinePromotionWatcher | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        cfg = section(raw_config, "registry", RegistryConfig)
        super().__init__(
            raw_config,
            metrics_port=metrics_port if metrics_port is not None else cfg.metrics_port,
        )
        self.config = cfg
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._watcher = watcher
        if cfg.automatic_promotion_enabled and watcher is None:
            raise RegistryStartupError(
                "registry.automatic_promotion_enabled=true requires an injected "
                "verified-CROWN BaselinePromotionWatcher"
            )
        if not cfg.automatic_promotion_enabled and watcher is not None:
            raise RegistryStartupError(
                "an automatic-promotion watcher was injected while "
                "registry.automatic_promotion_enabled=false"
            )
        if (
            cfg.automatic_promotion_enabled
            and cfg.allow_disabled_automatic_promotion_for_testnet
        ):
            raise RegistryStartupError(
                "the disabled-automatic-promotion testnet exception cannot be set "
                "when automatic promotion is enabled"
            )
        self._store = store or make_store(section(raw_config, "audit", AuditConfig))
        self._http_api_ok = True
        self._archives_ok = False
        self._conn: sqlite3.Connection | None = None

        try:
            seeds = cfg.genesis_seeds()
            self._conn = connect(cfg.db_path)
            migrate(self._conn)
            seed_genesis_baselines(self._conn, self._store, seeds, self._now())
            violations = baseline_invariant_violations(self._conn)
            if violations:
                raise RegistryStartupError("; ".join(violations))
            records = self._active_records()
            verify_active_archives(self._store, records)
            self._archives_ok = True
        except BaseException:
            self.close()
            raise

        self.health.register_check("http_api", lambda: self._http_api_ok)
        self.health.register_check("database", lambda: self._conn is not None)
        self.health.register_check("archives", lambda: self._archives_ok)
        self.health.register_check(
            "invariants", lambda: not self._invariant_violations()
        )
        registry = self.health.registry
        self._m_active = Gauge(
            "vidaio_registry_active_baselines",
            "Active executable baselines by protocol track",
            ["track"],
            registry=registry,
        )
        self._m_pending = Gauge(
            "vidaio_registry_pending_promotions",
            "Unresolved verified-CROWN baseline promotion latches by track",
            ["track"],
            registry=registry,
        )
        self._m_promotion_enabled = Gauge(
            "vidaio_registry_automatic_promotion_enabled",
            "1 only when verified-CROWN automatic promotion is configured",
            registry=registry,
        )
        self._m_promotion_adapter_wired = Gauge(
            "vidaio_registry_automatic_promotion_adapter_wired",
            "1 only when the internal verified chain watcher is injected",
            registry=registry,
        )
        self._m_promotion_enabled.set(
            1.0 if self.config.automatic_promotion_enabled else 0.0
        )
        self._m_promotion_adapter_wired.set(1.0 if self._watcher is not None else 0.0)
        self._refresh_metrics()
        self.app = self._build_app()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RegistryStartupError("baseline registry database is closed")
        return self._conn

    def _active_records(self) -> list[BaselineRecord]:
        records: list[BaselineRecord] = []
        for track in SUPPORTED_TRACKS:
            record = current_baseline(self.conn, track)
            if record is None:
                raise RegistryStartupError(f"{track} has no active baseline")
            records.append(record)
        return records

    def _invariant_violations(self) -> list[str]:
        if self._conn is None:
            return ["registry database is closed"]
        # HealthServer evaluates checks on its own request thread.  SQLite's
        # default connection is intentionally thread-affine, so asking that
        # thread to reuse the service loop's writer makes a healthy registry
        # look degraded.  Read the live WAL through a short-lived query-only
        # handle instead; API calls use the same path and therefore observe the
        # exact same committed invariant state as the health endpoint.
        try:
            with connect_read_only(self.config.db_path) as conn:
                return baseline_invariant_violations(conn)
        except sqlite3.Error as exc:
            return [f"registry database read failed: {type(exc).__name__}: {exc}"]

    def _refresh_metrics(self) -> None:
        for track in SUPPORTED_TRACKS:
            self._m_active.labels(track).set(
                1.0 if current_baseline(self.conn, track) is not None else 0.0
            )
            self._m_pending.labels(track).set(
                1.0 if pending_promotion(self.conn, track) is not None else 0.0
            )

    def public_state(self) -> dict[str, Any]:
        violations = self._invariant_violations()
        if violations:
            raise RegistryStartupError("; ".join(violations))
        self._refresh_metrics()
        return {
            "kind": REGISTRY_API_KIND,
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "automatic_promotion": self._promotion_state(),
            "baselines": {
                record.track: _record_json(record) for record in self._active_records()
            },
            "pending_promotions": {
                track: {
                    "competition_id": latch.competition_id,
                    "cycle": latch.cycle,
                    "snapshot_digest": latch.snapshot_digest,
                    "winner_uid": latch.winner_uid,
                    "winner_hotkey": latch.winner_hotkey,
                    "latched_at": latch.latched_at.isoformat(),
                }
                for track in SUPPORTED_TRACKS
                if (latch := pending_promotion(self.conn, track)) is not None
            },
        }

    def _promotion_state(self) -> dict[str, Any]:
        enabled = self.config.automatic_promotion_enabled
        wired = self._watcher is not None
        if enabled and wired:
            status = "active"
        elif self.config.allow_disabled_automatic_promotion_for_testnet:
            status = "disabled_testnet_exception"
        else:
            status = "disabled"
        return {
            "enabled": enabled,
            "adapter_wired": wired,
            "testnet_exception": (
                self.config.allow_disabled_automatic_promotion_for_testnet
            ),
            "status": status,
        }

    def _build_app(self) -> FastAPI:
        app = FastAPI(
            title="vidaio baseline registry",
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )

        @app.get("/healthz")
        async def healthz() -> JSONResponse:
            violations = self._invariant_violations()
            if violations:
                return JSONResponse(
                    status_code=503,
                    content={
                        "service": self.name,
                        "status": "degraded",
                        "schema_version": REGISTRY_SCHEMA_VERSION,
                        "violations": violations,
                    },
                )
            records = self._active_records()
            return JSONResponse(
                {
                    "service": self.name,
                    "status": "ok",
                    "schema_version": REGISTRY_SCHEMA_VERSION,
                    "active_versions": {
                        record.track: record.version for record in records
                    },
                    "archives_verified": self._archives_ok,
                    "automatic_promotion": self._promotion_state(),
                }
            )

        @app.get("/v1/baselines")
        async def baselines() -> JSONResponse:
            try:
                return JSONResponse(self.public_state())
            except RegistryStartupError as exc:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "registry_invariant_violation",
                        "message": str(exc),
                    },
                )

        return app

    def _on_task_exit(self, task: asyncio.Task[Any], *, component: str) -> None:
        error: BaseException | None = None
        try:
            error = task.exception()
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            pass
        detail = "" if error is None else f"{type(error).__name__}: {error}"
        if component == "http_api":
            self._http_api_ok = False
        self.fail_fatal(f"baseline registry {component} exited unexpectedly ({detail})")

    async def run(self) -> None:
        server = uvicorn.Server(
            uvicorn.Config(
                self.app,
                host=self.config.http_host,
                port=self.config.http_port,
                log_level="warning",
            )
        )

        async def _serve() -> None:
            try:
                await server.serve()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                raise RuntimeError(
                    f"uvicorn exited: {type(exc).__name__}: {exc}"
                ) from exc

        api = asyncio.create_task(_serve(), name="baseline-registry-http")
        stop = asyncio.create_task(self.stopping.wait(), name="baseline-registry-stop")
        tasks: set[asyncio.Task[Any]] = {api, stop}
        watcher_task: asyncio.Task[Any] | None = None
        if self._watcher is not None:
            watcher_task = asyncio.create_task(
                self._watcher.run(
                    db_path=self.config.db_path,
                    store=self._store,
                    stopping=self.stopping,
                ),
                name="baseline-registry-crown-watcher",
            )
            tasks.add(watcher_task)
        try:
            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            if not self.stopping.is_set():
                if api in done:
                    self._on_task_exit(api, component="http_api")
                elif watcher_task is not None and watcher_task in done:
                    self._on_task_exit(watcher_task, component="promotion_watcher")
        finally:
            server.should_exit = True
            stop.cancel()
            if watcher_task is not None:
                watcher_task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


__all__ = [
    "REGISTRY_API_KIND",
    "REGISTRY_SCHEMA_VERSION",
    "BaselinePromotionWatcher",
    "BaselineRegistryService",
    "RegistryStartupError",
    "verify_active_archives",
    "verify_genesis_archives",
]
