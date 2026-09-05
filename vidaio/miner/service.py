"""Reference miner service: the miner-facing wire endpoints.

  POST /v1/task/artifact  metadata header + raw input -> raw output (CANONICAL)
  POST /v1/task   DEPRECATED local-path API, absent unless the explicit
  POST /task      miner.enable_legacy_path_routes local-test opt-in is true.
                  Production callers use /v1/task/artifact, which never accepts
                  or dereferences a caller-supplied path.
  GET  /warrant   -> {"track": <warrant_track>}  the TaskWarrant probe: which
                  pool THIS miner identity competes in (see below).

Honest remote task handling, in order: authenticate (when a token is configured),
decode a small bounded metadata header, validate the task id, admit against the
concurrency bound, stream the body to a private task directory under a hard byte
cap while hashing it, verify the input digest, route to the track backend, bound
the work by the request deadline, hash/size-bound the output, and stream the
output in the response. No validator/gateway path or URL is ever dereferenced.

The endpoint is PUBLIC, so the ingress is bounded on every axis an anonymous
caller could push (see MinerConfig for the knobs):

  * task_id is matched against TASK_ID_PATTERN (^[A-Za-z0-9_:-]{1,128}$ — colon
    and 128 chars are deliberate: validator ids are "<challenge_id>:<hotkey>")
    AND the resolved output
    directory is re-checked to be under work_dir. Two independent barriers on
    purpose: the pattern rejects `..`, absolute paths, separators and unicode
    look-alikes outright, and the realpath prefix check catches anything a
    symlinked work_dir or a future pattern relaxation could still let through.
  * a semaphore caps concurrent tasks; a saturated miner answers 429 rather than
    piling up ffmpeg processes.
  * receiving and durably staging a streamed request body has a server-owned
    wall-clock cap, independent of the caller's claimed task deadline.
  * oversized inputs and outputs are refused before they can consume unbounded
    disk or leave the service.
  * task dirs are swept (startup + periodically) so disk cannot grow forever —
    always at the configured TTL, so a restart never deletes a result the caller
    was still promised time to read.

TaskWarrant: one pool per miner identity. This reference binary supports BOTH
track backends, but an instance DECLARES the single pool it competes in
(`miner.warrant_track`) and the validator buckets it there; a miner that wants
to serve both pools registers two identities. The validator never defaults an
unclassified miner into a track (vidaio/validator/inference.py) — an unknown or
garbage warrant answer means the miner is skipped for the round, so the declared
value is validated against the known tracks at config load.

Typed errors (JSON body {"detail": {"code", "message"}}):
  401 unauthorized      (only when miner.api_token is configured)
  400 invalid_metadata | invalid_content_length | body_length_mismatch
  408 ingress_timeout
  409 task_conflict
  413 input_too_large
  415 unsupported_media_type
  422 invalid_task_id | input_not_found | input_digest_mismatch |
      unknown_track | bad_params
  429 busy              (max_concurrent_tasks reached)
  500 processing_failed
  504 deadline_exceeded  (the validator would score us absent anyway)
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from prometheus_client import CollectorRegistry, Counter, Histogram

from vidaio.core import section
from vidaio.miner.backends import (
    BackendError,
    BackendTimeoutError,
    FfmpegCompressBackend,
    FfmpegUpscaleBackend,
    MinerBackend,
    sha256_file,
)
from vidaio.miner.config import MinerConfig
from vidaio.miner.remote_gpu import RemoteGpuBackend
from vidaio.services import BaseService
from vidaio.services.artifact_auth import (
    ArtifactAuthColdStart,
    ArtifactAuthError,
    ArtifactAuthExpired,
    ArtifactAuthInvalid,
    ArtifactAuthMissing,
    ArtifactReplay,
    ArtifactReplayCacheFull,
    ArtifactRequestClaims,
    ArtifactServerAuth,
    ArtifactUnregisteredValidator,
    ArtifactWrongMiner,
)
from vidaio.services.protocol import (
    MINER_ARTIFACT_AUTH_VERSION,
    MINER_ARTIFACT_ROUTE,
    MINER_ARTIFACT_VERSION,
    MINER_ARTIFACT_VERSION_HEADER,
    MINER_OUTPUT_DIGEST_HEADER,
    MINER_PROCESSING_SECONDS_HEADER,
    MINER_TASK_ID_HEADER,
    MINER_TASK_METADATA_HEADER,
    MinerArtifactTaskRequest,
    MinerTaskMetadataError,
    MinerTaskRequest,
    MinerTaskResponse,
    decode_miner_task_metadata,
)

#: Task ids are OUR directory names, so they are restricted to an alphabet that
#: cannot express a path: no separators (`/`, `\`), no dots at all (so `.` and
#: `..` are unspellable, not merely filtered), no NULs, no whitespace, no
#: unicode. Anything else is a 422 before a single byte is written.
#:
#: `:` is IN because the wire contract already carries composite ids of the form
#: "<challenge_id>:<hotkey>" (vidaio.services.protocol callers) and a colon
#: cannot begin a path component on the POSIX hosts this runs on — the resolved
#: prefix check in resolve_task_dir is the barrier that does not depend on that
#: judgement. The length bound is 128 for the same reason: a uuid4 plus a
#: hotkey suffix does not fit in 64.
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_:-]{1,128}$")

#: Header carrying the optional shared secret (miner.api_token).
AUTH_HEADER = "x-miner-token"


class TaskDirEscape(ValueError):
    """A task id resolved to a directory outside work_dir."""


def resolve_task_dir(work_dir: Path, task_id: str) -> Path:
    """work_dir/<task_id>, proven to stay under work_dir.

    Belt and braces (the pattern check runs first, in the handler): the parent is
    resolved through symlinks and the candidate must be a child of it. A resolved
    prefix check — not a string prefix on the unresolved path — because work_dir
    itself may be a symlink and `..` segments only collapse after resolution.
    """
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise TaskDirEscape(f"task_id {task_id!r} is not a safe directory name")
    root = Path(work_dir).resolve()
    candidate = (root / task_id).resolve()
    if candidate == root or root not in candidate.parents:
        raise TaskDirEscape(f"task_id {task_id!r} escapes work_dir {root}")
    return candidate


def sweep_task_dirs(work_dir: Path, *, ttl_seconds: float, now: float) -> int:
    """Delete task dirs untouched for longer than ttl_seconds. Returns the count.

    Called once at startup (clearing what a crashed process left behind ONCE IT
    HAS AGED OUT — a restart must not shorten the TTL a live response already
    promised its caller) and then periodically. Only direct children of work_dir
    whose names are valid task ids are considered — the sweeper never follows a
    path it did not make.
    """
    root = Path(work_dir)
    removed = 0
    if not root.is_dir():
        return 0
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        if not TASK_ID_PATTERN.fullmatch(child.name):
            continue
        try:
            age = now - child.stat().st_mtime
        except OSError:
            continue
        if age < ttl_seconds:
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed += 1
    return removed


def _discard_task_dir(cfg: MinerConfig, out_dir: Path) -> None:
    """Drop a completed/failed remote task directory unless debugging retains it."""
    if cfg.retain_task_dirs:
        return
    shutil.rmtree(out_dir, ignore_errors=True)


class _DiscardingFileResponse(FileResponse):
    """FileResponse whose private task directory is removed on every exit.

    Starlette background tasks run after a successful response, but a client
    disconnect/send failure can interrupt before that point. Cleanup belongs in
    ``finally`` so completed output is not stranded until the TTL after an
    aborted download.
    """

    def __init__(self, *args: Any, cleanup: Callable[[], None], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._cleanup()


def _sha256_file_bounded(path: Path, max_bytes: int) -> tuple[str, int]:
    """Hash a backend output without ever reading past the configured cap."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"output crossed {max_bytes} bytes")
            h.update(chunk)
    return h.hexdigest(), size


class _Metrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.tasks = Counter(
            "vidaio_miner_tasks_total",
            "Tasks handled, by track and outcome",
            # outcome: ok | rejected | busy | unauthorized | timeout | failed
            ["track", "outcome"],
            registry=registry,
        )
        self.swept_dirs = Counter(
            "vidaio_miner_task_dirs_swept_total",
            "Task directories deleted by the retention sweeper",
            registry=registry,
        )
        self.processing_seconds = Histogram(
            "vidaio_miner_processing_seconds",
            "Backend processing wall-clock seconds",
            ["track"],
            registry=registry,
        )


def build_backends(cfg: MinerConfig) -> dict[str, MinerBackend]:
    if cfg.backend_mode == "remote_gpu":
        assert cfg.remote_gpu_url is not None
        assert cfg.remote_gpu_auth_token is not None
        shared = {
            "endpoint_url": cfg.remote_gpu_url,
            "auth_token": cfg.remote_gpu_auth_token.get_secret_value(),
            "solution_variant": cfg.remote_gpu_solution_variant,
            "max_output_bytes": cfg.max_output_bytes,
            "request_timeout_seconds": cfg.ffmpeg_timeout_seconds,
            "connect_timeout_seconds": cfg.remote_gpu_connect_timeout_seconds,
            "allow_cpu_fallback": cfg.remote_gpu_allow_cpu_fallback,
        }
        return {
            "compression": RemoteGpuBackend(track="compression", **shared),
            "upscaling": RemoteGpuBackend(track="upscaling", **shared),
        }
    return {
        "compression": FfmpegCompressBackend(
            cfg.ffmpeg_path,
            cfg.ffmpeg_timeout_seconds,
            crf=cfg.compress_crf,
            preset=cfg.compress_preset,
        ),
        "upscaling": FfmpegUpscaleBackend(
            cfg.ffmpeg_path,
            cfg.ffmpeg_timeout_seconds,
            crf=cfg.upscale_crf,
            preset=cfg.upscale_preset,
        ),
    }


def create_app(
    cfg: MinerConfig,
    metrics: _Metrics,
    *,
    artifact_auth: ArtifactServerAuth | None = None,
) -> FastAPI:
    app = FastAPI(title="vidaio-reference-miner", docs_url=None, redoc_url=None)
    if artifact_auth is not None:
        if not cfg.artifact_hotkey:
            raise ValueError("miner.artifact_hotkey is required with artifact v2 auth")
        if artifact_auth.signer.hotkey != cfg.artifact_hotkey:
            raise ValueError(
                "artifact response signer hotkey does not match miner.artifact_hotkey "
                f"({artifact_auth.signer.hotkey!r} != {cfg.artifact_hotkey!r})"
            )
    backends = build_backends(cfg)
    # Bounds the compute an anonymous caller can start. Created here (not in
    # Miner) so the app is self-contained for ASGI tests.
    slots = asyncio.Semaphore(cfg.max_concurrent_tasks)
    # Exposed for observability (and so tests can await actual slot acquisition
    # rather than racing on a fixed number of event-loop yields).
    app.state.task_slots = slots

    def _reject(track: str, status: int, code: str, message: str) -> HTTPException:
        outcome = {
            401: "unauthorized",
            408: "timeout",
            422: "rejected",
            429: "busy",
            504: "timeout",
        }.get(status, "failed")
        metrics.tasks.labels(track=track, outcome=outcome).inc()
        return HTTPException(status_code=status, detail={"code": code, "message": message})

    def _authorize(request: Request, track: str) -> None:
        """Shared-secret gate. No token configured = open (documented local mode)."""
        if cfg.api_token is None:
            return
        presented = request.headers.get(AUTH_HEADER, "")
        if not hmac.compare_digest(presented, cfg.api_token):
            raise _reject(
                track, 401, "unauthorized",
                f"this miner requires a shared secret in the {AUTH_HEADER} header",
            )

    def _backend_and_dir(
        task_id: str, track: str
    ) -> tuple[MinerBackend, Path]:
        backend = backends.get(track)
        if backend is None:
            raise _reject(track, 422, "unknown_track", f"no backend for {track!r}")
        try:
            out_dir = resolve_task_dir(cfg.work_dir, task_id)
        except TaskDirEscape as exc:
            raise _reject(track, 422, "invalid_task_id", str(exc)) from exc
        return backend, out_dir

    @app.get("/warrant")
    async def warrant() -> dict[str, str]:
        """TaskWarrant probe: the single pool this miner identity competes in."""
        return {"track": cfg.warrant_track}

    async def task(req: MinerTaskRequest, request: Request) -> MinerTaskResponse:
        _authorize(request, req.track)
        backend, out_dir = _backend_and_dir(req.task_id, req.track)
        input_path = Path(req.input_path)
        if not input_path.is_file():
            raise _reject(req.track, 422, "input_not_found", f"no file at {input_path}")
        try:
            input_bytes = input_path.stat().st_size
        except OSError as exc:
            raise _reject(req.track, 422, "input_not_found", str(exc)) from exc
        if input_bytes > cfg.max_input_bytes:
            raise _reject(
                req.track, 422, "input_too_large",
                f"input is {input_bytes} bytes, this miner accepts at most "
                f"{cfg.max_input_bytes}",
            )
        # Admission control BEFORE any hashing or encoding: a saturated miner
        # says so immediately instead of queueing work it cannot start.
        if slots.locked():
            raise _reject(
                req.track, 429, "busy",
                f"all {cfg.max_concurrent_tasks} task slots are in use",
            )
        async with slots:
            return await _run_task(req, backend, input_path, out_dir)

    async def _receive_remote_input(
        request: Request,
        metadata: MinerArtifactTaskRequest,
        out_dir: Path,
    ) -> Path:
        declared_text = request.headers.get("content-length")
        declared: int | None = None
        if declared_text is not None:
            try:
                declared = int(declared_text)
            except ValueError as exc:
                raise _reject(
                    metadata.track,
                    400,
                    "invalid_content_length",
                    f"invalid Content-Length {declared_text!r}",
                ) from exc
            if declared < 0:
                raise _reject(
                    metadata.track,
                    400,
                    "invalid_content_length",
                    f"invalid Content-Length {declared_text!r}",
                )
            if declared > cfg.max_input_bytes:
                raise _reject(
                    metadata.track,
                    413,
                    "input_too_large",
                    f"input declares {declared} bytes, this miner accepts at most "
                    f"{cfg.max_input_bytes}",
                )

        out_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_dir.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError as exc:
            raise _reject(
                metadata.track,
                409,
                "task_conflict",
                f"task {metadata.task_id!r} already exists",
            ) from exc
        partial = out_dir / ".input.part"
        final = out_dir / "input.media"
        h = hashlib.sha256()
        received = 0
        try:
            with partial.open("xb") as sink:
                os.chmod(partial, 0o600)
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > cfg.max_input_bytes:
                        raise _reject(
                            metadata.track,
                            413,
                            "input_too_large",
                            f"input crossed {cfg.max_input_bytes} bytes",
                        )
                    h.update(chunk)
                    sink.write(chunk)
                sink.flush()
                os.fsync(sink.fileno())
            if declared is not None and received != declared:
                raise _reject(
                    metadata.track,
                    400,
                    "body_length_mismatch",
                    f"Content-Length was {declared}, received {received}",
                )
            if received < 1:
                raise _reject(
                    metadata.track, 422, "input_not_found", "request body is empty"
                )
            actual = h.hexdigest()
            if actual != metadata.input_digest:
                raise _reject(
                    metadata.track,
                    422,
                    "input_digest_mismatch",
                    f"input sha256 {actual} != declared {metadata.input_digest}",
                )
            os.replace(partial, final)
            return final
        except BaseException:
            _discard_task_dir(cfg, out_dir)
            raise

    async def artifact_task(request: Request):  # type: ignore[no-untyped-def]
        # Authentication precedes metadata parsing and body reads: an anonymous
        # caller learns nothing and cannot make us buffer/hash a byte.
        _authorize(request, "unknown")
        artifact_version = request.headers.get(MINER_ARTIFACT_VERSION_HEADER)
        if artifact_version not in {
            MINER_ARTIFACT_VERSION,
            MINER_ARTIFACT_AUTH_VERSION,
        }:
            raise _reject(
                "unknown",
                400,
                "invalid_metadata",
                f"{MINER_ARTIFACT_VERSION_HEADER} must be "
                f"{MINER_ARTIFACT_AUTH_VERSION} (or explicitly enabled dev v1)",
            )
        if (
            artifact_version == MINER_ARTIFACT_VERSION
            and not cfg.allow_unsigned_artifact_v1
        ):
            raise _reject(
                "unknown",
                426,
                "artifact_v2_required",
                "unsigned artifact v1 is disabled; use hotkey-authenticated v2",
            )
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != (
            "application/octet-stream"
        ):
            raise _reject(
                "unknown",
                415,
                "unsupported_media_type",
                "remote miner tasks require application/octet-stream",
            )
        try:
            metadata = decode_miner_task_metadata(
                request.headers.get(MINER_TASK_METADATA_HEADER)
            )
        except MinerTaskMetadataError as exc:
            raise _reject("unknown", 400, "invalid_metadata", str(exc)) from exc
        auth_claims: ArtifactRequestClaims | None = None
        if artifact_version == MINER_ARTIFACT_AUTH_VERSION:
            if artifact_auth is None:
                raise _reject(
                    metadata.track,
                    503,
                    "artifact_auth_unavailable",
                    "artifact v2 signer/verifier is not configured",
                )
            raw_length = request.headers.get("content-length")
            try:
                content_length = int(raw_length or "")
            except ValueError as exc:
                raise _reject(
                    metadata.track,
                    400,
                    "invalid_content_length",
                    "artifact v2 requires an integer Content-Length",
                ) from exc
            try:
                # Registration may refresh a chain snapshot. Keep that bounded
                # synchronous seam off the ASGI event loop; no body has been read.
                auth_claims = await asyncio.to_thread(
                    artifact_auth.verify_request,
                    request.headers,
                    metadata,
                    content_length=content_length,
                )
            except ArtifactReplay as exc:
                raise _reject(metadata.track, 409, "artifact_replay", str(exc)) from exc
            except ArtifactReplayCacheFull as exc:
                raise _reject(
                    metadata.track, 503, "artifact_auth_capacity", str(exc)
                ) from exc
            except ArtifactWrongMiner as exc:
                raise _reject(metadata.track, 403, "wrong_miner", str(exc)) from exc
            except ArtifactUnregisteredValidator as exc:
                raise _reject(
                    metadata.track, 403, "unregistered_validator", str(exc)
                ) from exc
            except ArtifactAuthExpired as exc:
                raise _reject(metadata.track, 401, "expired_signature", str(exc)) from exc
            except ArtifactAuthColdStart as exc:
                raise _reject(
                    metadata.track, 425, "artifact_auth_starting", str(exc)
                ) from exc
            except (ArtifactAuthMissing, ArtifactAuthInvalid, ArtifactAuthError) as exc:
                raise _reject(metadata.track, 401, "invalid_signature", str(exc)) from exc
            if metadata.track != cfg.warrant_track:
                raise _reject(
                    metadata.track,
                    422,
                    "warrant_track_mismatch",
                    f"this miner identity serves {cfg.warrant_track!r}, not "
                    f"{metadata.track!r}",
                )
        backend, out_dir = _backend_and_dir(metadata.task_id, metadata.track)
        if slots.locked():
            raise _reject(
                metadata.track,
                429,
                "busy",
                f"all {cfg.max_concurrent_tasks} task slots are in use",
            )
        request_started = time.perf_counter()
        async with slots:
            ingress_remaining = cfg.artifact_ingress_timeout_seconds - (
                time.perf_counter() - request_started
            )
            try:
                if ingress_remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(ingress_remaining):
                    input_path = await _receive_remote_input(
                        request, metadata, out_dir
                    )
            except TimeoutError as exc:
                _discard_task_dir(cfg, out_dir)
                raise _reject(
                    metadata.track,
                    408,
                    "ingress_timeout",
                    "input was not completely received and staged within the "
                    f"server limit of {cfg.artifact_ingress_timeout_seconds}s",
                ) from exc
            remaining = metadata.deadline_seconds - (time.perf_counter() - request_started)
            if remaining <= 0:
                _discard_task_dir(cfg, out_dir)
                raise _reject(
                    metadata.track,
                    504,
                    "deadline_exceeded",
                    "deadline exhausted while receiving input",
                )
            local = metadata.as_local_request(str(input_path)).model_copy(
                update={"deadline_seconds": remaining}
            )
            try:
                completed = await _run_task(local, backend, input_path, out_dir)
            except BaseException:
                _discard_task_dir(cfg, out_dir)
                raise
        output = Path(completed.output_path)
        processing_value = (
            ""
            if completed.processing_seconds is None
            else format(completed.processing_seconds, ".12g")
        )
        headers = {
            MINER_ARTIFACT_VERSION_HEADER: artifact_version,
            MINER_TASK_ID_HEADER: completed.task_id,
            MINER_OUTPUT_DIGEST_HEADER: completed.output_digest,
        }
        if processing_value:
            headers[MINER_PROCESSING_SECONDS_HEADER] = processing_value
        if auth_claims is not None:
            assert artifact_auth is not None
            try:
                headers.update(
                    artifact_auth.response_headers(
                        auth_claims,
                        metadata,
                        output_digest=completed.output_digest,
                        output_size=output.stat().st_size,
                        processing_seconds=processing_value,
                    )
                )
            except Exception as exc:
                _discard_task_dir(cfg, out_dir)
                raise _reject(
                    metadata.track,
                    503,
                    "artifact_signing_failed",
                    f"could not sign artifact v2 response: {exc}",
                ) from exc
        return _DiscardingFileResponse(
            output,
            media_type="application/octet-stream",
            headers=headers,
            cleanup=lambda: _discard_task_dir(cfg, out_dir),
        )

    async def _run_task(
        req: MinerTaskRequest, backend: MinerBackend, input_path: Path, out_dir: Path
    ) -> MinerTaskResponse:
        start = time.perf_counter()
        deadline = req.deadline_seconds

        def remaining() -> float:
            return deadline - (time.perf_counter() - start)

        actual = await asyncio.to_thread(sha256_file, input_path)
        if actual != req.input_digest:
            raise _reject(
                req.track, 422, "input_digest_mismatch",
                f"input sha256 {actual} != declared {req.input_digest}",
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "output.mp4"
        try:
            budget = remaining()
            if budget <= 0:
                raise BackendTimeoutError("deadline exhausted before processing began")
            # Two bounds on purpose: wait_for frees this handler at the deadline
            # even if the subprocess lingers; the backend's own subprocess timeout
            # (capped to the same budget) reaps the process itself.
            await asyncio.wait_for(
                asyncio.to_thread(
                    backend.process, str(input_path), str(output_path), req.params,
                    timeout=budget,
                ),
                timeout=budget,
            )
        except (TimeoutError, BackendTimeoutError) as exc:
            _discard_task_dir(cfg, out_dir)
            raise _reject(
                req.track, 504, "deadline_exceeded",
                f"could not finish within {deadline}s: {exc}",
            ) from exc
        except BackendError as exc:
            _discard_task_dir(cfg, out_dir)
            msg = str(exc)
            if "upscale_factor" in msg:
                raise _reject(req.track, 422, "bad_params", msg) from exc
            raise _reject(req.track, 500, "processing_failed", msg) from exc
        except BaseException:
            _discard_task_dir(cfg, out_dir)
            raise
        if not output_path.is_file():
            _discard_task_dir(cfg, out_dir)
            raise _reject(req.track, 500, "processing_failed", "backend produced no output")

        try:
            digest, _ = await asyncio.to_thread(
                _sha256_file_bounded, output_path, cfg.max_output_bytes
            )
        except ValueError as exc:
            _discard_task_dir(cfg, out_dir)
            raise _reject(
                req.track,
                500,
                "output_too_large",
                f"backend output exceeds {cfg.max_output_bytes} bytes",
            ) from exc
        elapsed = time.perf_counter() - start
        metrics.tasks.labels(track=req.track, outcome="ok").inc()
        metrics.processing_seconds.labels(track=req.track).observe(elapsed)
        # Internal result model shared by both handlers. The canonical remote
        # handler streams this file then deletes the task dir in a response
        # background task; only the deprecated JSON path handler needs the TTL.
        return MinerTaskResponse(
            task_id=req.task_id,
            output_path=str(output_path),
            output_digest=digest,
            processing_seconds=elapsed,
        )

    # Production/default route: byte streams in both directions, no shared path.
    app.post(MINER_ARTIFACT_ROUTE, name="artifact_task")(artifact_task)
    # Local/backward-compatibility routes are an explicit opt-in. They accept an
    # input_path and therefore must be completely absent from a public miner,
    # especially when the permissionless streamed endpoint has no shared token.
    if cfg.enable_legacy_path_routes:
        app.post(
            "/v1/task",
            response_model=MinerTaskResponse,
            name="task_legacy_local_path",
            deprecated=True,
            description="DEPRECATED local/shared-filesystem path contract.",
        )(task)
        app.post(
            "/task",
            response_model=MinerTaskResponse,
            name="task_deprecated_alias",
            deprecated=True,
            description="DEPRECATED alias of POST /v1/task — identical behavior.",
        )(task)

    return app


class Miner(BaseService):
    name = "reference-miner"

    def __init__(
        self,
        raw_config: dict[str, Any],
        *,
        artifact_auth: ArtifactServerAuth | None = None,
    ) -> None:
        cfg = section(raw_config, "miner", MinerConfig)
        super().__init__(raw_config, metrics_port=cfg.metrics_port)
        self.cfg = cfg
        self.metrics = _Metrics(self.health.registry)
        self.app = create_app(self.cfg, self.metrics, artifact_auth=artifact_auth)
        Path(self.cfg.work_dir).mkdir(parents=True, exist_ok=True)
        #: False once the HTTP server task has stopped unexpectedly — a bound
        #: failure must NOT leave a live process reporting healthy with no API.
        self._api_serving = True
        if self.cfg.backend_mode == "ffmpeg":
            self.health.register_check(
                "ffmpeg", lambda: shutil.which(self.cfg.ffmpeg_path) is not None
            )
        else:
            # Passive service health must not cold-start (and bill) the GPU.
            # ``the development-tree Modal preflight`` performs the explicit live
            # /healthz probe before advertisement and during a soak.
            self.health.register_check(
                "remote_gpu_configured",
                lambda: bool(
                    self.cfg.remote_gpu_url
                    and self.cfg.remote_gpu_auth_token is not None
                ),
            )
        self.health.register_check("work_dir", lambda: Path(self.cfg.work_dir).is_dir())
        self.health.register_check("http_api", lambda: self._api_serving)
        # Startup sweep, at the CONFIGURED TTL — never force. Canonical remote
        # responses normally delete their directories immediately after their
        # stream (including disconnects); this catches crash residue and retains
        # the promised grace window for deprecated local JSON-path callers.
        self.sweep_task_dirs()

    def sweep_task_dirs(self, *, force: bool = False) -> int:
        """Delete task dirs older than the configured TTL. Returns the count.

        `force` ignores the TTL and sweeps every task dir. It is an explicit
        admin/test path ONLY: it deletes outputs whose advertised TTL has not
        expired, so nothing on the normal service lifecycle (startup included)
        may call it.
        """
        if self.cfg.retain_task_dirs:
            return 0
        removed = sweep_task_dirs(
            Path(self.cfg.work_dir),
            ttl_seconds=0.0 if force else self.cfg.task_dir_ttl_seconds,
            now=time.time(),
        )
        if removed:
            self.metrics.swept_dirs.inc(removed)
            self.log.info("swept stale task dirs", extra={"removed": removed})
        return removed

    def _create_http_server(self) -> uvicorn.Server:
        """Seam: tests substitute a server whose serve() fails to bind."""
        return uvicorn.Server(
            uvicorn.Config(
                self.app,
                host=self.cfg.http_host,
                port=self.cfg.http_port,
                log_config=None,
                access_log=False,
            )
        )

    async def _sweep_loop(self) -> None:
        while not self.stopping.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self.stopping.wait(), self.cfg.task_sweep_interval_seconds
                )
                return
            try:
                self.sweep_task_dirs()
            except Exception:
                self.log.exception("task dir sweep failed")

    async def run(self) -> None:
        server = self._create_http_server()
        serve_task = asyncio.create_task(server.serve(), name="miner-http")
        sweep_task = asyncio.create_task(self._sweep_loop(), name="miner-sweep")
        stop_task = asyncio.create_task(self.stopping.wait(), name="miner-stop")
        try:
            # The serve task is MONITORED, not fired and forgotten: if it returns
            # or raises (a bind failure, say) health flips and the service dies
            # FATALLY (non-zero exit), so a supervisor restarts it. Returning
            # normally here would be exit 0 = "deliberate stop" and the miner
            # would stay down forever.
            await asyncio.wait(
                {serve_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if serve_task.done() and not self.stopping.is_set():
                self._api_serving = False
                exc = serve_task.exception() if not serve_task.cancelled() else None
                self.fail_fatal(
                    "http server exited unexpectedly; the miner has no API"
                    f" (error={repr(exc) if exc is not None else None})"
                )
        finally:
            server.should_exit = True
            self.request_stop()
            for pending in (serve_task, sweep_task):
                with contextlib.suppress(Exception):
                    await pending
            stop_task.cancel()


if __name__ == "__main__":
    # Keep ``python -m vidaio.miner.service`` as an operator convenience, but route
    # it through the same role-scoped production guard as the release container.
    # Calling ``BaseService.main`` here used to bypass every Bittensor identity,
    # transport and legacy-route invariant enforced by service_entrypoint.py.
    from scripts.service_entrypoint import main as guarded_service_main

    guarded_service_main(["reference-miner", *sys.argv[1:]])
