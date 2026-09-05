"""Fresh Modal GPU transform worker for vidaio-next inference miners.

Deployments MUST use a new ``vidaio-next-*`` app name and secret. This file does
not look up Volumes, Dicts, Queues, existing Functions, or any other persistent
resource. The only named dependency is the explicitly supplied fresh auth
secret.

The public Bittensor endpoint remains ``vidaio.miner.service`` on a small CPU
host: it authenticates validator requests and signs responses. This private
Modal endpoint performs only the CUDA media transform. Economic scoring and
all auditor recomputation stay on the canonical CPU scorer.
"""

import asyncio
import hashlib
import hmac
import importlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

RESOURCE_PREFIX = "vidaio-next-"
FRESH_CREATION_CONFIRMATION = "CREATE_FRESH_VIDAIO_NEXT_MODAL_RESOURCES"
_RESOURCE_RE = re.compile(r"^vidaio-next-[a-zA-Z0-9._-]{6,52}$")


def _required_fresh_name(environment_variable: str, *, what: str) -> str:
    value = os.environ.get(environment_variable, "").strip()
    if not _RESOURCE_RE.fullmatch(value) or len(value) >= 64:
        raise ValueError(
            f"{what} requires an explicit fresh vidaio-next-* name shorter than "
            f"64 characters in {environment_variable}; there is no reusable default"
        )
    return value


def _require_local_creation_authorization() -> None:
    """Reject every local construction without the create-only acknowledgement."""
    if (
        os.environ.get("VIDAIO_NEXT_FRESH_CREATION_CONFIRMATION", "")
        != FRESH_CREATION_CONFIRMATION
    ):
        raise ValueError(
            "Modal miner deployment is disabled until "
            "VIDAIO_NEXT_FRESH_CREATION_CONFIRMATION equals "
            f"{FRESH_CREATION_CONFIRMATION!r}"
        )


# Modal imports this source twice: once on the trusted deployment client and again
# inside each remote Function container. The create-only acknowledgement authorizes
# the first operation; it is deliberately not a runtime secret and therefore is not
# present during the second import. MODAL_IS_REMOTE is a platform-reserved hint that
# lets the remote import reach the SDK. modal.is_local() remains the authoritative
# boundary below, so setting the hint on an operator host cannot bypass authorization.
_REMOTE_RUNTIME_HINT = os.environ.get("MODAL_IS_REMOTE") == "1"
if not _REMOTE_RUNTIME_HINT:
    # Preserve the useful fail-closed error on an operator host that has not even
    # installed the optional Modal dependency.
    _require_local_creation_authorization()

modal = importlib.import_module("modal")
_IS_LOCAL = bool(modal.is_local())

if _IS_LOCAL:
    _require_local_creation_authorization()
    DEPLOYMENT_LABEL = _required_fresh_name(
        "VIDAIO_NEXT_DEPLOYMENT_LABEL", what="Modal deployment label"
    )
    SECRET_NAME = _required_fresh_name(
        "VIDAIO_NEXT_MODAL_SECRET_NAME", what="Modal auth secret"
    )
    if DEPLOYMENT_LABEL == SECRET_NAME:
        raise ValueError("Modal deployment and auth-secret names must be distinct")
    SOLUTION_VARIANT = os.environ.get(
        "VIDAIO_NEXT_SOLUTION_VARIANT", "balanced"
    ).strip()
    if SOLUTION_VARIANT not in {"quality", "balanced", "compact", "premium"}:
        raise ValueError(
            "solution variant must be quality, balanced, compact, or premium"
        )
    GPU_TYPE = os.environ.get("VIDAIO_NEXT_MODAL_GPU", "L4").strip()
    MAX_CONTAINERS = int(os.environ.get("VIDAIO_NEXT_MODAL_MAX_CONTAINERS", "4"))
    SCALEDOWN_WINDOW_SECONDS = int(
        os.environ.get("VIDAIO_NEXT_MODAL_SCALEDOWN_WINDOW_SECONDS", "120")
    )
    if not GPU_TYPE or not 1 <= MAX_CONTAINERS <= 100:
        raise ValueError(
            "Modal GPU type must be non-empty and max containers must be 1..100"
        )
    if not 1 <= SCALEDOWN_WINDOW_SECONDS <= 3600:
        raise ValueError("Modal scaledown window must be between 1 and 3600 seconds")
    _AUTH_SECRET = modal.Secret.from_name(SECRET_NAME)
else:
    # Function decorators are re-evaluated when Modal imports the source remotely.
    # The deployed definition was already bound to the exact local names/resources;
    # construct only inert placeholders here and perform no named lookup. This is the
    # pattern Modal documents for globals that differ between local and remote import.
    DEPLOYMENT_LABEL = "vidaio-next-remote-runtime"
    SECRET_NAME = "vidaio-next-remote-auth-placeholder"
    SOLUTION_VARIANT = "balanced"
    GPU_TYPE = "L4"
    MAX_CONTAINERS = 1
    SCALEDOWN_WINDOW_SECONDS = 120
    _AUTH_SECRET = modal.Secret.from_dict({})

APP_NAME = DEPLOYMENT_LABEL
_RUNTIME_ENV = {
    "VIDAIO_NEXT_DEPLOYMENT_LABEL": DEPLOYMENT_LABEL,
    "VIDAIO_NEXT_SOLUTION_VARIANT": SOLUTION_VARIANT,
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "curl", "xz-utils", "zstd")
    # Premium (P1.1a) toolchain, baked at build time (cold starts may not
    # download anything): pinned ab-av1 release + BtbN static ffmpeg n9.0 —
    # libvmaf (ab-av1's scorer) + libsvtav1/libx264/libx265/libvpx; Debian's
    # own ffmpeg ships NO libvmaf. /usr/local/bin shadows the apt ffmpeg, and
    # the n9.0 line matches the repo's pinned ffmpeg major (Dockerfile note).
    .run_commands(
        "curl -fsSL -o /tmp/abav1.tar.zst https://github.com/alexheretic/ab-av1/"
        "releases/download/v0.11.7/ab-av1-v0.11.7-x86_64-unknown-linux-musl.tar.zst"
        " && tar --zstd -xf /tmp/abav1.tar.zst -C /usr/local/bin ab-av1"
        " && chmod 0755 /usr/local/bin/ab-av1",
        "curl -fsSL -o /tmp/ff.tar.xz https://github.com/BtbN/FFmpeg-Builds/"
        "releases/download/latest/ffmpeg-n9.0-latest-linux64-gpl-9.0.tar.xz"
        " && tar -xJf /tmp/ff.tar.xz -C /tmp"
        " && install -m 0755 /tmp/ffmpeg-*/bin/ffmpeg /tmp/ffmpeg-*/bin/ffprobe /usr/local/bin/"
        " && rm -rf /tmp/ff.tar.xz /tmp/ffmpeg-* /tmp/abav1.tar.zst",
        "ab-av1 --version && ffmpeg -version | head -1"
        " && ffmpeg -hide_banner -filters 2>/dev/null | grep -q libvmaf",
    )
    .uv_pip_install(
        "fastapi==0.116.1",
        "httpx==0.28.1",
        "numpy==2.2.6",
        "prometheus-client==0.22.1",
        "pydantic==2.11.7",
        "PyYAML==6.0.3",
        "torch==2.8.0",
        "uvicorn==0.35.0",
    )
    .add_local_python_source("vidaio")
)

app = modal.App(APP_NAME)


@app.function(
    name="vidaio-next-gpu-miner-worker",
    image=image,
    gpu=GPU_TYPE,
    secrets=[_AUTH_SECRET],
    env=_RUNTIME_ENV,
    timeout=600,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    buffer_containers=0,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def gpu_miner_app():  # type: ignore[no-untyped-def]
    import torch
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, Response
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    from vidaio.miner.gpu_worker import (
        GpuTransformError,
        decode_gpu_metadata,
        transform_media,
    )
    from vidaio.miner.remote_gpu import (
        CPU_FALLBACK_DEVICE,
        GPU_ACCELERATED_HEADER,
        GPU_DEVICE_HEADER,
        GPU_INPUT_DIGEST_HEADER,
        GPU_METADATA_HEADER,
        GPU_OUTPUT_DIGEST_HEADER,
        GPU_PROTOCOL_HEADER,
        GPU_PROTOCOL_VERSION,
        GPU_TRACK_HEADER,
        GPU_VARIANT_HEADER,
    )

    runtime_deployment_label = _required_fresh_name(
        "VIDAIO_NEXT_DEPLOYMENT_LABEL", what="Modal runtime deployment label"
    )
    runtime_solution_variant = os.environ.get(
        "VIDAIO_NEXT_SOLUTION_VARIANT", ""
    ).strip()
    if runtime_solution_variant not in {"quality", "balanced", "compact", "premium"}:
        raise RuntimeError(
            "Modal runtime solution variant must be quality, balanced, compact, "
            "or premium"
        )
    auth_token = os.environ.get("VIDAIO_NEXT_GPU_AUTH_TOKEN", "")
    if not auth_token:
        raise RuntimeError(
            "fresh Modal secret must contain VIDAIO_NEXT_GPU_AUTH_TOKEN"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")

    max_input_bytes = int(
        os.environ.get("VIDAIO_NEXT_GPU_MAX_INPUT_BYTES", str(2 * 1024**3))
    )
    max_output_bytes = int(
        os.environ.get("VIDAIO_NEXT_GPU_MAX_OUTPUT_BYTES", str(4 * 1024**3))
    )
    max_raw_bytes = int(
        os.environ.get("VIDAIO_NEXT_GPU_MAX_RAW_BYTES", str(4 * 1024**3))
    )
    if min(max_input_bytes, max_output_bytes, max_raw_bytes) < 1:
        raise RuntimeError("GPU worker byte bounds must all be positive")
    fallback_setting = os.environ.get("VIDAIO_NEXT_GPU_ALLOW_CPU_FALLBACK", "false").lower()
    if fallback_setting not in {"true", "false"}:
        raise RuntimeError("VIDAIO_NEXT_GPU_ALLOW_CPU_FALLBACK must be true or false")
    allow_cpu_fallback = fallback_setting == "true"

    registry = CollectorRegistry()
    requests_total = Counter(
        "vidaio_next_modal_gpu_requests_total",
        "GPU transform requests by track, solution variant and outcome",
        ("track", "variant", "outcome"),
        registry=registry,
    )
    processing_seconds = Histogram(
        "vidaio_next_modal_gpu_processing_seconds",
        "End-to-end GPU worker processing time",
        ("track", "variant"),
        registry=registry,
    )
    gpu_kernel_seconds = Histogram(
        "vidaio_next_modal_gpu_kernel_seconds",
        "CUDA frame-transform time",
        ("track", "variant"),
        registry=registry,
    )
    input_bytes_total = Counter(
        "vidaio_next_modal_gpu_input_bytes_total",
        "Accepted input bytes",
        ("track", "variant"),
        registry=registry,
    )
    output_bytes_total = Counter(
        "vidaio_next_modal_gpu_output_bytes_total",
        "Produced output bytes",
        ("track", "variant"),
        registry=registry,
    )
    gpu_memory_bytes = Gauge(
        "vidaio_next_modal_gpu_memory_allocated_bytes",
        "PyTorch bytes currently allocated on CUDA device zero",
        registry=registry,
    )

    web = FastAPI(title="vidaio-next-modal-gpu-miner", docs_url=None, redoc_url=None)

    class CleaningFileResponse(FileResponse):
        """Remove private task bytes even when the caller disconnects mid-send."""

        def __init__(self, *args, task_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__(*args, **kwargs)
            self._task_dir = task_dir

        async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
            try:
                await super().__call__(scope, receive, send)
            finally:
                shutil.rmtree(self._task_dir, ignore_errors=True)

    def authorize(request: Request) -> None:
        presented = request.headers.get("authorization", "")
        expected = f"Bearer {auth_token}"
        if not hmac.compare_digest(presented, expected):
            raise HTTPException(status_code=401, detail="unauthorized")

    def log_event(event: str, **fields: object) -> None:
        # Modal captures stdout/stderr per app/function/container. JSON keeps the
        # soak filterable; input bytes, tokens and full digests are never logged.
        print(
            json.dumps(
                {"event": event, "service": runtime_deployment_label, **fields},
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    @web.get("/healthz")
    async def healthz(request: Request):  # type: ignore[no-untyped-def]
        authorize(request)
        available = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if available else None
        return {
            "service": runtime_deployment_label,
            "status": "ok" if available else "unavailable",
            "gpu_available": available,
            "device": device,
            "protocol": GPU_PROTOCOL_VERSION,
            "scoring_device": "none (canonical scoring/auditing is CPU-only)",
        }

    @web.get("/metrics")
    async def metrics(request: Request) -> Response:
        authorize(request)
        gpu_memory_bytes.set(float(torch.cuda.memory_allocated(0)))
        return Response(
            content=generate_latest(registry),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @web.post("/process")
    async def process(request: Request):  # type: ignore[no-untyped-def]
        authorize(request)
        started = time.perf_counter()
        try:
            metadata = decode_gpu_metadata(request.headers.get(GPU_METADATA_HEADER))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if metadata.solution_variant != runtime_solution_variant:
            raise HTTPException(
                status_code=422,
                detail=(
                    "solution variant does not match this isolated deployment: "
                    f"expected {runtime_solution_variant!r}"
                ),
            )
        if request.headers.get(GPU_PROTOCOL_HEADER) != GPU_PROTOCOL_VERSION:
            raise HTTPException(status_code=400, detail="GPU protocol header mismatch")
        declared = request.headers.get("content-length")
        try:
            declared_size = int(declared or "")
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="integer Content-Length is required"
            ) from exc
        if declared_size != metadata.input_size:
            raise HTTPException(status_code=400, detail="input size binding mismatch")
        if declared_size > max_input_bytes:
            raise HTTPException(status_code=413, detail="input exceeds worker cap")

        task_dir = Path(tempfile.mkdtemp(prefix="vidaio-next-gpu-task-"))
        input_path = task_dir / "input.media"
        output_path = task_dir / "output.mp4"
        received = 0
        digest = hashlib.sha256()
        try:
            try:
                async with asyncio.timeout(min(metadata.deadline_seconds, 120.0)):
                    with input_path.open("xb") as sink:
                        os.chmod(input_path, 0o600)
                        async for chunk in request.stream():
                            received += len(chunk)
                            if received > max_input_bytes:
                                raise HTTPException(
                                    status_code=413,
                                    detail="input crossed worker cap",
                                )
                            digest.update(chunk)
                            sink.write(chunk)
                        sink.flush()
                        os.fsync(sink.fileno())
            except TimeoutError as exc:
                raise HTTPException(status_code=408, detail="input upload timed out") from exc
            if received != metadata.input_size:
                raise HTTPException(status_code=400, detail="input length changed in flight")
            input_digest = digest.hexdigest()
            if input_digest != metadata.input_digest:
                raise HTTPException(status_code=422, detail="input digest mismatch")
            elapsed = time.perf_counter() - started
            remaining = metadata.deadline_seconds - elapsed
            if remaining <= 0:
                raise HTTPException(status_code=504, detail="deadline exhausted at ingress")
            bounded_metadata = metadata.model_copy(
                update={"deadline_seconds": remaining}
            )
            try:
                result = await asyncio.to_thread(
                    transform_media,
                    input_path,
                    output_path,
                    bounded_metadata,
                    maximum_raw_bytes=max_raw_bytes,
                    maximum_output_bytes=max_output_bytes,
                    allow_cpu_fallback=allow_cpu_fallback,
                )
            except GpuTransformError as exc:
                requests_total.labels(
                    metadata.track, metadata.solution_variant, "failed"
                ).inc()
                log_event(
                    "gpu_transform_failed",
                    track=metadata.track,
                    variant=metadata.solution_variant,
                    error_type=type(exc).__name__,
                    reason=str(exc).replace(auth_token, "<redacted>")[:2048],
                    input_digest_prefix=input_digest[:12],
                    params=metadata.params,
                )
                status = 504 if "deadline" in str(exc).lower() else 422
                raise HTTPException(status_code=status, detail=str(exc)) from exc
            except Exception as exc:
                requests_total.labels(
                    metadata.track, metadata.solution_variant, "failed"
                ).inc()
                log_event(
                    "gpu_runtime_failed",
                    track=metadata.track,
                    variant=metadata.solution_variant,
                    error_type=type(exc).__name__,
                )
                # A CUDA runtime fault can poison a process after the exception.
                # Ask Modal to drain this container instead of accepting another
                # input on suspect state; its platform GPU-health plane handles
                # host-level critical faults independently.
                if isinstance(exc, RuntimeError):
                    try:
                        modal.experimental.stop_fetching_inputs()
                    except Exception as drain_exc:  # noqa: BLE001 - SDK best effort
                        log_event(
                            "gpu_container_drain_failed",
                            error_type=type(drain_exc).__name__,
                        )
                raise HTTPException(status_code=500, detail="GPU runtime failed") from exc
            output_size = output_path.stat().st_size
            output_hash = hashlib.sha256()
            with output_path.open("rb") as output_stream:
                for chunk in iter(lambda: output_stream.read(1 << 20), b""):
                    output_hash.update(chunk)
            output_digest = output_hash.hexdigest()
            total_seconds = time.perf_counter() - started
            requests_total.labels(
                metadata.track, metadata.solution_variant, "ok"
            ).inc()
            processing_seconds.labels(
                metadata.track, metadata.solution_variant
            ).observe(total_seconds)
            gpu_kernel_seconds.labels(
                metadata.track, metadata.solution_variant
            ).observe(result.gpu_seconds)
            input_bytes_total.labels(
                metadata.track, metadata.solution_variant
            ).inc(received)
            output_bytes_total.labels(
                metadata.track, metadata.solution_variant
            ).inc(output_size)
            gpu_memory_bytes.set(float(torch.cuda.memory_allocated(0)))
            log_event(
                "gpu_transform_complete",
                track=metadata.track,
                variant=metadata.solution_variant,
                frames=result.frames,
                width=result.width,
                height=result.height,
                input_bytes=received,
                output_bytes=output_size,
                processing_seconds=round(total_seconds, 6),
                gpu_seconds=round(result.gpu_seconds, 6),
                device=result.device,
                input_digest_prefix=input_digest[:12],
                output_digest_prefix=output_digest[:12],
            )
            headers = {
                GPU_PROTOCOL_HEADER: GPU_PROTOCOL_VERSION,
                GPU_INPUT_DIGEST_HEADER: input_digest,
                GPU_OUTPUT_DIGEST_HEADER: output_digest,
                GPU_TRACK_HEADER: metadata.track,
                GPU_VARIANT_HEADER: metadata.solution_variant,
                GPU_ACCELERATED_HEADER: "false" if result.device == CPU_FALLBACK_DEVICE else "true",
                GPU_DEVICE_HEADER: result.device,
            }
            # FileResponse adds Content-Length; the subclass cleanup covers both
            # successful sends and caller disconnects.
            return CleaningFileResponse(
                output_path,
                media_type="video/mp4",
                headers=headers,
                task_dir=task_dir,
            )
        except BaseException:
            shutil.rmtree(task_dir, ignore_errors=True)
            raise

    return web


@app.local_entrypoint()
def print_release_contract():
    """Read-only local check: print names/operators must review before deploy."""
    print(
        json.dumps(
            {
                "app_name": APP_NAME,
                "deployment_label": DEPLOYMENT_LABEL,
                "secret_name": SECRET_NAME,
                "solution_variant": SOLUTION_VARIANT,
                "gpu": GPU_TYPE,
                "max_containers": MAX_CONTAINERS,
                "scaledown_window_seconds": SCALEDOWN_WINDOW_SECONDS,
                "persistent_modal_resources": [],
                "scoring": "CPU-only external scoring worker/auditor",
            },
            sort_keys=True,
        )
    )
