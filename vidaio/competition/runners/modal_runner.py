"""Create-only Modal implementation of the competition ``SandboxRunner``.

``ModalSandboxRunner`` owns lifecycle/integrity/fault logic and accepts an
injected runtime, so ordinary tests never import Modal or contact the cloud.
``ModalSdkRuntime`` is the explicit current-SDK adapter.  It can only be started
with fresh ``vidaio-next-*`` Environment/App/run names and an exact creation
confirmation.  It never lists or discovers resources.  Process recovery may
rehydrate only an exact immutable Image id that this competition previously
created and durably bound to its pinned source; Sandboxes/instances remain fresh
per batch and are never attached or reused.

GPU is contender execution only. Every batch gets a new GPU Sandbox and a fresh,
content-bound input Image. Modal exposes a sandbox-local writable overlay when an
Image is mounted, so the isolation probe destroys its writer and remounts the same
Image in a second fresh CPU Sandbox to prove that writes did not reach the base.
An upscaling input has only its low-resolution bytes and a hidden digest-bound
sidecar containing the committed 2x/4x factor; no reference identifier or bytes
enter the image. Output is watched, frozen into a short-lived image, then copied
by a separate fresh CPU-only collector after the untrusted Sandbox is terminated.
The orchestrator's existing trusted CPU scorer/audit path consumes the resulting
bytes; no scorer exists here.

Modal exposes create-time isolation options but no public full-config readback.
The probe therefore records the exact host-side create request and applies an
advisory-negative in-container probe. A request mismatch fails closed and this
limitation is explicit in ``IsolationProbeReport.details``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from vidaio.competition.interfaces import (
    BUILD_IDENTITY_SCHEME,
    BatchItem,
    BatchOutput,
    ContenderSpec,
    IsolationProbeReport,
    logical_build_identity,
    upscale_task_sidecar_bytes,
    upscale_task_sidecar_name,
)
from vidaio.competition.runners import safeio
from vidaio.competition.runners.errors import (
    BatchExecutionError,
    BatchTimeout,
    BuildError,
    BuildTimeout,
    ContenderBuildError,
    InputStagingError,
    OutputRejectedError,
    OversizeOutputError,
    RunnerUnavailableError,
    SandboxIsolationError,
    SandboxProbeUnavailableError,
    SolutionExitError,
    UnknownImageError,
    UnsafePathError,
)
from vidaio.competition.runners.repo import (
    RepoProvider,
    checkout_pinned,
    release_checkout,
)
from vidaio.core.logging import get_logger, log_fields

logger = get_logger("vidaio.competition.runners.modal")

RESOURCE_PREFIX = "vidaio-next-"
FRESH_CREATION_CONFIRMATION = "CREATE_FRESH_VIDAIO_NEXT_MODAL_RESOURCES"
_RESOURCE_RE = re.compile(r"^vidaio-next-[a-zA-Z0-9._-]{6,52}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_OBJECT_ID_RE = re.compile(r"^im-[A-Za-z0-9_-]{1,128}$")
_REMOTE_INPUT = "/evaluation-inputs"
_REMOTE_OUTPUT = "/output"
_COLLECTED_OUTPUT = "/collected-output"
_TAIL_BYTES = 2000
_REMOTE_TREE_RESCAN_LIMIT = 3
_KEEPALIVE_COMMAND = (
    "/bin/sh",
    "-c",
    "while :; do sleep 3600; done",
)

_SECRET_NAME_RE = re.compile(
    r"(^|_)(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|PAT)(_|$)"
    r"|API_KEY|PRIVATE_KEY|ACCESS_KEY|SESSION_KEY|AUTH_TOKEN",
)

_PROBE_SCRIPT = """\
#!/bin/sh
# Advisory-negative only; host-side requested controls are authoritative.
net=2
if command -v wget >/dev/null 2>&1; then
  wget -q -T 3 -t 1 -O /dev/null http://example.com/ >/dev/null 2>&1 && net=1 || net=0
elif command -v python3 >/dev/null 2>&1; then
  python3 -c 'import socket;socket.create_connection(("1.1.1.1",80),3)' \
    >/dev/null 2>&1 && net=1 || net=0
elif command -v getent >/dev/null 2>&1; then
  getent hosts example.com >/dev/null 2>&1 && net=1 || net=0
fi
echo "NETWORK_ATTEMPT=$net"
w=0; ( : > /evaluation-inputs/.probe-write ) 2>/dev/null && w=1
echo "INPUT_WRITE=$w"
m=0
for d in /reference /references /holdout /evaluation-reference; do
  [ -e "$d" ] && m=1
done
echo "REF_MOUNTS=$m"
i=0
for f in /evaluation-inputs/index.json /index.json /app/index.json; do
  [ -e "$f" ] && i=1
done
echo "INDEX_LEAK=$i"
echo "ENV_BEGIN"
env
echo "ENV_END"
echo "PROBE_DONE=1"
"""

_INPUT_OVERLAY_CHECK_SCRIPT = """\
#!/bin/sh
mutated=0
[ -e /evaluation-inputs/.probe-write ] && mutated=1
echo "INPUT_BASE_MUTATED=$mutated"
echo "OVERLAY_CHECK_DONE=1"
"""


def _validate_resource_name(value: str, *, what: str) -> str:
    value = value.strip()
    if not _RESOURCE_RE.fullmatch(value) or len(value) >= 64:
        raise ValueError(
            f"{what} must start with {RESOURCE_PREFIX!r}, contain only Modal-safe "
            "characters, include a unique run suffix, and be shorter than 64 "
            f"characters (got {value!r})"
        )
    return value


def _tail(value: str, limit: int = _TAIL_BYTES) -> str:
    value = value.strip()
    return value[-limit:] if len(value) > limit else value


def _modal_error_text(exc: BaseException) -> str:
    # A Dockerfile can influence build output. Never persist an unbounded message.
    return _tail(f"{type(exc).__name__}: {exc}")


@dataclass(frozen=True)
class ModalRunnerConfig:
    """Explicit resource and boundary limits for one fresh Modal run."""

    gpu: str = "L4"
    cpu: float = 2.0
    memory_mb: int = 8192
    build_timeout_seconds: float = 1200.0
    batch_timeout_seconds: float = 900.0
    probe_timeout_seconds: float = 120.0
    sandbox_lifetime_seconds: int = 23 * 3600 + 30 * 60
    idle_timeout_seconds: int = 300
    max_output_bytes: int = 512 * 1024 * 1024
    max_batch_output_bytes: int = 2 * 1024 * 1024 * 1024
    max_log_bytes: int = 8 * 1024 * 1024
    max_output_entries: int = 4096
    output_poll_seconds: float = 0.25
    snapshot_ttl_seconds: int = 3600

    def __post_init__(self) -> None:
        if not self.gpu.strip():
            raise ValueError("Modal contender GPU type must be non-empty")
        if not math.isfinite(float(self.cpu)) or self.cpu <= 0:
            raise ValueError("Modal cpu must be finite and positive")
        if self.memory_mb < 256:
            raise ValueError("Modal memory_mb must be >= 256")
        for name in (
            "build_timeout_seconds",
            "batch_timeout_seconds",
            "probe_timeout_seconds",
            "output_poll_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 60 <= self.sandbox_lifetime_seconds <= 23 * 3600 + 30 * 60:
            raise ValueError(
                "sandbox_lifetime_seconds must be between 60 and 84600 (roll over "
                "before Modal's 24-hour lifetime)"
            )
        if not 1 <= self.idle_timeout_seconds <= self.sandbox_lifetime_seconds:
            raise ValueError("idle_timeout_seconds must fit within sandbox lifetime")
        for name in (
            "max_output_bytes",
            "max_batch_output_bytes",
            "max_log_bytes",
            "max_output_entries",
            "snapshot_ttl_seconds",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_batch_output_bytes < self.max_output_bytes:
            raise ValueError("max_batch_output_bytes must be >= max_output_bytes")


@dataclass(frozen=True)
class SandboxRequestAttestation:
    """Security-relevant fields submitted in ``Sandbox.create``."""

    name: str
    role: str
    block_network: bool
    secret_count: int
    env_keys: tuple[str, ...]
    include_oidc_identity_token: bool
    volume_mounts: tuple[str, ...]
    network_filesystem_mounts: tuple[str, ...]
    ports: tuple[int, ...]
    gpu: str | None
    image_mounts: tuple[str, ...] = ()

    @property
    def base_isolated(self) -> bool:
        return (
            self.name.startswith(RESOURCE_PREFIX)
            and self.block_network
            and self.secret_count == 0
            and not self.env_keys
            and not self.include_oidc_identity_token
            and not self.volume_mounts
            and not self.network_filesystem_mounts
            and not self.ports
        )


@dataclass(frozen=True)
class RemoteFile:
    path: str
    kind: str
    size: int


@runtime_checkable
class RemoteProcess(Protocol):
    stdout: Iterable[bytes | str]
    stderr: Iterable[bytes | str]

    def poll(self) -> int | None: ...


@runtime_checkable
class ModalSandboxLease(Protocol):
    object_id: str

    def attestation(self) -> SandboxRequestAttestation: ...

    def mount_image(self, path: str, image: object) -> None: ...

    def make_directory(self, path: str) -> None: ...

    def write_text(self, path: str, value: str) -> None: ...

    def exec(self, args: Sequence[str], *, timeout_seconds: float) -> RemoteProcess: ...

    def list_files(self, path: str) -> Sequence[RemoteFile]: ...

    def stat(self, path: str) -> RemoteFile: ...

    def copy_to_local(self, remote_path: str, local_path: Path) -> None: ...

    def snapshot_directory(self, path: str, *, ttl_seconds: int) -> object: ...

    def terminate(self) -> None: ...

    def detach(self) -> None: ...


@runtime_checkable
class ModalRuntime(Protocol):
    """Minimal create-only Modal surface consumed by the runner."""

    run_label: str

    def available(self) -> bool: ...

    def build_contender_image(
        self, checkout: Path, dockerfile: Path
    ) -> tuple[object, str]: ...

    def restore_contender_image(self, image_object_id: str) -> tuple[object, str]: ...

    def build_input_image(self, staged_inputs: Path) -> object: ...

    def create_sandbox(
        self,
        *,
        image: object,
        name: str,
        role: str,
        tags: dict[str, str],
        gpu: str | None,
        cpu: float,
        memory_mb: int,
        timeout_seconds: int,
        idle_timeout_seconds: int,
    ) -> ModalSandboxLease: ...

    def create_collector_sandbox(
        self,
        *,
        name: str,
        tags: dict[str, str],
        timeout_seconds: int,
    ) -> ModalSandboxLease: ...

    def close(self) -> None: ...


@dataclass
class _LogCapture:
    cap: int
    tail_limit: int = _TAIL_BYTES
    total: int = 0
    overflow: bool = False
    errors: list[str] = field(default_factory=list)
    _data: bytearray = field(default_factory=bytearray)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def consume(self, stream: Iterable[bytes | str]) -> None:
        try:
            for raw in stream:
                chunk = (
                    raw.encode("utf-8", "replace")
                    if isinstance(raw, str)
                    else bytes(raw)
                )
                with self._lock:
                    self.total += len(chunk)
                    if self.total > self.cap:
                        self.overflow = True
                    remaining = max(0, self.cap - len(self._data))
                    if remaining:
                        self._data.extend(chunk[:remaining])
        except Exception as exc:  # noqa: BLE001 - converted to typed infra error
            with self._lock:
                self.errors.append(_modal_error_text(exc))

    def tail(self) -> str:
        with self._lock:
            return bytes(self._data[-self.tail_limit :]).decode("utf-8", "replace")

    def text(self) -> str:
        with self._lock:
            return bytes(self._data).decode("utf-8", "replace")


@dataclass(frozen=True)
class _ExecutionResult:
    returncode: int
    elapsed: float
    log_tail: str
    log_text: str


class ModalSandboxRunner:
    """Per-batch, create-only Modal GPU implementation of ``SandboxRunner``.

    The runtime is mandatory: there is intentionally no default that can discover
    credentials or contact Modal during ordinary local construction.
    """

    def __init__(
        self,
        repo_provider: RepoProvider,
        runtime: ModalRuntime,
        *,
        inputs_dir: str | Path,
        outputs_dir: str | Path,
        scratch_dir: str | Path,
        config: ModalRunnerConfig | None = None,
    ) -> None:
        self._repos = repo_provider
        self._runtime = runtime
        self._inputs_dir = Path(inputs_dir)
        self._outputs_dir = Path(outputs_dir)
        self._scratch_dir = Path(scratch_dir)
        self._cfg = config or ModalRunnerConfig()
        # ``run_label`` names the fresh Modal resource namespace, but it is not a
        # sufficient process-restart fence: an operator could accidentally reuse
        # the same configured label while constructing a new SDK runtime, whose
        # Python Image handles are necessarily unrelated to this object.  The
        # random session id is persisted by the orchestrator before any build and
        # changes for every runner construction, even when the label does not.
        # It is provenance only; it is never used to discover or attach to an
        # unrelated/pre-existing Modal object.
        self._runtime_session_id = hashlib.sha256(
            f"{runtime.run_label}:{uuid.uuid4().hex}".encode("utf-8")
        ).hexdigest()
        self._images: dict[str, object] = {}
        self._image_object_ids: dict[str, str] = {}
        # A SCHEDULED earning competition prebuilds its baseline before anchoring.
        # Reusing this exact handle later is what makes the committed opaque Modal
        # image id executable instead of forcing a second, potentially different
        # force_build. The cache belongs only to this newly-created run; no external
        # App/Image discovery or existing-resource lookup is performed.
        self._spec_digests: dict[tuple[str, str, str], str] = {}
        self._probed: set[str] = set()
        self._owned: dict[str, ModalSandboxLease] = {}
        self._closed = False
        self._lock = threading.RLock()
        for directory in (self._inputs_dir, self._outputs_dir, self._scratch_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if not runtime.run_label.startswith(RESOURCE_PREFIX):
            raise RunnerUnavailableError(
                f"Modal runtime run_label must start with {RESOURCE_PREFIX!r}"
            )
        if not runtime.available():
            raise RunnerUnavailableError(
                "the supplied fresh Modal runtime is unavailable"
            )

    @property
    def inputs_dir(self) -> Path:
        return self._inputs_dir

    @property
    def outputs_dir(self) -> Path:
        return self._outputs_dir

    @property
    def gpu(self) -> str:
        """Exact GPU string sent on every probe/contender Sandbox request."""
        return self._cfg.gpu

    @property
    def runtime_session_id(self) -> str:
        """Opaque identity of this in-process, fresh-only Modal runtime.

        A persisted image digest is not itself an executable Modal handle.  The
        orchestrator records this value and, after a restart, restores only exact
        competition-owned immutable Image ids, reprobes them, and resets the
        effective batch matrix before permitting another GPU batch.
        """
        return self._runtime_session_id

    @property
    def runtime_label(self) -> str:
        """Fresh resource label recorded alongside ``runtime_session_id``."""
        return self._runtime.run_label

    def has_live_image(self, image_digest: str) -> bool:
        """Whether this exact runner owns an executable handle for ``digest``.

        This intentionally checks only the in-memory cache. Restoration is a
        separate explicit operation over an append-only competition-owned Image
        id; no list/name lookup or App/Sandbox attachment is permitted here.
        """
        if not _DIGEST_RE.fullmatch(image_digest):
            return False
        with self._lock:
            return image_digest in self._images

    def image_object_id(self, image_digest: str) -> str | None:
        """Return the provider id for an image built/restored by this runner.

        This is persistence data, not a discovery seam.  Callers may durably bind
        it to the competition and later pass that exact id back to
        :meth:`restore_image`; no provider inventory lookup is ever performed.
        """
        if not _DIGEST_RE.fullmatch(image_digest):
            return None
        with self._lock:
            return self._image_object_ids.get(image_digest)

    def restore_image(
        self,
        contender: ContenderSpec,
        image_digest: str,
        image_object_id: str,
    ) -> str:
        """Rehydrate one immutable image created by this competition earlier.

        The caller must obtain ``image_object_id`` from the append-only local
        binding for this exact pinned contender.  The stable logical build digest
        is recomputed from the pinned source before any provider call, and the
        provider must return the identical separately-bound object id.
        This never restores a Sandbox or running instance; every execution batch
        still creates a new isolated GPU Sandbox.
        """
        self._assert_open()
        if not _DIGEST_RE.fullmatch(image_digest):
            raise UnknownImageError(f"invalid image_digest {image_digest!r}")
        if not _IMAGE_OBJECT_ID_RE.fullmatch(image_object_id):
            raise UnknownImageError(
                f"invalid Modal image object id {image_object_id!r}"
            )
        expected = logical_build_identity(
            repo_url=contender.repo_url,
            commit_sha=contender.commit_sha,
            tree_sha=contender.tree_sha,
        )
        if expected != image_digest:
            raise UnknownImageError(
                "persisted Modal logical build digest does not match the pinned source"
            )
        spec_key = (
            contender.repo_url,
            contender.commit_sha,
            contender.tree_sha,
        )
        with self._lock:
            cached = self._images.get(image_digest)
            cached_id = self._image_object_ids.get(image_digest)
            if cached is not None:
                if cached_id != image_object_id:
                    raise UnknownImageError(
                        "live Modal image handle disagrees with its persisted object id"
                    )
                self._spec_digests[spec_key] = image_digest
                return image_digest
        try:
            image, restored_id = self._runtime.restore_contender_image(image_object_id)
        except Exception as exc:
            raise BuildError(
                "competition-owned Modal image could not be rehydrated: "
                f"{_modal_error_text(exc)}"
            ) from exc
        if restored_id != image_object_id:
            raise BuildError(
                "Modal restored a different image object id: expected "
                f"{image_object_id}, got {restored_id}"
            )
        with self._lock:
            self._images[image_digest] = image
            self._image_object_ids[image_digest] = image_object_id
            self._spec_digests[spec_key] = image_digest
            self._probed.discard(image_digest)
        return image_digest

    def available(self) -> bool:
        return not self._closed and self._runtime.available()

    def build(self, contender: ContenderSpec) -> str:
        self._assert_open()
        spec_key = (
            contender.repo_url,
            contender.commit_sha,
            contender.tree_sha,
        )
        with self._lock:
            cached = self._spec_digests.get(spec_key)
            if cached is not None and cached in self._images:
                logger.info(
                    "reusing exact image built in this fresh Modal run",
                    extra=log_fields(
                        run_label=self._runtime.run_label,
                        contender_id=contender.contender_id,
                        tree_sha=contender.tree_sha,
                        image_digest=cached,
                    ),
                )
                return cached
        checkout = checkout_pinned(
            self._repos,
            contender.repo_url,
            contender.commit_sha,
            contender.tree_sha,
        )
        dockerfile = checkout / "Dockerfile"
        try:
            safeio.lstat_regular(dockerfile, what="Dockerfile")
            safeio.assert_safe_tree(checkout)
        except (FileNotFoundError, UnsafePathError) as exc:
            release_checkout(self._repos, checkout)
            raise ContenderBuildError(
                f"contender {contender.contender_id}: pinned checkout has no safe "
                f"Dockerfile/tree ({exc})"
            ) from exc
        except BaseException:
            release_checkout(self._repos, checkout)
            raise

        started = time.monotonic()
        try:
            image, image_id = self._bounded_build(checkout, dockerfile)
        except BuildTimeout:
            raise
        except Exception as exc:
            if isinstance(exc, _ModalContenderImageBuildError):
                raise ContenderBuildError(str(exc)) from exc
            raise BuildError(
                f"fresh Modal image build could not run: {_modal_error_text(exc)}"
            ) from exc
        if not isinstance(image_id, str) or not _IMAGE_OBJECT_ID_RE.fullmatch(image_id):
            raise BuildError(
                "fresh Modal image build returned a malformed image object id"
            )
        # Modal's ``im-*`` value is an opaque per-build handle, not a content
        # digest: rebuilding byte-identical pinned source in a fresh resource
        # namespace mints another id.  Commit the stable source identity to the
        # protocol and retain the exact object id separately for owned-image
        # restoration and run evidence.
        image_digest = logical_build_identity(
            repo_url=contender.repo_url,
            commit_sha=contender.commit_sha,
            tree_sha=contender.tree_sha,
        )
        with self._lock:
            self._images[image_digest] = image
            self._image_object_ids[image_digest] = image_id
            self._spec_digests[spec_key] = image_digest
            self._probed.discard(image_digest)
        logger.info(
            "fresh Modal contender image built",
            extra=log_fields(
                run_label=self._runtime.run_label,
                contender_id=contender.contender_id,
                tree_sha=contender.tree_sha,
                image_id=image_id,
                image_digest=image_digest,
                build_identity_scheme=BUILD_IDENTITY_SCHEME,
                build_seconds=round(time.monotonic() - started, 3),
            ),
        )
        return image_digest

    def _bounded_build(self, checkout: Path, dockerfile: Path) -> tuple[object, str]:
        box: dict[str, Any] = {}
        done = threading.Event()

        def invoke() -> None:
            try:
                box["value"] = self._runtime.build_contender_image(checkout, dockerfile)
            except BaseException as exc:
                box["error"] = exc
            finally:
                try:
                    # Image.build has returned, so Modal has completed consuming the
                    # local build context. It is now safe to remove the fresh clone.
                    release_checkout(self._repos, checkout)
                except BaseException as exc:
                    box["error"] = exc
                done.set()

        build_thread = threading.Thread(
            target=invoke,
            name=f"{self._runtime.run_label}-image-build",
            daemon=True,
        )
        try:
            build_thread.start()
        except BaseException:
            release_checkout(self._repos, checkout)
            raise
        if not done.wait(self._cfg.build_timeout_seconds):
            # Image.build has no per-build cancellation. Closing the fresh
            # ephemeral App is Modal's strongest available cancellation boundary.
            self.close()
            raise BuildTimeout(
                f"fresh Modal image build exceeded {self._cfg.build_timeout_seconds}s; "
                "the owned ephemeral App was closed and this runner must be replaced"
            )
        if "error" in box:
            raise box["error"]
        return box["value"]

    def isolation_probe(self, image_digest: str) -> IsolationProbeReport:
        self._assert_open()
        image = self._resolve_image(image_digest)
        probe_dir = self._new_scratch("modal-probe", image_digest, 0)
        empty_inputs = probe_dir / "inputs"
        empty_inputs.mkdir()
        lease: ModalSandboxLease | None = None
        overlay_lease: ModalSandboxLease | None = None
        attestation: SandboxRequestAttestation | None = None
        overlay_attestation: SandboxRequestAttestation | None = None
        note = ""
        stdout = ""
        overlay_note = ""
        overlay_stdout = ""
        try:
            try:
                input_image = self._runtime.build_input_image(empty_inputs)
                lease = self._create_owned_sandbox(
                    image=image,
                    image_digest=image_digest,
                    batch_index=None,
                    role="probe",
                    gpu=self._cfg.gpu,
                    timeout_seconds=math.ceil(self._cfg.probe_timeout_seconds + 60),
                )
                lease.mount_image(_REMOTE_INPUT, input_image)
                self._assert_request_isolation(lease, _REMOTE_INPUT, gpu=True)
                lease.write_text("/tmp/vidaio-next-isolation-probe.sh", _PROBE_SCRIPT)
                result = self._execute_watched(
                    lease,
                    ["/bin/sh", "/tmp/vidaio-next-isolation-probe.sh"],
                    timeout_seconds=self._cfg.probe_timeout_seconds,
                    output_root=None,
                    what="Modal isolation probe",
                )
                stdout = result.log_text
                if result.returncode != 0:
                    note = f"probe script exited {result.returncode}"

                # Modal Image mounts currently expose a writable sandbox-local
                # overlay. Destroy the writer, then remount the exact same Image
                # in a second fresh CPU Sandbox. The marker must not have reached
                # the content-bound base image. This is stronger and more honest
                # than claiming that Modal offers a read-only mount flag.
                attestation = lease.attestation()
                self._terminate_owned(lease, reason="probe_writer_complete")
                lease = None
                overlay_lease = self._create_owned_sandbox(
                    image=image,
                    image_digest=image_digest,
                    batch_index=None,
                    role="probe-overlay-check",
                    gpu=None,
                    timeout_seconds=math.ceil(self._cfg.probe_timeout_seconds + 60),
                )
                overlay_lease.mount_image(_REMOTE_INPUT, input_image)
                self._assert_request_isolation(overlay_lease, _REMOTE_INPUT, gpu=False)
                overlay_lease.write_text(
                    "/tmp/vidaio-next-input-overlay-check.sh",
                    _INPUT_OVERLAY_CHECK_SCRIPT,
                )
                overlay_result = self._execute_watched(
                    overlay_lease,
                    ["/bin/sh", "/tmp/vidaio-next-input-overlay-check.sh"],
                    timeout_seconds=self._cfg.probe_timeout_seconds,
                    output_root=None,
                    what="Modal input Image overlay persistence check",
                )
                overlay_stdout = overlay_result.log_text
                if overlay_result.returncode != 0:
                    overlay_note = (
                        f"overlay check script exited {overlay_result.returncode}"
                    )
                overlay_attestation = overlay_lease.attestation()
            except (BatchTimeout, OversizeOutputError) as exc:
                note = f"probe script aborted: {exc}"
            except SandboxIsolationError:
                raise
            except Exception as exc:
                raise SandboxProbeUnavailableError(
                    f"fresh Modal isolation probe could not run: {_modal_error_text(exc)}"
                ) from exc
            if attestation is None and lease is not None:
                attestation = lease.attestation()
            assert attestation is not None
            report = self._probe_report(
                attestation,
                stdout,
                note,
                overlay_attestation,
                overlay_stdout,
                overlay_note,
            )
            if report.passed:
                with self._lock:
                    self._probed.add(image_digest)
            return report
        finally:
            if lease is not None:
                self._terminate_owned(lease, reason="probe_complete")
            if overlay_lease is not None:
                self._terminate_owned(
                    overlay_lease, reason="probe_overlay_check_complete"
                )
            shutil.rmtree(probe_dir, ignore_errors=True)

    def run_batch(
        self, image_digest: str, items: Sequence[BatchItem], batch_index: int
    ) -> Sequence[BatchOutput]:
        self._assert_open()
        if batch_index < 0:
            raise ValueError("batch_index must be non-negative")
        image = self._resolve_image(image_digest)
        with self._lock:
            if image_digest not in self._probed:
                raise SandboxIsolationError(
                    f"image {image_digest} has not passed this runner's isolation probe"
                )
        if not items:
            return []

        run_dir = self._new_scratch("modal-run", image_digest, batch_index)
        in_dir = run_dir / "inputs"
        collected_dir = run_dir / "collected"
        in_dir.mkdir()
        collected_dir.mkdir()
        lease: ModalSandboxLease | None = None
        collector: ModalSandboxLease | None = None
        try:
            for item in items:
                self._stage_input(item, in_dir)
            try:
                input_image = self._runtime.build_input_image(in_dir)
            except Exception as exc:
                raise InputStagingError(
                    f"could not seal batch {batch_index} inputs into a fresh immutable "
                    f"Modal image: {_modal_error_text(exc)}"
                ) from exc

            self.rollover(image_digest)
            lease = self._create_owned_sandbox(
                image=image,
                image_digest=image_digest,
                batch_index=batch_index,
                role="contender",
                gpu=self._cfg.gpu,
                timeout_seconds=min(
                    self._cfg.sandbox_lifetime_seconds,
                    math.ceil(self._cfg.batch_timeout_seconds + 120),
                ),
            )
            lease.mount_image(_REMOTE_INPUT, input_image)
            lease.make_directory(_REMOTE_OUTPUT)
            self._assert_request_isolation(lease, _REMOTE_INPUT, gpu=True)
            result = self._execute_watched(
                lease,
                ["/bin/sh", "/app/run.sh", _REMOTE_INPUT, _REMOTE_OUTPUT],
                timeout_seconds=self._cfg.batch_timeout_seconds,
                output_root=_REMOTE_OUTPUT,
                what=f"batch {batch_index} ({image_digest[:16]})",
            )
            if result.returncode != 0:
                raise SolutionExitError(
                    f"batch {batch_index} ({image_digest[:16]}) solution exited "
                    f"{result.returncode}: {result.log_tail}"
                )
            frozen_output = self._snapshot_watched(
                lease,
                batch_index=batch_index,
            )
            # No untrusted code is alive while output paths are copied.
            self._terminate_owned(lease, reason="batch_output_frozen")
            lease = None

            collector = self._create_collector(image_digest, batch_index)
            collector.mount_image(_COLLECTED_OUTPUT, frozen_output)
            self._assert_request_isolation(collector, _COLLECTED_OUTPUT, gpu=False)
            total, entries = self._remote_tree_usage(collector, _COLLECTED_OUTPUT)
            if entries > self._cfg.max_output_entries:
                raise OversizeOutputError(
                    f"batch output contains {entries} entries, over the cap of "
                    f"{self._cfg.max_output_entries}"
                )
            if total > self._cfg.max_batch_output_bytes:
                raise OversizeOutputError(
                    f"batch outputs total {total} bytes, over the per-batch cap of "
                    f"{self._cfg.max_batch_output_bytes}"
                )
            outputs = self._collect_outputs(
                collector, collected_dir, items, result.elapsed
            )
            logger.info(
                "fresh Modal GPU batch executed and frozen output collected",
                extra=log_fields(
                    run_label=self._runtime.run_label,
                    image_digest=image_digest,
                    batch_index=batch_index,
                    items=len(items),
                    outputs=len(outputs),
                    output_bytes=sum(output.output_bytes for output in outputs),
                    wall_seconds=round(result.elapsed, 3),
                    scorer_device="none (CPU scorer is downstream)",
                ),
            )
            return outputs
        finally:
            if lease is not None:
                self._terminate_owned(lease, reason="batch_finally")
            if collector is not None:
                self._terminate_owned(collector, reason="collector_complete")
            shutil.rmtree(run_dir, ignore_errors=True)

    def rollover(self, image_digest: str) -> None:
        """Kill this image's owned Sandboxes before its next fresh batch."""
        prefix = f"image:{image_digest}:"
        with self._lock:
            leases = [
                lease for key, lease in self._owned.items() if key.startswith(prefix)
            ]
        for lease in leases:
            self._terminate_owned(lease, reason="per_batch_rollover")

    def terminate(self, image_digest: str | None = None) -> None:
        """Terminate only Sandboxes created and still owned by this runner."""
        with self._lock:
            if image_digest is None:
                leases = list(self._owned.values())
            else:
                prefix = f"image:{image_digest}:"
                leases = [
                    lease
                    for key, lease in self._owned.items()
                    if key.startswith(prefix)
                ]
        for lease in leases:
            self._terminate_owned(lease, reason="explicit_terminate")

    def close(self) -> None:
        if self._closed:
            return
        self.terminate()
        self._closed = True
        self._images.clear()
        self._image_object_ids.clear()
        self._probed.clear()
        self._runtime.close()

    def __enter__(self) -> ModalSandboxRunner:
        self._assert_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _assert_open(self) -> None:
        if self._closed:
            raise RunnerUnavailableError("the fresh Modal runner is closed")

    def _resolve_image(self, image_digest: str) -> object:
        if not _DIGEST_RE.fullmatch(image_digest):
            raise UnknownImageError(f"invalid image_digest {image_digest!r}")
        with self._lock:
            image = self._images.get(image_digest)
        if image is None:
            raise UnknownImageError(
                f"image_digest {image_digest} was not built by this live fresh "
                "runner; cloud image lookup/reuse is forbidden, so rebuild in a "
                "new competition run"
            )
        return image

    def _new_scratch(self, role: str, image_digest: str, batch_index: int) -> Path:
        path = self._scratch_dir / (
            f"{role}-{image_digest[:12]}-b{batch_index}-{uuid.uuid4().hex[:8]}"
        )
        path.mkdir(parents=True)
        return path

    def _resource_name(self, role: str) -> str:
        stem = re.sub(r"[^a-zA-Z0-9._-]", "-", self._runtime.run_label)
        suffix = uuid.uuid4().hex[:12]
        maximum_stem = 63 - len(role) - len(suffix) - 2
        return f"{stem[:maximum_stem]}-{role}-{suffix}"

    def _create_owned_sandbox(
        self,
        *,
        image: object,
        image_digest: str,
        batch_index: int | None,
        role: str,
        gpu: str | None,
        timeout_seconds: int,
    ) -> ModalSandboxLease:
        name = self._resource_name(role)
        try:
            lease = self._runtime.create_sandbox(
                image=image,
                name=name,
                role=role,
                tags={
                    "vidaio-resource": "vidaio-next",
                    "vidaio-run": self._runtime.run_label,
                    "vidaio-role": role,
                    "vidaio-image-digest": image_digest,
                    "vidaio-batch": "probe"
                    if batch_index is None
                    else str(batch_index),
                },
                gpu=gpu,
                cpu=self._cfg.cpu,
                memory_mb=self._cfg.memory_mb,
                timeout_seconds=timeout_seconds,
                idle_timeout_seconds=min(
                    self._cfg.idle_timeout_seconds, timeout_seconds
                ),
            )
        except Exception as exc:
            raise BatchExecutionError(
                f"could not create fresh Modal {role} Sandbox {name!r}: "
                f"{_modal_error_text(exc)}"
            ) from exc
        key = f"image:{image_digest}:{role}:{lease.object_id}"
        with self._lock:
            self._owned[key] = lease
        logger.info(
            "fresh Modal Sandbox created",
            extra=log_fields(
                run_label=self._runtime.run_label,
                sandbox_name=name,
                sandbox_id=lease.object_id,
                role=role,
                gpu=gpu,
            ),
        )
        return lease

    def _create_collector(
        self, image_digest: str, batch_index: int
    ) -> ModalSandboxLease:
        name = self._resource_name("collector")
        try:
            lease = self._runtime.create_collector_sandbox(
                name=name,
                tags={
                    "vidaio-resource": "vidaio-next",
                    "vidaio-run": self._runtime.run_label,
                    "vidaio-role": "collector",
                    "vidaio-image-digest": image_digest,
                    "vidaio-batch": str(batch_index),
                },
                timeout_seconds=max(60, math.ceil(self._cfg.batch_timeout_seconds)),
            )
        except Exception as exc:
            raise BatchExecutionError(
                f"could not create fresh CPU output collector {name!r}: "
                f"{_modal_error_text(exc)}"
            ) from exc
        key = f"image:{image_digest}:collector:{lease.object_id}"
        with self._lock:
            self._owned[key] = lease
        return lease

    def _terminate_owned(self, lease: ModalSandboxLease, *, reason: str) -> None:
        with self._lock:
            keys = [key for key, value in self._owned.items() if value is lease]
            if not keys:
                return
            for key in keys:
                self._owned.pop(key, None)
        errors: list[str] = []
        try:
            lease.terminate()
        except Exception as exc:
            errors.append(f"terminate: {_modal_error_text(exc)}")
        try:
            lease.detach()
        except Exception as exc:
            errors.append(f"detach: {_modal_error_text(exc)}")
        logger.info(
            "owned Modal Sandbox released",
            extra=log_fields(
                run_label=self._runtime.run_label,
                sandbox_id=lease.object_id,
                reason=reason,
                cleanup_errors=errors,
            ),
        )

    def _assert_request_isolation(
        self, lease: ModalSandboxLease, expected_mount: str, *, gpu: bool
    ) -> None:
        att = lease.attestation()
        problems: list[str] = []
        if not att.base_isolated:
            problems.append("base create request is not isolated")
        if att.image_mounts != (expected_mount,):
            problems.append(
                f"image mounts {att.image_mounts!r} != expected {(expected_mount,)!r}"
            )
        if gpu and att.gpu != self._cfg.gpu:
            problems.append(f"GPU {att.gpu!r} != required {self._cfg.gpu!r}")
        if not gpu and att.gpu is not None:
            problems.append(f"CPU collector unexpectedly requested GPU {att.gpu!r}")
        if problems:
            raise SandboxIsolationError(
                f"Modal request attestation failed for {att.name!r}: "
                + "; ".join(problems)
            )

    def _execute_watched(
        self,
        lease: ModalSandboxLease,
        args: Sequence[str],
        *,
        timeout_seconds: float,
        output_root: str | None,
        what: str,
    ) -> _ExecutionResult:
        started = time.monotonic()
        try:
            process = lease.exec(args, timeout_seconds=timeout_seconds)
        except Exception as exc:
            raise BatchExecutionError(
                f"{what} could not exec: {_modal_error_text(exc)}"
            ) from exc
        logs = _LogCapture(self._cfg.max_log_bytes)
        readers = [
            threading.Thread(
                target=logs.consume,
                args=(process.stdout,),
                name=f"{self._runtime.run_label}-stdout",
                daemon=True,
            ),
            threading.Thread(
                target=logs.consume,
                args=(process.stderr,),
                name=f"{self._runtime.run_label}-stderr",
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()
        returncode: int | None = None
        while returncode is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                lease.terminate()
                raise BatchTimeout(
                    f"{what} exceeded {timeout_seconds}s and was terminated"
                )
            if logs.overflow:
                lease.terminate()
                raise OversizeOutputError(
                    f"{what} wrote more than {self._cfg.max_log_bytes} bytes to "
                    "stdout/stderr and was terminated"
                )
            if output_root is not None:
                try:
                    total, entries = self._remote_tree_usage(lease, output_root)
                except (OutputRejectedError, OversizeOutputError):
                    lease.terminate()
                    raise
                except Exception as exc:
                    lease.terminate()
                    raise BatchExecutionError(
                        f"{what} output watchdog could not inspect the Sandbox: "
                        f"{_modal_error_text(exc)}"
                    ) from exc
                if entries > self._cfg.max_output_entries:
                    lease.terminate()
                    raise OversizeOutputError(
                        f"{what} created {entries} output entries, over the cap of "
                        f"{self._cfg.max_output_entries}"
                    )
                if total > self._cfg.max_batch_output_bytes:
                    lease.terminate()
                    raise OversizeOutputError(
                        f"{what} wrote {total} output bytes, over the cap of "
                        f"{self._cfg.max_batch_output_bytes}"
                    )
            try:
                returncode = process.poll()
            except Exception as exc:
                lease.terminate()
                raise BatchExecutionError(
                    f"{what} status could not be read: {_modal_error_text(exc)}"
                ) from exc
            if returncode is None:
                time.sleep(self._cfg.output_poll_seconds)

        join_deadline = time.monotonic() + min(5.0, timeout_seconds)
        for reader in readers:
            reader.join(max(0.0, join_deadline - time.monotonic()))
        if logs.overflow:
            lease.terminate()
            raise OversizeOutputError(
                f"{what} wrote more than {self._cfg.max_log_bytes} bytes to logs"
            )
        if logs.errors:
            raise BatchExecutionError(
                f"{what} log stream failed: {'; '.join(logs.errors[:2])}"
            )
        if output_root is not None:
            total, entries = self._remote_tree_usage(lease, output_root)
            if entries > self._cfg.max_output_entries:
                raise OversizeOutputError(
                    f"{what} created {entries} output entries, over the cap of "
                    f"{self._cfg.max_output_entries}"
                )
            if total > self._cfg.max_batch_output_bytes:
                raise OversizeOutputError(
                    f"{what} wrote {total} output bytes, over the cap of "
                    f"{self._cfg.max_batch_output_bytes}"
                )
        return _ExecutionResult(
            returncode=returncode,
            elapsed=time.monotonic() - started,
            log_tail=logs.tail(),
            log_text=logs.text(),
        )

    def _snapshot_watched(
        self, lease: ModalSandboxLease, *, batch_index: int
    ) -> object:
        """Freeze output while continuing the provider-side byte watchdog.

        A solution can leave a background process after ``run.sh`` exits.  The
        output directory therefore remains adversarial until the immutable
        snapshot exists and the GPU Sandbox is terminated.
        """
        box: dict[str, Any] = {}
        done = threading.Event()

        def invoke() -> None:
            try:
                box["value"] = lease.snapshot_directory(
                    _REMOTE_OUTPUT, ttl_seconds=self._cfg.snapshot_ttl_seconds
                )
            except BaseException as exc:
                box["error"] = exc
            finally:
                done.set()

        threading.Thread(
            target=invoke,
            name=f"{self._runtime.run_label}-output-snapshot",
            daemon=True,
        ).start()
        deadline = time.monotonic() + min(300.0, self._cfg.batch_timeout_seconds)
        while not done.wait(self._cfg.output_poll_seconds):
            if time.monotonic() >= deadline:
                lease.terminate()
                raise BatchExecutionError(
                    f"freezing batch {batch_index} output exceeded the bounded "
                    "snapshot deadline and the Sandbox was terminated"
                )
            try:
                total, entries = self._remote_tree_usage(lease, _REMOTE_OUTPUT)
            except (OutputRejectedError, OversizeOutputError):
                lease.terminate()
                raise
            except Exception as exc:
                lease.terminate()
                raise BatchExecutionError(
                    f"batch {batch_index} snapshot watchdog failed: "
                    f"{_modal_error_text(exc)}"
                ) from exc
            if entries > self._cfg.max_output_entries:
                lease.terminate()
                raise OversizeOutputError(
                    f"batch {batch_index} created {entries} output entries while "
                    f"freezing, over the cap of {self._cfg.max_output_entries}"
                )
            if total > self._cfg.max_batch_output_bytes:
                lease.terminate()
                raise OversizeOutputError(
                    f"batch {batch_index} wrote {total} output bytes while freezing, "
                    f"over the cap of {self._cfg.max_batch_output_bytes}"
                )
        if "error" in box:
            exc = box["error"]
            raise BatchExecutionError(
                f"could not freeze batch {batch_index} output before collection: "
                f"{_modal_error_text(exc)}"
            ) from exc
        return box["value"]

    def _remote_tree_usage(
        self, lease: ModalSandboxLease, root: str
    ) -> tuple[int, int]:
        root_path = PurePosixPath(root)
        for rescan in range(_REMOTE_TREE_RESCAN_LIMIT + 1):
            total = 0
            count = 0
            pending = [root_path]
            seen: set[str] = set()
            restart = False
            while pending:
                directory = pending.pop()
                key = str(directory)
                if key in seen:
                    raise OutputRejectedError(
                        f"remote output directory cycle at {key!r}"
                    )
                seen.add(key)
                try:
                    entries = lease.list_files(key)
                except FileNotFoundError:
                    if directory == root_path:
                        raise
                    # Contenders publish atomically by renaming a temporary child.
                    # Restart from the root so the watchdog charges the final name;
                    # silently ignoring the vanished child could undercount bytes.
                    restart = True
                    break
                for entry in entries:
                    count += 1
                    if count > self._cfg.max_output_entries:
                        return total, count
                    path = PurePosixPath(entry.path)
                    if path == root_path or root_path not in path.parents:
                        raise OutputRejectedError(
                            f"remote output entry {entry.path!r} escaped {root!r}"
                        )
                    total += max(0, int(entry.size))
                    if total > self._cfg.max_batch_output_bytes:
                        return total, count
                    if entry.kind == "directory":
                        pending.append(path)
            if not restart:
                return total, count
            if rescan == _REMOTE_TREE_RESCAN_LIMIT:
                break
        raise OutputRejectedError(
            "remote output tree kept changing during bounded watchdog traversal"
        )

    def _collect_outputs(
        self,
        collector: ModalSandboxLease,
        local_dir: Path,
        items: Sequence[BatchItem],
        elapsed: float,
    ) -> list[BatchOutput]:
        outputs: list[BatchOutput] = []
        total = 0
        for item in items:
            remote = f"{_COLLECTED_OUTPUT}/{item.input_sha256}"
            try:
                info = collector.stat(remote)
            except FileNotFoundError:
                continue
            except Exception as exc:
                raise BatchExecutionError(
                    f"item {item.item_id}: could not stat frozen output: "
                    f"{_modal_error_text(exc)}"
                ) from exc
            if info.kind != "file":
                raise OutputRejectedError(
                    f"item {item.item_id}: frozen output is {info.kind}, not a regular file"
                )
            if info.size > self._cfg.max_output_bytes:
                raise OversizeOutputError(
                    f"item {item.item_id}: output is {info.size} bytes, over the "
                    f"per-output cap of {self._cfg.max_output_bytes}"
                )
            total += info.size
            if total > self._cfg.max_batch_output_bytes:
                raise OversizeOutputError(
                    f"batch outputs total {total} bytes, over the per-batch cap of "
                    f"{self._cfg.max_batch_output_bytes}"
                )
            local = local_dir / item.input_sha256
            try:
                collector.copy_to_local(remote, local)
                st = safeio.lstat_regular(local, what="frozen Modal output")
                digest, size = safeio.hash_into_pool(
                    local,
                    st,
                    self._outputs_dir,
                    max_bytes=self._cfg.max_output_bytes,
                    what="frozen Modal output",
                )
            except UnsafePathError as exc:
                raise OutputRejectedError(f"item {item.item_id}: {exc}") from exc
            except OversizeOutputError:
                raise
            except Exception as exc:
                raise BatchExecutionError(
                    f"item {item.item_id}: could not copy frozen output: "
                    f"{_modal_error_text(exc)}"
                ) from exc
            if size != info.size:
                raise BatchExecutionError(
                    f"item {item.item_id}: immutable snapshot size changed during "
                    f"collection ({info.size} -> {size})"
                )
            outputs.append(
                BatchOutput(
                    item_id=item.item_id,
                    output_sha256=digest,
                    output_bytes=size,
                    wall_seconds=elapsed,
                )
            )
        return outputs

    def _stage_input(self, item: BatchItem, destination: Path) -> None:
        if not _DIGEST_RE.fullmatch(item.input_sha256):
            raise InputStagingError(
                f"item {item.item_id}: invalid input sha256 {item.input_sha256!r}"
            )
        source = self._inputs_dir / item.input_sha256
        target = destination / item.input_sha256
        try:
            st = safeio.lstat_regular(source, what="sealed input")
            safeio.assert_within(source, self._inputs_dir, what="sealed input")
            if st.st_size != item.input_bytes:
                raise InputStagingError(
                    f"item {item.item_id}: sealed input size {st.st_size} != declared "
                    f"{item.input_bytes}"
                )
            digest = hashlib.sha256()
            copied = 0
            with safeio.open_regular_nofollow(source, st, what="sealed input") as src:
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
                with os.fdopen(fd, "wb") as dst:
                    while chunk := src.read(safeio.CHUNK):
                        copied += len(chunk)
                        digest.update(chunk)
                        dst.write(chunk)
            if copied != item.input_bytes or digest.hexdigest() != item.input_sha256:
                target.unlink(missing_ok=True)
                raise InputStagingError(
                    f"item {item.item_id}: sealed input bytes do not match digest/size"
                )
            if item.upscale_factor is not None:
                if item.target_width is None or item.target_height is None:
                    raise InputStagingError(
                        f"item {item.item_id}: committed target geometry is missing"
                    )
                sidecar = destination / upscale_task_sidecar_name(item.input_sha256)
                payload = upscale_task_sidecar_bytes(
                    item.upscale_factor,
                    item.target_width,
                    item.target_height,
                )
                fd = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
                with os.fdopen(fd, "wb") as factor_file:
                    os.fchmod(factor_file.fileno(), 0o444)
                    factor_file.write(payload)
        except InputStagingError:
            raise
        except (FileNotFoundError, UnsafePathError, OSError, ValueError) as exc:
            raise InputStagingError(
                f"item {item.item_id}: could not stage sealed input "
                f"{item.input_sha256}: {exc}"
            ) from exc

    def _probe_report(
        self,
        att: SandboxRequestAttestation,
        stdout: str,
        note: str,
        overlay_att: SandboxRequestAttestation | None,
        overlay_stdout: str,
        overlay_note: str,
    ) -> IsolationProbeReport:
        parsed = _parse_probe(stdout)
        overlay_parsed = _parse_probe(overlay_stdout)
        completed = parsed.get("PROBE_DONE") == "1"
        overlay_completed = overlay_parsed.get("OVERLAY_CHECK_DONE") == "1"
        env_names = parsed.get("env_names", [])
        secret_shaped = sorted(
            name for name in env_names if _SECRET_NAME_RE.search(name)
        )
        base = att.base_isolated and att.image_mounts == (_REMOTE_INPUT,)
        overlay_base = (
            overlay_att is not None
            and overlay_att.base_isolated
            and overlay_att.gpu is None
            and overlay_att.image_mounts == (_REMOTE_INPUT,)
        )
        input_write = parsed.get("INPUT_WRITE")
        input_write_reported = input_write in {"0", "1"}
        input_base_unchanged = (
            overlay_base
            and overlay_completed
            and overlay_parsed.get("INPUT_BASE_MUTATED") == "0"
        )
        report = IsolationProbeReport(
            network_blocked=(
                base and completed and parsed.get("NETWORK_ATTEMPT") != "1"
            ),
            secrets_absent=base and completed and not secret_shaped,
            reference_mounts_absent=(
                base
                and completed
                and input_write_reported
                and parsed.get("REF_MOUNTS") == "0"
                and input_base_unchanged
            ),
            index_leak_absent=(base and completed and parsed.get("INDEX_LEAK") == "0"),
            details=json.dumps(
                {
                    "trust": (
                        "exact host-side Modal Sandbox.create request plus "
                        "advisory-negative in-container probe; Modal exposes no "
                        "public full-config readback API"
                    ),
                    "run_label": self._runtime.run_label,
                    "request": {
                        "name": att.name,
                        "role": att.role,
                        "block_network": att.block_network,
                        "secret_count": att.secret_count,
                        "env_keys": list(att.env_keys),
                        "include_oidc_identity_token": att.include_oidc_identity_token,
                        "volume_mounts": list(att.volume_mounts),
                        "network_filesystem_mounts": list(
                            att.network_filesystem_mounts
                        ),
                        "ports": list(att.ports),
                        "gpu": att.gpu,
                        "image_mounts": list(att.image_mounts),
                    },
                    "probe": {
                        "completed": completed,
                        "note": note,
                        "network_attempt": parsed.get("NETWORK_ATTEMPT"),
                        "input_write": parsed.get("INPUT_WRITE"),
                        "reference_mounts": parsed.get("REF_MOUNTS"),
                        "index_leak": parsed.get("INDEX_LEAK"),
                        "secret_shaped_env": secret_shaped,
                    },
                    "input_image_overlay_check": {
                        "completed": overlay_completed,
                        "note": overlay_note,
                        "input_base_mutated": overlay_parsed.get("INPUT_BASE_MUTATED"),
                        "request": None
                        if overlay_att is None
                        else {
                            "name": overlay_att.name,
                            "role": overlay_att.role,
                            "block_network": overlay_att.block_network,
                            "secret_count": overlay_att.secret_count,
                            "env_keys": list(overlay_att.env_keys),
                            "include_oidc_identity_token": (
                                overlay_att.include_oidc_identity_token
                            ),
                            "volume_mounts": list(overlay_att.volume_mounts),
                            "network_filesystem_mounts": list(
                                overlay_att.network_filesystem_mounts
                            ),
                            "ports": list(overlay_att.ports),
                            "gpu": overlay_att.gpu,
                            "image_mounts": list(overlay_att.image_mounts),
                        },
                        "semantics": (
                            "Modal may expose a sandbox-local writable overlay; "
                            "the fresh CPU remount proves that the write did not "
                            "persist into the content-bound input Image"
                        ),
                    },
                },
                sort_keys=True,
            ),
        )
        logger.info(
            "fresh Modal isolation probe finished",
            extra=log_fields(
                run_label=self._runtime.run_label,
                sandbox_name=att.name,
                passed=report.passed,
                details=report.details,
            ),
        )
        return report


class _ModalContenderImageBuildError(Exception):
    """Modal explicitly rejected the contender-controlled image build."""


class _SdkProcess:
    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self.stdout = raw.stdout
        self.stderr = raw.stderr

    def poll(self) -> int | None:
        return self._raw.poll()


class _SdkSandboxLease:
    def __init__(
        self, raw: Any, request: SandboxRequestAttestation, modal_module: Any
    ) -> None:
        self._raw = raw
        self._request = request
        self._modal = modal_module
        self._image_mounts: list[str] = []
        self._terminated = False
        self._detached = False
        self.object_id = str(raw.object_id)

    def attestation(self) -> SandboxRequestAttestation:
        return SandboxRequestAttestation(
            **{**self._request.__dict__, "image_mounts": tuple(self._image_mounts)}
        )

    def mount_image(self, path: str, image: object) -> None:
        if path in self._image_mounts:
            raise RuntimeError(f"image already mounted at {path}")
        self._raw.mount_image(path, image)
        self._image_mounts.append(path)

    def make_directory(self, path: str) -> None:
        self._raw.filesystem.make_directory(path)

    def write_text(self, path: str, value: str) -> None:
        self._raw.filesystem.write_text(value, path)

    def exec(self, args: Sequence[str], *, timeout_seconds: float) -> RemoteProcess:
        stream_type = importlib.import_module("modal.stream_type").StreamType
        raw = self._raw.exec(
            *args,
            stdout=stream_type.PIPE,
            stderr=stream_type.PIPE,
            timeout=max(1, math.ceil(timeout_seconds)),
            text=False,
            bufsize=-1,
            env={},
            secrets=[],
        )
        return _SdkProcess(raw)

    def list_files(self, path: str) -> Sequence[RemoteFile]:
        try:
            entries = self._raw.filesystem.list_files(path)
        except self._modal.exception.SandboxFilesystemNotFoundError as exc:
            raise FileNotFoundError(path) from exc
        return [
            RemoteFile(
                path=str(entry.path), kind=str(entry.type.value), size=int(entry.size)
            )
            for entry in entries
        ]

    def stat(self, path: str) -> RemoteFile:
        try:
            entry = self._raw.filesystem.stat(path)
        except self._modal.exception.SandboxFilesystemNotFoundError as exc:
            raise FileNotFoundError(path) from exc
        return RemoteFile(
            path=str(entry.path), kind=str(entry.type.value), size=int(entry.size)
        )

    def copy_to_local(self, remote_path: str, local_path: Path) -> None:
        self._raw.filesystem.copy_to_local(remote_path, local_path)

    def snapshot_directory(self, path: str, *, ttl_seconds: int) -> object:
        return self._raw.snapshot_directory(path, timeout=300, ttl=ttl_seconds)

    def terminate(self) -> None:
        if not self._terminated:
            self._raw.terminate(wait=True)
            self._terminated = True

    def detach(self) -> None:
        if not self._detached:
            self._raw.detach()
            self._detached = True


class ModalSdkRuntime:
    """Current Modal adapter with an explicit create-only ownership boundary."""

    def __init__(self) -> None:
        raise TypeError("use ModalSdkRuntime.start_fresh(...)")

    @classmethod
    def start_fresh(
        cls,
        *,
        environment_name: str,
        app_name: str,
        run_label: str,
        confirmation: str,
    ) -> ModalSdkRuntime:
        environment_name = _validate_resource_name(
            environment_name, what="fresh Modal Environment name"
        )
        app_name = _validate_resource_name(app_name, what="fresh Modal App name")
        run_label = _validate_resource_name(run_label, what="fresh Modal run label")
        if len({environment_name, app_name, run_label}) != 3:
            raise RunnerUnavailableError(
                "fresh Modal Environment/App/run names must be distinct; mint "
                "three new vidaio-next-* identities for this run"
            )
        if confirmation != FRESH_CREATION_CONFIRMATION:
            raise RunnerUnavailableError(
                "Modal creation is disabled until the caller supplies the exact "
                f"confirmation {FRESH_CREATION_CONFIRMATION!r}"
            )
        modal = importlib.import_module("modal")
        self = object.__new__(cls)
        self.run_label = run_label
        self._modal = modal
        self._environment_name = environment_name
        self._app_name = app_name
        self._app = modal.App(
            app_name,
            tags={
                "vidaio-resource": "vidaio-next",
                "vidaio-run": run_label,
                "vidaio-purpose": "competition-sandbox",
            },
        )
        self._context = self._app.run(
            name=app_name, environment_name=environment_name, detach=False
        )
        self._active = False
        self._leases: list[_SdkSandboxLease] = []
        self._collector_image: object | None = None
        try:
            self._context.__enter__()
            self._active = True
        except Exception as exc:
            try:
                # ``__enter__`` can fail after partially creating the fresh App.
                # Close that exact owned context; never discover or resolve it.
                self._context.__exit__(type(exc), exc, exc.__traceback__)
            except Exception as cleanup_exc:
                exc.add_note(
                    "fresh Modal App context cleanup also failed: "
                    f"{_modal_error_text(cleanup_exc)}"
                )
            raise RunnerUnavailableError(
                f"could not create fresh Modal App {app_name!r} in fresh "
                f"Environment {environment_name!r}; mint new names, do not inspect "
                f"or reuse a collision: {_modal_error_text(exc)}"
            ) from exc
        return self

    def available(self) -> bool:
        # Construction already did the authenticated create; no inventory probe.
        return bool(self._active)

    def build_contender_image(
        self, checkout: Path, dockerfile: Path
    ) -> tuple[object, str]:
        self._assert_active()
        try:
            image = (
                self._modal.Image.from_dockerfile(
                    dockerfile,
                    context_dir=checkout,
                    force_build=True,
                    secrets=[],
                    gpu=None,
                )
                .entrypoint([])
                .cmd(list(_KEEPALIVE_COMMAND))
            )
            image.build(self._app)
        except self._modal.exception.ImageBuildError as exc:
            raise _ModalContenderImageBuildError(
                f"contender Dockerfile was rejected by Modal: {_modal_error_text(exc)}"
            ) from exc
        return image, str(image.object_id)

    def restore_contender_image(self, image_object_id: str) -> tuple[object, str]:
        """Rehydrate one exact deployment-owned immutable Image; never discover.

        The orchestrator supplies an id from its append-only competition binding.
        No list/lookup-by-name call is made, and no Sandbox or instance is reused.
        """
        self._assert_active()
        if not _IMAGE_OBJECT_ID_RE.fullmatch(image_object_id):
            raise ValueError(f"invalid Modal image object id {image_object_id!r}")
        image = self._modal.Image.from_id(image_object_id)
        restored_id = str(image.object_id)
        if restored_id != image_object_id:
            raise RuntimeError(
                "Modal Image.from_id returned a different object identity"
            )
        return image, restored_id

    def build_input_image(self, staged_inputs: Path) -> object:
        self._assert_active()
        image = self._modal.Image.from_scratch(force_build=True).add_local_dir(
            staged_inputs,
            "/",
            copy=True,
            # Factor metadata is intentionally a dotfile. Modal's empty ignore
            # matcher includes it; keep that choice explicit in the live adapter.
            ignore=[],
        )
        image.build(self._app)
        return image

    def create_sandbox(
        self,
        *,
        image: object,
        name: str,
        role: str,
        tags: dict[str, str],
        gpu: str | None,
        cpu: float,
        memory_mb: int,
        timeout_seconds: int,
        idle_timeout_seconds: int,
    ) -> ModalSandboxLease:
        self._assert_active()
        _validate_resource_name(name, what="fresh Modal Sandbox name")
        request = SandboxRequestAttestation(
            name=name,
            role=role,
            block_network=True,
            secret_count=0,
            env_keys=(),
            include_oidc_identity_token=False,
            volume_mounts=(),
            network_filesystem_mounts=(),
            ports=(),
            gpu=gpu,
        )
        # A contender image's CMD may exit immediately (the shipped examples use
        # ``/bin/sh``). Keep the Sandbox alive under a trusted, argument-only
        # command so subsequent ``exec`` and filesystem calls have a live target.
        raw = self._modal.Sandbox.create(
            *_KEEPALIVE_COMMAND,
            app=self._app,
            name=name,
            tags=tags,
            image=image,
            env={},
            secrets=[],
            network_file_systems={},
            timeout=int(timeout_seconds),
            idle_timeout=int(idle_timeout_seconds),
            gpu=gpu,
            cpu=(float(cpu), float(cpu)),
            memory=(int(memory_mb), int(memory_mb)),
            block_network=True,
            volumes={},
            encrypted_ports=[],
            h2_ports=[],
            unencrypted_ports=[],
            include_oidc_identity_token=False,
        )
        lease = _SdkSandboxLease(raw, request, self._modal)
        self._leases.append(lease)
        return lease

    def create_collector_sandbox(
        self,
        *,
        name: str,
        tags: dict[str, str],
        timeout_seconds: int,
    ) -> ModalSandboxLease:
        self._assert_active()
        if self._collector_image is None:
            image = self._modal.Image.debian_slim(
                python_version="3.12", force_build=True
            )
            image.build(self._app)
            self._collector_image = image
        return self.create_sandbox(
            image=self._collector_image,
            name=name,
            role="collector",
            tags=tags,
            gpu=None,
            cpu=0.25,
            memory_mb=512,
            timeout_seconds=timeout_seconds,
            idle_timeout_seconds=min(120, timeout_seconds),
        )

    def close(self) -> None:
        if not getattr(self, "_active", False):
            return
        for lease in reversed(self._leases):
            try:
                lease.terminate()
            except Exception:  # noqa: BLE001 - cleanup best effort
                pass
            try:
                lease.detach()
            except Exception:  # noqa: BLE001 - cleanup best effort
                pass
        self._leases.clear()
        self._active = False
        self._context.__exit__(None, None, None)

    def _assert_active(self) -> None:
        if not self._active:
            raise RunnerUnavailableError("the fresh Modal App context is closed")


def _parse_probe(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    env_names: list[str] = []
    in_env = False
    for line in stdout.splitlines():
        line = line.strip()
        if line == "ENV_BEGIN":
            in_env = True
            continue
        if line == "ENV_END":
            in_env = False
            continue
        if in_env and "=" in line:
            env_names.append(line.split("=", 1)[0])
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            if key in {
                "NETWORK_ATTEMPT",
                "INPUT_WRITE",
                "REF_MOUNTS",
                "INDEX_LEAK",
                "PROBE_DONE",
                "INPUT_BASE_MUTATED",
                "OVERLAY_CHECK_DONE",
            }:
                parsed[key] = value
    parsed["env_names"] = env_names
    return parsed
