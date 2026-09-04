"""Real subprocess-backed metrics plus a CPU/CUDA-selectable PieAPP backend.

The scoring module stays pure (see :mod:`vidaio.scoring.backends`); everything that
shells out lives here. Discipline for every subprocess call:

* argv lists only — never ``shell=True``, never string interpolation into a shell;
* an explicit timeout on every call (an unbounded subprocess wait is a bug —
  the ``with_timeout`` rule of :mod:`vidaio.core.resilience` applied to processes);
* failures surface as typed errors carrying the argv and captured stderr, never as
  a silent default value;
* every child is started in its OWN session/process group (``start_new_session``),
  and a timeout (or an external cancellation) kills that whole GROUP — an ffmpeg
  that spawned helpers cannot outlive the call that started it.

Bounded output: a decode is an EXPANSION (raw y4m is orders of magnitude larger
than the encoding it came from), so a caller that has reserved disk for one can
pass ``max_output_bytes`` and have that reservation enforced — the process group
is killed the moment the file passes the bound and the call raises
:class:`CanonicalizationTooLarge` instead of running the volume out of space.

Scratch placement (:func:`use_media_scratch`): backends are composed once and
shared by every concurrent request, so the temp dirs they create are placed
through a THREAD-LOCAL directory the caller installs for the unit of work,
not through instance state two requests would race on. The scoring worker
installs each request's own scratch directory, which is what puts libvmaf's JSON
logs inside the same directory (and the same byte budget, and the same startup
sweep) as everything else that request writes.

Cancellation (:class:`MediaProcessScope`): the caller of a unit of media work
installs a scope for the duration of that work; every ``_run`` inside it registers
its child process with the scope. Cancelling the scope from another thread
SIGKILLs every registered process group immediately and makes the next ``_run``
refuse to start, so the worker thread unwinds with :class:`MediaWorkCancelled`
instead of continuing to burn CPU after its caller gave up. This is what makes an
HTTP request timeout bound *real work* and not just the awaiting coroutine.

VMAF determinism (spec §08 recomputability): the libvmaf filter is invoked with
``n_subsample=1`` (every frame — no sampling), ``n_threads=1`` (fixed accumulation
order), ``pool=mean`` and an explicitly pinned model, so the same (reference,
candidate) pair always yields the same pooled score. There is no randomness to seed;
``deterministic_seed`` is accepted for Protocol compatibility and ignored.

Secondary model choice (the ``vmaf_model_delta`` gate): ``vmaf_v0.6.1neg`` — the
NEG ("no enhancement gain") variant bundled with libvmaf. It clips exactly the
sharpening/contrast "enhancement" tricks that inflate the default model, so a large
primary-vs-NEG delta is precisely the model-gaming signal the gate hunts. The 4k
model was rejected as secondary: it is calibrated for a different viewing distance
and legitimately diverges from the default model at small resolutions, which would
make the delta gate noisy instead of adversarial-sensitive.

PieAPP is provided through PIQ/PyTorch and accepts an explicit ``cpu`` or
``cuda`` device. Auditors always select CPU. CUDA remains an experimental/dev
backend; the production guard requires CPU because no cross-device tolerance has
been launch-calibrated. Missing optional packages or unavailable model weights raise
:class:`NotConfiguredError`, never a substituted distance. The three perceptual
manipulation checks use deterministic CPU/OpenCV sampling and integer reductions.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Sequence

from vidaio.scoring.backends import MediaInfo, PerceptualCheckResult
from vidaio.scoring.perceptual_cpu import (
    CPU_PERCEPTUAL_ALGORITHM_VERSION,
    CpuPerceptualConfig,
    PerceptualStatistics,
    chroma_uv_result,
    grayscale_result,
    tone_manipulation_result,
)

#: Pinned libvmaf models (bundled with libvmaf — no external model files).
DEFAULT_VMAF_MODEL = "version=vmaf_v0.6.1"
SECONDARY_VMAF_MODEL = "version=vmaf_v0.6.1neg"

# PIQ v0.8.0's immutable release asset. We fetch it into torch.hub's normal
# cache ourselves so its full digest is verified *before* torch deserializes it.
PIEAPP_WEIGHTS_URL = (
    "https://github.com/photosynthesis-team/piq/releases/download/v0.5.4/PieAPPv0.1.pth"
)
PIEAPP_WEIGHTS_FILENAME = "PieAPPv0.1.pth"
PIEAPP_WEIGHTS_SHA256 = (
    "0937b01480c7a637ae3018af755faa8ecde4788b52bb246b7ae62cf96fb6baf0"
)

#: Default per-subprocess timeout (seconds). Every call site may narrow it.
DEFAULT_SUBPROCESS_TIMEOUT = 120.0

#: Grace between SIGTERM and SIGKILL when a subprocess overruns its own timeout.
#: (Scope cancellation does not wait: it SIGKILLs the group outright, because the
#: caller has already given up on the result and may be the event-loop thread.)
TERMINATE_GRACE_SECONDS = 1.0

#: Temp-dir prefixes this module creates. Exported so the scoring worker's
#: startup sweep can reclaim the ones a crash orphaned — a temp dir whose owning
#: process died is never cleaned by :class:`tempfile.TemporaryDirectory`.
VMAF_SCRATCH_PREFIX = "vmaf-"
VMAF_VERSION_SCRATCH_PREFIX = "vmafver-"

#: How often an output-size watchdog stats the file it is guarding. Short enough
#: that a runaway ffmpeg is stopped within a fraction of a second of crossing its
#: bound, long enough to be free next to the encode itself.
OUTPUT_WATCHDOG_POLL_SECONDS = 0.1

_STDERR_TAIL = 2000  # captured-stderr cap on typed errors (keep logs bounded)


# --- typed errors --------------------------------------------------------------------


class BackendError(Exception):
    """Base for every real-backend failure."""


class NotConfiguredError(BackendError):
    """The backend exists as a Protocol seam but is not runnable in this build."""


class MediaToolError(BackendError):
    """A media subprocess failed — carries the argv and captured stderr."""

    def __init__(self, message: str, *, argv: Sequence[str], stderr: str = "") -> None:
        self.argv = list(argv)
        self.stderr = stderr[-_STDERR_TAIL:]
        detail = f"{message}: {' '.join(self.argv)}"
        if self.stderr:
            detail = f"{detail}\nstderr: {self.stderr}"
        super().__init__(detail)


class MediaToolTimeout(MediaToolError):
    """A media subprocess exceeded its explicit timeout."""


class MediaOutputTooLarge(MediaToolError):
    """A media subprocess wrote more output than the caller reserved for it.

    Carries the bound and what was actually observed on disk, so the caller can
    turn it into a typed refusal that names real numbers.
    """

    def __init__(
        self,
        message: str,
        *,
        argv: Sequence[str],
        stderr: str = "",
        output_path: str,
        limit_bytes: int,
        observed_bytes: int,
    ) -> None:
        self.output_path = output_path
        self.limit_bytes = limit_bytes
        self.observed_bytes = observed_bytes
        super().__init__(message, argv=argv, stderr=stderr)


class CanonicalizationError(MediaToolError):
    """A canonicalization plan failed to execute."""


class CanonicalizationTimeout(CanonicalizationError, MediaToolTimeout):
    """A canonicalization plan exceeded its explicit timeout."""


class CanonicalizationTooLarge(CanonicalizationError, MediaOutputTooLarge):
    """A canonicalization plan expanded past the size that was reserved for it.

    The projection that reserved the scratch (see
    :func:`vidaio.scoring_worker.inputs.projected_canonical_bytes`) is exact for a
    well-formed source, but it is still a prediction of what a decoder will do
    with a file the miner controls. This is the enforcement that makes the
    reservation a BOUND: the process group is killed the moment its output passes
    the reserved cap, so a mis-projected request costs the volume the cap and not
    the disk.
    """


class MetricLogTooLarge(MediaOutputTooLarge):
    """A metric run's JSON log outgrew the scratch reserved for it.

    Round-4 an internal review: the 1 KiB/frame log estimate
    (:func:`vidaio.scoring_worker.inputs.projected_metric_log_bytes`) was
    reserved but never enforced, so a libvmaf configuration that logs more than
    the estimate (a wider model's feature set) wrote past its reservation for
    the whole length of a long clip. The caller installs the reserved per-run
    bound via :func:`use_metric_log_limit`; the same watchdog machinery that
    bounds a canonicalization then kills the process group past it, and the
    post-exit stat makes the verdict exact for runs that finish between polls.
    """


class MediaWorkCancelled(BackendError):
    """The unit of media work was cancelled — its process groups were killed.

    Deliberately NOT a :class:`MediaToolError`: nothing about the media tools
    failed, the caller withdrew. Callers map it to their own cancellation
    outcome (the scoring worker: a timeout), never to a tool error.
    """


# --- cancellation scope (process-group lifetime of one unit of work) -----------------


def kill_process_group(proc: "subprocess.Popen[Any]") -> None:
    """SIGKILL the process GROUP of `proc` (started with ``start_new_session``).

    Immediate and non-blocking: safe to call from the event-loop thread. Falls
    back to killing the single process if the group cannot be resolved (already
    reaped, or a platform without process groups).
    """
    if proc.poll() is not None:
        return
    _signal_group(proc, signal.SIGKILL)


def terminate_process_group(
    proc: "subprocess.Popen[Any]", *, grace: float = TERMINATE_GRACE_SECONDS
) -> None:
    """SIGTERM the process group, then SIGKILL it if it outlives `grace`.

    Blocking (up to `grace`) — for use from the worker thread that owns the
    process, e.g. when the subprocess overran its own timeout.
    """
    if proc.poll() is not None:
        return
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        _signal_group(proc, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=grace)


def _signal_group(proc: "subprocess.Popen[Any]", sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.send_signal(sig)


class MediaProcessScope:
    """Tracks the child processes of ONE unit of media work so they can be killed.

    Thread-safe by design: the work runs in a worker thread while
    :meth:`cancel` is typically called from the event loop when the request that
    asked for it timed out or was cancelled. After ``cancel()`` every registered
    process group is SIGKILLed and every subsequent ``_run`` inside the scope
    refuses to start, so the worker thread unwinds promptly.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[Any]] = set()
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def live_pids(self) -> list[int]:
        """PIDs currently registered (each is also its own process-group id)."""
        with self._lock:
            return [p.pid for p in self._processes]

    def register(self, proc: "subprocess.Popen[Any]") -> None:
        """Adopt `proc`; if the scope is already cancelled, kill it immediately."""
        with self._lock:
            self._processes.add(proc)
            cancelled = self._cancelled
        if cancelled:
            kill_process_group(proc)

    def unregister(self, proc: "subprocess.Popen[Any]") -> None:
        with self._lock:
            self._processes.discard(proc)

    def cancel(self) -> None:
        """Mark cancelled and SIGKILL every live process group. Non-blocking."""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            processes = list(self._processes)
        for proc in processes:
            kill_process_group(proc)

    def raise_if_cancelled(self, *, detail: str = "media work cancelled") -> None:
        if self.cancelled:
            raise MediaWorkCancelled(detail)


_ACTIVE_SCOPE = threading.local()


def current_process_scope() -> MediaProcessScope | None:
    """The scope installed on THIS thread, if any (``None`` outside a scope)."""
    return getattr(_ACTIVE_SCOPE, "scope", None)


@contextlib.contextmanager
def use_process_scope(
    scope: MediaProcessScope | None,
) -> Iterator[MediaProcessScope | None]:
    """Install `scope` for the calling thread for the duration of the block."""
    previous = current_process_scope()
    _ACTIVE_SCOPE.scope = scope
    try:
        yield scope
    finally:
        _ACTIVE_SCOPE.scope = previous


# --- per-request scratch placement ---------------------------------------------------
#
# Backends are composed ONCE and shared by every concurrent request, so a backend
# cannot own a per-request directory as instance state without two requests
# racing on it. The active directory is therefore thread-local, exactly like the
# process scope above and for the same reason: the unit of work is one executor
# thread, and installing it there is what lets the caller (the scoring worker)
# put every temp file a request creates INSIDE that request's own scratch
# directory — where it dies with the request and is visible to the same byte
# accounting and the same startup sweep.

_ACTIVE_SCRATCH = threading.local()


def current_media_scratch() -> str | None:
    """The scratch directory installed on THIS thread, if any."""
    return getattr(_ACTIVE_SCRATCH, "path", None)


@contextlib.contextmanager
def use_media_scratch(path: str | Path | None) -> Iterator[str | None]:
    """Install `path` as the temp-dir parent for media work on the calling thread."""
    previous = current_media_scratch()
    resolved = str(path) if path is not None else None
    _ACTIVE_SCRATCH.path = resolved
    try:
        yield resolved
    finally:
        _ACTIVE_SCRATCH.path = previous


# Thread-local like the scratch dir above and for the same reason: the backends
# are composed once and shared, so a per-request bound cannot live on the
# instance. The scoring worker installs the log bytes it RESERVED for one metric
# run; every libvmaf invocation on the thread is then hard-bounded to it
#.

_ACTIVE_METRIC_LOG_LIMIT = threading.local()


def current_metric_log_limit() -> int | None:
    """The per-run metric-log byte bound installed on THIS thread, if any."""
    return getattr(_ACTIVE_METRIC_LOG_LIMIT, "nbytes", None)


@contextlib.contextmanager
def use_metric_log_limit(nbytes: int | None) -> Iterator[int | None]:
    """Bound every metric run's log file on the calling thread to `nbytes`.

    ``None`` restores the unbounded historical behaviour (fake mode, version
    probes — places where nothing was reserved and the output is known small).
    """
    previous = current_metric_log_limit()
    _ACTIVE_METRIC_LOG_LIMIT.nbytes = nbytes
    try:
        yield nbytes
    finally:
        _ACTIVE_METRIC_LOG_LIMIT.nbytes = previous


# --- output-size watchdog ------------------------------------------------------------


class _OutputSizeWatchdog:
    """Kills a child's process group if the file it writes passes `limit` bytes.

    Polling, because there is no portable way to be woken by a write. That leaves
    a window between two polls in which the child can overshoot, so the caller
    ALSO re-stats the file after the process exits (:func:`_run`): the watchdog
    bounds how much a runaway can write, the post-exit check makes the verdict
    exact. Same shape as the log/output measurement rule elsewhere in this
    module — measure while it runs and again once it has stopped.
    """

    def __init__(
        self,
        proc: "subprocess.Popen[Any]",
        path: str,
        limit: int,
        *,
        poll_interval: float = OUTPUT_WATCHDOG_POLL_SECONDS,
    ) -> None:
        self._proc = proc
        self._path = path
        self._limit = limit
        self._poll_interval = poll_interval
        self._done = threading.Event()
        self._tripped = threading.Event()
        self.observed_bytes = 0
        self._thread = threading.Thread(
            target=self._watch, name="media-output-watchdog", daemon=True
        )

    @property
    def tripped(self) -> bool:
        return self._tripped.is_set()

    def _watch(self) -> None:
        while not self._done.wait(self._poll_interval):
            size = _file_size(self._path)
            if size > self._limit:
                self.observed_bytes = size
                self._tripped.set()
                kill_process_group(self._proc)
                return

    def __enter__(self) -> "_OutputSizeWatchdog":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._done.set()
        self._thread.join(timeout=self._poll_interval * 20)


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:  # not created yet, or already gone
        return 0


# --- shared subprocess runner --------------------------------------------------------


def _run(
    argv: Sequence[str],
    *,
    timeout: float,
    error_cls: type[MediaToolError] = MediaToolError,
    timeout_cls: type[MediaToolTimeout] = MediaToolTimeout,
    output_limit: tuple[str, int] | None = None,
    oversize_cls: type[MediaOutputTooLarge] = MediaOutputTooLarge,
) -> subprocess.CompletedProcess[str]:
    """Run one argv (no shell) with an explicit timeout; typed errors on failure.

    The child is its own session leader (``start_new_session=True``), so both the
    timeout path and :meth:`MediaProcessScope.cancel` kill the whole process group
    — no ffmpeg (or helper it spawned) survives the call that started it.

    `output_limit` is an optional ``(path, max_bytes)`` bound on the file this run
    writes: crossing it kills the process group and raises `oversize_cls`, with
    the partial output left in place for the caller to remove (this function does
    not own the file it guards). Both the running check and a final post-exit
    stat are applied, so an overshoot between two polls is still caught.
    """
    scope = current_process_scope()
    if scope is not None:
        scope.raise_if_cancelled(detail=f"cancelled before {argv[0]!r} started")
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv list, shell=False by default
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise error_cls(f"binary not found ({argv[0]!r})", argv=argv) from exc
    if scope is not None:
        scope.register(proc)
    watchdog: _OutputSizeWatchdog | None = None
    try:
        with contextlib.ExitStack() as guards:
            if output_limit is not None:
                watchdog = guards.enter_context(
                    _OutputSizeWatchdog(proc, output_limit[0], output_limit[1])
                )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate_process_group(proc)
                stdout, stderr = proc.communicate()
                raise timeout_cls(
                    f"timed out after {timeout}s", argv=argv, stderr=stderr or ""
                ) from None
    finally:
        if scope is not None:
            scope.unregister(proc)
    if scope is not None and scope.cancelled:
        # The group was killed out from under us — report the withdrawal, not a
        # media-tool failure (the nonzero exit code is our own SIGKILL). Checked
        # before the size verdict: a cancelled request was not refused on size.
        raise MediaWorkCancelled(f"cancelled while running {argv[0]!r}")
    if output_limit is not None:
        path, limit = output_limit
        # Measured again now the writer has stopped: the poll loop bounds the
        # overshoot, this makes the verdict exact for a run that finished between
        # two polls.
        observed = max(_file_size(path), watchdog.observed_bytes if watchdog else 0)
        if observed > limit or (watchdog is not None and watchdog.tripped):
            raise oversize_cls(
                f"output exceeded its {limit}-byte bound ({observed} bytes on disk)",
                argv=argv,
                stderr=stderr or "",
                output_path=path,
                limit_bytes=limit,
                observed_bytes=observed,
            )
    if proc.returncode != 0:
        raise error_cls(f"exited {proc.returncode}", argv=argv, stderr=stderr or "")
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)


def _escape_filter_option(value: str) -> str:
    """Escape a value for use inside an ffmpeg filter-graph option string."""
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


# --- ffprobe -------------------------------------------------------------------------


class FfprobeBackend:
    """ProbeBackend over ``ffprobe -print_format json -show_format -show_streams``."""

    name = "ffprobe"
    version = "unknown"

    def __init__(
        self,
        ffprobe_path: str = "ffprobe",
        *,
        timeout: float = DEFAULT_SUBPROCESS_TIMEOUT,
    ) -> None:
        self._ffprobe = ffprobe_path
        self._timeout = timeout

    def probe(self, path: str) -> MediaInfo:
        argv = [
            self._ffprobe,
            "-hide_banner",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ]
        completed = _run(argv, timeout=self._timeout)
        try:
            doc = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise MediaToolError("unparseable ffprobe output", argv=argv) from exc
        stream = next(
            (s for s in doc.get("streams", []) if s.get("codec_type") == "video"), None
        )
        if stream is None:
            raise MediaToolError("no video stream found", argv=argv)
        fmt = doc.get("format", {})

        fps = _parse_fraction(stream.get("avg_frame_rate", ""))
        if fps <= 0:
            fps = _parse_fraction(stream.get("r_frame_rate", ""))
        frame_count = _parse_int(stream.get("nb_frames"))
        if frame_count is None:
            frame_count = self._count_frames(path)
        duration = _parse_float(stream.get("duration"))
        if duration is None:
            duration = _parse_float(fmt.get("duration"))
        if duration is None:
            duration = frame_count / fps if fps > 0 else 0.0
        byte_size = _parse_int(fmt.get("size"))
        if byte_size is None:
            byte_size = os.path.getsize(path)
        pix_fmt = stream.get("pix_fmt") or "unknown"
        bit_depth = _parse_int(stream.get("bits_per_raw_sample"))
        if bit_depth is None:
            bit_depth = _bit_depth_from_pix_fmt(pix_fmt)

        return MediaInfo(
            codec=stream.get("codec_name") or "unknown",
            width=int(stream.get("width") or 0),
            height=int(stream.get("height") or 0),
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            byte_size=byte_size,
            bit_depth=bit_depth,
            pix_fmt=pix_fmt,
        )

    def _count_frames(self, path: str) -> int:
        """Fallback for containers without ``nb_frames`` (e.g. y4m): decode + count."""
        argv = [
            self._ffprobe,
            "-hide_banner",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-print_format",
            "json",
            path,
        ]
        completed = _run(argv, timeout=self._timeout)
        try:
            streams = json.loads(completed.stdout).get("streams", [])
            counted = _parse_int(streams[0].get("nb_read_frames")) if streams else None
        except (json.JSONDecodeError, IndexError, AttributeError) as exc:
            raise MediaToolError("unparseable frame-count output", argv=argv) from exc
        if counted is None:
            raise MediaToolError("frame count unavailable", argv=argv)
        return counted


def _parse_fraction(text: str) -> float:
    num, _, den = (text or "").partition("/")
    try:
        denominator = float(den) if den else 1.0
        return float(num) / denominator if denominator else 0.0
    except ValueError:
        return 0.0


def _parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bit_depth_from_pix_fmt(pix_fmt: str) -> int:
    match = re.search(r"(\d{1,2})(?:le|be)?$", pix_fmt)
    if match:
        depth = int(match.group(1))
        if 8 <= depth <= 16:
            return depth
    return 8


# --- ffmpeg libvmaf ------------------------------------------------------------------


class FfmpegVmafBackend:
    """VmafBackend over the ffmpeg libvmaf filter, pinned to one explicit model.

    One instance per model: the worker composes a primary instance
    (:data:`DEFAULT_VMAF_MODEL`) and a secondary instance
    (:data:`SECONDARY_VMAF_MODEL`) for the model-delta gate — both satisfy the
    ``VmafBackend`` Protocol exactly. See the module docstring for the NEG-model
    rationale and the determinism settings.
    """

    name = "ffmpeg-libvmaf"

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        *,
        model: str = DEFAULT_VMAF_MODEL,
        work_dir: str | Path | None = None,
        timeout: float = DEFAULT_SUBPROCESS_TIMEOUT,
    ) -> None:
        self._ffmpeg = ffmpeg_path
        self.model = model
        self._work_dir = str(work_dir) if work_dir is not None else None
        self._timeout = timeout
        self.version = "unknown"  # libvmaf version; cached from the first JSON log

    def scratch_root(self) -> str | None:
        """Where this backend's temp dirs go: the active request's scratch if any.

        Inside a scored request the worker installs that request's own directory
        (:func:`use_media_scratch`), so the JSON logs libvmaf writes live under
        the same directory as the snapshots and the y4m files — they die when it
        does, they are covered by the same byte accounting, and a crash leaves
        them where the same startup sweep finds them. Outside a request (version
        probing at composition time) it falls back to the configured work dir.
        """
        active = current_media_scratch()
        return active if active is not None else self._work_dir

    def compute(
        self, reference: str, candidate: str, *, deterministic_seed: int = 0
    ) -> float:
        """Pooled (mean) VMAF of `candidate` against `reference`, 0-100 scale.

        Full clip, no subsampling, single thread, pinned model — deterministic by
        construction; ``deterministic_seed`` is Protocol surface only (unused).
        """
        del deterministic_seed  # no stochastic settings exist to seed
        with tempfile.TemporaryDirectory(
            dir=self.scratch_root(), prefix=VMAF_SCRATCH_PREFIX
        ) as tmp:
            log_path = os.path.join(tmp, "vmaf.json")
            argv = self._argv(reference, candidate, log_path)
            # Round-4 an internal review: the JSON log grows with every frame, and the
            # caller reserved scratch for it from an estimate. When a per-run
            # bound is installed (use_metric_log_limit), the watchdog enforces
            # the reservation exactly like a canonicalization's: process group
            # killed past the bound, post-exit stat for exactness, typed error.
            log_limit = current_metric_log_limit()
            _run(
                argv,
                timeout=self._timeout,
                output_limit=None if log_limit is None else (log_path, log_limit),
                oversize_cls=MetricLogTooLarge,
            )
            try:
                with open(log_path, encoding="utf-8") as handle:
                    doc = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise MediaToolError("unreadable libvmaf JSON log", argv=argv) from exc
        return self._pooled_score(doc, argv)

    def _argv(self, reference: str, candidate: str, log_path: str) -> list[str]:
        # libvmaf input order: first (main) = distorted candidate, second = reference.
        graph = (
            "[0:v:0][1:v:0]libvmaf="
            f"model={_escape_filter_option(self.model)}"
            f":log_fmt=json:log_path={_escape_filter_option(log_path)}"
            ":n_threads=1:n_subsample=1:pool=mean"
        )
        return [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            candidate,
            "-i",
            reference,
            "-filter_complex",
            graph,
            "-f",
            "null",
            "-",
        ]

    def _pooled_score(self, doc: dict[str, Any], argv: Sequence[str]) -> float:
        version = doc.get("version")
        if isinstance(version, str) and version:
            self.version = version
        try:
            score = float(doc["pooled_metrics"]["vmaf"]["mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MediaToolError(
                "pooled VMAF absent from libvmaf log", argv=argv
            ) from exc
        if not 0.0 <= score <= 100.0:
            raise MediaToolError(f"pooled VMAF {score!r} outside [0, 100]", argv=argv)
        return score

    def probe_version(self) -> str:
        """Determine the libvmaf version once (tiny lavfi run) and cache it."""
        if self.version != "unknown":
            return self.version
        with tempfile.TemporaryDirectory(
            dir=self.scratch_root(), prefix=VMAF_VERSION_SCRATCH_PREFIX
        ) as tmp:
            log_path = os.path.join(tmp, "version.json")
            source = "color=c=gray:size=64x64:rate=10:duration=0.2"
            argv = [
                self._ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                source,
                "-f",
                "lavfi",
                "-i",
                source,
                "-filter_complex",
                (
                    "[0:v][1:v]libvmaf=log_fmt=json"
                    f":log_path={_escape_filter_option(log_path)}:n_threads=1"
                ),
                "-f",
                "null",
                "-",
            ]
            _run(argv, timeout=self._timeout)
            try:
                with open(log_path, encoding="utf-8") as handle:
                    self.version = str(json.load(handle).get("version") or "unknown")
            except (OSError, json.JSONDecodeError) as exc:
                raise MediaToolError("unreadable libvmaf JSON log", argv=argv) from exc
        return self.version


# --- canonicalization executor -------------------------------------------------------


class CanonicalizeExecutor:
    """Executes the scoring module's canonicalization argv plans via subprocess.

    The plan (from :func:`vidaio.scoring.canonicalize.build_canonicalization_plan`)
    is built and digested with the portable ``"ffmpeg"`` argv[0]; execution resolves
    that token to this executor's configured binary so the audit digest stays
    machine-independent. Failures raise :class:`CanonicalizationError` with the argv
    and captured stderr; timeouts raise :class:`CanonicalizationTimeout`.

    Canonicalization is also the step that EXPANDS: it decodes a compressed
    submission into raw y4m, which is where a small untrusted file becomes a huge
    one. ``max_output_bytes`` makes the caller's reservation a hard bound —
    :class:`CanonicalizationTooLarge` and a killed process group, not a full
    disk. Callers that do not pass one get the historical unbounded behaviour
    (used where the output is already known to be small).
    """

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        *,
        timeout: float = DEFAULT_SUBPROCESS_TIMEOUT,
    ) -> None:
        self._ffmpeg = ffmpeg_path
        self._timeout = timeout

    def run(
        self,
        plan: Sequence[str],
        timeout: float | None = None,
        *,
        output_path: str | None = None,
        max_output_bytes: int | None = None,
    ) -> None:
        if not plan:
            raise CanonicalizationError("empty canonicalization plan", argv=plan)
        argv = list(plan)
        if argv[0] == "ffmpeg":
            argv[0] = self._ffmpeg
        limit: tuple[str, int] | None = None
        if output_path is not None and max_output_bytes is not None:
            limit = (output_path, max_output_bytes)
        _run(
            argv,
            timeout=timeout if timeout is not None else self._timeout,
            error_cls=CanonicalizationError,
            timeout_cls=CanonicalizationTimeout,
            output_limit=limit,
            oversize_cls=CanonicalizationTooLarge,
        )


# --- PieAPP (same deterministic model on an explicitly selected CPU/CUDA device) -----


def _pieapp_package_version() -> str:
    required = ("torch", "cv2", "piq")
    if any(importlib.util.find_spec(name) is None for name in required):
        return "not-configured"
    try:
        return f"piq/{importlib.metadata.version('piq')}:pieapp"
    except importlib.metadata.PackageNotFoundError:
        return "not-configured"


def _verify_pieapp_weights(path: Path) -> None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise NotConfiguredError(
            f"cannot read PieAPP weights at {path}: {exc}"
        ) from exc
    actual = digest.hexdigest()
    if actual != PIEAPP_WEIGHTS_SHA256:
        raise NotConfiguredError(
            f"PieAPP weights digest mismatch at {path}: {actual} != "
            f"{PIEAPP_WEIGHTS_SHA256}"
        )


class _PiqPieAppRuntime:
    """Lazily imported PIQ runtime; construction may load cached model weights."""

    def __init__(self, device: Literal["cpu", "cuda"]) -> None:
        try:
            import cv2
            import piq
            import torch
        except (ImportError, OSError) as exc:
            raise NotConfiguredError(
                "PieAPP requires the optional 'media' dependencies "
                "(torch, torchvision, piq and opencv-python-headless)"
            ) from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise NotConfiguredError(
                "PieAPP was configured for CUDA but torch.cuda.is_available() is false"
            )
        # Re-verify immediately before model construction. The normal scorer
        # and auditor composition initializes this earlier so inter-op threads
        # are still configurable; this local import avoids a module cycle and
        # prevents a later process-global Torch mutation from slipping through.
        if device == "cpu":
            from vidaio.scoring_worker.runtime_identity import (
                canonical_release_marker_present,
                initialize_canonical_torch_cpu_runtime,
            )

            if canonical_release_marker_present():
                initialize_canonical_torch_cpu_runtime()
            else:
                # Native development/tests are not allowed to claim the
                # canonical release identity, but retain the historical
                # bounded CPU-thread behavior so they can exercise PieAPP.
                torch.set_num_threads(1)
        self._cv2 = cv2
        self._torch = torch
        self._device = torch.device(device)
        try:
            self._ensure_weights(torch)
            self._metric = piq.PieAPP(
                reduction="mean", data_range=1.0, stride=27, enable_grad=False
            ).to(self._device)
            self._metric.eval()
        except Exception as exc:  # missing/corrupt model cache is deployment config
            raise NotConfiguredError(
                "PieAPP model could not be initialized; pre-cache the pinned PIQ "
                f"weights in the scorer/auditor image: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _ensure_weights(torch: Any) -> None:
        checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        target = checkpoint_dir / PIEAPP_WEIGHTS_FILENAME
        if not target.exists():
            with tempfile.NamedTemporaryFile(
                prefix="pieapp-", suffix=".pth", dir=checkpoint_dir, delete=False
            ) as handle:
                temporary = Path(handle.name)
            try:
                torch.hub.download_url_to_file(
                    PIEAPP_WEIGHTS_URL, str(temporary), progress=False
                )
                _verify_pieapp_weights(temporary)
                os.replace(temporary, target)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    temporary.unlink()
        _verify_pieapp_weights(target)

    def _frames(self, path: str, start_frame: int, count: int) -> list[Any]:
        cv2 = self._cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise MediaToolError(
                "PieAPP could not open video", argv=["opencv", "VideoCapture", path]
            )
        frames: list[Any] = []
        try:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame)):
                raise MediaToolError(
                    f"PieAPP could not seek to frame {start_frame}",
                    argv=["opencv", "seek", path],
                )
            for offset in range(count):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise MediaToolError(
                        f"PieAPP could not decode frame {start_frame + offset}",
                        argv=["opencv", "decode", path],
                    )
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return frames

    def _tensor(self, frame: Any) -> Any:
        # cvtColor returns a contiguous positive-stride uint8 array. Keep the
        # conversion explicit so both devices receive identical RGB [0,1] values.
        return (
            self._torch.from_numpy(frame)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self._device, dtype=self._torch.float32)
            .div(255.0)
        )

    def compute(
        self, reference: str, candidate: str, *, start_frame: int, sample_window: int
    ) -> float:
        ref_frames = self._frames(reference, start_frame, sample_window)
        cand_frames = self._frames(candidate, start_frame, sample_window)
        scores: list[float] = []
        try:
            with self._torch.inference_mode():
                for ref, cand in zip(ref_frames, cand_frames, strict=True):
                    if ref.shape != cand.shape:
                        raise MediaToolError(
                            f"PieAPP frame shape mismatch {ref.shape!r} != {cand.shape!r}",
                            argv=["piq", "pieapp"],
                        )
                    # PIQ full-reference convention: distorted first, reference second.
                    raw = self._metric(self._tensor(cand), self._tensor(ref))
                    value = float(raw.detach().cpu().item())
                    if not math.isfinite(value):
                        raise MediaToolError(
                            f"PieAPP returned a non-finite distance {value!r}",
                            argv=["piq", "pieapp"],
                        )
                    scores.append(value)
        except (MediaToolError, NotConfiguredError):
            raise
        except Exception as exc:
            raise MediaToolError(
                "PieAPP inference failed",
                argv=["piq", "pieapp", str(self._device)],
                stderr=f"{type(exc).__name__}: {exc}",
            ) from exc
        if len(scores) != sample_window:
            raise MediaToolError(
                "PieAPP did not produce the configured sample window",
                argv=["piq", "pieapp"],
            )
        return math.fsum(scores) / len(scores)


class PieAppTorchBackend:
    """Windowed PieAPP distance with an explicit, auditable device selection.

    The runtime/model is initialized lazily on the first upscaling item and then
    shared under a lock. CPU is a first-class supported device, not a fallback;
    this is what lets validators audit every task without accepting GPU work.
    """

    name = "pieapp-torch"

    def __init__(
        self,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        sample_window: int = 4,
        _runtime_loader: Callable[[Literal["cpu", "cuda"]], Any] | None = None,
        _backend_version: str | None = None,
    ) -> None:
        if sample_window < 1:
            raise ValueError("sample_window must be >= 1")
        self.device = device
        self.sample_window = sample_window
        self.version = _backend_version or _pieapp_package_version()
        self._runtime_loader = _runtime_loader or _PiqPieAppRuntime
        self._runtime: Any | None = None
        self._lock = threading.Lock()

    def ensure_ready(self) -> None:
        """Load the model/weights without media; fail if the deployment is incomplete."""
        if self.version == "not-configured":
            raise NotConfiguredError(
                "PieAPP backend is not configured: install the optional 'media' "
                "dependency group and pre-cache the pinned PIQ PieAPP weights"
            )
        with self._lock:
            if self._runtime is None:
                self._runtime = self._runtime_loader(self.device)

    def preload(self) -> None:
        """Alias for :meth:`ensure_ready`, intended for image-build preflights."""
        self.ensure_ready()

    def compute(self, reference: str, candidate: str, *, start_frame: int) -> float:
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        self.ensure_ready()
        # PIQ models and CUDA contexts are shared process state. Serializing one
        # small deterministic window avoids concurrent model mutation/allocation races.
        with self._lock:
            assert self._runtime is not None
            return float(
                self._runtime.compute(
                    reference,
                    candidate,
                    start_frame=start_frame,
                    sample_window=self.sample_window,
                )
            )


def _opencv_package_version() -> str:
    if importlib.util.find_spec("cv2") is None:
        return "not-configured"
    for distribution in ("opencv-python-headless", "opencv-python"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


class CpuPerceptualCheckBackend:
    """Tone/grayscale/chroma manipulation checks on CPU-only integer samples.

    Canonical Y4M inputs are decoded at fixed, evenly-spaced frame positions.
    Each frame is sampled on a fixed integer pixel grid; the reductions use
    integer sums, so there is no GPU kernel or nondeterministic parallel float
    reduction for validators to reproduce.  The three gate calls share a small
    bounded statistics cache because they always inspect the same media pair.
    """

    name = "cpu-perceptual-checks"

    def __init__(
        self,
        config: CpuPerceptualConfig | None = None,
        *,
        _analyzer: Callable[[str, str, CpuPerceptualConfig], PerceptualStatistics]
        | None = None,
        _backend_version: str | None = None,
    ) -> None:
        self.config = config or CpuPerceptualConfig()
        opencv_version = _opencv_package_version()
        self.version = _backend_version or (
            "not-configured"
            if opencv_version == "not-configured"
            else (
                f"opencv/{opencv_version}:algorithm/"
                f"{CPU_PERCEPTUAL_ALGORITHM_VERSION}:config/"
                f"{self.config.digest()[:12]}"
            )
        )
        self._analyzer = _analyzer or self._analyze_opencv
        self._cache: dict[
            tuple[str, str, int, int, int, int], PerceptualStatistics
        ] = {}
        self._cache_lock = threading.Lock()

    @staticmethod
    def _file_identity(path: str) -> tuple[int, int]:
        try:
            stat = os.stat(path)
        except OSError as exc:
            raise MediaToolError(
                "CPU perceptual check cannot stat media",
                argv=["opencv", "stat", path],
                stderr=str(exc),
            ) from exc
        return stat.st_size, stat.st_mtime_ns

    def _statistics(self, reference: str, candidate: str) -> PerceptualStatistics:
        if self.version == "not-configured":
            raise NotConfiguredError(
                "CPU perceptual checks require the optional 'media' dependency "
                "group (opencv-python-headless)"
            )
        ref_size, ref_mtime = self._file_identity(reference)
        cand_size, cand_mtime = self._file_identity(candidate)
        key = (reference, candidate, ref_size, ref_mtime, cand_size, cand_mtime)
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        measured = self._analyzer(reference, candidate, self.config)
        with self._cache_lock:
            if len(self._cache) >= 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = measured
        return measured

    @staticmethod
    def _sample_positions(length: int, count: int) -> list[int]:
        count = min(length, count)
        if count <= 0:
            return []
        if count == 1:
            return [length // 2]
        return [i * (length - 1) // (count - 1) for i in range(count)]

    @classmethod
    def _analyze_opencv(
        cls, reference: str, candidate: str, config: CpuPerceptualConfig
    ) -> PerceptualStatistics:
        try:
            import cv2
            import numpy as np
        except (ImportError, OSError) as exc:
            raise NotConfiguredError(
                "CPU perceptual checks require opencv-python-headless and numpy"
            ) from exc

        ref_cap = cv2.VideoCapture(reference)
        cand_cap = cv2.VideoCapture(candidate)
        if not ref_cap.isOpened() or not cand_cap.isOpened():
            ref_cap.release()
            cand_cap.release()
            raise MediaToolError(
                "CPU perceptual check could not open canonical video",
                argv=["opencv", "VideoCapture", reference, candidate],
            )
        ref_frames = round(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cand_frames = round(cand_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if ref_frames < 1 or cand_frames != ref_frames:
            ref_cap.release()
            cand_cap.release()
            raise MediaToolError(
                f"CPU perceptual frame-count mismatch {ref_frames} != {cand_frames}",
                argv=["opencv", "frame-count", reference, candidate],
            )

        total = sum_y_ref = sum_y_cand = sum_y2_ref = sum_y2_cand = 0
        sum_chroma_ref = sum_chroma_cand = sum_chroma_diff = 0
        try:
            for frame_index in cls._sample_positions(ref_frames, config.sample_frames):
                scope = current_process_scope()
                if scope is not None:
                    scope.raise_if_cancelled(detail="CPU perceptual check cancelled")
                if not ref_cap.set(
                    cv2.CAP_PROP_POS_FRAMES, float(frame_index)
                ) or not cand_cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index)):
                    raise MediaToolError(
                        f"CPU perceptual check could not seek frame {frame_index}",
                        argv=["opencv", "seek", reference, candidate],
                    )
                ref_ok, ref = ref_cap.read()
                cand_ok, cand = cand_cap.read()
                if not ref_ok or not cand_ok or ref is None or cand is None:
                    raise MediaToolError(
                        f"CPU perceptual check could not decode frame {frame_index}",
                        argv=["opencv", "decode", reference, candidate],
                    )
                if ref.shape != cand.shape or len(ref.shape) != 3 or ref.shape[2] != 3:
                    raise MediaToolError(
                        f"CPU perceptual frame shape mismatch {ref.shape!r} != {cand.shape!r}",
                        argv=["opencv", "shape", reference, candidate],
                    )
                rows = np.asarray(
                    cls._sample_positions(ref.shape[0], config.sample_edge),
                    dtype=np.intp,
                )
                cols = np.asarray(
                    cls._sample_positions(ref.shape[1], config.sample_edge),
                    dtype=np.intp,
                )
                ref = ref[rows[:, None], cols[None, :]].astype(np.int32)
                cand = cand[rows[:, None], cols[None, :]].astype(np.int32)
                ref_b, ref_g, ref_r = (ref[:, :, i] for i in range(3))
                cand_b, cand_g, cand_r = (cand[:, :, i] for i in range(3))
                y_ref = (29 * ref_b + 150 * ref_g + 77 * ref_r + 128) >> 8
                y_cand = (29 * cand_b + 150 * cand_g + 77 * cand_r + 128) >> 8
                rg_ref, bg_ref = ref_r - ref_g, ref_b - ref_g
                rg_cand, bg_cand = cand_r - cand_g, cand_b - cand_g
                pixels = int(y_ref.size)
                total += pixels
                sum_y_ref += int(y_ref.sum(dtype=np.int64))
                sum_y_cand += int(y_cand.sum(dtype=np.int64))
                sum_y2_ref += int((y_ref * y_ref).sum(dtype=np.int64))
                sum_y2_cand += int((y_cand * y_cand).sum(dtype=np.int64))
                sum_chroma_ref += int(
                    (np.abs(rg_ref) + np.abs(bg_ref)).sum(dtype=np.int64)
                )
                sum_chroma_cand += int(
                    (np.abs(rg_cand) + np.abs(bg_cand)).sum(dtype=np.int64)
                )
                sum_chroma_diff += int(
                    (np.abs(rg_ref - rg_cand) + np.abs(bg_ref - bg_cand)).sum(
                        dtype=np.int64
                    )
                )
        finally:
            ref_cap.release()
            cand_cap.release()
        if total < 1:
            raise MediaToolError(
                "CPU perceptual check sampled no pixels",
                argv=["opencv", "sample", reference, candidate],
            )

        ref_mean_raw = sum_y_ref / total
        cand_mean_raw = sum_y_cand / total
        ref_variance = max(0.0, sum_y2_ref / total - ref_mean_raw**2)
        cand_variance = max(0.0, sum_y2_cand / total - cand_mean_raw**2)
        return PerceptualStatistics(
            sampled_pixels=total,
            reference_luma_mean=ref_mean_raw / 255.0,
            candidate_luma_mean=cand_mean_raw / 255.0,
            reference_luma_std=math.sqrt(ref_variance) / 255.0,
            candidate_luma_std=math.sqrt(cand_variance) / 255.0,
            reference_chroma_energy=sum_chroma_ref / (total * 510.0),
            candidate_chroma_energy=sum_chroma_cand / (total * 510.0),
            chroma_mae=sum_chroma_diff / (total * 1020.0),
        )

    def check_tone_manipulation(
        self, reference: str, candidate: str
    ) -> PerceptualCheckResult:
        return tone_manipulation_result(
            self._statistics(reference, candidate), self.config
        )

    def check_color_grayscale(
        self, reference: str, candidate: str
    ) -> PerceptualCheckResult:
        return grayscale_result(self._statistics(reference, candidate), self.config)

    def check_chroma_uv(self, reference: str, candidate: str) -> PerceptualCheckResult:
        return chroma_uv_result(self._statistics(reference, candidate), self.config)


class UnconfiguredPerceptualCheckBackend:
    """Explicit refusal seam for deliberately incomplete test compositions.

    Release scoring and auditing use :class:`CpuPerceptualCheckBackend`; this
    class remains only for tests that prove a missing required backend fails
    loudly instead of silently substituting a passing verdict.
    """

    name = "perceptual-checks"
    version = "not-configured"

    def _refuse(self, check: str) -> PerceptualCheckResult:
        raise NotConfiguredError(
            f"perceptual {check} check is not configured in this composition; "
            "refusing to substitute a pass/fail verdict."
        )

    def check_tone_manipulation(
        self, reference: str, candidate: str
    ) -> PerceptualCheckResult:
        return self._refuse("tone-manipulation")

    def check_color_grayscale(
        self, reference: str, candidate: str
    ) -> PerceptualCheckResult:
        return self._refuse("color-grayscale")

    def check_chroma_uv(self, reference: str, candidate: str) -> PerceptualCheckResult:
        return self._refuse("chroma-uv")


# --- tool version detection ----------------------------------------------------------


_VERSION_RE = re.compile(r"^ff\w+ version (\S+)")


def _binary_version(binary: str, *, timeout: float) -> str:
    completed = _run([binary, "-version"], timeout=timeout)
    match = _VERSION_RE.match(
        completed.stdout.splitlines()[0] if completed.stdout else ""
    )
    return match.group(1) if match else "unknown"


def detect_tool_versions(
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
    *,
    vmaf_backend: FfmpegVmafBackend | None = None,
    timeout: float = DEFAULT_SUBPROCESS_TIMEOUT,
) -> dict[str, str]:
    """ffmpeg/ffprobe/libvmaf versions for the audit ``backend_versions`` stamp.

    Probed once at worker startup; libvmaf's version comes from the given backend
    (cached from its JSON log) so the stamp matches the library that actually scored.
    """
    versions = {
        "ffmpeg": f"ffmpeg/{_binary_version(ffmpeg_path, timeout=timeout)}",
        "ffprobe": f"ffprobe/{_binary_version(ffprobe_path, timeout=timeout)}",
    }
    if vmaf_backend is not None:
        versions["libvmaf"] = f"libvmaf/{vmaf_backend.probe_version()}"
    return versions
