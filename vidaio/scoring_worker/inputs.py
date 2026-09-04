"""Verify-then-snapshot: the only artifacts the scoring pipeline is allowed to read.

A ``/score`` request names files on the shared filesystem that the MINER (or
whoever produced them) may still be able to write. Hashing a path and then
re-opening that same path for canonicalization/probing is a time-of-check /
time-of-use hole: swap the bytes in between and the packet's ``content_digest``
names one file while ffmpeg measured another. Since the digest is the only thing
binding a score to a submission, that turns the whole audit trail into fiction.

The fix implemented here is verify-then-snapshot, in this exact order:

  1. ``os.open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK)`` — one handle, taken
     once. ``O_NOFOLLOW`` refuses a symlinked final component (a miner cannot
     point us at the held-out reference or an operator file); ``O_NONBLOCK``
     means a FIFO opens instead of hanging forever waiting for a writer.
  2. ``os.fstat`` on THAT descriptor: anything that is not a regular file (fifo,
     device such as ``/dev/zero``, directory, socket) is a 422, not an
     unbounded read.
  3. hash AND copy in a single pass from that same descriptor into the worker's
     private per-request directory, mode ``0400``. Because both the bytes we
     hashed and the bytes we keep come from one open file description, no path
     swap can separate them: replacing/renaming the path afterwards affects a
     file we no longer have any reference to.
  4. compare the streamed digest against the request's claim (mismatch → 422,
     copy discarded), then re-hash the private copy as cheap insurance against a
     truncated/corrupt write (mismatch → 500).

Everything downstream — canonicalization, probing, VMAF/PieAPP, the gate facts —
is then handed ONLY the private paths, and the packet stamps the digest of the
copy that was actually measured. The copies live inside the request's temporary
directory and die with it; :func:`sweep_work_dir` clears anything a crash left
behind at worker startup so the work dir stays bounded.

BYTE BUDGETS (:class:`ByteLimits` / :class:`ScratchBudget`). Snapshotting is a
COPY, so verify-then-snapshot amplifies whatever a caller names into the worker's
own volume. Without a ceiling a miner that returns one enormous regular file
makes the validator hash it and the worker duplicate it — filling the scoring
volume long before ``request_timeout`` fires, which takes down every concurrent
request and the next ones too. The budget therefore covers EVERY byte a request
puts on that volume — not just the copies:

  * per file (``scoring_worker.max_input_bytes``): ``fstat`` on the same
    descriptor we will read decides it, so an oversize file is a typed 422 that
    never writes a byte. The copy loop carries the same number as a hard stop, so
    a file that GROWS after the fstat is aborted mid-stream (422) instead of
    running past its reservation.
  * per request, INPUTS (``scoring_worker.max_request_bytes``): the three inputs
    of one request together. 413 with the running total.
  * per request, ALL SCRATCH (``scoring_worker.max_request_scratch_bytes``): the
    snapshots PLUS everything the request generates from them — the canonicalized
    raw y4m files and the libvmaf JSON logs. 413, because a request that cannot
    fit inside one request's ceiling can never fit: shedding it would shed it
    forever. Capped at ``max_scratch_bytes`` (see
    :attr:`ByteLimits.request_scratch_ceiling`), so "bigger than the whole
    worker's volume" is always the deterministic refusal and never an eternal 503.
  * per worker (``scoring_worker.max_scratch_bytes``): every live request's
    snapshots AND generated files summed, so N concurrent requests cannot
    collectively fill the volume that one alone could not. 503 + Retry-After —
    "come back", exactly like queue saturation, because the budget frees when a
    request ends.

THE EXPANSION PROBLEM (why input caps alone are not a scratch bound). Every input
cap measures the ENCODED file. Scoring measures the DECODED one: both sides are
canonicalized to raw y4m before any metric runs, and raw video is three to four
orders of magnitude larger than its h264/h265 encoding. A 30 MB, 10-minute 4K
clip passes every input cap and expands to ~450 GB of y4m. So the projected raw
size is computed from the PROBE, before ffmpeg is started, and reserved against
the same budget (:func:`projected_canonical_bytes`); over the ceiling is refused
before a byte is written. The estimate is also enforced while it expands —
:class:`~vidaio.scoring.backends_real.CanonicalizeExecutor` takes the reserved
cap as a hard bound and kills the process group that exceeds it — because a
projection is a prediction and a prediction can be wrong.

Every reservation is held for the LIFE of the request's scratch directory (the
lease is released after the directory is removed, never when copying ends), and
an aborted copy discards its partial file and refunds its reservation.

RESIDUALS. The startup sweep deliberately never fails startup,
so a crash-left directory the worker CANNOT delete (permissions, a busy mount)
used to vanish from the accounting entirely: the fresh budget started at zero
while the leftover bytes sat on the volume, and a fully-admitted budget on top of
them was an overcommit ending in ENOSPC. Whatever the sweep leaves behind is
therefore MEASURED (:func:`measure_scratch_entries`) and PRE-CHARGED against the
worker budget (:meth:`ScratchBudget.charge_residual`) as a reservation that
admission cannot displace — the budget then admits only what genuinely fits
beside the leftovers. The charge is permanent until a later sweep succeeds:
every retry (:meth:`ScratchBudget.retry_residual_sweep`, driven per scored
request) deletes what it now can and releases exactly those bytes. Residuals
larger than the whole configured budget are an operator problem and fail the
worker fatally at startup instead of silently overcommitting the volume.
"""

from __future__ import annotations

import errno
import hashlib
import math
import os
import re
import shutil
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Mapping

from vidaio.scoring.backends import MediaInfo
from vidaio.scoring.backends_real import (
    VMAF_SCRATCH_PREFIX,
    VMAF_VERSION_SCRATCH_PREFIX,
)
from vidaio.scoring.canonicalize import CANONICAL_PIX_FMT

#: Streaming chunk for the hash-and-copy pass.
HASH_CHUNK = 1 << 20

#: Largest single input the worker will hash-and-copy (2 GiB). Deliberately the
#: same ceiling the reference miner puts on its own ingress
#: (``miner.max_input_bytes``): a miner cannot legitimately produce an output
#: from an input it would itself refuse, and a challenge clip is seconds of
#: video — orders of magnitude below this. Generous enough that no honest
#: submission is ever refused, small enough that one caller cannot hand the
#: worker an arbitrary amount of disk.
DEFAULT_MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024

#: Largest total across ONE request's three inputs (4 GiB). Not 3x the per-file
#: cap on purpose: a genuine item is a lossless reference plus the (smaller)
#: derived miner input plus an ENCODED output, so a maximum-size reference still
#: leaves 2 GiB for the other two combined, while three maximum-size files —
#: which no honest challenge produces — are refused.
DEFAULT_MAX_REQUEST_BYTES = 4 * 1024 * 1024 * 1024

#: ALL scratch bytes one request may hold live at once: its snapshots plus the
#: raw y4m files and metric logs it generates from them (16 GiB). With the
#: shipped ``max_concurrent=2`` two of these exactly fill
#: ``DEFAULT_MAX_SCRATCH_BYTES``, so a fully loaded worker never sheds its own
#: legitimate load. 4 GiB of it is the input allowance
#: (``DEFAULT_MAX_REQUEST_BYTES``); the remaining 12 GiB is the expansion
#: allowance — enough for two ~90-second 1080p y4m sides (1920x1080 yuv420p is
#: ~3.1 MB/frame, so 12 GiB is ~3900 frames per side), and far short of the
#: hundreds of gigabytes a long 4K clip would decode to.
DEFAULT_MAX_REQUEST_SCRATCH_BYTES = 16 * 1024 * 1024 * 1024

#: Scratch bytes the whole worker may hold live at once (32 GiB), across every
#: live request and covering snapshots AND generated files alike. Must be at
#: least ``max_concurrent x max_request_scratch_bytes`` or a fully loaded worker
#: would shed its own legitimate load.
DEFAULT_MAX_SCRATCH_BYTES = 32 * 1024 * 1024 * 1024

#: Private copies are read-only: nothing downstream (or beside us) rewrites the
#: bytes that were verified.
SNAPSHOT_MODE = 0o400

#: Per-request scratch directories live directly under ``scoring_worker.work_dir``.
WORK_PREFIX = "score-"

#: Health probes write (and immediately unlink) one of these in the work dir; a
#: crash between the two leaves it behind, so the sweep reclaims it too.
HEALTH_PROBE_PREFIX = ".healthz-probe-"

#: EVERY shape this worker can leave in ``work_dir``. A sweep that only knew the
#: per-request prefix left crash-orphaned libvmaf temp dirs on the volume for
#: the life of the deployment — an unbounded work dir is how the scorer
#: eventually fills its own disk.
SCRATCH_PREFIXES = (
    WORK_PREFIX,
    VMAF_SCRATCH_PREFIX,
    VMAF_VERSION_SCRATCH_PREFIX,
    HEALTH_PROBE_PREFIX,
)


# --- projecting the scratch a request will GENERATE ------------------------------------
#
# The canonicalized side of a comparison is raw y4m, and its size is not a guess:
# the y4m container is a header line, then one "FRAME\n" marker plus one
# uncompressed frame per frame. So
#
#     bytes = header + frames x (6 + width x height x bytes-per-pixel(pix_fmt))
#
# with bytes-per-pixel taken from the pixel format's plane geometry (yuv420p =
# one luma sample per pixel plus two quarter-resolution chroma planes = 1.5
# B/px). Verified exact against real ffmpeg output for several clips
# (tests/scoring_worker/test_inputs.py) — the only inexactness is the header
# line, which is bounded by Y4M_HEADER_BYTES.
#
# The pixel format is the CANONICAL one, not the source's: canonicalization
# pins ``format=yuv420p`` on both sides (vidaio.scoring.canonicalize), which is
# exactly what makes the projection knowable in advance.

#: Upper bound on the y4m header line. Measured 58 B for a real ffmpeg 320x240
#: clip ("YUV4MPEG2 W320 H240 F25:1 Ip A1:1 C420jpeg XYSCSS=420JPEG\n"); 128
#: covers the widest plausible field values (6-digit dimensions, a long frame-rate
#: fraction, an explicit chroma-siting tag).
Y4M_HEADER_BYTES = 128

#: The literal ``b"FRAME\n"`` that precedes every frame's planes.
Y4M_FRAME_HEADER_BYTES = 6

#: Frames added to every projection. Absorbs the one-frame rounding that
#: ``-fps_mode cfr`` can introduce when the container's declared frame rate and
#: its declared duration disagree in the last fractional frame. Small on purpose:
#: a WRONG projection is caught by the hard cap during expansion, not by padding.
CANONICAL_SLACK_FRAMES = 2

#: Plane geometry of the pixel formats canonicalization is allowed to emit:
#: ``(chroma x-subsampling, chroma y-subsampling, bytes per sample)``, with 0
#: subsampling meaning "no chroma planes" (monochrome). Only planar YUV and gray
#: exist in y4m, which is why this table is closed rather than a heuristic.
_PIX_FMT_PLANES: dict[str, tuple[int, int, int]] = {
    "gray": (0, 0, 1),
    "gray10le": (0, 0, 2),
    "gray12le": (0, 0, 2),
    "yuv420p": (2, 2, 1),
    "yuv422p": (2, 1, 1),
    "yuv444p": (1, 1, 1),
    "yuv420p10le": (2, 2, 2),
    "yuv422p10le": (2, 1, 2),
    "yuv444p10le": (1, 1, 2),
    "yuv420p12le": (2, 2, 2),
    "yuv422p12le": (2, 1, 2),
    "yuv444p12le": (1, 1, 2),
}

#: Bytes to reserve per frame per libvmaf run for the JSON log. Measured at ~718
#: B/frame for the default model (16 integer features plus vmaf, full float
#: precision); 1024 rounds that up with room for a wider feature set.
VMAF_LOG_BYTES_PER_FRAME = 1024

#: Floor on the log reservation, so a two-frame clip still reserves something for
#: the JSON envelope, the pooled-metrics block and the version string.
VMAF_LOG_FLOOR_BYTES = 64 * 1024

#: Suffix carried over from the source name so ffmpeg/ffprobe keep whatever
#: container hint they had. Anything exotic is dropped rather than reproduced.
_SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,8}$")


class ScoreRejected(Exception):
    """A /score request refused before scoring — carries the HTTP status + payload."""

    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        #: Response headers the refusal needs (Retry-After on a 503 budget shed).
        self.headers = headers
        super().__init__(payload.get("error", "rejected"))


class SnapshotCancelled(Exception):
    """The caller withdrew while inputs were still being copied.

    Snapshotting a multi-gigabyte submission is real, uninterruptible I/O; the
    request's deadline has to reach it too, or a timed-out request keeps its
    concurrency slot (and the disk) busy long after the 504.
    """


def y4m_frame_bytes(width: int, height: int, pix_fmt: str = CANONICAL_PIX_FMT) -> int:
    """Uncompressed bytes of ONE y4m frame (planes only, no ``FRAME\\n`` marker).

    Luma is one sample per pixel; each of the two chroma planes is the luma plane
    divided by the format's subsampling, rounded UP (a decoder never allocates a
    partial chroma row), times the format's bytes per sample.
    """
    geometry = _PIX_FMT_PLANES.get(pix_fmt)
    if geometry is None:
        raise ScoreRejected(
            422,
            {
                "error": "unsupported_canonical_pix_fmt",
                "pix_fmt": pix_fmt,
                "supported": sorted(_PIX_FMT_PLANES),
            },
        )
    sub_x, sub_y, bytes_per_sample = geometry
    luma = width * height * bytes_per_sample
    if sub_x == 0 or sub_y == 0:  # monochrome: no chroma planes at all
        return luma
    chroma = math.ceil(width / sub_x) * math.ceil(height / sub_y) * bytes_per_sample
    return luma + 2 * chroma


def projected_frame_count(info: MediaInfo) -> int:
    """Frames the CANONICALIZED copy of `info` will contain.

    ``max`` of the two independent statements the container makes about itself,
    because canonicalization runs ``-fps_mode cfr``: a variable-frame-rate source
    whose declared duration and frame rate imply more frames than it stores gets
    those frames DUPLICATED into the output. Trusting ``nb_frames`` alone would
    under-project exactly the file that expands the most.
    """
    counted = max(0, int(info.frame_count or 0))
    implied = 0
    if info.duration > 0 and info.fps > 0:
        implied = math.ceil(info.duration * info.fps)
    return max(counted, implied)


def projected_canonical_bytes(
    field: str,
    info: MediaInfo,
    *,
    pix_fmt: str = CANONICAL_PIX_FMT,
    slack_frames: int = CANONICAL_SLACK_FRAMES,
) -> int:
    """Bytes the canonicalized y4m of `info` will occupy — computed, not guessed.

    ``header + (frames + slack) x (6 + frame planes)``. Exact for a well-formed
    source (see the module's projection notes); the slack covers CFR rounding and
    the header bound covers the one variable-length field.

    A stream whose geometry cannot be projected (zero dimensions, or a container
    that reports neither a frame count nor a usable duration x rate) is a typed
    422: the worker refuses to start an expansion whose size it cannot bound.
    """
    frames = projected_frame_count(info)
    if info.width <= 0 or info.height <= 0 or frames <= 0:
        raise ScoreRejected(
            422,
            {
                "error": "unprojectable_stream",
                "field": field,
                "width": info.width,
                "height": info.height,
                "frame_count": info.frame_count,
                "duration": info.duration,
                "fps": info.fps,
                "detail": (
                    "canonical size cannot be bounded from the probed stream, so "
                    "the expansion cannot be reserved"
                ),
            },
        )
    per_frame = Y4M_FRAME_HEADER_BYTES + y4m_frame_bytes(info.width, info.height, pix_fmt)
    return Y4M_HEADER_BYTES + (frames + max(0, slack_frames)) * per_frame


def projected_metric_log_bytes(*, frames: int, runs: int) -> int:
    """Scratch to reserve for the libvmaf JSON logs of one request.

    libvmaf writes one JSON record per frame per run into the request's own
    scratch directory. Modest by design — the logs are text next to raw video —
    but accounted, because "modest" is not "free".

    `runs` SCALES the whole per-run amount (floor included): the reservation is
    exactly ``runs x projected_metric_log_bytes(frames=frames, runs=1)``, and
    that per-run amount is the hard bound each individual libvmaf run is held to
    (an internal review, enforced via
    :func:`vidaio.scoring.backends_real.use_metric_log_limit`). Per-run
    enforcement of a per-run share is what keeps the enforced total within the
    reservation with no run borrowing another's headroom.
    """
    per_run = max(VMAF_LOG_FLOOR_BYTES, max(0, frames) * VMAF_LOG_BYTES_PER_FRAME)
    return max(1, runs) * per_run


@dataclass(frozen=True)
class ByteLimits:
    """The scratch ceilings (module docstring). Defaults = shipped config."""

    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_request_scratch_bytes: int = DEFAULT_MAX_REQUEST_SCRATCH_BYTES
    max_scratch_bytes: int = DEFAULT_MAX_SCRATCH_BYTES

    @property
    def request_scratch_ceiling(self) -> int:
        """The largest total scratch ONE request may ever hold.

        Never above the worker-wide budget: a request bigger than the whole
        volume can never run, so it must be refused deterministically (413)
        instead of shed (503) on every retry until the end of time.
        """
        return min(self.max_request_scratch_bytes, self.max_scratch_bytes)


class ScratchBudget:
    """Worker-wide accounting of every scratch byte currently live in the work dir.

    One instance per app (created in :func:`vidaio.scoring_worker.service.create_app`)
    and shared by every request, so the guard is genuinely collective: two
    requests that individually fit still cannot exceed the volume together.
    Counters move under a lock because the accounting happens on the executor
    threads, not on the event loop.

    Accounted: the verified private copies AND everything a request generates
    from them (canonicalized y4m, metric logs). Reservations are taken BEFORE
    the bytes are written — from the file's ``fstat`` size for a copy, from the
    probed stream geometry for an expansion (a budget checked afterwards is not
    a budget) — and the surplus is refunded once the real byte count is known.
    """

    def __init__(self, limits: ByteLimits | None = None) -> None:
        self.limits = limits if limits is not None else ByteLimits()
        self._lock = threading.Lock()
        self._used = 0
        #: path -> bytes of startup leftovers a sweep could not delete. Charged
        #: into ``_used`` so admission genuinely shrinks.
        self._residual: dict[str, int] = {}

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used

    @property
    def residual_bytes(self) -> int:
        """Bytes currently held by undeletable startup leftovers (see module doc)."""
        with self._lock:
            return sum(self._residual.values())

    def charge_residual(self, entries: Mapping[str, int]) -> int:
        """Permanently reserve bytes a failed sweep left on the volume.

        `entries` maps each leftover path to its measured size
        (:func:`measure_scratch_entries`). Deliberately NEVER refused, even past
        the configured ceiling: the bytes are already on disk, so the charge
        records reality — what it changes is admission, which now only accepts
        requests that fit BESIDE the leftovers. The caller decides whether an
        over-budget residual is fatal (the scoring worker: yes, at startup).
        Returns the total charged; paths already charged are not charged twice.
        """
        with self._lock:
            charged = 0
            for path, nbytes in entries.items():
                if path in self._residual or nbytes < 0:
                    continue
                self._residual[path] = nbytes
                self._used += nbytes
                charged += nbytes
            return charged

    def retry_residual_sweep(self) -> int:
        """Try again to delete every charged leftover; release what is now gone.

        Runs on a worker thread (real disk I/O). Only the recorded paths are
        touched — live requests' directories are never in the recorded set, so
        a retry can never delete or double-count a concurrent request's scratch.
        A path that still cannot be deleted keeps its charge; a path that IS
        gone (deleted here, or by an operator fixing the permissions) releases
        exactly the bytes it was charged. Returns the bytes released.
        """
        with self._lock:
            entries = dict(self._residual)
        released = 0
        for path, nbytes in entries.items():
            target = Path(path)
            try:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target, onexc=_force_remove)
                else:
                    target.unlink()
            except OSError:
                pass  # still undeletable: the charge stays
            if _path_gone(target):
                with self._lock:
                    if self._residual.pop(path, None) is not None:
                        self._used = max(0, self._used - nbytes)
                        released += nbytes
        return released

    def lease(self) -> "ScratchLease":
        """A per-request slice of this budget (release it when its dir is gone)."""
        return ScratchLease(self)

    # -- used by ScratchLease only -----------------------------------------
    def _take(self, nbytes: int) -> int | None:
        """Reserve `nbytes`; returns the live total on refusal, None on success."""
        with self._lock:
            if self._used + nbytes > self.limits.max_scratch_bytes:
                return self._used
            self._used += nbytes
            return None

    def _give_back(self, nbytes: int) -> None:
        with self._lock:
            self._used = max(0, self._used - nbytes)


class ScratchLease:
    """One request's reservation against a :class:`ScratchBudget`.

    Held for the LIFETIME of the request's scratch directory — released after the
    directory is removed, not when snapshotting finishes — because the bytes are
    on the volume for exactly that long.

    Two counters, because the two ceilings mean different things.
    ``input_bytes`` is what the caller NAMED (checked against
    ``max_request_bytes``); ``held_bytes`` is everything the request will put on
    the volume, inputs and expansions alike (checked against
    ``request_scratch_ceiling`` and against the worker-wide budget). Keeping them
    apart stops a reserved expansion from retroactively making a legal input
    look oversize.
    """

    def __init__(self, budget: ScratchBudget) -> None:
        self._budget = budget
        self._held = 0
        self._input_bytes = 0

    @property
    def limits(self) -> ByteLimits:
        return self._budget.limits

    @property
    def held_bytes(self) -> int:
        """Every scratch byte this request has reserved (inputs + generated)."""
        return self._held

    @property
    def input_bytes(self) -> int:
        """The snapshot share of :attr:`held_bytes`."""
        return self._input_bytes

    def reserve(self, *, field: str, path_text: str, nbytes: int) -> None:
        """Claim `nbytes` for one input, or refuse with the typed error that fits."""
        limits = self._budget.limits
        if nbytes > limits.max_input_bytes:
            raise ScoreRejected(
                422,
                {
                    "error": "input_too_large",
                    "field": field,
                    "path": path_text,
                    "input_bytes": nbytes,
                    "limit": limits.max_input_bytes,
                },
            )
        if self._input_bytes + nbytes > limits.max_request_bytes:
            raise ScoreRejected(
                413,
                {
                    "error": "request_inputs_too_large",
                    "field": field,
                    "path": path_text,
                    "input_bytes": nbytes,
                    "request_bytes": self._input_bytes + nbytes,
                    "limit": limits.max_request_bytes,
                },
            )
        self._claim(
            nbytes,
            too_large=lambda ceiling: ScoreRejected(
                413,
                {
                    "error": "request_scratch_too_large",
                    "field": field,
                    "path": path_text,
                    "input_bytes": nbytes,
                    "request_scratch_bytes": self._held + nbytes,
                    "limit": ceiling,
                },
            ),
            unavailable=lambda live: ScoreRejected(
                503,
                {
                    "error": "scratch_budget_unavailable",
                    "field": field,
                    "input_bytes": nbytes,
                    "scratch_used_bytes": live,
                    "limit": limits.max_scratch_bytes,
                },
                headers={"Retry-After": "5"},
            ),
        )
        self._input_bytes += nbytes

    def reserve_generated(self, *, kind: str, nbytes: int, **detail: Any) -> None:
        """Claim `nbytes` for scratch this request is about to CREATE.

        Same two outcomes as an input reservation and for the same reasons: 413
        when the request could never fit inside one request's ceiling (shedding
        it would shed it on every retry, forever), 503 + Retry-After when it
        merely does not fit right now beside the other live requests.
        """
        limits = self._budget.limits
        self._claim(
            nbytes,
            too_large=lambda ceiling: ScoreRejected(
                413,
                {
                    "error": "request_scratch_too_large",
                    "kind": kind,
                    "projected_bytes": nbytes,
                    "request_scratch_bytes": self._held + nbytes,
                    "limit": ceiling,
                    **detail,
                },
            ),
            unavailable=lambda live: ScoreRejected(
                503,
                {
                    "error": "scratch_budget_unavailable",
                    "kind": kind,
                    "projected_bytes": nbytes,
                    "scratch_used_bytes": live,
                    "limit": limits.max_scratch_bytes,
                    **detail,
                },
                headers={"Retry-After": "5"},
            ),
        )

    def _claim(
        self,
        nbytes: int,
        *,
        too_large: Callable[[int], ScoreRejected],
        unavailable: Callable[[int], ScoreRejected],
    ) -> None:
        """Per-request ceiling first (permanent), then the worker budget (transient)."""
        ceiling = self._budget.limits.request_scratch_ceiling
        if self._held + nbytes > ceiling:
            raise too_large(ceiling)
        live = self._budget._take(nbytes)
        if live is not None:
            raise unavailable(live)
        self._held += nbytes

    def refund(self, nbytes: int) -> None:
        """Give back input bytes no longer on disk (short copy, discarded copy)."""
        given = self._give_back(nbytes)
        self._input_bytes = max(0, self._input_bytes - given)

    def refund_generated(self, nbytes: int) -> None:
        """Give back reserved-but-unused expansion bytes (the projection's surplus)."""
        self._give_back(nbytes)

    def _give_back(self, nbytes: int) -> int:
        if nbytes <= 0:
            return 0
        nbytes = min(nbytes, self._held)
        self._budget._give_back(nbytes)
        self._held -= nbytes
        return nbytes

    def release(self) -> None:
        self._give_back(self._held)
        self._input_bytes = 0

    def __enter__(self) -> "ScratchLease":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


@dataclass(frozen=True)
class VerifiedInput:
    """One request input, verified and materialized as an immutable private copy."""

    #: Protocol field this input came from ("reference" / "miner_input" / "output").
    field: str
    #: The mutable path the request named — recorded for diagnostics ONLY; the
    #: pipeline must never read it again.
    source_path: str
    #: The private read-only copy every downstream stage reads.
    path: str
    #: sha256 of the bytes in `path` — equal to the request's claim by construction.
    digest: str


@dataclass(frozen=True)
class InputSnapshot:
    """The three verified inputs of one /score request."""

    reference: VerifiedInput
    miner_input: VerifiedInput
    output: VerifiedInput


def snapshot_request_inputs(
    *,
    reference: tuple[str, str],
    miner_input: tuple[str, str],
    output: tuple[str, str],
    dest_dir: Path,
    cancelled: Callable[[], bool] | None = None,
    lease: ScratchLease | None = None,
) -> InputSnapshot:
    """Verify + snapshot the three inputs of a request into `dest_dir`.

    Each argument is a ``(path, expected_sha256)`` pair. Raises
    :class:`ScoreRejected` on the first input that is missing, not a regular
    file, a symlink, over one of the byte budgets, or whose bytes do not hash to
    the claimed digest, and :class:`SnapshotCancelled` if `cancelled()` turns
    true mid-copy.

    `lease` carries the request's byte budget and must outlive `dest_dir` (the
    caller releases it after the directory is gone). Omitting it gives this call
    a private lease with the DEFAULT limits — bounded, never unlimited.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    lease = lease if lease is not None else ScratchBudget().lease()
    return InputSnapshot(
        reference=snapshot_input(
            "reference", *reference, dest_dir=dest_dir, cancelled=cancelled, lease=lease
        ),
        miner_input=snapshot_input(
            "miner_input",
            *miner_input,
            dest_dir=dest_dir,
            cancelled=cancelled,
            lease=lease,
        ),
        output=snapshot_input(
            "output", *output, dest_dir=dest_dir, cancelled=cancelled, lease=lease
        ),
    )


def snapshot_input(
    field: str,
    path_text: str,
    expected_digest: str,
    *,
    dest_dir: Path,
    cancelled: Callable[[], bool] | None = None,
    lease: ScratchLease | None = None,
) -> VerifiedInput:
    """Verify one input and materialize it as an immutable private copy.

    The byte budgets are enforced on the way IN: the size comes from ``fstat`` on
    the descriptor that will be read (so the check cannot be raced by a rename),
    the reservation is taken before the first byte is written, and the copy loop
    stops the instant it would exceed that reservation.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    lease = lease if lease is not None else ScratchBudget().lease()
    dest = dest_dir / f"{field}{_carried_suffix(path_text)}"
    fd, size = _open_regular_file(field, path_text)
    try:
        lease.reserve(field=field, path_text=path_text, nbytes=size)
    except BaseException:
        os.close(fd)  # refused before a single byte was copied
        raise

    try:
        actual, written = _hash_and_copy(fd, dest, cancelled, limit=size)
    except _CopyLimitExceeded:
        # The source GREW after its fstat: stop at the reserved ceiling rather
        # than let a writer we do not control keep extending our copy.
        _discard(dest)
        lease.refund(size)
        raise ScoreRejected(
            422,
            {
                "error": "input_grew_during_snapshot",
                "field": field,
                "path": path_text,
                "limit": size,
            },
        ) from None
    except FileExistsError:  # O_EXCL lost: never delete a file we did not create
        lease.refund(size)
        raise
    except BaseException:  # cancellation, I/O error, disk full
        _discard(dest)
        lease.refund(size)
        raise
    finally:
        os.close(fd)
    lease.refund(size - written)  # the copy may be shorter than the fstat size

    try:
        if actual != expected_digest:
            raise ScoreRejected(
                422,
                {
                    "error": "digest_mismatch",
                    "field": field,
                    "path": path_text,
                    "expected": expected_digest,
                    "actual": actual,
                },
            )
        # Cheap insurance: prove the bytes we will actually measure are still the
        # bytes we hashed (a short write / full disk must not score as a match).
        copy_digest = digest_of(dest)
        if copy_digest != expected_digest:
            raise ScoreRejected(
                500,
                {
                    "error": "snapshot_corrupt",
                    "field": field,
                    "path": path_text,
                    "expected": expected_digest,
                    "actual": copy_digest,
                },
            )
    except BaseException:
        _discard(dest)
        lease.refund(written)
        raise
    return VerifiedInput(
        field=field, source_path=path_text, path=str(dest), digest=copy_digest
    )


def digest_of(path: Path | str) -> str:
    """sha256 of a file, read through a non-following descriptor."""
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        while chunk := os.read(fd, HASH_CHUNK):
            digest.update(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


def sweep_work_dir(work_dir: Path) -> int:
    """Delete scratch a previous run left behind. Returns the count.

    Called at worker startup. A crash mid-request leaves private copies, raw y4m
    intermediates, libvmaf temp dirs and health probes on disk; an unbounded work
    dir is how the scorer eventually fills the volume. So the sweep reclaims
    EVERY prefix this worker can create (:data:`SCRATCH_PREFIXES`) — per-request
    dirs (``score-``, which nest the y4m files and the libvmaf dirs of live
    requests), plus the ``vmaf-``/``vmafver-`` dirs libvmaf creates directly
    under the work dir outside a request, plus stale health probes.

    Anything else in the work dir is somebody ELSE's file and is left alone: the
    sweep is a reclamation of our own leftovers, not a directory wipe.
    """
    if not work_dir.is_dir():
        return 0
    removed = 0
    for entry in work_dir.iterdir():
        if not entry.name.startswith(SCRATCH_PREFIXES):
            continue
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, onexc=_force_remove)
            else:
                entry.unlink()
        except OSError:  # a dir another worker holds — leave it, never fail startup
            continue
        removed += 1
    return removed


def measure_scratch_entries(work_dir: Path) -> tuple[dict[str, int], list[str]]:
    """Bytes still sitting under OUR prefixes in `work_dir`, per entry.

    Called at startup, AFTER :func:`sweep_work_dir` and BEFORE any request runs:
    at that instant everything matching :data:`SCRATCH_PREFIXES` is a leftover
    the sweep failed to delete, and its size is exactly the scratch the fresh
    budget must not hand out twice. Returned per entry so
    :meth:`ScratchBudget.retry_residual_sweep` can later release precisely the
    entries that become deletable, and never touch a live request's directory.

    Sizes are lstat sums (symlinks never followed). Returns
    ``(measurable, unmeasurable)``: entries whose subtree was FULLY traversed
    (safe to pre-charge by their byte count) and the paths of entries that hid
    part of their contents (permissions/IO errors). An unmeasurable entry's
    true size is unbounded relative to its visible sum — a 0700 directory
    owned by another uid measures as zero while hiding gigabytes (an internal review) — so it must never be charged as if the sum bounded it; the
    worker refuses to start instead (operator must reclaim the work dir).
    """
    if not work_dir.is_dir():
        return {}, []
    measurable: dict[str, int] = {}
    unmeasurable: list[str] = []
    for entry in work_dir.iterdir():
        if not entry.name.startswith(SCRATCH_PREFIXES):
            continue
        size, complete = _tree_bytes(entry)
        if complete:
            measurable[str(entry)] = size
        else:
            unmeasurable.append(str(entry))
    return measurable, unmeasurable


# --- internals -----------------------------------------------------------------------


def _tree_bytes(root: Path) -> tuple[int, bool]:
    """(lstat-summed visible bytes, fully_measured) of `root` and its subtree.

    ``fully_measured`` is False when ANY entry could not be statted or any
    directory could not be traversed. The true size may then exceed the
    returned sum by an unbounded amount — a 0700 directory owned by another
    uid measures as ZERO while hiding gigabytes — so
    callers must never charge a partial sum as if it bounded the entry.
    """
    try:
        info = root.lstat()
    except OSError:
        return 0, False
    if not stat.S_ISDIR(info.st_mode):
        return int(info.st_size), True
    complete = True
    walk_errors: list[BaseException] = []
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, onerror=walk_errors.append):
        for name in filenames:
            try:
                total += int(os.lstat(os.path.join(dirpath, name)).st_size)
            except OSError:
                complete = False
    if walk_errors:
        complete = False
    return total, complete


def _path_gone(path: Path) -> bool:
    """True when nothing (not even a dangling symlink) exists at `path`.

    Only a positive "no such entry" counts: any other stat failure (EACCES on a
    broken parent, EIO) keeps the residual charge, because bytes that may still
    be on the volume must stay reserved.
    """
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _carried_suffix(path_text: str) -> str:
    suffix = Path(path_text).suffix
    return suffix if _SAFE_SUFFIX.match(suffix) else ""


class _CopyLimitExceeded(Exception):
    """The source produced more bytes than the copy was allowed to write."""


def _open_regular_file(field: str, path_text: str) -> tuple[int, int]:
    """Open `path_text` refusing symlinks; returns (fd, size) for a regular file.

    The size comes from ``fstat`` on the RETURNED descriptor, so it describes the
    bytes we are about to read — not whatever the path names a moment later.
    """
    try:
        # O_NONBLOCK: a FIFO must fail the regular-file check below, never block
        # the worker thread waiting for a writer that will never come.
        fd = os.open(path_text, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        raise ScoreRejected(
            422, {"error": "file_missing", "field": field, "path": path_text}
        ) from None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise ScoreRejected(
                422,
                {"error": "symlink_rejected", "field": field, "path": path_text},
            ) from None
        raise ScoreRejected(
            422,
            {
                "error": "unreadable_input",
                "field": field,
                "path": path_text,
                "detail": exc.strerror or str(exc),
            },
        ) from None
    try:
        info = os.fstat(fd)
    except OSError:
        os.close(fd)
        raise ScoreRejected(
            422, {"error": "unreadable_input", "field": field, "path": path_text}
        ) from None
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise ScoreRejected(
            422,
            {
                "error": "not_a_regular_file",
                "field": field,
                "path": path_text,
                "detail": _describe_mode(info.st_mode),
            },
        )
    return fd, info.st_size


def _describe_mode(mode: int) -> str:
    for predicate, label in (
        (stat.S_ISDIR, "directory"),
        (stat.S_ISFIFO, "fifo"),
        (stat.S_ISCHR, "character device"),
        (stat.S_ISBLK, "block device"),
        (stat.S_ISSOCK, "socket"),
    ):
        if predicate(mode):
            return label
    return "not a regular file"


def _hash_and_copy(
    fd: int,
    dest: Path,
    cancelled: Callable[[], bool] | None = None,
    *,
    limit: int,
) -> tuple[str, int]:
    """Stream fd -> dest (mode 0400) in one pass: returns (sha256, bytes written).

    `limit` is the reservation this copy was granted. A chunk that would carry
    the file past it is never written — the loop raises immediately, so a source
    that keeps growing costs the volume nothing beyond what was already budgeted.
    """
    digest = hashlib.sha256()
    written = 0
    out_fd = os.open(
        dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, SNAPSHOT_MODE
    )
    with os.fdopen(out_fd, "wb") as out:
        while chunk := os.read(fd, HASH_CHUNK):
            if cancelled is not None and cancelled():
                raise SnapshotCancelled(f"cancelled while snapshotting {dest.name}")
            if written + len(chunk) > limit:
                raise _CopyLimitExceeded(f"{dest.name} exceeded {limit} bytes")
            digest.update(chunk)
            out.write(chunk)
            written += len(chunk)
    os.chmod(dest, SNAPSHOT_MODE)  # defeat a permissive umask
    return digest.hexdigest(), written


def _discard(dest: Path) -> None:
    try:
        os.chmod(dest, 0o600)
    except OSError:
        pass
    try:
        dest.unlink()
    except OSError:
        pass


def _force_remove(func: Any, path: str, exc: BaseException) -> None:
    """rmtree hook: our snapshots are 0400, so clear the bit and retry once.

    Only unlink/rmdir are retryable with a bare path — shutil also reports
    traversal failures with func=os.open (which needs flags), and calling that
    with one argument raised TypeError THROUGH the sweep's `except OSError`
    guard, crashing startup (round-5). Anything non-retryable is swallowed
    here: the entry simply survives the sweep and is then measured/pre-charged
    or reported unmeasurable — never a startup crash.
    """
    try:
        os.chmod(path, 0o700)
        # Only bare-path deleters are retryable; os.open/os.lstat (traversal
        # failures) are not — leave the entry for measurement to deal with.
        if func in (os.unlink, os.rmdir, os.remove):
            func(path)
    except OSError:
        pass
