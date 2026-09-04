"""Typed sandbox-runner errors.

Every docker/subprocess boundary in the runners surfaces failure as one of these —
callers never have to parse stderr strings to branch. The orchestrator maps them
to CONTENDER-level outcomes (BUILD_FAILED, batch requeue); none of them may fail a
competition on its own (spec §14: infra failures requeue rather than fail).

FAULT CLASSES. Every error here belongs to exactly one
of two classes, and the class is carried by the TYPE, never by string matching:

- ``ContenderFaultError`` — the contender's own submission caused it (its solution
  exited non-zero, blew its per-contender time budget, wrote an unsafe or oversize
  output, or its own Dockerfile/build context failed to build). The orchestrator
  zero-scores that contender's item and the competition CONTINUES. One untrusted
  contender can never halt the pipeline.
- everything else — INFRA (docker daemon unreachable, image gone, a sealed input
  missing from the pool, a runner whose isolation flags did not take effect, a
  probe that could not be RUN). The orchestrator requeues with a bounded budget and
  HALTS on exhaustion; scores are never substituted around an infra blocker.

``vidaio.competition.orchestrator.failures.classify_failure`` is the single place
that maps an exception to its class.

BUILD + PROBE SPLIT
------------------------------------------------------
"the build failed" and "the probe failed" each cover two different faults, and the
orchestrator used to collapse both pairs onto the contender:

- ``ContenderBuildError`` — ``docker build`` itself rejected the contender's
  Dockerfile/context (or the build blew its bounded timeout). CONTENDER fault ->
  BUILD_FAILED, competition continues.
  ``BuildError`` stays INFRA — it now means "WE could not run a build" (the docker
  CLI is unusable, `image inspect`/`tag` failed after a successful build). That
  halts; it never marks a contender BUILD_FAILED, because the contender did
  nothing wrong.
- ``SandboxProbeUnavailableError`` — the isolation probe could not be RUN at all
  (image unresolvable, `docker run` refused, host inspection failed). INFRA: an
  unattestable boundary is OUR problem, so it retries/halts. A probe that RAN and
  reported a violated boundary is a different thing entirely: it returns a normal
  failing ``IsolationProbeReport`` and disqualifies that contender.
"""

from __future__ import annotations

from vidaio.audit import NotConfiguredError


class SandboxRunnerError(Exception):
    """Base class for all runner failures."""


class ContenderFaultError(SandboxRunnerError):
    """The CONTENDER's submission is at fault — never the infrastructure.

    Zero-scores that contender's affected item(s) with a reason code; the
    competition keeps running.
    """

    #: Stable machine-readable reason, persisted on the batch row and in the event
    #: log so an operator (and the audit trail) can tell the fault classes apart.
    code = "CONTENDER_FAULT"


class RunnerUnavailableError(NotConfiguredError, SandboxRunnerError):
    """The container runtime itself is unusable (docker missing/daemon down).

    Raised at CONSTRUCTION (fail fast — spec §14 health discipline): a runner that
    cannot reach its runtime must never be handed to the orchestrator.
    """


class CheckoutError(SandboxRunnerError):
    """The contender's pinned code identity could not be materialized locally."""


class CheckoutRejectedError(CheckoutError, ContenderFaultError):
    """CONTENDER: remote source bytes violate the pinned-checkout contract.

    The fetched commit/tree does not match enrollment, the checkout exceeds the
    byte cap, contains unsafe links/special files, or requests unpinned
    submodules. Auth, transport, Git availability and timeout failures remain the
    plain ``CheckoutError`` infrastructure class.
    """

    code = "CHECKOUT_REJECTED"


class BuildError(SandboxRunnerError):
    """INFRA: WE could not run a build for this contender.

    The docker CLI could not be executed, or a post-build `image inspect`/`tag`
    against our own daemon failed. Nothing here says the submission is bad, so it
    must NOT mark the contender BUILD_FAILED — it takes the bounded retry/halt
    path like any other infra blocker.
    Carries the stderr tail.
    """


class ContenderBuildError(BuildError, ContenderFaultError):
    """CONTENDER: the submission's own image failed to build.

    ``docker build`` exited non-zero on the contender's Dockerfile/context, or the
    pinned checkout has no usable Dockerfile. That is the submission's fault: the
    contender is marked BUILD_FAILED with a reason code and the competition keeps
    running. Subclasses BuildError so existing `except BuildError` sites still
    catch it, but `classify_failure` sees the ContenderFaultError base first.
    """

    code = "BUILD_FAILED"


class BuildTimeout(ContenderBuildError):
    """CONTENDER: the submission's build exceeded its bounded timeout and was killed.

    Same reasoning as BatchTimeout: the build runs with ``--pull=false`` against a
    local daemon, so the only thing that can make it take longer than the budget is
    the contender's own build graph. A contender does not get to stall (or halt)
    the competition by shipping an unbounded Dockerfile.
    """

    code = "BUILD_TIMEOUT"


class BatchExecutionError(SandboxRunnerError):
    """INFRA: the batch could not be run/collected for a reason outside the
    contender's control (docker refused the run, the image is gone, a sealed input
    is missing). Carries the stderr tail."""


class BatchTimeout(ContenderFaultError):
    """CONTENDER: the solution exceeded its bounded per-contender time budget; the
    container was force-removed. A contender that never finishes is scored zero —
    it does not get to stall (or halt) the competition."""

    code = "SOLUTION_TIMEOUT"


class SolutionExitError(ContenderFaultError):
    """CONTENDER: ``/app/run.sh`` exited non-zero. Carries the stderr tail.

    Deliberately NOT a BatchExecutionError: a solution's own `exit 1` is the most
    basic contender fault there is, and treating it as systemic infra is exactly
    the halt-everything bug of review service-review #14."""

    code = "SOLUTION_EXIT_NONZERO"


class OutputRejectedError(ContenderFaultError):
    """CONTENDER: an output entry is not a plain regular file inside the sandbox's
    own output root (symlink, hardlink to elsewhere, fifo, device, directory) — or
    it changed identity between inspection and read. Never followed, never
    archived."""

    code = "OUTPUT_REJECTED"


class OversizeOutputError(ContenderFaultError):
    """CONTENDER: the solution's output exceeded the per-output or per-batch byte
    cap. The container is force-removed mid-run when the
    cap is crossed, so a contender can never fill the validator's disk."""

    code = "OUTPUT_OVERSIZE"


class UnsafePathError(ContenderFaultError):
    """CONTENDER: a path inside contender-controlled bytes (submission tree,
    sandbox output) is not a regular file/dir and must never be read, followed or
    archived."""

    code = "UNSAFE_PATH"


class SandboxProbeUnavailableError(SandboxRunnerError):
    """INFRA: the isolation probe could not be RUN, so nothing was attested.

    The image could not be resolved, `docker run` refused to start the probe
    container, or the host `docker inspect` that produces the authoritative verdict
    failed. Distinct from a probe that RAN and reported a violated boundary (that
    is a normal failing report and disqualifies the contender): here we have no
    verdict at all, and manufacturing an all-False report out of OUR outage would
    disqualify an innocent contender.
    """


class SandboxIsolationError(SandboxRunnerError):
    """INFRA: the isolation flags did not take effect on a container we launched
    (host `docker inspect` disagrees with the contract). Any result produced under
    a broken boundary is tainted and is never scored — this halts rather than
    zeroing a contender, because the fault is ours."""


class UnknownImageError(BatchExecutionError):
    """run_batch/isolation_probe was asked for an image_digest never built here
    (or garbage-collected) — the caller must (re)build first."""


class InputStagingError(BatchExecutionError):
    """A sealed input referenced by the batch is missing from the local input pool
    or does not hash to its declared sha256 — an infra/integrity failure, never
    attributable to the contender."""
