"""Scoring-worker configuration (config section: ``scoring_worker``).

Ports follow the service map in :mod:`vidaio.services.protocol` (HTTP 8201,
metrics 9103). Load via ``section(raw, "scoring_worker", ScoringWorkerConfig)``;
every field is overridable from ``config/default.yaml`` or
``VIDAIO__SCORING_WORKER__<KEY>`` env vars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from vidaio.scoring.backends_real import DEFAULT_VMAF_MODEL, SECONDARY_VMAF_MODEL
from vidaio.scoring.perceptual_cpu import CpuPerceptualConfig
from vidaio.scoring_worker.inputs import (
    DEFAULT_MAX_INPUT_BYTES,
    DEFAULT_MAX_REQUEST_BYTES,
    DEFAULT_MAX_REQUEST_SCRATCH_BYTES,
    DEFAULT_MAX_SCRATCH_BYTES,
)


class ScoringWorkerConfig(BaseModel):
    # --- HTTP surface -----------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8201
    metrics_port: int = 9103

    # --- scoring pipeline -------------------------------------------------
    #: Scratch root for canonicalization temp dirs and libvmaf JSON logs.
    work_dir: Path = Path("./data/scoring-work")
    #: "real" shells out to ffmpeg/ffprobe; "fake" requires injected
    #: deterministic backends (tests / CI without media tools) and never
    #: invents one on its own.
    backend: Literal["real", "fake"] = "real"
    #: Whole-request budget for one /score call (snapshot + canonicalize +
    #: probes + metric runs), measured from the moment the request WINS a
    #: concurrency slot. Exceeding it kills the request's ffmpeg process groups
    #: and returns a typed 504 — the budget bounds real work, not just the wait.
    request_timeout: float = 300.0
    #: Per-subprocess budget (each ffmpeg/ffprobe invocation individually).
    subprocess_timeout: float = 120.0
    #: Maximum scorings in flight (asyncio.Semaphore) — scoring is
    #: subprocess-heavy, so this is the worker's genuine parallelism knob. The
    #: slot is held for the TRUE lifetime of the work (including a timed-out
    #: request's dying subprocesses), so this is a hard bound, not an advisory.
    max_concurrent: int = 2
    #: How long a request may wait for a concurrency slot before the worker
    #: sheds it with a 503 + Retry-After. Distinct from `request_timeout`: a 503
    #: means "we never started, come back", a 504 means "we started and gave
    #: up". Without this bound a saturated worker queues callers forever.
    queue_wait_timeout_seconds: float = 30.0

    # --- scratch byte budgets (vidaio.scoring_worker.inputs) ---------------
    # Scoring COPIES every input into the worker's private scratch and then
    # EXPANDS both sides into raw y4m before measuring them, so an unbounded
    # input is an unbounded write to our own volume — and the expansion, not the
    # copy, is the big one (raw video is ~1000x its encoding). Every ceiling is
    # enforced before the bytes exist: an oversize input never gets written, a
    # source that grows mid-copy is cut off at its reservation, and an expansion
    # whose PROJECTED size does not fit is refused before ffmpeg starts.
    #: Largest single input (422 `input_too_large`, decided from the fstat of
    #: the descriptor that would be read). 2 GiB = the reference miner's own
    #: ingress ceiling; a challenge clip is orders of magnitude smaller.
    max_input_bytes: int = Field(DEFAULT_MAX_INPUT_BYTES, gt=0)
    #: Largest total across ONE request's reference + miner_input + output
    #: (413 `request_inputs_too_large`). Inputs only — see the next field for
    #: what the request goes on to generate from them.
    max_request_bytes: int = Field(DEFAULT_MAX_REQUEST_BYTES, gt=0)
    #: ALL scratch ONE request may hold: its snapshots plus the canonicalized
    #: y4m of both sides plus the libvmaf logs (413 `request_scratch_too_large`).
    #: This is the ceiling a highly-compressed long/high-resolution clip hits —
    #: it passes every input cap and is then refused on what it would DECODE to.
    #: 413 rather than 503 on purpose: a request that cannot fit in one request's
    #: allowance can never fit, so shedding it would shed it forever.
    max_request_scratch_bytes: int = Field(DEFAULT_MAX_REQUEST_SCRATCH_BYTES, gt=0)
    #: Scratch bytes the whole worker may hold live at once, across all
    #: concurrent requests and covering generated files as well as snapshots
    #: (503 `scratch_budget_unavailable` + Retry-After). Keep it at or above
    #: ``max_concurrent * max_request_scratch_bytes`` or a fully loaded worker
    #: sheds its own legitimate load.
    max_scratch_bytes: int = Field(DEFAULT_MAX_SCRATCH_BYTES, gt=0)

    #: Perceptual manipulation gates (tone / color-grayscale / chroma-UV).
    #:   "required" (DEFAULT): the deterministic CPU checks must
    #:     actually run. Without a configured backend every request is refused
    #:     with a typed 501 — an honest refusal, never a substituted pass.
    #:   "skip": the three gates are CONSCIOUSLY not run, and each records a
    #:     GateSkip in the ItemScore packet naming this flag — exactly the
    #:     mechanism require_secondary_vmaf=False uses. Nothing is faked: a
    #:     "skip"-mode packet is permanently distinguishable from a packet that
    #:     genuinely passed the manipulation checks. It exists only for explicit
    #:     diagnostics/failure-injection; production preflight rejects it.
    perceptual_checks: Literal["required", "skip"] = "required"

    #: Fixed CPU sampling/threshold surface for the required manipulation gates.
    #: Its digest and backend algorithm/version are recorded in every packet and
    #: every field participates in the effective scorer identity.
    perceptual_cpu: CpuPerceptualConfig = Field(default_factory=CpuPerceptualConfig)

    #: Device used by the SCORING worker's PieAPP model. Auditors ignore this
    #: field and explicitly select CPU so validation never depends on CUDA.
    pieapp_device: Literal["cpu", "cuda"] = "cpu"

    # --- scorer identity --------------------------------------------------
    #: Human-readable NAME of this scorer build. The version the worker actually
    #: stamps into every packet is this name plus a digest of the scoring
    #: configuration AND canonical payout runtime that produced the score — see
    #: :func:`vidaio.scoring_worker.service.effective_scorer_version`. The worker
    #: NEVER stamps a version supplied by the caller: a request that names a
    #: different scorer is rejected 409 rather than answered by the wrong scorer.
    scorer_version: str = "vidaio-scorer/1"

    # --- media tooling ----------------------------------------------------
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    #: Pinned libvmaf models: primary scores, secondary feeds the
    #: vmaf_model_delta gate (NEG variant — see backends_real module docstring).
    vmaf_model_primary: str = DEFAULT_VMAF_MODEL
    vmaf_model_secondary: str = SECONDARY_VMAF_MODEL

    @model_validator(mode="after")
    def _sane(self) -> "ScoringWorkerConfig":
        for name, port in (("port", self.port), ("metrics_port", self.metrics_port)):
            if not 0 <= port <= 65535:  # 0 = ephemeral (tests)
                raise ValueError(f"{name} must be in [0, 65535], got {port}")
        for name, value in (
            ("request_timeout", self.request_timeout),
            ("subprocess_timeout", self.subprocess_timeout),
            ("queue_wait_timeout_seconds", self.queue_wait_timeout_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
        if self.max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {self.max_concurrent}")
        # The ceilings must nest, widest last:
        #
        #   max_scratch_bytes >= max_request_scratch_bytes
        #                     >= max_request_bytes >= max_input_bytes
        #
        # Any inversion turns a cap into a lie: a budget that cannot fit one
        # legal request refuses every request, and a per-request cap below the
        # per-file cap means the file cap can never be the binding one.
        if self.max_request_bytes < self.max_input_bytes:
            raise ValueError(
                "max_request_bytes must be >= max_input_bytes, got "
                f"{self.max_request_bytes} < {self.max_input_bytes}"
            )
        if self.max_scratch_bytes < self.max_request_bytes:
            raise ValueError(
                "max_scratch_bytes must be >= max_request_bytes (a worker that "
                "cannot hold one request's inputs can never score), got "
                f"{self.max_scratch_bytes} < {self.max_request_bytes}"
            )
        # An EXPLICIT per-request scratch ceiling above the worker-wide budget is
        # a contradiction the operator wrote and should hear about. Left at its
        # default it is simply clamped to the worker budget
        # (ByteLimits.request_scratch_ceiling), so a deployment that only tunes
        # max_scratch_bytes still gets a coherent pair.
        if (
            "max_request_scratch_bytes" in self.model_fields_set
            and self.max_scratch_bytes < self.max_request_scratch_bytes
        ):
            raise ValueError(
                "max_scratch_bytes must be >= max_request_scratch_bytes, got "
                f"{self.max_scratch_bytes} < {self.max_request_scratch_bytes}"
            )
        if self.request_scratch_ceiling < self.max_request_bytes:
            raise ValueError(
                "max_request_scratch_bytes must be >= max_request_bytes (a "
                "request allowed to snapshot more than its total scratch "
                "allowance could never canonicalize), got "
                f"{self.request_scratch_ceiling} < {self.max_request_bytes}"
            )
        if not self.scorer_version.strip():
            raise ValueError("scorer_version must be a non-empty name")
        return self

    @property
    def request_scratch_ceiling(self) -> int:
        """Effective per-request scratch ceiling — never above the worker budget.

        A request larger than the entire volume can never run, so it must be
        refused deterministically (413) rather than shed (503) on every retry.
        """
        return min(self.max_request_scratch_bytes, self.max_scratch_bytes)
