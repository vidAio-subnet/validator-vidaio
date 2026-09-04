"""Validity gates — run FIRST; any failure forces score 0 with a machine-readable reason.

Spec §18 ("validity gates first, then complementary metrics"): one excellent metric must
never offset a task failure, so gating precedes composition and is absolute. Every gate
emits :class:`ValidityViolation` records with a :class:`ReasonCode` so audits and
dashboards can classify failures without parsing prose.

The pipeline collects *all* violations (no short-circuit) — the full failure picture is
part of the audit record — then the item passes only when the list is empty.

Fail-closed principle: a metric the gates cannot actually evaluate — missing, NaN or
+/-inf on either side of a comparison — is a violation (``METRIC_MISSING`` /
``METRIC_NON_FINITE``), never a silent pass. The only sanctioned way to skip a check
is an explicit config flag, which records an informational :class:`GateSkip` on the
context so the omission itself is auditable
(``require_secondary_vmaf=False`` on the model-delta gate;
``scoring_worker.perceptual_checks="skip"`` via
:func:`pipeline_without_perceptual_checks` on the three perceptual gates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, Sequence

from pydantic import BaseModel

from vidaio.scoring.backends import (
    MediaInfo,
    PerceptualCheckBackend,
    PerceptualCheckResult,
)
from vidaio.scoring.config import TRACK_COMPRESSION, TRACK_UPSCALING, ScoringConfig


class ReasonCode(StrEnum):
    """Machine-readable zero-score reasons. Stable strings — stored in the audit record."""

    ENCODING_NOT_ALLOWED = "ENCODING_NOT_ALLOWED"
    FRAME_COUNT_MISMATCH = "FRAME_COUNT_MISMATCH"
    FILE_SIZE_CAP_EXCEEDED = "FILE_SIZE_CAP_EXCEEDED"
    COMPRESSION_RATE_TOO_HIGH = "COMPRESSION_RATE_TOO_HIGH"
    VMAF_BELOW_FLOOR = "VMAF_BELOW_FLOOR"
    VMAF_BELOW_THRESHOLD = "VMAF_BELOW_THRESHOLD"
    VMAF_MODEL_DELTA_EXCEEDED = "VMAF_MODEL_DELTA_EXCEEDED"
    TONE_MANIPULATION = "TONE_MANIPULATION"
    COLOR_GRAYSCALE = "COLOR_GRAYSCALE"
    CHROMA_UV_MANIPULATION = "CHROMA_UV_MANIPULATION"
    STREAM_DIMENSIONS_MISMATCH = "STREAM_DIMENSIONS_MISMATCH"
    STREAM_PIX_FMT_MISMATCH = "STREAM_PIX_FMT_MISMATCH"
    STREAM_DURATION_MISMATCH = "STREAM_DURATION_MISMATCH"
    STREAM_PTS_INCONSISTENT = "STREAM_PTS_INCONSISTENT"
    REPLAY_DUPLICATE = "REPLAY_DUPLICATE"
    MINER_TIMEOUT = "MINER_TIMEOUT"
    MINER_TRANSPORT_ERROR = "MINER_TRANSPORT_ERROR"
    MINER_TASK_ID_MISMATCH = "MINER_TASK_ID_MISMATCH"
    MINER_OUTPUT_DIGEST_MISMATCH = "MINER_OUTPUT_DIGEST_MISMATCH"
    METRIC_MISSING = "METRIC_MISSING"
    METRIC_NON_FINITE = "METRIC_NON_FINITE"
    UNSUPPORTED_SCALE_FACTOR = "UNSUPPORTED_SCALE_FACTOR"


class ValidityViolation(BaseModel):
    """One gate/validation failure — reason code plus the measured value and limit."""

    model_config = {"frozen": True}

    code: ReasonCode
    detail: str = ""
    measured: float | None = None
    limit: float | None = None


class GateSkip(BaseModel):
    """Informational record of a check consciously disabled by config.

    A skip is NOT a pass-by-omission: it exists precisely so the audit record shows
    which check did not run and which config flag turned it off.
    """

    model_config = {"frozen": True}

    gate: str
    detail: str = ""


def _non_finite_violation(name: str, value: float) -> ValidityViolation:
    """A METRIC_NON_FINITE violation. ``measured`` stays None — NaN/inf must not
    enter the JSON audit record as a number."""
    return ValidityViolation(
        code=ReasonCode.METRIC_NON_FINITE,
        detail=f"{name} is non-finite ({value!r})",
    )


@dataclass
class GateContext:
    """Everything the gates may inspect for one item. Pure data, no I/O.

    ``reference_info`` is the pristine held-out original; ``input_info`` and
    ``input_path`` are the payload the miner actually received. Size caps,
    compression rate, anti-gaming VMAF, and perceptual-manipulation gates all use
    that canonical miner-input basis. Scored quality remains pristine based.
    """

    track: str
    config: ScoringConfig
    reference_info: MediaInfo
    candidate_info: MediaInfo
    reference_path: str = ""
    candidate_path: str = ""
    input_info: MediaInfo | None = None
    #: Canonical miner input at candidate geometry. Legacy direct callers may
    #: omit it, in which case the pristine path remains the compatibility basis.
    input_path: str = ""
    vmaf_primary: float | None = None
    vmaf_secondary: float | None = None
    #: The anti-gaming model pair measured against the MINER INPUT. The scored
    #: ``vmaf_primary`` above remains pristine-reference based. Older direct
    #: GateContext callers may omit these and retain the legacy pair.
    vmaf_delta_primary: float | None = None
    vmaf_delta_secondary: float | None = None
    upscale_factor: int | None = None
    #: Violations produced upstream (e.g. canonicalize.validate_stream) — folded in.
    extra_violations: list[ValidityViolation] = field(default_factory=list)
    #: Informational skips appended by gates whose check a config flag disabled —
    #: part of the audit record, inspected after :meth:`GatePipeline.run`.
    skips: list[GateSkip] = field(default_factory=list)
    #: Measured CPU perceptual results, retained so the score packet records
    #: successful checks as well as violations.
    perceptual_results: dict[str, PerceptualCheckResult] = field(default_factory=dict)

    @property
    def effective_input(self) -> MediaInfo:
        return self.input_info if self.input_info is not None else self.reference_info

    @property
    def effective_input_path(self) -> str:
        return self.input_path if self.input_path else self.reference_path


class Gate(Protocol):
    name: str

    def check(self, ctx: GateContext) -> list[ValidityViolation]: ...


class EncodingGate:
    """Candidate codec must be on the allowlist."""

    name = "encoding"

    def check(self, ctx: GateContext) -> list[ValidityViolation]:
        codec = ctx.candidate_info.codec.lower()
        if codec not in ctx.config.codec_allowlist:
            return [
                ValidityViolation(
                    code=ReasonCode.ENCODING_NOT_ALLOWED,
                    detail=f"codec {codec!r} not in allowlist {list(ctx.config.codec_allowlist)}",
                )
            ]
        return []


class FrameCountGate:
    """Candidate must carry exactly the reference frame count (no drops/pads)."""

    name = "frame_count"

    def check(self, ctx: GateContext) -> list[ValidityViolation]:
        ref, cand = ctx.reference_info.frame_count, ctx.candidate_info.frame_count
        if cand != ref:
            return [
                ValidityViolation(
                    code=ReasonCode.FRAME_COUNT_MISMATCH,
                    detail=f"candidate has {cand} frames, reference has {ref}",
                    measured=float(cand),
                    limit=float(ref),
                )
            ]
        return []


class FileSizeCapGate:
    """Upscaling only: candidate bytes <= cap(upscale_factor) * input bytes (8x / 20x).

    Fails CLOSED: the supported upscale factors are exactly the keys of
    ``file_size_caps`` — an absent or unsupported factor has no cap to enforce, so it
    is an ``UNSUPPORTED_SCALE_FACTOR`` violation, never a pass. (The challenge DAG is
    separately constrained to only issue supported factors; scoring never trusts that.)
    """

    name = "file_size_cap"

    def check(self, ctx: GateContext) -> list[ValidityViolation]:
        if ctx.track != TRACK_UPSCALING:
            return []
        caps = ctx.config.file_size_caps
        if ctx.upscale_factor is None:
            return [
                ValidityViolation(
                    code=ReasonCode.UNSUPPORTED_SCALE_FACTOR,
                    detail=f"upscale factor absent; supported factors: {sorted(caps)}",
                )
            ]
        if ctx.upscale_factor not in caps:
            return [
                ValidityViolation(
                    code=ReasonCode.UNSUPPORTED_SCALE_FACTOR,
                    detail=(
                        f"upscale factor {ctx.upscale_factor} not in "
                        f"supported factors {sorted(caps)}"
                    ),
                    measured=float(ctx.upscale_factor),
                )
            ]
        limit = caps[ctx.upscale_factor] * ctx.effective_input.byte_size
        if not math.isfinite(limit):
            return [_non_finite_violation("file-size cap limit", limit)]
        measured = ctx.candidate_info.byte_size
        if measured > limit:
            return [
                ValidityViolation(
                    code=ReasonCode.FILE_SIZE_CAP_EXCEEDED,
                    detail=(
                        f"candidate {measured} bytes exceeds "
                        f"{caps[ctx.upscale_factor]}x cap over input"
                    ),
                    measured=float(measured),
                    limit=float(limit),
                )
            ]
        return []


class CompressionRateGate:
    """Compression only: rate = candidate_bytes / input_bytes must be < 0.80."""

    name = "compression_rate"

    def check(self, ctx: GateContext) -> list[ValidityViolation]:
        if ctx.track != TRACK_COMPRESSION:
            return []
        input_bytes = ctx.effective_input.byte_size
        if input_bytes <= 0:
            return [
                ValidityViolation(
                    code=ReasonCode.COMPRESSION_RATE_TOO_HIGH,
                    detail="input byte size is zero — rate undefined",
                )
            ]
        rate = ctx.candidate_info.byte_size / input_bytes
        if rate >= ctx.config.compression_rate_max:
            return [
                ValidityViolation(
                    code=ReasonCode.COMPRESSION_RATE_TOO_HIGH,
                    detail="compression rate at/above maximum",
                    measured=rate,
                    limit=ctx.config.compression_rate_max,
                )
            ]
        return []


class VmafFloorGate:
    """Per-track hard VMAF floor (upscaling: vmaf/100 < gate; compression: threshold-band).

    Fails CLOSED: a primary VMAF the gate cannot compare — missing or non-finite —
    is a violation, not a pass.
    """

    name = "vmaf_floor"

    def check(self, ctx: GateContext) -> list[ValidityViolation]:
        if ctx.vmaf_primary is None:
            return [
                ValidityViolation(
                    code=ReasonCode.METRIC_MISSING,
                    detail="primary VMAF was not measured",
                )
            ]
        if not math.isfinite(ctx.vmaf_primary):
            return [_non_finite_violation("primary VMAF", ctx.vmaf_primary)]
        floor = ctx.config.vmaf_floor(ctx.track)
        if not math.isfinite(floor):
            return [_non_finite_violation(f"{ctx.track} VMAF floor", floor)]
        if ctx.vmaf_primary < floor:
            return [
                ValidityViolation(
                    code=ReasonCode.VMAF_BELOW_FLOOR,
                    detail=f"vmaf below {ctx.track} floor",
                    measured=ctx.vmaf_primary,
                    limit=floor,
                )
            ]
        return []


class VmafModelDeltaGate:
    """Miner-added |primary - secondary| delta must be <= the configured maximum.

    Scoring quality remains measured against the pristine reference. This gate's
    two model runs instead use the payload the miner actually received as their
    reference, so degradation-DAG tone gain cancels and miner-added enhancement
    remains visible. Downscaled inputs are Lanczos-rescaled to output geometry.

    Fails CLOSED: an absent or non-finite run is a violation. A track whose pipeline
    genuinely has no secondary model run must disable the requirement explicitly via
    ``ScoringConfig.require_secondary_vmaf = False``, which records an informational
    :class:`GateSkip` on the context — never a silent pass by omission.
    """

    name = "vmaf_model_delta"

    def check(self, ctx: GateContext) -> list[ValidityViolation]:
        violations: list[ValidityViolation] = []
        delta_pair_supplied = (
            ctx.vmaf_delta_primary is not None or ctx.vmaf_delta_secondary is not None
        )
        primary = ctx.vmaf_delta_primary if delta_pair_supplied else ctx.vmaf_primary
        secondary = (
            ctx.vmaf_delta_secondary if delta_pair_supplied else ctx.vmaf_secondary
        )
        if primary is None:
            violations.append(
                ValidityViolation(
                    code=ReasonCode.METRIC_MISSING,
                    detail="primary VMAF was not measured (model-delta gate)",
                )
            )
        elif not math.isfinite(primary):
            violations.append(_non_finite_violation("delta-primary VMAF", primary))
        if secondary is None:
            if ctx.config.require_secondary_vmaf:
                violations.append(
                    ValidityViolation(
                        code=ReasonCode.METRIC_MISSING,
                        detail=(
                            "secondary VMAF run absent while "
                            "require_secondary_vmaf is enabled"
                        ),
                    )
                )
            else:
                ctx.skips.append(
                    GateSkip(
                        gate=self.name,
                        detail=(
                            "secondary VMAF run absent; check disabled by "
                            "require_secondary_vmaf=False"
                        ),
                    )
                )
        elif not math.isfinite(secondary):
            violations.append(_non_finite_violation("delta-secondary VMAF", secondary))
        if violations or primary is None or secondary is None:
            return violations
        delta = abs(primary - secondary)
        if not math.isfinite(ctx.config.vmaf_model_delta_max):
            return [
                _non_finite_violation(
                    "vmaf_model_delta_max", ctx.config.vmaf_model_delta_max
                )
            ]
        if delta > ctx.config.vmaf_model_delta_max:
            return [
                ValidityViolation(
                    code=ReasonCode.VMAF_MODEL_DELTA_EXCEEDED,
                    detail="two VMAF model runs disagree beyond tolerance",
                    measured=delta,
                    limit=ctx.config.vmaf_model_delta_max,
                )
            ]
        return []


class _PerceptualGate:
    """Base for backend-driven perceptual manipulation checks."""

    name = "perceptual"
    code: ReasonCode = ReasonCode.TONE_MANIPULATION

    def __init__(self, backend: PerceptualCheckBackend) -> None:
        self._backend = backend

    def _run(self, ctx: GateContext):  # pragma: no cover - overridden
        raise NotImplementedError

    def check(self, ctx: GateContext) -> list[ValidityViolation]:
        result = self._run(ctx)
        ctx.perceptual_results[self.name] = result
        if not result.passed:
            return [
                ValidityViolation(
                    code=self.code,
                    detail=result.detail or f"{self.name} check failed",
                    measured=result.measure,
                    limit=result.limit,
                )
            ]
        return []


class ToneManipulationGate(_PerceptualGate):
    name = "tone_manipulation"
    code = ReasonCode.TONE_MANIPULATION

    def _run(self, ctx: GateContext):
        return self._backend.check_tone_manipulation(
            ctx.effective_input_path, ctx.candidate_path
        )


class ColorGrayscaleGate(_PerceptualGate):
    name = "color_grayscale"
    code = ReasonCode.COLOR_GRAYSCALE

    def _run(self, ctx: GateContext):
        return self._backend.check_color_grayscale(
            ctx.effective_input_path, ctx.candidate_path
        )


class ChromaUvGate(_PerceptualGate):
    name = "chroma_uv"
    code = ReasonCode.CHROMA_UV_MANIPULATION

    def _run(self, ctx: GateContext):
        return self._backend.check_chroma_uv(
            ctx.effective_input_path, ctx.candidate_path
        )


class SkippedGate:
    """A gate whose check is consciously NOT run, by explicit configuration.

    It measures nothing, invents nothing and never passes anything: it records a
    :class:`GateSkip` on the context (the same sanctioned mechanism
    ``require_secondary_vmaf=False`` uses) so the audit packet shows exactly
    which check did not run and which config flag turned it off. A skipped gate
    can never contribute a violation, and it can never contribute a value.
    """

    def __init__(self, name: str, detail: str) -> None:
        self.name = name
        self._detail = detail

    def check(self, ctx: GateContext) -> list[ValidityViolation]:
        ctx.skips.append(GateSkip(gate=self.name, detail=self._detail))
        return []


class GatePipeline:
    """Runs every gate, collects all violations, returns (passed, violations)."""

    def __init__(self, gates: Sequence[Gate]) -> None:
        self.gates = list(gates)

    def run(self, ctx: GateContext) -> tuple[bool, list[ValidityViolation]]:
        violations: list[ValidityViolation] = list(ctx.extra_violations)
        for gate in self.gates:
            violations.extend(gate.check(ctx))
        return (len(violations) == 0, violations)


def _structural_gates() -> list[Gate]:
    """The measured, backend-free gates every pipeline runs, in order."""
    return [
        EncodingGate(),
        FrameCountGate(),
        FileSizeCapGate(),
        CompressionRateGate(),
        VmafFloorGate(),
        VmafModelDeltaGate(),
    ]


#: Names of the three backend-driven perceptual gates, in pipeline order.
PERCEPTUAL_GATE_NAMES = ("tone_manipulation", "color_grayscale", "chroma_uv")


def default_pipeline(perceptual_backend: PerceptualCheckBackend) -> GatePipeline:
    """The standard gate order for both tracks (track-specific gates self-skip)."""
    return GatePipeline(
        [
            *_structural_gates(),
            ToneManipulationGate(perceptual_backend),
            ColorGrayscaleGate(perceptual_backend),
            ChromaUvGate(perceptual_backend),
        ]
    )


def pipeline_without_perceptual_checks(*, reason: str) -> GatePipeline:
    """The standard pipeline with the three perceptual gates consciously SKIPPED.

    For explicit diagnostic/test compositions where the perceptual-check backend
    is disabled, the tone/color/chroma checks do not run and each records its own
    :class:`GateSkip` naming `reason`. Release scoring uses the deterministic CPU
    backend and production preflight rejects this skipped posture. NOTHING is
    faked: a diagnostic packet scored this way carries three skip records, so a
    reader can always distinguish a checked packet from one that never ran them.
    """
    return GatePipeline(
        [
            *_structural_gates(),
            *(SkippedGate(name, reason) for name in PERCEPTUAL_GATE_NAMES),
        ]
    )
