"""Scoring worker — the HTTP service that turns one miner output into an ItemScore.

POST /score executes the full honest pipeline over the shared-filesystem artifacts
named by :class:`~vidaio.services.protocol.ScoreRequest`:

  0. admission: the request must be for THIS scorer (409 otherwise, see
     "Scorer identity" below) and must win a concurrency slot within
     ``queue_wait_timeout_seconds`` (503 + Retry-After otherwise);
  1. verify-then-snapshot EVERY input: hash the bytes through a single
     symlink-refusing descriptor and copy them, from that same descriptor, into
     a private read-only working copy (:mod:`vidaio.scoring_worker.inputs`). Any
     mismatch, missing file, symlink or non-regular file is a 422 typed
     rejection; nothing unverified is ever scored, and nothing but the private
     copies is ever read again — the bytes that were verified are exactly the
     bytes that get measured. Because that step COPIES, it is bounded on three
     axes before and during the copy: per file (422), per request (413) and
     per worker across all live requests (503 + Retry-After), so no caller can
     amplify its way through the scoring volume;
  2. probe the originals — the gate facts (codec, bytes) AND the geometry that
     says how large step 3 is about to become;
  3. canonicalize reference + miner input + candidate via the scoring module's argv plans
     (executed by :class:`~vidaio.scoring.backends_real.CanonicalizeExecutor`;
     y4m/rawvideo output — genuinely lossless, so no encoder default can slip a
     lossy re-encode into the comparison). Raw video is ~1000x its encoding, so
     the size of both outputs is PROJECTED from step 2 and reserved against the
     same worker-wide scratch budget before ffmpeg runs (413 if it can never fit,
     503 + Retry-After if it just does not fit now) and bounded again while it
     runs (the process group dies if the projection was wrong);
  4. probe the canonical files (stream-consistency facts), fold
     ``validate_stream`` violations in;
  5. measure metrics per track — compression: primary + secondary-model VMAF
     (model-delta gate) and the byte ratio; upscaling: PieAPP at the
     deterministically derived start frame plus the VMAF gate runs;
  6. run the standard gate pipeline and compose the audit-grade
     :class:`~vidaio.scoring.result.ItemScore` — exact JSON bytes + sha256 back as
     :class:`~vidaio.services.protocol.ScoreResponse`.

Bounded scratch: every file a request creates lives under ONE per-request
directory — the snapshots, the canonicalized y4m, and the temp dirs libvmaf makes
for its JSON logs — and every one of them is reserved against the shared
:class:`~vidaio.scoring_worker.inputs.ScratchBudget` before it exists. The
directory dies with the request, its reservation is released after the directory
is gone, and whatever a crash leaves behind is reclaimed by the startup sweep.

Honesty boundaries: a backend whose pinned dependency/model cache is not configured
yields a 501 typed error, NEVER a substituted score; media-tool failures
are 502; budget overruns are 504. The only 200s are genuinely measured packets
(gate-failed packets included — a zero with reasons is a measurement).

Scorer identity: the worker stamps ITS OWN ``scorer_version``
(:func:`effective_scorer_version` — the configured name plus a digest of the
scoring configuration and canonical payout runtime that produced the score),
never the caller's. A request
that carries a *different* ``scorer_version`` is refused with 409: a validator
expecting scorer X must not silently receive packets measured by scorer Y, since
the audit bundle cross-checks packet against bundle scorer version. Callers that
do not care may omit the field; the worker's value is discoverable on /healthz.

Bounded work: the per-request budget starts when the request WINS a concurrency
slot, and a timeout kills that request's ffmpeg process groups
(:class:`~vidaio.scoring.backends_real.MediaProcessScope`). The slot is released
only once the worker thread has genuinely finished or died, so ``max_concurrent``
is a hard bound on real CPU/disk load — not merely on coroutines that are still
being awaited.

The one sanctioned diagnostic mode is ``scoring_worker.perceptual_checks="skip"``
(config.py): the three perceptual manipulation gates are consciously NOT run and
each records a GateSkip in the packet naming the flag. That keeps an explicit
failure-injection/diagnostic composition runnable while every such packet remains
distinguishable from one that actually passed those checks. Production keeps the
default "required" deterministic CPU implementation.

Recomputability: every field of the packet is a pure function of the request and
the pinned tool versions — no timestamps, no host paths (the canonicalization
digest recorded is the path-independent *template* digest), no randomness — so
re-scoring the same artifacts yields byte-identical packet JSON and digest.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
from typing import Any, Callable, Generator, Mapping

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, Counter, Histogram

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.core import log_fields, section
from vidaio.scoring import (
    TRACK_COMPRESSION,
    TRACK_UPSCALING,
    GateContext,
    MediaInfo,
    PerceptualCheckBackend,
    PieAppBackend,
    ProbeBackend,
    ReasonCode,
    ScoringConfig,
    ValidityViolation,
    VmafBackend,
    build_canonicalization_plan,
    compose_item_score,
    default_pipeline,
    derive_pieapp_start_frame,
    pipeline_without_perceptual_checks,
    plan_template_digest,
    score_compression,
    score_upscaling,
    usable_frames,
    validate_stream,
)
from vidaio.scoring.backends_real import (
    CanonicalizationTooLarge,
    CanonicalizeExecutor,
    CpuPerceptualCheckBackend,
    FfmpegVmafBackend,
    FfprobeBackend,
    MediaProcessScope,
    MediaToolError,
    MediaToolTimeout,
    MediaWorkCancelled,
    MetricLogTooLarge,
    NotConfiguredError,
    PieAppTorchBackend,
    detect_tool_versions,
    use_media_scratch,
    use_metric_log_limit,
    use_process_scope,
)
from vidaio.scoring.result import ItemScore, config_digest
from vidaio.scoring_worker.config import ScoringWorkerConfig
from vidaio.scoring_worker.inputs import (
    WORK_PREFIX,
    ByteLimits,
    InputSnapshot,
    ScoreRejected,
    ScratchBudget,
    ScratchLease,
    SnapshotCancelled,
    measure_scratch_entries,
    projected_canonical_bytes,
    projected_frame_count,
    projected_metric_log_bytes,
    snapshot_request_inputs,
    sweep_work_dir,
)
from vidaio.scoring_worker.runtime_identity import (
    canonical_release_marker_present,
    complete_payout_backend_versions,
    initialize_canonical_torch_cpu_runtime,
    payout_runtime_attestation,
    require_attested_backend_versions,
    require_canonical_release_runtime,
    runtime_backend_stamp,
    runtime_commitment_digest,
)
from vidaio.services.base import BaseService
from vidaio.services.protocol import ScoreRequest, ScoreResponse

_TRACKS = (TRACK_COMPRESSION, TRACK_UPSCALING)

# Bump whenever scoring semantics change without a ScoringConfig field changing.
# v2: VMAF model-delta pair moved from pristine-reference to miner-input basis.
# v3: required tone/grayscale/chroma checks became deterministic CPU metrics.
# v4: tone/grayscale/chroma gates moved to the canonical miner-input basis too.
SCORING_PIPELINE_VERSION = 4

#: Same name the service logger uses (BaseService: ``get_logger(self.name)``),
#: for the module-level paths (app factory, executor threads) that log without a
#: service instance in hand.
_LOG = logging.getLogger("scoring-worker")


# --- backend composition -------------------------------------------------------------


@dataclass(frozen=True)
class ScoringBackends:
    """Everything injectable the worker scores with (Protocol instances only).

    ``canonicalizer=None`` selects passthrough mode (fake/CI without media tools):
    no normalization runs, the original files stand in for the canonical ones and
    the packet honestly records ``canonicalization_plan_digest=None``.
    ``vmaf_secondary=None`` means no second model run exists; with
    ``require_secondary_vmaf`` enabled the delta gate then fails closed.
    """

    probe: ProbeBackend
    vmaf_primary: VmafBackend
    vmaf_secondary: VmafBackend | None
    pieapp: PieAppBackend
    perceptual: PerceptualCheckBackend
    canonicalizer: CanonicalizeExecutor | None
    #: Stamped into every packet (probed once at composition time, spec §08).
    versions: Mapping[str, str] = field(default_factory=dict)
    #: Complete executable-runtime preimage for the ``versions["runtime"]``
    #: commitment. Real scorer/auditor compositions populate it; injected fake
    #: test backends intentionally use the deterministic development identity.
    runtime_attestation: Mapping[str, Any] | None = None


def real_backends(
    config: ScoringWorkerConfig,
    *,
    scoring_config: ScoringConfig | None = None,
    pieapp_device: str | None = None,
) -> ScoringBackends:
    """Compose the subprocess-backed backends (probed versions included).

    PieAPP is real when the optional ``media`` dependencies/model cache exist.
    ``pieapp_device`` is an auditor-only override; normal scorers use the device
    selected by :class:`ScoringWorkerConfig`.

    The required tone/grayscale/chroma gates always run on CPU. PieAPP uses the
    configured scorer device, except that auditors pass the CPU override. Thus
    the shipped composition is complete on an ordinary validator host.
    """
    config.work_dir.mkdir(parents=True, exist_ok=True)
    primary = FfmpegVmafBackend(
        config.ffmpeg_path,
        model=config.vmaf_model_primary,
        work_dir=config.work_dir,
        timeout=config.subprocess_timeout,
    )
    secondary = FfmpegVmafBackend(
        config.ffmpeg_path,
        model=config.vmaf_model_secondary,
        work_dir=config.work_dir,
        timeout=config.subprocess_timeout,
    )
    versions = detect_tool_versions(
        config.ffmpeg_path,
        config.ffprobe_path,
        vmaf_backend=primary,
        timeout=config.subprocess_timeout,
    )
    secondary.version = primary.version  # same libvmaf library, probed once
    scoring_cfg = scoring_config or ScoringConfig()
    selected_device = pieapp_device or config.pieapp_device
    if selected_device == "cpu" and canonical_release_marker_present():
        # This must precede model construction (and therefore every payout
        # tensor operation): ATen capability and oneMKL CBWR are import-time
        # kernel choices, not settings an auditor can repair after the fact.
        # Native developer/test compositions remain explicitly noncanonical.
        initialize_canonical_torch_cpu_runtime()
    pieapp = PieAppTorchBackend(
        device=selected_device, sample_window=scoring_cfg.pieapp_sample_window
    )
    perceptual = CpuPerceptualCheckBackend(config.perceptual_cpu)
    versions = complete_payout_backend_versions(
        versions,
        pieapp=pieapp,
        perceptual=perceptual,
        device=selected_device,
    )
    runtime_attestation = payout_runtime_attestation(
        config, scoring_cfg, backend_versions=versions
    )
    if runtime_attestation["release"].get("marker_verified") is True:
        require_canonical_release_runtime(runtime_attestation)
    versions["runtime"] = runtime_backend_stamp(runtime_attestation)
    return ScoringBackends(
        probe=FfprobeBackend(config.ffprobe_path, timeout=config.subprocess_timeout),
        vmaf_primary=primary,
        vmaf_secondary=secondary,
        pieapp=pieapp,
        perceptual=perceptual,
        canonicalizer=CanonicalizeExecutor(
            config.ffmpeg_path, timeout=config.subprocess_timeout
        ),
        versions=versions,
        runtime_attestation=runtime_attestation,
    )


# --- scorer identity -----------------------------------------------------------------


def scorer_identity_digest(
    config: ScoringWorkerConfig,
    scoring_config: ScoringConfig,
    *,
    runtime_attestation: Mapping[str, Any] | None = None,
) -> str:
    """sha256 over every configured lever that can change a measured score.

    Built on the packet's own ``scoring_config_digest``
    (:func:`vidaio.scoring.result.config_digest`) plus the worker settings that
    change what is measured and the complete payout-runtime commitment. The
    latter binds the verified release runtime, Linux/amd64 execution policy and
    every native/Python metric backend version. Ports, paths, timeouts and
    concurrency remain excluded because they cannot change a packet.
    """
    attestation = (
        dict(runtime_attestation)
        if runtime_attestation is not None
        else payout_runtime_attestation(config, scoring_config)
    )
    payload = {
        "pipeline_version": SCORING_PIPELINE_VERSION,
        "scoring_config_digest": config_digest(scoring_config),
        "perceptual_checks": config.perceptual_checks,
        "perceptual_cpu": config.perceptual_cpu.model_dump(mode="json"),
        "pieapp_device": config.pieapp_device,
        "vmaf_model_primary": config.vmaf_model_primary,
        "vmaf_model_secondary": config.vmaf_model_secondary,
        "payout_runtime_commitment": runtime_commitment_digest(attestation),
    }
    return sha256_hex(canonical_json_bytes(payload))


def effective_scorer_version(
    config: ScoringWorkerConfig,
    scoring_config: ScoringConfig,
    *,
    runtime_attestation: Mapping[str, Any] | None = None,
) -> str:
    """The version THIS worker stamps: ``<configured name>+<identity digest[:12]>``.

    A caller can never influence it (see :func:`check_scorer_version`), and it
    moves whenever a lever that changes scores moves — so an audit bundle that
    pins a scorer version pins the actual scoring behaviour, not just a label
    somebody typed into a validator config.
    """
    return (
        f"{config.scorer_version}+"
        f"{scorer_identity_digest(config, scoring_config, runtime_attestation=runtime_attestation)[:12]}"
    )


def check_scorer_version(requested: str | None, effective: str) -> None:
    """Reject (409) a request that asks for a scorer this worker is not.

    Contract (see :mod:`vidaio.services.protocol`): ``scorer_version`` is a
    caller ASSERTION about which scorer it expects, never an instruction. Absent
    or empty means "whatever you are"; equal means agreement; anything else is a
    mismatch that must fail loudly, because the validator will later compare the
    packet's scorer version against its audit bundle and a silent substitution
    would only surface as an unexplained audit failure.
    """
    if requested is None or not requested.strip():
        return
    if requested != effective:
        raise ScoreRejected(
            409,
            {
                "error": "scorer_version_mismatch",
                "requested": requested,
                "scorer_version": effective,
            },
        )


# --- typed request rejection ---------------------------------------------------------


def _param_float(params: Mapping[str, Any], key: str) -> float | None:
    value = params.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ScoreRejected(
            422, {"error": "invalid_param", "param": key, "value": str(value)}
        ) from None


def _param_int(params: Mapping[str, Any], key: str) -> int | None:
    value = _param_float(params, key)
    if value is None:
        return None
    if not value.is_integer():
        raise ScoreRejected(
            422, {"error": "invalid_param", "param": key, "value": str(value)}
        )
    return int(value)


# --- the synchronous scoring pipeline (runs in a thread executor) --------------------


def _score_sync(
    request: ScoreRequest,
    config: ScoringWorkerConfig,
    scoring_config: ScoringConfig,
    backends: ScoringBackends,
    scorer_version: str,
    scope: MediaProcessScope | None = None,
    budget: ScratchBudget | None = None,
) -> ItemScore:
    """The whole scored request, start to finish, on ONE worker thread.

    Runs under `scope`: every subprocess it launches is registered there, so the
    caller can genuinely abort this work (and not merely stop waiting for it).

    Runs under a lease on `budget` too: EVERY byte this request puts on the
    shared volume — the snapshots it copies AND the raw y4m/metric logs it
    generates from them — is reserved before it is written and returned only
    after the scratch directory is gone (note the `with` order — the lease is
    the OUTER context, so it outlives the directory it accounts for).

    Everything the request writes lives under ONE directory, including the temp
    dirs libvmaf makes for its JSON logs (`use_media_scratch` installs it for
    this thread). One directory is what makes the accounting checkable, the
    cleanup total, and the startup sweep sufficient.
    """
    if request.track not in _TRACKS:
        raise ScoreRejected(422, {"error": "unsupported_track", "track": request.track})
    config.work_dir.mkdir(parents=True, exist_ok=True)
    scratch = budget if budget is not None else ScratchBudget(_byte_limits(config))
    if scratch.residual_bytes:
        # Round-4 an internal review: pre-charged startup leftovers are retried on every
        # scored request, BEFORE this request's admission — whatever became
        # deletable (an operator fixed the permissions, a mount came back) is
        # released so those bytes admit work again instead of staying reserved
        # forever.
        released = scratch.retry_residual_sweep()
        if released:
            _LOG.info(
                "a retry sweep reclaimed residual scratch a previous run left"
                " behind; the released bytes are available to admissions again",
                extra=log_fields(
                    released_bytes=released,
                    residual_bytes=scratch.residual_bytes,
                ),
            )
    with (
        use_process_scope(scope),
        scratch.lease() as lease,
        TemporaryDirectory(dir=config.work_dir, prefix=WORK_PREFIX) as tmp,
    ):
        tmp_path = Path(tmp)
        metrics_scratch = tmp_path / "metrics"
        metrics_scratch.mkdir(parents=True, exist_ok=True)
        with use_media_scratch(metrics_scratch):
            # 1) Verify-then-snapshot: from here on the pipeline reads ONLY the
            #    private read-only copies, so no post-verification swap of the
            #    caller-named paths can separate the digest from the measured
            #    bytes.
            try:
                snapshot = snapshot_request_inputs(
                    reference=(request.reference_path, request.reference_digest),
                    miner_input=(
                        request.miner_input_path,
                        request.miner_input_digest,
                    ),
                    output=(request.output_path, request.output_digest),
                    dest_dir=tmp_path / "inputs",
                    # Copying a multi-GB submission is real work; the request's
                    # deadline must reach it, not just the ffmpeg stages.
                    cancelled=None if scope is None else (lambda: scope.cancelled),
                    lease=lease,
                )
            except SnapshotCancelled as exc:
                raise MediaWorkCancelled(str(exc)) from exc
            if scope is not None:
                scope.raise_if_cancelled()
            return _score_in_dir(
                request,
                config,
                scoring_config,
                backends,
                tmp_path,
                snapshot,
                scorer_version,
                lease,
            )


def _byte_limits(config: ScoringWorkerConfig) -> ByteLimits:
    """The configured scratch ceilings (see scoring_worker.inputs)."""
    return ByteLimits(
        max_input_bytes=config.max_input_bytes,
        max_request_bytes=config.max_request_bytes,
        max_request_scratch_bytes=config.max_request_scratch_bytes,
        max_scratch_bytes=config.max_scratch_bytes,
    )


def _score_in_dir(
    request: ScoreRequest,
    config: ScoringWorkerConfig,
    scoring_config: ScoringConfig,
    backends: ScoringBackends,
    tmp: Path,
    snapshot: InputSnapshot,
    scorer_version: str,
    lease: ScratchLease,
) -> ItemScore:
    # Every path below is a private snapshot path — never a caller-named one.
    reference_path = snapshot.reference.path
    output_path = snapshot.output.path
    miner_input_path = snapshot.miner_input.path

    # 2) Probe the originals FIRST. They are the gate facts, and they are also
    #    the only way to know how big this request is about to become: the
    #    canonical y4m size is a function of the probed geometry, so probing
    #    before expanding is what turns "we hope it fits" into a reservation.
    ref_info = backends.probe.probe(reference_path)
    cand_info = backends.probe.probe(output_path)
    input_info = backends.probe.probe(miner_input_path)

    # 3) Canonicalize both sides (identical recipe) — or pass through in fake
    #    mode, which generates nothing and therefore reserves nothing.
    plan_digest_value: str | None = None
    log_run_cap: int | None = None  # per-run metric-log bound; None = unreserved
    if backends.canonicalizer is not None:
        canon_ref = str(tmp / "reference.y4m")
        canon_cand = str(tmp / "candidate.y4m")
        canon_input = str(tmp / "miner-input-for-delta.y4m")
        # VMAF requires equal geometry. Match the calibration probe exactly:
        # Downscale inputs are Lanczos-rescaled to candidate geometry before the
        # two anti-gaming model runs.
        delta_input_info = input_info.model_copy(
            update={"width": cand_info.width, "height": cand_info.height}
        )
        ref_cap, cand_cap, input_cap, log_run_cap = _reserve_expansion(
            lease=lease,
            backends=backends,
            reference=ref_info,
            candidate=cand_info,
            model_delta_input=delta_input_info,
        )
        ref_plan = build_canonicalization_plan(reference_path, canon_ref)
        cand_plan = build_canonicalization_plan(output_path, canon_cand)
        input_plan = build_canonicalization_plan(
            miner_input_path,
            canon_input,
            scale_width=(
                cand_info.width
                if (input_info.width, input_info.height)
                != (cand_info.width, cand_info.height)
                else None
            ),
            scale_height=(
                cand_info.height
                if (input_info.width, input_info.height)
                != (cand_info.width, cand_info.height)
                else None
            ),
        )
        _canonicalize(
            backends.canonicalizer,
            ref_plan,
            field="reference",
            output_path=canon_ref,
            max_output_bytes=ref_cap,
            timeout=config.subprocess_timeout,
        )
        _canonicalize(
            backends.canonicalizer,
            cand_plan,
            field="output",
            output_path=canon_cand,
            max_output_bytes=cand_cap,
            timeout=config.subprocess_timeout,
        )
        _canonicalize(
            backends.canonicalizer,
            input_plan,
            field="miner_input",
            output_path=canon_input,
            max_output_bytes=input_cap,
            timeout=config.subprocess_timeout,
        )
        # The projection is an upper bound; hand the surplus back so a
        # concurrent request is shed on what is really on the volume, not on
        # what this one might have needed.
        written = (
            _file_size(canon_ref) + _file_size(canon_cand) + _file_size(canon_input)
        )
        lease.refund_generated(max(0, ref_cap + cand_cap + input_cap - written))
        # The path-independent recipe digest (identical for both sides by
        # construction) — recorded so packets stay byte-stable across hosts.
        plan_digest_value = plan_template_digest(cand_plan, output_path, canon_cand)
        delta_input_plan_digest = plan_template_digest(
            input_plan, miner_input_path, canon_input
        )
    else:
        canon_ref = reference_path
        canon_cand = output_path
        canon_input = miner_input_path
        delta_input_plan_digest = None

    # 4) Probe the canonical files (stream consistency).
    canon_ref_info = backends.probe.probe(canon_ref)
    canon_cand_info = backends.probe.probe(canon_cand)
    canon_input_info = backends.probe.probe(canon_input)
    extra_violations = validate_stream(canon_ref_info, canon_cand_info)
    # libvmaf (and PieAPP) require same-geometry inputs; with mismatched dims the
    # metrics are unmeasurable and the gates fail closed on the missing values.
    dims_match = not any(
        v.code == ReasonCode.STREAM_DIMENSIONS_MISMATCH for v in extra_violations
    )
    delta_dims_match = (
        canon_input_info.width,
        canon_input_info.height,
    ) == (canon_cand_info.width, canon_cand_info.height)

    # 5) Metrics per track.
    params = request.params
    pieapp_value: float | None = None
    start_frame: int | None = None
    if request.track == TRACK_UPSCALING and dims_match:
        window = usable_frames(
            canon_ref_info.frame_count, scoring_config.pieapp_sample_window
        )
        if window >= 1:
            # Derived from the HELD-OUT reference digest (the miner never sees the
            # upscaling reference), so it is verifier-recomputable but not
            # miner-predictable; zero scoring-time randomness (spec §08).
            start_frame = derive_pieapp_start_frame(
                snapshot.reference.digest, request.challenge_id, window
            )
            pieapp_value = backends.pieapp.compute(
                canon_ref, canon_cand, start_frame=start_frame
            )
    if request.track == TRACK_UPSCALING and pieapp_value is None:
        extra_violations = extra_violations + [
            ValidityViolation(
                code=ReasonCode.METRIC_MISSING,
                detail="PieAPP was not measured (unusable stream geometry)",
            )
        ]

    vmaf_primary: float | None = None
    vmaf_delta_primary: float | None = None
    vmaf_delta_secondary: float | None = None
    if dims_match:
        # Round-4 an internal review: each VMAF run's JSON log is hard-bounded to the
        # per-run share of the metric-log reservation taken above — the same
        # watchdog that bounds a canonicalization kills the run past it. Blown
        # here means the reservation was genuinely too small for this request
        # (a wider model, an unprojected frame surplus): the deterministic 413
        # every retry would also get, never an unaccounted write.
        try:
            with use_metric_log_limit(log_run_cap):
                vmaf_primary = backends.vmaf_primary.compute(canon_ref, canon_cand)
                if backends.vmaf_secondary is not None and (
                    delta_dims_match or backends.canonicalizer is None
                ):
                    # The gate asks what the MINER added, not what the challenge
                    # DAG added. Both anti-gaming models therefore use miner input.
                    vmaf_delta_primary = backends.vmaf_primary.compute(
                        canon_input, canon_cand
                    )
                    vmaf_delta_secondary = backends.vmaf_secondary.compute(
                        canon_input, canon_cand
                    )
        except MetricLogTooLarge as exc:
            raise ScoreRejected(
                413,
                {
                    "error": "metric_log_too_large",
                    "reserved_bytes": exc.limit_bytes,
                    "observed_bytes": exc.observed_bytes,
                    "detail": (
                        "a metric run's JSON log exceeded the scratch reserved "
                        "for it from the projected frame count"
                    ),
                },
            ) from exc

    # 6) Gates, then composition (gates-first zeroing lives in compose_item_score).
    #    perceptual_checks="skip" swaps the three backend-driven perceptual gates
    #    for auditable GateSkip records (config.py); "required" runs them for real
    #    and lets an unconfigured backend raise its honest 501.
    gate_pipeline = (
        default_pipeline(backends.perceptual)
        if config.perceptual_checks == "required"
        else pipeline_without_perceptual_checks(
            reason='check disabled by scoring_worker.perceptual_checks="skip"'
        )
    )
    ctx = GateContext(
        track=request.track,
        config=scoring_config,
        reference_info=ref_info,
        candidate_info=cand_info,
        reference_path=canon_ref,
        candidate_path=canon_cand,
        input_info=input_info,
        input_path=canon_input,
        vmaf_primary=vmaf_primary,
        vmaf_secondary=vmaf_delta_secondary,
        vmaf_delta_primary=vmaf_delta_primary,
        vmaf_delta_secondary=vmaf_delta_secondary,
        upscale_factor=_param_int(params, "upscale_factor"),
        extra_violations=extra_violations,
    )
    passed, violations = gate_pipeline.run(ctx)

    breakdown = None
    metrics: dict[str, float | int | str | None] = {
        "vmaf": vmaf_primary,
        "vmaf_secondary": vmaf_delta_secondary,
        "vmaf_model_delta_primary": vmaf_delta_primary,
        "vmaf_model_delta_basis": "miner_input",
        "vmaf_model_delta_input_plan_digest": delta_input_plan_digest,
        "perceptual_gate_basis": "miner_input",
        "perceptual_gate_input_plan_digest": delta_input_plan_digest,
        "vmaf_model_delta": (
            abs(vmaf_delta_primary - vmaf_delta_secondary)
            if vmaf_delta_primary is not None and vmaf_delta_secondary is not None
            else None
        ),
        "candidate_bytes": cand_info.byte_size,
        "reference_bytes": input_info.byte_size,
        "perceptual_config_digest": config.perceptual_cpu.digest(),
    }
    for gate_name, result in sorted(ctx.perceptual_results.items()):
        metrics[f"{gate_name}_measure"] = result.measure
        metrics[f"{gate_name}_passed"] = "true" if result.passed else "false"
    if (
        request.track == TRACK_COMPRESSION
        and vmaf_primary is not None
        and input_info.byte_size > 0
    ):
        breakdown = score_compression(
            candidate_bytes=cand_info.byte_size,
            reference_bytes=input_info.byte_size,
            vmaf=vmaf_primary,
            config=scoring_config,
            vmaf_threshold=_param_float(params, "vmaf_threshold"),
        )
        metrics["compression_rate"] = breakdown.compression_rate
        metrics["final_score"] = breakdown.final
    elif request.track == TRACK_UPSCALING and pieapp_value is not None:
        content_length = _param_float(params, "content_length")
        if content_length is None:
            content_length = canon_ref_info.duration
        breakdown = score_upscaling(
            pieapp=pieapp_value, content_length=content_length, config=scoring_config
        )
        metrics["pieapp"] = pieapp_value
        metrics["content_length"] = content_length
        metrics["final_score"] = breakdown.final

    return compose_item_score(
        item_id=request.item_id,
        challenge_id=request.challenge_id,
        track=request.track,
        gate_passed=passed,
        violations=violations,
        skips=ctx.skips,
        breakdown=breakdown,
        config=scoring_config,
        miner_hotkey=request.miner_hotkey,
        # The digest of the bytes that were MEASURED (the private copy), which
        # by construction equals the digest the request claimed. Never re-read
        # from the caller-named path.
        content_digest=snapshot.output.digest,
        metrics=metrics,
        backend_versions=dict(backends.versions),
        canonicalization_plan_digest=plan_digest_value,
        pieapp_start_frame=start_frame,
        # THIS worker's version — never the caller's (see check_scorer_version).
        scorer_version=scorer_version,
    )


# --- the expansion budget ------------------------------------------------------------
#
# Everything above the input caps happens here. The input caps bound the ENCODED
# artifacts a caller names; scoring measures the DECODED ones, and decoding is an
# amplifier a miner controls. A 30 MB ten-minute 4K clip is inside every input
# cap and decodes to ~450 GB of y4m — twice, once per side. So the expansion is
# projected from the probe, reserved against the same worker-wide budget the
# snapshots use, and then bounded again while it actually runs.


def _reserve_expansion(
    *,
    lease: ScratchLease,
    backends: ScoringBackends,
    reference: MediaInfo,
    candidate: MediaInfo,
    model_delta_input: MediaInfo,
) -> tuple[int, int, int, int]:
    """Reserve the scratch canonicalization + metric logs will need.

    Returns ``(ref_cap, cand_cap, input_cap, log_run_cap)``: hard output
    bounds for all three canonicalizations plus the per-run metric log bound.
    held to (an internal review — the log reservation is enforced, not just
    taken; ``runs x log_run_cap`` is exactly the log share reserved below).

    Refuses BEFORE ffmpeg starts, with the typed error that fits (see
    :meth:`~vidaio.scoring_worker.inputs.ScratchLease.reserve_generated`): 413
    when the projection cannot fit in one request's allowance no matter how
    empty the worker is, 503 + Retry-After when it merely does not fit right
    now. Either way not one byte of y4m has been written.

    Reserved as ONE claim covering both sides plus the logs, because a partial
    reservation is not a reservation: taking the reference's share and then
    failing on the candidate's would let a request that can never complete hold
    the volume against requests that could.
    """
    ref_cap = projected_canonical_bytes("reference", reference)
    cand_cap = projected_canonical_bytes("output", candidate)
    input_cap = projected_canonical_bytes("miner_input", model_delta_input)
    # quality vs pristine (1), plus both model-delta runs vs miner input (2).
    vmaf_runs = 1 if backends.vmaf_secondary is None else 3
    frames = max(
        projected_frame_count(reference),
        projected_frame_count(candidate),
        projected_frame_count(model_delta_input),
    )
    log_run_cap = projected_metric_log_bytes(frames=frames, runs=1)
    log_cap = projected_metric_log_bytes(frames=frames, runs=vmaf_runs)
    lease.reserve_generated(
        kind="canonicalization",
        nbytes=ref_cap + cand_cap + input_cap + log_cap,
        reference_y4m_bytes=ref_cap,
        candidate_y4m_bytes=cand_cap,
        model_delta_input_y4m_bytes=input_cap,
        metric_log_bytes=log_cap,
    )
    return ref_cap, cand_cap, input_cap, log_run_cap


def _canonicalize(
    canonicalizer: CanonicalizeExecutor,
    plan: list[str],
    *,
    field: str,
    output_path: str,
    max_output_bytes: int,
    timeout: float,
) -> None:
    """Run one canonicalization plan under a HARD output bound.

    The projection that produced `max_output_bytes` is exact for a well-formed
    source, but it is still a prediction about a file the miner produced. If the
    output passes the cap, the process group is killed, the partial y4m is
    removed (so the volume owes nothing for a request that will not finish) and
    the request is refused 413 — the same verdict the projection would have
    given had it seen the truth.
    """
    try:
        canonicalizer.run(
            plan,
            timeout=timeout,
            output_path=output_path,
            max_output_bytes=max_output_bytes,
        )
    except CanonicalizationTooLarge as exc:
        _remove_partial(output_path)
        raise ScoreRejected(
            413,
            {
                "error": "canonicalized_output_too_large",
                "field": field,
                "projected_bytes": exc.limit_bytes,
                "observed_bytes": exc.observed_bytes,
                "detail": (
                    "the canonicalized raw video exceeded the scratch reserved "
                    "for it from the probed stream geometry"
                ),
            },
        ) from exc
    except BaseException:
        # Timeout, cancellation, a tool failure: whatever partial output exists
        # is unusable and must not sit on the volume until the dir is swept.
        _remove_partial(output_path)
        raise


def _remove_partial(path: str) -> None:
    with contextlib.suppress(OSError):
        Path(path).unlink()


def _file_size(path: str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


# --- metrics -------------------------------------------------------------------------


class WorkerMetrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.scorings = Counter(
            "scoring_worker_scorings_total",
            "Scoring requests by track and outcome",
            ["track", "outcome"],
            registry=registry,
        )
        self.gate_failures = Counter(
            "scoring_worker_gate_failures_total",
            "Gate violations by reason code",
            ["reason"],
            registry=registry,
        )
        self.duration = Histogram(
            "scoring_worker_scoring_duration_seconds",
            "Wall-clock duration of one /score request",
            buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
            registry=registry,
        )
        self.vmaf = Histogram(
            "scoring_worker_vmaf_score",
            "Primary-model VMAF of scored items",
            buckets=tuple(float(b) for b in range(0, 101, 10)),
            registry=registry,
        )


# --- health checks -------------------------------------------------------------------


def build_health_checks(
    config: ScoringWorkerConfig, backends: ScoringBackends
) -> dict[str, Callable[[], bool]]:
    """The worker's health checks — shared by /healthz and the HealthServer.

    Both are STATELESS: they read config and the filesystem only, never a
    connection or object owned by another thread. That matters because the
    HealthServer answers /health on its own thread while the event loop and the
    scoring threads are busy — a check that touched their state would report on
    the wrong thread's world (or corrupt it).
    """

    def work_dir_writable() -> bool:
        config.work_dir.mkdir(parents=True, exist_ok=True)
        # Unique per call: two health checks racing on one fixed probe path
        # would make each other spuriously "unhealthy".
        probe_path = config.work_dir / f".healthz-probe-{uuid.uuid4().hex}"
        try:
            probe_path.write_bytes(b"ok")
        finally:
            with contextlib.suppress(OSError):
                probe_path.unlink()
        return True

    def media_tools_present() -> bool:
        if backends.canonicalizer is None:  # fake mode needs no media tools
            return True
        return (
            which(config.ffmpeg_path) is not None
            and which(config.ffprobe_path) is not None
        )

    return {
        "work_dir_writable": work_dir_writable,
        "media_tools_present": media_tools_present,
    }


# --- app factory ---------------------------------------------------------------------


def create_app(
    config: ScoringWorkerConfig,
    backends: ScoringBackends,
    *,
    scoring_config: ScoringConfig | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    scoring_cfg = scoring_config if scoring_config is not None else ScoringConfig()
    metrics = WorkerMetrics(registry if registry is not None else CollectorRegistry())
    config.work_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="vidaio-scoring-worker")
    semaphore = asyncio.Semaphore(config.max_concurrent)
    # ONE budget for the whole app: the guard has to be collective, or N
    # requests that each fit would still fill the volume between them.
    scratch_budget = ScratchBudget(_byte_limits(config))
    # Round-4 an internal review: whatever the startup sweep could NOT delete is still on
    # the volume, so a budget that starts at zero would overcommit by exactly
    # those bytes. Measured before any request runs (everything under our
    # prefixes at this instant is a leftover) and pre-charged as a reservation
    # admission cannot displace; every scored request retries the deletion and
    # releases what succeeds (see _score_sync).
    residual, unmeasurable = measure_scratch_entries(config.work_dir)
    # Round-5 an internal review: an entry whose subtree could not be fully traversed has
    # an UNBOUNDED true size (a 0700 dir owned by another uid measures as zero
    # while hiding gigabytes) — charging its visible sum would overcommit the
    # volume. It cannot be charged, cannot be swept by us, and therefore the
    # WORKER refuses to start (ScoringWorker reads this and fail_fatal's);
    # bare create_app() users get the ERROR log.
    app.state.unmeasurable_scratch = list(unmeasurable)
    if unmeasurable:
        _LOG.error(
            "leftover scratch cannot be measured (permissions/IO) — its true"
            " size is unbounded and it CANNOT be budgeted; reclaim the work"
            " dir manually",
            extra=log_fields(unmeasurable_entries=unmeasurable),
        )
    if residual:
        charged = scratch_budget.charge_residual(residual)
        if charged:
            _LOG.warning(
                "undeletable scratch from a previous run remains in the work dir"
                " and is PRE-CHARGED against the scratch budget: admission is"
                " reduced by exactly these bytes until a retry sweep reclaims"
                " them",
                extra=log_fields(
                    residual_bytes=charged,
                    residual_entries=len(residual),
                    max_scratch_bytes=scratch_budget.limits.max_scratch_bytes,
                ),
            )
    app.state.scratch_budget = scratch_budget
    health_checks = build_health_checks(config, backends)
    runtime_attestation = (
        dict(backends.runtime_attestation)
        if backends.runtime_attestation is not None
        else payout_runtime_attestation(config, scoring_cfg)
    )
    scorer_version = effective_scorer_version(
        config, scoring_cfg, runtime_attestation=runtime_attestation
    )
    app.state.scorer_version = scorer_version
    app.state.runtime_attestation = runtime_attestation

    async def _acquire_slot(track: str) -> None:
        """Wait for a concurrency slot, but only for a bounded time.

        A saturated worker must SHED load (503 + Retry-After), not queue callers
        indefinitely behind work whose own timeout has not even started yet.
        """
        try:
            await asyncio.wait_for(
                semaphore.acquire(), timeout=config.queue_wait_timeout_seconds
            )
        except TimeoutError:
            metrics.scorings.labels(track=track, outcome="queue_timeout").inc()
            raise HTTPException(
                503,
                detail={
                    "error": "queue_saturated",
                    "detail": (
                        "no scoring slot within "
                        f"{config.queue_wait_timeout_seconds}s "
                        f"(max_concurrent={config.max_concurrent})"
                    ),
                },
                headers={
                    "Retry-After": str(max(1, int(config.queue_wait_timeout_seconds)))
                },
            ) from None

    async def _scored(request: ScoreRequest) -> ItemScore:
        """Run one scoring on a worker thread with a REAL, enforceable deadline.

        Two things make the deadline real. (a) The media work runs as child
        process groups registered on `scope`; cancelling the scope SIGKILLs them,
        so the thread unwinds instead of continuing to burn CPU behind a caller
        that already got its 504. (b) The semaphore slot is released by the
        future's done-callback — i.e. when the work has genuinely finished or
        died — not when this coroutine stops awaiting it. Releasing on the await
        is precisely how a worker ends up running more ffmpeg than
        `max_concurrent` allows.
        """
        loop = asyncio.get_running_loop()
        scope = MediaProcessScope()
        try:
            future = loop.run_in_executor(
                None,
                partial(
                    _score_sync,
                    request,
                    config,
                    scoring_cfg,
                    backends,
                    scorer_version,
                    scope,
                    scratch_budget,
                ),
            )
        except BaseException:  # never hold a slot for work that never started
            semaphore.release()
            raise

        def _release(done: "asyncio.Future[ItemScore]") -> None:
            semaphore.release()
            if not done.cancelled():
                done.exception()  # consume: an abandoned failure must not warn

        future.add_done_callback(_release)
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=config.request_timeout
            )
        except (TimeoutError, asyncio.CancelledError):
            # Kill the work itself. Non-blocking: `scope.cancel()` signals the
            # process groups and returns; the slot frees when the thread does.
            scope.cancel()
            raise

    @app.post("/score", response_model=ScoreResponse)
    async def score(request: ScoreRequest) -> ScoreResponse:
        track = request.track if request.track in _TRACKS else "unknown"
        started = time.perf_counter()
        try:
            # Admission BEFORE any work: this worker answers only for itself.
            check_scorer_version(request.scorer_version, scorer_version)
            await _acquire_slot(track)
            item = await _scored(request)
        except ScoreRejected as exc:
            # 503 = shed (a budget/queue that frees itself), 500 = our fault,
            # everything else = the request was refused on its merits.
            outcome = {503: "shed", 500: "internal_error"}.get(
                exc.status_code, "rejected"
            )
            metrics.scorings.labels(track=track, outcome=outcome).inc()
            raise HTTPException(
                exc.status_code, detail=exc.payload, headers=exc.headers
            ) from exc
        except NotConfiguredError as exc:
            metrics.scorings.labels(track=track, outcome="not_configured").inc()
            raise HTTPException(
                501, detail={"error": "backend_not_configured", "detail": str(exc)}
            ) from exc
        except (TimeoutError, MediaToolTimeout, MediaWorkCancelled) as exc:
            metrics.scorings.labels(track=track, outcome="timeout").inc()
            raise HTTPException(
                504, detail={"error": "scoring_timeout", "detail": str(exc)}
            ) from exc
        except MediaToolError as exc:
            metrics.scorings.labels(track=track, outcome="tool_error").inc()
            raise HTTPException(
                502, detail={"error": "media_tool_failed", "detail": str(exc)}
            ) from exc
        finally:
            metrics.duration.observe(time.perf_counter() - started)

        outcome = "scored" if item.gate_passed else "gate_failed"
        metrics.scorings.labels(track=track, outcome=outcome).inc()
        for violation in item.violations:
            metrics.gate_failures.labels(reason=violation.code.value).inc()
        vmaf_value = item.metrics.get("vmaf")
        if isinstance(vmaf_value, (int, float)):
            metrics.vmaf.observe(float(vmaf_value))

        packet = item.to_json()
        return ScoreResponse(
            item_score_json=packet,
            packet_digest=hashlib.sha256(packet.encode("utf-8")).hexdigest(),
        )

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        checks: dict[str, bool] = {}
        ok = True
        for name, check in health_checks.items():
            try:
                healthy = bool(check())
            except Exception:  # noqa: BLE001 - a failing check is an unhealthy check
                healthy = False
            checks[name] = healthy
            ok = ok and healthy
        return JSONResponse(
            {
                "service": ScoringWorker.name,
                "status": "ok" if ok else "degraded",
                "checks": checks,
                # Published so a caller can align its ScoreRequest.scorer_version
                # (or its audit bundle) with the scorer that will actually run,
                # instead of discovering the mismatch as a 409 per request.
                "scorer_version": scorer_version,
                # Full preimage for the runtime stamp in every measured packet.
                # It contains only public release/tool identity, never secrets.
                "runtime_commitment": {
                    "digest": runtime_commitment_digest(runtime_attestation),
                    "attestation": runtime_attestation,
                },
            },
            status_code=200 if ok else 503,
        )

    return app


# --- service -------------------------------------------------------------------------


class _EmbeddedServer(uvicorn.Server):
    """Uvicorn server that leaves signal handling to BaseService."""

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        yield


class ApiServerFailed(RuntimeError):
    """The worker's HTTP server stopped serving (bind failure, protocol crash)."""


class ScoringWorker(BaseService):
    """The long-running scoring worker process (BaseService lifecycle).

    Config sections: ``scoring_worker`` (this service) + ``scoring`` (the shared
    scoring levers). ``backends`` may be injected for tests; otherwise the
    ``backend: real`` composition shells out to ffmpeg/ffprobe. ``backend: fake``
    without injected backends is refused — the worker never invents a fake on its
    own (a fake must be an explicit test-time composition).
    """

    name = "scoring-worker"

    def __init__(
        self,
        raw_config: dict[str, Any],
        *,
        backends: ScoringBackends | None = None,
    ) -> None:
        config = section(raw_config, "scoring_worker", ScoringWorkerConfig)
        super().__init__(raw_config, metrics_port=config.metrics_port)
        self.config = config
        self.scoring_config = section(raw_config, "scoring", ScoringConfig)
        #: Flipped when the HTTP server task ends without a stop request — a live
        #: process with no API must report unhealthy so a supervisor replaces it.
        self._api_failed = False
        config.work_dir.mkdir(parents=True, exist_ok=True)
        swept = sweep_work_dir(config.work_dir)
        if swept:
            self.log.info(
                "swept stale scoring scratch dirs", extra=log_fields(removed=swept)
            )
        if backends is None:
            if config.backend != "real":
                raise ValueError(
                    "backend 'fake' requires explicitly injected backends — the "
                    "worker never composes fake metric backends on its own"
                )
            backends = real_backends(config, scoring_config=self.scoring_config)
        self.backends = backends
        supplied_backend_attestation = backends.runtime_attestation
        backend_attestation = (
            dict(backends.runtime_attestation)
            if supplied_backend_attestation is not None
            else payout_runtime_attestation(config, self.scoring_config)
        )
        # An earning scorer is not allowed to mint packets from a developer
        # checkout, a native process, or an otherwise unqualified runtime.  The
        # report-mode seam remains available for unit tests and local tooling;
        # the real-chain composition is fail-closed at the constructor itself so
        # an entrypoint cannot accidentally bypass a separate preflight.
        from vidaio.chain.factory import ChainConfig

        if section(raw_config, "chain", ChainConfig).mode == "bittensor":
            # Never let an injected backend merely *claim* a release identity.
            # Re-probe this process and require an exact commitment match.
            runtime_attestation = payout_runtime_attestation(
                config, self.scoring_config
            )
            require_canonical_release_runtime(runtime_attestation)
            if supplied_backend_attestation is None:
                raise RuntimeError(
                    "Bittensor scoring backends must carry the canonical runtime "
                    "attestation produced by real_backends"
                )
            if getattr(backends.pieapp, "device", None) != "cpu":
                raise RuntimeError(
                    "Bittensor payout PieAPP backend must execute on CPU"
                )
            require_attested_backend_versions(runtime_attestation, backends.versions)
            if runtime_commitment_digest(backend_attestation) != runtime_commitment_digest(
                runtime_attestation
            ):
                raise RuntimeError(
                    "scoring backend runtime attestation differs from the "
                    "independently probed canonical payout runtime"
                )
        else:
            runtime_attestation = backend_attestation
        self.scorer_version = effective_scorer_version(
            config,
            self.scoring_config,
            runtime_attestation=runtime_attestation,
        )
        self.app = create_app(
            config,
            backends,
            scoring_config=self.scoring_config,
            registry=self.health.registry,
        )
        # Round-4 an internal review: leftovers bigger than the WHOLE budget mean not one
        # request can ever be admitted honestly — that is an operator problem
        # (reclaim the work dir), not something to paper over by overcommitting
        # the volume. Fail fatally so a supervisor restarts us into a fixed
        # world instead of us serving 503s forever.
        residual_bytes = self.app.state.scratch_budget.residual_bytes
        if residual_bytes > config.max_scratch_bytes:
            self.fail_fatal(
                "undeletable scratch left over from a previous run exceeds the"
                f" entire scratch budget ({residual_bytes} residual bytes >"
                f" max_scratch_bytes={config.max_scratch_bytes});"
                f" an operator must reclaim {config.work_dir} by hand"
            )
        # Round-5 an internal review: an UNMEASURABLE leftover is worse than a big one —
        # its true size is unbounded, so no budget can honestly admit anything
        # next to it. Refuse to serve until an operator reclaims it.
        unmeasurable = list(getattr(self.app.state, "unmeasurable_scratch", ()))
        if unmeasurable:
            self.fail_fatal(
                "leftover scratch cannot be measured (permissions/IO) and its"
                " true size is unbounded — no admission next to it is honest;"
                f" an operator must reclaim by hand: {', '.join(unmeasurable)}"
            )
        for check_name, check in build_health_checks(config, backends).items():
            self.health.register_check(check_name, check)
        self.health.register_check("http_api_serving", lambda: not self._api_failed)

    async def _serve_api(self, server: uvicorn.Server) -> None:
        """Run uvicorn, converting its startup ``sys.exit`` into a normal error.

        uvicorn signals a bind failure with ``sys.exit(3)``. Raised inside a
        task, that SystemExit travels asyncio's BaseException path and tears the
        event loop down on the spot — before we could mark ourselves unhealthy
        or shut anything down cleanly. Turning it into an ordinary exception
        keeps the failure ours to handle (and still ends the process non-zero,
        because :meth:`run` re-raises it).
        """
        try:
            await server.serve()
        except SystemExit as exc:
            raise ApiServerFailed(
                f"http server exited during startup (code {exc.code})"
            ) from exc

    async def run(self) -> None:
        server = _EmbeddedServer(
            uvicorn.Config(
                self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="warning",
                access_log=False,
            )
        )
        serve_task = asyncio.create_task(
            self._serve_api(server), name="scoring-worker-http"
        )
        stop_task = asyncio.create_task(
            self.stopping.wait(), name="scoring-worker-stop"
        )
        try:
            await asyncio.wait(
                {serve_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if serve_task.done() and not self.stopping.is_set():
                # The API died on its own (bind failure, protocol crash). Staying
                # alive-and-"healthy" with no /score is the worst outcome: the
                # supervisor would never replace us and callers would just time
                # out. Fail closed AND fatally — a plain stop exits 0, which the
                # supervisor reads as "deliberate shutdown, do not restart"
                #.
                self._api_failed = True
                self.fail_fatal(
                    "http api exited unexpectedly; the worker has no /score"
                    f" (port={self.config.port} error={_task_error(serve_task)})"
                )
        finally:
            server.should_exit = True
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task
            try:
                await serve_task
            except asyncio.CancelledError:
                raise
            except BaseException:
                # Re-raised deliberately: this worker's API failure has its own
                # typed error (ApiServerFailed) and callers assert on it. It also
                # satisfies the exit-code contract on its own — an exception out
                # of run() is a NON-ZERO exit, which is all the supervisor needs
                # to restart us. `fail_fatal` above has already flipped health and
                # logged the reason CRITICAL; this just keeps the typed cause.
                self._api_failed = True
                raise


def _task_error(task: "asyncio.Task[Any]") -> str:
    """Readable reason a finished task ended (SystemExit included)."""
    if task.cancelled():
        return "cancelled"
    exc = task.exception()
    return f"{type(exc).__name__}: {exc}" if exc is not None else "clean exit"
