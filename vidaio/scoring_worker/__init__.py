"""Scoring worker service: HTTP /score over the pure scoring engine + real backends.

The worker is the only place where the pure scoring module meets subprocesses
(:mod:`vidaio.scoring.backends_real`) and the wire protocol
(:mod:`vidaio.services.protocol`). Backends are injected — deterministic fakes for
tests/CI, ffmpeg/ffprobe plus pinned CPU media metrics for real runs — and missing
dependencies/model weights surface as typed 501 errors, never substituted scores.
"""

from vidaio.scoring_worker.config import ScoringWorkerConfig
from vidaio.scoring_worker.inputs import (
    ByteLimits,
    InputSnapshot,
    ScoreRejected,
    ScratchBudget,
    ScratchLease,
    SnapshotCancelled,
    VerifiedInput,
    measure_scratch_entries,
    projected_canonical_bytes,
    projected_frame_count,
    projected_metric_log_bytes,
    snapshot_request_inputs,
    sweep_work_dir,
    y4m_frame_bytes,
)
from vidaio.scoring_worker.service import (
    ScoringBackends,
    ScoringWorker,
    WorkerMetrics,
    build_health_checks,
    check_scorer_version,
    create_app,
    effective_scorer_version,
    real_backends,
    scorer_identity_digest,
)
from vidaio.scoring_worker.runtime_identity import (
    canonical_release_marker_present,
    canonical_runtime_problems,
    initialize_canonical_torch_cpu_runtime,
    payout_runtime_attestation,
    require_canonical_release_runtime,
    runtime_backend_stamp,
    runtime_commitment_digest,
)

__all__ = [
    "ScoringWorkerConfig",
    "ScoringBackends",
    "ScoringWorker",
    "ByteLimits",
    "ScoreRejected",
    "ScratchBudget",
    "ScratchLease",
    "SnapshotCancelled",
    "WorkerMetrics",
    "InputSnapshot",
    "VerifiedInput",
    "build_health_checks",
    "check_scorer_version",
    "canonical_release_marker_present",
    "canonical_runtime_problems",
    "create_app",
    "effective_scorer_version",
    "initialize_canonical_torch_cpu_runtime",
    "measure_scratch_entries",
    "payout_runtime_attestation",
    "projected_canonical_bytes",
    "projected_frame_count",
    "projected_metric_log_bytes",
    "real_backends",
    "require_canonical_release_runtime",
    "runtime_backend_stamp",
    "runtime_commitment_digest",
    "scorer_identity_digest",
    "snapshot_request_inputs",
    "sweep_work_dir",
    "y4m_frame_bytes",
]
