"""Challenge generation: procedural degradation DAG (private seeds), commit-reveal,
content pool + provenance log, per-cycle scheduler (spec: design spec §18, plan §7).

Pure logic + SQLite persistence. ffmpeg is never executed here — operators and the
ingest contract emit command PLANS for an external executor.
"""

from pathlib import Path

from vidaio.challenge.commitment import (
    CHALLENGE_ANCHOR_DOMAIN,
    ChallengeAnchor,
    ChallengeCommitment,
    RevealBeforeResolutionError,
    RevealBeforeRetireError,
    RevealedCommitment,
    challenge_anchor_payload,
    deep_reveal_verifier,
    record_commitment_anchor,
    record_commitment,
    reveal_commitment,
    verify_reveal,
    verify_reveal_deep,
)
from vidaio.challenge.config import (
    LAUNCH_MAX_ELIGIBILITY_SCAN_ASSETS,
    LAUNCH_UPSCALING_MIN_CLIP_SECONDS,
    MAX_CLIP_DURATION_OVERSHOOT_SECONDS,
    ChallengeConfig,
)
from vidaio.challenge.dag import (
    DAG_VERSION,
    LAUNCH_UPSCALE_FACTORS,
    OPERATOR_REGISTRY,
    TRACK_RULES,
    UPSCALE_FACTORS,
    DegradationDag,
    DegradationOp,
    build_dag,
    canonical_json_dumps,
    dag_rng_from_seed,
    seed_to_bytes,
    to_ffmpeg_plan,
)
from vidaio.challenge.pool import (
    Asset,
    FingerprintIndex,
    NearDuplicateError,
    NoFreshAssetError,
    StaticFingerprintIndex,
    UnresolvedChallengeError,
    add_asset,
    append_provenance,
    assign_split,
    check_near_duplicate,
    checkout_asset,
    get_asset,
    provenance_log,
    release_asset,
    retire_asset,
    source_key,
)
from vidaio.challenge.scheduler import (
    MIN_SEED_BITS,
    Challenge,
    ChallengeIntegrityError,
    DispatchPayload,
    IngestResult,
    PayloadLeakError,
    WeakSeedError,
    confirm_ingest_step,
    make_challenge,
    record_challenge,
    register_asset,
    resolve_challenge,
)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

__all__ = [
    "MIGRATIONS_DIR",
    # config
    "ChallengeConfig",
    "LAUNCH_MAX_ELIGIBILITY_SCAN_ASSETS",
    "LAUNCH_UPSCALING_MIN_CLIP_SECONDS",
    "MAX_CLIP_DURATION_OVERSHOOT_SECONDS",
    # dag
    "DAG_VERSION",
    "LAUNCH_UPSCALE_FACTORS",
    "OPERATOR_REGISTRY",
    "TRACK_RULES",
    "UPSCALE_FACTORS",
    "DegradationDag",
    "DegradationOp",
    "build_dag",
    "canonical_json_dumps",
    "dag_rng_from_seed",
    "seed_to_bytes",
    "to_ffmpeg_plan",
    # commitment
    "CHALLENGE_ANCHOR_DOMAIN",
    "ChallengeAnchor",
    "ChallengeCommitment",
    "RevealedCommitment",
    "RevealBeforeResolutionError",
    "RevealBeforeRetireError",
    "challenge_anchor_payload",
    "record_commitment_anchor",
    "record_commitment",
    "reveal_commitment",
    "verify_reveal",
    "verify_reveal_deep",
    "deep_reveal_verifier",
    # pool
    "Asset",
    "FingerprintIndex",
    "StaticFingerprintIndex",
    "NearDuplicateError",
    "NoFreshAssetError",
    "UnresolvedChallengeError",
    "add_asset",
    "get_asset",
    "append_provenance",
    "provenance_log",
    "assign_split",
    "source_key",
    "checkout_asset",
    "release_asset",
    "retire_asset",
    "check_near_duplicate",
    # scheduler
    "MIN_SEED_BITS",
    "Challenge",
    "ChallengeIntegrityError",
    "DispatchPayload",
    "IngestResult",
    "PayloadLeakError",
    "WeakSeedError",
    "confirm_ingest_step",
    "make_challenge",
    "record_challenge",
    "register_asset",
    "resolve_challenge",
]
