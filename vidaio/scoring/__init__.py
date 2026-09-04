"""Scoring engine: pure, deterministic, auditable composition/gating/aggregation.

Spec authority: the design spec §02 (formulas), §08 (recomputability),
§18 (anti-gaming: canonicalize first, validity gates first, worst-decile aggregation,
cross-miner dedup). Expensive metric computation lives behind the injectable Protocols
in :mod:`vidaio.scoring.backends`; nothing in this package performs I/O, shells out, or
uses randomness.

Honest-rebuild boundary: scores come only from measured metric inputs — there is no
substitution path and no per-hotkey special case anywhere in this package.
"""

from vidaio.scoring.aggregate import (
    competition_final_score,
    length_weighted_mean,
    worst_decile_from_config,
    worst_decile_score,
)
from vidaio.scoring.backends import (
    DeterministicFakeBackend,
    MediaInfo,
    PerceptualCheckBackend,
    PerceptualCheckResult,
    PerceptualHashBackend,
    PieAppBackend,
    ProbeBackend,
    VmafBackend,
    derive_pieapp_start_frame,
    usable_frames,
)
from vidaio.scoring.canonicalize import (
    SecondaryDecoderBackend,
    build_canonicalization_plan,
    cross_check_decoders,
    plan_digest,
    plan_template_digest,
    validate_stream,
)
from vidaio.scoring.compression import (
    CompressionBreakdown,
    compression_rate,
    compression_score_from_rate,
    score_compression,
)
from vidaio.scoring.config import (
    TRACK_COMPRESSION,
    TRACK_UPSCALING,
    AggregateWeights,
    CompressionWeights,
    ScoringConfig,
)
from vidaio.scoring.dedup import (
    DedupEntry,
    DedupVerdict,
    dedup_responses,
)
from vidaio.scoring.duplicate_evidence import (
    DUPLICATE_EVIDENCE_RULE,
    DUPLICATE_SCORER_PREFIX,
    DUPLICATE_WITNESS_METRIC,
    DuplicateWitness,
    InvalidDuplicateEvidence,
    canonical_receipt_digest,
    duplicate_identity,
    duplicate_order_key,
    duplicate_witness_from_packet,
    is_duplicate_identity,
    mint_duplicate_packet,
    parse_duplicate_witness,
)
from vidaio.scoring.finite import require_finite
from vidaio.scoring.gates import (
    ChromaUvGate,
    ColorGrayscaleGate,
    CompressionRateGate,
    EncodingGate,
    FileSizeCapGate,
    FrameCountGate,
    Gate,
    GateContext,
    GatePipeline,
    GateSkip,
    PERCEPTUAL_GATE_NAMES,
    ReasonCode,
    SkippedGate,
    ToneManipulationGate,
    ValidityViolation,
    VmafFloorGate,
    VmafModelDeltaGate,
    default_pipeline,
    pipeline_without_perceptual_checks,
)
from vidaio.scoring.perceptual_cpu import (
    CPU_PERCEPTUAL_ALGORITHM_VERSION,
    CpuPerceptualConfig,
    PerceptualStatistics,
    chroma_uv_result,
    grayscale_result,
    tone_manipulation_result,
)
from vidaio.scoring.phash_cpu import (
    CPU_VIDEO_PHASH_VERSION,
    CpuVideoPhash,
    PerceptualHashUnavailable,
)
from vidaio.scoring.result import ItemScore, compose_item_score, config_digest
from vidaio.scoring.upscaling import (
    UpscalingBreakdown,
    final_from_pre,
    length_score,
    quality_from_pieapp,
    score_upscaling,
)

__all__ = [
    # config
    "ScoringConfig",
    "CompressionWeights",
    "AggregateWeights",
    "TRACK_COMPRESSION",
    "TRACK_UPSCALING",
    # backends
    "MediaInfo",
    "PerceptualCheckResult",
    "VmafBackend",
    "PieAppBackend",
    "ProbeBackend",
    "PerceptualCheckBackend",
    "PerceptualHashBackend",
    "DeterministicFakeBackend",
    "derive_pieapp_start_frame",
    "usable_frames",
    "CPU_PERCEPTUAL_ALGORITHM_VERSION",
    "CpuPerceptualConfig",
    "PerceptualStatistics",
    "CPU_VIDEO_PHASH_VERSION",
    "CpuVideoPhash",
    "PerceptualHashUnavailable",
    "tone_manipulation_result",
    "grayscale_result",
    "chroma_uv_result",
    # canonicalize
    "build_canonicalization_plan",
    "plan_digest",
    "plan_template_digest",
    "validate_stream",
    "cross_check_decoders",
    "SecondaryDecoderBackend",
    # gates
    "ReasonCode",
    "ValidityViolation",
    "GateSkip",
    "GateContext",
    "Gate",
    "GatePipeline",
    "EncodingGate",
    "FrameCountGate",
    "FileSizeCapGate",
    "CompressionRateGate",
    "VmafFloorGate",
    "VmafModelDeltaGate",
    "ToneManipulationGate",
    "ColorGrayscaleGate",
    "ChromaUvGate",
    "SkippedGate",
    "PERCEPTUAL_GATE_NAMES",
    "default_pipeline",
    "pipeline_without_perceptual_checks",
    # formulas
    "CompressionBreakdown",
    "compression_rate",
    "compression_score_from_rate",
    "score_compression",
    "UpscalingBreakdown",
    "quality_from_pieapp",
    "length_score",
    "final_from_pre",
    "score_upscaling",
    # aggregation
    "worst_decile_score",
    "worst_decile_from_config",
    "length_weighted_mean",
    "competition_final_score",
    # dedup
    "DedupEntry",
    "DedupVerdict",
    "dedup_responses",
    "DUPLICATE_EVIDENCE_RULE",
    "DUPLICATE_SCORER_PREFIX",
    "DUPLICATE_WITNESS_METRIC",
    "DuplicateWitness",
    "InvalidDuplicateEvidence",
    "canonical_receipt_digest",
    "duplicate_identity",
    "duplicate_order_key",
    "duplicate_witness_from_packet",
    "is_duplicate_identity",
    "mint_duplicate_packet",
    "parse_duplicate_witness",
    # result
    "ItemScore",
    "compose_item_score",
    "config_digest",
    # finite
    "require_finite",
]
