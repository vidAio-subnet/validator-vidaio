"""vidaio.auditor — the thin validator's second hat: independent honesty check.

The auditor is the decentralized check on the central Scoring Authority
(the project design record rule 10, the project design record §1c, build wave 6). Each
epoch it:

1. reads the epoch log's audit manifest and DETERMINISTICALLY samples its items
   (``vidaio.auditor.sampling``, seeded from epoch_id + auditor identity so a run is
   reproducible and un-steerable);
2. RECOMPUTES each sampled score over the ACTUAL scoring engine
   (:class:`RealScoreRecomputer` → ``vidaio.audit.recompute.verify_bundle``), which
   makes an injected/substituted score fail an independent audit (SCORE_MISMATCH) —
   and honestly REFUSES (SKIP) an item when a required CPU backend/artifact is unavailable,
   never a false CLEAN;
3. RE-DERIVES the log's own weight vector from its stated inputs (a cheap,
   media-free check that catches a WEIGHT_DERIVATION_MISMATCH before any recompute);
4. aggregates a signed, deterministic :class:`AuditReport` and submits it through the
   :class:`AuditResultsClient` seam (real API: wave 7).

Public API below; the Scoring Authority API (``vidaio.authority``) is a separate,
concurrently-built package the auditor never imports.
"""

from vidaio.auditor.beacon import (
    AnchorBlockReadable,
    AnchorReadable,
    BeaconGrindRisk,
    BeaconUnavailable,
    BlockHashReadable,
    chain_beacon,
)
from vidaio.auditor.client import (
    AuditResultsClient,
    RecordingAuditResultsClient,
    SubmitAck,
)
from vidaio.auditor.config import AuditorConfig, SamplePolicy
from vidaio.auditor.chronology import (
    CHALLENGE_CHRONOLOGY_INVALID,
    CHALLENGE_CHRONOLOGY_UNVERIFIED,
    ChronologyKind,
    ChronologyResult,
    verify_challenge_chronology,
)
from vidaio.auditor.recomputer import RealScoreRecomputer, RecomputeUnavailable
from vidaio.auditor.report import (
    BURN_UID_MISMATCH,
    BURN_UID_UNVERIFIED,
    CENSUS_MISMATCH,
    CREATED_AT_MISMATCH,
    CREATED_AT_UNVERIFIED,
    DUPLICATE_AUDIT_IDENTITY,
    EPOCH_LOG_INVALID,
    EPOCH_LOG_UNVERIFIED,
    EARNING_PACKET_REPLAY,
    EARNING_STATE_MISMATCH,
    EARNING_STATE_RESET,
    EARNING_STATE_UNVERIFIED,
    FOLD_CURSOR_MISMATCH,
    METAGRAPH_DEDUP_MISMATCH,
    METAGRAPH_TRACK_MISMATCH,
    PREDECESSOR_CHAIN_BROKEN,
    PREDECESSOR_UNVERIFIED,
    REWARD_WINDOW_MISMATCH,
    SNAPSHOT_UNVERIFIED,
    UNKNOWN_TRACK,
    WEIGHT_DERIVATION_MISMATCH,
    AuditMode,
    AuditReport,
    AuditStatus,
    ItemVerdict,
    ItemVerdictKind,
    ReportSigner,
    Sha256Signer,
    WeightVerdict,
    overall_status,
)
from vidaio.auditor.sampling import (
    AuditItem,
    DuplicateAuditIdentity,
    ManifestIncomplete,
    manifest_items,
    sample_items,
)
from vidaio.auditor.service import (
    Auditor,
    BundleSource,
    BundleUnavailable,
    InMemoryBundleSource,
    StoredBundleSource,
    persist_bundle,
)

__all__ = [
    # config
    "AuditorConfig",
    "SamplePolicy",
    "CHALLENGE_CHRONOLOGY_INVALID",
    "CHALLENGE_CHRONOLOGY_UNVERIFIED",
    "ChronologyKind",
    "ChronologyResult",
    "verify_challenge_chronology",
    # recomputer
    "RealScoreRecomputer",
    "RecomputeUnavailable",
    # sampling
    "AuditItem",
    "DuplicateAuditIdentity",
    "ManifestIncomplete",
    "manifest_items",
    "sample_items",
    # chain-derived sampling beacon (#10; future-finalized-block-hash round-6 #2)
    "AnchorReadable",
    "AnchorBlockReadable",
    "BlockHashReadable",
    "BeaconUnavailable",
    "BeaconGrindRisk",
    "chain_beacon",
    # service
    "Auditor",
    "BundleSource",
    "BundleUnavailable",
    "InMemoryBundleSource",
    "StoredBundleSource",
    "persist_bundle",
    # report
    "AuditMode",
    "AuditReport",
    "AuditStatus",
    "ItemVerdict",
    "ItemVerdictKind",
    "WeightVerdict",
    "WEIGHT_DERIVATION_MISMATCH",
    "EARNING_STATE_MISMATCH",
    "EARNING_PACKET_REPLAY",
    "EARNING_STATE_RESET",
    "DUPLICATE_AUDIT_IDENTITY",
    "EPOCH_LOG_INVALID",
    "EPOCH_LOG_UNVERIFIED",
    "EARNING_STATE_UNVERIFIED",
    "FOLD_CURSOR_MISMATCH",
    "REWARD_WINDOW_MISMATCH",
    "METAGRAPH_DEDUP_MISMATCH",
    "METAGRAPH_TRACK_MISMATCH",
    "SNAPSHOT_UNVERIFIED",
    "UNKNOWN_TRACK",
    "CREATED_AT_MISMATCH",
    "CREATED_AT_UNVERIFIED",
    "CENSUS_MISMATCH",
    "BURN_UID_MISMATCH",
    "BURN_UID_UNVERIFIED",
    "PREDECESSOR_CHAIN_BROKEN",
    "PREDECESSOR_UNVERIFIED",
    "overall_status",
    "ReportSigner",
    "Sha256Signer",
    # client seam
    "AuditResultsClient",
    "RecordingAuditResultsClient",
    "SubmitAck",
]
