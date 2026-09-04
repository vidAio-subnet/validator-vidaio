"""The shared per-epoch EPOCH-LOG model (dependency-light: pydantic + stdlib).

Imported by BOTH the central Scoring Authority (produces the log via
`vidaio.authority.EpochFinalizer`) and the thin validator/auditor (consume it:
converge from the weight vector, verify from the audit manifest). See
the project design record §3.1 and the project design record §2.2 (build-wave 3).
"""

from vidaio.epoch.log import (
    EPOCH_LOG_SCHEMA_VERSION,
    AuditFileKind,
    AuditFileRef,
    AuditManifest,
    AvailabilityInput,
    CompetitionAuditItem,
    CompetitionAuditSubject,
    CompetitionInput,
    CycleScore,
    EarningInput,
    EpochLog,
    EpochLogInputs,
    EpochLogInvalid,
    MinerCensusEntry,
    weight_vector_digest,
)

__all__ = [
    "EpochLog",
    "EpochLogInputs",
    "EpochLogInvalid",
    "MinerCensusEntry",
    "AuditFileRef",
    "AuditFileKind",
    "AuditManifest",
    "AvailabilityInput",
    "CompetitionAuditItem",
    "CompetitionAuditSubject",
    "CompetitionInput",
    "CycleScore",
    "EarningInput",
    "EPOCH_LOG_SCHEMA_VERSION",
    "weight_vector_digest",
]
