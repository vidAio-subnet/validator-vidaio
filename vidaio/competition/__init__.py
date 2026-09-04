"""Competition core: manifest schema, phase lifecycle, persistence, event log,
human review (spec: the design spec §04-§06; locked choices: the project design record).

The sandbox runner and scoring client are later phases — only their Protocols are
defined here (vidaio.competition.interfaces).
"""

from vidaio.competition.config import CompetitionConfig
from vidaio.competition.engine import IllegalTransition, LifecycleEngine
from vidaio.competition.interfaces import (
    BUILD_IDENTITY_SCHEME,
    BatchItem,
    BatchOutput,
    CompetitionScoringClient,
    ContenderSpec,
    IsolationProbeReport,
    SandboxRunner,
    ScorePacket,
    logical_build_identity,
)
from vidaio.competition.manifest import (
    ArchivedBaseline,
    CompetitionManifest,
    EvaluationBatchSizeBounds,
    ManifestBoundsError,
    ScoringFactors,
    validate_against_config,
)
from vidaio.competition.item_commitment import (
    EVALUATION_ITEM_COMMITMENT_DOMAIN,
    evaluation_item_commitment,
    evaluation_item_preimage,
)
from vidaio.competition.repository import (
    CompetitionRecord,
    ContenderRecord,
    EnrollmentError,
    MIGRATIONS_DIR,
    ScorePacketError,
    ScorePacketPayload,
    migrate,
    verify_review_chain,
)
from vidaio.competition.review import (
    ReviewError,
    ReviewWindowClosed,
    recalculate_ranks,
    submit_review,
)
from vidaio.competition.states import (
    Phase,
    RUNNING_PHASES,
    TERMINAL_PHASES,
    TRANSITIONS,
    is_allowed,
)

__all__ = [
    "CompetitionConfig",
    "LifecycleEngine",
    "IllegalTransition",
    "CompetitionManifest",
    "ArchivedBaseline",
    "ScoringFactors",
    "EvaluationBatchSizeBounds",
    "ManifestBoundsError",
    "validate_against_config",
    "EVALUATION_ITEM_COMMITMENT_DOMAIN",
    "evaluation_item_commitment",
    "evaluation_item_preimage",
    "Phase",
    "RUNNING_PHASES",
    "TERMINAL_PHASES",
    "TRANSITIONS",
    "is_allowed",
    "migrate",
    "MIGRATIONS_DIR",
    "CompetitionRecord",
    "ContenderRecord",
    "EnrollmentError",
    "ScorePacketError",
    "ScorePacketPayload",
    "verify_review_chain",
    "submit_review",
    "recalculate_ranks",
    "ReviewError",
    "ReviewWindowClosed",
    "SandboxRunner",
    "BUILD_IDENTITY_SCHEME",
    "logical_build_identity",
    "CompetitionScoringClient",
    "ContenderSpec",
    "BatchItem",
    "BatchOutput",
    "IsolationProbeReport",
    "ScorePacket",
]
