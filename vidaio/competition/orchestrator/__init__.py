"""Competition orchestrator service (spec §14)
lifecycle engine/repository with a SandboxRunner and a CompetitionScoringClient.

Local-first (the project design record workflow rule 7): wire it with
vidaio.competition.runners.DockerSandboxRunner via build_docker_runner.
"""

from vidaio.competition.orchestrator.config import OrchestratorConfig
from vidaio.competition.orchestrator.failures import (
    Fault,
    classify_failure,
    fault_code,
)
from vidaio.competition.orchestrator.results import (
    ActiveBaselineProvenance,
    ResultNotReady,
    build_competition_result,
    result_payload,
)
from vidaio.competition.orchestrator.scoring_client import (
    HttpScoringClient,
    ScoringClientError,
)
from vidaio.competition.orchestrator.service import (
    EMPTY_SHA256,
    AnchorClaimRefused,
    AnchorError,
    AnchorResult,
    EarningManifestError,
    Orchestrator,
    build_docker_runner,
    reward_parameter_digest,
)
from vidaio.competition.orchestrator.zero_packets import (
    ORCHESTRATOR_ZERO_SCORER_NAME,
    ReservedScorerIdentity,
    is_orchestrator_zero_identity,
    orchestrator_zero_identity,
)

__all__ = [
    "Orchestrator",
    "OrchestratorConfig",
    "HttpScoringClient",
    "ScoringClientError",
    "build_docker_runner",
    "EMPTY_SHA256",
    "AnchorClaimRefused",
    "AnchorError",
    "AnchorResult",
    "EarningManifestError",
    "reward_parameter_digest",
    "Fault",
    "classify_failure",
    "fault_code",
    "ORCHESTRATOR_ZERO_SCORER_NAME",
    "ReservedScorerIdentity",
    "is_orchestrator_zero_identity",
    "orchestrator_zero_identity",
    "build_competition_result",
    "result_payload",
    "ActiveBaselineProvenance",
    "ResultNotReady",
]
