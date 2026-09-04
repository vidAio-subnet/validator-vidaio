"""Inference validator service: round loop, miner registry, process supervisor.

Spec: the design spec §01 (round loop, rebuilt honest), §07 (the fixed
TaskWarrant — unknown-track miners are skipped, never defaulted to upscaling),
§13 (process-isolation topology). EWMA decay and eligibility rules come from
vidaio.tokenomics; wire contracts from vidaio.services.protocol.
"""

from pathlib import Path

from vidaio.validator.availability import (
    AvailabilityFailureReason,
    AvailabilityObservation,
    DispatchAttempt,
    build_availability_observation,
    verify_availability_observation,
)
from vidaio.validator.config import ValidatorConfig
from vidaio.validator.evidence import (
    DEFAULT_LOOKBACK_SECONDS,
    AvailabilityFoldEvidence,
    ScorePacketEvidence,
)
from vidaio.validator.inference import (
    AuditStoreFailure,
    ChallengeAlreadyTerminal,
    ChallengeClient,
    ChallengeItem,
    ChallengeOwnershipRefused,
    DispatchedChallenge,
    HttpChallengeClient,
    HttpMinerClient,
    HttpScoringClient,
    InferenceValidator,
    MinerClient,
    PacketEvidence,
    RoundReport,
    ScoringClient,
    dedup_miners,
    sha256_file,
)
from vidaio.validator.miner_manager import (
    KNOWN_TRACKS,
    RESOLVE_OUTCOMES,
    RegistryUpdate,
    RoundLedgerError,
    apply_scores,
    begin_round,
    clear_inflight_challenge,
    clear_scorer_pin,
    commit_round,
    connection_factory,
    get_miner,
    inflight_challenges,
    load_scorer_pin,
    normalize_track,
    planned_tracks,
    record_inflight_challenge,
    record_scorer_pin,
    record_track,
    set_inflight_outcome,
    snapshot,
    sync_neurons,
    track_of,
    transaction,
    uncommitted_rounds,
    utc_now_iso,
)
from vidaio.validator.supervisor import (
    STATE_BACKOFF,
    STATE_PARKED,
    STATE_RUNNING,
    STATE_STOPPED,
    ChildParkedError,
    ChildSpec,
    Supervisor,
)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

__all__ = [
    "MIGRATIONS_DIR",
    # config
    "ValidatorConfig",
    "AvailabilityFailureReason",
    "AvailabilityObservation",
    "DispatchAttempt",
    "build_availability_observation",
    "verify_availability_observation",
    # inference
    "InferenceValidator",
    "ChallengeItem",
    "ChallengeClient",
    "ChallengeAlreadyTerminal",
    "ChallengeOwnershipRefused",
    "DispatchedChallenge",
    "AuditStoreFailure",
    "MinerClient",
    "ScoringClient",
    "HttpChallengeClient",
    "HttpMinerClient",
    "HttpScoringClient",
    "RoundReport",
    "PacketEvidence",
    "dedup_miners",
    "sha256_file",
    # score-packet evidence (the weight-setter's PublicationInputs source)
    "ScorePacketEvidence",
    "AvailabilityFoldEvidence",
    "DEFAULT_LOOKBACK_SECONDS",
    # miner manager
    "KNOWN_TRACKS",
    "RESOLVE_OUTCOMES",
    "normalize_track",
    "sync_neurons",
    "get_miner",
    "record_track",
    "track_of",
    "planned_tracks",
    "RegistryUpdate",
    "load_scorer_pin",
    "record_scorer_pin",
    "clear_scorer_pin",
    "apply_scores",
    "snapshot",
    "transaction",
    "connection_factory",
    "utc_now_iso",
    "begin_round",
    "commit_round",
    "uncommitted_rounds",
    "RoundLedgerError",
    "record_inflight_challenge",
    "set_inflight_outcome",
    "inflight_challenges",
    "clear_inflight_challenge",
    # supervisor
    "Supervisor",
    "ChildParkedError",
    "ChildSpec",
    "STATE_RUNNING",
    "STATE_BACKOFF",
    "STATE_PARKED",
    "STATE_STOPPED",
]
