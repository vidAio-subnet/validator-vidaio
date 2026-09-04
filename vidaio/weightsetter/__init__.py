"""Weight-setter + publication service.

Composes the weight vector from injected miner snapshots + persisted reward-window state +
the latest ingested competition result, submits it through the ChainAdapter on a
tempo-gated cadence, and publishes the EXACT submitted vector through the audit
store + commitment ledger so third parties can reproduce the chain weights.
Configuration section: `weightsetter` (see config.py).
"""

from vidaio.weightsetter import intents
from vidaio.weightsetter.config import WeightSetterConfig
from vidaio.weightsetter.crown_store import (
    ResultConflictError,
    ingest_competition_result,
    latest_result,
    load_reward_window,
    migrate,
    save_reward_window,
)
from vidaio.weightsetter.service import (
    EMPTY_SCORE_PACKET_MARKER,
    EMPTY_SCORE_PACKET_SET_ROOT,
    WEIGHT_VECTOR_DOMAIN,
    ChainConfirmation,
    PublicationInputs,
    SnapshotProvider,
    WeightSetter,
    weight_vector_document,
)
from vidaio.weightsetter.shared_snapshot import (
    EpochAnchorReader,
    EpochInputs,
    EpochLogStore,
    HttpScoringAuthorityClient,
    InMemoryChainAnchorReader,
    ScoringAuthorityClient,
    SharedSnapshotError,
    SharedSnapshotProvider,
    SnapshotDigestMismatch,
    SnapshotUnavailable,
    make_snapshot_provider,
)

__all__ = [
    "WeightSetterConfig",
    "WeightSetter",
    "SnapshotProvider",
    "PublicationInputs",
    # shared-snapshot convergence provider (build-wave 5)
    "SharedSnapshotProvider",
    "EpochInputs",
    "ScoringAuthorityClient",
    "HttpScoringAuthorityClient",
    "EpochLogStore",
    "EpochAnchorReader",
    "InMemoryChainAnchorReader",
    "SharedSnapshotError",
    "SnapshotUnavailable",
    "SnapshotDigestMismatch",
    "make_snapshot_provider",
    # tri-state verdict of a post-write chain read
    "ChainConfirmation",
    "WEIGHT_VECTOR_DOMAIN",
    "EMPTY_SCORE_PACKET_MARKER",
    "EMPTY_SCORE_PACKET_SET_ROOT",
    "weight_vector_document",
    "migrate",
    "load_reward_window",
    "save_reward_window",
    "ingest_competition_result",
    "latest_result",
    "ResultConflictError",
    # weight-submission intent ledger (durability across set_weights)
    "intents",
]
