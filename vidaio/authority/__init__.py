"""The Scoring Authority (central, owner-run) — the epoch-log producer side.

Build-wave 3 delivers the finalizer: it assembles the per-epoch `EpochLog` from the
central scorer's folded snapshots + competition reward-window state + audit files, writes it
to the object store behind a `_FINALIZED` marker, and hands back the pointer
(object key + `log_digest` + `weight_vector_digest`) the Scoring Authority API
publishes and the on-chain anchor binds. See the project design record §1(a), §4.
"""

from vidaio.authority.anchoring import ANCHOR_DOMAIN, anchor_epoch, anchor_payload
from vidaio.authority.api import (
    AnchorPointer,
    AnchorRecord,
    EpochPointer,
    anchor_from_record,
    pointer_from_record,
)
from vidaio.authority.config import AuthorityConfig
from vidaio.authority.finalizer import (
    EPOCH_LOG_MEMBER,
    AuditFileMissingError,
    ChallengeCommitmentSource,
    EpochFinalizer,
    FinalizedEpoch,
    ScoredItem,
    build_audit_manifest,
    epoch_prefix,
)
from vidaio.authority.index import EpochIndex, EpochIndexConflict, EpochRecord
from vidaio.authority.service import ScoringAuthority

__all__ = [
    # producer (wave 3)
    "EpochFinalizer",
    "FinalizedEpoch",
    "ScoredItem",
    "ChallengeCommitmentSource",
    "build_audit_manifest",
    "AuditFileMissingError",
    "epoch_prefix",
    "EPOCH_LOG_MEMBER",
    # config
    "AuthorityConfig",
    # epoch index
    "EpochIndex",
    "EpochRecord",
    "EpochIndexConflict",
    # anchoring
    "anchor_epoch",
    "anchor_payload",
    "ANCHOR_DOMAIN",
    # pointer API
    "ScoringAuthority",
    "EpochPointer",
    "AnchorPointer",
    "AnchorRecord",
    "pointer_from_record",
    "anchor_from_record",
]
