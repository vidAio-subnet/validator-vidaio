"""Chain access layer.

Every service talks to the chain ONLY through the ChainAdapter Protocol; the real
bittensor implementation is a later, thin adapter (dep not installed yet — see
DEPS.md). InMemoryChain is the local-testing implementation: deterministic,
records every set_weights call and anchored payload for assertions.

Report mode (the project design record rule 8 — the DEFAULT): make_chain_adapter builds
HttpChainAdapter against a running chainsim (vidaio.chainsim), or an
EmbeddedReportingChain (JSONL-journaled InMemoryChain) for single-process runs.

Two properties every caller should know about the Protocol:
- freshness is explicit — `has_fresh_snapshot(now, max_age)` / `snapshot_age(now)`,
  and `neurons()` raises ChainStateUnavailable when nothing was ever fetched
  (never an empty list standing in for "the chain is unreachable");
- mutations are authenticated — HttpChainAdapter carries the hotkey's chainsim
  token (`chain.auth_token`, or the one captured by `register()`), the seat the
  real adapter's signing keypair will take.

Plus one OPTIONAL surface, `SubmittedWeightsReader.submitted_weights(hotkey)`:
the vector the chain currently records for a hotkey. It is what lets the
weight-setter prove that a SPECIFIC weight vector landed before publishing it;
without it every confirmation is UNKNOWN. InMemoryChain, EmbeddedReportingChain
and HttpChainAdapter all implement it.
"""

from vidaio.chain.adapter import (
    BlockPinnedNeuronsReadable,
    BurnUidReadable,
    ChainAdapter,
    ChainCommitmentRecord,
    ChainNeuron,
    ChainStateUnavailable,
    CommitmentCapacity,
    CommitmentCapacityReadable,
    CommitmentRecordReadable,
    CommitmentRateLimitReadable,
    CommitRevealReadable,
    EpochBoundary,
    EpochBoundaryReadable,
    FinalizedBlockReadable,
    HistoricalEpochAnchorReadable,
    InMemoryChain,
    PendingWeightReveal,
    SetWeightsResult,
    SubmittedWeights,
    SubmittedWeightsReader,
    resolve_burn_uid,
)

# Safe to import at module top: bittensor_adapter imports bittensor LAZILY (inside
# the transport), so this line never drags the heavy SDK in. `quantize_u16` is the
# deterministic u16 quantization the convergence phase also depends on.
from vidaio.chain.bittensor_adapter import (
    BittensorAdapterConfig,
    BittensorChainAdapter,
    BittensorHotkeySigner,
    BittensorReadOnlyChainAdapter,
    ReadOnlyChainError,
    quantize_u16,
)
from vidaio.chain.client import EmbeddedReportingChain, HttpChainAdapter
from vidaio.chain.factory import (
    ChainConfig,
    make_chain_adapter,
    make_read_only_chain_adapter,
)

__all__ = [
    "BittensorAdapterConfig",
    "BittensorChainAdapter",
    "BittensorHotkeySigner",
    "BittensorReadOnlyChainAdapter",
    "BlockPinnedNeuronsReadable",
    "BurnUidReadable",
    "ChainAdapter",
    "ChainConfig",
    "ChainCommitmentRecord",
    "ChainNeuron",
    "ChainStateUnavailable",
    "CommitmentCapacity",
    "CommitmentCapacityReadable",
    "CommitmentRecordReadable",
    "CommitmentRateLimitReadable",
    "CommitRevealReadable",
    "EpochBoundary",
    "EpochBoundaryReadable",
    "FinalizedBlockReadable",
    "HistoricalEpochAnchorReadable",
    "EmbeddedReportingChain",
    "HttpChainAdapter",
    "InMemoryChain",
    "PendingWeightReveal",
    "ReadOnlyChainError",
    "SetWeightsResult",
    "SubmittedWeights",
    "SubmittedWeightsReader",
    "make_chain_adapter",
    "make_read_only_chain_adapter",
    "quantize_u16",
    "resolve_burn_uid",
]
