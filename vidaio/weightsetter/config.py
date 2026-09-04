"""Weight-setter configuration — schema for the `weightsetter:` section of config.

Cadence anchor (spec: design spec §01): the legacy validator attempts a weight set every
72 minutes against a 100-block (~20 min) on-chain tempo gate; the attempt interval
default preserves that cadence. Chain writes are timeout-guarded and retried with
bounded backoff (vidaio.core.resilience) — the timeout/retry envelope lives here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WeightSetterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: How often a weight-set is attempted (design spec §01: 72 min against a ~20 min tempo).
    attempt_interval_seconds: float = Field(default=72 * 60.0, gt=0)

    #: Timeout around anchor_commitment. set_weights is deliberately not caller-
    #: bounded; both live writes may span inclusion + finalization. Keep this at
    #: least 180s so a normal anchor is not abandoned with an ambiguous outcome.
    chain_timeout_seconds: float = Field(default=180.0, gt=0)

    #: Bounded retry envelope for chain writes (vidaio.core.resilience.retry_async).
    chain_retry_attempts: int = Field(default=3, ge=1)
    chain_retry_base_delay_seconds: float = Field(default=1.0, gt=0)

    #: Fleet convergence fence, synchronized with EPOCH_LOG_SCHEMA_VERSION.
    #: Report/test overlays may explicitly use zero; live defaults never do.
    version_key: int = Field(default=16, ge=0)

    #: This validator's hotkey. Used ONLY to read our own weight vector back off
    #: the chain when a set_weights attempt was ambiguous (a timeout leaves us
    #: unable to tell whether the extrinsic landed). Empty means the
    #: chain cannot be asked at all, so every ambiguous attempt stays UNKNOWN:
    #: its intent is never abandoned AND never published (round 3 — an
    #: unconfirmed vector is not publishable). Set it in any deployment that
    #: publishes.
    validator_hotkey: str = ""

    #: Chain-snapshot staleness gate: skip the whole attempt rather
    #: than submit a vector composed over a stale/empty metagraph. 0 disables it.
    max_chain_snapshot_age_seconds: float = Field(default=3600.0, ge=0)

    #: How long a POSITIVELY DENIED weight intent must have existed before the
    #: reconciliation pass is allowed to abandon it.
    #:
    #: Abandonment is terminal — an abandoned vector is never published — so it
    #: may only follow a positive denial from a FRESH post-write snapshot, and
    #: only once enough chain time has passed that a genuinely-accepted write
    #: could not still be in flight. An intent whose fate is UNKNOWN is never
    #: abandoned at any age: it stays pending and keeps being re-checked. Default
    #: is several tempos (~20 min each), comfortably longer than propagation.
    abandon_denied_intent_after_seconds: float = Field(default=3600.0, ge=0)

    #: Window of score-packet evidence a publication commits to when the
    #: PublicationInputs provider supports `recent_packet_digests(since)` and no
    #: previous publication watermark exists yet.
    publication_lookback_seconds: float = Field(default=24 * 3600.0, gt=0)

    #: Overall wall-clock budget for one best-effort post-submit publication.
    #: Object-store work runs off the emissions event loop and a timeout leaves
    #: the durable accepted intent queued.  This prevents a wedged evidence store
    #: or commitment lane from interrupting later scheduled weight attempts.
    publication_attempt_timeout_seconds: float = Field(default=300.0, gt=0)

    #: Report/local fallback for the canonical empty-epoch recipient. Production
    #: resolves the subnet owner's current uid directly from chain state; a missing
    #: or unreadable owner uid HOLDs the submission and is never replaced with zero.
    #: Report overlays explicitly pin uid 0 for dependency-free adapters.
    burn_uid: int | None = Field(default=None, ge=0)

    #: Metrics/health port (service port map: vidaio/services/protocol.py).
    metrics_port: int = 9102

    #: The auditable weight path (design spec §15 review fix "publish the exact weight
    #: vector"): store the submitted vector, ledger a PublicationRecord, anchor it
    #: on chain. Disable ONLY in dev/test environments without an audit store.
    publication_enabled: bool = True

    #: Health check: degraded once the last successful weight set (or service start)
    #: is older than this. Default = 4 attempt intervals.
    max_last_success_age_seconds: float = Field(default=4 * 72 * 60.0, gt=0)

    # -- snapshot source (build-wave 5, the project design record §1(b)) --------
    #: WHERE the miner snapshots + crown/result come from.
    #:  - "shared": the SharedSnapshotProvider fetches the Scoring Authority's epoch
    #:    pointer, mirrors the epoch-log bytes, verifies the three-way digest chain,
    #:    and hands the weight-setter the log's inputs so every validator builds the
    #:    IDENTICAL vector (convergence, the project design record rule 9). PRODUCTION.
    #:  - "local": the existing miner_manager path — per-validator EWMA folds that do
    #:    NOT converge; kept for report-mode / dryrun / third-party recompute overlays
    #:    (the project design record rule 8). Both modes drive the identical service code.
    #: The provider itself is injected into WeightSetter; this key drives the
    #: `make_snapshot_provider` selection at wiring time.
    provider: Literal["local", "shared"] = "local"

    #: The Scoring Authority pointer API base URL (provider = "shared"). Empty in
    #: local mode. Carries no credentials.
    authority_url: str = ""
    #: Bearer token presented to the Scoring Authority (its `authority.api_token`).
    #: Empty = send no Authorization header (only against an open/dev authority).
    authority_token: str = ""
    #: The subnet id stamped in the on-chain anchor payload; the anchor reader keys
    #: off it when verifying the third digest leg.
    authority_netuid: int = Field(default=85, ge=0)
    #: Timeout around every pointer fetch from the authority.
    authority_timeout_seconds: float = Field(default=10.0, gt=0)
    #: Verify the on-chain anchored digest as the third, independent leg of the
    #: tamper-evidence chain (sha256(bytes) == pointer digest == anchored digest).
    #: When True an anchor READER MUST be wired (the provider HOLDS rather than skip
    #: the leg — #3); a finalized-but-not-yet-anchored epoch HOLDs; a mismatch
    #: REFUSES. Disable ONLY in overlays with genuinely no chain anchor.
    verify_anchor: bool = True

    # -- convergence-health observation (observe-only; never a submit gate) -------
    #: After submitting our OWN honest vector, read peer validators' on-chain
    #: vectors and emit `vidaio_weightsetter_convergence` = fraction agreeing. This
    #: surfaces divergence for the dashboard/operator BEFORE it costs emissions; it
    #: NEVER changes what this validator submits (it always submits its own vector).
    #: On an honest network all peer validators reading the same epoch converge
    #: to 1.0.
    convergence_observe_enabled: bool = False
    #: Explicit peer validator hotkeys to sample. Empty + metagraph peers off =
    #: nothing to observe (the gauge stays unset).
    convergence_peer_hotkeys: list[str] = Field(default_factory=list)
    #: Also sample every validator neuron in the metagraph (excluding ours) as a
    #: peer, in addition to any explicit hotkeys.
    convergence_use_metagraph_peers: bool = False
