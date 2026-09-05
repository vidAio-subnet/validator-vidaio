"""ChainAdapter — the only boundary any service crosses to reach the chain.

Design rules:
- Reads (block, neurons) are synchronous snapshots of the adapter's cached state;
  refreshing that cache is the adapter's own concern (`refresh()`).
- Writes (set_weights, anchor_commitment) are async and MUST be wrapped in
  vidaio.core.resilience.with_timeout by callers — an adapter never guards for you.
- Freshness is explicit: `refresh()` never raises (a flaky chain must not crash a
  service loop), so the adapter instead reports whether its snapshot can be
  trusted — `has_fresh_snapshot()` / `snapshot_age()` — and `neurons()` raises
  ChainStateUnavailable rather than returning an empty list when NOTHING has ever
  been fetched. An empty list means "no neurons registered"; it must never be
  the way "we could not reach the chain" is expressed.
- No service imports bittensor. The real adapter lives behind this Protocol and
  is added as a thin module once the bittensor dependency is introduced.

OPTIONAL EXTENSION — `SubmittedWeightsReader`. An adapter MAY
also implement `submitted_weights(hotkey) -> SubmittedWeights | None`: the weight
vector the chain currently records for that hotkey. It is the only evidence that
can prove a specific weight write LANDED — "our last_update advanced" merely says
somebody's write did, which is not the same claim and is exactly how an
unconfirmed vector used to get published. The weight-setter feature-detects it
(`isinstance(chain, SubmittedWeightsReader)`); an adapter that cannot answer makes
every confirmation UNKNOWN, never CONFIRMED and never DENIED.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import ClassVar, Protocol, runtime_checkable

from vidaio.tokenomics.quantize import max_normalize_u16, quantize_u16


class ChainStateUnavailable(RuntimeError):
    """The adapter has no chain snapshot at all — never successfully refreshed.

    Raised by `neurons()` (and any read that would otherwise substitute an empty
    view of the chain). Callers should treat it as "chain state unknown": skip
    the round / refuse to set weights, never as "the subnet is empty".
    """


class PendingWeightReveal(ChainStateUnavailable):
    """Finalized storage positively contains a weight commit awaiting reveal."""


@dataclass(frozen=True, slots=True)
class ChainCommitmentRecord:
    """Raw contents and original inclusion height of one commitment slot.

    Unlike :class:`EpochAnchorReadable`, this record deliberately does not parse a
    domain-specific payload.  Competition commitments use the already-public
    ``vidaio.commitment.v1:competition:<root>`` wire format, so a raw record is the
    compatibility-preserving seam needed to prove that exact payload at head and
    again from archive state at its inclusion block.
    """

    payload: bytes
    block: int

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes):
            raise TypeError("chain commitment payload must be bytes")
        if not isinstance(self.block, int) or isinstance(self.block, bool) or self.block < 0:
            raise ValueError("chain commitment block must be a non-negative integer")


@dataclass(frozen=True)
class CommitmentCapacity:
    """Block-pinned Commitments-pallet byte budget for one account and subnet.

    The pallet charges ``max(100, payload_bytes)`` for every commitment and
    resets an account's usage when the subnet epoch changes.  ``used_space`` is
    the *effective* usage at ``current_epoch``; ``reported_used_space`` preserves
    a stale tracker's value so operators can see why it was reset to zero.

    This object deliberately carries no default maximum.  ``MaxSpace`` is
    mutable runtime storage and an unreadable value is UNKNOWN, not 3100.
    """

    MIN_WRITE_SPACE: ClassVar[int] = 100

    netuid: int
    hotkey: str
    block: int
    current_epoch: int
    usage_epoch: int | None
    max_space: int
    reported_used_space: int
    used_space: int

    def __post_init__(self) -> None:
        if not isinstance(self.hotkey, str) or not self.hotkey.strip():
            raise ValueError("commitment-capacity hotkey must be non-empty")
        for name in (
            "netuid",
            "block",
            "current_epoch",
            "max_space",
            "reported_used_space",
            "used_space",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}"
                )
        if self.usage_epoch is not None and (
            isinstance(self.usage_epoch, bool)
            or not isinstance(self.usage_epoch, int)
            or self.usage_epoch < 0
        ):
            raise ValueError(
                "usage_epoch must be None or a non-negative integer, got "
                f"{self.usage_epoch!r}"
            )
        expected_used = (
            self.reported_used_space if self.usage_epoch == self.current_epoch else 0
        )
        if self.used_space != expected_used:
            raise ValueError(
                "effective used_space disagrees with the epoch tracker: "
                f"expected {expected_used}, got {self.used_space}"
            )
        if self.used_space > self.max_space:
            raise ValueError(
                f"effective commitment usage {self.used_space} exceeds MaxSpace "
                f"{self.max_space}"
            )

    @property
    def remaining_space(self) -> int:
        """Bytes available for writes in the current subnet epoch."""
        return self.max_space - self.used_space

    @classmethod
    def required_space(cls, payload_bytes: int) -> int:
        """Exact pallet charge for a single Raw commitment payload."""
        if (
            isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes < 0
        ):
            raise ValueError(
                f"payload_bytes must be a non-negative integer, got {payload_bytes!r}"
            )
        return max(cls.MIN_WRITE_SPACE, payload_bytes)

    def can_fit(self, payload_bytes: int) -> bool:
        """Whether one payload can be accepted without ``SpaceLimitExceeded``."""
        return self.required_space(payload_bytes) <= self.remaining_space

    def writes_remaining(self, payload_bytes: int) -> int:
        """How many equal-sized writes fit before this epoch's budget is spent."""
        return self.remaining_space // self.required_space(payload_bytes)


@dataclass(frozen=True)
class EpochBoundary:
    """One chain-native subnet epoch transition.

    ``epoch_id`` is the runtime's monotonic ``SubnetEpochIndex`` *after* the
    transition and ``close_block`` is the exact block where that transition
    happened.  On Bittensor this pair is accepted only when archive state proves
    ``index(close_block - 1) == epoch_id - 1``,
    ``index(close_block) == epoch_id`` and
    ``LastEpochBlock(close_block) == close_block``.

    Keeping the pair together prevents callers from combining an epoch counter
    read at one head with a schedule block read at another head.
    """

    epoch_id: int
    close_block: int


@dataclass(frozen=True)
class ChainNeuron:
    uid: int
    hotkey: str
    coldkey: str
    ip: str
    alpha_stake: float
    emission: float
    #: Validator-permit capability reported by the metagraph. It is deliberately
    #: named for adapter compatibility but is NOT an exclusive economic role: a
    #: serving miner may acquire a permit as stake changes and remains miner-eligible.
    is_validator: bool = False
    #: block of the neuron's last weight update (validators) / registration info
    last_update: int = 0
    #: The block this neuron REGISTERED at. The auditor cross-checks
    #: the committed window evidence's `registration_block` against THIS and re-derives
    #: `has_full_retention_window` from `(close_block - registration_block)`. Safe default 0
    #: (a genesis-era registration) keeps every existing `ChainNeuron(...)` caller working.
    registration_block: int = 0
    #: Axon-advertised TCP port. ``None`` is the compatibility shape for
    #: chainless fakes/older report payloads; real metagraph views preserve the
    #: advertised port and HTTP clients prefer it over their configured fallback.
    axon_port: int | None = None
    #: Metagraph incentive share (the miner-side of emission). Informational —
    #: surfaced by the dashboard's miner-stats feed; never used for dispatch.
    incentive: float = 0.0


@dataclass(frozen=True)
class SetWeightsResult:
    #: True only when the new vector is active in ``Weights`` and can be
    #: independently read back.  A finalized CRv4 timelocked commitment is NOT a
    #: success yet; it is represented by ``pending_reveal=True`` below.
    success: bool
    block: int
    message: str = ""
    #: The vector ACTUALLY emitted to chain (uid -> max-grid u16), after exact-target
    #: verification + VIDAIO sum-grid + pinned-SDK max-grid steps. The live adapter
    #: never drops/requantizes a target: churn rejects the whole attempt before a
    #: write, while a successful result is the exact authority target set. Empty on a
    #: failed/rejected submit (nothing landed). Also populated for a finalized
    #: timelocked commitment so recovery knows the exact vector it must eventually
    #: read back, but it MUST NOT be published until ``success`` is true or an
    #: independent ``submitted_weights`` read confirms it.
    submitted: dict[int, int] = field(default_factory=dict)
    #: The chain accepted a commit-reveal commitment, but the vector is not active
    #: in ``Weights`` yet. Callers must keep their durable intent pending, must not
    #: publish it, and must not submit another vector while this flag is true.
    pending_reveal: bool = False


@dataclass(frozen=True)
class SubmittedWeights:
    """What the chain CURRENTLY records as one hotkey's weight vector.

    `weights` is uid -> weight in whatever scale the adapter finds cheapest to
    report: the raw u16 values the chain stores, a sum-normalized view, or the
    untouched floats of an in-process fake. The comparison the weight-setter runs
    is SCALE-INVARIANT (max-normalized onto the chain's u16 grid — see
    vidaio.weightsetter.intents.quantize_weights), so any positive rescaling of
    the same vector still matches. What must NOT be done is reporting a vector
    that has been re-weighted, filtered or padded: that is a different vector and
    will read as one.

    An EMPTY mapping means "a record exists but carries no positive weight".
    "This hotkey has no weight record at all" is `None` from
    `submitted_weights()` — a POSITIVE answer, and the only one that can deny an
    intent on its own.

    `block` is the block at which that vector was recorded, or None when the
    adapter cannot report it cheaply. It is what disambiguates two attempts that
    carry an identical vector, so report it when you can.
    """

    weights: dict[int, float]
    block: int | None = None


@runtime_checkable
class SubmittedWeightsReader(Protocol):
    """OPTIONAL ChainAdapter extension: read a hotkey's on-chain weight vector.

    Round-3 an internal review Confirming a weight write from block bookkeeping alone
    ("our last_update is at/after the attempt block") proves that SOME write of
    ours landed at some point — never that THIS attempt's exact vector did. Only
    a read of the vector itself can decide that, so this is the surface that
    turns "probably landed" into publishable evidence.

    Contract:
    - returns the CURRENT record for `hotkey` (the latest accepted vector);
    - returns None to say POSITIVELY that the chain holds no weights for it;
    - RAISES (ChainStateUnavailable or any transport/decoding error) when the
      answer cannot be read. Never substitute None for a failed read: None denies
      an intent, and a denied intent is eventually abandoned unpublished.
    """

    def submitted_weights(self, hotkey: str) -> SubmittedWeights | None: ...


@runtime_checkable
class CommitRevealReadable(Protocol):
    """Optional live-chain visibility into weight commit-reveal state.

    The v10 SDK's ``set_weights`` call returns after a CRv4 timelocked commit is
    finalized. Until the automatic reveal executes, the ordinary ``Weights``
    storage still contains the previous vector. These two probes let a caller
    distinguish that expected pending state from an accepted active vector and
    prevent a second submission from being fired into the reveal window.
    """

    def commit_reveal_enabled(self) -> bool: ...

    def weight_commit_pending(self, hotkey: str) -> bool: ...


@runtime_checkable
class CommitmentRateLimitReadable(Protocol):
    """Optional generic transaction-rate-limit diagnostic.

    Current Commitments writes are governed by :class:`CommitmentCapacity`, not
    this Subtensor-wide value. It must never be used to authorize or schedule an
    anchor.
    """

    def commitment_rate_limit(self) -> int: ...


@runtime_checkable
class CommitmentCapacityReadable(Protocol):
    """Optional exact Commitments-pallet capacity visibility.

    Reads must be pinned to one chain block and must fail closed when MaxSpace,
    UsedSpaceOf, or the subnet epoch cannot be decoded.
    """

    def commitment_capacity(self, netuid: int, hotkey: str) -> CommitmentCapacity: ...


def _last_matching_anchor_index(
    payloads: object, *, netuid: int, epoch_id: int, domain: str
) -> int | None:
    """Index of the LAST payload matching `<domain>:<netuid>:<epoch_id>:` (or None).

    The single matching rule `parse_anchor_digest` and `read_anchor_block` share:
    the digest read wants the payload's TAIL, the anchor-block read wants WHICH anchor
    matched (so it can look up the inclusion block recorded alongside it). Factoring
    the match here keeps the two reads byte-for-byte consistent about which anchor is
    "the epoch's anchor". `payloads` is materialized to a list so a
    one-shot iterable is not consumed twice.
    """
    prefix = f"{domain}:{netuid}:{epoch_id}:"
    found: int | None = None
    for i, payload in enumerate(list(payloads or ())):
        try:
            text = bytes(payload).decode("ascii")
        except (UnicodeDecodeError, TypeError, ValueError):
            continue
        if text.startswith(prefix):
            found = i
    return found


def earliest_reanchor_block(
    entries: object, *, netuid: int, epoch_id: int, domain: str
) -> int | None:
    """Inclusion block for `(netuid, epoch_id)`'s committed anchor, choosing the EARLIEST
    block among anchors of the SAME committed payload.

    `entries` is an iterable of `(payload_bytes, block)`. The COMMITTED anchor is the one
    `read_anchor` / `parse_anchor_digest` select — the LAST payload matching this epoch's
    `<domain>:<netuid>:<epoch_id>:` prefix (idempotent anchoring: at most one distinct digest
    is the epoch's commitment). A crash between the chain write and the index update makes the
    runner RE-ANCHOR the SAME payload, so the chain records a DUPLICATE identical anchor at a
    LATER block. The beacon's "committed before the beacon block" grind check must treat that
    idempotent re-anchor as the ORIGINAL, timely commitment — otherwise an honestly-timely
    anchor is permanently mis-read as a grind. So return the EARLIEST block among anchors whose
    payload EQUALS the committed payload.

    This does NOT weaken genuine late-anchor detection: matching by the COMMITTED PAYLOAD (not
    merely the epoch prefix) means a DIFFERENT/later payload is a genuinely-new commitment whose
    OWN earliest block still applies, and a first anchor genuinely after the beacon block keeps
    that later block. Only `read_anchor_block` (the inclusion-BLOCK selection for the grind
    check) uses this; the digest legs stay on the LAST-matching payload, unchanged.
    """
    prefix = f"{domain}:{netuid}:{epoch_id}:"
    materialized = list(entries or ())
    committed: bytes | None = None
    for payload, _block in materialized:
        try:
            text = bytes(payload).decode("ascii")
        except (UnicodeDecodeError, TypeError, ValueError):
            continue
        if text.startswith(prefix):
            committed = bytes(
                payload
            )  # LAST matching wins (mirrors parse_anchor_digest)
    if committed is None:
        return None
    return min(
        int(block) for payload, block in materialized if bytes(payload) == committed
    )


def parse_anchor_digest(
    payloads: object, *, netuid: int, epoch_id: int, domain: str
) -> str | None:
    """Extract the anchored log_digest for `(netuid, epoch_id)` from raw payloads.

    The authority anchors a domain-tagged commitment
    ``<domain>:<netuid>:<epoch_id>:<log_digest>`` (ascii). Given an iterable of
    recorded commitment payload bytes (a fake chain's full anchor list, or the
    single latest commitment a real chain returns), this returns the log_digest of
    the LAST matching anchor (anchoring is idempotent, so there is at most one), or
    None to say POSITIVELY that none matched. Shared by every adapter's
    ``read_anchor`` so the payload contract lives in one place (the chain layer never
    imports the authority package).
    """
    materialized = list(payloads or ())
    idx = _last_matching_anchor_index(
        materialized, netuid=netuid, epoch_id=epoch_id, domain=domain
    )
    if idx is None:
        return None
    prefix = f"{domain}:{netuid}:{epoch_id}:"
    return bytes(materialized[idx]).decode("ascii")[len(prefix) :]


def synthetic_block_hash(block_number: int) -> str:
    """The deterministic synthetic block HASH for a height.

    The round-6 sampling beacon is ``block_hash(close_block + K)`` — the hash of a
    FUTURE FINALIZED block, fixed by the epoch's ``close_block`` (recorded in the
    anchored log) and a fixed confirmation depth K. Because it depends ONLY on the
    epoch's close_block, an authority that RE-ANCHORS the same payload in a later block
    cannot reroll it (that closes the round-5 inclusion-block hole: each re-anchor there
    minted a fresh inclusion-block beacon to grind against).

    Report/in-memory mode has no real substrate block hash, so this derives one
    DETERMINISTICALLY from the block HEIGHT: ``sha256("block:<n>")``. `InMemoryChain`
    and the chainsim's `HttpChainAdapter` both use THIS function, so report-mode and
    in-memory agree byte-for-byte. The live bittensor adapter returns the REAL
    substrate block hash instead (still a future-finalized-block hash). This is a
    general height->hash map; a block not yet produced returns None from the adapters'
    ``block_hash`` (never a hash for a block that does not exist yet).
    """
    return hashlib.sha256(f"block:{block_number}".encode("ascii")).hexdigest()


@runtime_checkable
class EpochAnchorReadable(Protocol):
    """OPTIONAL ChainAdapter extension: read a per-epoch anchored digest back.

    The genuinely-independent third leg of the tamper-evidence chain
    (the project design record §5): after the authority anchors an epoch log's
    digest on chain, a validator reads it BACK from the chain and checks it equals
    the pointer/bytes digest. Every real adapter (InMemoryChain, HttpChainAdapter to
    the sim, BittensorChainAdapter to the live commitment pallet) implements it over
    its own storage; the `ChainAdapterAnchorReader` (weightsetter.shared_snapshot)
    wraps any adapter that satisfies this.

    Contract (mirrors SubmittedWeightsReader):
    - returns the 64-hex anchored digest for `(netuid, epoch_id)`;
    - returns None to say POSITIVELY the chain holds NO such anchor (a substituted
      pointer is then caught);
    - RAISES on a read/transport failure — never substitutes None for a failed read
      (an unreadable chain must HOLD, not be mistaken for "no anchor").
    """

    def read_anchor(self, *, netuid: int, epoch_id: int, domain: str) -> str | None: ...


@runtime_checkable
class EpochAnchorBlockReadable(Protocol):
    """OPTIONAL ChainAdapter extension: read an epoch anchor's INCLUSION BLOCK NUMBER.

    an internal review (finding #2, step 3). The round-6 beacon is
    ``block_hash(close_block + K)`` — independent of when the authority anchors — but the
    auditor must still confirm the item set was COMMITTED before the beacon block could be
    known: the anchor's inclusion block must be ``<= close_block + K``. This returns that
    inclusion BLOCK NUMBER (not entropy), using the SAME domain-tag matching as
    ``read_anchor``.

    Contract (mirrors `EpochAnchorReadable`):
    - returns the block number the epoch's anchor commitment was recorded at;
    - returns None to say POSITIVELY the chain holds NO such anchor yet;
    - RAISES on a read/transport failure — never substitutes None for a failed read
      (an unreadable chain must HOLD, not be mistaken for "no anchor").
    """

    def read_anchor_block(
        self, *, netuid: int, epoch_id: int, domain: str
    ) -> int | None: ...


@runtime_checkable
class HistoricalEpochAnchorReadable(Protocol):
    """OPTIONAL extension: verify an anchor at its claimed inclusion block.

    The Commitments pallet keeps one current slot per ``(netuid, account)``. A
    backfill therefore has to query archive state at the pointer's block instead
    of reading head state. Implementations return a digest only when the payload
    matches and the record's stored inclusion block equals ``block_number``;
    unavailable/pruned state raises ``ChainStateUnavailable``.
    """

    def read_anchor_at(
        self,
        *,
        netuid: int,
        epoch_id: int,
        domain: str,
        block_number: int,
    ) -> str | None: ...


@runtime_checkable
class CommitmentRecordReadable(Protocol):
    """OPTIONAL extension: read the authority's raw mutable commitment record.

    ``block_number=None`` returns the current record.  A concrete block number
    returns the slot state from that archive block, including the record's own
    original inclusion height.  Returning ``None`` positively means the slot was
    empty at that state; transport, decoding, and pruned-history failures raise
    :class:`ChainStateUnavailable`.

    Keeping payload and inclusion height in one result prevents a caller from
    accidentally pairing two generations of the one-slot pallet record.  It also
    lets non-epoch protocols retain their existing public payload bytes instead
    of forcing them into the epoch ``domain:netuid:id:digest`` envelope.
    """

    def read_commitment_record(
        self, *, netuid: int, block_number: int | None = None
    ) -> ChainCommitmentRecord | None: ...


@runtime_checkable
class BlockHashReadable(Protocol):
    """OPTIONAL ChainAdapter extension: read a block HASH by height (round-6 #2).

    The un-grindable sampling-beacon source: ``block_hash(close_block + K)``. The
    authority cannot precompute the hash of a block that has not been produced yet, and
    the beacon block is fixed by the epoch's ``close_block`` (recorded in the anchored
    log), so re-anchoring the same payload in a later block cannot reroll it.

    Contract:
    - returns the block's hash (64-hex; report/in-memory uses `synthetic_block_hash`,
      the live chain uses the real substrate hash) for a block that HAS been produced;
    - returns None to say POSITIVELY the block is NOT YET produced (``n`` > current head)
      — the beacon is then simply not finalized, so the auditor HOLDS and retries later;
    - RAISES on a read/transport failure — an unreadable chain HOLDS, never a substituted
      None for a produced block.
    """

    def block_hash(self, block_number: int) -> str | None: ...


@runtime_checkable
class FinalizedBlockReadable(Protocol):
    """OPTIONAL adapter extension: the latest consensus-finalized block height.

    A best/current head is not finality.  Production sampling and epoch-close
    reads use this height so a short reorg cannot change a supposedly immutable
    close snapshot or beacon.  Report/in-memory adapters may treat their own
    deterministic head as finalized.
    """

    def finalized_block(self) -> int: ...


@runtime_checkable
class EpochBoundaryReadable(Protocol):
    """OPTIONAL adapter extension for chain-native subnet epoch boundaries.

    The Bittensor epoch schedule is stateful: tempo changes reset its anchor,
    owners may trigger an early epoch, and per-block capacity may defer it.  A
    zero-offset ``head // tempo`` grid is therefore not authoritative.

    Implementations return only archive-proven transitions.  ``None`` means the
    requested epoch has not closed at the finalized head; an unreadable/pruned or
    internally inconsistent history raises :class:`ChainStateUnavailable`.
    """

    def latest_closed_epoch(self, *, netuid: int) -> EpochBoundary | None: ...

    def epoch_close_block(self, *, netuid: int, epoch_id: int) -> int | None: ...


@runtime_checkable
class BlockTimeReadable(Protocol):
    """OPTIONAL ChainAdapter extension: read a block's WALL-CLOCK TIME by height (round-9 #6).

    The auditor binds ``EpochLog.created_at`` to the epoch's CLOSE BLOCK time so a BACKDATED
    ``created_at`` (which would keep an expired PODIUM/CROWN reward window active)
    is caught: ``created_at`` must agree with ``block_time(close_block)`` within a tolerance.

    Contract:
    - returns the block's UTC timestamp for a block whose time the chain can determine
      (report/in-memory derive it from the deterministic block clock; a live chain reads the
      block's on-chain timestamp);
    - returns None when the time cannot be determined (block not produced / no clock) — the
      auditor then fails CLOSED to INCONCLUSIVE, never a PASS;
    - RAISES on a read/transport failure — an unreadable chain HOLDs, never a substituted time.
    """

    def block_time(self, block_number: int) -> datetime | None: ...


@runtime_checkable
class BlockPinnedNeuronsReadable(Protocol):
    """OPTIONAL adapter extension for a metagraph at an exact chain block.

    Epoch finalization must not relabel current-head registration/stake state as
    if it came from the epoch-close block. Implementations either return the exact
    historical view or raise ``ChainStateUnavailable`` (for example on a pruned
    endpoint); they never silently fall back to the head snapshot.
    """

    def neurons_at(self, block_number: int) -> list[ChainNeuron]: ...


@runtime_checkable
class BurnUidReadable(Protocol):
    """OPTIONAL adapter extension for the subnet owner's current registered uid.

    The empty-epoch convergence vector pays 100% to this uid.  It must therefore
    come from chain state, not a shared authority/validator config value that can
    be self-consistently wrong.  Implementations raise ``ChainStateUnavailable``
    when the owner hotkey or its uid cannot be read; they never guess uid 0.
    """

    def get_burn_uid(self) -> int: ...


def resolve_burn_uid(chain: object, *, report_fallback: int | None = None) -> int:
    """Resolve and validate the canonical empty-epoch recipient.

    A chain adapter that implements :class:`BurnUidReadable` is authoritative; a
    failed chain read is propagated and is never masked by config.  The optional
    fallback exists only for dependency-free report/test adapters that have no
    subnet-owner registry.  Production's Bittensor adapter implements the read.
    """
    reader = getattr(chain, "get_burn_uid", None)
    if callable(reader):
        uid = reader()
    elif report_fallback is not None:
        uid = report_fallback
    else:
        raise ChainStateUnavailable(
            "the chain adapter cannot resolve the subnet-owner burn uid and no "
            "report-mode fallback was configured"
        )
    if isinstance(uid, bool):
        raise ChainStateUnavailable("chain returned a boolean instead of a burn uid")
    try:
        resolved = int(uid)
    except (TypeError, ValueError) as exc:
        raise ChainStateUnavailable(
            f"chain returned an invalid burn uid: {uid!r}"
        ) from exc
    if resolved < 0:
        raise ChainStateUnavailable(f"chain returned a negative burn uid: {resolved}")
    return resolved


@runtime_checkable
class ChainAdapter(Protocol):
    def current_block(self) -> int: ...

    def neurons(self) -> list[ChainNeuron]: ...

    def refresh(self) -> None:
        """Refresh the cached metagraph snapshot (throttling is the adapter's job).

        MUST NOT raise on transport/decoding failure: the previous snapshot is
        kept and the failure is reported through `has_fresh_snapshot()`.
        """
        ...

    def has_fresh_snapshot(self, now: float, max_age_seconds: float) -> bool:
        """Is the cached snapshot usable at `now`?

        Contract:
        - False when the adapter has NEVER successfully refreshed;
        - False when the last successful refresh is older than max_age_seconds;
        - True otherwise.
        `now` is WALL-CLOCK EPOCH SECONDS (`time.time()`) — the one clock family
        this surface speaks, since the timestamp crosses a process boundary.
        Adapters whose state is authoritative in-process (InMemoryChain) are
        trivially fresh and ignore both arguments.
        """
        ...

    def snapshot_age(self, now: float) -> float | None:
        """Seconds since the last successful refresh; None if never refreshed.

        `now` is wall-clock epoch seconds, as in has_fresh_snapshot.
        """
        ...

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        """Submit a weight vector.

        `hotkeys` is the OPTIONAL intended uid -> hotkey binding the vector was
        scored against. A live-chain adapter uses it to reject the complete attempt
        if a uid was deregistered/recycled between scoring and submission; it never
        drops and renormalizes a subset. Adapters without that hazard (the sim,
        in-memory fakes) accept and ignore it.
        """
        ...

    async def anchor_commitment(self, payload: bytes) -> str:
        """Anchor <=128 payload bytes on chain; returns a transaction/extrinsic id."""
        ...


@dataclass
class InMemoryChain:
    """Deterministic fake chain for local runs and tests.

    Records everything: `weight_calls` is every accepted set_weights vector with
    its block; `anchored` is every payload in order. Time is advanced explicitly
    with advance_blocks() — nothing moves on its own.
    """

    _neurons: list[ChainNeuron] = field(default_factory=list)
    _block: int = 1
    tempo: int = 100
    #: set_weights fails while block <= last_set + tempo (mirrors the tempo gate)
    weight_calls: list[tuple[int, dict[int, float]]] = field(default_factory=list)
    anchored: list[bytes] = field(default_factory=list)
    #: Inclusion block per entry of `anchored`. The
    #: auditor must confirm the item set was COMMITTED before the beacon block could be
    #: known (anchor_block <= close_block + K) — so `read_anchor_block` needs the block
    #: each anchor landed at. Kept SEPARATE from `anchored` (which STAYS a list[bytes]
    #: so read_anchor / parse_anchor_digest and every other iterator over it are
    #: unchanged); the two lists advance together in `anchor_commitment`.
    _anchor_blocks: list[int] = field(default_factory=list)
    _last_weight_block: int = -(10**9)
    #: set to a string to make the next set_weights fail with that message
    fail_next_set_weights: str | None = None
    #: block->time clock anchor: `(block, utc_time)` pins one block to a
    #: wall-clock time; `block_time(n)` extrapolates linearly at `block_seconds`. None ⇒ the
    #: fake exposes NO block clock, so `block_time` returns None and the auditor's created_at
    #: binding fails CLOSED to INCONCLUSIVE (never a PASS). Tests set it so the honest
    #: close_block time equals the log's created_at.
    block_time_anchor: tuple[int, datetime] | None = None
    block_seconds: float = 12.0

    def current_block(self) -> int:
        return self._block

    def neurons(self) -> list[ChainNeuron]:
        return list(self._neurons)

    def refresh(self) -> None:  # snapshot is always current in the fake
        return None

    # -- freshness (trivially fresh: this adapter IS the chain) -------------------

    def has_fresh_snapshot(self, now: float, max_age_seconds: float) -> bool:
        """Always True — in-process state cannot be stale relative to itself."""
        return True

    def snapshot_age(self, now: float) -> float | None:
        """Always 0.0 — see has_fresh_snapshot."""
        return 0.0

    @property
    def last_refresh_error(self) -> str | None:
        """Always None — refresh() cannot fail here."""
        return None

    @property
    def last_successful_refresh(self) -> float | None:
        """Always None: there is no fetch to timestamp.

        Do NOT read this as "never refreshed" — `has_fresh_snapshot()` is the
        freshness contract; this property exists only for surface uniformity.
        """
        return None

    # -- submitted-weights read (SubmittedWeightsReader) -------------------------

    def submitted_weights(self, hotkey: str) -> SubmittedWeights | None:
        """The latest accepted vector, with the block it was recorded at.

        This fake has a SINGLE writing identity (`set_weights` carries no hotkey),
        so it answers for whatever hotkey is asked. None means nothing has ever
        been accepted — a positive "no weights on chain", not a failed read.
        """
        if not self.weight_calls:
            return None
        block, vector = self.weight_calls[-1]
        return SubmittedWeights(weights=dict(vector), block=block)

    # -- anchor read (EpochAnchorReadable) --------------------------------------

    def read_anchor(self, *, netuid: int, epoch_id: int, domain: str) -> str | None:
        """The anchored digest for `(netuid, epoch_id)`, parsed from `self.anchored`.

        A REAL read of what this fake actually recorded (never a None-skip): the
        third verification leg is genuinely exercised in report/tests.
        """
        return parse_anchor_digest(
            self.anchored, netuid=netuid, epoch_id=epoch_id, domain=domain
        )

    # -- anchor inclusion-block read (EpochAnchorBlockReadable) ------------------

    def read_anchor_block(
        self, *, netuid: int, epoch_id: int, domain: str
    ) -> int | None:
        """The INCLUSION BLOCK of `(netuid, epoch_id)`'s anchor.

        Returns the EARLIEST block among anchors whose payload EQUALS the COMMITTED anchor
        (the last-matching payload `read_anchor` selects). The auditor uses this to confirm the
        item set was committed BEFORE the beacon block could be known
        (`anchor_block <= close_block + K`); an idempotent recovery RE-ANCHOR of the same payload
        (a crash between the chain write and the index update) must NOT move the anchor block past
        the beacon. Returns None (a positive "no anchor yet") when nothing
        matches; a real read of what this fake recorded.
        """
        return earliest_reanchor_block(
            zip(self.anchored, self._anchor_blocks),
            netuid=netuid,
            epoch_id=epoch_id,
            domain=domain,
        )

    def read_anchor_at(
        self,
        *,
        netuid: int,
        epoch_id: int,
        domain: str,
        block_number: int,
    ) -> str | None:
        """Emulate one-slot archive state at one exact inclusion block.

        Multiple writes to the same slot in one block collapse to the LAST value,
        matching substrate storage semantics. A receipt for an earlier same-block
        write therefore fails, which is why the challenge service fences writes
        onto strictly increasing finalized blocks.
        """
        payloads = [
            payload
            for payload, block in zip(self.anchored, self._anchor_blocks)
            if block == block_number
        ]
        if not payloads:
            return None
        return parse_anchor_digest(
            [payloads[-1]], netuid=netuid, epoch_id=epoch_id, domain=domain
        )

    def read_commitment_record(
        self, *, netuid: int, block_number: int | None = None
    ) -> ChainCommitmentRecord | None:
        """Return the raw one-slot record at head or one deterministic block.

        The in-memory fake journals writes rather than storing snapshots.  To
        emulate substrate storage at block ``N``, select the last write at or
        before ``N``; its stored ``block`` remains the write's original inclusion
        height.  Same-block writes naturally collapse to the last append.
        """

        del netuid  # this single-subnet fake has one authority commitment lane
        if block_number is not None:
            if isinstance(block_number, bool) or block_number < 0:
                raise ValueError("block_number must be a non-negative integer")
            candidates = [
                index
                for index, included_at in enumerate(self._anchor_blocks)
                if included_at <= block_number
            ]
            if not candidates:
                return None
            index = candidates[-1]
        else:
            if not self.anchored:
                return None
            index = len(self.anchored) - 1
        return ChainCommitmentRecord(
            payload=bytes(self.anchored[index]), block=int(self._anchor_blocks[index])
        )

    def finalized_block(self) -> int:
        """The deterministic in-process head is final immediately."""
        return self.current_block()

    # -- block-hash read (BlockHashReadable) ------------------------------------

    def block_hash(self, block_number: int) -> str | None:
        """The synthetic hash of a PRODUCED block, else None.

        `synthetic_block_hash` for `n <= current_block()`, None for a block not yet
        produced (`n > current_block()`) — the beacon block is then simply not finalized
        yet, so the auditor HOLDS and retries. Uses the SAME derivation as the chainsim's
        `HttpChainAdapter`, so report-mode and in-memory agree byte-for-byte.
        """
        if block_number > self.current_block():
            return None
        return synthetic_block_hash(block_number)

    # -- block-time read (BlockTimeReadable) ------------------------------------

    def block_time(self, block_number: int) -> datetime | None:
        """The deterministic wall-clock time of a block from the clock anchor (round-9 #6).

        `anchor_time + (block_number - anchor_block) * block_seconds`, or None when no
        anchor is configured (the fake then exposes no clock, so the auditor's created_at
        binding is INCONCLUSIVE, never a false PASS).
        """
        if self.block_time_anchor is None:
            return None
        anchor_block, anchor_time = self.block_time_anchor
        return anchor_time + timedelta(
            seconds=(block_number - anchor_block) * self.block_seconds
        )

    def set_neurons(self, neurons: list[ChainNeuron]) -> None:
        self._neurons = list(neurons)

    def advance_blocks(self, n: int) -> None:
        if n < 0:
            raise ValueError("blocks only move forward")
        self._block += n

    def update_neuron(self, uid: int, **changes: object) -> None:
        self._neurons = [
            replace(n, **changes) if n.uid == uid else n  # type: ignore[arg-type]
            for n in self._neurons
        ]

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        del hotkeys  # the in-process fake has no recycle hazard to reconcile
        if self.fail_next_set_weights is not None:
            message, self.fail_next_set_weights = self.fail_next_set_weights, None
            return SetWeightsResult(success=False, block=self._block, message=message)
        if self._block <= self._last_weight_block + self.tempo:
            return SetWeightsResult(
                success=False, block=self._block, message="tempo gate: too soon"
            )
        self.weight_calls.append((self._block, dict(weights)))
        self._last_weight_block = self._block
        # Report the EXACT max-normalized u16 vector Bittensor 10.5 emits, not the
        # caller's pre-quantization float intent. VIDAIO first builds its stable
        # sum-grid, then the SDK max-normalizes that grid before the extrinsic.
        # This fake keeps the raw floats in `weight_calls` (its read-back store); the
        # submitted grid vector is what the publication commits to.
        return SetWeightsResult(
            success=True,
            block=self._block,
            submitted=max_normalize_u16(quantize_u16(dict(weights))),
        )

    async def anchor_commitment(self, payload: bytes) -> str:
        if len(payload) > 128:
            raise ValueError("chain payload must be <= 128 bytes")
        self.anchored.append(bytes(payload))
        # Record the inclusion block in lockstep so `read_anchor_block` can report it for
        # the anchor-before-beacon-block check. The two lists are only
        # ever appended together.
        self._anchor_blocks.append(self.current_block())
        return "0x" + hashlib.sha256(payload).hexdigest()[:16]
