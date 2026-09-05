"""Test doubles for the weight-setter suite (importable under importlib mode)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Sequence

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind, LocalFsStore
from vidaio.chain import (
    ChainNeuron,
    ChainStateUnavailable,
    PendingWeightReveal,
    InMemoryChain,
    SetWeightsResult,
    SubmittedWeights,
)
from vidaio.tokenomics import MinerSnapshot, TokenomicsConfig

T0 = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)

#: Shared authority constants for the wave-5 shared-snapshot / convergence suites.
NETUID = 85
SCORER = "scoring-1.0.0+abc123def456"
NOW = T0


class FakeSnapshots:
    """Test SnapshotProvider: a fixed miner list."""

    def __init__(self, miners: Sequence[MinerSnapshot]) -> None:
        self.miners = list(miners)

    def miner_snapshots(self) -> Sequence[MinerSnapshot]:
        return list(self.miners)


class FakePublicationInputs:
    """Test PublicationInputs: a fixed score-packet digest list."""

    def __init__(self, digests: Sequence[str] = ()) -> None:
        self.digests = list(digests)

    def score_packet_digests(self) -> Sequence[str]:
        return list(self.digests)


class Clock:
    """Deterministic injected clock (tz-aware)."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class _ChainProxy:
    """Base for ChainAdapter wrappers over an InMemoryChain.

    Deliberately does NOT forward `submitted_weights`: the base proxy models an
    adapter that cannot read a vector back off the chain, which must make every
    confirmation UNKNOWN. Doubles that CAN answer mix in
    `_VectorReadable` below.
    """

    def __init__(self, inner: InMemoryChain) -> None:
        self.inner = inner

    def current_block(self) -> int:
        return self.inner.current_block()

    def neurons(self) -> list[ChainNeuron]:
        return self.inner.neurons()

    def refresh(self) -> None:
        self.inner.refresh()

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        return await self.inner.set_weights(
            weights, version_key=version_key, hotkeys=hotkeys
        )

    async def anchor_commitment(self, payload: bytes) -> str:
        return await self.inner.anchor_commitment(payload)


class _VectorReadable:
    """Mixin: the optional `submitted_weights` read, answered from the fake chain.

    This is the surface that makes a CONFIRMED verdict possible at all — it
    reports the vector the inner chain actually accepted, plus its block.
    """

    inner: InMemoryChain

    def submitted_weights(self, hotkey: str) -> SubmittedWeights | None:
        return self.inner.submitted_weights(hotkey)


class RecyclingChain(_ChainProxy, _VectorReadable):
    """A chain that detects target churn and rejects before any rewritten write.

    The production adapter may not drop/requantize recycled targets: that would
    donate their fixed share to survivors. This fake proves the WeightSetter never
    publishes a locally rewritten subset after such a clean pre-write rejection.
    """

    def __init__(self, inner: InMemoryChain, *, recycled=()) -> None:
        super().__init__(inner)
        self.recycled = set(recycled)

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        del hotkeys
        affected = sorted(
            uid
            for uid, value in weights.items()
            if value > 0 and uid in self.recycled
        )
        if not affected:
            return await self.inner.set_weights(weights, version_key=version_key)
        return SetWeightsResult(
            success=False,
            block=self.inner.current_block(),
            message=f"target binding changed for recycled uids {affected}",
        )


class FixedSubmittedChain(_ChainProxy, _VectorReadable):
    """DIRECT-path double: set_weights ACCEPTS and reports a FIXED `submitted` u16.

    Round-6 an internal review Pairs with ReportedVectorChain (the RECOVERY-path double) to
    stand for the SAME accepted write exposed in DIFFERENT permitted representations:
    this reports `SetWeightsResult.submitted` as the raw u16 grid (the direct-success
    surface, service.py ~1130), while ReportedVectorChain reports the
    `submitted_weights()` readback as scale-equivalent floats (the recovery surface,
    service.py ~1811). The test asserts BOTH paths persist byte-identical canonical
    `vector_json`/`vector_digest`. The vector is pinned regardless of the caller's
    computed float vector so the two doubles describe one identical vector.
    """

    def __init__(self, inner: InMemoryChain, *, submitted: dict[int, int]) -> None:
        super().__init__(inner)
        self.submitted = dict(submitted)

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        result = await self.inner.set_weights(
            weights, version_key=version_key, hotkeys=hotkeys
        )
        if not result.success:
            return result
        return SetWeightsResult(
            success=True,
            block=result.block,
            message=result.message,
            submitted=dict(self.submitted),
        )


class HangingChain(_ChainProxy):
    """ChainAdapter wrapper whose first N set_weights calls hang (timeout path)."""

    def __init__(self, inner: InMemoryChain, hang_first: int) -> None:
        super().__init__(inner)
        self.hang_remaining = hang_first

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        if self.hang_remaining > 0:
            self.hang_remaining -= 1
            # A hung submit: the write never lands. set_weights is no longer
            # caller-timeout-bounded (chain #11), so the adapter raises a transport
            # timeout — the ambiguous outcome _submit handles — instead of hanging.
            raise TimeoutError("set_weights hung (ambiguous)")
        return await self.inner.set_weights(
            weights, version_key=version_key, hotkeys=hotkeys
        )


class LostResponseChain(_ChainProxy):
    """The an internal review ambiguity: the write LANDS, its response never comes back.

    The first set_weights is applied to the chain and then hangs, so the caller
    times out with no way to tell whether the extrinsic landed. The retry hits the
    chain's own tempo gate — evidence that SOME write occupies the window, which
    round 3 no longer accepts as proof that the write was ours: this adapter
    cannot report a vector, so the attempt's fate stays UNKNOWN.
    """

    def __init__(self, inner: InMemoryChain) -> None:
        super().__init__(inner)
        self.calls = 0

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        self.calls += 1
        result = await self.inner.set_weights(
            weights, version_key=version_key, hotkeys=hotkeys
        )
        if self.calls == 1:
            # The write LANDED (recorded on the inner chain above); its response is
            # lost. set_weights is no longer caller-timeout-bounded (chain #11), so the
            # adapter surfaces the lost response as a transport error — the SAME
            # ambiguous outcome _submit handles, without a 30s wall-clock sleep.
            raise TimeoutError("set_weights response lost in transit (ambiguous)")
        return result


class ConfirmingChain(LostResponseChain, _VectorReadable):
    """LostResponseChain that CAN read its vector back (and dates it)."""

    def last_weight_block(self, hotkey: str) -> int | None:
        for block, _vector in reversed(self.inner.weight_calls):
            return block
        return None


class DelayedConfirmationChain(LostResponseChain, _VectorReadable):
    """The vector read only starts answering after `readable_after` attempts.

    Models a chain that cannot be read at the instant the retry is prepared but
    can be once the tempo rejection comes back — the shape in which a tempo
    rejection after an ambiguous write is legitimately reconciled as a success,
    because the chain SHOWS us our own vector.
    """

    def __init__(self, inner: InMemoryChain, *, readable_after: int = 1) -> None:
        super().__init__(inner)
        self.reads = 0
        self.readable_after = readable_after

    def submitted_weights(self, hotkey: str) -> SubmittedWeights | None:
        self.reads += 1
        if self.reads <= self.readable_after:
            raise ChainStateUnavailable("weights storage query timed out")
        return self.inner.submitted_weights(hotkey)


class RejectingRetryChain(_ChainProxy, _VectorReadable):
    """First write LANDS and hangs; every retry is rejected 401-style.

    The review scenario that abandoned a live intent on the spot: the synchronous
    rejection belongs to the RETRY, not to the ambiguous write before it.
    `weights_readable` starts False (the chain cannot be read while the attempt
    runs) and the test flips it to prove the intent is later confirmed by vector
    match — and published.
    """

    def __init__(
        self, inner: InMemoryChain, *, message: str = "HTTP 401: invalid token"
    ) -> None:
        super().__init__(inner)
        self.calls = 0
        self.message = message
        self.weights_readable = False

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        self.calls += 1
        if self.calls == 1:
            result = await self.inner.set_weights(
                weights, version_key=version_key, hotkeys=hotkeys
            )
            # The write LANDED (recorded on the inner chain above); its response is
            # lost. set_weights is no longer caller-timeout-bounded (chain #11), so the
            # adapter surfaces the lost response as a transport error — the SAME
            # ambiguous outcome _submit handles, without a 30s wall-clock sleep.
            raise TimeoutError("set_weights response lost in transit (ambiguous)")
            return result
        return SetWeightsResult(
            success=False, block=self.inner.current_block(), message=self.message
        )

    def submitted_weights(self, hotkey: str) -> SubmittedWeights | None:
        if not self.weights_readable:
            raise ChainStateUnavailable("weights storage query timed out")
        return self.inner.submitted_weights(hotkey)


class DenyingChain(_ChainProxy, _VectorReadable):
    """An adapter that POSITIVELY reports we have never set weights.

    The only shape that may lead to a terminal `abandoned`: a fresh read
    answering the question rather than failing to. `submitted_weights` returns
    None (no record on chain), which is an answer, not a failure.
    """

    def last_weight_block(self, hotkey: str) -> int | None:
        return None

    def submitted_weights(self, hotkey: str) -> SubmittedWeights | None:
        return None


class OverwrittenVectorChain(_ChainProxy):
    """Reports a DIFFERENT vector than ours, recorded at `set_block`.

    "A later intent B landed, so the chain now holds B's vector" — which says
    nothing about whether A ever landed (it may have been overwritten). A must be
    neither confirmed nor denied on this evidence.
    """

    def __init__(
        self, inner: InMemoryChain, *, weights: dict[int, float], block: int | None
    ) -> None:
        super().__init__(inner)
        self.reported = dict(weights)
        self.block = block

    def submitted_weights(self, hotkey: str) -> SubmittedWeights | None:
        return SubmittedWeights(weights=dict(self.reported), block=self.block)


class ReportedVectorChain(_ChainProxy):
    """Reports a FIXED vector back off the chain at `block` — a real u16 read.

    Unlike OverwrittenVectorChain (whose report is a DIFFERENT vector, to force an
    UNKNOWN), this reports a vector that MATCHES an intent under `weights_match` yet
    is not byte-equal to its float form: the shape that proves a recovery
    confirmation reconciles the stored row to the chain's EXACT u16 before it is
    published, instead of anchoring the pre-quantization float.
    """

    def __init__(
        self, inner: InMemoryChain, *, weights: dict[int, float], block: int | None
    ) -> None:
        super().__init__(inner)
        self.reported = dict(weights)
        self.block = block

    def submitted_weights(self, hotkey: str) -> SubmittedWeights | None:
        return SubmittedWeights(weights=dict(self.reported), block=self.block)


class VectorReadingChain(_ChainProxy, _VectorReadable):
    """A plain chain proxy that CAN read its own weight vector back."""


class CommitRevealChain(_ChainProxy):
    """Accepts a CRv4 commit, then exposes its vector only after ``reveal``.

    This models the pinned SDK with MEV protection disabled: ``set_weights``
    finalizes the timelocked commitment, while active ``Weights`` and
    ``LastUpdate`` remain unchanged until the chain's automatic reveal round.
    """

    def __init__(self, inner: InMemoryChain) -> None:
        super().__init__(inner)
        self.calls = 0
        self.pending = False
        self.staged: dict[int, float] | None = None
        self.staged_block: int | None = None

    def weight_commit_pending(self, hotkey: str) -> bool:  # noqa: ARG002
        return self.pending

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        del version_key, hotkeys
        from vidaio.tokenomics.quantize import max_normalize_u16, quantize_u16

        self.calls += 1
        self.pending = True
        self.staged = {
            uid: float(value)
            for uid, value in max_normalize_u16(quantize_u16(weights)).items()
        }
        self.staged_block = self.inner.current_block()
        return SetWeightsResult(
            success=False,
            block=self.staged_block,
            message="commit finalized; awaiting reveal",
            submitted=dict(self.staged),
            pending_reveal=True,
        )

    def submitted_weights(self, hotkey: str) -> SubmittedWeights | None:  # noqa: ARG002
        if self.pending:
            raise PendingWeightReveal("weight commit pending reveal")
        if self.staged is None:
            return None
        return SubmittedWeights(weights=dict(self.staged), block=self.staged_block)

    def reveal(self) -> None:
        if self.staged is None or self.staged_block is None:
            raise AssertionError("no staged weight commit")
        self.pending = False


class RefreshCountingChain(_ChainProxy, _VectorReadable):
    """Counts refresh() calls — the pre/post-write snapshot distinction (#10)."""

    def __init__(self, inner: InMemoryChain) -> None:
        super().__init__(inner)
        self.refreshes = 0

    def refresh(self) -> None:
        self.refreshes += 1
        self.inner.refresh()

    def last_weight_block(self, hotkey: str) -> int | None:
        for block, _vector in reversed(self.inner.weight_calls):
            return block
        return None


class StaleChainConfirms(_ChainProxy):
    """The an internal review false negative, exactly.

    The write LANDS and its response is lost. The adapter's cached view is the
    PRE-WRITE one and only `refresh()` moves it forward, so a confirmation that
    does not refresh sees nothing — "no weights" — and abandons a vector that is
    live on chain. The retry is tempo-rejected, as the real chain would.

    Both reads are refresh-gated: the block AND (round 3) the vector itself.
    """

    def __init__(self, inner: InMemoryChain) -> None:
        super().__init__(inner)
        self.calls = 0
        #: what a NON-refreshing reader would see: the pre-write state
        self._visible_weight_block: int | None = None
        self._visible_vector: dict[int, float] | None = None

    def refresh(self) -> None:
        self.inner.refresh()
        for block, vector in reversed(self.inner.weight_calls):
            self._visible_weight_block = block
            self._visible_vector = dict(vector)
            return

    def last_weight_block(self, hotkey: str) -> int | None:
        return self._visible_weight_block

    def submitted_weights(self, hotkey: str) -> SubmittedWeights | None:
        if self._visible_vector is None:
            return None
        return SubmittedWeights(
            weights=dict(self._visible_vector), block=self._visible_weight_block
        )

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        self.calls += 1
        result = await self.inner.set_weights(
            weights, version_key=version_key, hotkeys=hotkeys
        )
        if self.calls == 1:
            # The write LANDED (recorded on the inner chain above); its response is
            # lost. set_weights is no longer caller-timeout-bounded (chain #11), so the
            # adapter surfaces the lost response as a transport error — the SAME
            # ambiguous outcome _submit handles, without a 30s wall-clock sleep.
            raise TimeoutError("set_weights response lost in transit (ambiguous)")
        return result


class StaleChain(_ChainProxy):
    """ChainAdapter with the freshness surface, reporting an unusable snapshot."""

    def __init__(self, inner: InMemoryChain, *, fresh: bool = False) -> None:
        super().__init__(inner)
        self.fresh = fresh
        self.freshness_calls: list[tuple] = []

    def has_fresh_snapshot(self, now: float, max_age: float) -> bool:
        self.freshness_calls.append((now, max_age))
        return self.fresh


class HangingAnchorChain(_ChainProxy):
    """set_weights works; anchor_commitment hangs until `anchor_ok` is set."""

    def __init__(self, inner: InMemoryChain) -> None:
        super().__init__(inner)
        self.anchor_ok = False

    async def anchor_commitment(self, payload: bytes) -> str:
        if not self.anchor_ok:
            await asyncio.sleep(30)
        return await self.inner.anchor_commitment(payload)


class FailingSnapshots:
    """SnapshotProvider whose read raises — chain state is unavailable."""

    def miner_snapshots(self):
        raise RuntimeError("ChainStateUnavailable: no metagraph snapshot yet")


# --------------------------------------------------------------------------------------
# Wave-5 shared-snapshot / convergence doubles.
# --------------------------------------------------------------------------------------


class FakeScoringAuthorityClient:
    """Test ScoringAuthorityClient: serves real pointers produced by the authority.

    The pointers are made by the REAL finalize->index pipeline (see
    `authority_epoch` in the test module), so only the HTTP transport is faked. A
    missing pointer raises `SnapshotUnavailable`, exactly like the Http client's 404.
    """

    def __init__(self, latest=None, by_epoch=None) -> None:
        self._latest = latest
        self._by_epoch = dict(by_epoch or {})
        if latest is not None:
            self._by_epoch.setdefault(latest.epoch_id, latest)

    def set_latest(self, pointer) -> None:
        self._latest = pointer
        if pointer is not None:
            self._by_epoch[pointer.epoch_id] = pointer

    def latest_pointer(self):
        from vidaio.weightsetter.shared_snapshot import SnapshotUnavailable

        if self._latest is None:
            raise SnapshotUnavailable("no finalized pointer yet")
        return self._latest

    def pointer_for(self, epoch_id: int):
        from vidaio.weightsetter.shared_snapshot import SnapshotUnavailable

        pointer = self._by_epoch.get(epoch_id)
        if pointer is None:
            raise SnapshotUnavailable(f"no pointer for epoch {epoch_id}")
        return pointer


class PeerChain(_ChainProxy, _VectorReadable):
    """A readable chain that ALSO answers per-peer-hotkey on-chain vectors.

    Our own submissions go through the inner InMemoryChain (and are read back for
    our own hotkey); configured peer hotkeys report their own vectors. Unknown
    hotkeys report None (no record). A hotkey listed in `unreadable` raises, to model
    a peer whose vector cannot be read this epoch (never counted as a disagreement).
    """

    def __init__(
        self, inner, *, peer_vectors=None, unreadable=(), no_record=()
    ) -> None:
        super().__init__(inner)
        self.peer_vectors = dict(peer_vectors or {})
        self.unreadable = set(unreadable)
        self.no_record = set(no_record)

    def submitted_weights(self, hotkey: str):
        if hotkey in self.unreadable:
            raise ChainStateUnavailable(f"peer {hotkey} weights unreadable")
        if hotkey in self.no_record:
            return None  # POSITIVELY no on-chain vector for this peer yet
        if hotkey in self.peer_vectors:
            return SubmittedWeights(
                weights=dict(self.peer_vectors[hotkey]),
                block=self.inner.current_block(),
            )
        return self.inner.submitted_weights(hotkey)

    def neurons(self) -> list[ChainNeuron]:
        return self.inner.neurons()


# --------------------------------------------------------------------------------------
# The real epoch PRODUCER wired to test backends (no service/HTTP app) — used by both
# the shared-snapshot and convergence suites so the pointer/bytes/anchor a validator
# mirrors are exactly what the honest pipeline wrote.
# --------------------------------------------------------------------------------------


#: The finalizer now REQUIRES every nonzero-weight uid to carry an earning input that
#: EWMA-folds to its accumulate_score (#1). These builders couple `make_miner(uid)` and
#: `make_item(uid)` through this single, order-independent cycle score: the miner's
#: accumulate is its genesis fold and `make_item` attests the SAME value, so the produced
#: log is internally consistent. (A caller-passed `make_miner` score is cosmetic — no
#: weightsetter test asserts inter-uid weight magnitudes, only byte-identity/burn.)
_CYCLE_SCORE = 0.8


def make_miner(
    uid: int, score: float = _CYCLE_SCORE, track: str = "compression"
) -> MinerSnapshot:
    from vidaio.tokenomics.ewma import accumulate

    return MinerSnapshot(
        uid=uid,
        hotkey=f"hk{uid}",
        coldkey=f"ck{uid}",
        ip=f"10.0.0.{uid}",
        track=track,
        accumulate_score=accumulate(0.0, _CYCLE_SCORE, TokenomicsConfig().ewma_decay),
    )


def make_item(uid: int, store: LocalFsStore, *, seq: int = 0):
    """A ScoredItem backed by a REAL, RESOLVABLE committed-evidence bundle.

    ``seq`` is the committed dispatch ordering_key (== the packet's cycle_sequence). The
    committed dispatch key is MONOTONIC per uid across epochs, so a multi-epoch chain that
    folds a NEW cycle for the SAME uid in a later epoch must pass a strictly higher ``seq``
    (and gets a distinct item_id/bundle) — otherwise the auditor's cross-epoch REPLAY guard
    (round-22 #1) correctly flags re-folding the same ordering_key as a double-count. Default
    0 for a single-epoch fixture.

    review round-4 #1 made the auditor fail CLOSED: a nonzero-weight uid whose committed
    challenge evidence (bundle -> DAG_REVEAL -> commitment preimage carrying the
    pre-dispatch (track, dispatch_ordering_key)) can't be resolved is INCONCLUSIVE, not a
    pass-via-fallback. So an honest fixture must ship the full committed chain: a persisted
    POST_RETIREMENT bundle whose DAG_REVEAL IS the challenge-commitment preimage, resolvable
    by the auditor's StoredBundleSource. (The media artifact bytes are placeholders, so a
    rate>0 media recompute still can't reproduce them -> SKIP; the rate-0 earning re-fold is
    what verifies here.) Import lazily: authority/auditor pull in fastapi/uvicorn.
    """
    from vidaio.audit.bundle import LifecycleStage, build_bundle
    from vidaio.authority import ScoredItem
    from vidaio.auditor.service import persist_bundle
    from vidaio.challenge.commitment import ChallengeCommitment

    cycle_score = _CYCLE_SCORE
    track = "compression"
    dispatch_key = (
        seq  # == the packet's cycle_sequence; monotonic per uid across epochs
    )

    # A distinct item_id per (uid, seq) so a later epoch's cycle is a DISTINCT committed bundle,
    # never a re-fold of the same packet (item_id stays `i{uid}` at seq 0 for existing fixtures).
    challenge_id = "c1"
    item_id = f"i{uid}" if seq == 0 else f"i{uid}s{seq}"
    hotkey = f"hk{uid}"
    # The DAG_REVEAL IS the pre-dispatch commitment preimage binding (track, dispatch key)
    # BEFORE any score exists -> the auditor reads the committed fold order/track from here.
    dag_bytes = ChallengeCommitment.preimage_payload(
        f"asset-{item_id}",
        sha256_hex(b"dag-" + item_id.encode()),
        1,
        SCORER,
        track,
        dispatch_key,
    )
    # A REAL, FULLY-VALID ItemScore packet: carries the committed earning fields (score
    # + cycle_sequence) the auditor re-reads for the evidence-bound fold
    # AND satisfies the ItemScore contract so a rate>0 media pass parses it (a malformed
    # packet would FAIL the contract before recompute; here recompute-ability, not the
    # packet shape, is what a media sample turns on).
    packet = canonical_json_bytes(
        {
            "item_id": item_id,
            "challenge_id": challenge_id,
            "track": track,
            "score": cycle_score,
            "gate_passed": True,
            "violations": [],
            "skips": [],
            "miner_hotkey": hotkey,
            # Packet identity names the exact output archived in the bundle below.
            # Keeping these different used to be tolerated by this fixture, but
            # strict audit now (correctly) rejects that substitution shape.
            "content_digest": sha256_hex(f"out-{item_id}".encode()),
            "breakdown": {"kind": track, "final": cycle_score, "vmaf_threshold": 90.0},
            "metrics": {
                "compression_rate": 0.125,
                "vmaf": 93.42,
                "final_score": cycle_score,
            },
            "scorer_version": SCORER,
            "backend_versions": {},
            "pieapp_start_frame": None,
            "scoring_config_digest": sha256_hex(b"scoring config"),
            "canonicalization_plan_digest": sha256_hex(b"plan"),
            "cycle_sequence": dispatch_key,
            "excluded": False,
        }
    )
    bundle = build_bundle(
        challenge_id=challenge_id,
        item_id=item_id,
        miner_hotkey=hotkey,
        commitment_hash=sha256_hex(dag_bytes),
        stage=LifecycleStage.POST_RETIREMENT,
        challenge_input=store.put(
            f"in-{item_id}".encode(), ArtifactKind.CHALLENGE_INPUT
        ),
        miner_output=store.put(f"out-{item_id}".encode(), ArtifactKind.MINER_OUTPUT),
        manifest=store.put(
            canonical_json_bytes({"item": item_id}), ArtifactKind.MANIFEST
        ),
        score_packet=store.put(packet, ArtifactKind.SCORE_PACKET),
        reference_original=store.put(
            f"ref-{item_id}".encode(), ArtifactKind.REFERENCE_ORIGINAL
        ),
        dag_reveal=store.put(dag_bytes, ArtifactKind.DAG_REVEAL),
        scorer_version=SCORER,
        backend_versions={},
        created_at=NOW.isoformat(),
    )
    persist_bundle(
        store, bundle
    )  # store digest IS bundle_digest -> StoredBundleSource resolves it
    return ScoredItem(
        uid=uid,
        hotkey=hotkey,
        challenge_id=challenge_id,
        item_id=item_id,
        bundle_digest=bundle.bundle_digest(),
        packet_digest=sha256_hex(packet),
        committed_track=track,  # REQUIRED (#9)
        score=cycle_score,  # the same cycle score make_miner folded
        cycle_sequence=dispatch_key,
    )


class AuthorityHarness:
    """Real EpochFinalizer + LocalFsStore + InMemoryChain anchor + EpochIndex."""

    def __init__(self, tmp_path, *, burn_uid: int = 0) -> None:
        from vidaio.authority import EpochIndex
        from vidaio.authority.finalizer import EpochFinalizer

        self.store = LocalFsStore(tmp_path / "audit")
        self.chain = InMemoryChain()
        self.index = EpochIndex.open(tmp_path / "authority.db")
        self.finalizer = EpochFinalizer(TokenomicsConfig(), scorer_version=SCORER)
        self.burn_uid = burn_uid

    async def finalize(
        self,
        *,
        epoch_id: int,
        close_block: int,
        miners,
        items=None,
        now: datetime = NOW,
        anchor: bool = True,
        prior_accumulate=None,
        prior_log_digest=None,
    ):
        # Most weight-setter fixtures intentionally build inference-only epochs.
        from vidaio.authority import (
            EPOCH_LOG_MEMBER,
            build_audit_manifest,
            epoch_prefix,
        )
        from vidaio.authority.anchoring import anchor_epoch
        from vidaio.epoch import EpochLog

        prior_log = None
        latest = self.index.latest()
        if (
            prior_log_digest is not None
            and latest is not None
            and latest.log_digest == prior_log_digest
        ):
            prior_log = EpochLog.from_json(
                self.store.get_set_member(
                    epoch_prefix(latest.epoch_id),
                    EPOCH_LOG_MEMBER,
                    expected_digest=latest.log_digest,
                )
            )
        prior_fold_cursors = (
            prior_log.audit_manifest.fold_cursors if prior_log is not None else {}
        )
        prior_earning = {
            miner.uid: (miner.hotkey, float(miner.accumulate_score))
            for miner in (prior_log.miners if prior_log is not None else ())
        }

        # A CHAINED (non-genesis) epoch carries the prior epoch's per-uid accumulate as the
        # carry-in AND references the prior log's digest, so the earning re-fold verifies the
        # nonzero carry-in against the prior epoch.
        manifest = build_audit_manifest(
            items or (),
            store=self.store,
            prior_accumulate=prior_accumulate,
            prior_fold_cursors=prior_fold_cursors,
        )
        finalized = self.finalizer.finalize(
            epoch_id=epoch_id,
            close_block=close_block,
            snapshots=miners,
            burn_uid=self.burn_uid,
            audit_manifest=manifest,
            store=self.store,
            now=now,
            prior_log_digest=prior_log_digest,
            prior_earning=prior_earning,
            prior_fold_cursors=prior_fold_cursors,
        )
        self.index.record_finalized(finalized, finalized_at=now.isoformat())
        if anchor:
            await anchor_epoch(
                finalized, chain=self.chain, index=self.index, netuid=NETUID, now=now
            )
        return finalized

    def pointer(self, epoch_id: int):
        from vidaio.authority.api import pointer_from_record

        return pointer_from_record(self.index.get(epoch_id))

    def latest_pointer(self):
        from vidaio.authority.api import pointer_from_record

        return pointer_from_record(self.index.latest())

    def anchor_reader(self):
        from vidaio.weightsetter.shared_snapshot import InMemoryChainAnchorReader

        return InMemoryChainAnchorReader(self.chain)

    def provider(self, *, epoch_id=None, verify_anchor=True, anchor_reader=None):
        from vidaio.weightsetter.shared_snapshot import SharedSnapshotProvider

        pointer = (
            self.pointer(epoch_id) if epoch_id is not None else self.latest_pointer()
        )
        return SharedSnapshotProvider(
            client=FakeScoringAuthorityClient(latest=pointer),
            store=self.store,
            netuid=NETUID,
            anchor_reader=self.anchor_reader()
            if anchor_reader is None
            else anchor_reader,
            verify_anchor=verify_anchor,
        )

    def close(self) -> None:
        self.index.close()
