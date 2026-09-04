"""Shared fakes for the validator suite: neurons, challenge/miner/scoring clients.

Everything is deterministic and in-process; the fakes write/read real files so
the validator's digest verification runs for real.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from vidaio.chain.adapter import ChainNeuron
from vidaio.challenge import ChallengeAnchor, DispatchPayload
from vidaio.scoring import ItemScore
from vidaio.services.artifact_auth import (
    CallableHotkeySigner,
    MinerArtifactReceipt,
    ValidatorArtifactRequestReceipt,
)
from vidaio.services.miner_artifacts import (
    MinerArtifactColdStart,
    MinerPeerAddressError,
)
from vidaio.services.protocol import (
    MINER_ARTIFACT_AUTH_VERSION,
    MinerArtifactTaskRequest,
    MinerTaskRequest,
    MinerTaskResponse,
    ScorerIdentityUnavailable,
    ScoreRequest,
    ScoreResponse,
)
from vidaio.validator.inference import (
    ChallengeAlreadyTerminal,
    ChallengeItem,
    ChallengeOwnershipRefused,
    DispatchedChallenge,
)
from vidaio.validator.availability import (
    AvailabilityFailureReason,
    AvailabilityObservation,
    DispatchAttempt,
    build_availability_observation,
)


#: This validator's identity in the suite (its hotkey, in production) — stamped
#: on every /challenge/next and the ownership boundary of the orphan sweep.
VALIDATOR_IDENTITY = "5ValidatorSelf"
#: A SECOND validator sharing the same challenge service.
OTHER_VALIDATOR_IDENTITY = "5ValidatorOther"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mk_neuron(
    uid: int,
    *,
    hotkey: str | None = None,
    coldkey: str | None = None,
    ip: str | None = None,
    alpha_stake: float = 10.0,
    emission: float = 1.0,
    is_validator: bool = False,
    axon_port: int | None = None,
) -> ChainNeuron:
    return ChainNeuron(
        uid=uid,
        hotkey=hotkey if hotkey is not None else f"hk{uid}",
        coldkey=coldkey if coldkey is not None else f"ck{uid}",
        ip=ip if ip is not None else f"10.0.0.{uid}",
        alpha_stake=alpha_stake,
        emission=emission,
        is_validator=is_validator,
        axon_port=axon_port,
    )


class FakeChallengeClient:
    """Serves one deterministic challenge item per track, with real files.

    Also models the challenge service's resolve contract: `resolves` records every
    terminated challenge in order, a challenge can only be terminated ONCE (a
    second attempt raises ChallengeAlreadyTerminal, as the real 409 does), and
    `fail_resolve_ids` makes the call fail so the retry path can be exercised.

    OWNERSHIP: `/challenge/next` records the requesting
    validator's `owner` on the challenge, and `GET /challenges` both filters by
    it and NAMES it on every row — the two halves the validator's sweep needs to
    prove a challenge is its own. `supports_owner=False` models the service
    before that contract landed: the parameter is ignored and no row names an
    owner, so the sweep must expire nothing.

    `/challenge/{id}/resolve` ENFORCES that same ownership exactly as the real
    service does: an owned challenge resolved by anybody else raises (the real
    403 `not_owner`), which is what catches a validator that stamps its identity
    on production and then drops it on resolve.
    """

    def __init__(self, root: Path, *, supports_owner: bool = True) -> None:
        self.root = root
        self.fetches: list[str] = []
        self.fail_tracks: set[str] = set()
        #: (challenge_id, outcome) in call order
        self.resolves: list[tuple[str, str]] = []
        self.fail_resolve_ids: set[str] = set()
        self.terminal: set[str] = set()
        #: owners seen on /challenge/{id}/resolve, in call order
        self.resolve_owners: list[str] = []
        #: challenge_id -> recorded owner, SURVIVING resolution (the real service
        #: keeps the row; `dispatched` is only the not-yet-terminal view)
        self.recorded_owners: dict[str, str] = {}
        self._items: dict[str, ChallengeItem] = {}
        #: SERVICE-SIDE state for GET /challenges: challenge_id -> (track, age, owner).
        #: A challenge lands here when the service produces it, which is BEFORE
        #: the response reaches the validator — that gap is the lost-response
        #: blind spot the sweep exists to close.
        self.dispatched: dict[str, tuple[str, float, str]] = {}
        #: raise on list_dispatched (an old service / a flaky call)
        self.fail_listing = False
        self.list_calls: list[tuple[float, str]] = []
        #: owners seen on /challenge/next, in call order
        self.fetch_owners: list[str] = []
        self.supports_owner = supports_owner
        #: When False the service ACCEPTS `owner` but does not filter on it — the
        #: realistic mid-migration shape (and what an HTTP framework does with an
        #: unknown query parameter). The validator's own row-level ownership check
        #: is the thing under test then.
        self.filters_by_owner = True

    async def resolve_challenge(
        self, challenge_id: str, outcome: str, owner: str = ""
    ) -> None:
        self.resolve_owners.append(owner)
        if challenge_id in self.fail_resolve_ids:
            raise RuntimeError(f"challenge service down for {challenge_id}")
        recorded = self.recorded_owners.get(challenge_id, "")
        if recorded and owner != recorded:
            # The real service's 403 not_owner, typed exactly as HttpChallenge
            # Client types it. NOT ChallengeAlreadyTerminal: this is a PERMANENT
            # refusal — the row must not be dropped as if the challenge had
            # drained, and it must not be retried forever either (round-3 #5).
            raise ChallengeOwnershipRefused(
                f"not_owner: {challenge_id} belongs to {recorded!r},"
                f" resolve claimed {owner!r}"
            )
        if challenge_id in self.terminal:
            raise ChallengeAlreadyTerminal(
                f"challenge {challenge_id} is already terminal"
            )
        self.terminal.add(challenge_id)
        self.dispatched.pop(challenge_id, None)  # no longer dispatched
        self.resolves.append((challenge_id, outcome))

    async def list_dispatched(self, older_than_seconds: float, owner: str = ""):
        """GET /challenges?status=dispatched&older_than_seconds=N[&owner=](real shape)."""
        self.list_calls.append((older_than_seconds, owner))
        if self.fail_listing:
            raise RuntimeError("challenge service has no /challenges route")
        rows = []
        for cid, (track, age, row_owner) in sorted(self.dispatched.items()):
            if age < older_than_seconds:
                continue
            if not self.supports_owner:
                # An older service ignores the filter AND names no owner — the
                # unfiltered list the validator must refuse to act on.
                rows.append(
                    DispatchedChallenge(challenge_id=cid, track=track, age_seconds=age)
                )
                continue
            if owner and self.filters_by_owner and row_owner != owner:
                continue
            rows.append(
                DispatchedChallenge(
                    challenge_id=cid, track=track, age_seconds=age, owner=row_owner
                )
            )
        return rows

    def lose_response(
        self, challenge_id: str, *, track: str, age_seconds: float, owner: str = ""
    ) -> None:
        """Model a `/challenge/next` whose RESPONSE never arrived.

        The service produced and dispatched the challenge (so it is on the sweep
        list and its asset is checked out), but the validator never learned the
        id — there is no in-flight row and nothing to retry. `owner` is the
        validator the service attributed it to.
        """
        self.dispatched[challenge_id] = (track, age_seconds, owner)
        self.recorded_owners[challenge_id] = owner

    def item_for(self, track: str) -> ChallengeItem:
        if track not in self._items:
            ref = self.root / f"{track}-reference.bin"
            ref.write_bytes(f"reference:{track}".encode())
            inp = self.root / f"{track}-input.bin"
            inp.write_bytes(f"input:{track}".encode())
            self._items[track] = ChallengeItem(
                dispatch=DispatchPayload(
                    challenge_id=f"ch-{track}",
                    task_type=track,
                    input_ref=f"serve/ch-{track}",
                ),
                track=track,
                reference_path=str(ref),
                reference_digest=sha256_bytes(ref.read_bytes()),
                miner_input_path=str(inp),
                miner_input_digest=sha256_bytes(inp.read_bytes()),
                commitment_anchor=ChallengeAnchor(
                    netuid=85,
                    dispatch_ordering_key=1,
                    commitment_hash=sha256_bytes(f"commitment:{track}".encode()),
                    block=1,
                    block_hash="ab" * 32,
                ),
                params={"round": 1},
            )
        return self._items[track]

    async def next_challenge(self, track: str, owner: str = "") -> ChallengeItem:
        self.fetches.append(track)
        self.fetch_owners.append(owner)
        if track in self.fail_tracks:
            raise RuntimeError(f"challenge service down for {track}")
        item = self.item_for(track)
        # Service-side effect of a successful production: the challenge is
        # dispatched (age 0 — a live round's challenge is never old enough for
        # the sweep, which is exactly why the sweep is age-gated) and ATTRIBUTED
        # to the validator that asked for it.
        recorded = owner if self.supports_owner else ""
        self.dispatched[item.resolve_id] = (track, 0.0, recorded)
        self.recorded_owners[item.resolve_id] = recorded
        return item


class LegacyChallengeClient(FakeChallengeClient):
    """A challenge client from BEFORE the owner contract.

    All three routes keep their old signatures, so passing `owner` raises
    TypeError — the shape the validator must feature-detect. It must then fetch
    the old way, resolve the old way, and expire NOTHING, rather than sweeping an
    unattributed list.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(root, supports_owner=False)

    async def next_challenge(self, track: str) -> ChallengeItem:  # type: ignore[override]
        return await FakeChallengeClient.next_challenge(self, track)

    async def list_dispatched(self, older_than_seconds: float):  # type: ignore[override]
        return await FakeChallengeClient.list_dispatched(self, older_than_seconds)

    async def resolve_challenge(  # type: ignore[override]
        self, challenge_id: str, outcome: str
    ) -> None:
        return await FakeChallengeClient.resolve_challenge(self, challenge_id, outcome)


class FakeMinerClient:
    """Configurable per-uid behaviour: track, output bytes, timeouts, bad digests."""

    def __init__(self, outdir: Path) -> None:
        self.outdir = outdir
        #: uid -> warrant answer (any string; garbage stays unclassified)
        self.tracks: dict[int, str] = {}
        #: uid -> output bytes (default: unique per uid)
        self.outputs: dict[int, bytes] = {}
        self.warrant_fail_uids: set[int] = set()
        self.task_timeout_uids: set[int] = set()
        self.task_cold_start_uids: set[int] = set()
        self.task_unreachable_endpoint_uids: set[int] = set()
        self.bad_digest_uids: set[int] = set()
        #: uid -> the task_id this miner ECHOES instead of the dispatched one
        self.swap_task_ids: dict[int, str] = {}
        self.warrant_calls: list[int] = []
        self.task_calls: list[tuple[int, MinerTaskRequest]] = []

    async def probe_warrant(self, neuron: ChainNeuron) -> str:
        self.warrant_calls.append(neuron.uid)
        if neuron.uid in self.warrant_fail_uids:
            raise TimeoutError("warrant probe unreachable")
        return self.tracks[neuron.uid]

    async def submit_task(
        self, neuron: ChainNeuron, request: MinerTaskRequest
    ) -> MinerTaskResponse:
        self.task_calls.append((neuron.uid, request))
        if neuron.uid in self.task_timeout_uids:
            await asyncio.sleep(30)
        if neuron.uid in self.task_cold_start_uids:
            raise MinerArtifactColdStart("restart fence remained active")
        if neuron.uid in self.task_unreachable_endpoint_uids:
            raise MinerPeerAddressError("chain-advertised endpoint is undialable")
        payload = self.outputs.get(neuron.uid, f"output:{neuron.uid}".encode())
        out = self.outdir / f"out-{request.task_id.replace(':', '-')}.bin"
        out.write_bytes(payload)
        digest = sha256_bytes(payload)
        if neuron.uid in self.bad_digest_uids:
            digest = sha256_bytes(payload + b"tampered")
        # The swap case: a miner echoing a DIFFERENT task id, so the
        # validator would score this output under somebody else's dispatch.
        receipt = MinerArtifactReceipt(
            version=MINER_ARTIFACT_AUTH_VERSION,
            validator_hotkey=VALIDATOR_IDENTITY,
            miner_hotkey=neuron.hotkey,
            timestamp=1,
            nonce=f"{neuron.uid:032x}",
            input_size=Path(request.input_path).stat().st_size,
            metadata=MinerArtifactTaskRequest.from_local_request(request),
            request_signature="00" * 64,
            output_digest=digest,
            output_size=len(payload),
            processing_seconds="0",
            response_signature="00" * 64,
        )
        return MinerTaskResponse(
            task_id=self.swap_task_ids.get(neuron.uid, request.task_id),
            output_path=str(out),
            output_digest=digest,
            artifact_receipt=receipt.model_dump(mode="json"),
        )

    def availability_observation(
        self,
        neuron: ChainNeuron,
        request: MinerTaskRequest,
        reason: AvailabilityFailureReason,
        *,
        exception: BaseException | None = None,
        response: MinerTaskResponse | None = None,
        returned_task_id: str | None = None,
        observed_output_digest: str | None = None,
    ) -> AvailabilityObservation:
        del exception
        miner_receipt = None
        if response is not None and response.artifact_receipt is not None:
            miner_receipt = MinerArtifactReceipt.model_validate(
                response.artifact_receipt
            )
            request_receipt = miner_receipt.request_receipt()
            assert request_receipt is not None
        else:
            request_receipt = ValidatorArtifactRequestReceipt(
                version=MINER_ARTIFACT_AUTH_VERSION,
                validator_hotkey=VALIDATOR_IDENTITY,
                miner_hotkey=neuron.hotkey,
                timestamp=1,
                nonce=f"{neuron.uid:032x}",
                input_size=Path(request.input_path).stat().st_size,
                metadata=MinerArtifactTaskRequest.from_local_request(request),
                request_signature="00" * 64,
            )
        attempt = DispatchAttempt(
            uid=neuron.uid,
            miner_hotkey=neuron.hotkey,
            endpoint=f"fake://{neuron.ip}:{neuron.axon_port or 0}",
            challenge_id=request.task_id.rsplit(":", 1)[0],
            item_id=request.task_id,
            track=request.track,
            request=request_receipt,
        )
        return build_availability_observation(
            attempt=attempt,
            reason=reason,
            signer=CallableHotkeySigner(VALIDATOR_IDENTITY, lambda _payload: "00" * 64),
            returned_task_id=returned_task_id,
            observed_output_digest=observed_output_digest,
            miner_receipt=miner_receipt,
        )


class FakeScoringClient:
    """Returns a genuine ItemScore packet; per-hotkey scores configurable.

    `spoof` models a compromised or MITM'd scoring endpoint: it replaces named
    fields of the returned packet (hotkey -> {field: value}) while keeping the
    packet's own sha256 self-consistent — exactly the shape of a REPLAYED packet
    that belongs to a different miner/item/challenge and would previously have
    been accumulated for the current uid.
    """

    #: what a real worker stamps: <configured name>+<scoring-config digest>
    EFFECTIVE_SCORER_VERSION = "vidaio-scorer/1+0123456789ab"

    def __init__(self) -> None:
        self.default_score = 0.8
        self.scores: dict[str, float] = {}  # hotkey -> score
        self.fail_hotkeys: set[str] = set()
        self.corrupt_digest_hotkeys: set[str] = set()
        self.spoof: dict[str, dict[str, object]] = {}
        self.requests: list[ScoreRequest] = []
        self.effective_scorer_version = self.EFFECTIVE_SCORER_VERSION
        #: True = GET /healthz unreachable (the worker is not up yet). Discovery
        #: then pins nothing and retries, rather than failing the validator.
        self.identity_unavailable = False
        self.identity_calls = 0

    async def scorer_identity(self) -> str:
        """GET /healthz's `scorer_version` — the discovery half of the contract."""
        self.identity_calls += 1
        if self.identity_unavailable:
            raise ScorerIdentityUnavailable("scoring worker /healthz unreachable")
        return self.effective_scorer_version

    async def score(self, request: ScoreRequest) -> ScoreResponse:
        self.requests.append(request)
        hotkey = request.miner_hotkey or ""
        if hotkey in self.fail_hotkeys:
            raise RuntimeError("scoring worker down")
        # services.protocol contract: scorer_version is a caller ASSERTION —
        # absent/empty accepts whatever this worker is, a different value is 409.
        if (
            request.scorer_version
            and request.scorer_version != self.effective_scorer_version
        ):
            raise RuntimeError(
                f"409 scorer_version_mismatch: requested={request.scorer_version}"
                f" effective={self.effective_scorer_version}"
            )
        fields: dict[str, object] = {
            "item_id": request.item_id,
            "challenge_id": request.challenge_id,
            "track": request.track,
            "miner_hotkey": request.miner_hotkey,
            "content_digest": request.output_digest,
            "score": self.scores.get(hotkey, self.default_score),
            "gate_passed": True,
            # the worker always stamps its OWN version, never the caller's
            "scorer_version": self.effective_scorer_version,
        }
        fields.update(self.spoof.get(hotkey, {}))
        item = ItemScore(**fields)  # type: ignore[arg-type]
        packet = item.to_json()
        digest = sha256_bytes(packet.encode("utf-8"))
        if hotkey in self.corrupt_digest_hotkeys:
            digest = sha256_bytes(b"tampered")
        return ScoreResponse(item_score_json=packet, packet_digest=digest)
