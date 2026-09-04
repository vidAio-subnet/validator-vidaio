"""InferenceValidator — the synthetic scoring round loop (spec design spec §01, rebuilt).

One round: refresh chain (throttled) → filter + dedup miners → sync registry /
retention windows → warrant-probe unknown tracks → per track: fetch a challenge
item, dispatch to every eligible miner, verify output digests, response-time
dedup, score via the scoring worker → EWMA-accumulate → persist. Weights are NOT
set here — the weight-setter is its own supervised process (spec §13); the DB is
the only shared state.

Failure discipline (SN44 edge rules): every external await is with_timeout-
wrapped; idempotent reads (warrant probe, scoring — a pure recompute) get
retry_async. Miner-attributable timeout, transport, protocol, task-id, digest and
receipt failures fold zero only when a canonical observation carries the exact
validator-signed request and finalized anchor. Scoring-worker, audit-store,
chain/challenge and local-input trouble remains non-punitive. Duplicate penalties
remain separately backed by two miner-signed outputs and a CPU-recomputed witness.

The fixed TaskWarrant: a miner whose track is unknown (probe timeout, no record,
garbage value) is SKIPPED for the round with a structured log and the
`vidaio_validator_skipped_unknown_track_total` metric — never silently bucketed
as upscaling (the old validator.py:844-849 confirmed bug, design spec §07).

review service-review fixes carried here:

- #5  every challenge the round FETCHES is resolved. Fetching consumes an asset
      from the pool and leaves it `in_use` with its commitment unrevealed until
      POST /challenge/{id}/resolve arrives, so the round wraps its whole body in
      try/finally and drains a persisted in-flight table on success, failure,
      timeout and shutdown. A crashed round's stranded challenges are drained by
      `recover_inflight_challenges()` on the next startup.
- #6  a score packet is only accepted when it is BOUND to the request that asked
      for it: after the digest check the packet is parsed and its challenge_id,
      item_id, miner_hotkey, track, content_digest and scorer_version must equal
      what this validator sent. A replayed/foreign packet from a compromised or
      MITM'd scoring endpoint is rejected, counted as a scoring failure (which is
      NON-punitive to the miner), and never accumulated.
      ROUND 2: the binding must not be CIRCULAR. The item id came from the
      MINER's `MinerTaskResponse.task_id`, which was then "verified" against the
      packet built from that same miner-supplied value. The dispatched task id is
      now the only id that exists: a response echoing anything else is rejected as
      a miner failure, and ScoreRequest.item_id is always the VALIDATOR's id.
- #7  the packet BYTES + digest are persisted per (round, uid, item) and archived
      as SCORE_PACKET artifacts, so published weights are reproducible.
      `vidaio.validator.evidence.ScorePacketEvidence` serves them to the
      weight-setter's PublicationInputs instead of the empty-set sentinel.
      ROUND 2: when an audit store IS configured, a storage failure FAILS THE ITEM
      CLOSED — the score is not accumulated and no evidence row claims an archive
      that does not exist. DB-only operation (no store configured at all) stays
      allowed and is flagged at startup, in every round log and on a metric.
- #9  a round's whole OBSERVABLE state — the registry sync, the retention fold,
      the warrant-probe tracks, the EWMA folds, the packet evidence and the
      round-ledger stamp — is ONE BEGIN IMMEDIATE transaction (round 2: the
      registry half used to commit separately, so a weight-setter could read a
      hotkey reset from a round that then died). Registry effects are STAGED in
      memory during the round and applied by `commit_round`; the only pre-commit
      write is the `rounds` marker row, which readers ignore until it is stamped.
- #21 a FAILED chain refresh is never recorded as a successful one, and a stale
      or unavailable snapshot SKIPS the round with a structured reason instead of
      reporting a successful empty round.
- #22 health checks get their own per-thread connection: they run on the
      HealthServer's thread and must not touch the round loop's handle.

Round-2 findings also carried here:

- new-2 the scorer pin is DURABLE (validator DB) and REQUIRED. No pin -> the
      round is skipped with a structured reason rather than scored unbound; a
      discovered identity that disagrees with the persisted pin refuses to score
      until an operator acknowledges it (`validator.reset_scorer_pin`), so two
      scorers' results can never merge into one accumulator across a restart.
- new-4 the orphan sweep has an OWNERSHIP boundary: this validator's `identity`
      is sent as `owner` on every `/challenge/next`, and only challenges the
      service attributes to us are ever expired. A service (or a client) that
      cannot attribute ownership disables the sweep instead of expiring another
      validator's live challenge. The SAME identity is stamped on every
      `/challenge/{id}/resolve` — the service enforces ownership there too, so a
      validator that omits it is refused (403 `not_owner`) on its OWN challenges.

Round-3:

- #5  that owner is PERSISTED with the obligation (`inflight_challenges.owner`,
      migration 0004) and recovery resolves with the RECORDED owner, not the
      current config. An identity rotation between the fetch and the restart used
      to 403 the recovery forever — retried every round, asset stranded `in_use`,
      commitment never revealed. A rotation now logs a WARNING and still uses the
      fetcher's identity; a genuine 403 is counted
      (`vidaio_validator_challenge_resolve_forbidden_total`) and left for an
      operator instead of being retried against a boundary that cannot move.

Round-4:

- #5  "left for an operator" is now DURABLE: a genuine 403 PARKS the row
      (`inflight_challenges.parked_at`, migration 0005) instead of merely being
      skipped for one attempt — the retained row used to be re-selected by every
      later round AND every restart, producing the same impossible resolve,
      metric bump and WARNING forever. Parked rows are excluded from the
      drain/recovery selection but stay visible: the
      `vidaio_validator_parked_challenges` gauge, a startup recovery log line,
      and `miner_manager.parked_challenges()`. The operator's way out is
      `validator.unpark_challenges = true` at startup or the
      `unpark_challenges()` admin method, both of which return every parked row
      to the normal drain (a refusal that still stands re-parks on its next 403).

All dependencies are Protocols; the service runs a full deterministic round
in-process against fakes (InMemoryChain + fake clients) in tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import httpx
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, ConfigDict, Field

from vidaio.audit import ArtifactKind, ArtifactRef, AuditStore
from vidaio.chain.adapter import ChainAdapter, ChainNeuron
from vidaio.challenge import ChallengeAnchor, DispatchPayload
from vidaio.core import connect, retry_async, section, with_timeout
from vidaio.core.logging import log_fields
from vidaio.scoring import (
    InvalidDuplicateEvidence,
    ItemScore,
    ScoringConfig,
    duplicate_order_key,
    mint_duplicate_packet,
)
from vidaio.services.artifact_auth import (
    ArtifactClientAuth,
    CallableHotkeySigner,
    MinerArtifactReceipt,
)
from vidaio.services.base import BaseService
from vidaio.services.miner_artifacts import (
    MinerArtifactColdStart,
    MinerArtifactInputError,
    MinerArtifactIntegrityError,
    MinerArtifactProtocolError,
    MinerPeerAddressError,
    discard_downloaded_artifact,
    format_miner_peer_host,
    sign_miner_artifact_request_receipt,
    submit_miner_artifact,
)
from vidaio.services.protocol import (
    SHA256_HEX,
    MinerTaskRequest,
    MinerTaskResponse,
    ScorerIdentityMismatch,
    ScorerIdentityUnavailable,
    ScorerRuntimeContract,
    ScorerRuntimeMismatch,
    ScoreRequest,
    ScoreResponse,
    fetch_scorer_runtime_contract_async,
    require_matching_scorer_runtime_contract,
)
from vidaio.tokenomics import TokenomicsConfig, dedup_ip_key
from vidaio.validator import miner_manager
from vidaio.validator.availability import (
    AvailabilityFailureReason,
    AvailabilityObservation,
    DispatchAttempt,
    build_availability_observation,
)
from vidaio.validator.config import ValidatorConfig
from vidaio.validator.evidence import ScorePacketEvidence


class ChallengeItem(BaseModel):
    """Challenge-service response for one round item: the miner-facing dispatch
    payload plus the validator-private reference/input artifacts (local paths +
    digests — services.protocol local-first contract).

    Parsed directly from the challenge service's ChallengeNextResponse, whose
    `dispatch` field carries the producer-authored (and producer-leak-guarded)
    DispatchPayload verbatim — the validator never re-derives it. commitment_hash
    is ignored here: it is the service's own bookkeeping, not a round input.
    """

    model_config = ConfigDict(frozen=True)

    dispatch: DispatchPayload
    #: The service's own challenge id — the key /challenge/{id}/resolve uses
    #:. Equal to dispatch.challenge_id by construction; carried
    #: separately so the id we resolve is the one the SERVICE named, and
    #: defaulted so payloads without it still parse.
    challenge_id: str = ""
    track: str
    reference_path: str
    reference_digest: str = Field(pattern=SHA256_HEX)
    miner_input_path: str
    miner_input_digest: str = Field(pattern=SHA256_HEX)
    #: Finalized external receipt minted before the challenge service returned.
    #: Optional for legacy/report fixtures; production finalization refuses it.
    commitment_anchor: ChallengeAnchor | None = None
    #: track params forwarded to both the miner request and the score request
    params: dict[str, float | int | str] = Field(default_factory=dict)

    @property
    def resolve_id(self) -> str:
        """The challenge id to resolve/score against (service-authored first)."""
        return self.challenge_id or self.dispatch.challenge_id


class DispatchedChallenge(BaseModel):
    """One row of the challenge service's dispatched-challenge sweep list.

    `age_seconds` is the service's own measure of how long the challenge has
    been dispatched — the validator never guesses it from its own clock, since
    the two processes may not share one.

    `owner` is the validator identity the service recorded when it dispatched
    the challenge. It is the ONLY thing that makes a sweep
    safe with more than one validator on the subnet, so it is treated as
    positive evidence: a row without it is nobody's as far as we are concerned
    and is never expired. Defaulted so a service that does not implement
    ownership yet still parses (and simply yields an empty sweep).
    """

    model_config = ConfigDict(frozen=True)

    challenge_id: str
    track: str = ""
    age_seconds: float = 0.0
    owner: str = ""


class ChallengeOwnershipRefused(Exception):
    """The challenge service refused a resolve: 403 `not_owner`.

    A PERMANENT refusal, and the one failure the retry loop must not treat as
    transient: the row is left for an operator (counted + logged) instead of
    being retried against a boundary that will never move. It normally means the
    validator's identity rotated between fetching a challenge and recovering it,
    which is exactly why the fetching owner is now persisted with the row.

    `response` (when the raiser is the HTTP client) is the refusing httpx
    response, so callers can read the service's typed `not_owner` detail.
    """

    def __init__(self, message: str, *, response: Any | None = None) -> None:
        super().__init__(message)
        self.response = response


class ChallengeAlreadyTerminal(Exception):
    """The challenge service reports this challenge as unknown or already terminal.

    Nothing is left to drive: the in-flight row is dropped rather than retried
    forever (a 404 means the service never had it; a 409 means someone — usually
    this validator's own earlier attempt whose response was lost — already
    resolved it).
    """


class AuditStoreFailure(Exception):
    """A configured audit store could not archive a packet.

    Fails the ITEM closed: an accumulated score whose evidence exists only in the
    validator's own database is a score a third party cannot reproduce, which is
    exactly what the audit store is for. DB-only operation (no store configured)
    is a different, explicitly-flagged mode — not this.
    """


@runtime_checkable
class ChallengeClient(Protocol):
    async def next_challenge(self, track: str, owner: str = "") -> ChallengeItem:
        """Produce/check out the next challenge for `track`.

        `owner` is the requesting validator's identity, recorded by the service
        so the dispatched-challenge sweep can be scoped to its owner (an internal review). Clients predating the parameter are feature-detected.
        """
        ...

    async def resolve_challenge(
        self, challenge_id: str, outcome: str, owner: str = ""
    ) -> None:
        """Terminate a fetched challenge: releases the checked-out asset and
        unblocks the commit-reveal.

        `outcome` is 'resolved' (this round scored the item) or 'expired' (the
        round aborted). MUST raise ChallengeAlreadyTerminal when the service does
        not know the challenge or it is already terminal — that is a drained
        state, not a failure to retry.

        `owner` is the resolving validator's identity. The service enforces the
        SAME ownership boundary here as on the sweep: a challenge recorded with
        an owner may only be terminated by that owner (or the operator token),
        so a validator that omits it is 403'd on its own challenges. Clients
        predating the parameter are feature-detected exactly like
        `next_challenge`'s.
        """
        ...

    async def list_dispatched(
        self, older_than_seconds: float, owner: str = ""
    ) -> list[DispatchedChallenge]:
        """Dispatched challenges older than `older_than_seconds` (may be empty).

        OPTIONAL surface: a client that does not implement it is feature-detected
        and the orphan sweep is skipped (see `sweep_orphaned_challenges`). When
        `owner` is given the service filters to that validator's challenges; the
        validator additionally requires each row to NAME its owner, because an
        unknown query parameter is silently ignored by most HTTP frameworks and
        an unfiltered list must never be mistaken for a filtered one.
        """
        ...


@runtime_checkable
class MinerClient(Protocol):
    async def probe_warrant(self, neuron: ChainNeuron) -> str:
        """TaskWarrant: which track the miner serves. Raise on unreachable."""
        ...

    async def submit_task(
        self, neuron: ChainNeuron, request: MinerTaskRequest
    ) -> MinerTaskResponse: ...

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
    ) -> AvailabilityObservation | None: ...


@runtime_checkable
class ScoringClient(Protocol):
    async def score(self, request: ScoreRequest) -> ScoreResponse: ...

    async def scorer_identity(self) -> str:
        """The worker's effective scorer identity, from its GET /healthz.

        The discovery half of the scorer-identity contract (services.protocol).
        A client without it is feature-detected: the validator then has no pin
        and falls back to the configured operator pin (or to none at all).
        """
        ...


# -- HTTP implementations (the real wiring; tests inject fakes) ----------------


class HttpChallengeClient:
    """POST {challenge_service_url}/challenge/next {"track": ..., "owner": ...}
    -> ChallengeItem, POST {challenge_service_url}/challenge/{id}/resolve
    {"outcome": ..., "owner": ...}, and GET {challenge_service_url}/challenges
    (the dispatched-orphan sweep).

    `owner` is this validator's identity on all three: the service records it at
    production time and then REQUIRES it to resolve (403 `not_owner` otherwise),
    which is the same boundary that makes the sweep safe.

    EVERY challenge-service route is token-gated (that service hands out the
    held-out reference, so it fails closed): `api_token` is sent as
    `Authorization: Bearer <token>` on every request. It comes from
    `ValidatorConfig.challenge_service_token` and must equal the service's
    `challenge_service.api_token`; without it every call is a 401.

    P2 hotkey auth: when a `signer` is wired (`hotkey` + `sign(bytes)->hex`,
    the `CallableHotkeySigner` shape), every request additionally carries the
    four Scheme A signed headers over the exact bytes sent, so the service's
    hotkey-auth `enforce` mode keeps serving this validator. The bearer stays
    as a second factor.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float,
        api_token: str = "",
        *,
        signer: object | None = None,
    ) -> None:
        token = (api_token or "").strip()
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )
        self._signer = signer
        # The guard verifies over `request.url.path`, which includes any base-url
        # path prefix — sign the same thing.
        from urllib.parse import urlsplit

        self._base_path = urlsplit(base_url).path.rstrip("/")

    def _signed(self, method: str, path: str, body: bytes = b"") -> dict[str, str]:
        if self._signer is None:
            return {}
        from vidaio.services.hotkey_auth import sign_request_headers

        return sign_request_headers(
            self._signer, method=method, path=self._base_path + path, body=body
        )

    async def _post_json(self, path: str, body: dict[str, str]) -> httpx.Response:
        # Serialize once and send those exact bytes: Scheme A signs the body.
        payload = json.dumps(body, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"}
        headers.update(self._signed("POST", path, payload))
        return await self._client.post(path, content=payload, headers=headers)

    async def next_challenge(self, track: str, owner: str = "") -> ChallengeItem:
        body: dict[str, str] = {"track": track}
        if owner:
            body["owner"] = owner  # recorded by the service; ignored by older ones
        response = await self._post_json("/challenge/next", body)
        response.raise_for_status()
        return ChallengeItem.model_validate(response.json())

    async def list_dispatched(
        self, older_than_seconds: float, owner: str = ""
    ) -> list[DispatchedChallenge]:
        params: dict[str, Any] = {
            "status": "dispatched",
            "older_than_seconds": older_than_seconds,
        }
        if owner:
            params["owner"] = owner
        response = await self._client.get(
            "/challenges", params=params, headers=self._signed("GET", "/challenges")
        )
        response.raise_for_status()
        body = response.json()
        return [
            DispatchedChallenge.model_validate(row)
            for row in body.get("challenges", [])
        ]

    async def resolve_challenge(
        self, challenge_id: str, outcome: str, owner: str = ""
    ) -> None:
        body: dict[str, str] = {"outcome": outcome}
        if owner:
            # The service refuses (403 not_owner) when an OWNED challenge is
            # resolved by anybody else; omitting it on our own challenge is a
            # permanent failure, not a retryable one. Older services ignore it.
            body["owner"] = owner
        response = await self._post_json(f"/challenge/{challenge_id}/resolve", body)
        # 404 unknown_challenge / 409 not_resolvable are both "nothing left to
        # drive" — the challenge service's own contract (challenge_service.service).
        if response.status_code in (404, 409):
            raise ChallengeAlreadyTerminal(
                f"challenge {challenge_id} is unknown or already terminal:"
                f" HTTP {response.status_code} {response.text}"
            )
        # Only a POSITIVE ownership refusal (detail.code == "not_owner") is
        # PERMANENT and may park the obligation. Any other
        # 403 — an auth proxy, an RBAC layer, a misconfigured token — is
        # transient infrastructure and must stay on the normal retry path, or
        # a passing outage would durably park live challenges.
        if response.status_code == 403:
            try:
                detail = response.json().get("detail")
                detail_code = detail.get("code") if isinstance(detail, dict) else None
            except (ValueError, AttributeError):
                detail_code = None
            if detail_code == "not_owner":
                raise ChallengeOwnershipRefused(
                    f"challenge {challenge_id} refused for owner {owner!r}:"
                    f" HTTP 403 {response.text}",
                    response=response,
                )
        response.raise_for_status()


class HttpMinerClient:
    """GET warrant + path-free, bounded byte-stream task exchange.

    The input path is opened only on the validator and its bytes are streamed to
    ``/v1/task/artifact``. The response body is streamed into a validator-owned
    artifact directory and verified against task id, length, and sha256 before
    a local ``MinerTaskResponse`` is returned. No path or URL supplied by a
    remote miner is ever followed.

    `api_token` (ValidatorConfig.miner_api_token) is sent as `X-Miner-Token` on
    every call when non-empty — the SAME optional shared secret the organic
    gateway presents (`gateway.miner_api_token`), so a miner that closes its
    public task endpoint with `miner.api_token` stays reachable by both of its
    legitimate callers. Empty sends no header, which is what an open loopback
    miner expects.
    """

    def __init__(
        self,
        port: int,
        timeout: float,
        api_token: str = "",
        *,
        scheme: str = "http",
        artifact_dir: str | Path = "./data/validator/miner-artifacts",
        max_input_bytes: int = 2 * 1024 * 1024 * 1024,
        max_output_bytes: int = 4 * 1024 * 1024 * 1024,
        allow_non_public_addresses: bool = False,
        artifact_auth: ArtifactClientAuth | None = None,
        allow_unsigned_artifact_v1: bool = False,
    ) -> None:
        self._port = port
        self._timeout = timeout
        normalized_scheme = scheme.strip().lower()
        if normalized_scheme not in {"http", "https"}:
            raise ValueError("miner URL scheme must be exactly 'http' or 'https'")
        self._scheme = normalized_scheme
        self._artifact_dir = Path(artifact_dir)
        self._max_input_bytes = max_input_bytes
        self._max_output_bytes = max_output_bytes
        self._allow_non_public_addresses = allow_non_public_addresses
        self._artifact_auth = artifact_auth
        self._allow_unsigned_artifact_v1 = allow_unsigned_artifact_v1
        token = (api_token or "").strip()
        self._headers = {"X-Miner-Token": token} if token else {}
        self._client = httpx.AsyncClient(timeout=timeout, headers=self._headers)

    def _base(self, neuron: ChainNeuron) -> str:
        host = format_miner_peer_host(
            neuron.ip, allow_non_public=self._allow_non_public_addresses
        )
        port = self._port if neuron.axon_port is None else neuron.axon_port
        if not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65536:
            raise MinerPeerAddressError(
                f"invalid miner axon port {port!r} for uid {neuron.uid}"
            )
        return f"{self._scheme}://{host}:{port}"

    async def probe_warrant(self, neuron: ChainNeuron) -> str:
        response = await self._client.get(f"{self._base(neuron)}/warrant")
        response.raise_for_status()
        return str(response.json().get("track", ""))

    async def submit_task(
        self, neuron: ChainNeuron, request: MinerTaskRequest
    ) -> MinerTaskResponse:
        try:
            base_url = self._base(neuron)
        except MinerPeerAddressError as exc:
            # An unusable chain advertisement is miner-attributable, but only an
            # exact signed request whose local input was independently validated
            # may affect EWMA. Prepare that proof off the event loop; a local
            # file/wallet failure stays typed as validator input trouble.
            if self._artifact_auth is not None:
                receipt = await asyncio.to_thread(
                    sign_miner_artifact_request_receipt,
                    request,
                    max_input_bytes=self._max_input_bytes,
                    artifact_auth=self._artifact_auth,
                    expected_miner_hotkey=neuron.hotkey,
                )
                exc.artifact_request_receipt = receipt
                raw_port = self._port if neuron.axon_port is None else neuron.axon_port
                target_digest = hashlib.sha256(
                    f"{neuron.uid}\0{neuron.ip}\0{raw_port}".encode()
                ).hexdigest()
                exc.artifact_target_endpoint = (
                    f"chain-axon-invalid://sha256/{target_digest}"
                )
            raise
        return await submit_miner_artifact(
            self._client,
            base_url,
            request,
            output_dir=self._artifact_dir,
            max_input_bytes=self._max_input_bytes,
            max_output_bytes=self._max_output_bytes,
            headers=self._headers,
            timeout=self._timeout,
            artifact_auth=self._artifact_auth,
            expected_miner_hotkey=(
                neuron.hotkey if self._artifact_auth is not None else None
            ),
            allow_unsigned_v1=self._allow_unsigned_artifact_v1,
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
    ) -> AvailabilityObservation | None:
        """Sign one request-bound availability zero, or refuse without v2 proof."""
        if self._artifact_auth is None:
            return None
        request_receipt = None
        miner_receipt = None
        endpoint = None
        if exception is not None:
            # ``with_timeout`` may translate the inner cancellation into a fresh
            # TimeoutError. Walk the bounded exception chain to recover the request
            # receipt attached inside ``submit_miner_artifact`` before cancellation.
            current: BaseException | None = exception
            seen: set[int] = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                request_receipt = getattr(
                    current, "artifact_request_receipt", request_receipt
                )
                endpoint = getattr(current, "artifact_target_endpoint", endpoint)
                current = current.__cause__ or current.__context__
        if response is not None and response.artifact_receipt is not None:
            try:
                miner_receipt = MinerArtifactReceipt.model_validate(
                    response.artifact_receipt
                )
            except Exception:
                miner_receipt = None
            if miner_receipt is not None:
                request_receipt = miner_receipt.request_receipt()
        if request_receipt is None:
            return None
        endpoint = str(endpoint or self._base(neuron))
        attempt = DispatchAttempt(
            uid=neuron.uid,
            miner_hotkey=neuron.hotkey,
            endpoint=endpoint,
            challenge_id=request.task_id.rsplit(":", 1)[0],
            item_id=request.task_id,
            track=request.track,
            request=request_receipt,
        )
        return build_availability_observation(
            attempt=attempt,
            reason=reason,
            signer=self._artifact_auth.signer,
            returned_task_id=returned_task_id,
            observed_output_digest=observed_output_digest,
            miner_receipt=miner_receipt,
        )


class HttpScoringClient:
    """HTTP scorer bound to an exact locally derived payout-runtime contract."""

    def __init__(
        self,
        base_url: str,
        timeout: float,
        *,
        expected_runtime_contract: ScorerRuntimeContract | None = None,
        allow_noncanonical_runtime_for_report_or_tests: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if expected_runtime_contract is None and not (
            allow_noncanonical_runtime_for_report_or_tests
        ):
            raise ValueError(
                "HttpScoringClient requires a locally derived canonical payout-"
                "runtime contract; only explicit report/test callers may opt out"
            )
        if expected_runtime_contract is not None and (
            allow_noncanonical_runtime_for_report_or_tests
        ):
            raise ValueError(
                "expected_runtime_contract and the report/test opt-out are mutually "
                "exclusive"
            )
        self._base_url = base_url
        self._expected_runtime_contract = expected_runtime_contract
        self._allow_noncanonical_runtime = (
            allow_noncanonical_runtime_for_report_or_tests
        )
        self._runtime_contract: ScorerRuntimeContract | None = None
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
        )

    async def score(self, request: ScoreRequest) -> ScoreResponse:
        contract = await self._runtime_contract_for_score()
        response = await self._client.post(
            "/score", json=request.model_dump(mode="json")
        )
        response.raise_for_status()
        result = ScoreResponse.model_validate(response.json())
        packet_bytes = result.item_score_json.encode("utf-8")
        if hashlib.sha256(packet_bytes).hexdigest() != result.packet_digest:
            raise RuntimeError("scoring worker returned a packet digest mismatch")
        packet = ItemScore.from_json(result.item_score_json)
        if packet.scorer_version != contract.scorer_version:
            raise RuntimeError(
                "scoring worker packet identity differs from its validated health "
                "runtime contract"
            )
        if packet.backend_versions != contract.backend_versions:
            raise RuntimeError(
                "scoring worker packet backend_versions differs from its exact "
                "validated health runtime contract"
            )
        return result

    async def scorer_identity(self) -> str:
        """Refresh and exact-match the worker's complete payout runtime."""

        self._runtime_contract = None
        contract = await self._fetch_runtime_contract()
        self._runtime_contract = contract
        return contract.scorer_version

    async def _fetch_runtime_contract(self) -> ScorerRuntimeContract:
        contract = await fetch_scorer_runtime_contract_async(
            self._base_url,
            client=self._client,
            require_canonical=not self._allow_noncanonical_runtime,
        )
        expected = self._expected_runtime_contract
        if expected is not None:
            require_matching_scorer_runtime_contract(
                contract,
                expected,
                context="inference scoring worker healthz",
            )
        return contract

    async def _runtime_contract_for_score(self) -> ScorerRuntimeContract:
        if self._runtime_contract is None:
            self._runtime_contract = await self._fetch_runtime_contract()
        return self._runtime_contract


# -- round bookkeeping ---------------------------------------------------------


@dataclass
class RoundReport:
    """What one round did — returned for tests/observability, persisted via DB."""

    scored: dict[int, float] = field(default_factory=dict)
    #: uid -> reason for an evidence-backed economic zero: a byte-exact duplicate,
    #: or ``availability:<reason>`` backed by the exact signed dispatch observation.
    zeroed: dict[int, str] = field(default_factory=dict)
    #: uid -> excused system/protocol failure which deliberately did NOT affect EWMA.
    #: Miner-attributable reasons move to ``zeroed`` only with signed request evidence.
    non_punitive_skips: dict[int, str] = field(default_factory=dict)
    #: uids skipped because their track is unknown (NOT scored, NOT defaulted)
    skipped_unknown_track: list[int] = field(default_factory=list)
    #: uids whose scoring-worker call failed (no accumulation this round)
    scoring_failed: list[int] = field(default_factory=list)
    #: uid -> the packet field that did not match this validator's request; a
    #: replayed/foreign packet. These uids are also in scoring_failed.
    rejected_packets: dict[int, str] = field(default_factory=dict)
    #: uids whose packet could not be ARCHIVED while an audit store is configured
    #:. Failed closed: not accumulated, no evidence row. Also
    #: listed in scoring_failed — like any validator-side infra trouble it is
    #: non-punitive to the miner.
    audit_store_failed: list[int] = field(default_factory=list)
    #: set when the WHOLE round was skipped before any scoring (e.g. the chain
    #: snapshot was stale/unavailable). No EWMA was touched.
    skipped_reason: str | None = None
    #: ledger id of this round (None when the round was skipped before it opened)
    round_id: str | None = None
    #: challenge ids this round fetched, mapped to the outcome they were resolved as
    resolved_challenges: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PacketEvidence:
    """One accepted, request-bound score packet plus the bytes that prove it."""

    uid: int
    item_id: str
    challenge_id: str
    track: str
    miner_hotkey: str
    content_digest: str
    packet_digest: str
    packet_json: str
    scorer_version: str
    score: float
    #: audit-store backend key when archived as a SCORE_PACKET artifact, else None
    audit_ref: str | None = None
    #: Serialized ArtifactRefs for the exact media bytes. None only in explicitly
    #: visible DB-only mode; production refuses to finalize without all three.
    challenge_input_ref: str | None = None
    miner_output_ref: str | None = None
    reference_original_ref: str | None = None
    #: Canonical JSON for the miner-signed artifact-v2 response. Its signed
    #: request embeds ``commitment_anchor``, proving the result followed anchor.
    miner_receipt_json: str | None = None

    def as_row(self) -> dict[str, object]:
        return {
            "uid": self.uid,
            "item_id": self.item_id,
            "challenge_id": self.challenge_id,
            "track": self.track,
            "miner_hotkey": self.miner_hotkey,
            "content_digest": self.content_digest,
            "packet_digest": self.packet_digest,
            "packet_json": self.packet_json,
            "scorer_version": self.scorer_version,
            "score": self.score,
            "audit_ref": self.audit_ref,
            "challenge_input_ref": self.challenge_input_ref,
            "miner_output_ref": self.miner_output_ref,
            "reference_original_ref": self.reference_original_ref,
            "miner_receipt_json": self.miner_receipt_json,
        }


@dataclass(frozen=True, slots=True)
class DuplicateCandidate:
    """A byte-exact loser paired with the anchor-salted identity winner."""

    loser_uid: int
    loser_response: MinerTaskResponse
    winner_uid: int
    winner_response: MinerTaskResponse


@dataclass(frozen=True, slots=True)
class DedupOutcome:
    kept: tuple[tuple[int, MinerTaskResponse], ...]
    duplicates: tuple[DuplicateCandidate, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dedup_miners(neurons: list[ChainNeuron]) -> list[ChainNeuron]:
    """First-uid-per-IP and per-coldkey, uid ascending — the same deterministic
    rule as tokenomics.rank_curve.eligible_for_ranking, applied at dispatch time
    (without the positive-score gate: new miners must still get dispatched to)."""
    seen_ips: set[str] = set()
    seen_coldkeys: set[str] = set()
    kept: list[ChainNeuron] = []
    for n in sorted(neurons, key=lambda x: x.uid):
        ip_key = dedup_ip_key(n.ip)
        if (ip_key is not None and ip_key in seen_ips) or n.coldkey in seen_coldkeys:
            continue
        if ip_key is not None:
            seen_ips.add(ip_key)
        seen_coldkeys.add(n.coldkey)
        kept.append(n)
    return kept


def permit_holder_has_serving_axon(neuron: ChainNeuron) -> bool:
    """Prove a permit-holding identity explicitly advertises a miner endpoint.

    Validator permit is non-exclusive, so it can never reject a *serving* miner.
    Conversely, control/authority hotkeys usually have no axon. Letting one of
    those enter pre-warrant IP/coldkey dedup can shadow a real miner (and a report
    fallback port could even route the control uid to somebody else's service).
    Require the chain-advertised port and a specified address only for permit
    holders; ordinary miners retain the legacy fallback-port path in report tests.
    """
    port = neuron.axon_port
    return (
        isinstance(port, int)
        and not isinstance(port, bool)
        and 0 < port < 65536
        and dedup_ip_key(neuron.ip) is not None
    )


class InferenceValidator(BaseService):
    name = "inference-validator"

    def __init__(
        self,
        raw_config: dict[str, Any],
        *,
        chain: ChainAdapter,
        challenge_client: ChallengeClient | None = None,
        miner_client: MinerClient | None = None,
        scoring_client: ScoringClient | None = None,
        conn: Any | None = None,
        conn_factory: Callable[[], Any] | None = None,
        store: AuditStore | None = None,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        artifact_auth: ArtifactClientAuth | None = None,
        allow_unsigned_artifact_v1: bool = False,
        expected_scorer_runtime_contract: ScorerRuntimeContract | None = None,
    ) -> None:
        self.config = section(raw_config, "validator", ValidatorConfig)
        super().__init__(raw_config, metrics_port=self.config.metrics_port)
        self.tokenomics = section(raw_config, "tokenomics", TokenomicsConfig)
        self.scoring = section(raw_config, "scoring", ScoringConfig)
        self.chain = chain
        cfg = self.config
        # One identity-bound signer for every authenticated client surface:
        # artifact-v2 receipts AND the P2 signed headers on challenge routes.
        chain_sign = getattr(chain, "sign", None)
        hotkey_signer: CallableHotkeySigner | None = None
        if cfg.identity and callable(chain_sign):
            hotkey_signer = CallableHotkeySigner(cfg.identity, chain_sign)
        self.challenge_client: ChallengeClient = (
            challenge_client
            or HttpChallengeClient(
                cfg.challenge_service_url,
                cfg.challenge_request_timeout_seconds,
                cfg.challenge_service_token,
                signer=hotkey_signer,
            )
        )
        if artifact_auth is None and miner_client is None and hotkey_signer is not None:
            artifact_auth = ArtifactClientAuth(hotkey_signer)
        if (
            miner_client is None
            and artifact_auth is None
            and str((raw_config.get("chain") or {}).get("mode", "report"))
            == "bittensor"
        ):
            raise RuntimeError(
                "bittensor inference validator has no artifact-v2 hotkey signer"
            )
        self.miner_client: MinerClient = miner_client or HttpMinerClient(
            cfg.miner_port,
            cfg.miner_request_timeout_seconds,
            cfg.miner_api_token,
            scheme=cfg.miner_url_scheme,
            artifact_dir=cfg.miner_artifact_dir,
            max_input_bytes=cfg.miner_max_input_bytes,
            max_output_bytes=cfg.miner_max_output_bytes,
            allow_non_public_addresses=cfg.allow_non_public_miner_addresses,
            artifact_auth=artifact_auth,
            allow_unsigned_artifact_v1=allow_unsigned_artifact_v1,
        )
        if scoring_client is None:
            bittensor_mode = str(
                (raw_config.get("chain") or {}).get("mode", "report")
            ) == "bittensor"
            if bittensor_mode and expected_scorer_runtime_contract is None:
                raise RuntimeError(
                    "bittensor inference validator requires an independently "
                    "derived local canonical scorer runtime contract"
                )
            scoring_client = HttpScoringClient(
                cfg.scoring_worker_url,
                cfg.scoring_request_timeout_seconds,
                expected_runtime_contract=expected_scorer_runtime_contract,
                allow_noncanonical_runtime_for_report_or_tests=not bittensor_mode,
            )
        self.scoring_client = scoring_client
        self.conn = conn if conn is not None else connect(self.core.db_path)
        # Health checks run on the HealthServer's THREAD; sqlite3 connections are
        # not shareable across threads, so they get their own handle.
        self._conn_factory: Callable[[], Any] | None = (
            conn_factory
            if conn_factory is not None
            else miner_manager.connection_factory(self.conn)
        )
        self._thread_local = threading.local()
        #: audit store for SCORE_PACKET artifacts. None = DB-only.
        self._store = store
        #: inject a seeded random.Random for deterministic sleep jitter
        self._rng = rng if rng is not None else random.Random()
        self._clock = clock
        #: wall clock (epoch seconds) — only used to talk to the chain adapter's
        #: optional freshness surface, which keeps its own clock
        self._wall_clock = wall_clock
        self._started_at = clock()
        self._last_round_at: float | None = None
        self._last_refresh_at: float | None = None
        self._recovered_inflight = False
        #: The identity the LAST `/challenge/next` actually carried ('' when the
        #: fetch was anonymous). Persisted with the in-flight row so recovery
        #: resolves as the fetcher, not as the current config.
        self._last_fetch_owner = ""
        #: The scorer identity this validator is bound to — PIN ON FIRST CONTACT
        #: (services.protocol, scorer-identity contract), DURABLE across restarts
        #:. None only until the worker has been reached once
        #: on a validator that has never pinned.
        self.pinned_scorer_version: str | None = None
        #: Set when the persisted pin and the live worker disagree: scoring is
        #: refused until an operator acknowledges the change. Never auto-cleared.
        self.scorer_pin_conflict: str | None = None
        #: Has the live worker confirmed our identity in THIS process? A reloaded
        #: pin is not evidence about the worker running right now.
        self._identity_verified = False
        self._load_scorer_pin()

        if self._store is None:
            self.log.warning(
                "no audit store configured: score packets are persisted to the"
                " validator DB only, so published weights cannot be reproduced from"
                " the audit store alone"
            )

        registry = self.health.registry
        self.m_rounds = Counter(
            "vidaio_validator_rounds_total",
            "Completed synthetic rounds",
            registry=registry,
        )
        self.m_scored = Counter(
            "vidaio_validator_scored_total",
            "Miner responses scored (including independently auditable duplicate zeroes)",
            ["track"],
            registry=registry,
        )
        self.m_skipped_unknown = Counter(
            "vidaio_validator_skipped_unknown_track_total",
            "Miners skipped for the round because their warrant track is unknown"
            " (the deliberate fix of the old default-to-upscaling bug)",
            registry=registry,
        )
        self.m_miner_timeouts = Counter(
            "vidaio_validator_miner_timeouts_total",
            "Miner task requests that timed out (non-punitive skip)",
            registry=registry,
        )
        self.m_digest_mismatch = Counter(
            "vidaio_validator_digest_mismatch_total",
            "Miner responses whose output file did not match its claimed digest",
            registry=registry,
        )
        self.m_round_duration = Histogram(
            "vidaio_validator_round_duration_seconds",
            "Wall-clock duration of one synthetic round",
            registry=registry,
        )
        self.m_rounds_skipped = Counter(
            "vidaio_validator_rounds_skipped_total",
            "Rounds skipped before any scoring, by structured reason"
            " (stale/unavailable chain state — never reported as a successful round)",
            ["reason"],
            registry=registry,
        )
        self.m_chain_refresh_failures = Counter(
            "vidaio_validator_chain_refresh_failures_total",
            "Chain refreshes that failed (the snapshot is NOT marked fresh)",
            registry=registry,
        )
        self.m_packet_rejected = Counter(
            "vidaio_validator_score_packet_rejected_total",
            "Score packets rejected because they are not bound to this validator's"
            " request (replay / MITM guard), by the field that mismatched",
            ["field"],
            registry=registry,
        )
        self.m_challenges_resolved = Counter(
            "vidaio_validator_challenges_resolved_total",
            "Challenges terminated through /challenge/{id}/resolve, by outcome",
            ["outcome"],
            registry=registry,
        )
        self.m_challenge_resolve_failures = Counter(
            "vidaio_validator_challenge_resolve_failures_total",
            "Failed /challenge/{id}/resolve calls (the in-flight row is retained"
            " and retried by the next round / startup recovery pass)",
            registry=registry,
        )
        self.m_challenge_resolve_forbidden = Counter(
            "vidaio_validator_challenge_resolve_forbidden_total",
            "Resolves refused 403 not_owner: the recorded owner of an in-flight"
            " challenge is not accepted by the service. Retrying cannot fix it —"
            " the row is left for an operator",
            registry=registry,
        )
        self.m_challenges_parked = Gauge(
            "vidaio_validator_parked_challenges",
            "In-flight challenge rows PARKED by a permanent ownership refusal"
            " (403 not_owner) and excluded from the drain until an operator"
            " unparks them (`validator.unpark_challenges` / the"
            " unpark_challenges() admin method) — each one is a service-side"
            " asset stranded in_use",
            registry=registry,
        )
        self.m_packets_persisted = Counter(
            "vidaio_validator_score_packets_persisted_total",
            "Score packets whose exact bytes were persisted as round evidence",
            registry=registry,
        )
        self.m_audit_store_failures = Counter(
            "vidaio_validator_audit_store_failures_total",
            "SCORE_PACKET artifact writes that failed (evidence stays DB-only)",
            registry=registry,
        )
        self.m_orphans_swept = Counter(
            "vidaio_validator_orphaned_challenges_swept_total",
            "Dispatched challenges with no in-flight record of ours, expired by the"
            " startup sweep (the lost-/challenge/next-RESPONSE blind spot)",
            registry=registry,
        )
        self.m_sweep_skipped = Counter(
            "vidaio_validator_orphan_sweep_skipped_total",
            "Dispatched challenges the sweep deliberately did NOT expire because"
            " they are not provably ours, by reason (ownership boundary, new-4)",
            ["reason"],
            registry=registry,
        )
        self.m_task_id_mismatch = Counter(
            "vidaio_validator_task_id_mismatch_total",
            "Miner responses echoing a task id this validator never dispatched"
            "",
            registry=registry,
        )
        self.m_audit_store_configured = Gauge(
            "vidaio_validator_audit_store_configured",
            "1 when SCORE_PACKET artifacts are archived to an audit store, 0 when"
            " this validator runs DB-ONLY (published weights are then not"
            " reproducible from the audit store alone)",
            registry=registry,
        )
        self.m_audit_store_configured.set(0.0 if self._store is None else 1.0)
        self.m_scorer_pinned = Gauge(
            "vidaio_validator_scorer_pinned",
            "1 when a scorer identity is pinned and scoring may run, 0 when it is"
            " unknown or in conflict",
            registry=registry,
        )
        self._publish_pin_metric()

        self.health.register_check("db", self._db_reachable)
        self.health.register_check("last_round_age", self._round_fresh)
        # An unresolved pin conflict is not a transient condition: the process is
        # deliberately not scoring, and /health must say so.
        self.health.register_check(
            "scorer_pin", lambda: self.scorer_pin_conflict is None
        )

    # -- health ----------------------------------------------------------------

    def _health_conn(self) -> Any:
        """A connection owned by the CALLING thread.

        The HealthServer answers /health on its own thread; reusing the round
        loop's sqlite3 handle there raises ProgrammingError and made /health
        report the DB unhealthy even when it was fine. Falls back to the round
        loop's handle only for ':memory:' databases, which cannot be reopened.
        """
        if self._conn_factory is None:
            return self.conn
        conn = getattr(self._thread_local, "conn", None)
        if conn is None:
            conn = self._conn_factory()
            self._thread_local.conn = conn
        return conn

    def _db_reachable(self) -> bool:
        try:
            self._health_conn().execute("SELECT COUNT(*) FROM miners").fetchone()
            return True
        except Exception:
            # Drop a poisoned per-thread handle so the next check reconnects.
            self._thread_local.conn = None
            return False

    def _round_fresh(self) -> bool:
        reference = (
            self._last_round_at if self._last_round_at is not None else self._started_at
        )
        allowance = 2 * self.config.cycle_sleep_max_seconds + 600.0
        return (self._clock() - reference) < allowance

    # -- chain -----------------------------------------------------------------

    def _maybe_refresh_chain(self) -> None:
        """Refresh (throttled). A FAILED refresh is NOT recorded as a success.

        review #21: the old code stamped `_last_refresh_at` unconditionally, so a
        startup race against the chain sim left an EMPTY snapshot considered
        fresh for a full 30-minute throttle window and the validator happily
        reported successful empty rounds.
        """
        now = self._clock()
        if (
            self._last_refresh_at is not None
            and now - self._last_refresh_at < self.config.metagraph_refresh_seconds
        ):
            return
        try:
            self.chain.refresh()
        except Exception as exc:
            self.m_chain_refresh_failures.inc()
            self.log.warning(
                "chain refresh failed; snapshot NOT marked fresh",
                extra=log_fields(error=str(exc)),
            )
            return
        self._last_refresh_at = now

    def _adapter_reports_fresh(self, max_age: float) -> bool | None:
        """Ask the chain adapter's OPTIONAL freshness surface. None = unavailable.

        `ChainAdapter.has_fresh_snapshot(now, max_age)` is being added by the
        chain owner; this feature-detects it so the validator is
        correct whether or not it has landed, and tolerates either the
        (now, max_age) or the no-argument spelling. `now` is passed as WALL-clock
        epoch seconds — the adapter keeps its own clock, not the round loop's
        monotonic one.
        """
        probe = getattr(self.chain, "has_fresh_snapshot", None)
        if not callable(probe):
            return None
        for args in ((self._wall_clock(), max_age), (max_age,), ()):
            try:
                return bool(probe(*args))
            except TypeError:
                continue  # signature mismatch: try the next spelling
            except Exception as exc:
                self.log.warning(
                    "chain freshness probe raised; falling back to local bookkeeping",
                    extra=log_fields(error=str(exc)),
                )
                return None
        return None

    def _chain_state_reason(self) -> str | None:
        """None when the cached snapshot may be scored against; else a skip reason."""
        max_age = self.config.max_chain_snapshot_age_seconds
        if max_age <= 0:
            return None  # gating explicitly disabled
        adapter = self._adapter_reports_fresh(max_age)
        if adapter is False:
            return "chain_snapshot_stale"
        if adapter is None:
            if self._last_refresh_at is None:
                return "chain_snapshot_never_refreshed"
            if self._clock() - self._last_refresh_at > max_age:
                return "chain_snapshot_stale"
        return None

    def _skip_round(
        self, report: RoundReport, reason: str, **fields: Any
    ) -> RoundReport:
        """Abandon the round with a structured reason. NO EWMA, NO round ledger row."""
        report.skipped_reason = reason
        self.m_rounds_skipped.labels(reason=reason).inc()
        self.log.warning(
            "round skipped before any scoring — no scores were recorded",
            extra=log_fields(reason=reason, **fields),
        )
        return report

    def eligible_neurons(self) -> list[ChainNeuron]:
        miners = [
            n
            for n in self.chain.neurons()
            if n.alpha_stake >= self.config.min_stake
            and (not n.is_validator or permit_holder_has_serving_axon(n))
        ]
        return dedup_miners(miners)

    # -- warrant probing (the fixed TaskWarrant) -------------------------------

    async def _probe_warrant(self, neuron: ChainNeuron) -> tuple[int, str | None]:
        """Probe one miner's track. Returns (uid, track|None). NEVER defaults.

        The result is STAGED by the caller and written by `commit_round` (an internal review
        round 2): a warrant track is part of the round's observable state, so it
        must not appear in the registry ahead of the round that learned it.
        """
        cfg = self.config
        try:
            raw = await retry_async(
                lambda: with_timeout(
                    self.miner_client.probe_warrant(neuron),
                    cfg.warrant_probe_timeout_seconds,
                    f"warrant probe uid={neuron.uid}",
                ),
                attempts=2,
                base_delay=0.05,
            )
        except Exception as exc:
            self.log.warning(
                "warrant probe failed; track stays unknown",
                extra=log_fields(uid=neuron.uid, error=str(exc)),
            )
            return neuron.uid, None
        track = miner_manager.normalize_track(raw)
        if track is None:
            self.log.warning(
                "warrant probe returned an unknown track; miner stays unclassified",
                extra=log_fields(uid=neuron.uid, raw_track=str(raw)),
            )
            return neuron.uid, None
        return neuron.uid, track

    # -- scorer identity (services.protocol: PIN ON FIRST CONTACT) -------------

    def _publish_pin_metric(self) -> None:
        self.m_scorer_pinned.set(1.0 if self.scoring_pin() else 0.0)

    def scoring_pin(self) -> str:
        """The identity every packet of a scored round must carry, or "".

        An unresolved pin CONFLICT answers "" — the validator refuses to score
        until an operator decides, rather than picking one of the two scorers.
        """
        if self.scorer_pin_conflict is not None:
            return ""
        return self.pinned_scorer_version or self.config.scorer_version.strip()

    def _load_scorer_pin(self) -> None:
        """Adopt the DURABLE pin at construction.

        `validator.reset_scorer_pin` is the operator's explicit acknowledgement
        that this validator may bind to another scorer: it drops the row so the
        next discovery re-pins, and says loudly that the accumulators built under
        the old identity are now mixed unless the operator wipes them.
        """
        try:
            if self.config.reset_scorer_pin:
                cleared = miner_manager.clear_scorer_pin(self.conn)
                if cleared:
                    self.log.critical(
                        "OPERATOR RESET the durable scorer pin: this validator may"
                        " now bind to a different scorer. EWMA accumulators built"
                        " under the previous identity are NOT reset and will mix"
                        " two scorers unless they are wiped deliberately",
                        extra=log_fields(cleared_scorer_version=cleared),
                    )
            row = miner_manager.load_scorer_pin(self.conn)
        except Exception as exc:  # a validator DB without the migration applied
            self.log.error(
                "durable scorer pin unreadable; falling back to in-process pinning",
                extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
            )
            return
        if row is None:
            return
        persisted = str(row["scorer_version"])
        configured = self.config.scorer_version.strip()
        if configured and persisted != configured:
            # The operator pin and the accumulators disagree: refuse both.
            self.scorer_pin_conflict = (
                f"validator.scorer_version={configured!r} does not match the"
                f" persisted pin {persisted!r}"
            )
            self.log.critical(
                "SCORER PIN CONFLICT at startup: the operator pin names a different"
                " scorer than the one this validator's accumulators were built"
                " under. NOTHING will be scored until an operator resolves it"
                " (validator.reset_scorer_pin acknowledges the change)",
                extra=log_fields(persisted=persisted, configured=configured),
            )
            return
        self.pinned_scorer_version = persisted
        self.log.info(
            "durable scorer pin loaded; every packet this process accepts must"
            " carry it (the pin survives restarts by design)",
            extra=log_fields(scorer_version=persisted, pinned_at=row["pinned_at"]),
        )

    async def discover_scorer_identity(self) -> str | None:
        """Read the worker's identity from GET /healthz and PIN it. Returns it.

        The discovery half of the scorer-identity contract (services.protocol):

        * unreachable/unusable worker -> returns None, nothing is pinned, and the
          next call tries again (a scoring worker that is not up yet must not
          make the validator unstartable);
        * `config.scorer_version` empty -> whatever the worker advertises becomes
          THE identity for this process; every later packet must carry exactly it;
        * `config.scorer_version` non-empty -> that is an explicit operator pin.
          A worker advertising anything else is a CONFIGURATION ERROR and raises
          ScorerIdentityMismatch, because silently scoring against a scorer the
          operator did not choose is precisely the drift the pin exists to stop.

        Re-pinning to a DIFFERENT identity is refused the same way: a worker that
        changed its scoring levers under a running validator would otherwise
        split one EWMA accumulation across two scorers. Round-2 review new-2: that
        refusal now spans RESTARTS too — the pin is persisted on first contact and
        a disagreement with the persisted value latches `scorer_pin_conflict`, so
        the validator stops scoring instead of quietly adopting whatever answered.
        """
        if self.scorer_pin_conflict is not None:
            return None  # latched: only an operator acknowledgement clears it
        probe = getattr(self.scoring_client, "scorer_identity", None)
        if not callable(probe):
            return None
        try:
            discovered = await with_timeout(
                probe(),
                self.config.scorer_identity_timeout_seconds,
                "scorer identity discovery",
            )
        except ScorerRuntimeMismatch as exc:
            self.scorer_pin_conflict = (
                "the live scoring worker payout runtime does not match this "
                f"validator's pinned release image: {exc}"
            )
            self._publish_pin_metric()
            self.log.critical(
                "SCORER RUNTIME CONTRACT MISMATCH: refusing to score until an "
                "operator resolves the deployment",
                extra=log_fields(
                    scoring_worker_url=self.config.scoring_worker_url,
                    error=str(exc),
                ),
            )
            raise
        except (
            ScorerIdentityUnavailable,
            TimeoutError,
            OSError,
            httpx.HTTPError,
        ) as exc:
            self.log.warning(
                "scorer identity not discoverable yet; no pin taken this attempt",
                extra=log_fields(
                    scoring_worker_url=self.config.scoring_worker_url, error=str(exc)
                ),
            )
            return None
        identity = str(discovered).strip()
        if not identity:
            return None
        configured = self.config.scorer_version.strip()
        if configured and identity != configured:
            self.scorer_pin_conflict = (
                f"validator.scorer_version={configured!r} but the live worker"
                f" advertises {identity!r}"
            )
            self._publish_pin_metric()
            self.log.critical(
                "the live scoring worker is NOT the scorer this operator pinned —"
                " refusing to score against a scorer nobody chose",
                extra=log_fields(configured=configured, discovered=identity),
            )
            raise ScorerIdentityMismatch(
                configured, identity, context="validator.scorer_version operator pin"
            )
        if self.pinned_scorer_version and self.pinned_scorer_version != identity:
            # Latch the conflict BEFORE raising: whether or not the caller
            # survives the exception, this process must not score again.
            self.scorer_pin_conflict = (
                f"the scoring worker now advertises {identity!r} but this"
                f" validator is pinned to {self.pinned_scorer_version!r}"
            )
            self._publish_pin_metric()
            self.log.critical(
                "SCORER IDENTITY CHANGED under a pinned validator: refusing to"
                " score. Folding two scorers' packets into one EWMA accumulator is"
                " the drift the pin exists to prevent; an operator must resolve it"
                " (validator.reset_scorer_pin acknowledges the change)",
                extra=log_fields(
                    pinned=self.pinned_scorer_version, discovered=identity
                ),
            )
            raise ScorerIdentityMismatch(
                self.pinned_scorer_version,
                identity,
                context="the scoring worker changed identity under a pinned validator",
            )
        if self.pinned_scorer_version is None:
            self.log.info(
                "scorer identity pinned on first contact; every packet must carry it",
                extra=log_fields(
                    scorer_version=identity, operator_pin=bool(configured)
                ),
            )
        self.pinned_scorer_version = identity
        self._persist_scorer_pin(
            identity, source="operator" if configured else "discovered"
        )
        # The LIVE worker has now confirmed who it is, in this process.
        self._identity_verified = self.scorer_pin_conflict is None
        self._publish_pin_metric()
        return identity

    def _persist_scorer_pin(self, identity: str, *, source: str) -> None:
        """Make the pin outlive the process. A write failure must not score blind."""
        try:
            miner_manager.record_scorer_pin(
                self.conn,
                identity,
                pinned_at=miner_manager.utc_now_iso(),
                source=source,
            )
        except Exception as exc:
            self.scorer_pin_conflict = f"the scorer pin could not be persisted: {exc}"
            self.log.critical(
                "the durable scorer pin could not be written — refusing to score:"
                " an unpersisted pin cannot protect the accumulators across a"
                " restart, which is the whole point of pinning",
                extra=log_fields(scorer_version=identity, error=str(exc)),
            )

    async def _ensure_scorer_identity(self) -> None:
        """Pin on first contact AND verify a reloaded pin against the live worker.

        Round-2 review new-2: a restart that reloads the DURABLE pin must still ask
        the worker who it is — otherwise a swap onto a different scoring worker is
        invisible for the whole life of the process and both scorers' packets end
        up in one accumulator. Verified once per process; retried every round
        while the worker is unreachable.

        A disagreement is latched as a conflict by `discover_scorer_identity` and
        swallowed here: the round then SKIPS with a structured reason instead of
        crashing the loop (startup keeps failing loudly — see `run`).
        """
        if self._identity_verified or self.scorer_pin_conflict is not None:
            return
        with contextlib.suppress(ScorerIdentityMismatch, ScorerRuntimeMismatch):
            await self.discover_scorer_identity()

    # -- one full round --------------------------------------------------------

    async def run_round(self) -> RoundReport:
        start = self._clock()
        report = RoundReport()
        # Reconnect-safe: a worker that was down at startup is picked up here,
        # and an operator-pin disagreement still fails loudly rather than
        # accumulating packets from a scorer nobody chose.
        await self._ensure_scorer_identity()

        # -- scorer-identity gate ------------------------
        # A round with no pin would accumulate packets bound to NOTHING: any
        # scorer's output would be accepted, and a later restart onto a different
        # worker would fold both into the same EWMA. Skip instead.
        if not self.scoring_pin():
            return self._skip_round(
                report,
                "scorer_pin_conflict"
                if self.scorer_pin_conflict
                else "scorer_identity_unknown",
                detail=self.scorer_pin_conflict or "",
                scoring_worker_url=self.config.scoring_worker_url,
            )

        # -- chain state gate --------------------------------------
        self._maybe_refresh_chain()
        reason = self._chain_state_reason()
        if reason is not None:
            return self._skip_round(report, reason)
        try:
            block = self.chain.current_block()
            miners = self.eligible_neurons()
        except Exception as exc:
            # A never-refreshed adapter raises ChainStateUnavailable rather than
            # serving an empty snapshot; an empty round is NOT a successful round.
            return self._skip_round(
                report, "chain_state_unavailable", error=f"{type(exc).__name__}: {exc}"
            )

        round_id = uuid.uuid4().hex
        report.round_id = round_id
        miner_manager.begin_round(
            self.conn, round_id, block, miner_manager.utc_now_iso()
        )
        purged: list[int] = []
        try:
            # review #9 (round 2): the registry sync, the retention fold and the
            # warrant tracks are STAGED, not written — a reader must never see a
            # hotkey reset (or a new miner, or a probed track) belonging to a round
            # that then dies. `planned_tracks` gives this round the post-sync view
            # it needs without publishing any of it.
            planned = miner_manager.planned_tracks(self.conn, miners)
            probed: dict[int, str] = {}
            unknown = [n for n in miners if planned.get(n.uid) is None]
            if unknown:
                for uid, track in await asyncio.gather(
                    *(self._probe_warrant(n) for n in unknown)
                ):
                    if track is not None:
                        probed[uid] = track

            by_track: dict[str, list[ChainNeuron]] = {}
            for n in miners:
                track = probed.get(n.uid) or planned.get(n.uid)
                if track is None:
                    # The deliberate fix of validator.py:844: unknown track -> SKIP
                    # the miner for this round. No default bucket exists.
                    report.skipped_unknown_track.append(n.uid)
                    self.m_skipped_unknown.inc()
                    self.log.warning(
                        "miner skipped this round: warrant track unknown (never defaulted)",
                        extra=log_fields(uid=n.uid, hotkey=n.hotkey),
                    )
                    continue
                by_track.setdefault(track, []).append(n)

            scores: dict[int, float] = {}
            evidence: list[PacketEvidence] = []
            availability: list[AvailabilityObservation] = []
            for track in sorted(by_track):
                track_scores = await self._run_track(
                    track,
                    by_track[track],
                    report,
                    round_id=round_id,
                    evidence=evidence,
                    availability=availability,
                )
                scores.update(track_scores)

            # ONE transaction for the WHOLE round: registry sync + retention fold
            # + warrant tracks + EWMA folds + packet evidence + the ledger stamp
            # that makes this round readable. Nothing this round did is
            # observable before this call, and everything is after it.
            purged = miner_manager.commit_round(
                self.conn,
                round_id,
                scores=scores,
                decay=self.tokenomics.ewma_decay,
                packets=[e.as_row() for e in evidence],
                availability_observations=[o.as_row() for o in availability],
                committed_at=miner_manager.utc_now_iso(),
                registry=miner_manager.RegistryUpdate(
                    neurons=tuple(miners),
                    block=block,
                    tracks=probed,
                ),
            )
            self.m_packets_persisted.inc(len(evidence))
            if purged:
                self.log.info(
                    "hotkey changes purged", extra=log_fields(uids=purged, block=block)
                )
        finally:
            # review #5: every challenge this round fetched is resolved — on
            # success, on failure, on timeout AND on shutdown. Anything still
            # in-flight after a hard crash is drained by the startup recovery pass.
            await self._drain_inflight_challenges(report)

        self._last_round_at = self._clock()
        self.m_rounds.inc()
        self.m_round_duration.observe(self._last_round_at - start)
        self.log.info(
            "round complete",
            extra=log_fields(
                round_id=round_id,
                block=block,
                scored=len(report.scored),
                zeroed=len(report.zeroed),
                non_punitive_skips=len(report.non_punitive_skips),
                skipped_unknown_track=len(report.skipped_unknown_track),
                scoring_failed=len(report.scoring_failed),
                rejected_packets=len(report.rejected_packets),
                audit_store_failed=len(report.audit_store_failed),
                packets_persisted=len(evidence),
                availability_folds=len(availability),
                scorer_version=self.scoring_pin(),
                # DB-only is legitimate but must never be invisible.
                evidence_mode="db_only" if self._store is None else "audit_store",
            ),
        )
        return report

    async def _run_track(
        self,
        track: str,
        neurons: list[ChainNeuron],
        report: RoundReport,
        *,
        round_id: str,
        evidence: list[PacketEvidence],
        availability: list[AvailabilityObservation],
    ) -> dict[int, float]:
        cfg = self.config
        try:
            # Fetching the next challenge CONSUMES from the pool — not idempotent,
            # so timeout only, no retry.
            item = await with_timeout(
                self._next_challenge(track),
                cfg.challenge_request_timeout_seconds,
                f"challenge fetch track={track}",
            )
        except Exception as exc:
            self.log.error(
                "challenge fetch failed; track skipped this round",
                extra=log_fields(track=track, error=str(exc)),
            )
            return {}

        # The asset is now checked out and its commitment unrevealed: record the
        # obligation to resolve BEFORE doing anything that can fail,
        # WITH the identity it was actually fetched under —
        # that is the only identity the service will accept a resolve from, and
        # this process may not be that identity when it next starts.
        challenge_id = item.resolve_id
        miner_manager.record_inflight_challenge(
            self.conn,
            challenge_id=challenge_id,
            round_id=round_id,
            track=track,
            fetched_at=miner_manager.utc_now_iso(),
            owner=self._last_fetch_owner,
        )

        scores: dict[int, float] = {}
        responses = await self._dispatch_all(
            item,
            neurons,
            report,
            scores=scores,
            availability=availability,
        )
        try:
            neuron_by_uid = {n.uid: n for n in neurons}
            verified = await self._verify_digests(
                item,
                responses,
                report,
                neurons=neuron_by_uid,
                scores=scores,
                availability=availability,
            )
            dedup = await self._dedup(
                track,
                verified,
                report,
                hotkeys={uid: neuron.hotkey for uid, neuron in neuron_by_uid.items()},
                anchor_block_hash=(
                    item.commitment_anchor.block_hash
                    if item.commitment_anchor is not None
                    else None
                ),
            )
            for uid, response in dedup.kept:
                packet = await self._score_one(
                    item,
                    neuron_by_uid[uid],
                    response,
                    report,
                    scores=scores,
                    availability=availability,
                )
                if packet is None:
                    continue  # scoring infra failed / packet unbound — miner unharmed
                evidence.append(packet)
                scores[uid] = packet.score
                report.scored[uid] = packet.score
                self.m_scored.labels(track=track).inc()
            # A duplicate changes EWMA only after both real signed outputs have
            # been archived into a canonical witness. Any missing/invalid evidence
            # converts the claimed penalty into a visible non-punitive skip.
            for duplicate in dedup.duplicates:
                uid = duplicate.loser_uid
                try:
                    packet = self._attribute_duplicate(
                        item,
                        neuron_by_uid[duplicate.loser_uid],
                        neuron_by_uid[duplicate.winner_uid],
                        duplicate,
                    )
                except (AuditStoreFailure, InvalidDuplicateEvidence) as exc:
                    report.non_punitive_skips[uid] = "duplicate_evidence_unavailable"
                    if uid not in report.scoring_failed:
                        report.scoring_failed.append(uid)
                    if (
                        isinstance(exc, AuditStoreFailure)
                        and uid not in report.audit_store_failed
                    ):
                        report.audit_store_failed.append(uid)
                    self.log.error(
                        "duplicate verdict lacked independently auditable signed evidence;"
                        " penalty discarded",
                        extra=log_fields(
                            uid=uid,
                            duplicate_of=duplicate.winner_uid,
                            error=str(exc),
                            violation="DUPLICATE_EVIDENCE_UNAVAILABLE",
                        ),
                    )
                    continue
                evidence.append(packet)
                scores[uid] = 0.0
                report.zeroed[uid] = "duplicate"
                self.m_scored.labels(track=track).inc()
            # This track's scoring ran to completion: the challenge is 'resolved',
            # not 'expired'. The actual resolve call happens in run_round's finally.
            miner_manager.set_inflight_outcome(self.conn, challenge_id, "resolved")
            return scores
        finally:
            # Digest failures, dedup losers, scoring errors, audit-store errors,
            # successful archive, and cancellation all converge here. Scoring and
            # `_archive_media` have finished (or failed) before deletion.
            for _, response, _ in responses:
                self._discard_miner_artifact(response)

    async def _next_challenge(self, track: str) -> ChallengeItem:
        """`/challenge/next` STAMPED WITH OUR IDENTITY.

        The owner is what lets the service tell this validator's dispatched
        challenges from another's, which is what makes the orphan sweep safe.
        Feature-detected: a client predating the parameter is called the old way,
        and the sweep then finds nothing attributed to us and expires nothing.

        Records `_last_fetch_owner` = the identity the fetch ACTUALLY carried (''
        for an anonymous/feature-detected fetch), which `_run_track` persists on
        the in-flight row. The recovery pass must resolve as the fetcher, not as
        whoever this process is configured to be later.
        """
        owner = self.config.identity.strip()
        self._last_fetch_owner = ""
        if owner:
            try:
                item = await self.challenge_client.next_challenge(track, owner=owner)
            except TypeError:
                self.log.warning(
                    "the challenge client does not accept an owner — challenges"
                    " will not be attributed to this validator and the orphan"
                    " sweep stays disabled",
                    extra=log_fields(track=track, owner=owner),
                )
            else:
                self._last_fetch_owner = owner
                return item
        return await self.challenge_client.next_challenge(track)

    # -- challenge resolution ---------------------------------------

    async def _resolve_challenge(
        self, challenge_id: str, outcome: str, *, label: str, owner: str | None = None
    ) -> None:
        """`/challenge/{id}/resolve` STAMPED WITH OUR IDENTITY.

        The mirror image of `_next_challenge`: the service records the owner when
        it dispatches and then ENFORCES it on resolve (403 `not_owner`), so a
        validator that produced a challenge under `owner=<identity>` and then
        resolved it anonymously would be refused on every one of its OWN
        challenges — the in-flight row would never drain and the asset would stay
        checked out forever.

        Feature-detected the same way: a client predating the parameter is called
        the old way (and is talking to a service old enough not to enforce it).
        Raises ChallengeAlreadyTerminal / whatever the client raises; the callers
        own the retry policy.

        `owner` overrides the configured identity with the one PERSISTED on the
        in-flight row. Empty/None falls back to the current
        config, which is right for rows written before ownership was recorded and
        for the sweep (whose candidates are, by construction, ours right now).
        """
        owner = (owner if owner else self.config.identity).strip()
        timeout = self.config.challenge_resolve_timeout_seconds
        if owner:
            try:
                coro = self.challenge_client.resolve_challenge(
                    challenge_id, outcome, owner=owner
                )
            except TypeError:
                self.log.warning(
                    "the challenge client does not accept an owner — resolving"
                    " anonymously, which an ownership-enforcing service refuses",
                    extra=log_fields(challenge_id=challenge_id, owner=owner),
                )
            else:
                await with_timeout(coro, timeout, label)
                return
        await with_timeout(
            self.challenge_client.resolve_challenge(challenge_id, outcome),
            timeout,
            label,
        )

    async def _drain_inflight_challenges(
        self, report: RoundReport | None = None
    ) -> int:
        """Resolve every recorded in-flight challenge. Returns how many drained.

        Rows survive a failed resolve so the next round (or the next startup)
        retries them; the pool can never be permanently stranded by one flaky
        call. An unknown/already-terminal challenge is dropped, not retried.

        OWNERSHIP: each row is resolved as the identity that
        FETCHED it, which is not necessarily the identity this process is
        configured with — a rotation between the fetch and the recovery would
        otherwise 403 forever. A rotation is logged (WARNING) but does not change
        which owner is used, and an actual 403 is counted and parked for an
        operator instead of being retried against a boundary that cannot move.
        """
        drained = 0
        current = self.config.identity.strip()
        for row in miner_manager.inflight_challenges(self.conn):
            challenge_id = str(row["challenge_id"])
            outcome = str(row["outcome"])
            recorded = str(row["owner"] or "") if "owner" in row.keys() else ""
            if recorded and current and recorded != current:
                self.log.warning(
                    "in-flight challenge was fetched under a DIFFERENT identity;"
                    " resolving as the recorded owner (this validator's identity"
                    " rotated, or two validators share this database)",
                    extra=log_fields(
                        challenge_id=challenge_id,
                        recorded_owner=recorded,
                        current_identity=current,
                    ),
                )
            try:
                await self._resolve_challenge(
                    challenge_id,
                    outcome,
                    label=f"challenge resolve {challenge_id}",
                    owner=recorded or None,
                )
            except ChallengeOwnershipRefused as exc:
                # PERMANENT, and NOT a drained challenge: the row is kept (it is
                # the only record that an asset is stranded) but it is counted on
                # its own metric rather than filed with the flaky-resolve failures
                # that a later pass genuinely does fix, and it is PARKED durably
                # so no later round or restart re-attempts a
                # resolve that can never succeed. Visibility replaces retry: the
                # parked gauge, the startup recovery log and
                # miner_manager.parked_challenges() are the operator's signal
                # that this needs a human, not another round.
                self.m_challenge_resolve_forbidden.inc()
                miner_manager.park_inflight_challenge(
                    self.conn,
                    challenge_id,
                    parked_at=miner_manager.utc_now_iso(),
                    reason=f"403 not_owner as {(recorded or current)!r}: {exc}"[:500],
                )
                self._publish_parked_metric()
                self.log.warning(
                    "challenge resolve REFUSED (403 not_owner): the challenge"
                    " service does not accept this owner for this challenge. The"
                    " row is PARKED — it will not be retried by any round or"
                    " restart, because no number of retries can move an ownership"
                    " boundary. The service-side asset needs an operator; unpark"
                    " with validator.unpark_challenges once resolved",
                    extra=log_fields(
                        challenge_id=challenge_id,
                        outcome=outcome,
                        recorded_owner=recorded,
                        current_identity=current,
                        error=str(exc),
                    ),
                )
                continue
            except ChallengeAlreadyTerminal as exc:
                try:
                    self._release_retired_references(challenge_id)
                except Exception as release_exc:
                    self.log.error(
                        "terminal challenge holdout could not be published; keeping"
                        " the in-flight row so release is retried",
                        extra=log_fields(
                            challenge_id=challenge_id,
                            error=f"{type(release_exc).__name__}: {release_exc}",
                        ),
                    )
                    continue
                miner_manager.clear_inflight_challenge(self.conn, challenge_id)
                self.log.info(
                    "challenge already terminal at the service; in-flight row dropped",
                    extra=log_fields(challenge_id=challenge_id, detail=str(exc)),
                )
                continue
            except Exception as exc:
                self.m_challenge_resolve_failures.inc()
                self.log.error(
                    "challenge resolve failed; the asset stays in_use until this"
                    " row is drained by a later round or the startup recovery pass",
                    extra=log_fields(
                        challenge_id=challenge_id, outcome=outcome, error=str(exc)
                    ),
                )
                continue
            try:
                self._release_retired_references(challenge_id)
            except Exception as exc:
                # The resolve landed, but public recomputability did not. Keep
                # the durable obligation: the next pass receives already-terminal
                # and retries release before clearing it.
                self.log.error(
                    "challenge resolved but its retired holdout could not be"
                    " published; keeping the in-flight row for retry",
                    extra=log_fields(
                        challenge_id=challenge_id,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
                continue
            miner_manager.clear_inflight_challenge(self.conn, challenge_id)
            self.m_challenges_resolved.labels(outcome=outcome).inc()
            if report is not None:
                report.resolved_challenges[challenge_id] = outcome
            drained += 1
        return drained

    def _release_retired_references(self, challenge_id: str) -> None:
        """Publish this terminal challenge's sealed holdouts for public audit."""
        if self._store is None:
            return
        refs = ScorePacketEvidence(self.conn).reference_original_refs(challenge_id)
        if not refs:
            return
        release = getattr(self._store, "release", None)
        if not callable(release):
            raise AuditStoreFailure(
                "configured audit store has no post-retirement release capability"
            )
        for encoded in refs:
            try:
                ref = ArtifactRef.model_validate_json(encoded)
            except Exception as exc:
                raise AuditStoreFailure(
                    f"challenge {challenge_id} has an invalid reference_original_ref: {exc}"
                ) from exc
            if ref.kind is not ArtifactKind.REFERENCE_ORIGINAL:
                raise AuditStoreFailure(
                    f"challenge {challenge_id} reference is {ref.kind.value}, expected "
                    f"{ArtifactKind.REFERENCE_ORIGINAL.value}"
                )
            release(ref)

    async def sweep_orphaned_challenges(self) -> list[str]:
        """Expire dispatched challenges this validator has NO in-flight row for.

        The one gap the in-flight table cannot close: if the RESPONSE to
        `POST /challenge/next` is lost (connection reset, process killed between
        the service's commit and our `record_inflight_challenge`), the service
        holds a dispatched challenge whose id we never learned. Nothing to
        resolve, nothing to retry — the asset stays `in_use` and its commitment
        unrevealed forever, and a single-asset pool answers `pool_exhausted` from
        then on. The service's own `recover_orphans` cannot help either: the media
        is present and the challenge is perfectly valid, it is simply abandoned.

        So the validator ASKS: GET /challenges?status=dispatched&older_than_seconds
        (`orphan_sweep_age_seconds`, comfortably longer than a round so a live
        round's own challenge is never in the list), and anything on that list
        with no in-flight row of ours is resolved as `expired`.

        OWNERSHIP. "No in-flight row of ours" is not the
        same as "ours": a second validator's still-live challenge also has no row
        here, and expiring it kills its round mid-flight. The sweep therefore
        only ever touches challenges the SERVICE attributes to this validator:

        * no `validator.identity` configured -> the sweep is disabled outright;
        * the request carries `owner=<identity>`, and every listed row must NAME
          that owner. A row with no owner is NOT swept — an unknown query
          parameter is silently ignored by most HTTP frameworks, so an unfiltered
          list must never be mistaken for a filtered one.

        Feature-detected at both ends: a ChallengeClient without
        `list_dispatched`, or one that does not accept `owner`, skips the sweep
        rather than failing startup or expiring somebody else's work.
        """
        max_age = self.config.orphan_sweep_age_seconds
        if max_age <= 0:
            return []
        owner = self.config.identity.strip()
        if not owner:
            self.m_sweep_skipped.labels(reason="no_identity").inc()
            self.log.warning(
                "orphan sweep DISABLED: no validator.identity is configured, so"
                " this validator cannot tell its own dispatched challenges from"
                " another validator's and must expire none of them",
            )
            return []
        lister = getattr(self.challenge_client, "list_dispatched", None)
        if not callable(lister):
            return []
        try:
            dispatched = await with_timeout(
                lister(max_age, owner=owner),
                self.config.challenge_request_timeout_seconds,
                "dispatched-challenge sweep",
            )
        except TypeError:
            self.m_sweep_skipped.labels(reason="owner_unsupported").inc()
            self.log.warning(
                "orphan sweep SKIPPED: this challenge client cannot filter by"
                " owner, and an unscoped sweep could expire another validator's"
                " live challenge",
                extra=log_fields(owner=owner),
            )
            return []
        except Exception as exc:
            self.log.warning(
                "dispatched-challenge sweep unavailable; orphans (if any) stay"
                " checked out until the next startup",
                extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
            )
            return []
        # Parked rows are excluded from the DRAIN, not from existence: the sweep
        # must still treat them as ours-and-recorded, or it would expire a
        # challenge whose stranded row an operator is deliberately holding.
        known = {
            str(row["challenge_id"])
            for row in miner_manager.inflight_challenges(self.conn)
        } | {
            str(row["challenge_id"])
            for row in miner_manager.parked_challenges(self.conn)
        }
        expired: list[str] = []
        for entry in dispatched:
            challenge_id = str(getattr(entry, "challenge_id", "") or "")
            if not challenge_id or challenge_id in known:
                continue  # ours and already recorded: the normal drain owns it
            entry_owner = str(getattr(entry, "owner", "") or "")
            if entry_owner != owner:
                # Either another validator's challenge or a service that cannot
                # attribute ownership; both mean "not ours to expire".
                reason = "unattributed" if not entry_owner else "foreign_owner"
                self.m_sweep_skipped.labels(reason=reason).inc()
                self.log.info(
                    "dispatched challenge left alone by the sweep: it is not"
                    " provably ours",
                    extra=log_fields(
                        challenge_id=challenge_id,
                        listed_owner=entry_owner,
                        reason=reason,
                    ),
                )
                continue
            try:
                await self._resolve_challenge(
                    challenge_id, "expired", label=f"challenge expire {challenge_id}"
                )
            except ChallengeAlreadyTerminal:
                continue  # somebody else drained it between the list and now
            except Exception as exc:
                self.m_challenge_resolve_failures.inc()
                self.log.error(
                    "orphaned challenge could not be expired; it stays checked out",
                    extra=log_fields(challenge_id=challenge_id, error=str(exc)),
                )
                continue
            expired.append(challenge_id)
            self.m_orphans_swept.inc()
            self.m_challenges_resolved.labels(outcome="expired").inc()
            self.log.warning(
                "expired an orphaned dispatched challenge: the service issued it but"
                " this validator has no in-flight record of it (a lost"
                " /challenge/next response)",
                extra=log_fields(
                    challenge_id=challenge_id,
                    age_seconds=float(getattr(entry, "age_seconds", 0.0)),
                    track=str(getattr(entry, "track", "")),
                ),
            )
        return expired

    def _publish_parked_metric(self) -> list[Any]:
        """Refresh the parked gauge from the table; returns the parked rows."""
        rows = miner_manager.parked_challenges(self.conn)
        self.m_challenges_parked.set(float(len(rows)))
        return rows

    def unpark_challenges(self) -> list[str]:
        """OPERATOR PATH: retry every parked obligation.

        Returns each parked row to the normal drain selection so the next drain
        (a round's finally, or the startup recovery pass) attempts its resolve
        again. To be used AFTER the service-side ownership state has been fixed
        — or accepted; a refusal that still stands simply re-parks the row on
        its next 403. Reached from config via `validator.unpark_challenges =
        true` (applied at startup by `recover_inflight_challenges`). Returns the
        unparked challenge ids.
        """
        ids = miner_manager.unpark_inflight_challenges(self.conn)
        self._publish_parked_metric()
        if ids:
            self.log.warning(
                "parked challenge obligations UNPARKED by operator"
                " acknowledgement; the next drain retries their resolves",
                extra=log_fields(challenge_ids=ids),
            )
        return ids

    async def recover_inflight_challenges(self) -> int:
        """Startup pass: resolve challenges stranded by a crashed round.

        Without this, a process killed mid-round leaves its assets `in_use` and
        their commitments unrevealed forever, and the pool answers every later
        round with `pool_exhausted`.

        Two halves, because a crash can strand a challenge in two ways: the ones
        we RECORDED (drained from the in-flight table) and the ones we never
        learned the id of (`sweep_orphaned_challenges` — a lost /challenge/next
        response). The drain runs first so its rows are in `known` when the sweep
        decides what is nobody's.

        PARKED rows are honored, not drained: they are only
        returned to the drain when the operator has set
        `validator.unpark_challenges`, and whatever is (still) parked afterwards
        is surfaced on the gauge and in a startup log line so the stranded
        service-side assets are never silently forgotten.
        """
        if self.config.unpark_challenges:
            self.unpark_challenges()
        stranded = miner_manager.inflight_challenges(self.conn)
        partial = miner_manager.uncommitted_rounds(self.conn)
        if stranded or partial:
            self.log.warning(
                "recovering after an unclean shutdown",
                extra=log_fields(
                    stranded_challenges=[str(r["challenge_id"]) for r in stranded],
                    uncommitted_rounds=[str(r["round_id"]) for r in partial],
                ),
            )
        drained = await self._drain_inflight_challenges()
        await self.sweep_orphaned_challenges()
        parked = self._publish_parked_metric()
        if parked:
            self.log.warning(
                "PARKED challenge obligations exist: their resolves were"
                " positively refused (403 not_owner) and they will NOT be retried"
                " — each one is a service-side asset stranded in_use. Fix the"
                " ownership state, then set validator.unpark_challenges (or call"
                " unpark_challenges()) to retry them",
                extra=log_fields(
                    parked_challenges=[str(r["challenge_id"]) for r in parked],
                    park_reasons=[str(r["park_reason"] or "") for r in parked],
                ),
            )
        self._recovered_inflight = True
        return drained

    @staticmethod
    def task_id_for(item: ChallengeItem, uid: int) -> str:
        """THE task id for (challenge, miner) — authored here and nowhere else.

        review #6 (round 2): this is the only id the validator will accept back
        and the only one that ever reaches `ScoreRequest.item_id`. Deriving the
        scored id from the miner's echo made the binding circular — the packet
        was checked against a value the miner itself chose.
        """
        return f"{item.dispatch.challenge_id}:{uid}"

    def _availability_zero(
        self,
        neuron: ChainNeuron,
        request: MinerTaskRequest,
        reason: AvailabilityFailureReason,
        report: RoundReport,
        scores: dict[int, float],
        availability: list[AvailabilityObservation],
        *,
        exception: BaseException | None = None,
        response: MinerTaskResponse | None = None,
        returned_task_id: str | None = None,
        observed_output_digest: str | None = None,
    ) -> bool:
        """Record a zero only when the client can produce signed request evidence.

        The protocol taxonomy is closed by ``AvailabilityFailureReason``. System,
        scorer, storage and challenge failures never call this helper. A legacy or
        test client without artifact-v2 request proof remains an explicit excused skip;
        production guards already forbid unsigned artifact v1.
        """
        factory = getattr(self.miner_client, "availability_observation", None)
        observation = None
        if callable(factory):
            try:
                observation = factory(
                    neuron,
                    request,
                    reason,
                    exception=exception,
                    response=response,
                    returned_task_id=returned_task_id,
                    observed_output_digest=observed_output_digest,
                )
            except Exception as exc:
                self.log.error(
                    "miner failure could not be converted into canonical availability"
                    " evidence; skipped without economic effect",
                    extra=log_fields(
                        uid=neuron.uid,
                        task_id=request.task_id,
                        reason=reason.value,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
        if observation is None:
            report.non_punitive_skips[neuron.uid] = (
                f"{reason.value}_evidence_unavailable"
            )
            return False

        availability.append(observation)
        scores[neuron.uid] = 0.0
        report.zeroed[neuron.uid] = f"availability:{reason.value}"
        report.non_punitive_skips.pop(neuron.uid, None)
        self.log.warning(
            "miner-attributable failure recorded as an evidence-backed EWMA zero",
            extra=log_fields(
                uid=neuron.uid,
                hotkey=neuron.hotkey,
                task_id=request.task_id,
                reason=reason.value,
                observation_digest=observation.digest(),
            ),
        )
        return True

    async def _dispatch_all(
        self,
        item: ChallengeItem,
        neurons: list[ChainNeuron],
        report: RoundReport,
        *,
        scores: dict[int, float],
        availability: list[AvailabilityObservation],
    ) -> list[tuple[int, MinerTaskResponse, int]]:
        """Dispatch to every miner concurrently; (uid, response, arrival_order)."""
        cfg = self.config
        arrivals = iter(range(1_000_000_000))
        downloaded: list[MinerTaskResponse] = []

        async def one(neuron: ChainNeuron) -> tuple[int, MinerTaskResponse, int] | None:
            request = MinerTaskRequest(
                task_id=self.task_id_for(item, neuron.uid),
                track=item.track,
                input_path=item.miner_input_path,
                input_digest=item.miner_input_digest,
                params=item.params,
                commitment_anchor=item.commitment_anchor,
                deadline_seconds=cfg.miner_request_timeout_seconds,
            )
            try:
                response = await with_timeout(
                    self.miner_client.submit_task(neuron, request),
                    cfg.miner_request_timeout_seconds,
                    f"miner task uid={neuron.uid}",
                )
                downloaded.append(response)
            except MinerArtifactColdStart as exc:
                # A miner may legitimately need one fresh request after restart,
                # but an endpoint that returns 425 until the whole task deadline
                # has declined the round. The latest signed attempt makes that
                # miner-attributable zero independently auditable.
                self._availability_zero(
                    neuron,
                    request,
                    AvailabilityFailureReason.RESTART_FENCE_EXHAUSTED,
                    report,
                    scores,
                    availability,
                    exception=exc,
                )
                self.log.warning(
                    "miner restart fence did not clear inside the request deadline;"
                    " availability zero attempted",
                    extra=log_fields(uid=neuron.uid, error=str(exc)),
                )
                return None
            except TimeoutError as exc:
                self.m_miner_timeouts.inc()
                self._availability_zero(
                    neuron,
                    request,
                    AvailabilityFailureReason.TIMEOUT,
                    report,
                    scores,
                    availability,
                    exception=exc,
                )
                return None
            except MinerArtifactInputError as exc:
                report.non_punitive_skips[neuron.uid] = "validator_input_error"
                self.log.error(
                    "validator could not safely stream its local challenge input;"
                    " miner skipped without economic effect",
                    extra=log_fields(uid=neuron.uid, error=str(exc)),
                )
                return None
            except MinerPeerAddressError as exc:
                self._availability_zero(
                    neuron,
                    request,
                    AvailabilityFailureReason.UNREACHABLE_ENDPOINT,
                    report,
                    scores,
                    availability,
                    exception=exc,
                )
                return None
            except MinerArtifactIntegrityError as exc:
                self._availability_zero(
                    neuron,
                    request,
                    AvailabilityFailureReason.PROTOCOL_ERROR,
                    report,
                    scores,
                    availability,
                    exception=exc,
                )
                return None
            except MinerArtifactProtocolError as exc:
                self._availability_zero(
                    neuron,
                    request,
                    AvailabilityFailureReason.PROTOCOL_ERROR,
                    report,
                    scores,
                    availability,
                    exception=exc,
                )
                return None
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                reason = (
                    AvailabilityFailureReason.TIMEOUT
                    if isinstance(exc, httpx.TimeoutException)
                    else AvailabilityFailureReason.TRANSPORT_ERROR
                )
                self._availability_zero(
                    neuron,
                    request,
                    reason,
                    report,
                    scores,
                    availability,
                    exception=exc,
                )
                return None
            except Exception as exc:
                # An unknown exception is not protocol-enumerated and therefore may
                # be local/system trouble. Fail excused rather than mint an arbitrary
                # validator-attributed zero.
                report.non_punitive_skips[neuron.uid] = "system_error"
                self.log.warning(
                    "miner call failed outside the economic failure taxonomy;"
                    " skipped non-punitively",
                    extra=log_fields(uid=neuron.uid, error=str(exc)),
                )
                return None
            # review #6 (round 2): the response must answer the task we sent. A
            # miner echoing another task id is trying to have a different item
            # (its own earlier work, or another miner's) scored under this
            # dispatch. The response is unusable, but an authority-only assertion
            # cannot support an economic penalty, so it is skipped.
            if response.task_id != request.task_id:
                self.m_task_id_mismatch.inc()
                self._availability_zero(
                    neuron,
                    request,
                    AvailabilityFailureReason.TASK_ID_MISMATCH,
                    report,
                    scores,
                    availability,
                    response=response,
                    returned_task_id=response.task_id,
                )
                self.log.warning(
                    "miner answered with a task id we never dispatched; response"
                    " discarded and availability zero recorded",
                    extra=log_fields(
                        uid=neuron.uid,
                        dispatched_task_id=request.task_id,
                        returned_task_id=response.task_id,
                        violation="MINER_TASK_ID_MISMATCH",
                    ),
                )
                self._discard_miner_artifact(response)
                return None
            return neuron.uid, response, next(arrivals)

        try:
            results = await asyncio.gather(*(one(n) for n in neurons))
        except BaseException:
            # Cancellation or an unexpected caller-side exception must not
            # strand outputs downloaded by sibling dispatches that already
            # completed. The containment proof ignores fake/local paths outside
            # this validator's configured landing zone.
            for response in downloaded:
                self._discard_miner_artifact(response)
            raise
        return [r for r in results if r is not None]

    def _discard_miner_artifact(self, response: MinerTaskResponse) -> None:
        discard_downloaded_artifact(
            response.output_path, self.config.miner_artifact_dir
        )

    async def _verify_digests(
        self,
        item: ChallengeItem,
        responses: list[tuple[int, MinerTaskResponse, int]],
        report: RoundReport,
        *,
        neurons: dict[int, ChainNeuron],
        scores: dict[int, float],
        availability: list[AvailabilityObservation],
    ) -> list[tuple[int, MinerTaskResponse, int]]:
        verified: list[tuple[int, MinerTaskResponse, int]] = []
        for uid, response, order in responses:
            try:
                actual = await asyncio.to_thread(sha256_file, response.output_path)
            except OSError as exc:
                actual = ""
                detail = f"unreadable output: {exc}"
            else:
                detail = "digest mismatch"
            if actual != response.output_digest:
                self.m_digest_mismatch.inc()
                if not actual:
                    report.non_punitive_skips[uid] = "validator_output_read_error"
                    self.log.error(
                        "validator could not read its downloaded artifact; miner"
                        " skipped without economic effect",
                        extra=log_fields(
                            uid=uid,
                            task_id=response.task_id,
                            detail=detail,
                        ),
                    )
                    continue
                request = MinerTaskRequest(
                    task_id=self.task_id_for(item, uid),
                    track=item.track,
                    input_path=item.miner_input_path,
                    input_digest=item.miner_input_digest,
                    params=item.params,
                    commitment_anchor=item.commitment_anchor,
                    deadline_seconds=self.config.miner_request_timeout_seconds,
                )
                self._availability_zero(
                    neurons[uid],
                    request,
                    AvailabilityFailureReason.OUTPUT_DIGEST_MISMATCH,
                    report,
                    scores,
                    availability,
                    response=response,
                    observed_output_digest=actual,
                )
                self.log.warning(
                    "output digest verification failed; availability zero recorded",
                    extra=log_fields(
                        uid=uid,
                        task_id=response.task_id,
                        claimed=response.output_digest,
                        actual=actual,
                        detail=detail,
                        violation="OUTPUT_DIGEST_MISMATCH",
                    ),
                )
                continue
            verified.append((uid, response, order))
        return verified

    async def _dedup(
        self,
        track: str,
        responses: list[tuple[int, MinerTaskResponse, int]],
        report: RoundReport,
        *,
        hotkeys: dict[int, str],
        anchor_block_hash: str | None,
    ) -> DedupOutcome:
        """Economically dedup only equal, independently verified SHA-256 bytes.

        Perceptual decoding is absent from this critical path: honest
        restorations preserve the same scene and naturally look similar.
        Corpus-ingest pHash remains a separate producer-side protection.
        """
        if not responses:
            return DedupOutcome(kept=(), duplicates=())

        groups: dict[str, list[tuple[int, MinerTaskResponse]]] = {}
        for uid, response, _ in responses:
            groups.setdefault(response.output_digest, []).append((uid, response))

        kept: list[tuple[int, MinerTaskResponse]] = []
        duplicates: list[DuplicateCandidate] = []
        for digest_group in groups.values():
            if len(digest_group) == 1:
                kept.append(digest_group[0])
                continue
            if anchor_block_hash is None:
                for uid, _ in digest_group:
                    report.non_punitive_skips[uid] = "duplicate_anchor_unavailable"
                self.log.error(
                    "exact duplicate group has no anchor salt; skipped non-punitively",
                    extra=log_fields(track=track),
                )
                continue
            try:
                ranked = sorted(
                    digest_group,
                    key=lambda pair: duplicate_order_key(
                        anchor_block_hash, hotkeys[pair[0]]
                    ),
                )
                order_keys = [
                    duplicate_order_key(anchor_block_hash, hotkeys[uid])
                    for uid, _ in ranked
                ]
            except (InvalidDuplicateEvidence, KeyError) as exc:
                for uid, _ in digest_group:
                    report.non_punitive_skips[uid] = "duplicate_identity_unavailable"
                self.log.error(
                    "exact duplicate identity ordering unavailable; group skipped",
                    extra=log_fields(track=track, error=str(exc)),
                )
                continue
            if len(set(order_keys)) != len(order_keys):
                for uid, _ in digest_group:
                    report.non_punitive_skips[uid] = "duplicate_identity_collision"
                self.log.error(
                    "exact duplicate identities collide under deterministic ordering;"
                    " group skipped non-punitively",
                    extra=log_fields(track=track),
                )
                continue

            winner_uid, winner_response = ranked[0]
            kept.append((winner_uid, winner_response))
            for uid, response in ranked[1:]:
                duplicates.append(
                    DuplicateCandidate(
                        loser_uid=uid,
                        loser_response=response,
                        winner_uid=winner_uid,
                        winner_response=winner_response,
                    )
                )
                self.log.warning(
                    "byte-exact duplicate awaits signed two-output audit evidence",
                    extra=log_fields(
                        uid=uid,
                        track=track,
                        duplicate_of=winner_uid,
                        reason="exact_output_digest",
                    ),
                )
        return DedupOutcome(kept=tuple(kept), duplicates=tuple(duplicates))

    def _binding_mismatch(self, packet: ItemScore, request: ScoreRequest) -> str | None:
        """The first request-binding field the packet does not match, else None.

        review #6: verifying the packet's SELF-REPORTED sha256 proves only that
        the bytes and the digest agree — a compromised or MITM'd scoring endpoint
        can hand back a *valid* high-score packet belonging to a different miner,
        item or challenge and have it accumulated for this uid. The packet is
        accepted only when it is bound to exactly what this validator asked for.
        """
        for name, want in self._expected_bindings(request).items():
            if getattr(packet, name, None) != want:
                return name
        return None

    def _expected_bindings(self, request: ScoreRequest) -> dict[str, object]:
        """The packet fields that MUST equal what this validator sent/expects.

        `scorer_version` is bound to the PINNED identity — the one discovered
        from the worker's GET /healthz on first contact, or the operator pin in
        `validator.scorer_version` (which discovery had to agree with). Once a
        pin exists, a packet stamped by any other scorer is rejected: the audit
        bundle later cross-checks packet against bundle scorer version, so a
        silent substitution would surface only as an unexplained audit failure.

        A round only runs with a pin, so `scorer_version`
        is always among the bound fields here; the conditional remains only so
        the helper stays usable from tests that build a request by hand.
        """
        expected: dict[str, object] = {
            "challenge_id": request.challenge_id,
            "item_id": request.item_id,
            "miner_hotkey": request.miner_hotkey,
            "track": request.track,
            "content_digest": request.output_digest,
        }
        pinned = self.scoring_pin()
        if pinned:
            expected["scorer_version"] = pinned
        return expected

    def _archive_packet(self, packet_json: str, packet_digest: str) -> str | None:
        """Store the packet bytes as a SCORE_PACKET artifact.

        Returns the backend key, or None in DB-ONLY mode (no store configured —
        a legitimate, explicitly-flagged deployment). When a store IS configured,
        a failure RAISES `AuditStoreFailure`: an internal review round 2, an audit-store
        write that fails must fail the item CLOSED rather than committing a score
        whose evidence exists nowhere a third party can reach. Fail-open here
        produced exactly the "audited" weights nobody could reproduce.
        """
        if self._store is None:
            return None
        try:
            ref = self._store.put(
                packet_json.encode("utf-8"), ArtifactKind.SCORE_PACKET
            )
        except Exception as exc:
            self.m_audit_store_failures.inc()
            raise AuditStoreFailure(
                f"SCORE_PACKET {packet_digest} could not be archived: {exc}"
            ) from exc
        if ref.digest != packet_digest:
            # Content addressing makes this impossible unless the digest we
            # verified was computed over different bytes — surface it loudly and
            # do NOT accept the score: the evidence would not match the packet.
            self.m_audit_store_failures.inc()
            raise AuditStoreFailure(
                "archived score packet digest differs from the verified digest:"
                f" stored={ref.digest} verified={packet_digest}"
            )
        return ref.backend_key

    def _archive_media(
        self, request: ScoreRequest
    ) -> tuple[str | None, str | None, str | None]:
        """Archive the exact three media legs required for independent recompute."""
        if self._store is None:
            return None, None, None
        specs = (
            (
                "challenge_input",
                request.miner_input_path,
                request.miner_input_digest,
                ArtifactKind.CHALLENGE_INPUT,
            ),
            (
                "miner_output",
                request.output_path,
                request.output_digest,
                ArtifactKind.MINER_OUTPUT,
            ),
            (
                "reference_original",
                request.reference_path,
                request.reference_digest,
                ArtifactKind.REFERENCE_ORIGINAL,
            ),
        )
        encoded: list[str] = []
        try:
            for name, path, expected, kind in specs:
                ref = self._store.put_file(path, kind)
                if ref.digest != expected:
                    raise AuditStoreFailure(
                        f"archived {name} digest differs from the request binding: "
                        f"stored={ref.digest} expected={expected}"
                    )
                encoded.append(ref.model_dump_json())
        except AuditStoreFailure:
            self.m_audit_store_failures.inc()
            raise
        except Exception as exc:
            self.m_audit_store_failures.inc()
            raise AuditStoreFailure(
                f"full audit media could not be archived: {exc}"
            ) from exc
        return encoded[0], encoded[1], encoded[2]

    def _bound_receipt(
        self,
        item: ChallengeItem,
        neuron: ChainNeuron,
        response: MinerTaskResponse,
        *,
        required: bool,
    ) -> MinerArtifactReceipt | None:
        raw = response.artifact_receipt
        if raw is None:
            if required or item.commitment_anchor is not None:
                raise InvalidDuplicateEvidence(
                    "miner-signed artifact receipt is missing"
                )
            return None
        try:
            receipt = MinerArtifactReceipt.model_validate(raw)
            output_size = Path(response.output_path).stat().st_size
            input_size = Path(item.miner_input_path).stat().st_size
        except Exception as exc:
            raise InvalidDuplicateEvidence(
                f"miner artifact receipt is malformed: {exc}"
            ) from exc
        if (
            receipt.metadata.commitment_anchor != item.commitment_anchor
            or receipt.metadata.task_id != self.task_id_for(item, neuron.uid)
            or receipt.metadata.input_digest != item.miner_input_digest
            or receipt.metadata.track != item.track
            or receipt.miner_hotkey != neuron.hotkey
            or receipt.output_digest != response.output_digest
            or receipt.output_size != output_size
            or receipt.input_size != input_size
        ):
            raise InvalidDuplicateEvidence(
                "miner artifact receipt is not bound to the dispatched challenge,"
                " identity and media"
            )
        return receipt

    def _archive_duplicate_media(
        self,
        item: ChallengeItem,
        duplicate: DuplicateCandidate,
    ) -> tuple[str, str, str, ArtifactRef]:
        """Archive loser bundle media plus the second signed output witness."""
        if self._store is None:
            raise AuditStoreFailure(
                "duplicate penalties require a content-addressed audit store"
            )
        specs = (
            (
                "challenge_input",
                item.miner_input_path,
                item.miner_input_digest,
                ArtifactKind.CHALLENGE_INPUT,
            ),
            (
                "loser_output",
                duplicate.loser_response.output_path,
                duplicate.loser_response.output_digest,
                ArtifactKind.MINER_OUTPUT,
            ),
            (
                "reference_original",
                item.reference_path,
                item.reference_digest,
                ArtifactKind.REFERENCE_ORIGINAL,
            ),
            (
                "winner_output",
                duplicate.winner_response.output_path,
                duplicate.winner_response.output_digest,
                ArtifactKind.MINER_OUTPUT,
            ),
        )
        refs: list[ArtifactRef] = []
        try:
            for name, path, expected, kind in specs:
                ref = self._store.put_file(path, kind)
                if ref.digest != expected:
                    raise AuditStoreFailure(
                        f"archived {name} digest differs from response binding:"
                        f" stored={ref.digest} expected={expected}"
                    )
                refs.append(ref)
        except AuditStoreFailure:
            self.m_audit_store_failures.inc()
            raise
        except Exception as exc:
            self.m_audit_store_failures.inc()
            raise AuditStoreFailure(
                f"duplicate audit media could not be archived: {exc}"
            ) from exc
        return (
            refs[0].model_dump_json(),
            refs[1].model_dump_json(),
            refs[2].model_dump_json(),
            refs[3],
        )

    def _attribute_duplicate(
        self,
        item: ChallengeItem,
        loser: ChainNeuron,
        winner: ChainNeuron,
        duplicate: DuplicateCandidate,
    ) -> PacketEvidence:
        """Mint one exact-only zero backed by two real signed outputs."""
        loser_receipt = self._bound_receipt(
            item, loser, duplicate.loser_response, required=True
        )
        winner_receipt = self._bound_receipt(
            item, winner, duplicate.winner_response, required=True
        )
        assert loser_receipt is not None and winner_receipt is not None
        (
            challenge_input_ref,
            miner_output_ref,
            reference_original_ref,
            winner_output_ref,
        ) = self._archive_duplicate_media(item, duplicate)
        packet = mint_duplicate_packet(
            item_id=self.task_id_for(item, loser.uid),
            challenge_id=item.dispatch.challenge_id,
            track=item.track,
            loser_uid=loser.uid,
            loser_hotkey=loser.hotkey,
            loser_output_digest=duplicate.loser_response.output_digest,
            loser_output_size=Path(duplicate.loser_response.output_path).stat().st_size,
            loser_receipt=loser_receipt,
            winner_uid=winner.uid,
            winner_hotkey=winner.hotkey,
            winner_output=winner_output_ref,
            winner_receipt=winner_receipt,
            committed_scorer_version=self.scoring_pin(),
            config=self.scoring,
        )
        packet_json = packet.to_json()
        packet_digest = hashlib.sha256(packet_json.encode("utf-8")).hexdigest()
        audit_ref = self._archive_packet(packet_json, packet_digest)
        return PacketEvidence(
            uid=loser.uid,
            item_id=packet.item_id,
            challenge_id=packet.challenge_id,
            track=packet.track,
            miner_hotkey=loser.hotkey,
            content_digest=duplicate.loser_response.output_digest,
            packet_digest=packet_digest,
            packet_json=packet_json,
            scorer_version=packet.scorer_version or "",
            score=0.0,
            audit_ref=audit_ref,
            challenge_input_ref=challenge_input_ref,
            miner_output_ref=miner_output_ref,
            reference_original_ref=reference_original_ref,
            miner_receipt_json=loser_receipt.model_dump_json(),
        )

    async def _score_one(
        self,
        item: ChallengeItem,
        neuron: ChainNeuron,
        response: MinerTaskResponse,
        report: RoundReport,
        *,
        scores: dict[int, float],
        availability: list[AvailabilityObservation],
    ) -> PacketEvidence | None:
        cfg = self.config
        dispatch_request = MinerTaskRequest(
            task_id=self.task_id_for(item, neuron.uid),
            track=item.track,
            input_path=item.miner_input_path,
            input_digest=item.miner_input_digest,
            params=item.params,
            commitment_anchor=item.commitment_anchor,
            deadline_seconds=cfg.miner_request_timeout_seconds,
        )
        receipt_json: str | None = None
        if response.artifact_receipt is not None:
            try:
                receipt = MinerArtifactReceipt.model_validate(response.artifact_receipt)
            except Exception as exc:
                self._availability_zero(
                    neuron,
                    dispatch_request,
                    AvailabilityFailureReason.RECEIPT_INVALID,
                    report,
                    scores,
                    availability,
                    response=response,
                )
                self.log.warning(
                    "miner artifact receipt is malformed; availability zero attempted",
                    extra=log_fields(uid=neuron.uid, error=str(exc)),
                )
                return None
            if (
                receipt.metadata.commitment_anchor != item.commitment_anchor
                or receipt.metadata.task_id != self.task_id_for(item, neuron.uid)
                or receipt.metadata.input_digest != item.miner_input_digest
                or receipt.metadata.track != item.track
                or receipt.miner_hotkey != neuron.hotkey
                or receipt.output_digest != response.output_digest
            ):
                self._availability_zero(
                    neuron,
                    dispatch_request,
                    AvailabilityFailureReason.RECEIPT_INVALID,
                    report,
                    scores,
                    availability,
                    response=response,
                )
                self.log.warning(
                    "miner artifact receipt is not bound to the dispatched challenge;"
                    " availability zero attempted",
                    extra=log_fields(uid=neuron.uid),
                )
                return None
            receipt_json = receipt.model_dump_json()
        elif item.commitment_anchor is not None:
            self._availability_zero(
                neuron,
                dispatch_request,
                AvailabilityFailureReason.RECEIPT_INVALID,
                report,
                scores,
                availability,
                response=response,
            )
            self.log.warning(
                "externally anchored challenge has no miner-signed artifact receipt;"
                " availability zero attempted",
                extra=log_fields(uid=neuron.uid),
            )
            return None
        request = ScoreRequest(
            track=item.track,
            challenge_id=item.dispatch.challenge_id,
            # THE VALIDATOR'S id, never the miner's echo of it.
            # `_dispatch_all` has already rejected any response that did not carry
            # exactly this value, so the two agree by construction — but the id
            # that is scored, bound and persisted originates here.
            item_id=self.task_id_for(item, neuron.uid),
            miner_hotkey=neuron.hotkey,
            reference_path=item.reference_path,
            reference_digest=item.reference_digest,
            miner_input_path=item.miner_input_path,
            miner_input_digest=item.miner_input_digest,
            output_path=response.output_path,
            output_digest=response.output_digest,
            params=item.params,
            # Only an EXPLICIT OPERATOR PIN is asserted on the wire (so the worker
            # itself 409s a stranger). Under pin-on-first-contact the field is
            # omitted — asserting the discovered identity back at the worker that
            # just published it proves nothing, while the packet's own
            # scorer_version is bound in _expected_bindings either way.
            scorer_version=cfg.scorer_version or None,
        )
        try:
            # Scoring is a pure recompute over pinned digests — idempotent, retry-safe.
            result = await retry_async(
                lambda: with_timeout(
                    self.scoring_client.score(request),
                    cfg.scoring_request_timeout_seconds,
                    f"score uid={neuron.uid}",
                ),
                attempts=2,
                base_delay=0.1,
            )
        except Exception as exc:
            report.scoring_failed.append(neuron.uid)
            self.log.error(
                "scoring worker failed; miner not accumulated this round",
                extra=log_fields(uid=neuron.uid, error=str(exc)),
            )
            return None
        expected = hashlib.sha256(result.item_score_json.encode("utf-8")).hexdigest()
        if expected != result.packet_digest:
            report.scoring_failed.append(neuron.uid)
            self.log.error(
                "score packet digest mismatch from scoring worker; result discarded",
                extra=log_fields(uid=neuron.uid, claimed=result.packet_digest),
            )
            return None
        try:
            packet = ItemScore.from_json(result.item_score_json)
        except Exception as exc:
            report.scoring_failed.append(neuron.uid)
            report.rejected_packets[neuron.uid] = "unparseable"
            self.m_packet_rejected.labels(field="unparseable").inc()
            self.log.warning(
                "score packet could not be parsed; result discarded",
                extra=log_fields(uid=neuron.uid, error=str(exc)),
            )
            return None
        mismatch = self._binding_mismatch(packet, request)
        if mismatch is not None:
            # NOT punitive: a bad packet is validator-side infra trouble, so the
            # miner is skipped for the round rather than zeroed (same discipline
            # as a scoring-worker outage).
            report.scoring_failed.append(neuron.uid)
            report.rejected_packets[neuron.uid] = mismatch
            self.m_packet_rejected.labels(field=mismatch).inc()
            self.log.warning(
                "score packet is not bound to this validator's request — REJECTED;"
                " not accumulated (replayed or foreign packet)",
                extra=log_fields(
                    uid=neuron.uid,
                    field=mismatch,
                    expected=str(self._expected_bindings(request)[mismatch]),
                    got=str(getattr(packet, mismatch, None)),
                    packet_digest=result.packet_digest,
                    violation="SCORE_PACKET_NOT_BOUND",
                ),
            )
            return None
        try:
            audit_ref = self._archive_packet(
                result.item_score_json, result.packet_digest
            )
            challenge_input_ref, miner_output_ref, reference_original_ref = (
                self._archive_media(request)
            )
        except AuditStoreFailure as exc:
            # FAIL CLOSED. Non-punitive, like any other
            # validator-side infra failure: the miner is simply not accumulated.
            report.scoring_failed.append(neuron.uid)
            report.audit_store_failed.append(neuron.uid)
            self.log.error(
                "score packet could not be archived while an audit store IS"
                " configured — the score is DISCARDED, not committed unaudited:"
                " a weight nobody can reproduce from the audit store is worse than"
                " a missing one",
                extra=log_fields(
                    uid=neuron.uid,
                    packet_digest=result.packet_digest,
                    error=str(exc),
                    violation="SCORE_PACKET_UNARCHIVED",
                ),
            )
            return None
        return PacketEvidence(
            uid=neuron.uid,
            item_id=request.item_id,
            challenge_id=request.challenge_id,
            track=request.track,
            miner_hotkey=neuron.hotkey,
            content_digest=request.output_digest,
            packet_digest=result.packet_digest,
            packet_json=result.item_score_json,
            # the OBSERVED scorer (the worker stamps its own) — evidence must
            # record what actually ran, not what we asked for
            scorer_version=packet.scorer_version or "",
            score=packet.score,
            audit_ref=audit_ref,
            challenge_input_ref=challenge_input_ref,
            miner_output_ref=miner_output_ref,
            reference_original_ref=reference_original_ref,
            miner_receipt_json=receipt_json,
        )

    # -- service loop ----------------------------------------------------------

    async def run(self) -> None:
        # PIN ON FIRST CONTACT before anything is scored, and VERIFY a reloaded
        # durable pin against the live worker.
        #
        # Two different failures, two different responses:
        #  * `validator.scorer_version` names a scorer this worker is not — an
        #    operator CONFIG error, so the process refuses to start at all;
        #  * the worker disagrees with the DURABLE pin — the deployment is live
        #    and only the scoring half is wrong, so the process stays up with the
        #    conflict latched: every round skips, /health reports it and the
        #    metric goes to 0, which an operator can actually see. Crash-looping
        #    would take the health surface down with it.
        try:
            await self.discover_scorer_identity()
        except (ScorerIdentityMismatch, ScorerRuntimeMismatch):
            if self.config.scorer_version.strip():
                raise
            self.log.critical(
                "starting in a REFUSING-TO-SCORE state: the live scoring worker"
                " is not the scorer this validator is pinned to. No round will be"
                " scored until an operator resolves it",
                extra=log_fields(conflict=self.scorer_pin_conflict or ""),
            )
        # review #5: before the first round, resolve whatever a previous process
        # left checked out — otherwise the pool answers `pool_exhausted` forever.
        try:
            await self.recover_inflight_challenges()
        except Exception:
            self.log.exception("startup challenge recovery failed; continuing")
        try:
            while not self.stopping.is_set():
                try:
                    await self.run_round()
                except Exception:
                    # A broken round must never kill the loop; the next cycle retries.
                    self.log.exception("round crashed; continuing after sleep")
                sleep = self._rng.uniform(
                    self.config.cycle_sleep_min_seconds,
                    self.config.cycle_sleep_max_seconds,
                )
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.stopping.wait(), timeout=sleep)
        finally:
            # Shutdown drain: a stop signal received mid-round must not strand the
            # challenges that round fetched.
            with contextlib.suppress(Exception):
                await self._drain_inflight_challenges()
