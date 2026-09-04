"""SharedSnapshotProvider — the thin validator's convergence source (build-wave 5).

The weight-setter's `SnapshotProvider` used to be the validator's LOCAL
`miner_manager.snapshot()` — per-validator EWMA folds that never converge
(the project design record §2.1). This module adds the SHARED provider that makes
convergence emergent: it fetches the Scoring Authority's per-epoch POINTER, mirrors
the immutable epoch-log bytes directly from the object store, verifies the
tamper-evidence chain, and hands the weight-setter the authenticated `weight_u16` plus
the `MinerSnapshot` / `RewardWindowState` / `CompetitionResult` / `burn_uid` evidence the log
carries. Two validators reading the SAME finalized epoch submit that exact vector, so
their u16 bytes are identical and Yuma sees one agreed
vector (the project design record §1(b), §3.1, §4/§5; the project design record rules 9-11).

The three verification legs (the core guarantee, §5):

    sha256(mirrored bytes) == pointer.snapshot_digest == on-chain anchored digest

Any inequality means someone tampered → a typed `SnapshotDigestMismatch` is raised
and NOTHING is submitted. An unreachable authority/object store/anchor raises
`SnapshotUnavailable`, which the weight-setter also treats as HOLD — it NEVER falls
back to local sampling, because a locally-improvised vector is exactly what diverges
(§6, the project design record Part 7). Once those authentication legs pass, a narrow parser
validates the canonical current-schema identity, exact u16 sum-grid, pointer binding,
and uid/hotkey census needed for a safe chain write. Full economic/evidence validation
belongs to isolated auditors: a disagreement is alerted and reported centrally but
cannot interrupt submission of that authenticated authority vector (Decision 24).

This wave is ADDITIVE. The production provider becomes the shared one (config
`weightsetter.provider = "shared"`); `provider = "local"` keeps the existing
`miner_manager` path for report-mode / dryrun / third-party-recompute overlays
(the project design record rule 8 — same service code, only the provider swaps). The
weight-setter's tri-state confirmation, publication and reconcile are UNCHANGED;
only the snapshot SOURCE changes.

Seams (all injectable, so the whole provider is tested with a fake authority client
+ `LocalFsStore` + `InMemoryChain`, no boto3 / bittensor):

- `ScoringAuthorityClient` — fetch the epoch pointer (Protocol + `Http` impl +
  a test fake). Carries the bearer token.
- `EpochLogStore` — the object store the epoch-log bytes are mirrored from (the
  `_FINALIZED`-gated set convention from `vidaio.audit.store`).
- `EpochAnchorReader` — read the on-chain anchored digest for the epoch (the third,
  genuinely-independent verification leg). `InMemoryChainAnchorReader` reads it back
  out of an `InMemoryChain`'s recorded commitments; the real bittensor reader is
  wave 8.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol, Sequence, runtime_checkable

import httpx

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.commitments import merkle_root
from vidaio.audit.store import set_member_key
from vidaio.authority import (
    ANCHOR_DOMAIN,
    EPOCH_LOG_MEMBER,
    EpochPointer,
    epoch_prefix,
)
from vidaio.chain.adapter import EpochBoundary
from vidaio.core import get_logger, log_fields
from vidaio.epoch import (
    EPOCH_LOG_SCHEMA_VERSION,
    AuditFileKind,
    EpochLog,
    EpochLogInvalid,
    MinerCensusEntry,
    weight_vector_digest,
)
from vidaio.tokenomics import CompetitionResult, MinerSnapshot, RewardWindowState

logger = get_logger("vidaio.weightsetter.shared_snapshot")

MAX_EPOCH_LOG_BYTES = 64 * 1024 * 1024


# --------------------------------------------------------------------------------------
# Typed failures — the weight-setter maps each to a SAFE behaviour (§6).
# --------------------------------------------------------------------------------------


class SharedSnapshotError(Exception):
    """Base: the shared snapshot could not be provided this attempt.

    Never a signal to improvise — every subclass makes the weight-setter HOLD.
    """


class SnapshotUnavailable(SharedSnapshotError):
    """The authoritative input is UNREACHABLE or not yet published → HOLD (skip).

    The authority API is down / 404s, the object set is not `_FINALIZED`, the
    bytes could not be mirrored, or the on-chain anchor could not be read. The
    last confirmed vector stays live on chain; the validator re-converges as soon
    as it can mirror. NEVER a fall-back to local sampling (that is what diverges).
    """


class PointerNotFound(SnapshotUnavailable):
    """A POSITIVE not-found: the authority answered that this epoch has no pointer (404).

    A subclass of `SnapshotUnavailable` so every caller HOLDS on it (the weight-setter and
    the auditor's `pointer_for` path treat "no pointer" as HOLD). an internal review: a 404 is
    an UNAUTHENTICATED, AUTHORITY-controlled response, so it must NEVER advance the auditor's
    audit floor — the fresh-cursor 404-binary-search discovery that treated a 404 as proof of
    pruning (`discover_earliest_served_epoch`, round-7/8 #4) was REMOVED as a silent coverage
    bypass. A 404 for an epoch at/above the trustworthy floor is now the authority WITHHOLDING
    ⇒ HOLD/retry (never a floor-advance/skip). The typed distinction from a bare TRANSIENT
    `SnapshotUnavailable` (transport error / non-200 / malformed body) is retained for callers
    that want to tell a definite not-found apart from a temporary read failure; both HOLD.
    """


class SnapshotDigestMismatch(SharedSnapshotError):
    """Authority-vector authentication/safety BROKE → REFUSE to submit (CRITICAL).

    `sha256(bytes)` != the pointer digest, the pointer's anchor digest disagrees,
    the on-chain anchored digest disagrees, or the mirrored bytes are not a valid
    epoch log, or its narrow u16/census submission view is not chain-safe. Economic
    and evidence disagreements in otherwise authenticated bytes are not this error;
    auditors report those under Decision 24 while the exact safe vector still flows.
    """


# --------------------------------------------------------------------------------------
# Seams.
# --------------------------------------------------------------------------------------


@runtime_checkable
class ScoringAuthorityClient(Protocol):
    """Fetch the Scoring Authority's per-epoch POINTER (never the bytes).

    The pointer carries the object-store `snapshot_key`, the `snapshot_digest`,
    the `weight_vector_digest`, and the on-chain `anchor` (§3.1). The validator
    mirrors the bytes itself from the object store and verifies the digest chain —
    the API is a cheap, cacheable index, not the source of trust.

    Implementations RAISE `SnapshotUnavailable` when no finalized pointer exists
    (404) or the API cannot be reached; they never substitute a pointer.
    """

    def latest_pointer(self) -> EpochPointer:
        """The pointer to the newest FINALIZED epoch (the convergence target)."""
        ...

    def pointer_for(self, epoch_id: int) -> EpochPointer:
        """The pointer for a specific epoch (unknown / unfinalized -> unavailable)."""
        ...


@runtime_checkable
class EpochLogStore(Protocol):
    """The slice of `vidaio.audit.store.AuditStore` the provider mirrors from.

    Both `LocalFsStore` and the S3/Hippius `_TransportBackedStore` satisfy it via
    the shared `_SetConventionMixin` (the `_FINALIZED` half-write guard).
    """

    def is_finalized(self, prefix: str) -> bool: ...

    def get_set_member(
        self,
        prefix: str,
        name: str,
        *,
        expected_digest: str | None = None,
        byte_size: int | None = None,
        max_bytes: int | None = None,
    ) -> bytes: ...


@runtime_checkable
class EpochAnchorReader(Protocol):
    """Read the on-chain anchored `log_digest` for an epoch (the third leg).

    Returns the 64-hex digest anchored for `(netuid, epoch_id)`, or None to say
    POSITIVELY that the chain holds no such anchor. RAISES on a read failure
    (never substitutes None — a None while the pointer claims a txid is treated as
    tampering, whereas an unreadable chain must only HOLD).
    """

    def read_epoch_anchor(self, *, netuid: int, epoch_id: int) -> str | None: ...

    def read_epoch_anchor_at(
        self, *, netuid: int, epoch_id: int, block_number: int
    ) -> str | None: ...


@runtime_checkable
class EpochBoundaryReader(Protocol):
    """Read archive-proven finalized subnet epoch boundaries.

    This is deliberately narrower than the full chain adapter.  The convergence
    path needs only enough independent chain state to prove that an authority
    response advertised as ``latest`` really is the latest finalized epoch, and
    that its close block is the exact runtime transition block.  Historical
    ``pointer_for(epoch_id)`` reads do not use this seam.
    """

    def latest_closed_epoch(self, *, netuid: int) -> EpochBoundary | None: ...

    def epoch_close_block(self, *, netuid: int, epoch_id: int) -> int | None: ...


# --------------------------------------------------------------------------------------
# Concrete seam implementations.
# --------------------------------------------------------------------------------------


class HttpScoringAuthorityClient:
    """`ScoringAuthorityClient` over HTTP (httpx), carrying the bearer token.

    Points at the Scoring Authority pointer API (`GET /epoch/latest`,
    `GET /epoch/{id}`). A 404 (no finalized epoch / unknown epoch) and any transport
    error both surface as `SnapshotUnavailable` — the validator HOLDS. A malformed
    body (fails `EpochPointer` validation) is treated as unavailable too: a pointer
    we cannot parse is not a pointer.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout: float = 10.0,
        client: httpx.Client | None = None,
        signer: object | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("HttpScoringAuthorityClient needs a non-empty base_url")
        self._base = base_url.rstrip("/")
        self._token = token
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)
        # P2 Scheme A: when a hotkey signer is wired (`hotkey` + `sign(bytes)->hex`,
        # the `CallableHotkeySigner` shape), every pointer read carries the four
        # signed headers, so the authority's hotkey-auth `enforce` mode keeps
        # serving this client. Signing covers the URL path the server verifies
        # (`request.url.path`), including any base-url prefix.
        self._signer = signer

    def _headers(self, path: str) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        if self._signer is not None:
            from urllib.parse import urlsplit

            from vidaio.services.hotkey_auth import sign_request_headers

            headers.update(
                sign_request_headers(
                    self._signer,
                    method="GET",
                    path=urlsplit(self._base + path).path,
                )
            )
        return headers

    def _get(self, path: str) -> EpochPointer:
        try:
            resp = self._client.get(self._base + path, headers=self._headers(path))
        except httpx.HTTPError as exc:
            raise SnapshotUnavailable(
                f"scoring authority {path} unreachable: {type(exc).__name__}: {exc}"
            ) from exc
        if resp.status_code == 404:
            # an internal review: a 404 is a DEFINITE not-found (pruned / never finalized)
            # — a positive lower bound for history discovery. Distinct from the transient
            # failures below, which are NOT proof the epoch is gone. `PointerNotFound`
            # subclasses `SnapshotUnavailable`, so HOLD-on-unavailable callers are unchanged.
            raise PointerNotFound(
                f"scoring authority has no finalized pointer for {path} (404)"
            )
        if resp.status_code != 200:
            raise SnapshotUnavailable(
                f"scoring authority {path} returned HTTP {resp.status_code}"
            )
        try:
            return EpochPointer.model_validate(resp.json())
        except Exception as exc:  # malformed body: a pointer we cannot parse
            raise SnapshotUnavailable(
                f"scoring authority {path} returned an unparseable pointer: {exc}"
            ) from exc

    def latest_pointer(self) -> EpochPointer:
        return self._get("/epoch/latest")

    def pointer_for(self, epoch_id: int) -> EpochPointer:
        return self._get(f"/epoch/{epoch_id}")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class ChainAdapterAnchorReader:
    """THE real `EpochAnchorReader`: reads the anchored digest back off any chain.

    Delegates to the chain adapter's `read_anchor(netuid, epoch_id, domain)` — the
    genuinely-independent third verification leg (the project design record §5).
    ONE reader for every mode: it wraps the `HttpChainAdapter` (sim `/state`) in
    report/local-stack, and the `BittensorChainAdapter` (Commitments pallet) in
    production — differing only in the adapter behind it, never a None-skip. An
    adapter that cannot answer (missing `read_anchor`) is a wiring error and raises.

    Fail semantics propagate straight through: a positively-absent anchor returns
    None (a substituted pointer is caught upstream), while a read/transport failure
    raises `SnapshotUnavailable` (HOLD; never mistaken for "no anchor").
    """

    def __init__(self, chain: object, *, domain: str = ANCHOR_DOMAIN) -> None:
        reader = getattr(chain, "read_anchor", None)
        if not callable(reader):
            raise ValueError(
                f"chain adapter {type(chain).__name__} does not implement read_anchor()"
                " — cannot verify the on-chain anchor (the third tamper-evidence leg)"
            )
        self._chain = chain
        self._domain = domain

    def read_epoch_anchor(self, *, netuid: int, epoch_id: int) -> str | None:
        try:
            return self._chain.read_anchor(
                netuid=netuid, epoch_id=epoch_id, domain=self._domain
            )
        except SnapshotUnavailable:
            raise
        except Exception as exc:  # an unreadable chain HOLDS, never "no anchor"
            raise SnapshotUnavailable(
                f"could not read the on-chain anchor for epoch {epoch_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def read_epoch_anchor_at(
        self, *, netuid: int, epoch_id: int, block_number: int
    ) -> str | None:
        """Verify an overwritten single-slot commitment from archive state."""
        reader = getattr(self._chain, "read_anchor_at", None)
        if not callable(reader):
            # Report chains retain a per-epoch journal, so their normal read is
            # already historical. Production guards require read_anchor_at.
            return self.read_epoch_anchor(netuid=netuid, epoch_id=epoch_id)
        try:
            return reader(
                netuid=netuid,
                epoch_id=epoch_id,
                domain=self._domain,
                block_number=block_number,
            )
        except SnapshotUnavailable:
            raise
        except Exception as exc:
            raise SnapshotUnavailable(
                f"could not read the on-chain anchor for epoch {epoch_id} at block "
                f"{block_number}: {type(exc).__name__}: {exc}"
            ) from exc

    @property
    def supports_epoch_boundaries(self) -> bool:
        """Whether the wrapped adapter exposes both archive boundary reads."""
        return callable(getattr(self._chain, "latest_closed_epoch", None)) and callable(
            getattr(self._chain, "epoch_close_block", None)
        )

    def latest_closed_epoch(self, *, netuid: int) -> EpochBoundary | None:
        """Delegate the finalized latest-boundary read to the wrapped adapter."""
        reader = getattr(self._chain, "latest_closed_epoch", None)
        if not callable(reader):
            raise SnapshotUnavailable(
                f"chain adapter {type(self._chain).__name__} cannot read the latest "
                "archive-proven epoch boundary"
            )
        try:
            return reader(netuid=netuid)
        except SnapshotUnavailable:
            raise
        except Exception as exc:
            raise SnapshotUnavailable(
                f"could not read subnet {netuid}'s latest finalized epoch boundary: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def epoch_close_block(self, *, netuid: int, epoch_id: int) -> int | None:
        """Delegate the independent exact-close read to the wrapped adapter."""
        reader = getattr(self._chain, "epoch_close_block", None)
        if not callable(reader):
            raise SnapshotUnavailable(
                f"chain adapter {type(self._chain).__name__} cannot read exact epoch "
                "close blocks"
            )
        try:
            return reader(netuid=netuid, epoch_id=epoch_id)
        except SnapshotUnavailable:
            raise
        except Exception as exc:
            raise SnapshotUnavailable(
                f"could not read subnet {netuid} epoch {epoch_id}'s exact close block: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


class InMemoryChainAnchorReader(ChainAdapterAnchorReader):
    """Back-compat alias: `ChainAdapterAnchorReader` over an `InMemoryChain`.

    The InMemoryChain now implements `read_anchor` (a real read of its recorded
    `anchored` payloads), so this is exactly `ChainAdapterAnchorReader(chain)` — kept
    as a named type for the test/e2e call sites that reference it.
    """


# --------------------------------------------------------------------------------------
# The inputs the shared provider hands the weight-setter.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpochInputs:
    """The convergence INPUTS the epoch log carries, for the weight-setter.

    The weight-setter submits the authenticated authority-published `weight_u16`
    vector unchanged. Standalone auditor workers independently consume the epoch log to
    re-derive its economics and report findings for manual remediation.

    `composed_at` is the epoch log's `created_at`. `burn_uid`, when present, is the
    canonical subnet-owner sink for every fixed share the epoch could not allocate
    (including IDLE's fixed sink pool, absent track pools, and missing podium ranks).

    Schema-v14 convergence includes the CPU-rederived packet-economic competition
    result and predecessor-derived global reward window in addition to inference miner
    snapshots and the conditional sink path. The authority's cycle clock is not a
    separate canonical input: result application and window activity are evaluated at
    the epoch's chain-bound ``composed_at``.
    """

    epoch_id: int
    close_block: int
    burn_uid: int | None
    weight_shares: dict[int, float]
    weight_u16: dict[int, int]
    composed_at: datetime
    #: Complete close-block uid -> hotkey identity binding from the authenticated
    #: ``EpochLog.miner_census``.  This is deliberately broader than ``miners``:
    #: competition-window recipients can carry positive weight without having an
    #: inference snapshot in the epoch.  Weight submission must bind every such
    #: uid so a recycled slot cannot pay a different hotkey.
    miner_census_hotkeys: dict[int, str] = field(default_factory=dict)
    competition_result: CompetitionResult | None = None
    reward_window_state: RewardWindowState = field(default_factory=RewardWindowState)


@dataclass(frozen=True, slots=True)
class _ResolvedEpoch:
    pointer: EpochPointer
    #: Strict economic/evidence model. None only when authenticated canonical bytes
    #: fail an audit derivation invariant but still expose a safe submission view.
    log: EpochLog | None
    submission: "AuthoritySubmissionView"


@dataclass(frozen=True, slots=True)
class AuthoritySubmissionView:
    """Minimal authenticated fields needed to submit one authority vector safely.

    This view deliberately does *not* re-derive scores, earning state, competition,
    reward-window state, evidence completeness, or ``weight_shares``. Those are auditor findings
    under Decision 24. It does retain every condition required to make a chain write
    well-defined and identity-safe: exact current schema/canonical bytes, pointer
    identity, a digest-bound non-empty u16 sum-grid, and a unique census hotkey for
    every positive non-owner target. The owner/sink target is independently rebound
    to live chain state by ``WeightSetter`` immediately before submission.
    """

    epoch_id: int
    close_block: int
    scorer_version: str
    created_at: datetime
    burn_uid: int | None
    weight_u16: dict[int, int]
    weight_vector_digest: str
    miner_census_hotkeys: dict[int, str]
    packet_digests: tuple[str, ...] | None
    score_packet_merkle_root: str | None
    strict_error: str | None = None


_EPOCH_LOG_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "epoch_id",
        "close_block",
        "scorer_version",
        "created_at",
        "prior_log_digest",
        "gap_epochs",
        "burn_uid",
        "competition_result",
        "reward_window_state",
        "miner_census",
        "miners",
        "weight_shares",
        "weight_u16",
        "weight_vector_digest",
        "audit_manifest",
    }
)


def _lower_hex_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _bounded_error(exc: Exception) -> str:
    detail = f"{type(exc).__name__}: {exc}"
    return detail if len(detail) <= 4096 else detail[:4093] + "..."


def _raw_packet_commitment(
    manifest: object,
) -> tuple[tuple[str, ...] | None, str | None]:
    """Best-effort packet commitment extraction; malformed evidence stays unresolved.

    Evidence consistency is report-only for the emissions submission. Returning
    ``None`` (rather than ``()``) ensures a malformed manifest can never be published
    as the honest empty-packet sentinel after the vector lands.
    """
    if not isinstance(manifest, dict):
        return None, None
    expected_keys = {
        "per_uid",
        "baseline_bundles",
        "score_packet_merkle_root",
        "earning_inputs",
        "availability_inputs",
        "competition_input",
        "competition_bundles",
        "fold_cursors",
    }
    if set(manifest) != expected_keys:
        return None, None
    root = manifest.get("score_packet_merkle_root")
    if root is not None and not _lower_hex_digest(root):
        return None, None
    per_uid = manifest.get("per_uid")
    baseline = manifest.get("baseline_bundles")
    competition = manifest.get("competition_bundles")
    if (
        not isinstance(per_uid, dict)
        or not isinstance(baseline, list)
        or not isinstance(competition, dict)
    ):
        return None, None
    groups: list[object] = [baseline, *per_uid.values(), *competition.values()]
    digests: list[str] = []
    for refs in groups:
        if not isinstance(refs, list):
            return None, None
        for ref in refs:
            if not isinstance(ref, dict) or not isinstance(ref.get("kind"), str):
                return None, None
            if ref["kind"] != AuditFileKind.SCORE_PACKET.value:
                continue
            digest = ref.get("digest")
            if not _lower_hex_digest(digest):
                return None, None
            digests.append(str(digest))
    return tuple(digests), root


def parse_authority_submission(
    data: bytes, pointer: EpochPointer
) -> AuthoritySubmissionView:
    """Parse only chain-feasibility/authentication fields from canonical log bytes."""
    try:
        obj = json.loads(data)
    except Exception as exc:
        raise SnapshotDigestMismatch(
            f"epoch {pointer.epoch_id} log bytes are not JSON: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(obj, dict) or set(obj) != _EPOCH_LOG_TOP_LEVEL_FIELDS:
        fields = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        raise SnapshotDigestMismatch(
            f"epoch {pointer.epoch_id} log does not have the exact schema-v"
            f"{EPOCH_LOG_SCHEMA_VERSION} top-level shape: {fields!r}"
        )
    try:
        canonical = canonical_json_bytes(obj)
    except Exception as exc:
        raise SnapshotDigestMismatch(
            f"epoch {pointer.epoch_id} log cannot be canonically encoded: {exc}"
        ) from exc
    if canonical != data:
        raise SnapshotDigestMismatch(
            f"epoch {pointer.epoch_id} anchored log bytes are NOT canonical"
        )

    schema = obj["schema_version"]
    epoch_id = obj["epoch_id"]
    close_block = obj["close_block"]
    if type(schema) is not int or schema != EPOCH_LOG_SCHEMA_VERSION:
        raise SnapshotDigestMismatch(
            f"epoch {pointer.epoch_id} schema_version {schema!r} is not "
            f"{EPOCH_LOG_SCHEMA_VERSION}"
        )
    if type(epoch_id) is not int or epoch_id < 0 or epoch_id != pointer.epoch_id:
        raise SnapshotDigestMismatch(
            f"pointer epoch_id {pointer.epoch_id} does not match usable log epoch "
            f"{epoch_id!r}"
        )
    if (
        type(close_block) is not int
        or close_block < 0
        or close_block != pointer.close_block
    ):
        raise SnapshotDigestMismatch(
            f"pointer close_block {pointer.close_block} does not match usable log close "
            f"{close_block!r}"
        )
    scorer_version = obj["scorer_version"]
    if not isinstance(scorer_version, str) or not scorer_version.strip():
        raise SnapshotDigestMismatch(
            f"epoch {epoch_id} scorer_version is not a non-empty string"
        )
    raw_created_at = obj["created_at"]
    if not isinstance(raw_created_at, str):
        raise SnapshotDigestMismatch(f"epoch {epoch_id} created_at is not a string")
    try:
        created_at = datetime.fromisoformat(raw_created_at)
    except Exception as exc:
        raise SnapshotDigestMismatch(
            f"epoch {epoch_id} created_at is not parseable: {exc}"
        ) from exc
    if created_at.tzinfo is None or created_at.tzinfo.utcoffset(created_at) is None:
        raise SnapshotDigestMismatch(f"epoch {epoch_id} created_at is timezone-naive")
    if created_at.isoformat() != raw_created_at:
        raise SnapshotDigestMismatch(
            f"epoch {epoch_id} created_at is not in the canonical schema encoding"
        )

    raw_vector = obj["weight_u16"]
    if not isinstance(raw_vector, list) or not raw_vector:
        raise SnapshotDigestMismatch(
            f"epoch {epoch_id} authority weight_u16 is not a non-empty pair list"
        )
    weight_u16: dict[int, int] = {}
    previous_uid = -1
    for position, pair in enumerate(raw_vector):
        if not isinstance(pair, list) or len(pair) != 2:
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} weight_u16[{position}] is not [uid, value]"
            )
        uid, value = pair
        if type(uid) is not int or not 0 <= uid <= 65535 or type(value) is not int:
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} weight_u16[{position}] has invalid integer fields"
            )
        if uid <= previous_uid:
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} weight_u16 uids are not strictly ascending"
            )
        if not 1 <= value <= 65535:
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} weight_u16 uid {uid} value {value} is outside 1..65535"
            )
        if uid in weight_u16:
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} weight_u16 repeats uid {uid}"
            )
        weight_u16[uid] = value
        previous_uid = uid
    if sum(weight_u16.values()) != 65535:
        raise SnapshotDigestMismatch(
            f"epoch {epoch_id} weight_u16 sum {sum(weight_u16.values())} is not 65535"
        )
    published_digest = obj["weight_vector_digest"]
    expected_digest = weight_vector_digest(weight_u16)
    if not _lower_hex_digest(published_digest) or published_digest != expected_digest:
        raise SnapshotDigestMismatch(
            f"epoch {epoch_id} weight_vector_digest does not bind its exact u16 vector"
        )
    if pointer.weight_vector_digest != published_digest:
        raise SnapshotDigestMismatch(
            f"epoch {epoch_id} pointer weight_vector_digest does not match the log vector"
        )

    burn_uid = obj["burn_uid"]
    if burn_uid is not None and (
        type(burn_uid) is not int or not 0 <= burn_uid <= 65535
    ):
        raise SnapshotDigestMismatch(f"epoch {epoch_id} burn_uid is invalid")
    raw_census = obj["miner_census"]
    if not isinstance(raw_census, list):
        raise SnapshotDigestMismatch(f"epoch {epoch_id} miner_census is not a list")
    census: dict[int, str] = {}
    census_hotkeys: set[str] = set()
    previous_census_uid = -1
    for position, raw_entry in enumerate(raw_census):
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "uid",
            "hotkey",
            "coldkey",
            "ip",
        }:
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} miner_census[{position}] has an invalid shape"
            )
        if (
            type(raw_entry["uid"]) is not int
            or not 0 <= raw_entry["uid"] <= 65535
            or not all(
                isinstance(raw_entry[field], str)
                for field in ("hotkey", "coldkey", "ip")
            )
        ):
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} miner_census[{position}] has invalid field types"
            )
        try:
            entry = MinerCensusEntry.model_validate(raw_entry)
        except Exception as exc:
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} miner_census[{position}] is malformed: {exc}"
            ) from exc
        if entry.uid in census:
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} miner_census repeats uid {entry.uid}"
            )
        if entry.uid <= previous_census_uid:
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} miner_census uids are not strictly ascending"
            )
        if not entry.hotkey.strip():
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} miner_census uid {entry.uid} has an empty hotkey"
            )
        if entry.hotkey in census_hotkeys:
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} miner_census repeats hotkey {entry.hotkey!r}"
            )
        census[entry.uid] = entry.hotkey
        census_hotkeys.add(entry.hotkey)
        previous_census_uid = entry.uid
    missing = sorted(
        set(weight_u16) - set(census) - ({burn_uid} if burn_uid is not None else set())
    )
    if missing:
        raise SnapshotDigestMismatch(
            f"epoch {epoch_id} positive authority targets lack census hotkey bindings: "
            f"{missing}"
        )

    packet_digests, packet_root = _raw_packet_commitment(obj["audit_manifest"])
    return AuthoritySubmissionView(
        epoch_id=epoch_id,
        close_block=close_block,
        scorer_version=scorer_version,
        created_at=created_at,
        burn_uid=burn_uid,
        weight_u16=weight_u16,
        weight_vector_digest=str(published_digest),
        miner_census_hotkeys=census,
        packet_digests=packet_digests,
        score_packet_merkle_root=packet_root,
    )


# --------------------------------------------------------------------------------------
# The provider.
# --------------------------------------------------------------------------------------


class SharedSnapshotProvider:
    """The convergence `SnapshotProvider`: pointer -> mirror -> verify -> inputs.

    Implements the weight-setter's `SnapshotProvider` protocol (`miner_snapshots()`)
    plus `epoch_inputs()` — the feature-detected extension that hands over the
    narrow authenticated `burn_uid`, uid/hotkey census and stated u16 vector so the
    weight-setter submits the IDENTICAL authority input. Strictly decoded reward-window,
    competition and miner fields remain available for diagnostics when valid.

    Every `miner_snapshots()` call RE-RESOLVES the latest finalized epoch (fetch
    pointer, mirror bytes, verify the three-way digest chain, parse the safe view).
    It raises `SnapshotUnavailable` (HOLD) or `SnapshotDigestMismatch` (REFUSE) only
    when the authority input cannot be authenticated or safely expressed. Strict
    economic/evidence parse failures are logged for manual remediation and do not
    suppress that safe vector. The resolution is cached for the matching
    `epoch_inputs()` call in the same attempt.
    """

    def __init__(
        self,
        *,
        client: ScoringAuthorityClient,
        store: EpochLogStore,
        netuid: int = 85,
        anchor_reader: EpochAnchorReader | None = None,
        boundary_reader: EpochBoundaryReader | None = None,
        verify_anchor: bool = True,
    ) -> None:
        self._client = client
        self._store = store
        self._netuid = netuid
        self._anchor_reader = anchor_reader
        if boundary_reader is not None:
            self._boundary_reader: EpochBoundaryReader | None = boundary_reader
        else:
            # Production already wraps the real adapter in
            # ChainAdapterAnchorReader.  Reuse that wrapper without forcing report
            # adapters to implement archive history.  A direct test/custom reader
            # may structurally implement the two narrow methods without exposing
            # the capability property.
            supports = getattr(anchor_reader, "supports_epoch_boundaries", None)
            structurally_supported = (
                callable(getattr(anchor_reader, "latest_closed_epoch", None))
                and callable(getattr(anchor_reader, "epoch_close_block", None))
            )
            self._boundary_reader = (
                anchor_reader  # type: ignore[assignment]
                if structurally_supported and supports is not False
                else None
            )
        self._verify_anchor = verify_anchor
        self._resolved: _ResolvedEpoch | None = None
        self._resolved_latest_boundary: tuple[int, int] | None = None

    # -- SnapshotProvider protocol ---------------------------------------------

    def miner_snapshots(self) -> Sequence[MinerSnapshot]:
        """The strict log's snapshots, or none after a report-only audit mismatch.

        Shared-mode submission consumes :meth:`epoch_inputs`' authenticated u16
        vector directly.  The snapshots are therefore diagnostics/economic inputs,
        not an emissions gate.  Returning an empty sequence when strict parsing
        rejects lets ``WeightSetter`` continue with the narrow submission view;
        standalone auditors still parse the complete log strictly and report the
        exact mismatch.
        """
        resolved = self._resolve()
        return resolved.log.miners if resolved.log is not None else ()

    def resolved_log(self) -> EpochLog | None:
        """Return the epoch log resolved and verified in this attempt.

        None until `miner_snapshots()`/`epoch_inputs()` has resolved an epoch this
        attempt. Retained as a legacy/diagnostic access surface; production standalone
        auditors resolve their own bytes and ``WeightSetter`` does not use this for audit.
        """
        return self._resolved.log if self._resolved is not None else None

    def resolved_snapshot_digest(self) -> str:
        """The exact verified epoch-log digest backing this attempt's vector.

        Publication freezes this value into the durable weight intent before the
        chain write. It comes from the pointer whose bytes have already passed the
        three-leg digest verification; resolving here only covers callers that use
        the publication surface before ``miner_snapshots()``.
        """
        resolved = self._resolved or self._resolve()
        return resolved.pointer.snapshot_digest

    def resolved_latest_boundary(self) -> tuple[int, int] | None:
        """Re-prove and return the archive boundary for this latest resolution.

        ``None`` means the provider was used with a report/custom chain that has
        no archive-boundary seam.  Production ``chain.mode=bittensor`` treats that
        absence as a HOLD in :class:`WeightSetter`; it is never silently accepted.

        This deliberately performs a second archive read after the epoch bytes
        have been mirrored and parsed.  If a new epoch closed during that work,
        ``_verify_latest_pointer_boundary`` now rejects the formerly-latest pointer
        before the service records an intent or writes weights.
        """
        if self._resolved is None:
            return None
        boundary = self._verify_latest_pointer_boundary(self._resolved.pointer)
        self._resolved_latest_boundary = boundary
        return boundary

    def score_packet_digests(self) -> Sequence[str]:
        """The exact packet leaf set committed by the resolved epoch log.

        Thin validators do not run inference and therefore have no authoritative
        packet rows in their process-local validator database. The shared log is
        their source of truth: extract every SCORE_PACKET ref (earning and baseline),
        then independently recompute the manifest root before returning the set.
        A missing/extra ref or substituted root is a typed mismatch and prevents
        publication of a false commitment. This verification runs only after the
        authority vector's chain submission, so it can never gate emissions.
        """
        resolved = self._resolved or self._resolve()
        return self._validated_packet_digests(resolved)

    def score_packet_digests_for_epoch(
        self, epoch_id: int, *, expected_snapshot_digest: str
    ) -> Sequence[str]:
        """Re-resolve and validate the packet leaves for one durable intent.

        Publication can be retried after a crash or after ``_resolved`` has moved to
        a newer authority epoch.  The intent therefore supplies both the exact epoch
        id and the snapshot digest it authenticated before submission.  Fetching
        ``latest_pointer()`` here would risk attaching a newer packet set to an older
        accepted vector; a digest mismatch is likewise a hard refusal.  Failures only
        leave the already-accepted publication queued for another retry.
        """
        pointer = self._pointer_for(int(epoch_id))
        if pointer.snapshot_digest != expected_snapshot_digest:
            raise SnapshotDigestMismatch(
                f"epoch {epoch_id} pointer snapshot_digest {pointer.snapshot_digest} "
                f"does not match durable intent snapshot {expected_snapshot_digest}"
            )
        resolved = self._resolve_from_pointer(pointer, allow_economic_disagreement=True)
        return self._validated_packet_digests(resolved)

    @staticmethod
    def _packet_digests_from_resolved(resolved: _ResolvedEpoch) -> list[str]:
        digests = resolved.submission.packet_digests
        if digests is None:
            raise SnapshotDigestMismatch(
                f"epoch {resolved.submission.epoch_id} packet evidence is malformed; "
                "the vector is still submission-safe, but publication must remain "
                "unresolved until an operator repairs/releases the exact evidence"
            )
        return list(digests)

    @classmethod
    def _validated_packet_digests(cls, resolved: _ResolvedEpoch) -> list[str]:
        digests = cls._packet_digests_from_resolved(resolved)
        expected = resolved.submission.score_packet_merkle_root
        if not digests:
            if expected is not None:
                raise SnapshotDigestMismatch(
                    f"epoch {resolved.submission.epoch_id} manifest commits packet root "
                    f"{expected} but exposes no SCORE_PACKET refs"
                )
            return []
        if expected is None:
            raise SnapshotDigestMismatch(
                f"epoch {resolved.submission.epoch_id} exposes {len(digests)} SCORE_PACKET "
                "refs but has no score_packet_merkle_root"
            )
        actual = merkle_root(digests)
        if actual != expected:
            raise SnapshotDigestMismatch(
                f"epoch {resolved.submission.epoch_id} SCORE_PACKET refs yield merkle root "
                f"{actual}, not the manifest's committed root {expected}"
            )
        return digests

    def committed_packet_digests(self) -> Sequence[str]:
        """Copy packet refs from the already-authenticated authority log.

        This is the weight-intent durability seam, not an auditor verdict: it never
        fetches, recomputes, or compares the packet Merkle root. ``WeightSetter``
        calls it only after the shared provider resolved the signed/anchored log and
        catches every failure so publication bookkeeping can never block the
        authority vector's scheduled chain submission. The ordinary
        :meth:`score_packet_digests` method retains independent Merkle validation for
        post-submit publication recovery and diagnostics.
        """
        if self._resolved is None:
            raise SnapshotUnavailable(
                "no authenticated authority epoch is resolved for this weight attempt"
            )
        return self._packet_digests_from_resolved(self._resolved)

    # -- the epoch-inputs extension (feature-detected by the weight-setter) -----

    def epoch_inputs(self) -> EpochInputs:
        """Return the complete schema-v15 vector convergence inputs.

        Reuses the log resolved by `miner_snapshots()` in the same attempt (the
        weight-setter calls them back to back); resolves on its own if called first.
        """
        resolved = self._resolved or self._resolve()
        log = resolved.log
        view = resolved.submission
        # The exact authority u16 and census binding come from the narrow authenticated
        # view even if strict economic/evidence validation rejected the rest.  Those
        # rejected fields remain useful to an independent auditor, but never gate the
        # scheduled authority-vector write (Decision 24).
        return EpochInputs(
            epoch_id=view.epoch_id,
            close_block=view.close_block,
            burn_uid=view.burn_uid,
            weight_shares=dict(log.weight_shares) if log is not None else {},
            weight_u16=dict(view.weight_u16),
            composed_at=view.created_at,
            miner_census_hotkeys=dict(view.miner_census_hotkeys),
            competition_result=(log.competition_result if log is not None else None),
            reward_window_state=(
                log.reward_window_state if log is not None else RewardWindowState()
            ),
        )

    def resolve_epoch(self, epoch_id: int) -> EpochLog:
        """Fetch and verify a specific finalized epoch's log.

        Uses `pointer_for(epoch_id)` (not `latest_pointer()`) and runs the SAME three-leg
        verification as `_resolve`. This legacy/diagnostic API supported the former
        in-process own-audit backfill; standalone auditors now resolve history through
        their own provider/cursor. It does NOT overwrite the cached
        `_resolved` latest epoch (that stays the convergence target for `epoch_inputs()` /
        `resolved_log()` this attempt). Raises `SnapshotUnavailable` (HOLD — includes a 404
        `PointerNotFound`) or `SnapshotDigestMismatch` (REFUSE) exactly like `_resolve`, so
        legacy callers fail closed on an unavailable/tampered predecessor.
        """
        pointer = self._pointer_for(epoch_id)
        resolved = self._resolve_from_pointer(pointer)
        # ``allow_economic_disagreement`` defaults false above, so a missing strict
        # log is impossible. Keep the assertion local instead of widening the public
        # return type used by standalone audit/history callers.
        assert resolved.log is not None
        return resolved.log

    # -- resolve = fetch pointer -> mirror bytes -> verify -> parse -------------

    def _resolve(self) -> _ResolvedEpoch:
        # A failed new attempt must not leave a prior attempt's freshness proof
        # observable through the feature-detected service seam.
        self._resolved = None
        self._resolved_latest_boundary = None
        pointer = self._fetch_pointer()
        boundary = self._verify_latest_pointer_boundary(pointer)
        resolved = self._resolve_from_pointer(pointer, allow_economic_disagreement=True)
        self._resolved = resolved
        self._resolved_latest_boundary = boundary
        return resolved

    def _verify_latest_pointer_boundary(
        self, pointer: EpochPointer
    ) -> tuple[int, int] | None:
        """Bind ``GET /epoch/latest`` to the archive chain's finalized latest epoch.

        A historical pointer remains cryptographically valid forever, so digest
        and historical-anchor checks alone cannot distinguish it from the current
        convergence target.  When the archive seam is present, require two exact
        chain facts: the paired latest ``(epoch_id, close_block)`` and a second
        ``epoch_close_block`` lookup for that id.  Any unreadable, missing or
        disagreeing view HOLDS; explicit historical-by-id APIs bypass this method.
        """
        reader = self._boundary_reader
        if reader is None:
            return None
        try:
            latest = reader.latest_closed_epoch(netuid=self._netuid)
        except SharedSnapshotError:
            raise
        except Exception as exc:
            raise SnapshotUnavailable(
                f"could not verify the authority latest pointer against the archive "
                f"chain: {type(exc).__name__}: {exc}"
            ) from exc
        if latest is None:
            raise SnapshotUnavailable(
                f"archive chain has no finalized closed epoch for subnet {self._netuid}; "
                "cannot accept an authority latest pointer"
            )
        epoch_id = getattr(latest, "epoch_id", None)
        close_block = getattr(latest, "close_block", None)
        if (
            isinstance(epoch_id, bool)
            or not isinstance(epoch_id, int)
            or epoch_id < 1
            or isinstance(close_block, bool)
            or not isinstance(close_block, int)
            or close_block < 1
        ):
            raise SnapshotUnavailable(
                f"archive chain returned an invalid latest epoch boundary: {latest!r}"
            )
        try:
            exact_close = reader.epoch_close_block(
                netuid=self._netuid, epoch_id=epoch_id
            )
        except SharedSnapshotError:
            raise
        except Exception as exc:
            raise SnapshotUnavailable(
                f"could not independently verify subnet {self._netuid} epoch "
                f"{epoch_id}'s close block: {type(exc).__name__}: {exc}"
            ) from exc
        if (
            isinstance(exact_close, bool)
            or not isinstance(exact_close, int)
            or exact_close < 1
        ):
            raise SnapshotUnavailable(
                f"archive chain did not return an exact finalized close block for "
                f"subnet {self._netuid} epoch {epoch_id}"
            )
        if exact_close != close_block:
            raise SnapshotUnavailable(
                f"archive boundary views disagree for subnet {self._netuid} epoch "
                f"{epoch_id}: latest_closed_epoch={close_block}, "
                f"epoch_close_block={exact_close}"
            )
        if pointer.epoch_id != epoch_id or pointer.close_block != close_block:
            raise SnapshotUnavailable(
                f"authority latest pointer is not the archive chain's latest finalized "
                f"epoch: pointer=({pointer.epoch_id}, {pointer.close_block}), "
                f"chain=({epoch_id}, {close_block})"
            )
        return epoch_id, close_block

    def _resolve_from_pointer(
        self,
        pointer: EpochPointer,
        *,
        allow_economic_disagreement: bool = False,
    ) -> _ResolvedEpoch:
        if not pointer.finalized:
            raise SnapshotUnavailable(
                f"epoch {pointer.epoch_id} pointer is not finalized yet"
            )
        prefix = epoch_prefix(pointer.epoch_id)
        expected_key = set_member_key(prefix, EPOCH_LOG_MEMBER)
        if pointer.snapshot_key != expected_key:
            # The pointer's key does not match the canonical object layout — refuse
            # rather than mirror an adversary-chosen key.
            raise SnapshotDigestMismatch(
                f"pointer snapshot_key {pointer.snapshot_key!r} does not match the "
                f"canonical epoch-log key {expected_key!r}"
            )

        # The `_FINALIZED` half-write guard: never mirror a half-written set.
        try:
            finalized = self._store.is_finalized(prefix)
        except Exception as exc:
            raise SnapshotUnavailable(
                f"object store could not be probed for epoch {pointer.epoch_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not finalized:
            raise SnapshotUnavailable(
                f"epoch {pointer.epoch_id} object set is not _FINALIZED yet"
            )

        try:
            data = self._store.get_set_member(
                prefix, EPOCH_LOG_MEMBER, max_bytes=MAX_EPOCH_LOG_BYTES
            )
        except Exception as exc:
            raise SnapshotUnavailable(
                f"could not mirror epoch {pointer.epoch_id} log bytes: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._verify_digest_chain(pointer, data)

        # This parser is the submission boundary. It validates only authentication,
        # canonical/current schema identity, exact chain-expressible u16 bytes, pointer
        # binding and the complete uid/hotkey census needed for a safe write. It does
        # NOT derive economics or judge evidence completeness.
        submission = parse_authority_submission(data, pointer)

        try:
            log = EpochLog.from_json(data)
        except EpochLogInvalid as exc:
            if not allow_economic_disagreement:
                raise SnapshotDigestMismatch(
                    f"epoch {pointer.epoch_id} log failed its own invariants: {exc}"
                ) from exc
            log = None
            submission = replace(
                submission,
                strict_error=_bounded_error(exc),
            )
            logger.critical(
                "authenticated authority epoch failed strict economic/evidence "
                "validation; reporting for manual remediation while submitting "
                "its exact safe authority u16 vector on schedule",
                extra=log_fields(
                    epoch_id=submission.epoch_id,
                    close_block=submission.close_block,
                    snapshot_digest=pointer.snapshot_digest,
                    error=submission.strict_error,
                    submission_action="continue",
                ),
            )
        except Exception as exc:
            if not allow_economic_disagreement:
                raise SnapshotDigestMismatch(
                    f"epoch {pointer.epoch_id} log bytes are not a valid epoch log: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            log = None
            submission = replace(
                submission,
                strict_error=_bounded_error(exc),
            )
            logger.critical(
                "authenticated authority epoch could not be parsed as a strict audit "
                "model; reporting for manual remediation while submitting its exact "
                "safe authority u16 vector on schedule",
                extra=log_fields(
                    epoch_id=submission.epoch_id,
                    close_block=submission.close_block,
                    snapshot_digest=pointer.snapshot_digest,
                    error=submission.strict_error,
                    submission_action="continue",
                ),
            )

        resolved = _ResolvedEpoch(pointer=pointer, log=log, submission=submission)
        logger.info(
            "mirrored + verified finalized epoch log",
            extra=log_fields(
                epoch_id=submission.epoch_id,
                close_block=submission.close_block,
                snapshot_digest=pointer.snapshot_digest,
                miners=len(log.miners) if log is not None else 0,
                strict_audit_model_valid=log is not None,
                anchor_verified=self._verify_anchor and self._anchor_reader is not None,
            ),
        )
        return resolved

    def _fetch_pointer(self) -> EpochPointer:
        try:
            return self._client.latest_pointer()
        except SharedSnapshotError:
            raise
        except Exception as exc:
            raise SnapshotUnavailable(
                f"scoring authority pointer fetch failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _pointer_for(self, epoch_id: int) -> EpochPointer:
        """Fetch a SPECIFIC epoch's pointer (backfill), normalising failures like above."""
        try:
            return self._client.pointer_for(epoch_id)
        except SharedSnapshotError:
            raise
        except Exception as exc:
            raise SnapshotUnavailable(
                f"scoring authority pointer fetch for epoch {epoch_id} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _verify_digest_chain(self, pointer: EpochPointer, data: bytes) -> None:
        """The core guarantee: sha256(bytes) == pointer digest == on-chain anchor.

        Any inequality is tampering → `SnapshotDigestMismatch` (REFUSE). An
        unreadable anchor (not a positive disagreement) is `SnapshotUnavailable`
        (HOLD).
        """
        d_bytes = sha256_hex(data)
        d_api = pointer.snapshot_digest
        if d_bytes != d_api:
            raise SnapshotDigestMismatch(
                f"mirrored bytes sha256 {d_bytes} != pointer snapshot_digest {d_api} "
                "— the epoch log was tampered with; refusing to submit"
            )
        # The pointer's own anchor field must agree with its snapshot_digest (API
        # self-consistency — a cheap check that costs no chain read).
        if pointer.anchor.digest != d_api:
            raise SnapshotDigestMismatch(
                f"pointer anchor digest {pointer.anchor.digest} != snapshot_digest "
                f"{d_api} — the pointer is internally inconsistent; refusing to submit"
            )
        if not self._verify_anchor:
            return
        if pointer.anchor.txid is None:
            # Finalized but not yet anchored: the tamper-evidence root is not on
            # chain yet, so we cannot verify it — HOLD until it is.
            raise SnapshotUnavailable(
                f"epoch {pointer.epoch_id} is finalized but not yet anchored on chain"
            )
        if self._anchor_reader is None:
            # verify_anchor is ON but NO independent chain reader is wired (#3): we
            # verified only two legs — bytes == API == the pointer's OWN anchor field,
            # all supplied by the same (possibly untrusted) authority. Without the
            # genuinely-independent on-chain read the tamper-evidence chain is not
            # closed, so we CANNOT trust the pointer — HOLD, never submit. Production
            # always wires the reader (fail-fast at startup, run_weightsetter); a test
            # that wants two-leg-only must set verify_anchor=False explicitly.
            raise SnapshotUnavailable(
                f"epoch {pointer.epoch_id}: verify_anchor is on but no on-chain anchor "
                "reader is wired — cannot verify the third (independent) digest leg; "
                "HOLDING rather than trusting authority-supplied fields alone"
            )
        try:
            historical = getattr(self._anchor_reader, "read_epoch_anchor_at", None)
            if pointer.anchor.block is not None and callable(historical):
                on_chain = historical(
                    netuid=self._netuid,
                    epoch_id=pointer.epoch_id,
                    block_number=pointer.anchor.block,
                )
            else:
                on_chain = self._anchor_reader.read_epoch_anchor(
                    netuid=self._netuid, epoch_id=pointer.epoch_id
                )
        except SnapshotUnavailable:
            raise
        except Exception as exc:
            raise SnapshotUnavailable(
                f"could not read the on-chain anchor for epoch {pointer.epoch_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if on_chain is None:
            # The pointer claims a txid, but the chain positively holds NO anchor for
            # this epoch — a substituted pointer. Refuse.
            raise SnapshotDigestMismatch(
                f"pointer claims anchor txid {pointer.anchor.txid} but the chain holds "
                f"NO anchored digest for epoch {pointer.epoch_id} — refusing to submit"
            )
        if on_chain != d_api:
            raise SnapshotDigestMismatch(
                f"on-chain anchored digest {on_chain} != pointer/snapshot digest {d_api} "
                "— the epoch log was tampered with; refusing to submit"
            )


def make_snapshot_provider(
    config: object,
    *,
    store: EpochLogStore | None = None,
    local_provider: object,
    authority_client: ScoringAuthorityClient | None = None,
    anchor_reader: EpochAnchorReader | None = None,
    signer: object | None = None,
) -> object:
    """Select the weight-setter's `SnapshotProvider` by config (rule 8).

    `config` is a `WeightSetterConfig`. `provider = "shared"` builds a
    `SharedSnapshotProvider` (fetch from the authority + mirror + verify);
    `provider = "local"` returns `local_provider` (the existing `miner_manager`
    path, for report-mode / dryrun / third-party recompute) UNCHANGED. Both drive
    the exact same weight-setter code — only the provider swaps.

    An injected `authority_client` (tests) is used as-is; otherwise an
    `HttpScoringAuthorityClient` is built from `config.authority_url` /
    `config.authority_token`.
    """
    provider_mode = getattr(config, "provider", "local")
    if provider_mode != "shared":
        return local_provider
    if store is None:
        raise ValueError(
            "provider = 'shared' needs an object store to mirror epoch-log bytes from"
        )
    client = authority_client
    if client is None:
        client = HttpScoringAuthorityClient(
            getattr(config, "authority_url", ""),
            token=getattr(config, "authority_token", ""),
            timeout=getattr(config, "authority_timeout_seconds", 10.0),
            signer=signer,
        )
    return SharedSnapshotProvider(
        client=client,
        store=store,
        netuid=getattr(config, "authority_netuid", 85),
        anchor_reader=anchor_reader,
        verify_anchor=getattr(config, "verify_anchor", True),
    )


__all__ = [
    "SharedSnapshotProvider",
    "EpochInputs",
    "ScoringAuthorityClient",
    "HttpScoringAuthorityClient",
    "EpochLogStore",
    "EpochAnchorReader",
    "EpochBoundaryReader",
    "ChainAdapterAnchorReader",
    "InMemoryChainAnchorReader",
    "SharedSnapshotError",
    "SnapshotUnavailable",
    "SnapshotDigestMismatch",
    "AuthoritySubmissionView",
    "parse_authority_submission",
    "make_snapshot_provider",
]
