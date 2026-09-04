"""ScoringAuthority — the THIN POINTER API over the epoch object store (wave 4).

A `BaseService` that publishes, per finalized epoch, a POINTER (object key + digests
+ on-chain anchor) — never the epoch-log bytes. Validators fetch a pointer here, then
mirror the bytes directly from the object store by `snapshot_key` and verify
`sha256(bytes) == snapshot_digest == on-chain anchored digest` before submitting
(the project design record §1(a), §3.1, §4/§5). The store is the content plane; this
API is a cheap, cacheable index.

It also composes the authority's epoch-CLOSE sequence — `finalize_and_anchor`: run the
finalizer (write the immutable `_FINALIZED` epoch-log set to the object store), record
the pointer in the epoch index, and anchor the `log_digest` on chain. Idempotent per
epoch (finalize, index-record, and anchor are each idempotent), so a re-run is a full
no-op returning the same pointer.

Modes (the project design record rule 8): both report and bittensor drive this exact code —
only the injected `AuditStore` / `ChainAdapter` differ. Tests wire a `LocalFsStore` +
`InMemoryChain`; production builds them from the shared `audit:` / `chain:` sections.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from prometheus_client import Counter

from vidaio.audit.config import AuditConfig
from vidaio.audit.store import ArtifactKind, AuditStore, make_public_store, make_store
from vidaio.authority.anchoring import anchor_epoch
from vidaio.authority.api import (
    AnchorRecord,
    EpochPointer,
    anchor_from_record,
    pointer_from_record,
)
from vidaio.authority.config import AuthorityConfig
from vidaio.authority.finalizer import (
    EPOCH_LOG_MEMBER,
    EpochFinalizer,
    FinalizedEpoch,
    epoch_prefix,
)
from vidaio.authority.index import EpochIndex
from vidaio.chain.adapter import ChainAdapter, resolve_burn_uid
from vidaio.chain.factory import ChainConfig, make_chain_adapter
from vidaio.competition.anchor_evidence import (
    CompetitionAnchorMismatch,
    CompetitionAnchorUnavailable,
    verify_competition_anchor_on_chain,
)
from vidaio.competition.manifest import CompetitionManifest
from vidaio.core import section
from vidaio.epoch.log import AuditManifest, EpochLog, EpochLogInvalid, MinerCensusEntry
from vidaio.services.base import BaseService
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import MinerSnapshot
from vidaio.tokenomics.state import CompetitionResult, RewardWindowState

#: Typed authorization failures (the `error` field of every 401/403 detail).
AUTH_MISSING = "auth_token_missing"
AUTH_INVALID = "auth_token_invalid"

#: Deliberately merged 404: an unknown epoch and a not-yet-finalized epoch are
#: never distinguished, so an in-progress epoch cannot leak (§3.1).
EPOCH_NOT_FOUND = "epoch_not_found"
_MAX_COMPETITION_MANIFEST_BYTES = 16 * 1024 * 1024


class _DerivePriorDigest:
    """Sentinel default for `finalize_and_anchor(prior_log_digest=...)`.

    Distinguishes "caller did not specify" (derive the prior digest from the authority's own
    epoch index) from an EXPLICIT `None` (genesis — no prior epoch to chain to). Without this,
    the parameter's `None` default would silently make EVERY epoch produced through the public
    path a genesis, which the own-audit genesis gate correctly rejects as a broken chain.
    """


#: Sole instance of the derive-from-index sentinel.
_DERIVE_PRIOR = _DerivePriorDigest()


def _bearer(authorization: str | None) -> str | None:
    """Extract `Authorization: Bearer <token>`; None if absent or malformed."""
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


def _tokens_match(presented: str, configured: str) -> bool:
    """Constant-time compare (over digests, so length leaks nothing either)."""
    return hmac.compare_digest(
        hashlib.sha256(presented.encode("utf-8")).digest(),
        hashlib.sha256(configured.encode("utf-8")).digest(),
    )


class ScoringAuthority(BaseService):
    name = "scoring-authority"

    def __init__(
        self,
        raw_config: dict[str, Any],
        *,
        metrics_port: int | None = None,
        store: AuditStore | None = None,
        public_store: AuditStore | None = None,
        chain: ChainAdapter | None = None,
        index: EpochIndex | None = None,
        finalizer: EpochFinalizer | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        cfg = section(raw_config, "authority", AuthorityConfig)
        super().__init__(
            raw_config,
            metrics_port=metrics_port if metrics_port is not None else cfg.metrics_port,
        )
        self.config = cfg
        chain_config = section(raw_config, "chain", ChainConfig)
        self._anchor_hotkey = (
            chain_config.anchor_hotkey or chain_config.validator_hotkey
        )
        self._anchor_writer_lock_path = chain_config.anchor_writer_lock_path
        self._anchor_writer_lock_timeout_seconds = (
            chain_config.anchor_writer_lock_timeout_seconds
        )
        self._now = now or (lambda: datetime.now(timezone.utc))

        # The content plane + chain + index + producer. Injected in tests
        # (LocalFsStore + InMemoryChain); built from the shared config sections in
        # production — both paths drive identical service code.
        audit_config = section(raw_config, "audit", AuditConfig)
        self._store: AuditStore = (
            store
            if store is not None
            else make_store(audit_config)
        )
        # A distinct keyless view is load-bearing for CROWN publication.  It uses
        # unsigned S3 requests in production, so the finalizer proves that the
        # released winner is reachable by independent auditors before it publishes
        # or anchors the earning epoch.
        self._public_store: AuditStore = (
            public_store
            if public_store is not None
            else make_public_store(audit_config)
        )
        self._chain: ChainAdapter = (
            chain if chain is not None else make_chain_adapter(raw_config)
        )
        # P2 registered-hotkey auth: one cached, fail-closed registry per process.
        # Mode `off` skips construction entirely (dev/report); `log` observes;
        # `enforce` refuses. The static bearer stays as a second factor during
        # the migration (the design notes §3.2).
        from vidaio.services.hotkey_auth import (
            HotkeyAuthConfig,
            HotkeyAuthGuard,
            RegisteredHotkeyRegistry,
        )

        hk_cfg = section(raw_config, "hotkey_auth", HotkeyAuthConfig)
        self._hotkey_guard: HotkeyAuthGuard | None = None
        if hk_cfg.mode != "off":
            self._hotkey_guard = HotkeyAuthGuard(
                RegisteredHotkeyRegistry(
                    self._chain,
                    ttl_seconds=hk_cfg.registry_ttl_seconds,
                    max_stale_seconds=hk_cfg.registry_max_stale_seconds,
                ),
                hk_cfg,
            )
        self._index: EpochIndex = (
            index if index is not None else EpochIndex.open(cfg.db_path)
        )
        self._finalizer: EpochFinalizer = (
            finalizer
            if finalizer is not None
            else EpochFinalizer(
                section(raw_config, "tokenomics", TokenomicsConfig),
                scorer_version=cfg.scorer_version,
            )
        )

        self._http_api_ok = True
        self.health.register_check("http_api", lambda: self._http_api_ok)

        reg = self.health.registry
        self._m_pointer_reads = Counter(
            "vidaio_authority_pointer_reads_total",
            "Epoch-pointer reads by route and outcome",
            ["route", "outcome"],
            registry=reg,
        )
        self._m_finalized = Counter(
            "vidaio_authority_epochs_finalized_total",
            "Epochs finalized+anchored by this authority (new vs idempotent re-run)",
            ["outcome"],
            registry=reg,
        )
        self._m_auth_failures = Counter(
            "vidaio_authority_auth_failures_total",
            "Rejected pointer reads by typed authorization error",
            ["error"],
            registry=reg,
        )
        self.app = self._build_app()

    # -- the epoch-close sequence: finalize -> index -> anchor ------------------

    async def finalize_and_anchor(
        self,
        *,
        epoch_id: int,
        close_block: int,
        snapshots: Sequence[MinerSnapshot],
        miner_census: Sequence[MinerCensusEntry] | None = None,
        audit_manifest: AuditManifest,
        now: datetime | None = None,
        prior_log_digest: "str | None | _DerivePriorDigest" = _DERIVE_PRIOR,
        competition_result: CompetitionResult | None = None,
        prior_reward_window_state: RewardWindowState | None = None,
        competition_packet_scores: Mapping[str, float] | None = None,
    ) -> FinalizedEpoch:
        """Close an epoch: write the `_FINALIZED` log, index the pointer, anchor it.

        Idempotent per epoch — each step (finalize / index / anchor) is a no-op on a
        re-run, so calling this twice for the same epoch returns the same pointer and
        never double-writes the store or the chain.

        `prior_log_digest` chains this epoch to its predecessor: the log's
        `EarningInput.prior_accumulate_score` carry-in is verifiable back to genesis only when the
        log commits the prior epoch's `log_digest`, and the own-audit genesis gate DISPUTES any
        NON-genesis epoch that omits it. When left DEFAULT it is DERIVED from the authority's own
        durable epoch index (the newest finalized epoch strictly before this one — the same source
        the local-stack finalizer's `_prior_digest` resumes from), so successive epochs produced
        through this public path chain properly. Genesis (no prior finalized epoch) derives `None`;
        pass `None` explicitly to force genesis.
        """
        when = now or self._now()
        prior_log: EpochLog | None = None
        if isinstance(prior_log_digest, _DerivePriorDigest):
            prior_log = self._prior_log_for(epoch_id)
            prior_log_digest = prior_log.log_digest() if prior_log is not None else None
        elif prior_log_digest is not None:
            # An explicit predecessor digest still has to resolve to this authority's durable
            # predecessor bytes.  Otherwise neither cumulative replay history nor carry state
            # can be derived, and publishing with the finalizer's pure-model compatibility
            # defaults would silently turn schema-v11 continuity off on this production path.
            prior_log = self._prior_log_for(epoch_id)
            if prior_log is None or prior_log.log_digest() != prior_log_digest:
                raise EpochLogInvalid(
                    f"epoch {epoch_id} explicitly references predecessor digest "
                    f"{prior_log_digest}, but that log is not the authority index's readable "
                    "predecessor — refusing to publish without verifiable cumulative replay "
                    "history (schema v15)"
                )

        prior_fold_cursors = (
            prior_log.audit_manifest.fold_cursors if prior_log is not None else {}
        )
        census_uids = (
            (entry.uid for entry in miner_census)
            if miner_census is not None
            else (miner.uid for miner in snapshots)
        )
        audit_manifest = self._bind_fold_cursors(
            audit_manifest, prior_fold_cursors, census_uids
        )
        prior_earning = {
            miner.uid: (miner.hotkey, float(miner.accumulate_score))
            for miner in (prior_log.miners if prior_log is not None else ())
        }
        anchored_prior_reward_state = (
            prior_log.reward_window_state
            if prior_log is not None
            else RewardWindowState()
        )
        if (
            prior_reward_window_state is not None
            and prior_reward_window_state != anchored_prior_reward_state
        ):
            raise EpochLogInvalid(
                "caller-supplied prior reward window does not equal the immutable "
                "predecessor epoch state"
            )
        resolved_prior_reward_state = anchored_prior_reward_state
        if audit_manifest.competition_input is not None:
            self._verify_competition_anchor(
                audit_manifest, epoch_close_block=close_block
            )
        # Empty-epoch emission identity is mutable chain state (the subnet owner
        # hotkey may move and uids may be recycled). A production adapter must
        # resolve it now; only report/test adapters may use an explicit fallback.
        canonical_burn_uid = resolve_burn_uid(
            self._chain, report_fallback=self.config.burn_uid
        )
        finalized = self._finalizer.finalize(
            epoch_id=epoch_id,
            close_block=close_block,
            snapshots=snapshots,
            miner_census=miner_census,
            burn_uid=canonical_burn_uid,
            audit_manifest=audit_manifest,
            store=self._store,
            public_store=self._public_store,
            now=when,
            prior_log_digest=prior_log_digest,
            prior_earning=prior_earning,
            prior_fold_cursors=prior_fold_cursors,
            competition_result=competition_result,
            prior_reward_window_state=resolved_prior_reward_state,
            competition_packet_scores=competition_packet_scores,
        )
        self._index.record_finalized(finalized, finalized_at=when.isoformat())
        await anchor_epoch(
            finalized,
            chain=self._chain,
            index=self._index,
            netuid=self.config.netuid,
            now=when,
            anchor_hotkey=self._anchor_hotkey,
            writer_lock_path=self._anchor_writer_lock_path,
            writer_lock_timeout_seconds=(self._anchor_writer_lock_timeout_seconds),
        )
        self._m_finalized.labels(
            "idempotent" if finalized.already_finalized else "new"
        ).inc()
        return finalized

    def _verify_competition_anchor(
        self, manifest: AuditManifest, *, epoch_close_block: int
    ) -> None:
        """Refuse an earning input not independently proven from archive state."""

        competition_input = manifest.competition_input
        assert competition_input is not None
        try:
            manifest_raw = self._store.get_digest_limited(
                ArtifactKind.MANIFEST,
                competition_input.manifest_digest,
                max_bytes=_MAX_COMPETITION_MANIFEST_BYTES,
            )
            competition_manifest = CompetitionManifest.model_validate_json(manifest_raw)
            if competition_manifest.canonical_json().encode("utf-8") != manifest_raw:
                raise ValueError("competition manifest is not its canonical preimage")
            if (
                competition_manifest.manifest_digest()
                != competition_input.manifest_digest
                or competition_manifest.competition_id
                != competition_input.competition_id
                or competition_manifest.track != competition_input.track
            ):
                raise ValueError(
                    "competition manifest identity does not bind CompetitionInput"
                )
            verify_competition_anchor_on_chain(
                self._chain,
                competition_input,
                expected_netuid=self.config.netuid,
                competition_start_time=competition_manifest.start_time,
                epoch_close_block=epoch_close_block,
            )
        except (
            CompetitionAnchorMismatch,
            CompetitionAnchorUnavailable,
            FileNotFoundError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise EpochLogInvalid(
                "earning competition pre-enrollment anchor receipt cannot be "
                f"independently proven: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _bind_fold_cursors(
        manifest: AuditManifest,
        prior_fold_cursors: dict[int, int | None],
        current_census_uids: Iterable[int],
    ) -> AuditManifest:
        """Carry the schema-v15 total replay boundary into a caller-built manifest.

        ``finalize_and_anchor`` accepts an already-built manifest, and historical callers use
        ``build_audit_manifest(items)`` before this service has resolved the predecessor.  That
        builder can only claim this epoch's cycle maxima.  Accept exactly that current-only form
        (or a predecessor-complete/already-total form), add every current census uid as an
        observed-never-folded ``None`` cursor, carry predecessor tombstones, and reject every
        other claimed map. The finalizer independently checks the resulting exact map.
        """
        expected = {
            int(uid): None if cursor is None else int(cursor)
            for uid, cursor in prior_fold_cursors.items()
        }
        predecessor_plus_current = dict(expected)
        for uid in current_census_uids:
            expected.setdefault(int(uid), None)
        current_only: dict[int, int] = {}
        for uid, earning_input in manifest.earning_inputs.items():
            keys = [cycle.ordering_key for cycle in earning_input.cycle_scores]
            if not keys:
                continue
            prior = expected.get(uid)
            if prior is not None and min(keys) <= prior:
                raise EpochLogInvalid(
                    f"uid {uid} folds ordering_key(s) {sorted(keys)} at/below cumulative "
                    f"predecessor cursor {prior} — cross-epoch packet replay"
                )
            current_max = max(keys)
            current_only[int(uid)] = current_max
            predecessor_plus_current[int(uid)] = current_max
            expected[int(uid)] = current_max
        claimed = {
            int(uid): None if cursor is None else int(cursor)
            for uid, cursor in manifest.fold_cursors.items()
        }
        if claimed not in (current_only, predecessor_plus_current, expected):
            raise EpochLogInvalid(
                "audit manifest carries a partial, invented, or regressed fold_cursors map: "
                f"expected current-only {current_only}, predecessor-complete "
                f"{predecessor_plus_current}, or total {expected}; got {claimed}"
            )
        if claimed == expected:
            return manifest
        return manifest.model_copy(update={"fold_cursors": expected})

    def _prior_log_for(self, epoch_id: int) -> EpochLog | None:
        """Load and verify the durable predecessor bytes used for all chained state."""
        latest = self._index.latest()
        if latest is None or latest.epoch_id >= epoch_id:
            return None
        data = self._store.get_set_member(  # type: ignore[attr-defined]
            epoch_prefix(latest.epoch_id),
            EPOCH_LOG_MEMBER,
            expected_digest=latest.log_digest,
        )
        log = EpochLog.from_json(data)
        if log.epoch_id != latest.epoch_id or log.close_block != latest.close_block:
            raise EpochLogInvalid(
                "authority predecessor index metadata does not match its immutable log bytes: "
                f"index epoch/close=({latest.epoch_id}, {latest.close_block}), log="
                f"({log.epoch_id}, {log.close_block})"
            )
        return log

    def _prior_log_digest_for(self, epoch_id: int) -> str | None:
        """The prior epoch's finalized `log_digest` to chain `epoch_id` to.

        Derived from the authority's OWN durable index — the newest finalized epoch strictly
        BEFORE this one (mirrors the local-stack finalizer resuming `_prior_digest` from
        `index.latest()`). `None` when no earlier finalized epoch exists (genesis / a configured
        genesis floor), which correctly leaves the log a genesis the own-audit gate accepts. The
        `< epoch_id` guard keeps an idempotent RE-run of an already-finalized epoch from chaining
        to itself (the re-run short-circuits in the finalizer anyway, but this stays correct).
        """
        latest = self._index.latest()
        if latest is None or latest.epoch_id >= epoch_id:
            return None
        return latest.log_digest

    # -- authorization ---------------------------------------------------------

    def _authorize(self, authorization: str | None) -> None:
        """Gate every epoch-pointer route on `authority.api_token` when configured.

        401 on a missing/malformed bearer, 403 on a present-but-wrong token. Open
        (no gate) when `api_token` is unset — a loopback/dev posture; production
        sets it (validators carry it).
        """
        configured = (self.config.api_token or "").strip()
        if not configured:
            return
        presented = _bearer(authorization)
        if presented is None:
            self._m_auth_failures.labels(AUTH_MISSING).inc()
            raise HTTPException(
                status_code=401,
                detail={
                    "error": AUTH_MISSING,
                    "message": "this endpoint requires 'Authorization: Bearer "
                    "<authority.api_token>'",
                },
            )
        if not _tokens_match(presented, configured):
            self._m_auth_failures.labels(AUTH_INVALID).inc()
            raise HTTPException(
                status_code=403,
                detail={
                    "error": AUTH_INVALID,
                    "message": "the presented bearer token is not valid for the "
                    "scoring authority",
                },
            )

    def _require_hotkey_auth(
        self, request: Request, body: bytes = b""
    ) -> None:
        """P2: verify the caller is a registered (permitted) hotkey.

        `log` mode observes and never refuses; `enforce` maps the typed refusal
        onto its HTTP status. Validator permit is required on every pointer
        route — these are validator/auditor surfaces (owner decision 2026-08-26).
        """
        if self._hotkey_guard is None:
            return
        from vidaio.services.hotkey_auth import HotkeyAuthError

        try:
            self._hotkey_guard.require(
                dict(request.headers),
                method=request.method,
                path=request.url.path,
                body=body,
                require_validator_permit=True,
            )
        except HotkeyAuthError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc

    def _not_found(self, route: str, epoch_id: int | None) -> HTTPException:
        self._m_pointer_reads.labels(route, "not_found").inc()
        target = "any finalized epoch" if epoch_id is None else f"epoch {epoch_id}"
        return HTTPException(
            status_code=404,
            detail={
                "error": EPOCH_NOT_FOUND,
                "message": f"no finalized pointer for {target} (unknown or not yet "
                "finalized — never distinguished, so an in-progress epoch cannot leak)",
            },
        )

    # -- the pointer API -------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="vidaio scoring authority", docs_url=None, redoc_url=None)

        @app.get("/healthz")
        async def healthz() -> dict[str, Any]:
            return {"service": self.name, "status": "ok"}

        @app.get("/epoch/latest", response_model=EpochPointer)
        async def epoch_latest(
            request: Request,
            authorization: str | None = Header(default=None),
        ) -> EpochPointer:
            self._authorize(authorization)
            self._require_hotkey_auth(request)
            record = self._index.latest()
            if record is None:
                raise self._not_found("latest", None)
            self._m_pointer_reads.labels("latest", "ok").inc()
            return pointer_from_record(record)

        @app.get("/epoch/{epoch_id}", response_model=EpochPointer)
        async def epoch_get(
            request: Request,
            epoch_id: int,
            authorization: str | None = Header(default=None),
        ) -> EpochPointer:
            self._authorize(authorization)
            self._require_hotkey_auth(request)
            record = self._index.get(epoch_id)
            if record is None:
                raise self._not_found("get", epoch_id)
            self._m_pointer_reads.labels("get", "ok").inc()
            return pointer_from_record(record)

        @app.get("/epoch/{epoch_id}/anchor", response_model=AnchorRecord)
        async def epoch_anchor(
            request: Request,
            epoch_id: int,
            authorization: str | None = Header(default=None),
        ) -> AnchorRecord:
            self._authorize(authorization)
            self._require_hotkey_auth(request)
            record = self._index.get(epoch_id)
            if record is None:
                raise self._not_found("anchor", epoch_id)
            self._m_pointer_reads.labels("anchor", "ok").inc()
            return anchor_from_record(record)

        @app.post("/auth/challenge")
        async def auth_challenge() -> dict[str, Any]:
            """Scheme B step 1: a server nonce the caller signs to mint a token.

            Open by design: the nonce proves nothing by itself, the store is
            bounded, and gating it behind the static bearer would break the
            bearer's planned retirement. Disabled entirely when hotkey auth is
            off."""
            if self._hotkey_guard is None:
                raise HTTPException(status_code=404, detail="hotkey auth disabled")
            return {"challenge": self._hotkey_guard.mint_challenge()}

        @app.post("/auth/token")
        async def auth_token(request: Request) -> dict[str, Any]:
            """Scheme B step 2: redeem a signed challenge for a session token."""
            if self._hotkey_guard is None:
                raise HTTPException(status_code=404, detail="hotkey auth disabled")
            from vidaio.services.hotkey_auth import HotkeyAuthError

            body = await request.body()
            challenge = body.decode("ascii", errors="replace").strip()
            try:
                token = self._hotkey_guard.redeem_challenge(
                    dict(request.headers),
                    challenge=challenge,
                    path=request.url.path,
                )
            except HotkeyAuthError as exc:
                raise HTTPException(
                    status_code=exc.status_code,
                    detail={"code": exc.code, "message": str(exc)},
                ) from exc
            return {"token": token}

        return app

    # -- lifecycle -------------------------------------------------------------

    def _on_api_exit(self, api: asyncio.Task[Any]) -> None:
        """The uvicorn task ended without a stop being requested — fatal exit so a
        supervisor restarts an authority whose port nobody answers (the exit-code
        contract, vidaio.services.base)."""
        self._http_api_ok = False
        error: BaseException | None = None
        try:
            error = api.exception()
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            error = None
        detail = "" if error is None else f"{type(error).__name__}: {error}"
        self.fail_fatal(
            "scoring-authority HTTP API exited unexpectedly — no pointer API is serving"
            f" (port={self.config.http_port} error={detail})"
        )

    async def run(self) -> None:
        server = uvicorn.Server(
            uvicorn.Config(
                self.app,
                host=self.config.http_host,
                port=self.config.http_port,
                log_level="warning",
            )
        )

        # SystemExit (uvicorn's bind-failure exit) is a BaseException: awaited bare
        # it would tear the loop down instead of being reported.
        async def _serve() -> None:
            try:
                await server.serve()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                raise RuntimeError(
                    f"uvicorn exited: {type(exc).__name__}: {exc}"
                ) from exc

        api = asyncio.create_task(_serve(), name="scoring-authority-http")
        stop = asyncio.create_task(self.stopping.wait(), name="scoring-authority-stop")
        try:
            await asyncio.wait({api, stop}, return_when=asyncio.FIRST_COMPLETED)
            if api.done() and not self.stopping.is_set():
                self._on_api_exit(api)
        finally:
            server.should_exit = True
            stop.cancel()
            await asyncio.gather(api, return_exceptions=True)
            self.close()

    def close(self) -> None:
        self._index.close()
