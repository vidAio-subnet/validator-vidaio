"""Competition orchestrator — the bulletproof service of spec §14.

Drives LifecycleEngine.tick on an interval plus the phase work between ticks:
submission backup, validation intake, sandbox builds + isolation probes,
batched evaluation, scoring + audit-bundle linkage. Composes over the
review-shipped engine/repository/review modules — it never modifies them.

Bulletproofing invariants (spec §14, fault-injection checklist):
- Every stage is IDEMPOTENT re-entry: what is already built/evaluated/scored is
  read from the DB (contender status, batch rows + batch_outputs events,
  performance rows) before anything is redone — a crash mid-stage loses at most
  the operation in flight and never corrupts committed state. Modal image handles
  are the deliberate exception to "nothing depends on memory": a persisted digest
  is not executable. A unique persisted runtime-session fence plus an exact
  competition-owned immutable Image-id binding lets a fresh process rehydrate and
  reprobe only its own images and, for EVALUATING, append-only-reset and rerun the
  whole batch matrix before it can continue; it never mixes runtime outputs,
  inventories cloud resources, or restores a Sandbox/instance.
- Per-contender failures (build, probe) mark THAT contender failed; the engine
  fails the competition only when ALL builds failed.
- Infra failures on batches requeue with a bounded budget; scoring transport
  failures retry with bounded backoff. Exhaustion HALTS the pipeline (CRITICAL
  log + `orchestrator_halted` event, cleared by an operator through authenticated
  `POST /competitions/{id}/halt/clear`, with operator + reason in the append-only
  log) — the competition is never marked FAILED by an infra blocker.
- Every boundary is bounded (runner-internal subprocess timeouts + retry_async
  budgets); scores enter the DB only as verbatim scorer packets
  (repository.record_item_score) and every persisted score is audit-linked in
  the same transaction (build_bundle -> set_audit_bundle_digest), so the
  engine's completion gate can never see a half-recorded score.

Local-first artifact convention (shared with DockerSandboxRunner):
  <work_dir>/inputs/<sha256>   sealed input pool (staged by add_evaluation_item)
  <work_dir>/outputs/<sha256>  collected output pool (written by the runner)

--------------------------------------------------------------------------------
FAULT CLASSIFICATION
--------------------------------------------------------------------------------
EVERY failure crossing a stage is classified by orchestrator.failures — no stage
hard-codes a verdict any more (an internal review, round 2: four stages used to bypass the
classifier, and each bypass blamed a contender for something we broke):
- CONTENDER fault (the solution exited non-zero, timed out, wrote an unsafe or
  oversize output, produced no output, shipped a Dockerfile that does not build,
  failed an isolation probe that actually RAN, or had ITS OWN OUTPUT rejected by
  the trusted scorer) -> THAT contender's batch is failed and its items are
  zero-scored with a reason code. The competition CONTINUES.
- INFRA fault (docker down, image gone, sealed input missing, a checkout we could
  not materialize, the sandbox isolation contract not holding, a probe that could
  not be RUN, a scorer-identity disagreement, a scorer rejection naming OUR half
  of the request, scorer 5xx/transport, DB errors) -> the bounded requeue/retry
  path, then HALT. Never a FAILED competition, never a substituted score.
A missing (or empty) output is NEVER sent to the scorer: it is zero-scored
locally with a machine-readable reason code, so an untrusted contender can no
longer escalate ffmpeg's 502 into a pipeline halt.

--------------------------------------------------------------------------------
SUBMISSION ARCHIVE INVARIANT
--------------------------------------------------------------------------------
EVERY CONTENDER THAT CAN STILL WIN HAS AN ARCHIVED SUBMISSION. Finalization used
to skip a contender whose checkout failed and still record the combined
backup_ref, certifying an archive set that did not exist — a contender could then
compete and win with nothing archived to audit. Now: a CONTENDER-fault tree
(symlinks, oversize) REJECTS that contender before it can advance; an INFRA
failure HALTS finalization (the phase does not advance at all) instead of
certifying a partial set. The per-contender evidence lives in the append-only
event log (`contender_submission_archived`), so re-entry after a crash archives
exactly what is still missing.

--------------------------------------------------------------------------------
CONTROL SURFACE + CHAIN
--------------------------------------------------------------------------------
- `orchestrator/control.py` serves the competition control API (create, enroll,
  anchor, audited halt-clear, status, review, result) on
  `orchestrator.control_port`, token-authed.
  It is only started when a control token is configured. The serve task is
  MONITORED: if it exits unexpectedly the `control_api` health check flips and the
  service requests stop, instead of staying "healthy" with no control plane
.
- Commitment anchoring goes through the injected ChainAdapter — `anchor_competition`
  is THE single anchor path shared by chainless report mode and the future real
  chain (the project design record rule 8). It builds the CompetitionCommitment from the
  PERSISTED manifest, archives the canonical JSON, CLAIMS THE ANCHORING RIGHT IN
  THE DB, calls the adapter, and only then records the root.

  ANCHOR CLAIM PROTOCOL — the chain write used to happen
  BEFORE the guarded DB transition, so two concurrent requests carrying different
  payloads could both anchor while only the first to return was recorded:
    1. BEGIN IMMEDIATE. Verify: competition exists, still SCHEDULED, no
       commitment_root, and no open anchor claim. Record the claim with the EXACT
       payload digest about to be written. COMMIT.
       A second concurrent request fails this step and is refused (409) BEFORE it
       can touch the chain.
    2. Submit at most once, then independently poll the raw commitment record
       through finality and exact inclusion-block archive read-back.
    3. Atomically record the root + complete `commitment_anchored_onchain`
       receipt, which RESOLVES the claim.
  A crash between 2 and 3 leaves an open claim naming the payload digest, which is
  re-checkable: if the root is already recorded the anchor is simply done;
  otherwise the claim is AMBIGUOUS (the extrinsic may or may not have landed —
  the same ambiguity the weight-setter handles for set_weights). While the claim
  is fresh, every request is refused. Once it is stale
  (`anchor_claim_stale_seconds`) the IDENTICAL payload may be checked again in
  READ-ONLY mode; it is never resubmitted. A DIFFERENT payload is refused until
  an operator proves nothing landed and calls `release_anchor_claim`.
- `build_result` (orchestrator/results.py) converts a COMPLETED competition into
  the tokenomics CompetitionResult so the weight-setter side can ingest it.

--------------------------------------------------------------------------------
SCORER IDENTITY (vidaio/services/protocol.py — THE SCORER-IDENTITY CONTRACT)
--------------------------------------------------------------------------------
`CompetitionManifest.scoring_version` is the FULL effective scorer identity
(`<name>+<identity digest[:12]>`), not a label: the manifest digest is anchored
before enrollment, so the competition commits to the scorer that will measure it.
`_check_scorer_identity` compares the live worker's advertised identity to the
persisted manifest at competition start and again before SCORING; disagreement is
an INFRA halt with an explicit reason. `scorer_identity()` is the helper callers
use to AUTHOR a manifest against the worker that will actually run.

ORCHESTRATOR-MINTED ZEROS. The orchestrator mints its own
packet in exactly one case — an item with no measurable bytes (no output, or an
output the worker rejected). Those packets are stamped with a RESERVED, distinct
identity, `orchestrator-zero/1+<digest12>`, never with the worker's:
see orchestrator/zero_packets.py, which defines the convention, and note that
such a packet's `scorer_version` differing from the manifest's `scoring_version`
is CORRECT — it is an orchestrator fact, not a measurement. Both directions of
impersonation are refused: this service never stamps the worker's identity on a
packet it minted, and a manifest or worker claiming the reserved namespace halts
the pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import sqlite3
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import uvicorn
from prometheus_client import Counter, Gauge, Histogram

from vidaio.audit import ArtifactKind, ArtifactRef, AuditConfig, AuditStore
from vidaio.audit import (
    CompetitionItemBinding,
    LifecycleStage,
    build_bundle,
    make_store,
)
from vidaio.audit import (
    CompetitionCommitment,
    NotConfiguredError,
    build_competition_commitment,
    load_competition_commitment,
    pin_git_sha,
    reward_parameter_digest,
)
from vidaio.audit.store import backend_key
from vidaio.audit.canonical import canonical_json_bytes
from vidaio.chain.adapter import ChainAdapter
from vidaio.chain.anchor_receipt import (
    AnchorReceiptVerificationError,
    wait_for_finalized_commitment_receipt,
)
from vidaio.chain.anchor_writer import anchor_writer_lock
from vidaio.chain.factory import ChainConfig
from vidaio.core import connect, section
from vidaio.core.logging import log_fields
from vidaio.core.resilience import RetriesExhausted, retry_async, with_timeout
from vidaio.competition import (
    CompetitionConfig,
    CompetitionManifest,
    LifecycleEngine,
    migrate,
)
from vidaio.competition import repository as repo
from vidaio.competition.interfaces import (
    BUILD_IDENTITY_SCHEME,
    BatchOutput,
    CompetitionScoringClient,
    ContenderSpec,
    SandboxRunner,
    logical_build_identity,
)
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.orchestrator.results import ResultNotReady
from vidaio.competition.orchestrator.config import OrchestratorConfig
from vidaio.competition.orchestrator.failures import Fault, classify_failure, fault_code
from vidaio.competition.orchestrator.zero_packets import (
    ReservedScorerIdentity,
    assert_not_reserved,
    is_orchestrator_zero_identity,
    mint_zero_packet,
)
from vidaio.competition.review import submit_review as _submit_review
from vidaio.services.commitment_capacity import (
    CommitmentCapacityError,
    EPOCH_ANCHOR_CAPACITY_RESERVE_BYTES,
    require_commitment_capacity,
)
from vidaio.competition.runners import safeio
from vidaio.competition.runners.docker_runner import DockerSandboxRunner
from vidaio.competition.runners.repo import (
    RepoProvider,
    checkout_pinned,
    release_checkout,
)
from vidaio.competition.states import Phase
from vidaio.epoch.log import MinerCensusEntry
from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.gates import ReasonCode
from vidaio.scoring.result import ItemScore, config_digest
from vidaio.services.base import BaseService
from vidaio.services.protocol import ScorerIdentityUnavailable, ScorerRuntimeMismatch
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import CompetitionResult

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
MAX_COMPETITION_INGEST_BYTES = 2 * 1024 * 1024 * 1024

#: Appended to the halt reason when someone claims the reserved zero namespace.
_RESERVED_FIX = (
    "rename the scorer (its identity is the worker's to mint, but not this name), "
    "then clear_halt."
)


class AnchorError(Exception):
    """An anchor did not obtain an exact finalized/archive receipt."""


class EarningManifestError(ValueError):
    """A manifest cannot safely feed the enabled competition-emissions path."""


class AnchorClaimRefused(Exception):
    """Anchoring was refused BEFORE any chain write.

    Raised by the claim step: another anchor is in flight, the competition is
    already anchored or past SCHEDULED, or an ambiguous claim from a crashed
    attempt blocks a DIFFERENT payload. The control API maps it to 409 — the
    caller must never read it as "the chain write failed", because there was none.
    """

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        #: machine-readable: anchor_in_progress | already_anchored | not_scheduled
        #: | anchor_ambiguous | unknown_competition
        self.code = code


@dataclass(frozen=True)
class AnchorResult:
    """Outcome of the SINGLE anchor path (report mode and real chain alike)."""

    root: str
    tx_id: str | None
    payload: bytes
    canonical_json: bytes
    baseline_image_digest: str
    anchor_block: int
    anchor_block_hash: str
    finalized_block: int
    write_response_recovered: bool
    #: False when the lifecycle engine refused to record it (already anchored, or
    #: the competition has left SCHEDULED). The chain write still happened.
    recorded: bool


def build_docker_runner(
    cfg: OrchestratorConfig,
    repo_provider: RepoProvider,
    *,
    docker_path: str = "docker",
) -> DockerSandboxRunner:
    """Wire a DockerSandboxRunner to the orchestrator's work-dir convention and
    resource limits. Raises RunnerUnavailableError when docker is unusable."""
    work = Path(cfg.work_dir)
    return DockerSandboxRunner(
        repo_provider,
        inputs_dir=work / "inputs",
        outputs_dir=work / "outputs",
        scratch_dir=work / "scratch",
        docker_path=docker_path,
        build_timeout=cfg.build_timeout_seconds,
        batch_timeout=cfg.batch_timeout_seconds,
        probe_timeout=cfg.probe_timeout_seconds,
        memory=cfg.sandbox_memory,
        cpus=cfg.sandbox_cpus,
        tmpfs_size=cfg.sandbox_tmpfs_size,
        pids_limit=cfg.sandbox_pids_limit,
        max_output_bytes=cfg.sandbox_output_max_bytes,
        max_batch_output_bytes=cfg.sandbox_batch_output_max_bytes,
        max_log_bytes=cfg.sandbox_log_max_bytes,
        output_poll_seconds=cfg.sandbox_output_poll_seconds,
    )


class Orchestrator(BaseService):
    name = "competition-orchestrator"

    def __init__(
        self,
        raw_config: dict[str, Any],
        *,
        runner: SandboxRunner,
        scoring_client: CompetitionScoringClient,
        repo_provider: RepoProvider,
        store: AuditStore | None = None,
        conn: sqlite3.Connection | None = None,
        chain: ChainAdapter | None = None,
        clock: Any = None,
    ) -> None:
        cfg = section(raw_config, "orchestrator", OrchestratorConfig)
        super().__init__(raw_config, metrics_port=cfg.metrics_port)
        self.cfg = cfg
        self.engine = LifecycleEngine(
            section(raw_config, "competition", CompetitionConfig)
        )
        # Zero records are protocol outputs too: bind their identity/config digest
        # to the exact scoring policy used by the trusted worker and CPU auditors.
        self.scoring_config = section(raw_config, "scoring", ScoringConfig)
        # The shipping testnet configuration enables real competition emissions.
        # Library/report callers retain the model's explicit opt-in default (false),
        # but an earning orchestrator applies additional manifest and commitment
        # invariants before it can create or execute a competition.
        self.tokenomics = section(raw_config, "tokenomics", TokenomicsConfig)
        chain_cfg = section(raw_config, "chain", ChainConfig)
        self._chain_mode = chain_cfg.mode
        self._anchor_netuid = chain_cfg.netuid
        self._anchor_hotkey = chain_cfg.anchor_hotkey or chain_cfg.validator_hotkey
        self._anchor_writer_lock_path = chain_cfg.anchor_writer_lock_path
        self._anchor_writer_lock_timeout_seconds = (
            chain_cfg.anchor_writer_lock_timeout_seconds
        )
        self.runner = runner
        self.scoring_client = scoring_client
        self.repo_provider = repo_provider
        self.store = store or make_store(section(raw_config, "audit", AuditConfig))
        self.conn = conn or connect(self.core.db_path)
        #: The ONE chain seam (the project design record rule 8): report mode injects the
        #: chainsim adapter, production the real one — same call path either way.
        self.chain = chain
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        migrate(self.conn)
        work = Path(cfg.work_dir)
        self.inputs_dir = work / "inputs"
        self.outputs_dir = work / "outputs"
        self.ingest_dir = work / "ingest"
        for d in (self.inputs_dir, self.outputs_dir, self.ingest_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.ingest_dir.chmod(0o700)
        self._last_step_monotonic: float | None = None
        #: competition ids whose manifest scoring_version has been checked against
        #: the live worker at least once (the "at competition start" half).
        self._identity_checked: set[str] = set()
        reg = self.health.registry
        self.m_builds = Counter(
            "vidaio_orchestrator_builds_total",
            "Contender image builds by result (ok / failed / probe_dq)",
            ["result"],
            registry=reg,
        )
        self.m_batches = Counter(
            "vidaio_orchestrator_batches_total",
            "Evaluation batches by result (ok / requeued / contender_failed)",
            ["result"],
            registry=reg,
        )
        self.m_contender_faults = Counter(
            "vidaio_orchestrator_contender_faults_total",
            "Failures attributed to a contender's own submission, by reason code",
            ["code"],
            registry=reg,
        )
        self.m_zero_scored = Counter(
            "vidaio_orchestrator_zero_scored_items_total",
            "Items zero-scored locally with a reason code (never sent to the scorer)",
            ["code"],
            registry=reg,
        )
        self.m_anchors = Counter(
            "vidaio_orchestrator_anchors_total",
            "Competition commitments anchored through the ChainAdapter",
            ["result"],
            registry=reg,
        )
        self.m_scorings = Counter(
            "vidaio_orchestrator_scorings_recorded_total",
            "Per-item score packets persisted (audit-linked)",
            registry=reg,
        )
        self.m_halts = Counter(
            "vidaio_orchestrator_halts_total",
            "Pipeline halts on systemic infra blockers",
            registry=reg,
        )
        self.m_step_errors = Counter(
            "vidaio_orchestrator_step_errors_total",
            "Unexpected exceptions escaping a step",
            registry=reg,
        )
        self.m_stage_seconds = Histogram(
            "vidaio_orchestrator_stage_seconds",
            "Wall seconds spent on one phase's work within a step",
            ["phase"],
            registry=reg,
        )
        self.m_last_tick = Gauge(
            "vidaio_orchestrator_last_tick_timestamp_seconds",
            "Unix time of the last completed engine tick",
            registry=reg,
        )
        #: False once the control-API serve task has exited unexpectedly (review
        #: #22). None-app services are trivially healthy on this axis.
        self._control_api_healthy = True
        self.health.register_check("db", self._check_db)
        self.health.register_check("runner", self._check_runner)
        self.health.register_check("engine_tick", self._check_tick_age)
        self.health.register_check("control_api", self._check_control_api)
        self.control_app = self._build_control_app()

    # ---- control API -----------------------------------------------------------

    def _build_control_app(self) -> Any | None:
        """The control app, or None when no token is configured (fail closed)."""
        if not self.cfg.control_token:
            self.log.warning(
                "competition control API NOT served: orchestrator.control_token is "
                "empty. Competitions can then only be driven in-process — set a "
                "token to expose create/enroll/anchor/halt-clear/status/review/result"
            )
            return None
        from vidaio.competition.orchestrator.control import create_control_app

        # P2: self-signed enrollment needs the registered-hotkey guard; only a
        # real chain can back the registry (report mode enrolls in-process).
        hotkey_guard = None
        if self._chain_mode == "bittensor" and self.chain is not None:
            from vidaio.services.hotkey_auth import (
                HotkeyAuthConfig,
                HotkeyAuthGuard,
                RegisteredHotkeyRegistry,
            )

            hk_cfg = section(self.raw_config, "hotkey_auth", HotkeyAuthConfig)
            if hk_cfg.mode != "off":
                hotkey_guard = HotkeyAuthGuard(
                    RegisteredHotkeyRegistry(
                        self.chain,
                        ttl_seconds=hk_cfg.registry_ttl_seconds,
                        max_stale_seconds=hk_cfg.registry_max_stale_seconds,
                    ),
                    hk_cfg,
                )
        return create_control_app(
            self, token=self.cfg.control_token, hotkey_guard=hotkey_guard
        )

    def now(self) -> datetime:
        """The service clock (injectable for tests); always timezone-aware UTC."""
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("orchestrator clock must return timezone-aware datetimes")
        return value.astimezone(timezone.utc)

    # ---- health ----------------------------------------------------------------
    #
    # HealthServer runs its checks on its own HTTP thread (ThreadingHTTPServer), so
    # a check may NEVER touch `self.conn`: that connection belongs to the event
    # loop's thread, and sqlite3 either raises ProgrammingError across threads or —
    # worse, with check_same_thread=False — interleaves with a live transaction.
    # Either way the answer says nothing about whether the DB is usable, which is
    # the one thing this check exists to report.

    def _check_db(self) -> bool:
        """Open a SHORT-LIVED connection of our own and read real schema.

        Deliberately not a bare `SELECT 1`: that would pass against an empty or
        half-migrated file. Reading the competitions table proves the database the
        orchestrator actually works on is openable and migrated.
        """
        conn = sqlite3.connect(str(self.core.db_path), timeout=5)
        try:
            conn.execute("SELECT COUNT(*) FROM competitions").fetchone()
        finally:
            conn.close()
        return True

    def _check_runner(self) -> bool:
        available = getattr(self.runner, "available", None)
        return bool(available()) if callable(available) else True

    def _check_control_api(self) -> bool:
        """False once a configured control API has stopped serving.

        A live process with no reachable control plane is not healthy — nothing
        could create a competition or anchor a commitment — and used to look
        perfectly fine to every probe.
        """
        return self._control_api_healthy

    def _check_tick_age(self) -> bool:
        if self._last_step_monotonic is None:
            return True  # starting up; the loop hasn't ticked yet
        return (time.monotonic() - self._last_step_monotonic) < max(
            5 * self.cfg.tick_seconds, 30.0
        )

    # ---- service loop ----------------------------------------------------------

    def _create_control_server(self) -> uvicorn.Server:
        """Seam: tests substitute a server whose serve() fails (e.g. bind error)."""
        return uvicorn.Server(
            uvicorn.Config(
                self.control_app,
                host=self.cfg.control_host,
                port=self.cfg.control_port,
                log_config=None,
                access_log=False,
            )
        )

    async def run(self) -> None:
        """Tick loop + (when configured) the control API, in one event loop.

        Both touch the same SQLite connection from the same thread; no transaction
        in this package is held across an await, so they cannot interleave
        mid-transaction (see control.py CONCURRENCY). Health checks are the one
        exception and use their own connection (see the health section).

        The control serve task is MONITORED: a bind failure, or any
        other early exit, used to leave a live process reporting perfect health
        with nobody able to reach the control plane. It now flips the `control_api`
        health check and fails FATALLY — a non-zero exit, because exit 0 is the
        supervisor's "deliberate stop, do not restart".
        """
        control_task: asyncio.Task[Any] | None = None
        tasks = [asyncio.create_task(self._tick_loop(), name="orchestrator-tick")]
        server: uvicorn.Server | None = None
        if self.control_app is not None:
            server = self._create_control_server()
            control_task = asyncio.create_task(
                server.serve(), name="orchestrator-control"
            )
            tasks.append(control_task)
            self.log.info(
                "competition control API listening",
                extra=log_fields(
                    host=self.cfg.control_host, port=self.cfg.control_port
                ),
            )
        stop_task = asyncio.create_task(self.stopping.wait(), name="orchestrator-stop")
        try:
            await asyncio.wait({*tasks, stop_task}, return_when=asyncio.FIRST_COMPLETED)
            if (
                control_task is not None
                and control_task.done()
                and not self.stopping.is_set()
            ):
                self._control_api_healthy = False
                exc = control_task.exception() if not control_task.cancelled() else None
                self.fail_fatal(
                    "competition control API exited unexpectedly; the orchestrator"
                    " has no control plane"
                    f" (error={repr(exc) if exc is not None else None})"
                )
        finally:
            self.request_stop()
            if server is not None:
                server.should_exit = True
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(task, timeout=30)
            stop_task.cancel()

    async def _tick_loop(self) -> None:
        while not self.stopping.is_set():
            started = time.monotonic()
            try:
                await self.step(self.now())
            except Exception:
                self.m_step_errors.inc()
                self.log.exception("step failed; state is resumable — continuing")
            delay = max(0.05, self.cfg.tick_seconds - (time.monotonic() - started))
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.stopping.wait(), timeout=delay)

    async def step(self, now: datetime) -> None:
        """One orchestration pass: engine tick + the running competition's phase
        work. Deterministic and re-entrant — tests drive it with a fake clock."""
        if now.tzinfo is None:
            raise ValueError("step requires a timezone-aware `now`")
        # A due upscaling competition may become COMPLETED inside engine.tick.
        # Publish every pristine reference first; if storage release fails, halt and
        # leave the phase AWAITING_END_TIME so no earning evidence can exist without
        # keyless CPU recomputation bytes.
        if not self._release_due_upscaling_references(now):
            self._last_step_monotonic = time.monotonic()
            self.m_last_tick.set(time.time())
            return
        self.engine.tick(self.conn, now)
        self._last_step_monotonic = time.monotonic()
        self.m_last_tick.set(time.time())
        competition_id = repo.running_competition_id(self.conn)
        if competition_id is None:
            return
        if pers.is_halted(self.conn, competition_id):
            self.log.debug(
                "pipeline halted; skipping phase work",
                extra=log_fields(competition_id=competition_id),
            )
            return
        comp = repo.get_competition(self.conn, competition_id)
        assert comp is not None
        # Scorer identity, EARLY and EXPLICIT (services.protocol contract): the
        # manifest digest is anchored on chain before enrollment, so the
        # `scoring_version` it commits to must be the identity that will actually
        # measure. Checked on the first step that sees this competition and again
        # before SCORING — a disagreement halts here with a readable reason
        # instead of surfacing as a 409 in the middle of scoring.
        if comp.status is Phase.SCORING or competition_id not in self._identity_checked:
            if not self._check_scorer_identity(competition_id, now):
                return
        handler = {
            Phase.FINALIZING_SUBMISSIONS: self._stage_finalizing,
            Phase.VALIDATING: self._stage_validating,
            Phase.BUILDING: self._stage_building,
            Phase.EVALUATING: self._stage_evaluating,
            Phase.SCORING: self._stage_scoring,
        }.get(comp.status)
        if handler is None:
            return  # ENROLLING / AWAITING_END_TIME: nothing between ticks
        stage_started = time.monotonic()
        try:
            await handler(competition_id, now)
        finally:
            self.m_stage_seconds.labels(phase=comp.status.value).observe(
                time.monotonic() - stage_started
            )

    def _release_due_upscaling_references(self, now: datetime) -> bool:
        """Release due holdouts before the lifecycle can mark them COMPLETED.

        Returns ``False`` only when a due upscaling competition must stay closed.
        Releases are content-addressed and idempotent; a crash after object release
        but before the event/phase transition safely retries the same refs.
        """
        for comp in repo.list_competitions_in(self.conn, [Phase.AWAITING_END_TIME]):
            deadline = comp.end_time
            if comp.human_review_deadline is not None:
                deadline = max(deadline, comp.human_review_deadline)
            if now < deadline:
                continue
            manifest = repo.get_manifest(self.conn, comp.competition_id)
            if manifest.track != "upscaling":
                continue
            if pers.is_halted(self.conn, comp.competition_id):
                return False
            try:
                rows = repo.validate_evaluation_item_bindings(
                    self.conn, comp.competition_id
                )
                released: list[str] = []
                for row in rows:
                    ref = ArtifactRef(
                        digest=str(row["reference_sha256"]),
                        kind=ArtifactKind.REFERENCE_ORIGINAL,
                        byte_size=int(row["reference_bytes"]),
                        backend_key=backend_key(
                            ArtifactKind.REFERENCE_ORIGINAL,
                            str(row["reference_sha256"]),
                        ),
                    )
                    if not self.store.exists(ref):
                        raise FileNotFoundError(
                            f"sealed pristine reference {ref.digest} is absent"
                        )
                    self.store.release(ref)
                    if not self.store.is_released(ref):
                        raise RuntimeError(
                            f"pristine reference {ref.digest} was not publicly released"
                        )
                    released.append(ref.digest)
            except Exception as exc:
                self._halt(
                    comp.competition_id,
                    "upscaling completion blocked: pristine reference release/"
                    f"binding verification failed: {type(exc).__name__}: {exc}",
                    now,
                )
                return False

            already_recorded = self.conn.execute(
                "SELECT 1 FROM events WHERE competition_id = ? AND event_type = ? LIMIT 1",
                (
                    comp.competition_id,
                    pers.EVENT_COMPETITION_REFERENCES_RELEASED,
                ),
            ).fetchone()
            if already_recorded is None:
                with pers.txn(self.conn):
                    repo.record_event(
                        self.conn,
                        comp.competition_id,
                        pers.EVENT_COMPETITION_REFERENCES_RELEASED,
                        now,
                        payload={"reference_sha256": released},
                    )
        return True

    # ---- scorer identity -------------------------------------------------------

    def scorer_identity(self) -> str:
        """The live scoring worker's effective identity (services.protocol).

        Exposed so callers/tests can AUTHOR a manifest against the scorer that
        will actually run — `CompetitionManifest.scoring_version` is committed to
        by the anchored manifest digest, so it has to be the real identity, and
        guessing it is exactly what this method exists to stop.
        """
        probe = getattr(self.scoring_client, "scorer_identity", None)
        if not callable(probe):
            raise ScorerIdentityUnavailable(
                "the configured scoring client cannot report a scorer identity "
                "(no scorer_identity()); point the orchestrator at a real worker"
            )
        return str(probe())

    def _check_scorer_identity(self, competition_id: str, now: datetime) -> bool:
        """True when scoring may proceed; False when the pipeline was HALTED.

        Disagreement is an INFRA fault by the fault-classification table (a
        scorer-identity disagreement is systemic — the same worker scores
        everyone), so it halts rather than failing the competition. A worker that
        is merely UNREACHABLE is not a disagreement: we cannot prove one, so the
        check is deferred and the scoring stage's own bounded retry/halt path
        owns that failure.
        """
        manifest = repo.get_manifest(self.conn, competition_id)
        # Nobody but the orchestrator may answer under the orchestrator-zero
        # namespace: a worker (or a manifest) that claims it would make MEASURED
        # packets indistinguishable from orchestrator gate-failure records
        # (zero_packets.py). Refused as an INFRA halt, never silently accepted.
        try:
            assert_not_reserved(
                manifest.scoring_version,
                what="the manifest's committed scoring_version",
            )
        except ReservedScorerIdentity as exc:
            self._halt(competition_id, f"{exc} Operator action: {_RESERVED_FIX}", now)
            return False
        try:
            identity = self.scorer_identity()
        except ScorerRuntimeMismatch as exc:
            self._halt(
                competition_id,
                "canonical scorer runtime disagreement: the live worker returned "
                "a positive payout-runtime contract that differs from the "
                f"orchestrator's marker-qualified release image ({exc}). Scoring "
                "cannot start. Operator action: deploy the exact pinned CPU release "
                "image, then clear_halt.",
                now,
            )
            return False
        except Exception as exc:  # noqa: BLE001 - unreachable != disagreeing
            self.log.warning(
                "scorer identity not readable; deferring the manifest check",
                extra=log_fields(
                    competition_id=competition_id,
                    expected=manifest.scoring_version,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            return True
        try:
            assert_not_reserved(identity, what="the live scoring worker's identity")
        except ReservedScorerIdentity as exc:
            self._halt(competition_id, f"{exc} Operator action: {_RESERVED_FIX}", now)
            return False
        self._identity_checked.add(competition_id)
        if identity == manifest.scoring_version:
            return True
        self._halt(
            competition_id,
            "scorer identity disagreement: the manifest commits to scoring_version "
            f"{manifest.scoring_version!r} (the anchored manifest digest covers it) "
            f"but the scoring worker advertises {identity!r}. Scoring against a "
            "different scorer than the one this competition committed to would "
            "produce packets no audit can reconcile. Operator action: run the "
            "committed scorer (or re-run the competition under a manifest naming "
            "this one), then clear_halt.",
            now,
        )
        return False

    # ---- FINALIZING_SUBMISSIONS -> VALIDATING ----------------------------------

    async def _stage_finalizing(self, competition_id: str, now: datetime) -> None:
        """Back up every contender's pinned tree as a deterministic tarball in the
        audit store; the combined digest is the evidence-carrying backup_ref.

        The tarball is built by runners.safeio: REGULAR
        FILES ONLY, read through O_NOFOLLOW descriptors, symlinks/hardlinks/devices
        rejected outright rather than followed — a contender repo shipping
        ``creds -> /etc/passwd`` can no longer get host files archived into the
        audit store. A tree that needs filtering is REJECTED (never silently
        pruned).

        THE INVARIANT: every contender that can still win
        has an archived submission. A skipped backup used to be shrugged off while
        finalization went on to record the combined backup_ref and advance, so a
        contender whose checkout merely blipped could go on to compete — and win —
        with nothing archived for anyone to audit. Now the failure is CLASSIFIED:

        - CONTENDER fault (unsafe tree, oversize tree): that contender is REJECTED
          here, with the reason in the event log. It can no longer advance, so the
          invariant holds and everyone else proceeds untouched.
        - INFRA fault (checkout unreachable, audit store down, anything unknown):
          the phase does NOT advance and the pipeline HALTS. We refuse to certify
          a backup set we could not produce.

        Re-entry is idempotent: what is already archived is read from the
        append-only event log, so a restart archives only what is still missing and
        the combined ref is stable.
        """
        archived = pers.archived_submissions(self.conn, competition_id)
        for contender in repo.list_contenders(self.conn, competition_id):
            if contender.status == "REJECTED":
                continue  # cannot win: no archive required (and none exists)
            if contender.contender_id in archived:
                continue  # idempotent re-entry
            try:
                # A CONTENDER-fault tree (symlink, oversize) is a VERDICT, not a
                # transient failure: retrying it reaches the same answer, so it is
                # carried out as a result rather than spending the budget.
                tarball = _unwrap_fault(
                    await retry_async(
                        lambda c=contender: _fault_as_result(
                            asyncio.to_thread(self._archive_submission_bytes, c)
                        ),
                        attempts=self.cfg.submission_backup_attempts,
                        base_delay=self.cfg.retry_base_delay_seconds,
                        max_delay=self.cfg.retry_max_delay_seconds,
                    )
                )
            except (RetriesExhausted, Exception) as exc:
                if classify_failure(exc) is Fault.CONTENDER:
                    if contender.is_calibration:
                        self._halt(
                            competition_id,
                            "non-earning baseline submission backup was rejected: "
                            f"{_submission_reject_reason(exc)}. An earning "
                            "competition cannot continue without its exact baseline "
                            "baseline; fix/cancel this competition before resuming.",
                            now,
                        )
                        return
                    self._reject_contender(
                        competition_id,
                        contender.contender_id,
                        _submission_reject_reason(exc),
                        now,
                    )
                    self.m_contender_faults.labels(code=fault_code(exc)).inc()
                    continue
                self._halt(
                    competition_id,
                    f"submission backup failed for contender "
                    f"{contender.contender_id} ({contender.repo_url}) after "
                    f"{self.cfg.submission_backup_attempts} attempt(s): {exc}. "
                    "Finalization did NOT advance: every contender that can win "
                    "must have an archived submission, and certifying a partial "
                    "backup set would let an unarchived contender compete.",
                    now,
                )
                return
            # Submission source stays sealed while the competition is unresolved.
            # Only the independently verified CROWN promotion path releases the
            # winning archive; non-winners remain private indefinitely.
            ref = self.store.put(tarball, ArtifactKind.SUBMISSION_ARCHIVE)
            with pers.txn(self.conn):
                pers.record_submission_archived(
                    self.conn,
                    competition_id,
                    contender.contender_id,
                    ref.digest,
                    len(tarball),
                    now,
                )
            archived[contender.contender_id] = ref.digest
        # Only contenders that can still win are certified; a REJECTED one has no
        # archive and cannot compete, so the set is complete by construction.
        eligible = {
            c.contender_id
            for c in repo.list_contenders(self.conn, competition_id)
            if c.status != "REJECTED"
        }
        missing = sorted(eligible - archived.keys())
        if missing:  # defensive: the loop above halts or rejects, never leaves gaps
            self._halt(
                competition_id,
                f"refusing to certify the submission backup set: contenders "
                f"{missing} can still compete but have no archived submission",
                now,
            )
            return
        digests = [f"{cid_}:{archived[cid_]}" for cid_ in sorted(eligible)]
        combined = hashlib.sha256(
            "\n".join(sorted(digests)).encode("utf-8")
        ).hexdigest()
        backup_ref = f"audit://submissions/sha256:{combined}"
        self.engine.mark_submissions_backed_up(
            self.conn, competition_id, backup_ref, now
        )

    def _archive_submission_bytes(self, contender: repo.ContenderRecord) -> bytes:
        """Checkout + deterministic tarball for one contender (blocking)."""
        checkout = checkout_pinned(
            self.repo_provider,
            contender.repo_url,
            contender.commit_sha,
            contender.tree_sha,
        )
        try:
            return safeio.deterministic_tarball(
                checkout, max_bytes=self.cfg.submission_backup_max_bytes
            )
        finally:
            release_checkout(self.repo_provider, checkout)

    def _reject_contender(
        self, competition_id: str, contender_id: int, reason: str, now: datetime
    ) -> None:
        """Terminally REJECT one contender with a reason (never the competition).

        Shared by finalization and validation so a rejection looks the same in the
        event log wherever the submission was found unusable.
        """
        with pers.txn(self.conn):
            repo.set_contender_status(self.conn, contender_id, "REJECTED", now)
            repo.record_event(
                self.conn,
                competition_id,
                "contender_validated",
                now,
                payload={
                    "contender_id": contender_id,
                    "status": "REJECTED",
                    "reason": reason,
                },
            )
        self.log.warning(
            "contender REJECTED; the competition continues",
            extra=log_fields(
                competition_id=competition_id, contender_id=contender_id, reason=reason
            ),
        )

    # ---- VALIDATING -> BUILDING -------------------------------------------------

    async def _stage_validating(self, competition_id: str, now: datetime) -> None:
        """Local-first static validation: the pinned checkout must materialize,
        contain a Dockerfile, and hold ONLY regular files/directories.

        The tree-safety check is part of validation, not
        an afterthought: a submission containing a symlink, fifo or device node is
        REJECTED with a reason. That is a contender-level outcome — the other
        contenders are unaffected.

        review #14 (round 2): an ARBITRARY checkout failure is no longer a rejection.
        `except Exception -> REJECTED` meant that our own outage — an unreachable
        git host, a full disk, a bug in the provider — permanently eliminated a
        contender from a competition it had legitimately entered, silently and
        with a plausible-looking reason string. Every failure now goes through
        `classify_failure`: CONTENDER faults reject that contender, everything else
        (unknown included) HALTS the pipeline so an operator can fix it and the
        contender still gets its run.
        """
        for contender in repo.list_contenders(self.conn, competition_id):
            if contender.status != "ENROLLED":
                continue  # idempotent re-entry
            ok, reason = True, ""
            checkout: Path | None = None
            try:
                try:
                    checkout = checkout_pinned(
                        self.repo_provider,
                        contender.repo_url,
                        contender.commit_sha,
                        contender.tree_sha,
                    )
                    safeio.assert_safe_tree(checkout)
                    dockerfile = checkout / "Dockerfile"
                    try:
                        safeio.lstat_regular(dockerfile, what="Dockerfile")
                    except FileNotFoundError:
                        ok, reason = False, "no Dockerfile in pinned checkout"
                finally:
                    if checkout is not None:
                        release_checkout(self.repo_provider, checkout)
            except Exception as exc:
                if classify_failure(exc) is not Fault.CONTENDER:
                    self._halt(
                        competition_id,
                        f"validation could not inspect contender "
                        f"{contender.contender_id} ({contender.repo_url}): {exc}. "
                        "This is OUR failure, not a bad submission, so the "
                        "contender is NOT rejected — fix the blocker and "
                        "clear_halt to resume validation.",
                        now,
                    )
                    return
                ok, reason = False, _submission_reject_reason(exc)
            if not ok:
                if contender.is_calibration:
                    self._halt(
                        competition_id,
                        "non-earning baseline failed static validation: "
                        f"{reason}. An earning competition cannot advance without "
                        "its exact calibration baseline.",
                        now,
                    )
                    return
                self._reject_contender(
                    competition_id, contender.contender_id, reason, now
                )
                continue
            with pers.txn(self.conn):
                repo.set_contender_status(
                    self.conn, contender.contender_id, "ACCEPTED", now
                )
                repo.record_event(
                    self.conn,
                    competition_id,
                    "contender_validated",
                    now,
                    payload={
                        "contender_id": contender.contender_id,
                        "status": "ACCEPTED",
                    },
                )
        self.engine.mark_validation_complete(self.conn, competition_id, now)

    # ---- BUILDING -> EVALUATING -------------------------------------------------

    def _modal_runtime_fence(self) -> tuple[str, str, Any] | None:
        """Return the fresh Modal runtime identity/live-handle check, if present.

        Docker/report runners deliberately have no such surface: Docker image
        digests are resolvable local daemon objects.  Modal image objects are
        Python handles scoped to one fresh SDK runtime, so a persisted digest
        alone must never be treated as executable after process restart.
        """
        session_id = getattr(self.runner, "runtime_session_id", None)
        runtime_label = getattr(self.runner, "runtime_label", None)
        has_live_image = getattr(self.runner, "has_live_image", None)
        if session_id is None and runtime_label is None and has_live_image is None:
            return None
        if (
            not isinstance(session_id, str)
            or len(session_id) != 64
            or any(char not in "0123456789abcdef" for char in session_id)
            or not isinstance(runtime_label, str)
            or not runtime_label.startswith("vidaio-next-")
            or not callable(has_live_image)
        ):
            raise ValueError(
                "Modal runner restart-fence identity is incomplete or malformed"
            )
        return session_id, runtime_label, has_live_image

    def _modal_image_controls(self) -> tuple[Any, Any] | None:
        """Return exact owned-Image persistence/restore controls, when Modal.

        Report and Docker runners expose neither method.  A Modal runner must
        expose both: persisting an opaque id without a restore seam (or vice
        versa) would make restart behavior either unusable or capable of
        attaching to an unproven object.
        """
        image_object_id = getattr(self.runner, "image_object_id", None)
        restore_image = getattr(self.runner, "restore_image", None)
        if image_object_id is None and restore_image is None:
            return None
        if not callable(image_object_id) or not callable(restore_image):
            raise ValueError(
                "Modal runner owned-image persistence/restore controls are incomplete"
            )
        return image_object_id, restore_image

    @staticmethod
    def _require_modal_binding_matches(
        binding: Mapping[str, object],
        spec: ContenderSpec,
        image_digest: str,
        *,
        is_calibration: bool,
    ) -> str:
        expected_digest = logical_build_identity(
            repo_url=spec.repo_url,
            commit_sha=spec.commit_sha,
            tree_sha=spec.tree_sha,
        )
        if image_digest != expected_digest:
            raise ValueError(
                "persisted Modal logical build digest does not match the pinned "
                "contender source"
            )
        expected: dict[str, object] = {
            "repo_url": spec.repo_url,
            "commit_sha": spec.commit_sha,
            "tree_sha": spec.tree_sha,
            "build_identity_scheme": BUILD_IDENTITY_SCHEME,
            "image_digest": image_digest,
            "provider": "modal",
            "is_calibration": is_calibration,
        }
        for field, value in expected.items():
            if binding.get(field) != value:
                raise ValueError(
                    f"persisted Modal image binding {field} does not match the "
                    "pinned contender"
                )
        object_id = binding.get("image_object_id")
        if (
            not isinstance(object_id, str)
            or not object_id.startswith("im-")
            or len(object_id) <= 3
            or len(object_id) > 131
            or any(not (char.isalnum() or char in "_-") for char in object_id[3:])
        ):
            raise ValueError("persisted Modal image object id is malformed")
        return object_id

    async def _restore_bound_modal_image(
        self,
        competition_id: str,
        spec: ContenderSpec,
        image_digest: str,
        *,
        is_calibration: bool,
    ) -> str:
        controls = self._modal_image_controls()
        if controls is None:
            raise ValueError("runner has no owned Modal image restore seam")
        _image_object_id, restore_image = controls
        binding = pers.latest_modal_image_binding(
            self.conn,
            competition_id,
            image_digest,
            is_calibration=is_calibration,
        )
        if binding is None:
            raise ValueError(
                "no append-only competition-owned Modal image binding exists for "
                f"{image_digest}"
            )
        object_id = self._require_modal_binding_matches(
            binding,
            spec,
            image_digest,
            is_calibration=is_calibration,
        )
        restored = await retry_async(
            lambda: asyncio.to_thread(restore_image, spec, image_digest, object_id),
            attempts=self.cfg.build_retry_attempts,
            base_delay=self.cfg.retry_base_delay_seconds,
            max_delay=self.cfg.retry_max_delay_seconds,
        )
        if restored != image_digest:
            raise ValueError("Modal restore returned a different image evidence digest")
        return restored

    def _record_modal_image_binding(
        self,
        competition_id: str,
        spec: ContenderSpec,
        image_digest: str,
        *,
        is_calibration: bool,
        now: datetime,
    ) -> None:
        controls = self._modal_image_controls()
        if controls is None:
            return
        image_object_id, _restore_image = controls
        object_id = image_object_id(image_digest)
        if (
            not isinstance(object_id, str)
            or not object_id.startswith("im-")
            or len(object_id) <= 3
            or len(object_id) > 131
            or any(not (char.isalnum() or char in "_-") for char in object_id[3:])
        ):
            raise ValueError(
                "live Modal image has no valid provider object id to persist for restart"
            )
        fence = self._modal_runtime_fence()
        if fence is None:
            raise ValueError("Modal image controls exist without a runtime fence")
        session_id, runtime_label, _has_live_image = fence
        current = pers.latest_modal_image_binding(
            self.conn,
            competition_id,
            image_digest,
            is_calibration=is_calibration,
        )
        if current is not None:
            current_id = self._require_modal_binding_matches(
                current,
                spec,
                image_digest,
                is_calibration=is_calibration,
            )
            current_session = current.get("runtime_session_id")
            if current_id != object_id and current_session == session_id:
                raise ValueError(
                    "one Modal runtime produced multiple provider object ids for "
                    "the same logical build identity"
                )
            if current_id == object_id:
                return
            # A later wholly fresh runtime is allowed to rebuild identical pinned
            # source and therefore mint another provider id. Preserve both rows;
            # restart recovery always restores the newest exact owned binding.
        pers.record_modal_image_binding(
            self.conn,
            competition_id,
            contender_id=spec.contender_id,
            is_calibration=is_calibration,
            repo_url=spec.repo_url,
            commit_sha=spec.commit_sha,
            tree_sha=spec.tree_sha,
            image_digest=image_digest,
            image_object_id=object_id,
            runtime_session_id=session_id,
            runtime_label=runtime_label,
            now=now,
        )

    async def _ensure_fresh_modal_runtime(
        self, competition_id: str, phase: Phase, now: datetime
    ) -> bool:
        """Bind or recover the runtime-scoped Modal image handles fail-closed.

        A new process never discovers or lists an old resource.  It rehydrates
        only exact immutable Image ids that this competition durably bound to each
        pinned repo/commit/tree, then reprobes them. In EVALUATING, all batches
        are atomically reset behind an append-only event fence, so the effective
        matrix is rerun wholly on the new runtime; prior output events remain
        auditable but are never mixed into scoring. Any uncertainty halts before
        a new GPU batch executes.
        """
        try:
            fence = self._modal_runtime_fence()
        except Exception as exc:
            self._halt(
                competition_id,
                f"fresh Modal runtime restart fence is unusable: {exc}",
                now,
            )
            return False
        if fence is None:
            return True
        session_id, runtime_label, has_live_image = fence
        try:
            binding = pers.latest_modal_runtime_binding(self.conn, competition_id)
            previous_session: str | None = None
            if binding is not None:
                raw_previous = binding.get("runtime_session_id")
                if (
                    not isinstance(raw_previous, str)
                    or len(raw_previous) != 64
                    or any(char not in "0123456789abcdef" for char in raw_previous)
                ):
                    raise ValueError(
                        "persisted modal_runtime_bound session id is malformed"
                    )
                previous_session = raw_previous
            built = [
                contender
                for contender in repo.list_contenders(self.conn, competition_id)
                if contender.status == "BUILT"
            ]
            if self.tokenomics.competition_emissions_enabled and phase in {
                Phase.BUILDING,
                Phase.EVALUATING,
            }:
                manifest = repo.get_manifest(self.conn, competition_id)
                commitment = self._load_earning_commitment(competition_id, manifest)
                built_baselines = [
                    contender for contender in built if contender.is_calibration
                ]
                if phase is Phase.EVALUATING and len(built_baselines) != 1:
                    raise ValueError(
                        "expected exactly one BUILT baseline before evaluation, found "
                        f"{len(built_baselines)}"
                    )
                if any(
                    contender.image_digest != commitment.baseline_image_digest
                    for contender in built_baselines
                ):
                    raise ValueError(
                        "persisted baseline image differs from the pre-enrollment "
                        "earning commitment"
                    )
            missing_handles = [
                contender
                for contender in built
                if contender.image_digest is None
                or not has_live_image(contender.image_digest)
            ]
        except Exception as exc:
            self._halt(
                competition_id,
                "fresh Modal runtime binding/live-image evidence could not be "
                f"validated: {exc}",
                now,
            )
            return False

        if previous_session == session_id and not missing_handles:
            return True
        if not built:
            # Must precede the first image build.  If the process dies immediately
            # afterward, the next fresh runner observes a different session and
            # repeats this harmless pre-build bind.
            with pers.txn(self.conn):
                pers.record_modal_runtime_binding(
                    self.conn,
                    competition_id,
                    runtime_session_id=session_id,
                    runtime_label=runtime_label,
                    phase=phase.value,
                    now=now,
                    reason="initial_prebuild_binding"
                    if binding is None
                    else "fresh_runtime_before_first_build",
                    previous_runtime_session_id=previous_session,
                )
            return True

        try:
            modal_image_controls = self._modal_image_controls()
        except Exception as exc:
            self._halt(
                competition_id,
                f"fresh Modal owned-image controls are unusable: {exc}",
                now,
            )
            return False

        rebound_images: list[tuple[int, str]] = []
        probe_records: list[tuple[int, str, str]] = []
        for contender in built:
            if contender.image_digest is None:
                self._halt(
                    competition_id,
                    "fresh Modal restart recovery found a BUILT contender with no "
                    f"persisted image digest (contender {contender.contender_id})",
                    now,
                )
                return False
            spec = ContenderSpec(
                contender_id=contender.contender_id,
                repo_url=contender.repo_url,
                commit_sha=contender.commit_sha,
                tree_sha=contender.tree_sha,
            )
            try:
                if has_live_image(contender.image_digest):
                    rebuilt_digest = contender.image_digest
                elif modal_image_controls is not None:
                    rebuilt_digest = await self._restore_bound_modal_image(
                        competition_id,
                        spec,
                        contender.image_digest,
                        is_calibration=contender.is_calibration,
                    )
                else:
                    # Backward-compatible test/runtime seam. Production Modal
                    # exposes durable owned-image restoration and never relies on
                    # force-build returning the same opaque provider id.
                    rebuilt_digest = await retry_async(
                        lambda spec=spec: asyncio.to_thread(self.runner.build, spec),
                        attempts=self.cfg.build_retry_attempts,
                        base_delay=self.cfg.retry_base_delay_seconds,
                        max_delay=self.cfg.retry_max_delay_seconds,
                    )
            except Exception as exc:
                self._halt(
                    competition_id,
                    "fresh Modal restart recovery could not restore persisted "
                    f"contender {contender.contender_id}: {exc}. No batch was "
                    "executed on the replacement runtime.",
                    now,
                )
                return False
            if rebuilt_digest != contender.image_digest:
                self._halt(
                    competition_id,
                    "fresh Modal restart recovery restored a different image for "
                    f"contender {contender.contender_id}: persisted="
                    f"{contender.image_digest}, restored={rebuilt_digest}. No "
                    "replacement-runtime batch was executed.",
                    now,
                )
                return False
            try:
                report = await asyncio.to_thread(
                    self.runner.isolation_probe, rebuilt_digest
                )
            except Exception as exc:
                self._halt(
                    competition_id,
                    "fresh Modal restart recovery could not reprobe persisted "
                    f"contender {contender.contender_id}: {exc}. No batch was "
                    "executed on the replacement runtime.",
                    now,
                )
                return False
            probe_json = _probe_json(report)
            if not report.passed:
                with pers.txn(self.conn):
                    pers.record_sandbox_probe(
                        self.conn,
                        competition_id,
                        contender.contender_id,
                        rebuilt_digest,
                        probe_json,
                        passed=False,
                        now=now,
                    )
                self._halt(
                    competition_id,
                    "fresh Modal restart recovery reprobe failed for persisted "
                    f"contender {contender.contender_id}. No batch was executed "
                    "on the replacement runtime.",
                    now,
                )
                return False
            rebound_images.append((contender.contender_id, rebuilt_digest))
            probe_records.append((contender.contender_id, rebuilt_digest, probe_json))

        if phase is Phase.EVALUATING:
            pers.reset_evaluation_for_modal_runtime(
                self.conn,
                competition_id,
                previous_runtime_session_id=previous_session,
                runtime_session_id=session_id,
                runtime_label=runtime_label,
                phase=phase.value,
                rebound_images=rebound_images,
                probe_records=probe_records,
                now=now,
            )
        else:
            with pers.txn(self.conn):
                for contender_id, image_digest, probe_json in probe_records:
                    pers.record_sandbox_probe(
                        self.conn,
                        competition_id,
                        contender_id,
                        image_digest,
                        probe_json,
                        passed=True,
                        now=now,
                    )
                pers.record_modal_runtime_binding(
                    self.conn,
                    competition_id,
                    runtime_session_id=session_id,
                    runtime_label=runtime_label,
                    phase=phase.value,
                    now=now,
                    reason="fresh_runtime_owned_images_restored_and_reprobed",
                    previous_runtime_session_id=previous_session,
                    rebound_images=rebound_images,
                )
        self.log.warning(
            "fresh Modal runtime recovered from exact deployment-owned image bindings",
            extra=log_fields(
                competition_id=competition_id,
                phase=phase.value,
                previous_runtime_session_id=previous_session,
                runtime_session_id=session_id,
                rebound_images=len(rebound_images),
                evaluation_reset=phase is Phase.EVALUATING,
            ),
        )
        return True

    async def _stage_building(self, competition_id: str, now: datetime) -> None:
        """Build each accepted contender's image and attest its sandbox boundary.

        review #14 (round 2) — two bypasses removed:
        - an ARBITRARY build failure no longer becomes BUILD_FAILED. Only a
          ContenderBuildError (docker build rejected THIS Dockerfile/context, or
          the build blew its budget) marks the contender; a BuildError — our docker
          CLI is unusable, our daemon lost the image it just built — is INFRA and
          halts, because eliminating a contender for our outage is unrecoverable
          once the competition moves on.
        - an isolation probe that could NOT RUN no longer disqualifies anyone. The
          runner raises SandboxProbeUnavailableError instead of returning an
          all-False report, so "we could not attest the boundary" (INFRA, halt) is
          distinguishable from "we attested it and it was violated" (contender
          fault, disqualified with the evidence recorded).
        """
        manifest = repo.get_manifest(self.conn, competition_id)
        if self._halt_on_manifest_gpu_problem(competition_id, manifest, now):
            return
        if not await self._ensure_fresh_modal_runtime(
            competition_id, Phase.BUILDING, now
        ):
            return
        earning_commitment: CompetitionCommitment | None = None
        if self.tokenomics.competition_emissions_enabled:
            try:
                earning_commitment = self._load_earning_commitment(
                    competition_id, manifest
                )
            except Exception as exc:
                self._halt(
                    competition_id,
                    "earning competition commitment could not be proven against "
                    f"the persisted manifest and active reward policy: {exc}. "
                    "No contender image was executed; restore the exact anchored "
                    "artifacts/configuration before clearing the halt.",
                    now,
                )
                return
        contenders = repo.list_contenders(self.conn, competition_id)
        if earning_commitment is not None:
            # Prove the non-earning baseline first.  A mismatched baseline makes every
            # contender margin economically meaningless, so no contender should run
            # before the committed baseline identity is known to match.
            contenders.sort(
                key=lambda contender: (
                    0 if contender.is_calibration else 1,
                    contender.contender_id,
                )
            )
        for contender in contenders:
            if contender.status != "ACCEPTED":
                continue  # BUILT/BUILD_FAILED rows are done — idempotent re-entry
            spec = ContenderSpec(
                contender_id=contender.contender_id,
                repo_url=contender.repo_url,
                commit_sha=contender.commit_sha,
                tree_sha=contender.tree_sha,
            )
            try:
                if contender.is_calibration and earning_commitment is not None:
                    modal_controls = self._modal_image_controls()
                    modal_fence = self._modal_runtime_fence()
                    if modal_controls is not None and modal_fence is not None:
                        _session, _label, has_live_image = modal_fence
                        if not has_live_image(earning_commitment.baseline_image_digest):
                            await self._restore_bound_modal_image(
                                competition_id,
                                spec,
                                earning_commitment.baseline_image_digest,
                                is_calibration=True,
                            )
                image_digest = await retry_async(
                    lambda spec=spec: asyncio.to_thread(self.runner.build, spec),
                    attempts=self.cfg.build_retry_attempts,
                    base_delay=self.cfg.retry_base_delay_seconds,
                    max_delay=self.cfg.retry_max_delay_seconds,
                )
            except (RetriesExhausted, Exception) as exc:
                if classify_failure(exc) is not Fault.CONTENDER:
                    self._halt(
                        competition_id,
                        f"building contender {contender.contender_id} failed for an "
                        f"INFRASTRUCTURE reason after "
                        f"{self.cfg.build_retry_attempts} attempt(s): {exc}. The "
                        "contender is NOT marked BUILD_FAILED — its submission was "
                        "never judged. Fix the blocker and clear_halt.",
                        now,
                    )
                    return
                if contender.is_calibration:
                    self._halt(
                        competition_id,
                        "non-earning baseline image failed to build: "
                        f"{type(exc).__name__}: {exc}. The baseline is never an "
                        "ordinary disqualified contender; without it the economic "
                        "baseline is undefined, so the competition remains halted.",
                        now,
                    )
                    return
                self._mark_build_failed(
                    competition_id,
                    contender.contender_id,
                    f"build failed ({fault_code(exc)}): {exc}",
                    now,
                )
                self.m_builds.labels(result="failed").inc()
                self.m_contender_faults.labels(code=fault_code(exc)).inc()
                continue
            if (
                contender.is_calibration
                and earning_commitment is not None
                and image_digest != earning_commitment.baseline_image_digest
            ):
                self._halt(
                    competition_id,
                    "built baseline image digest does not match the pre-enrollment "
                    f"commitment: built={image_digest}, "
                    f"anchored={earning_commitment.baseline_image_digest}. The baseline "
                    "baseline was not executed and no earning score may be derived.",
                    now,
                )
                return
            try:
                report = await asyncio.to_thread(
                    self.runner.isolation_probe, image_digest
                )
            except Exception as exc:
                if classify_failure(exc) is not Fault.CONTENDER:
                    self._halt(
                        competition_id,
                        f"the isolation probe for contender "
                        f"{contender.contender_id} could not be RUN: {exc}. Nothing "
                        "was attested, so the contender is NOT disqualified — an "
                        "unattestable boundary is our failure, and a sandbox we "
                        "cannot attest is never used. Fix the blocker and "
                        "clear_halt.",
                        now,
                    )
                    return
                if contender.is_calibration:
                    self._halt(
                        competition_id,
                        "non-earning baseline isolation probe aborted: "
                        f"{type(exc).__name__}: {exc}. The baseline is never an "
                        "ordinary probe disqualification; the competition remains "
                        "halted without a baseline.",
                        now,
                    )
                    return
                # A probe that raised a CONTENDER fault (e.g. its image flooded the
                # log cap) is the contender's own doing: disqualify with the reason.
                self._mark_build_failed(
                    competition_id,
                    contender.contender_id,
                    f"isolation probe aborted by contender fault "
                    f"({fault_code(exc)}): {exc}",
                    now,
                )
                self.m_builds.labels(result="probe_dq").inc()
                self.m_contender_faults.labels(code=fault_code(exc)).inc()
                continue
            probe_json = _probe_json(report)
            if not report.passed:
                if contender.is_calibration:
                    with pers.txn(self.conn):
                        pers.record_sandbox_probe(
                            self.conn,
                            competition_id,
                            contender.contender_id,
                            image_digest,
                            probe_json,
                            passed=False,
                            now=now,
                        )
                        repo.record_event(
                            self.conn,
                            competition_id,
                            "baseline_isolation_probe_failed",
                            now,
                            payload={
                                "contender_id": contender.contender_id,
                                "image_digest": image_digest,
                                "probe": probe_json,
                            },
                        )
                    self._halt(
                        competition_id,
                        "non-earning baseline failed its isolation probe. The baseline is "
                        "never an ordinary disqualified contender; the competition "
                        "remains halted without a trustworthy baseline.",
                        now,
                    )
                    return
                # Probe failure DISQUALIFIES the build — recorded, never a crash
                # (spec §05: a sandbox that fails its probe is never used).
                with pers.txn(self.conn):
                    pers.record_sandbox_probe(
                        self.conn,
                        competition_id,
                        contender.contender_id,
                        image_digest,
                        probe_json,
                        passed=False,
                        now=now,
                    )
                    repo.set_contender_status(
                        self.conn, contender.contender_id, "BUILD_FAILED", now
                    )
                    repo.record_event(
                        self.conn,
                        competition_id,
                        "isolation_probe_failed",
                        now,
                        payload={
                            "contender_id": contender.contender_id,
                            "image_digest": image_digest,
                            "probe": probe_json,
                        },
                    )
                self.log.warning(
                    "isolation probe failed; contender disqualified",
                    extra=log_fields(
                        competition_id=competition_id,
                        contender_id=contender.contender_id,
                        image_digest=image_digest,
                        probe=probe_json,
                    ),
                )
                self.m_builds.labels(result="probe_dq").inc()
                continue
            with pers.txn(self.conn):
                self._record_modal_image_binding(
                    competition_id,
                    spec,
                    image_digest,
                    is_calibration=contender.is_calibration,
                    now=now,
                )
                pers.record_sandbox_probe(
                    self.conn,
                    competition_id,
                    contender.contender_id,
                    image_digest,
                    probe_json,
                    passed=True,
                    now=now,
                )
                repo.set_contender_image_digest(
                    self.conn, contender.contender_id, image_digest, now
                )
                repo.record_event(
                    self.conn,
                    competition_id,
                    "contender_built",
                    now,
                    payload={
                        "contender_id": contender.contender_id,
                        "image_digest": image_digest,
                    },
                )
            self.m_builds.labels(result="ok").inc()
        n_built = sum(
            1
            for c in repo.list_contenders(self.conn, competition_id)
            if c.status == "BUILT"
        )
        self.engine.mark_builds_complete(self.conn, competition_id, n_built, now)

    def _mark_build_failed(
        self, competition_id: str, contender_id: int, reason: str, now: datetime
    ) -> None:
        with pers.txn(self.conn):
            repo.set_contender_status(self.conn, contender_id, "BUILD_FAILED", now)
            repo.record_event(
                self.conn,
                competition_id,
                "contender_build_failed",
                now,
                payload={"contender_id": contender_id, "reason": reason[:500]},
            )
        self.log.warning(
            "contender build failed",
            extra=log_fields(
                competition_id=competition_id, contender_id=contender_id, reason=reason
            ),
        )

    # ---- EVALUATING -> SCORING --------------------------------------------------

    async def _stage_evaluating(self, competition_id: str, now: datetime) -> None:
        manifest = repo.get_manifest(self.conn, competition_id)
        if self._halt_on_manifest_gpu_problem(competition_id, manifest, now):
            return
        if not await self._ensure_fresh_modal_runtime(
            competition_id, Phase.EVALUATING, now
        ):
            return
        try:
            # The factor exposed to untrusted code must be the value bound by the
            # pre-enrollment manifest, not merely a mutable database column.
            items = repo.validate_evaluation_item_bindings(self.conn, competition_id)
        except repo.EvaluationItemBindingError as exc:
            self._halt(
                competition_id,
                "evaluation blocked: committed item binding verification failed: "
                f"{exc}",
                now,
            )
            return
        if not items:
            self.log.info(
                "evaluation waiting: no evaluation items seeded yet",
                extra=log_fields(competition_id=competition_id),
            )
            return
        batch_size = manifest.evaluation_batch_size.max
        built = [
            c
            for c in repo.list_contenders(self.conn, competition_id)
            if c.status == "BUILT"
        ]
        with pers.txn(self.conn):
            for contender in built:
                pers.ensure_batches(
                    self.conn,
                    competition_id,
                    contender.contender_id,
                    n_items=len(items),
                    batch_size=batch_size,
                    now=now,
                )
        by_id = {c.contender_id: c for c in built}
        for batch in pers.runnable_batches(self.conn, competition_id):
            contender = by_id.get(batch["contender_id"])
            if contender is None or contender.image_digest is None:
                continue  # defensive: batch for a contender no longer BUILT
            batch_items = pers.batch_items_for(items, batch["batch_index"], batch_size)
            pers.set_batch_status(
                self.conn, batch["batch_id"], "RUNNING", now, started=True
            )
            try:
                outputs = _unwrap_fault(
                    await retry_async(
                        lambda c=contender, bi=batch_items, idx=batch["batch_index"]: (
                            _fault_as_result(
                                asyncio.to_thread(
                                    self.runner.run_batch, c.image_digest, bi, idx
                                )
                            )
                        ),
                        attempts=self.cfg.batch_retry_attempts,
                        base_delay=self.cfg.retry_base_delay_seconds,
                        max_delay=self.cfg.retry_max_delay_seconds,
                    )
                )
            except (RetriesExhausted, Exception) as exc:
                if classify_failure(exc) is Fault.CONTENDER:
                    # The SUBMISSION failed (exit != 0, timeout, unsafe/oversize
                    # output): fail THIS batch terminally and keep going. One
                    # untrusted contender never halts the competition (#14).
                    code = fault_code(exc)
                    pers.fail_batch_contender_fault(
                        self.conn,
                        competition_id,
                        batch["batch_id"],
                        batch["contender_id"],
                        code,
                        str(exc),
                        now,
                    )
                    self.m_batches.labels(result="contender_failed").inc()
                    self.m_contender_faults.labels(code=code).inc()
                    self.log.warning(
                        "batch failed by CONTENDER fault; competition continues",
                        extra=log_fields(
                            competition_id=competition_id,
                            contender_id=batch["contender_id"],
                            batch_id=batch["batch_id"],
                            code=code,
                            reason=str(exc)[:500],
                        ),
                    )
                    continue
                requeues = pers.requeue_count(
                    self.conn, competition_id, batch["batch_id"]
                )
                if requeues >= self.cfg.max_batch_requeues:
                    pers.set_batch_status(
                        self.conn,
                        batch["batch_id"],
                        "REQUEUED",
                        now,
                        failure_code=str(exc)[:200],
                    )
                    self._halt(
                        competition_id,
                        f"batch {batch['batch_id']} (contender "
                        f"{batch['contender_id']}) infra-failed after {requeues} "
                        f"requeue(s) and {self.cfg.batch_retry_attempts} in-step "
                        f"attempt(s): {exc}",
                        now,
                    )
                    return
                pers.requeue_batch(
                    self.conn, competition_id, batch["batch_id"], str(exc), now
                )
                self.m_batches.labels(result="requeued").inc()
                continue
            pers.complete_batch(
                self.conn,
                competition_id,
                batch["batch_id"],
                batch["contender_id"],
                list(outputs),
                now,
            )
            self.m_batches.labels(result="ok").inc()
        expected = len(built) * pers.batch_count(len(items), batch_size)
        total = self.conn.execute(
            "SELECT COUNT(*) AS n FROM batches WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()["n"]
        if (
            total >= expected
            and repo.count_non_terminal_batches(self.conn, competition_id) == 0
        ):
            self.engine.mark_evaluation_complete(self.conn, competition_id, now)

    # ---- SCORING -> AWAITING_END_TIME -------------------------------------------

    async def _stage_scoring(self, competition_id: str, now: datetime) -> None:
        """Score every (contender, item, output) via the trusted scoring client,
        persist the verbatim packet, archive the artifacts, and audit-link the
        row — packet persistence and bundle linkage commit ATOMICALLY, so a
        crash can never leave a score the completion gate would count unlinked.

        review service-review #14: an item with NO output (or a zero-byte one) is
        never handed to the scorer — sending an empty file to real ffmpeg produced
        a 502 that halted the whole competition. Such items are zero-scored
        locally with a machine-readable reason code (see _zero_score_item), and a
        rejection of the CONTENDER'S OWN OUTPUT zeroes that item the same way.

        Round 2 narrows that second case: only a scorer rejection whose typed error
        names the `output` field is contender-attributable. A 422 about the
        reference/miner input we named, an invalid_param, or any 422 we cannot type
        is OUR failure and halts — zeroing a contender for our own bad request
        would silently corrupt the result (see orchestrator.failures)."""
        manifest = repo.get_manifest(self.conn, competition_id)
        manifest_ref = self.store.put(
            manifest.canonical_json().encode("utf-8"), ArtifactKind.MANIFEST
        )
        items = pers.list_items(self.conn, competition_id)
        batch_size = manifest.evaluation_batch_size.max
        targets = [
            c
            for c in repo.list_contenders(self.conn, competition_id)
            if c.status == "BUILT"
        ]
        for contender in targets:
            outputs = pers.outputs_for_contender(
                self.conn, competition_id, contender.contender_id
            )
            for item_row in items:
                item_id = item_row["item_id"]
                if pers.has_score_row(self.conn, contender.contender_id, item_id):
                    continue  # idempotent re-entry: already recorded
                item = pers.batch_items_for([item_row], 0, 1)[0]
                digest, nbytes = outputs.get(item_id, (EMPTY_SHA256, 0))
                if nbytes <= 0 or digest == EMPTY_SHA256:
                    # NO OUTPUT: never hand an empty file to ffmpeg (#14). Zero it
                    # here, with a reason code, and keep going.
                    self._zero_score_item(
                        competition_id,
                        contender,
                        item_row,
                        manifest,
                        manifest_ref,
                        batch_size,
                        code=ReasonCode.METRIC_MISSING,
                        detail=(
                            "no output produced by the solution for this item "
                            "(absent or zero-byte) — scored zero without invoking "
                            "the scorer"
                        ),
                        now=now,
                    )
                    continue
                output = BatchOutput(
                    item_id=item_id, output_sha256=digest, output_bytes=nbytes
                )
                try:
                    packet = _unwrap_scoring_fault(
                        await retry_async(
                            lambda i=item, o=output, c=contender: (
                                _scoring_fault_as_result(
                                    asyncio.to_thread(
                                        self.scoring_client.score_item,
                                        competition_id,
                                        c.contender_id,
                                        i,
                                        o,
                                    )
                                )
                            ),
                            attempts=self.cfg.scoring_retry_attempts,
                            base_delay=self.cfg.retry_base_delay_seconds,
                            max_delay=self.cfg.retry_max_delay_seconds,
                        )
                    )
                    # The worker may not answer under the orchestrator's reserved
                    # zero namespace — that is the packet-level half of the
                    # anti-impersonation rule (the /healthz check is the other).
                    _assert_packet_identity_not_reserved(packet.packet_bytes)
                    self._persist_scored_item(
                        competition_id,
                        contender,
                        item_row,
                        output,
                        packet.packet_bytes,
                        manifest_ref,
                        manifest.scoring_version,
                        batch_size,
                        now,
                    )
                except ReservedScorerIdentity as exc:
                    self._halt(
                        competition_id, f"{exc} Operator action: {_RESERVED_FIX}", now
                    )
                    return
                except Exception as exc:
                    if classify_failure(exc) is Fault.CONTENDER:
                        # The trusted scorer refused the CONTENDER'S OWN OUTPUT:
                        # that item is zero, the competition continues (#14).
                        code = fault_code(exc)
                        self.m_contender_faults.labels(code=code).inc()
                        self._zero_score_item(
                            competition_id,
                            contender,
                            item_row,
                            manifest,
                            manifest_ref,
                            batch_size,
                            code=ReasonCode.METRIC_MISSING,
                            detail=f"scorer rejected the output ({code}): {exc}"[:400],
                            now=now,
                        )
                        continue
                    # Scoring infra is systemic: the same worker scores everyone,
                    # so an exhausted budget (or a malformed packet) halts — the
                    # competition is NOT failed (spec §14).
                    self._halt(
                        competition_id,
                        f"scoring failed for contender {contender.contender_id} "
                        f"item {item_id}: {exc}",
                        now,
                    )
                    return
                self.m_scorings.inc()
        if (
            repo.count_missing_item_scores(self.conn, competition_id) == 0
            and repo.count_missing_calibration_rows(self.conn, competition_id) == 0
        ):
            self.engine.mark_scores_persisted(self.conn, competition_id, now)

    def _zero_score_item(
        self,
        competition_id: str,
        contender: repo.ContenderRecord,
        item_row: sqlite3.Row,
        manifest: CompetitionManifest,
        manifest_ref: ArtifactRef,
        batch_size: int,
        *,
        code: ReasonCode,
        detail: str,
        now: datetime,
    ) -> None:
        """Persist a gate-failed ZERO for one (contender, item), with a reason code.

        This packet is minted by the ORCHESTRATOR, not by the scoring worker, and
        it is the only packet that ever is. It is honest to do so because it
        asserts no measurement: gate_passed=False forces score 0.0 by the
        gates-first invariant (compose_item_score / record_item_score both enforce
        it), the violation carries the machine-readable reason, and the canonical
        empty artifact stands in for "there were no bytes to measure". The
        alternative — feeding an empty file to ffmpeg — produced a 502 that halted
        the competition.

        IT IS ALSO ATTRIBUTED HONESTLY: the packet carries
        the orchestrator's OWN reserved identity, `orchestrator-zero/1+<digest12>`,
        never `manifest.scoring_version` — stamping the worker's identity made a
        SCORE_PACKET artifact claim the worker had produced bytes it never saw. The
        item's audit bundle is built with the SAME identity, so packet and bundle
        agree (audit/recompute.py cross-checks them) while the bundle's manifest
        still names the committed worker. See orchestrator/zero_packets.py.

        The row is audit-linked exactly like a measured one. The CPU auditor's
        reserved-zero path independently verifies the empty output and closed
        packet shape before its zero may influence an economic packet mean.
        """
        output = BatchOutput(
            item_id=item_row["item_id"], output_sha256=EMPTY_SHA256, output_bytes=0
        )
        self._ensure_output_bytes(output)
        packet, scorer_version = mint_zero_packet(
            scoring_item_id=item_row["scoring_item_id"],
            challenge_id=item_row["challenge_id"],
            track=manifest.track,
            committed_scoring_version=manifest.scoring_version,
            miner_hotkey=contender.hotkey,
            empty_digest=EMPTY_SHA256,
            code=code,
            detail=detail,
            config=self.scoring_config,
        )
        self._persist_scored_item(
            competition_id,
            contender,
            item_row,
            output,
            packet.to_json().encode("utf-8"),
            manifest_ref,
            scorer_version,
            batch_size,
            now,
        )
        with pers.txn(self.conn):
            repo.record_event(
                self.conn,
                competition_id,
                pers.EVENT_ITEM_ZEROED,
                now,
                payload={
                    "contender_id": contender.contender_id,
                    "item_id": item_row["item_id"],
                    "code": str(code),
                    "detail": detail[:500],
                    # On the record, in the append-only log: this zero is an
                    # ORCHESTRATOR attribution, and the identity that says so.
                    "minted_by": scorer_version,
                    "committed_scoring_version": manifest.scoring_version,
                },
            )
        self.m_zero_scored.labels(code=str(code)).inc()
        self.m_scorings.inc()
        self.log.warning(
            "item zero-scored by the ORCHESTRATOR with a reason code (no bytes to "
            "measure); competition continues",
            extra=log_fields(
                competition_id=competition_id,
                contender_id=contender.contender_id,
                item_id=item_row["item_id"],
                code=str(code),
                detail=detail[:500],
                minted_by=scorer_version,
            ),
        )

    def _persist_scored_item(
        self,
        competition_id: str,
        contender: repo.ContenderRecord,
        item_row: sqlite3.Row,
        output: BatchOutput,
        packet_bytes: bytes,
        manifest_ref: ArtifactRef,
        scorer_version: str,
        batch_size: int,
        now: datetime,
    ) -> None:
        """Archive the artifacts and commit the score row + audit bundle atomically.

        `scorer_version` is WHO MINTED THESE EXACT PACKET BYTES — the committed
        worker identity for a measured packet, the orchestrator-zero identity for a
        locally minted gate-failure record. It is stamped on
        the audit bundle so packet and bundle always agree; it is deliberately NOT
        always the manifest's `scoring_version`, and a consumer must not read the
        difference as drift.

        Before ANY score/bundle artifact or DB write, re-parse and bind the packet
        to the local manifest/item/contender/output/config facts.
        HttpScoringClient already performs the same checks against its exact
        request and health runtime; this second boundary prevents a future or
        injected client from turning a moved but self-consistent packet into
        persisted/ranked economic state.
        """
        # The bundle is the independently fetched provenance envelope.  Copy the
        # complete backend stamp from the exact packet bytes it links; leaving the
        # bundle's default empty mapping made packet/backend verification vacuous
        # for measured competition scores.
        packet = ItemScore.model_validate_json(packet_bytes)
        manifest = repo.get_manifest(self.conn, competition_id)
        expected_config_digest = config_digest(self.scoring_config)
        binding_problems: list[str] = []
        expected_bindings: dict[str, object] = {
            "challenge_id": str(item_row["challenge_id"]),
            "item_id": str(item_row["scoring_item_id"]),
            "track": manifest.track,
            "miner_hotkey": contender.hotkey,
            "content_digest": output.output_sha256,
            "scorer_version": scorer_version,
            "scoring_config_digest": expected_config_digest,
        }
        for field, expected in expected_bindings.items():
            observed = getattr(packet, field, None)
            if observed != expected:
                binding_problems.append(f"{field}={observed!r}, expected {expected!r}")
        orchestrator_zero = is_orchestrator_zero_identity(scorer_version)
        if not orchestrator_zero and scorer_version != manifest.scoring_version:
            binding_problems.append(
                f"measured packet attribution {scorer_version!r} differs from "
                f"manifest scorer {manifest.scoring_version!r}"
            )

        # Production HttpScoringClient exposes the exact health-derived map it
        # already checked. Re-read it here so packet -> bundle persistence cannot
        # accidentally weaken complete-map equality. In-process test clients have
        # no such method and remain an explicit non-production seam.
        expected_backends_probe = getattr(
            self.scoring_client, "expected_backend_versions", None
        )
        if not orchestrator_zero and callable(expected_backends_probe):
            try:
                expected_backends = dict(expected_backends_probe())
            except Exception as exc:  # noqa: BLE001 - a missing pin is fail-closed
                raise repo.ScorePacketError(
                    "cannot establish the scoring worker's health-bound backend "
                    f"contract before persistence: {type(exc).__name__}: {exc}"
                ) from exc
            if packet.backend_versions != expected_backends:
                changed = sorted(
                    key
                    for key in (set(packet.backend_versions) | set(expected_backends))
                    if packet.backend_versions.get(key) != expected_backends.get(key)
                )
                binding_problems.append(
                    "backend_versions differs from the complete health runtime map "
                    f"at key(s) {changed!r}"
                )
        if binding_problems:
            raise repo.ScorePacketError(
                "score packet is not bound to the local competition request: "
                + "; ".join(binding_problems)
            )

        # Content-addressed artifact puts are idempotent and crash-safe on their
        # own; only the DB rows need the transaction. The output put deliberately
        # precedes the digest check: an output race may leave an unreferenced object,
        # but it can never produce a packet, bundle or economic row.
        input_ref = self._challenge_input_ref(item_row)
        output_ref = self.store.put_file(
            self.outputs_dir / output.output_sha256, ArtifactKind.MINER_OUTPUT
        )
        if (
            output_ref.digest != output.output_sha256
            or output_ref.byte_size != output.output_bytes
        ):
            raise repo.ScorePacketError(
                "materialized miner output differs from the scored BatchOutput: "
                f"stored={output_ref.digest}/{output_ref.byte_size}, "
                f"expected={output.output_sha256}/{output.output_bytes}"
            )
        packet_ref = self.store.put(packet_bytes, ArtifactKind.SCORE_PACKET)
        reference_ref: ArtifactRef | None = None
        competition_item: CompetitionItemBinding | None = None
        stage = LifecycleStage.PRE_REVEAL
        if manifest.track == "upscaling":
            # Covers local zero packets too: even if no contender output exists, the
            # economic record still binds the same immutable item preimage.
            repo.validate_evaluation_item_bindings(self.conn, competition_id)
            reference_ref = self._reference_original_ref(item_row)
            competition_item = CompetitionItemBinding(
                item_index=int(item_row["item_index"]),
                input_sha256=str(item_row["input_sha256"]),
                reference_sha256=str(item_row["reference_sha256"]),
                upscale_factor=int(item_row["upscale_factor"]),
                target_width=int(item_row["target_width"]),
                target_height=int(item_row["target_height"]),
                item_commitment=str(item_row["item_commitment"]),
            )
            stage = LifecycleStage.COMPETITION_SEALED
        bundle = build_bundle(
            challenge_id=item_row["challenge_id"],
            item_id=item_row["scoring_item_id"],
            miner_hotkey=contender.hotkey,
            commitment_hash=item_row["threshold_commitment"],
            stage=stage,
            challenge_input=input_ref,
            miner_output=output_ref,
            manifest=manifest_ref,
            score_packet=packet_ref,
            reference_original=reference_ref,
            competition_item=competition_item,
            execution_image_digest=contender.image_digest,
            scorer_version=scorer_version,
            backend_versions=dict(packet.backend_versions),
            created_at=repo.iso(now),
        )
        bundle_digest = bundle.bundle_digest()
        # Persist the bundle itself, not only its digest/event-log copy.  Epoch
        # manifests address AUDIT_BUNDLE objects by this digest and a remote CPU
        # auditor must be able to resolve those bytes from the shared audit store.
        # The canonical bundle JSON hashes to ``bundle_digest`` by construction.
        bundle_ref = self.store.put(
            canonical_json_bytes(bundle.model_dump(mode="json")),
            ArtifactKind.AUDIT_BUNDLE,
        )
        if bundle_ref.digest != bundle_digest:
            raise RuntimeError(
                f"persisted audit bundle digest {bundle_ref.digest} does not match "
                f"the canonical bundle digest {bundle_digest}"
            )
        batch_id = pers.batch_id_for_item(
            self.conn,
            competition_id,
            contender.contender_id,
            item_row["item_index"],
            batch_size,
        )
        with pers.txn(self.conn):
            performance_id = repo.record_item_score(
                self.conn,
                competition_id,
                contender_id=contender.contender_id,
                item_id=item_row["item_id"],
                packet_bytes=packet_bytes,
                now=now,
                output_bytes=output.output_bytes,
                batch_id=batch_id,
            )
            # The full bundle goes into the append-only event log so the digest
            # on the score row is independently checkable from the DB alone.
            repo.record_event(
                self.conn,
                competition_id,
                "audit_bundle_built",
                now,
                payload={
                    "performance_id": performance_id,
                    "bundle_digest": bundle_digest,
                    "bundle": bundle.model_dump(mode="json"),
                },
            )
            repo.set_audit_bundle_digest(self.conn, performance_id, bundle_digest)

    def _challenge_input_ref(self, item_row: sqlite3.Row) -> ArtifactRef:
        digest = item_row["input_sha256"]
        ref = ArtifactRef(
            digest=digest,
            kind=ArtifactKind.CHALLENGE_INPUT,
            byte_size=item_row["input_bytes"],
            backend_key=backend_key(ArtifactKind.CHALLENGE_INPUT, digest),
        )
        if not self.store.exists(ref):
            # Item was seeded out-of-band (tests / direct repo use): archive the
            # pooled sealed input now — the bundle must reference stored bytes.
            ref = self.store.put_file(
                self.inputs_dir / digest, ArtifactKind.CHALLENGE_INPUT
            )
        return ref

    def _reference_original_ref(self, item_row: sqlite3.Row) -> ArtifactRef:
        digest = str(item_row["reference_sha256"])
        ref = ArtifactRef(
            digest=digest,
            kind=ArtifactKind.REFERENCE_ORIGINAL,
            byte_size=int(item_row["reference_bytes"]),
            backend_key=backend_key(ArtifactKind.REFERENCE_ORIGINAL, digest),
        )
        if not self.store.exists(ref):
            # Trusted scorer staging is not miner-visible; recover an out-of-band
            # repository seed while retaining sealed-at-rest storage semantics.
            ref = self.store.put_file(
                self.inputs_dir / digest, ArtifactKind.REFERENCE_ORIGINAL
            )
        return ref

    def _ensure_output_bytes(self, output: BatchOutput) -> None:
        """Materialize the canonical empty-output artifact for an ABSENT output.

        Used only on the local zero-score path: the audit bundle must reference
        stored bytes, and "no output" is honestly represented by the empty blob.
        These bytes are never sent to the scorer (that 502'd and halted the
        competition)."""
        path = self.outputs_dir / output.output_sha256
        if output.output_bytes == 0 and not path.exists():
            path.write_bytes(b"")

    # ---- halt ------------------------------------------------------------------

    def _halt(self, competition_id: str, reason: str, now: datetime) -> None:
        if pers.record_halt(self.conn, competition_id, reason, now):
            self.m_halts.inc()
            self.log.critical(
                "pipeline HALTED on systemic infra blocker — competition left in "
                "its current phase (never failed by infra); operator action: fix "
                "the blocker, then clear_halt",
                extra=log_fields(competition_id=competition_id, reason=reason),
            )

    def clear_halt(
        self,
        competition_id: str,
        operator: str,
        now: datetime,
        *,
        reason: str,
    ) -> bool:
        cleared = pers.clear_halt(
            self.conn, competition_id, operator, now, reason=reason
        )
        if cleared:
            self.log.info(
                "pipeline halt cleared",
                extra=log_fields(
                    competition_id=competition_id,
                    operator=operator.strip(),
                    reason=reason.strip(),
                ),
            )
        return cleared

    # ---- repository/engine passthroughs ----------------------------------------

    def _manifest_gpu_problem(self, manifest: CompetitionManifest) -> str | None:
        """Return a live-Modal GPU policy mismatch, if any.

        Docker and injected fake runners remain the explicit report/test path. A
        live Modal composition has two independent facts: the operator config and
        the runner's actual request value. Both must agree and the resulting exact
        GPU string must be committed in ``manifest.allowed_gpus``.
        """
        if self.cfg.sandbox_backend != "modal":
            return None
        configured = self.cfg.modal_gpu.strip()
        actual_value = getattr(self.runner, "gpu", None)
        actual = actual_value.strip() if isinstance(actual_value, str) else ""
        if not actual:
            return (
                "live Modal runner does not attest its requested GPU; refusing "
                "manifest execution"
            )
        if actual != configured:
            return (
                f"live Modal runner GPU {actual!r} does not match configured "
                f"orchestrator.modal_gpu {configured!r}"
            )
        if actual not in manifest.allowed_gpus:
            return (
                f"live Modal runner GPU {actual!r} is not committed in manifest."
                f"allowed_gpus={manifest.allowed_gpus!r}"
            )
        return None

    def _assert_manifest_gpu(self, manifest: CompetitionManifest) -> None:
        problem = self._manifest_gpu_problem(manifest)
        if problem is not None:
            raise ValueError(problem)

    def _halt_on_manifest_gpu_problem(
        self, competition_id: str, manifest: CompetitionManifest, now: datetime
    ) -> bool:
        problem = self._manifest_gpu_problem(manifest)
        if problem is None:
            return False
        self._halt(
            competition_id,
            "competition GPU policy mismatch before contender execution: " + problem,
            now,
        )
        return True

    def create_competition(self, manifest: CompetitionManifest, now: datetime) -> None:
        # Reject before persistence/anchoring: the committed manifest must permit
        # the exact GPU this live runner will request from Modal.
        self._assert_manifest_gpu(manifest)
        if self.tokenomics.competition_emissions_enabled and manifest.baseline is None:
            raise EarningManifestError(
                "competition emissions are enabled, so the manifest requires "
                "exactly one non-earning archived baseline with pinned repo, commit, "
                "and tree identities"
            )
        if self.tokenomics.competition_emissions_enabled and (
            self._chain_mode == "bittensor"
            or self.cfg.baseline_registry_db_path is not None
        ):
            self._require_active_baseline(manifest)
        self.engine.create_competition(self.conn, manifest, now)

    def _require_active_baseline(self, manifest: CompetitionManifest) -> None:
        """Bind a new earning cycle to the registry's exact serving executable."""
        baseline = manifest.baseline
        if baseline is None:  # narrowed by ``create_competition``; defensive seam.
            raise EarningManifestError("earning manifest has no archived baseline")
        path = self.cfg.baseline_registry_db_path
        if path is None:
            raise EarningManifestError(
                "earning competition creation requires orchestrator.baseline_registry_db_path"
            )
        uri = f"file:{Path(path).resolve()}?mode=ro"
        try:
            registry = sqlite3.connect(uri, uri=True, timeout=5)
            registry.row_factory = sqlite3.Row
            row = registry.execute(
                "SELECT version, artifact_digest, artifact_bytes, image_digest, "
                "provenance_digest, provenance_bytes, repo_url, commit_sha, tree_sha "
                "FROM baselines "
                "WHERE track = ? AND status = 'active'",
                (manifest.track,),
            ).fetchone()
            pending = registry.execute(
                "SELECT 1 FROM baseline_promotion_latches "
                "WHERE track = ? AND status = 'pending' LIMIT 1",
                (manifest.track,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise EarningManifestError(
                f"active baseline registry is unreadable for {manifest.track}: {exc}"
            ) from exc
        finally:
            if "registry" in locals():
                registry.close()
        if pending is not None:
            raise EarningManifestError(
                f"{manifest.track} has an unresolved CROWN promotion; the next "
                "competition cannot start until its executable is activated"
            )
        if row is None:
            raise EarningManifestError(
                f"baseline registry has no active {manifest.track} executable"
            )
        committed = (
            baseline.version,
            baseline.artifact_digest,
            baseline.artifact_bytes,
            baseline.image_digest,
            baseline.provenance_digest,
            baseline.provenance_bytes,
            baseline.repo_url,
            baseline.commit_sha,
            baseline.tree_sha,
        )
        active = (
            int(row["version"]),
            str(row["artifact_digest"]),
            int(row["artifact_bytes"]),
            str(row["image_digest"]),
            str(row["provenance_digest"]),
            int(row["provenance_bytes"]),
            str(row["repo_url"]),
            str(row["commit_sha"]),
            str(row["tree_sha"]),
        )
        if committed != active:
            raise EarningManifestError(
                "competition baseline does not exactly match the registry's active "
                f"{manifest.track} row: manifest={committed!r}, registry={active!r}"
            )

    def _load_earning_commitment(
        self, competition_id: str, manifest: CompetitionManifest
    ) -> CompetitionCommitment:
        """Open and validate the anchored commitment before any earning build.

        The root alone is insufficient: execution must prove that the openable
        preimage names this manifest, its baseline tree, the active reward policy, and
        ultimately the baseline image actually returned by the sandbox runner.
        """
        competition = repo.get_competition(self.conn, competition_id)
        if competition is None or competition.commitment_root is None:
            raise ValueError("competition has no anchored commitment root")
        if manifest.baseline is None:
            raise EarningManifestError("earning manifest has no archived baseline")
        commitment = load_competition_commitment(
            self.store, competition.commitment_root
        )
        expected = {
            "manifest_digest": manifest.manifest_digest(),
            "baseline_version": manifest.baseline.version,
            "baseline_artifact_digest": manifest.baseline.artifact_digest,
            "baseline_provenance_digest": manifest.baseline.provenance_digest,
            "baseline_tree_digest": pin_git_sha(manifest.baseline.tree_sha),
            "dataset_selection_seed_commitment": manifest.scoring_seed_commitment,
            "reward_param_digest": reward_parameter_digest(self.tokenomics),
        }
        for field, value in expected.items():
            observed = getattr(commitment, field)
            if observed != value:
                raise ValueError(
                    f"anchored {field} {observed} does not match active {value}"
                )
        return commitment

    def anchor_commitment(
        self, competition_id: str, commitment_root: str, now: datetime
    ) -> bool:
        """LOW-LEVEL: record a root that was ALREADY anchored elsewhere.

        This is not an anchor path — it writes SQLite and nothing else. Callers
        that own the chain write themselves (the e2e harness anchors through the
        fake chain directly, to assert the payload bytes) use it; everything else
        must use `anchor_competition`, which is THE anchor path (review
        service-review #11).
        """
        return self.engine.mark_commitment_anchored(
            self.conn, competition_id, commitment_root, now
        )

    async def anchor_competition(
        self,
        competition_id: str,
        *,
        baseline_image_digest: str | None = None,
        reward_param_digest: str,
        baseline_tree_digest: str | None = None,
        now: datetime | None = None,
    ) -> AnchorResult:
        """THE single anchor path — chainless report mode and the real chain share it.

        Builds the pre-enrollment CompetitionCommitment from the PERSISTED manifest
        (manifest digest and dataset-selection seed commitment are never taken from
        the caller), keeps the canonical JSON openable in the audit store, CLAIMS
        THE ANCHORING RIGHT IN THE DB, anchors the payload through the injected
        ChainAdapter once, independently proves its exact finalized inclusion and
        archive state, and only then records the root. If proof fails, nothing is
        marked anchored — enrollment stays closed,
        which is the fail-closed behaviour the schema's SCHEDULED->ENROLLING guard
        already assumes.

        SERIALIZED BEFORE THE CHAIN, NOT AFTER. The claim is
        taken inside a BEGIN IMMEDIATE transaction that re-verifies SCHEDULED + no
        commitment_root + no in-flight claim, and records the EXACT payload digest.
        A second concurrent request — with the same payload or a different one —
        fails that step and is refused BEFORE it can reach the chain, so a
        competition can never end up with two on-chain commitments of which only
        one is tracked. See the module docstring for the full protocol and the
        crash-recovery rules.

        Raises NotConfiguredError when no ChainAdapter is wired (there is no
        DB-only anchor: that was exactly the report/real-chain drift of review
        service-review #11), ValueError for an unusable commitment,
        AnchorClaimRefused when the claim step says no (no chain write happened),
        AnchorError when the submitted/ambiguous write cannot be independently
        verified.
        """
        at = now or self.now()
        if self.chain is None:
            raise NotConfiguredError(
                "no ChainAdapter is wired into the orchestrator: anchoring is the "
                "one path report mode and the real chain must share, so there is "
                "no SQLite-only fallback (the project design record rule 8)"
            )
        manifest = repo.get_manifest(self.conn, competition_id)
        if manifest.baseline is None:
            raise EarningManifestError(
                "schema-v14 anchoring requires the persisted manifest's exact "
                "archived executable baseline and provenance"
            )
        expected_reward_digest = reward_parameter_digest(self.tokenomics)
        if reward_param_digest != expected_reward_digest:
            raise ValueError(
                "reward_param_digest does not match the active canonical "
                f"TokenomicsConfig: expected {expected_reward_digest}"
            )
        if baseline_tree_digest is None:
            if manifest.baseline is None:
                raise ValueError(
                    "baseline_tree_digest is required: the manifest declares no baseline "
                    "baseline to derive it from"
                )
            baseline_tree_digest = pin_git_sha(manifest.baseline.tree_sha)
        elif manifest.baseline is not None:
            expected_baseline_tree = pin_git_sha(manifest.baseline.tree_sha)
            if baseline_tree_digest != expected_baseline_tree:
                raise ValueError(
                    "baseline_tree_digest does not match the baseline tree pinned by the "
                    f"persisted manifest: expected {expected_baseline_tree}"
                )
        if manifest.baseline is not None:
            # Every declared baseline is built before the chain claim. Persist the
            # process-local Modal runtime fence first so even this pre-enrollment
            # handle has explicit provenance; report/Docker runners are a no-op.
            if not await self._ensure_fresh_modal_runtime(
                competition_id, Phase.SCHEDULED, at
            ):
                raise ValueError(
                    "the baseline runtime could not be bound before its "
                    "pre-enrollment build; no chain write was attempted"
                )
            baseline_spec = ContenderSpec(
                contender_id=0,
                repo_url=manifest.baseline.repo_url,
                commit_sha=manifest.baseline.commit_sha,
                tree_sha=manifest.baseline.tree_sha,
            )
            try:
                built_baseline_digest: str | None = None
                modal_controls = self._modal_image_controls()
                if modal_controls is not None:
                    if baseline_image_digest is not None:
                        binding = pers.latest_modal_image_binding(
                            self.conn,
                            competition_id,
                            baseline_image_digest,
                            is_calibration=True,
                        )
                    else:
                        binding = pers.latest_modal_calibration_binding(
                            self.conn, competition_id
                        )
                    if binding is not None:
                        bound_digest = binding.get("image_digest")
                        if not isinstance(bound_digest, str):
                            raise ValueError(
                                "persisted baseline Modal image digest is malformed"
                            )
                        self._require_modal_binding_matches(
                            binding,
                            baseline_spec,
                            bound_digest,
                            is_calibration=True,
                        )
                        built_baseline_digest = await self._restore_bound_modal_image(
                            competition_id,
                            baseline_spec,
                            bound_digest,
                            is_calibration=True,
                        )
                if built_baseline_digest is None:
                    built_baseline_digest = await retry_async(
                        lambda: asyncio.to_thread(self.runner.build, baseline_spec),
                        attempts=self.cfg.build_retry_attempts,
                        base_delay=self.cfg.retry_base_delay_seconds,
                        max_delay=self.cfg.retry_max_delay_seconds,
                    )
            except Exception as exc:
                raise ValueError(
                    "the baseline could not be prebuilt before anchoring; no "
                    f"chain write was attempted: {exc}"
                ) from exc
            if built_baseline_digest != manifest.baseline.image_digest:
                raise ValueError(
                    "the archived baseline image identity does not match the image "
                    f"rebuilt by this fresh runner: registry={manifest.baseline.image_digest}, "
                    f"rebuilt={built_baseline_digest}"
                )
            if (
                baseline_image_digest is not None
                and built_baseline_digest != baseline_image_digest
            ):
                raise ValueError(
                    "baseline_image_digest does not match the image prebuilt by this "
                    f"fresh runner: expected {built_baseline_digest}"
                )
            baseline_image_digest = built_baseline_digest
            with pers.txn(self.conn):
                self._record_modal_image_binding(
                    competition_id,
                    baseline_spec,
                    baseline_image_digest,
                    is_calibration=True,
                    now=at,
                )
        elif baseline_image_digest is None:
            raise ValueError(
                "baseline_image_digest is required when the manifest declares no "
                "buildable archived baseline"
            )
        assert baseline_image_digest is not None
        payload = build_competition_commitment(
            CompetitionCommitment(
                manifest_digest=manifest.manifest_digest(),
                baseline_version=manifest.baseline.version,
                baseline_artifact_digest=manifest.baseline.artifact_digest,
                baseline_provenance_digest=manifest.baseline.provenance_digest,
                baseline_tree_digest=baseline_tree_digest,
                baseline_image_digest=baseline_image_digest,
                dataset_selection_seed_commitment=manifest.scoring_seed_commitment,
                reward_param_digest=reward_param_digest,
            )
        )
        # The on-chain root is always backed by an openable stored document
        # (commitments.py convention).
        self.store.put(payload.canonical_json, ArtifactKind.MANIFEST)
        payload_digest = hashlib.sha256(payload.payload).hexdigest()
        # Capacity check + claim + write share the authority's cross-process lane.
        # This makes the epoch-anchor reserve atomic with respect to challenge and
        # competition writers. The adapter's nested use of the same lock is
        # re-entrant in this task.
        async with anchor_writer_lock(
            self._anchor_writer_lock_path,
            timeout_seconds=self._anchor_writer_lock_timeout_seconds,
        ):
            try:
                await require_commitment_capacity(
                    self.chain,
                    netuid=self._anchor_netuid,
                    hotkey=self._anchor_hotkey,
                    payload=payload.payload,
                    operation=f"competition {competition_id} pre-enrollment anchor",
                    reserve_payload_bytes=EPOCH_ANCHOR_CAPACITY_RESERVE_BYTES,
                )
            except CommitmentCapacityError as exc:
                self.m_anchors.labels(result="capacity_refused").inc()
                self.log.warning(
                    "competition anchor refused before claim/write to preserve "
                    "commitment capacity",
                    extra=log_fields(
                        competition_id=competition_id,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
                raise
            # STEP 1 — claim, guarded, BEFORE anything external happens. A stale
            # identical claim enters READ-ONLY recovery; it never submits the
            # payload again. Raises AnchorClaimRefused before chain I/O otherwise.
            recover_only = self._claim_anchor(
                competition_id, payload_digest, payload.root, at
            )

            # STEP 2 — at most ONE external write for a new claim. A timeout or
            # transport error is ambiguous, so independent read-back is the
            # authority: it may prove the write landed, but we never blind-retry it.
            tx_id: str | None = None
            write_error: Exception | None = None
            if not recover_only:
                try:
                    raw_tx_id = await with_timeout(
                        self.chain.anchor_commitment(payload.payload),  # type: ignore[union-attr]
                        self.cfg.chain_timeout_seconds,
                        "anchor_commitment",
                    )
                    tx_id = str(raw_tx_id)
                except asyncio.CancelledError:
                    # The adapter holds its own non-cancellable writer worker to
                    # completion before propagating cancellation. Leave the claim
                    # open; a later stale request can only read/verify, never write.
                    raise
                except Exception as exc:  # response loss may still mean LANDED
                    write_error = exc

            try:
                receipt = await wait_for_finalized_commitment_receipt(
                    self.chain,  # type: ignore[arg-type]
                    netuid=self._anchor_netuid,
                    expected_payload=payload.payload,
                    operation=f"competition {competition_id} pre-enrollment anchor",
                    timeout_seconds=self.cfg.anchor_receipt_timeout_seconds,
                    poll_seconds=self.cfg.anchor_receipt_poll_seconds,
                )
            except AnchorReceiptVerificationError as exc:
                self.m_anchors.labels(result="failed").inc()
                detail = (
                    f"; write response was {type(write_error).__name__}: {write_error}"
                    if write_error is not None
                    else ""
                )
                with pers.txn(self.conn):
                    pers.record_anchor_failure(
                        self.conn,
                        competition_id,
                        payload_digest=payload_digest,
                        reason=f"receipt verification failed: {exc}{detail}",
                        now=at,
                    )
                raise AnchorError(
                    f"anchoring competition {competition_id} did not obtain an exact "
                    "finalized/archive receipt; nothing was marked anchored and "
                    f"enrollment stays closed: {exc}{detail}. The claim for payload "
                    f"{payload_digest[:12]} stays OPEN. Recovery is read-only; a new "
                    "chain write requires an operator to verify that nothing landed "
                    "and explicitly release the claim."
                ) from exc

        recovered = recover_only or write_error is not None
        if write_error is not None:
            self.log.warning(
                "recovered a competition anchor whose write response was lost",
                extra=log_fields(
                    competition_id=competition_id,
                    root=payload.root,
                    anchor_block=receipt.block,
                    error=f"{type(write_error).__name__}: {write_error}",
                ),
            )

        # STEP 3 — atomically record lifecycle root + complete receipt event. The
        # resolving event cannot be lost in a crash after the root becomes earning.
        evidence = {
            "root": payload.root,
            "tx_id": tx_id,
            "anchor_netuid": self._anchor_netuid,
            "payload_hex": payload.payload.hex(),
            "payload_digest": payload_digest,
            "anchor_block": receipt.block,
            "anchor_block_hash": receipt.block_hash,
            "finalized_block": receipt.finalized_block,
            "archive_verified": True,
            "write_response_recovered": recovered,
        }
        try:
            recorded = self.engine.mark_commitment_anchored(
                self.conn,
                competition_id,
                payload.root,
                at,
                onchain_evidence=evidence,
            )
        except Exception:  # noqa: BLE001 - the chain already holds it; say so
            self.log.exception(
                "the chain anchor is independently proven but the atomic lifecycle/"
                "receipt transaction failed; the unresolved claim remains for "
                "read-only recovery",
                extra=log_fields(competition_id=competition_id, root=payload.root),
            )
            recorded = False
        self.m_anchors.labels(result="ok" if recorded else "not_recorded").inc()
        self.log.info(
            "competition commitment anchored through the ChainAdapter",
            extra=log_fields(
                competition_id=competition_id,
                root=payload.root,
                tx_id=tx_id,
                anchor_block=receipt.block,
                anchor_block_hash=receipt.block_hash,
                archive_verified=1,
                recorded=recorded,
            ),
        )
        return AnchorResult(
            root=payload.root,
            tx_id=tx_id,
            payload=payload.payload,
            canonical_json=payload.canonical_json,
            baseline_image_digest=baseline_image_digest,
            anchor_block=receipt.block,
            anchor_block_hash=receipt.block_hash,
            finalized_block=receipt.finalized_block,
            write_response_recovered=recovered,
            recorded=recorded,
        )

    def _claim_anchor(
        self, competition_id: str, payload_digest: str, root: str, at: datetime
    ) -> bool:
        """Take the exclusive right to anchor THIS payload.

        Returns ``False`` for a brand-new claim (the caller may make exactly one
        write) and ``True`` for a stale identical claim (READ-ONLY receipt recovery;
        the caller must not resubmit). Refusals perform no chain I/O.

        One BEGIN IMMEDIATE transaction, no awaits inside: concurrent requests on
        this event loop cannot interleave within it, and concurrent PROCESSES are
        serialized by SQLite's write lock. Checks, in order:

        1. the competition exists, is still SCHEDULED and has no commitment_root
           (an already-anchored competition needs no second chain write — that is
           the whole bug: the old code hit the chain first and asked afterwards);
        2. no open anchor claim. A FRESH claim refuses every request; a STALE claim
           (a crashed attempt whose outcome is unknown) admits only a READ-BACK of
           the IDENTICAL payload. It never writes again. A different payload remains
           refused until an operator proves nothing landed and resolves the claim.
        """
        with pers.txn(self.conn):
            comp = repo.get_competition(self.conn, competition_id)
            if comp is None:
                raise AnchorClaimRefused(
                    f"unknown competition {competition_id}", code="unknown_competition"
                )
            if comp.commitment_root is not None:
                raise AnchorClaimRefused(
                    f"competition {competition_id} is already anchored to "
                    f"{comp.commitment_root}; no second chain write is performed "
                    "(re-anchoring would leave an untracked commitment on chain)",
                    code="already_anchored",
                )
            if comp.status is not Phase.SCHEDULED:
                raise AnchorClaimRefused(
                    f"competition {competition_id} is {comp.status.value}, not "
                    "SCHEDULED: the pre-enrollment commitment can only be anchored "
                    "before enrollment opens",
                    code="not_scheduled",
                )
            claim = pers.open_anchor_claim(self.conn, competition_id)
            if claim is not None:
                age = _claim_age_seconds(claim, at)
                if age < self.cfg.anchor_claim_stale_seconds:
                    raise AnchorClaimRefused(
                        f"an anchor for competition {competition_id} is already in "
                        f"flight (payload {str(claim.get('payload_digest'))[:12]}, "
                        f"claimed {age:.0f}s ago). Refusing to touch the chain: two "
                        "concurrent anchors would leave a second, untracked "
                        "commitment on chain.",
                        code="anchor_in_progress",
                    )
                if str(claim.get("payload_digest")) != payload_digest:
                    raise AnchorClaimRefused(
                        f"competition {competition_id} has an UNRESOLVED anchor "
                        f"claim for payload "
                        f"{str(claim.get('payload_digest'))[:12]} (claimed "
                        f"{age:.0f}s ago) whose outcome is unknown — it may already "
                        f"be on chain. Refusing to anchor a DIFFERENT payload "
                        f"({payload_digest[:12]}) over it: check the chain for root "
                        f"{str(claim.get('root'))[:16]}, then call "
                        "release_anchor_claim to resolve it.",
                        code="anchor_ambiguous",
                    )
                self.log.warning(
                    "resuming a stale anchor claim in READ-ONLY verification mode; "
                    "the commitment will not be resubmitted",
                    extra=log_fields(
                        competition_id=competition_id,
                        payload_digest=payload_digest,
                        claim_age_seconds=round(age, 1),
                    ),
                )
                return True
            pers.record_anchor_claim(
                self.conn,
                competition_id,
                payload_digest=payload_digest,
                root=root,
                now=at,
            )
            return False

    def release_anchor_claim(
        self, competition_id: str, operator: str, reason: str, now: datetime
    ) -> bool:
        """Operator resolution of an AMBIGUOUS anchor claim.

        A chain write that timed out may or may not have landed. The orchestrator
        performs bounded exact read-back and stale identical requests remain
        read-only. Once an operator has independently proved that nothing landed,
        this closes the claim so a new anchor can proceed —
        exactly like clear_halt, and just as deliberately manual: silently
        abandoning the claim would restore the double-anchor hazard.
        """
        with pers.txn(self.conn):
            released = pers.release_anchor_claim(
                self.conn,
                competition_id,
                operator=operator,
                reason=reason,
                now=now,
            )
        if released:
            self.log.warning(
                "anchor claim released by an operator",
                extra=log_fields(
                    competition_id=competition_id, operator=operator, reason=reason
                ),
            )
        return released

    def build_result(
        self,
        competition_id: str,
        *,
        census_by_hotkey: Mapping[str, "MinerCensusEntry"] | None = None,
        applied_at: datetime | None = None,
    ) -> CompetitionResult:
        """Return an auditable packet-derived economic preview for a supplied census.

        Stored human ranks/review eligibility and paired operational margins are
        deliberately ignored.  Every subject packet and bundle must resolve from
        the audit store, and every BUILT contender must exist in the supplied/current
        census. Failure to establish that evidence is an error, never an UNKNOWN_UID
        or a smaller hand-picked podium. Unless the caller supplies the authority's
        exact close-block census and time, this is not the authoritative already-emitted
        result; the finalized schema-v15 epoch log is that source.
        """
        from vidaio.competition.epoch_evidence import (
            CompetitionEvidenceError,
            build_competition_epoch_evidence,
        )

        comp = repo.get_competition(self.conn, competition_id)
        if comp is None:
            raise ResultNotReady(f"unknown competition {competition_id}")
        if comp.status is not Phase.COMPLETED:
            raise ResultNotReady(
                f"competition {competition_id} is {comp.status.value}; an economic "
                "result exists only once it is COMPLETED"
            )

        if applied_at is None:
            finalized_reader = getattr(self.chain, "finalized_block", None)
            block_time_reader = getattr(self.chain, "block_time", None)
            if not callable(finalized_reader) or not callable(block_time_reader):
                raise CompetitionEvidenceError(
                    "an economic preview requires an explicit epoch close time or a "
                    "chain adapter with finalized_block/block_time"
                )
            finalized_block = int(finalized_reader())
            applied_at = block_time_reader(finalized_block)
            if applied_at is None:
                raise CompetitionEvidenceError(
                    f"finalized block {finalized_block} has no archive-readable chain time"
                )

        if census_by_hotkey is None:
            if self.chain is None:
                raise CompetitionEvidenceError(
                    "an auditable economic result requires a current chain census"
                )
            try:
                census_by_hotkey = {
                    neuron.hotkey: MinerCensusEntry(
                        uid=neuron.uid,
                        hotkey=neuron.hotkey,
                        coldkey=neuron.coldkey,
                        ip=neuron.ip,
                    )
                    for neuron in self.chain.neurons()
                }
            except Exception as exc:
                raise CompetitionEvidenceError(
                    f"current chain census is unavailable: {exc}"
                ) from exc
        evidence = build_competition_epoch_evidence(
            self.conn,
            census_by_hotkey=census_by_hotkey,
            store=self.store,
            tokenomics=self.tokenomics,
            competition_id=competition_id,
            through_time=applied_at,
        )
        if evidence is None:
            raise CompetitionEvidenceError(
                f"competition {competition_id!r} has no payable auditable result"
            )
        return evidence.result

    def enroll_contender(
        self,
        competition_id: str,
        *,
        hotkey: str,
        repo_url: str,
        commit_sha: str,
        tree_sha: str,
        stake: float,
        now: datetime,
    ) -> int:
        """Repository-level enrollment intake (the public HTTP API arrives with
        the organic gateway; until then callers enroll through the orchestrator)."""
        with pers.txn(self.conn):
            return repo.enroll_contender(
                self.conn,
                competition_id,
                hotkey=hotkey,
                repo_url=repo_url,
                commit_sha=commit_sha,
                tree_sha=tree_sha,
                stake=stake,
                now=now,
            )

    def add_evaluation_item(
        self,
        competition_id: str,
        *,
        input_path: str | Path,
        item_index: int,
        threshold_commitment: str,
        now: datetime,
        challenge_id: str | None = None,
        length_seconds: float | None = None,
        reference_path: str | Path | None = None,
        upscale_factor: int | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> int:
        """Seed a manifest-bound evaluation item.

        The miner-visible ``input_path`` is staged as ``CHALLENGE_INPUT``.  An
        upscaling item additionally requires a distinct pristine ``reference_path``;
        it is stored as sealed ``REFERENCE_ORIGINAL`` and staged only in the trusted
        scorer pool.  Sandbox batches are constructed solely from the low-resolution
        input digest, so contender code never receives the pristine bytes.
        """
        manifest = repo.get_manifest(self.conn, competition_id)
        src = Path(input_path)
        input_ref = self.store.put_file(src, ArtifactKind.CHALLENGE_INPUT)
        digest = input_ref.digest
        input_bytes = input_ref.byte_size
        pooled = self.inputs_dir / digest
        if not pooled.exists():
            shutil.copy2(src, pooled)

        reference_digest: str | None = None
        reference_bytes: int | None = None
        if manifest.track == "upscaling":
            if reference_path is None:
                raise ValueError(
                    "upscaling evaluation item requires a pristine reference_path"
                )
            reference_src = Path(reference_path)
            reference_ref = self.store.put_file(
                reference_src, ArtifactKind.REFERENCE_ORIGINAL
            )
            reference_digest = reference_ref.digest
            reference_bytes = reference_ref.byte_size
            if reference_digest == digest:
                raise ValueError(
                    "upscaling pristine reference and miner input must be distinct"
                )
            reference_pooled = self.inputs_dir / reference_digest
            if not reference_pooled.exists():
                shutil.copy2(reference_src, reference_pooled)
            if (
                reference_ref.digest != reference_digest
                or reference_ref.byte_size != reference_bytes
            ):
                raise RuntimeError(
                    "audit store returned a reference address that does not match "
                    "the pristine bytes"
                )
        elif (
            reference_path is not None
            or upscale_factor is not None
            or target_width is not None
            or target_height is not None
        ):
            raise ValueError(
                "reference_path/upscale_factor/target geometry are valid only for "
                "upscaling competitions"
            )
        with pers.txn(self.conn):
            return repo.add_evaluation_item(
                self.conn,
                competition_id,
                item_index=item_index,
                input_sha256=digest,
                input_bytes=input_bytes,
                threshold_commitment=threshold_commitment,
                challenge_id=challenge_id or f"chal-{competition_id}",
                length_seconds=length_seconds,
                now=now,
                reference_sha256=reference_digest,
                reference_bytes=reference_bytes,
                upscale_factor=upscale_factor,
                target_width=target_width,
                target_height=target_height,
            )

    def ingest_evaluation_item(
        self,
        competition_id: str,
        *,
        input_name: str,
        item_index: int,
        threshold_commitment: str,
        now: datetime,
        challenge_id: str | None = None,
        length_seconds: float | None = None,
        reference_name: str | None = None,
        upscale_factor: int | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> int:
        """Safely ingest operator-provisioned files from ``<work_dir>/ingest``.

        Control-plane callers supply basenames, never host paths.  Each entry is
        required to be a bounded regular file inside the dedicated 0700 root and
        is copied through an ``O_NOFOLLOW`` descriptor into the trusted digest
        pool before the ordinary manifest-binding path sees it.
        """
        comp = repo.get_competition(self.conn, competition_id)
        if comp is None:
            raise KeyError(f"unknown competition {competition_id}")
        if comp.status is not Phase.SCHEDULED or comp.commitment_root is not None:
            raise ValueError(
                "evaluation items may be ingested only while SCHEDULED and before "
                "the competition commitment is anchored"
            )

        input_pooled = self._ingest_file(input_name, what="competition miner input")
        reference_pooled = (
            None
            if reference_name is None
            else self._ingest_file(
                reference_name, what="competition pristine reference"
            )
        )
        return self.add_evaluation_item(
            competition_id,
            input_path=input_pooled,
            item_index=item_index,
            threshold_commitment=threshold_commitment,
            now=now,
            challenge_id=challenge_id,
            length_seconds=length_seconds,
            reference_path=reference_pooled,
            upscale_factor=upscale_factor,
            target_width=target_width,
            target_height=target_height,
        )

    def _ingest_file(self, name: str, *, what: str) -> Path:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(
                f"{what} name must be one basename inside {self.ingest_dir}"
            )
        source = self.ingest_dir / name
        safeio.assert_within(source, self.ingest_dir, what=what)
        inspected = safeio.lstat_regular(source, what=what)
        _digest, _size = safeio.hash_into_pool(
            source,
            inspected,
            self.inputs_dir,
            max_bytes=MAX_COMPETITION_INGEST_BYTES,
            what=what,
        )
        return self.inputs_dir / _digest

    def submit_review(self, competition_id: str, **kwargs: Any) -> int:
        """Human-review passthrough (AWAITING_END_TIME window); re-ranks inside."""
        return _submit_review(self.conn, competition_id, **kwargs)


def _claim_age_seconds(claim: Mapping[str, Any], at: datetime) -> float:
    """Seconds since an anchor claim was taken, from the service clock.

    An unparseable timestamp is treated as age 0 — i.e. the claim is FRESH and
    blocks everything. Fail closed: guessing "stale" on a bad timestamp would
    reopen the double-anchor window this whole protocol exists to close.
    """
    raw = claim.get("claimed_at")
    if not isinstance(raw, str):
        return 0.0
    try:
        claimed = datetime.fromisoformat(raw)
    except ValueError:
        return 0.0
    if claimed.tzinfo is None:
        claimed = claimed.replace(tzinfo=timezone.utc)
    return max(0.0, (at - claimed).total_seconds())


def _assert_packet_identity_not_reserved(packet_bytes: bytes) -> None:
    """Refuse a WORKER packet stamped with the orchestrator-zero identity.

    review round 2, new-3: an orchestrator-minted zero must be distinguishable from
    a measured packet, which requires the reserved namespace to be exclusively
    ours in BOTH directions. A worker whose /healthz advertises an honest identity
    could still stamp `orchestrator-zero/1+...` into the packet it returns; that
    packet would then read as an orchestrator gate-failure record while carrying a
    score somebody else chose. Refused as an INFRA halt — never persisted.

    Malformed packet bytes are NOT this function's problem (record_item_score
    rejects them with a typed error); it only inspects a readable identity.
    """
    import json

    try:
        identity = json.loads(packet_bytes.decode("utf-8")).get("scorer_version")
    except Exception:  # noqa: BLE001 - packet validity is repository.record_item_score's job
        return
    if isinstance(identity, str):
        assert_not_reserved(identity, what="the scoring worker's returned packet")


def _submission_reject_reason(exc: BaseException) -> str:
    """The rejection reason recorded when a submission is unusable BY ITS OWN DOING.

    Only ever called after `classify_failure` said CONTENDER, so every branch here
    describes the submission, never our infrastructure.
    """
    from vidaio.competition.orchestrator.failures import unwrap
    from vidaio.competition.runners.errors import OversizeOutputError, UnsafePathError

    root = unwrap(exc)
    if isinstance(root, UnsafePathError):
        return f"unsafe submission tree: {root}"[:500]
    if isinstance(root, OversizeOutputError):
        return f"oversize submission tree: {root}"[:500]
    return f"submission unusable ({fault_code(exc)}): {root}"[:500]


async def _fault_as_result(awaitable: Any) -> Any:
    """Turn a CONTENDER fault into a RESULT so retry budgets are not spent on it.

    A solution's `exit 1`, its blown timeout or its oversize output is a verdict,
    not a transient failure: retrying it burns sandbox time to reach the same
    answer. Infra failures still raise and are still retried.
    """
    from vidaio.competition.runners.errors import ContenderFaultError

    try:
        return await awaitable
    except ContenderFaultError as exc:
        return exc


def _unwrap_fault(outcome: Any) -> Any:
    from vidaio.competition.runners.errors import ContenderFaultError

    if isinstance(outcome, ContenderFaultError):
        raise outcome
    return outcome


async def _scoring_fault_as_result(awaitable: Any) -> Any:
    """Same idea for the scorer: a 4xx verdict on the contender's bytes is
    deterministic, so it must not consume the transport retry budget."""
    try:
        return await awaitable
    except Exception as exc:  # noqa: BLE001 - re-raised unless contender-attributable
        if classify_failure(exc) is Fault.CONTENDER:
            return exc
        raise


def _unwrap_scoring_fault(outcome: Any) -> Any:
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome


def _probe_json(report: Any) -> str:
    import json

    return json.dumps(
        {
            "network_blocked": report.network_blocked,
            "secrets_absent": report.secrets_absent,
            "reference_mounts_absent": report.reference_mounts_absent,
            "index_leak_absent": report.index_leak_absent,
            "passed": report.passed,
            "details": report.details,
        },
        sort_keys=True,
    )
