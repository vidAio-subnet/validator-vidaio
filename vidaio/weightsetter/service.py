"""Weight-setter service: compose, submit, and PUBLISH the exact weight vector.

Cadence (spec design spec §01): attempt a weight set every `attempt_interval_seconds`
(default 72 min) against the chain's tempo gate — a tempo rejection is a normal
reschedule, never an error. Composition is pure tokenomics.build_weight_vector
over injected MinerSnapshots (SnapshotProvider), the persisted crown, and the
latest ingested CompetitionResult; a validation failure (UnknownCompetitionUid)
is logged CRITICAL and the round is SKIPPED — a vector is never submitted after
a failed validation.

Publication: after a
SUCCESSFUL set_weights, the exact submitted vector is serialized as canonical
JSON and stored as a WEIGHT_VECTOR artifact; a PublicationRecord over
{score-packet merkle root, weight-vector digest} is recorded in the
CommitmentLedger (pending_chain) and anchored via ChainAdapter.anchor_commitment
(-> anchored), so third parties can reproduce the chain weights from the audit
store alone.

Empty-competition sentinel: audit.merkle_root requires >= 1 leaf. When no score
packets back this publication, the CONVENTION is the merkle root over the single
leaf sha256(EMPTY_SCORE_PACKET_MARKER) — see EMPTY_SCORE_PACKET_SET_ROOT. The
marker is versioned and public, so a third party can both recompute the sentinel
and prove that no real packet set could collide with it. It is a SENTINEL, not a
default: a PublicationInputs provider that can enumerate real packet digests (the
validator's `ScorePacketEvidence`) makes real publications carry the real merkle
set.

Durability: `set_weights` is a non-idempotent chain write behind a
retry envelope, so every attempt writes an INTENT record first
(vidaio.weightsetter.intents) carrying the exact vector, its digest, the backing
packet digests (or an exact authority epoch recovery key when their best-effort
copy failed) and the attempt block. A retry after an AMBIGUOUS attempt
(timeout / transport error) consults the chain before re-submitting instead of
blindly re-writing; a tempo rejection that follows an ambiguous attempt is
reconciled as an acceptance rather than recorded as a failure. Publication is
driven from the intent row, and `reconcile()` runs at startup and on every loop
iteration to finish half-done publications and re-drive pending anchors — the
recovery loop that was previously described but missing.

Round 2 of an internal review — WHAT THE CHAIN CHECK MAY CONCLUDE. The check used to read
the adapter's PRE-WRITE cached snapshot and answer a plain bool, so "I looked at
a stale snapshot" was indistinguishable from "the chain does not have it". That
false negative abandoned intents whose weights WERE live on chain, permanently
unpublished. The answer is now tri-state (`ChainConfirmation`) and the snapshot is
always REFRESHED first.

Round 3 of an internal review — WHAT THE CHAIN CHECK IS ASKED. Freshness was not enough,
because the question itself was wrong: "has our hotkey's `last_update` reached the
attempt block?" proves that SOME write of ours landed at some point. It does not
identify a vector, so it confirmed intents that had never reached the chain — an
intent that timed out, hit the tempo gate on retry and was published anyway; an
intent silently "confirmed" months later by a DIFFERENT intent's success. The
check now compares THIS INTENT'S OWN VECTOR against what the chain reports, via
the optional `ChainAdapter` read `submitted_weights(hotkey) -> SubmittedWeights`
(vidaio.chain.SubmittedWeightsReader). Both vectors are put on the chain's u16
grid first (intents.quantize_weights), so a chain that stores integers and a
submission of floats still compare equal.

    CONFIRMED  the chain reports OUR vector (same uids, same u16 weights within
               one step), recorded at/after this attempt's block when the adapter
               dates it, and no other unsettled intent carrying an identical
               vector could equally be its author
    DENIED     the chain positively reports NO weights for our hotkey, or its
               current vector predates this attempt (whether it differs from ours
               or matches an earlier attempt's identical one) — in both cases
               nothing of ours landed since we tried
    UNKNOWN    everything else: no hotkey configured, no fresh snapshot, an
               adapter that cannot report vectors AT ALL, a failed read, a
               different vector that postdates our attempt (ours may have landed
               and been overwritten), or an identical-vector ambiguity

Only DENIED may settle an intent as `abandoned`, and only through reconcile()'s
age-bounded path (`abandon_denied_intent_after_seconds`, logged CRITICAL with its
evidence). A synchronous rejection may bury an intent on the spot ONLY when no
attempt was ambiguous — a retry's rejection describes the retry, never the write
before it. UNKNOWN never abandons anything: the intent stays `pending` and every
later reconciliation pass re-checks it, because a vector that may be live on chain
must remain publishable. Only DENIED freely permits a re-submission; under UNKNOWN
at most one further probing attempt is made, and only because the chain's own
tempo gate cannot accept a second write inside the same window.

THE PUBLICATION RULE (the point of the whole fix — we may only publish what we
can show landed):

    CONFIRMED  publish: store the vector, ledger the PublicationRecord, anchor it
    UNKNOWN    HOLD. Nothing is stored, ledgered or anchored; the intent stays
               pending and re-checked, and becomes publishable the moment the
               chain can show its vector
    DENIED     never published — and after the age bound, never publishable

"CONFIRMED" includes the ordinary path where `set_weights` itself returns success:
that is the chain answering about this exact call, the most direct evidence there
is. What no longer counts is an inference — a tempo rejection, a neighbouring
intent's acceptance, or a block number.

Chain-state gate: a stale or unavailable metagraph snapshot SKIPS the
attempt with a structured reason. An empty or partial vector is never submitted
just because the chain read failed.

Competition reward activity is evaluated against the chain-bound epoch close time.
Operational completion clocks and an independently guessed cycle clock never enter
weight composition.

All timestamps come from an injected clock (tz-aware by default); no logic path
reads the wall clock directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import inspect
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from prometheus_client import Counter, Gauge

from vidaio.audit import (
    ArtifactKind,
    AuditStore,
    CommitmentLedger,
    CommitmentPayload,
    CommitmentStatus,
    PublicationRecord,
    build_publication_record,
    canonical_json_bytes,
    merkle_root,
    sha256_hex,
)
from vidaio.chain.adapter import ChainAdapter, resolve_burn_uid
from vidaio.chain.anchor_writer import anchor_writer_lock
from vidaio.chain.factory import ChainConfig
from vidaio.core import log_fields, section, with_timeout
from vidaio.core.db import connect
from vidaio.services.base import BaseService
from vidaio.services.commitment_capacity import (
    CommitmentCapacityError,
    require_commitment_capacity,
)
from vidaio.tokenomics import (
    MinerSnapshot,
    TokenomicsConfig,
    build_weight_vector,
)
from vidaio.weightsetter import crown_store, intents
from vidaio.weightsetter.config import WeightSetterConfig
from vidaio.weightsetter.shared_snapshot import (
    EpochInputs,
    SharedSnapshotError,
    SnapshotDigestMismatch,
)

#: Versioned domain tag of the published weight-vector document — bump on any
#: change to weight_vector_document's shape (it invalidates recorded digests).
WEIGHT_VECTOR_DOMAIN = "vidaio.weight_vector.v1"

#: Empty-competition sentinel (public convention, see module docstring): when a
#: publication has NO score packets, its merkle root is the root over the single
#: leaf sha256(EMPTY_SCORE_PACKET_MARKER) — merkle_root requires >= 1 leaf.
EMPTY_SCORE_PACKET_MARKER = b"vidaio.weightsetter.no-score-packets.v1"
EMPTY_SCORE_PACKET_SET_ROOT = merkle_root([sha256_hex(EMPTY_SCORE_PACKET_MARKER)])


def weight_vector_document(
    weights: dict[int, float], *, version_key: int, block: int
) -> dict[str, Any]:
    """The published form of one submitted vector — EXACTLY what went to the chain.

    `block` is the chain-reported block of the accepted set_weights; uids are JSON
    object keys (strings) mapping to the untouched float weights. Serialized via
    audit.canonical_json_bytes, whose digest is the PublicationRecord's
    weight_vector_digest.
    """
    return {
        "domain": WEIGHT_VECTOR_DOMAIN,
        "version_key": version_key,
        "block": block,
        "weights": {str(uid): weights[uid] for uid in sorted(weights)},
    }


@runtime_checkable
class SnapshotProvider(Protocol):
    """Source of the miner snapshots one weight composition runs over.

    Two implementations (the project design record rule 8, selected by config):
    - LOCAL — the validator's `miner_manager` (per-validator EWMA scores, retention
      windows, exclusions); report-mode / dryrun / third-party recompute.
    - SHARED — `vidaio.weightsetter.shared_snapshot.SharedSnapshotProvider`, which
      mirrors the Scoring Authority's finalized epoch log so every validator
      converges on the identical vector (build-wave 5).

    OPTIONAL, feature-detected extension — a provider MAY also implement

        epoch_inputs() -> EpochInputs | None

    handing the weight-setter the epoch log's `RewardWindowState` / `CompetitionResult` /
    `burn_uid` and its STATED u16 vector, so composition runs over the shared reward
    state (not local persistence) and the re-derived vector is cross-checked
    against the log's. The local provider does not implement it (returns None
    semantics), keeping the existing crown-store path unchanged.
    """

    def miner_snapshots(self) -> Sequence[MinerSnapshot]: ...


@runtime_checkable
class PublicationInputs(Protocol):
    """Audit inputs for one publication: the score packets backing these weights.

    `score_packet_digests` returns the sha256 hex digests of every score packet
    backing the weights being published. An empty sequence means no packets back
    this publication — it then uses the documented EMPTY_SCORE_PACKET_SET_ROOT
    sentinel.

    OPTIONAL, feature-detected: a provider may also implement

        recent_packet_digests(since: datetime | None) -> Sequence[str]

    which the weight-setter prefers, passing the watermark of the last published
    intent (falling back to `publication_lookback_seconds`) so each publication
    commits to the packets produced SINCE the previous one rather than to a
    provider-chosen window. `vidaio.validator.ScorePacketEvidence` implements
    both.

    The production shared provider additionally implements

        committed_packet_digests() -> Sequence[str]
        score_packet_digests_for_epoch(epoch_id, *, expected_snapshot_digest) -> Sequence[str]

    The first is a no-I/O copy from the already-authenticated log and is best-effort
    before submission. The second independently re-resolves that exact historical
    log after submission and validates its packet Merkle root. Neither result is an
    input to the authority vector's chain write.
    """

    def score_packet_digests(self) -> Sequence[str]: ...


class ChainConfirmation(enum.Enum):
    """What a FRESH chain read can say about ONE attempted weight write (#10).

    The distinction UNKNOWN vs DENIED is half the fix: they used to be the same
    `False`, so an unreadable or stale snapshot silently became "the chain does
    not have our weights" — and abandoned, unpublished, a vector that was live.
    Round 3 is the other half: every verdict is about THIS INTENT'S OWN VECTOR,
    read back off the chain, never about block bookkeeping.
    """

    #: The chain reports THIS intent's vector, recorded at/after its attempt.
    CONFIRMED = "confirmed"
    #: The chain positively does not hold it (no record at all, or its newest
    #: record is a different vector that predates this attempt).
    DENIED = "denied"
    #: Anything that is not one of those two positive answers.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class _ChainEvidence:
    """A confirmation verdict WITH the evidence that produced it.

    `set_block` is the block the chain says its current vector was recorded at
    (None when unreportable) — it becomes the intent's `accepted_block`, so a
    published document carries the chain's own block rather than ours.

    `reported_weights` is the EXACT vector read back off the chain on a CONFIRMED
    verdict — the `SubmittedWeights.weights` the adapter returned,
    the chain's own u16 as it stores them. A recovery confirmation must reconcile the
    stored intent to THIS before publication reads the row, so an intent confirmed on
    restart anchors chain state rather than its pre-quantization float. None when the
    verdict is not CONFIRMED, or when confirmation rested on block bookkeeping and no
    vector was read (then there is nothing to persist and the intent's own,
    match-proven vector stands).
    """

    verdict: ChainConfirmation
    set_block: int | None = None
    detail: str = ""
    reported_weights: dict[int, float] | None = None


@dataclass(frozen=True)
class _Submission:
    """Outcome of the whole (retried) set_weights attempt, with its provenance."""

    success: bool
    block: int
    #: how the outcome was established — recorded on the intent row
    resolution: str
    message: str = ""
    #: True when at least one attempt's fate was unknowable (timeout / transport)
    ambiguous: bool = False
    tempo_gated: bool = False
    #: The EXACT vector the chain holds for this acceptance — what the durable intent
    #: row is atomically rewritten to (accept_with_vector) so publication/anchoring
    #: commits chain state byte-for-byte. On a DIRECT acceptance it is the u16 the
    #: adapter submitted (binding-verified + quantized). Target churn is rejected before
    #: a write, so a direct success retains the exact uid set. On a RECONCILED acceptance
    #: (an ambiguous write later confirmed by a chain
    #: READ, an internal review) it is the vector that read reported back, carried from
    #: `_ChainEvidence.reported_weights` so a reconciled success also anchors chain
    #: state, not the pre-quantization float. None when a reconciliation confirmed by
    #: block bookkeeping alone could read no vector: then the intent's own,
    #: match-proven vector stands. Empty/None both leave the durable intent untouched.
    submitted: dict[int, float] | None = None
    #: For an UNSUCCESSFUL attempt: what a fresh, VECTOR-SPECIFIC chain read said.
    #: DENIED settles the intent immediately ONLY when nothing was ambiguous (a
    #: synchronous rejection of the single write we issued). After an ambiguous
    #: write, even DENIED goes through reconcile()'s age-bounded CRITICAL path,
    #: and UNKNOWN never settles anything: the intent must stay publishable.
    confirmation: ChainConfirmation = ChainConfirmation.DENIED


def _is_tempo(message: str) -> bool:
    return "tempo" in message.lower()


def _accepts_cutoff(callable_: Callable[..., Any]) -> bool:
    """Does `recent_packet_digests` take a `since` cutoff? (feature detection)

    By SIGNATURE, not by catching TypeError from the call: a TypeError raised
    inside a provider's own body would otherwise be misread as "this provider
    takes no cutoff", silently widening the evidence window.
    """
    try:
        parameters = inspect.signature(callable_).parameters
    except (TypeError, ValueError):
        return True  # unintrospectable (a C callable / mock): assume the contract
    return any(
        p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
        for p in parameters.values()
    )


def _connection_factory(
    conn: sqlite3.Connection,
) -> Callable[[], sqlite3.Connection] | None:
    """A factory opening a NEW handle on the same database file, or None.

    sqlite3 connections are not thread-shareable, so health checks answered on
    the HealthServer's thread need their own. ':memory:' databases
    have no reopenable file, so they get no DB health check at all rather than a
    check that lies about a different, empty database.
    """
    for row in conn.execute("PRAGMA database_list").fetchall():
        if row[1] == "main" and row[2]:
            path = str(row[2])
            return lambda: connect(path)
    return None


class WeightSetter(BaseService):
    """The weight-setter loop: compose -> set_weights -> publish, every interval."""

    name = "weight-setter"

    def __init__(
        self,
        raw_config: dict[str, Any],
        *,
        chain: ChainAdapter,
        snapshots: SnapshotProvider,
        conn: sqlite3.Connection,
        conn_factory: Callable[[], sqlite3.Connection] | None = None,
        store: AuditStore | None = None,
        ledger: CommitmentLedger | None = None,
        publication_inputs: PublicationInputs | None = None,
        clock: Callable[[], datetime] | None = None,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        config = section(raw_config, "weightsetter", WeightSetterConfig)
        super().__init__(raw_config, metrics_port=config.metrics_port)
        self.config = config
        chain_config = section(raw_config, "chain", ChainConfig)
        self._chain_mode = chain_config.mode
        self.tokenomics = section(raw_config, "tokenomics", TokenomicsConfig)
        if config.publication_enabled and (store is None or ledger is None):
            raise ValueError(
                "publication_enabled requires an audit store AND a commitment ledger —"
                " disable weightsetter.publication_enabled only in dev/test"
            )
        self._chain = chain
        self._snapshots = snapshots
        self._conn = conn
        self._store = store
        self._ledger = ledger
        self._publication_inputs = publication_inputs
        # Publication commitments are signed by this thin validator, not the
        # authority. Keep capacity-check + write in the same per-wallet lane as
        # the adapter's nested lock so another local writer cannot invalidate the
        # precheck between those operations.
        self._anchor_netuid = chain_config.netuid
        self._anchor_hotkey = chain_config.validator_hotkey
        self._anchor_writer_lock_path = chain_config.anchor_writer_lock_path
        self._anchor_writer_lock_timeout_seconds = (
            chain_config.anchor_writer_lock_timeout_seconds
        )
        self._publication_task: asyncio.Task[bool] | None = None
        self._publication_task_intent_id: int | None = None
        self._clock: Callable[[], datetime] = clock or (
            lambda: datetime.now(timezone.utc)
        )
        self._monotonic_clock = time.monotonic
        #: wall clock (epoch seconds) for the chain adapter's own freshness surface
        self._wall_clock = wall_clock
        crown_store.migrate(conn)

        # Health checks are served on the HealthServer's THREAD and must not touch
        # the loop's sqlite3 handle: give them their own connection.
        self._conn_factory = (
            conn_factory if conn_factory is not None else _connection_factory(conn)
        )
        self._thread_local = threading.local()

        self._last_success_at: datetime | None = None
        self._last_refresh_at: float | None = None
        self._age_anchor = self._clock()  # health/gauge anchor before the first success

        registry = self.health.registry
        self.metric_attempts = Counter(
            "weightsetter_attempts_total",
            "Weight-set attempts started",
            registry=registry,
        )
        self.metric_successes = Counter(
            "weightsetter_successes_total",
            "Chain-accepted set_weights calls",
            registry=registry,
        )
        self.metric_tempo_gated = Counter(
            "weightsetter_tempo_gated_total",
            "Attempts skipped by the chain tempo gate (rescheduled, not errors)",
            registry=registry,
        )
        self.metric_chain_failures = Counter(
            "weightsetter_chain_failures_total",
            "Chain writes that failed for non-tempo reasons (after retries)",
            registry=registry,
        )
        self.metric_validation_failures = Counter(
            "weightsetter_validation_failures_total",
            "Rounds skipped because weight composition failed validation",
            registry=registry,
        )
        self.metric_publications = Counter(
            "weightsetter_publications_total",
            "Weight vectors published and anchored (ledger advanced to anchored)",
            registry=registry,
        )
        self.metric_publication_errors = Counter(
            "weightsetter_publication_errors_total",
            "Accepted weight vectors whose post-submit publication raised; the"
            " durable accepted intent remains queued for reconciliation",
            registry=registry,
        )
        self.metric_loop_errors = Counter(
            "weightsetter_loop_errors_total",
            "Unexpected attempt-loop exceptions recovered without terminating the"
            " emissions-critical process",
            registry=registry,
        )
        self.metric_last_success_age = Gauge(
            "weightsetter_last_success_age_seconds",
            "Seconds since the last chain-accepted set_weights (service start before any)",
            registry=registry,
        )
        self.metric_reconciled = Counter(
            "weightsetter_reconciled_total",
            "Ambiguous set_weights attempts reconciled as chain acceptances"
            " (a lost response, not a failed write), by how it was established",
            ["resolution"],
            registry=registry,
        )
        self.metric_redriven = Counter(
            "weightsetter_redriven_publications_total",
            "Publications/anchors completed by the reconciliation pass rather than"
            " by the attempt that created them",
            registry=registry,
        )
        self.metric_chain_state_skips = Counter(
            "weightsetter_chain_state_skips_total",
            "Attempts skipped because the chain snapshot was stale/unavailable"
            " (NOTHING was submitted — never an empty or partial vector)",
            ["reason"],
            registry=registry,
        )
        self.metric_publication_input_failures = Counter(
            "weightsetter_publication_input_failures_total",
            "Score-packet evidence reads that failed (never flattened into an empty"
            " packet set; shared authority vectors still submit and queue recovery)",
            registry=registry,
        )
        self.metric_reward_read_model_failures = Counter(
            "weightsetter_reward_read_model_failures_total",
            "Authenticated authority reward/result read-model mirrors that failed;"
            " reporting is degraded but the authority vector still submits",
            registry=registry,
        )
        self.metric_unresolved_intents = Counter(
            "weightsetter_unresolved_intents_total",
            "Attempts whose chain fate is UNKNOWN: the intent stays pending and is"
            " re-checked, never abandoned on a guess",
            registry=registry,
        )
        self.metric_pending_intents = Gauge(
            "weightsetter_pending_intents",
            "Weight intents whose chain fate is still undecided. A number that"
            " does not come back down means the chain cannot be read at all",
            registry=registry,
        )
        self.metric_abandoned = Counter(
            "weightsetter_abandoned_intents_total",
            "Intents settled as abandoned — only ever on a POSITIVE chain denial,"
            " by resolution",
            ["resolution"],
            registry=registry,
        )
        self.metric_snapshot_tampered = Counter(
            "weightsetter_snapshot_tampered_total",
            "Shared-snapshot attempts REFUSED because the tamper-evidence chain broke"
            " (sha256(bytes) != pointer digest != on-chain anchor). NOTHING submitted"
            " — a mutated epoch log is never quantized into a vector (wave 5)",
            registry=registry,
        )
        self.metric_convergence = Gauge(
            "vidaio_weightsetter_convergence",
            "Fraction of observed peer validators whose on-chain vector matches"
            " ours at this epoch (observe-only health signal; 1.0 on an honest"
            " network where every peer validator reads the same finalized epoch)",
            registry=registry,
        )
        self.metric_last_success_age.set_function(self._last_success_age_seconds)
        self.health.register_check(
            "last_success_age",
            lambda: (
                self._last_success_age_seconds() <= config.max_last_success_age_seconds
            ),
        )
        if self._conn_factory is not None:
            self.health.register_check("db", self._db_reachable)

    # -- health ----------------------------------------------------------------

    def _health_conn(self) -> sqlite3.Connection:
        """A connection owned by the CALLING thread."""
        assert self._conn_factory is not None
        conn = getattr(self._thread_local, "conn", None)
        if conn is None:
            conn = self._conn_factory()
            self._thread_local.conn = conn
        return conn

    def _db_reachable(self) -> bool:
        try:
            self._health_conn().execute(
                "SELECT COUNT(*) FROM weight_intents"
            ).fetchone()
            return True
        except Exception:
            self._thread_local.conn = None
            return False

    def _sync_pending_intents_metric(self) -> None:
        """Make the gauge reflect the durable ledger at the mutation boundary.

        Reconciliation already refreshes this gauge, but a newly-created intent
        can remain pending for the entire 72-minute cadence after an ambiguous
        write.  Updating only on the *next* reconciliation made the live metric
        report zero throughout that window.  The database remains authoritative;
        metrics failure is observation-only and never interrupts weight setting.
        """
        try:
            self.metric_pending_intents.set(len(intents.unsettled_intents(self._conn)))
        except Exception as exc:
            self.log.warning(
                "could not refresh the pending-intent gauge from its durable ledger",
                extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
            )

    # -- time ------------------------------------------------------------------

    def _last_success_age_seconds(self) -> float:
        anchor = self._last_success_at or self._age_anchor
        return max((self._clock() - anchor).total_seconds(), 0.0)

    def _iso_now(self) -> str:
        """Clock instant as tz-aware ISO-8601 (the ledger rejects naive stamps)."""
        now = self._clock()
        if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.isoformat()

    # -- main loop ---------------------------------------------------------------

    async def run(self) -> None:
        try:
            await self._run_until_stopped()
        finally:
            # A timed-out publication is deliberately detached so it cannot hold
            # the emissions cadence. On orderly shutdown, retain ownership and let
            # its non-cancellable chain worker finish before the event loop closes.
            await self._finish_publication_task()

    async def _run_until_stopped(self) -> None:
        # Startup half of the recovery loop: finish any publication or
        # anchor a previous process left owed BEFORE composing a new vector. The
        # refresh comes first so the reconciliation's chain checks read a real
        # snapshot rather than an empty startup one.
        try:
            await self._refresh_chain_async()
            # Resolve ambiguous chain writes before creating another intent, but
            # never put S3/publication work in front of the first scheduled
            # weight attempt. Accepted intents are drained after the attempt.
            await self.reconcile(publish_accepted=False)
        except Exception:
            self.log.exception("startup reconciliation failed; continuing")
        while not self.stopping.is_set():
            started = self._monotonic_clock()
            try:
                await self.attempt_once()
            except Exception:
                # A malformed local row, storage bug, or other unexpected Python
                # failure must not kill the critical process and freeze all later
                # scheduled writes. Typed operational HOLDs remain inside
                # attempt_once; this is the final process-liveness boundary.
                self.metric_loop_errors.inc()
                self.log.exception(
                    "weight-setting attempt raised unexpectedly; the process remains"
                    " live and will retry on the fixed cadence"
                )
            try:
                await self._drain_one_accepted_publication()
            except Exception:
                self.metric_publication_errors.inc()
                self.log.exception(
                    "background publication drain failed; weight cadence remains"
                    " unaffected and the accepted intent stays durable"
                )
            elapsed = max(self._monotonic_clock() - started, 0.0)
            await self._wait_for_next_attempt(
                max(self.config.attempt_interval_seconds - elapsed, 0.0)
            )

    async def _wait_for_next_attempt(self, delay_seconds: float) -> None:
        """Wait interruptibly while preserving a fixed start-to-start cadence."""
        if delay_seconds <= 0.0:
            await asyncio.sleep(0)
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.stopping.wait(), timeout=delay_seconds)

    # -- chain state -------------------------------------------------

    def _refresh_chain(self) -> bool:
        """Refresh the snapshot. A FAILED refresh is not recorded as a success."""
        try:
            self._chain.refresh()
        except Exception as exc:
            self.log.warning(
                "chain refresh failed; snapshot NOT marked fresh",
                extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
            )
            return False
        self._last_refresh_at = self._wall_clock()
        return True

    async def _refresh_chain_async(self) -> bool:
        """Run the synchronous SDK refresh off the service event loop.

        The live adapter's coherent refresh can perform a metagraph, head, uid,
        LastUpdate, and rate-limit sequence. Each RPC is bounded, but executing
        that sequence inline used to freeze health/stop handling for the sum of
        those bounds. The chain adapter remains synchronous for non-async users;
        this service never performs the blocking refresh on its event-loop thread.
        """
        return await asyncio.to_thread(self._refresh_chain)

    def _adapter_reports_fresh(self, max_age: float) -> bool | None:
        """Feature-detect the adapter's optional `has_fresh_snapshot`. None = absent."""
        probe = getattr(self._chain, "has_fresh_snapshot", None)
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
                    extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
                )
                return None
        return None

    def _chain_state_reason(self) -> str | None:
        """None when the snapshot may be composed over; else a structured reason."""
        max_age = self.config.max_chain_snapshot_age_seconds
        if max_age <= 0:
            return None
        adapter = self._adapter_reports_fresh(max_age)
        if adapter is False:
            return "chain_snapshot_stale"
        if adapter is None:
            if self._last_refresh_at is None:
                return "chain_snapshot_never_refreshed"
            if self._wall_clock() - self._last_refresh_at > max_age:
                return "chain_snapshot_stale"
        return None

    def _reported_set_block(self, hotkey: str) -> int | None:
        """When the chain last recorded a vector for `hotkey`, if it will say.

        Corroboration ONLY — this number can never confirm anything by itself
        (that was the round-3 bug). It is used to place the reported vector in
        time: before our attempt (so our write did not land) or after it, and to
        tell two attempts carrying an identical vector apart.

        Prefers an adapter-provided `last_weight_block(hotkey)`, else our own
        neuron's `last_update`. None when nothing readable says.
        """
        getter = getattr(self._chain, "last_weight_block", None)
        if callable(getter):
            try:
                last = getter(hotkey)
            except Exception as exc:
                self.log.warning(
                    "last_weight_block lookup failed",
                    extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
                )
            else:
                return None if last is None else int(last)
        try:
            neurons = self._chain.neurons()
        except Exception as exc:
            self.log.warning(
                "chain neuron read failed while dating an on-chain weight vector",
                extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
            )
            return None
        for neuron in neurons:
            if neuron.hotkey == hotkey:
                return int(neuron.last_update)
        return None

    def _identical_twin_exists(
        self,
        *,
        intent_id: int | None,
        reported: dict[int, float],
        attempt_block: int,
        set_block: int | None,
    ) -> bool:
        """Could ANOTHER intent be the one whose vector is on chain?

        Rule 3 of the round-3 fix: a later intent's success must never confirm an
        earlier one. When the chain-reported vector could equally be another
        intent's, the intents are indistinguishable on chain, so the only
        tie-breaker is time: the attempt that sits CLOSEST BELOW the chain's
        set-block is the plausible author. Any other candidate intent with an
        attempt block in [ours, set_block] is an equally good candidate — so
        nobody is confirmed. Already-settled intents count too: the twin that
        published this vector last cycle must not hand a free confirmation to an
        older attempt.

        Round-4 an internal review: "could equally be another intent's" is decided by the
        SAME equivalence that just matched the chain's vector to ours —
        `weights_match(reported, other)`, tolerance included — never by exact
        fingerprint equality. A stricter twin rule than the match rule is a hole:
        an intent one u16 step away from ours matches the chain report exactly
        like we do, yet compares as a different fingerprint, so its landing used
        to CONFIRM (and publish) our vector, which never landed. Whenever the
        chain-reported vector matches more than one candidate intent under the
        match tolerance, the verdict is UNKNOWN for all of them; block dating may
        disambiguate only when it positively can (below).

        With no set-block to compare against, ANY twin makes the answer a guess.
        A failed lookup also counts as a twin: preferring UNKNOWN costs a re-check,
        preferring CONFIRMED publishes a vector we cannot show landed.
        """
        try:
            rows = intents.other_intents(
                self._conn, exclude_id=intent_id, max_attempt_block=set_block
            )
        except Exception as exc:
            self.log.warning(
                "could not read the intent ledger while checking for an identical"
                " twin — treating the confirmation as UNKNOWN",
                extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
            )
            return True
        for row in rows:
            if intent_id is not None and int(row["id"]) == intent_id:
                continue
            other_block = int(row["attempt_block"])
            try:
                # The ONE equivalence relation: the same
                # tolerance-based match that tied the chain report to OUR vector
                # decides whether it ties to this candidate's too.
                same_vector = intents.weights_match(reported, intents.load_vector(row))
            except Exception:
                continue  # an unreadable row is not evidence of a twin
            if not same_vector:
                continue
            if set_block is None or other_block >= attempt_block:
                return True
        return False

    async def _chain_evidence(
        self,
        *,
        vector: dict[int, float],
        attempt_block: int,
        intent_id: int | None = None,
    ) -> _ChainEvidence:
        """Does the chain hold THIS INTENT'S OWN VECTOR?

        The old check asked whether our hotkey's `last_update` had reached the
        attempt block. That proves somebody's write landed at some point — it says
        nothing about WHICH vector, so an intent that never reached the chain was
        confirmed (and published) by a later, different intent's success. The
        question is now answered by reading the vector back:

            CONFIRMED  the chain reports a vector that matches ours (same uids,
                       same u16 weights) and, when it will say, recorded it at or
                       after our attempt block — and no identical twin intent
                       could equally be its author.
            DENIED     the chain positively holds NO weights for us, or the
                       vector it holds PREDATES this attempt (whether it differs
                       from ours or is an earlier attempt's identical one) — in
                       both cases nothing of ours landed since we tried.
            UNKNOWN    everything else: no hotkey, no fresh snapshot, an adapter
                       that cannot report vectors, a failed read, a differing
                       vector that POSTDATES our attempt (ours may have landed
                       and been overwritten), or an identical-twin ambiguity.

        ALWAYS refreshes first: the question is about the state AFTER our write.
        """
        hotkey = self.config.validator_hotkey
        if not hotkey:
            self.log.warning(
                "no weightsetter.validator_hotkey is configured, so the chain"
                " cannot be asked whether an attempted write landed — the answer"
                " is UNKNOWN, and an unknown intent is never abandoned",
            )
            return _ChainEvidence(
                ChainConfirmation.UNKNOWN, detail="no_validator_hotkey"
            )
        # Freshness first: the pre-write snapshot answers a
        # question about the pre-write world.
        refreshed = await self._refresh_chain_async()
        reason = self._chain_state_reason()
        if not refreshed or reason is not None:
            self.log.warning(
                "cannot obtain a FRESH post-write chain snapshot; the fate of this"
                " weight write is UNKNOWN (deliberately not read as a denial)",
                extra=log_fields(
                    attempt_block=attempt_block,
                    refreshed=refreshed,
                    reason=reason or "",
                ),
            )
            return _ChainEvidence(
                ChainConfirmation.UNKNOWN, detail=reason or "chain_refresh_failed"
            )
        reader = getattr(self._chain, "submitted_weights", None)
        if not callable(reader):
            # An adapter that cannot report the vector cannot decide ANYTHING
            # here — not even a denial. Block-only evidence is what published
            # unconfirmed vectors, and it is not accepted as a substitute.
            self.log.warning(
                "this chain adapter cannot report the weight vector it holds"
                " (no submitted_weights), so no weight write can ever be shown to"
                " have landed: every intent stays UNKNOWN and unpublished until an"
                " adapter that can read vectors back is wired in",
                extra=log_fields(attempt_block=attempt_block, hotkey=hotkey),
            )
            return _ChainEvidence(
                ChainConfirmation.UNKNOWN, detail="adapter_cannot_read_weights"
            )
        try:
            report = await asyncio.to_thread(reader, hotkey)
            if report is None:
                # A POSITIVE answer: this hotkey has no weight record at all, so
                # no vector of ours — this one included — has ever landed.
                return _ChainEvidence(
                    ChainConfirmation.DENIED, detail="chain_reports_no_weights"
                )
            reported = {
                int(uid): float(w) for uid, w in dict(report.weights or {}).items()
            }
            set_block = None if report.block is None else int(report.block)
        except Exception as exc:
            # Includes a report we cannot decode: an adapter answering nonsense is
            # a failed read, never a denial.
            self.log.warning(
                "reading our on-chain weight vector failed; the fate of this write"
                " is UNKNOWN (a failed read is never a denial)",
                extra=log_fields(
                    attempt_block=attempt_block, error=f"{type(exc).__name__}: {exc}"
                ),
            )
            return _ChainEvidence(
                ChainConfirmation.UNKNOWN, detail="weights_read_failed"
            )
        if set_block is None:
            # The adapter reported a vector but cannot date it: fall back to the
            # block surfaces. Corroboration only — it can place the vector in
            # time, never confirm one on its own.
            set_block = await asyncio.to_thread(self._reported_set_block, hotkey)
        if intents.weights_match(reported, vector):
            if set_block is not None and set_block < attempt_block:
                # Our own vector IS on chain — but it was recorded before we even
                # attempted, so it is an EARLIER attempt's landing, not this one's.
                return _ChainEvidence(
                    ChainConfirmation.DENIED,
                    set_block=set_block,
                    detail="matching_vector_predates_this_attempt",
                )
            if self._identical_twin_exists(
                intent_id=intent_id,
                reported=reported,
                attempt_block=attempt_block,
                set_block=set_block,
            ):
                self.log.warning(
                    "the chain holds this vector, but another intent carries the"
                    " SAME vector and could equally be the one that landed —"
                    " refusing to guess: UNKNOWN",
                    extra=log_fields(
                        intent_id=intent_id,
                        attempt_block=attempt_block,
                        set_block=set_block,
                    ),
                )
                return _ChainEvidence(
                    ChainConfirmation.UNKNOWN,
                    set_block=set_block,
                    detail="identical_vector_ambiguity",
                )
            # an internal review: carry the EXACT vector the chain reported. A recovery
            # confirmation reconciles the stored intent to this before publishing, so
            # a crash-recovered intent anchors chain state, not the float it was
            # composed from. `reported` is the u16 (or renormalized) vector the adapter
            # read back — the same bytes twin-detection and the match ran against.
            return _ChainEvidence(
                ChainConfirmation.CONFIRMED,
                set_block=set_block,
                detail="vector_match",
                reported_weights=dict(reported),
            )
        if set_block is not None and set_block < attempt_block:
            # The newest thing on chain is a DIFFERENT vector, recorded before our
            # attempt: nothing of ours has landed since. A positive denial.
            return _ChainEvidence(
                ChainConfirmation.DENIED,
                set_block=set_block,
                detail="different_vector_predates_this_attempt",
            )
        # A different vector recorded at/after our attempt (or undatable): ours may
        # have landed and been overwritten by a later intent. We cannot show it
        # landed, so we may not publish it — and we cannot show it did not, so we
        # may not bury it either.
        self.log.warning(
            "the chain holds a DIFFERENT weight vector than this intent's and"
            " cannot be shown to predate it — the intent stays UNKNOWN: it is"
            " neither publishable nor deniable on this evidence",
            extra=log_fields(
                intent_id=intent_id, attempt_block=attempt_block, set_block=set_block
            ),
        )
        return _ChainEvidence(
            ChainConfirmation.UNKNOWN,
            set_block=set_block,
            detail="different_vector_on_chain",
        )

    async def attempt_once(self) -> bool:
        """One full attempt: compose, submit, publish. True on a chain-accepted set."""
        self.metric_attempts.inc()
        await self._refresh_chain_async()
        # Periodic half of the recovery loop: never let an owed publication sit
        # behind a chain that happens to be tempo-gated for hours.
        try:
            await self.reconcile(publish_accepted=False)
        except Exception:
            self.log.exception(
                "reconciliation pass failed; continuing with this attempt"
            )
        reason = self._chain_state_reason()
        if reason is not None:
            self.metric_chain_state_skips.labels(reason=reason).inc()
            self.log.warning(
                "chain state is not usable — SKIPPING this attempt; no weight vector"
                " will be submitted (an empty/partial vector is never a valid set)",
                extra=log_fields(reason=reason),
            )
            return False

        # CRv4 commits are automatically revealed by the chain after their drand
        # round. A second set_weights while one is pending can create another
        # intent and make exact readback authorship ambiguous. Probe live state on
        # every cycle (including restart recovery) and HOLD until the current
        # commitment reveals; an unreadable pending-state probe also HOLDs.
        pending_probe = getattr(self._chain, "weight_commit_pending", None)
        if callable(pending_probe) and self.config.validator_hotkey:
            try:
                reveal_pending = bool(
                    await asyncio.to_thread(pending_probe, self.config.validator_hotkey)
                )
            except Exception as exc:
                self.metric_chain_state_skips.labels(
                    reason="commit_reveal_state_unavailable"
                ).inc()
                self.log.error(
                    "cannot determine whether a CRv4 weight commitment is pending; "
                    "HOLDING instead of risking a second write",
                    extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
                )
                return False
            if reveal_pending:
                self.metric_chain_state_skips.labels(
                    reason="commit_reveal_pending"
                ).inc()
                self.log.info(
                    "a finalized CRv4 weight commitment is awaiting automatic "
                    "reveal; no new intent is created and nothing is published yet"
                )
                return False
        # Snapshot SOURCE (wave 5). The SharedSnapshotProvider raises a typed
        # SharedSnapshotError: SnapshotDigestMismatch (tamper) is REFUSED loudly;
        # anything else UNAVAILABLE is a plain HOLD. The local miner_manager provider
        # raises ordinary exceptions and is handled exactly as before. Its optional
        # `epoch_inputs()` extension hands over the epoch log's crown/result/burn_uid
        # + stated u16 vector so this validator builds the IDENTICAL vector.
        try:
            miners = list(await asyncio.to_thread(self._snapshots.miner_snapshots))
            epoch_inputs = self._epoch_inputs()
            snapshot_digest = self._resolved_snapshot_digest()
            if epoch_inputs is not None and snapshot_digest is None:
                raise SnapshotDigestMismatch(
                    "shared epoch inputs do not expose their verified snapshot digest"
                )
        except SnapshotDigestMismatch as exc:
            self.metric_snapshot_tampered.inc()
            self.metric_chain_state_skips.labels(reason="snapshot_tampered").inc()
            self.log.critical(
                "REFUSING to submit: the shared epoch log's tamper-evidence chain"
                " broke (someone tampered). Nothing is submitted; the last confirmed"
                " vector stays live on chain and an operator is alerted",
                extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
            )
            return False
        except SharedSnapshotError as exc:
            self.metric_chain_state_skips.labels(reason="snapshot_unavailable").inc()
            self.log.warning(
                "shared epoch snapshot unavailable — SKIPPING this attempt, nothing"
                " submitted (HOLD; never fall back to local sampling — that diverges)",
                extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
            )
            return False
        except Exception as exc:
            self.metric_chain_state_skips.labels(reason="snapshots_unavailable").inc()
            self.log.warning(
                "miner snapshots unavailable — SKIPPING this attempt, nothing submitted",
                extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
            )
            return False

        reward_state = (
            epoch_inputs.reward_window_state
            if epoch_inputs is not None
            else crown_store.load_reward_window(self._conn)
        )
        composed_at = (
            epoch_inputs.composed_at if epoch_inputs is not None else self._clock()
        )

        # Shared convergence is AUTHORITY-VECTOR convergence. Once the EpochLog bytes
        # have passed the pointer/digest/on-chain-anchor checks above, the exact published
        # u16 vector is the scheduled chain input. Convert each integer to an exactly
        # representable float without re-normalizing it through local tokenomics; the chain
        # adapter's canonical quantizer therefore reproduces these same u16 pairs.
        #
        # Local composition and own-audit remain valuable DETECTORS, but owner policy makes
        # every finding report-only: a bug or disagreement in validator-side scoring/audit
        # code must never freeze the fleet. Dedicated beacon and own-audit processes sign
        # and POST their reports to the Audit Results API for manual remediation.
        if epoch_inputs is not None:
            vector = {
                int(uid): float(value)
                for uid, value in epoch_inputs.weight_u16.items()
                if int(value) > 0
            }
            if not vector:
                # There is literally no chain input to submit. This is not an audit verdict;
                # an empty authenticated authority vector cannot be expressed as set_weights.
                self.log.critical(
                    "authenticated authority snapshot contains an empty/all-zero weight"
                    " vector — no chain submission is possible",
                    extra=log_fields(epoch_id=epoch_inputs.epoch_id),
                )
                return False

        else:
            # Local/report mode has no authenticated authority vector to follow, so it
            # still composes its own vector and retains the existing fail-closed source
            # and canonical-sink rules.
            # The pure composer no longer exposes an incomplete vector: if any fixed
            # allocation is unpayable, omitting the sink would let the chain normalize
            # that share into the remaining earners. Resolve the canonical identity
            # before composition and pass it unconditionally. This is local/report mode
            # only; production follows the authenticated authority vector above.
            try:
                burn_uid = await asyncio.to_thread(
                    resolve_burn_uid,
                    self._chain,
                    report_fallback=self.config.burn_uid,
                )
            except Exception as exc:
                self.metric_chain_state_skips.labels(reason="burn_uid_unavailable").inc()
                self.log.error(
                    "cannot resolve the canonical subnet-owner burn uid from chain"
                    " state — HOLDING local composition; uid 0 is never guessed",
                    extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
                )
                return False
            vector = build_weight_vector(
                self.tokenomics,
                miners,
                burn_uid=burn_uid,
                reward_state=reward_state,
                now=composed_at,
            )
            if not vector or sum(vector.values()) <= 0.0:
                self.log.info(
                    "weight vector is empty/all-zero — nothing to submit this round",
                    extra=log_fields(miners=len(miners)),
                )
                return False

        binding = await self._submission_hotkey_binding(vector, miners, epoch_inputs)
        if binding is None:
            return False

        # Packet evidence is an INPUT to publication, not to the authority vector's
        # scheduled chain submission. In shared/production mode, copy refs only from
        # the already-authenticated in-memory log as a best-effort durability aid.
        # Any failure is reported but NEVER gates the write; the exact epoch id +
        # snapshot digest below let the post-submit worker recover and validate the
        # leaves later. Local/report mode retains its historical fail-closed capture
        # because it has no immutable authority epoch from which exact recovery is
        # possible.
        frozen_at = self._iso_now()
        snapshot_epoch_id: int | None = None
        if epoch_inputs is not None:
            snapshot_epoch_id = int(epoch_inputs.epoch_id)
            digests = self._committed_authority_packet_digests()
            if digests is None:
                self.metric_publication_input_failures.inc()
                self.log.error(
                    "could not durably copy score-packet refs before submission;"
                    " submitting the authenticated authority vector anyway and"
                    " queueing exact post-submit recovery from its frozen epoch",
                    extra=log_fields(
                        epoch_id=snapshot_epoch_id,
                        snapshot_digest=snapshot_digest or "",
                        submission_action="continue",
                    ),
                )
        else:
            digests = self._publication_digests()
            if digests is None:
                self.metric_publication_input_failures.inc()
                self.log.error(
                    "local/report score-packet evidence is unreadable — SKIPPING"
                    " this local composition. An empty digest set would falsely"
                    " claim there was no evidence, and this mode has no immutable"
                    " authority epoch from which to recover it",
                    extra=log_fields(miners=len(miners)),
                )
                return False

        # This is a dashboard/read-model mirror, never a composition or submission
        # input in shared mode.  Persist the exact predecessor-folded state already
        # carried by the authenticated EpochLog; do not re-derive it from whatever
        # subset of competition history happens to exist in this validator's DB.
        # Decision 24 still controls the failure boundary: observation can degrade,
        # but it cannot interrupt the scheduled authority-vector write.
        if epoch_inputs is not None:
            self._mirror_authority_reward_read_model(epoch_inputs)

        # Final convergence fence: the provider now re-reads the archive boundary,
        # after mirror/parse/vector binding and immediately before the durable intent
        # + non-idempotent write. If an epoch closed during that preparation, HOLD.
        # The DB half also applies the highest-seen epoch floor before any new row.
        if epoch_inputs is not None and not self._shared_epoch_is_safe(epoch_inputs):
            return False

        # review #10: the intent is durable BEFORE the non-idempotent chain write,
        # so a lost response or a crash never leaves an unexplainable vector.
        attempt_block = await asyncio.to_thread(self._chain.current_block)
        intent_id = intents.record_intent(
            self._conn,
            created_at=self._iso_now(),
            attempt_block=attempt_block,
            version_key=self.config.version_key,
            weights=vector,
            packet_digests=digests,
            # review new-6: the watermark for the NEXT publication is when THIS
            # list was frozen, not when this intent eventually settles.
            packets_frozen_at=frozen_at,
            # Freeze the exact verified EpochLog identity BEFORE the non-idempotent
            # chain write. Recovery publishes from this durable value, never from a
            # provider that may have advanced to a later epoch after a crash.
            snapshot_digest=snapshot_digest,
            snapshot_epoch_id=snapshot_epoch_id,
        )
        # The non-idempotent write may now block for inclusion/finalized proof.
        # Expose its durable obligation immediately, not one cadence later.
        self._sync_pending_intents_metric()

        # The intended uid -> hotkey binding covers every positive target: the
        # complete authenticated census plus the separately re-resolved subnet
        # owner sink.  This includes competition-only recipients absent from the
        # inference snapshot set.
        outcome = await self._submit(
            vector, attempt_block=attempt_block, intent_id=intent_id, hotkeys=binding
        )
        if not outcome.success:
            if (
                outcome.confirmation is ChainConfirmation.DENIED
                and not outcome.ambiguous
            ):
                # A SYNCHRONOUS rejection of the only write we issued: nothing
                # ever reached the chain, so this is settled here and now.
                intents.mark_abandoned(
                    self._conn,
                    intent_id,
                    at=self._iso_now(),
                    resolution=outcome.resolution,
                )
                self._sync_pending_intents_metric()
                return False
            # Either UNKNOWN (the vector may be live on chain) or a denial that
            # followed an AMBIGUOUS write. Both stay PENDING: a retry's rejection
            # says nothing about the earlier attempt, and burying an intent is
            # reserved for reconcile()'s positive-denial + age-bound + CRITICAL
            # path.
            intents.note_check(
                self._conn,
                intent_id,
                at=self._iso_now(),
                verdict=outcome.confirmation.value,
            )
            self._sync_pending_intents_metric()
            if outcome.confirmation is ChainConfirmation.UNKNOWN:
                self.metric_unresolved_intents.inc()
            self.log.warning(
                "this weight intent is NOT settled here: its vector cannot be shown"
                " to have landed, and it is not abandoned on this evidence either."
                " It stays PENDING and every reconciliation pass re-checks it",
                extra=log_fields(
                    intent_id=intent_id,
                    resolution=outcome.resolution,
                    chain_check=outcome.confirmation.value,
                    ambiguous=outcome.ambiguous,
                ),
            )
            return False
        # review round-3 #3 / round-4 #3: publish/anchor the EXACT u16 vector the
        # adapter reported as submitted — what actually landed on chain — NEVER the
        # pre-quantization float intent. The ordinary case is SCALE-EQUIVALENCE: the
        # adapter quantized ours with the same uids — the
        #    intent {1:0.4,2:0.6} and the u16 {1:26214,2:39321} are the same vector,
        #    but the anchored/audited document must record the u16 the chain holds,
        #    byte-for-byte, not the floats it was derived from.
        # review round-5 #4: acceptance and that rewrite must be ONE commit. The
        # connection is autocommit, so a separate mark_accepted + reconcile_vector is
        # two commits, and a crash between them left an ACCEPTED intent still carrying
        # its FLOAT vector — which startup reconciliation then anchored verbatim. The
        # atomic `accept_with_vector` closes that window; when NO vector was reported
        # (`submitted` falsy) it degrades to a plain mark_accepted, leaving the intent's
        # own vector untouched. A target-set divergence after a reported success is an
        # adapter-contract breach; it is still published as landed chain evidence but
        # alerted critically.
        submitted = outcome.submitted
        if submitted and not intents.weights_match(submitted, vector):
            self.log.critical(
                "the adapter reported a submitted vector that DIVERGED from the exact"
                " authority intent despite the pre-write target-binding contract; the"
                " durable record is rewritten to what actually landed and operators"
                " must investigate",
                extra=log_fields(
                    intent_id=intent_id,
                    intent_uids=sorted(vector),
                    submitted_uids=sorted(submitted),
                ),
            )
        intents.accept_with_vector(
            self._conn,
            intent_id,
            accepted_block=outcome.block,
            resolution=outcome.resolution,
            weights=submitted,
        )
        self._sync_pending_intents_metric()
        if submitted:
            vector = {int(uid): float(w) for uid, w in submitted.items()}
        self.metric_successes.inc()
        self._last_success_at = self._clock()
        self.log.info(
            "weights submitted",
            extra=log_fields(
                intent_id=intent_id,
                block=outcome.block,
                uids=len(vector),
                total=sum(vector.values()),
                resolution=outcome.resolution,
            ),
        )
        try:
            await self._publish_intent_bounded(intent_id)
        except Exception as exc:
            # Chain acceptance is already durable at this point. Publication is
            # independently re-driven from the accepted intent on every later
            # reconciliation pass; a storage/ledger implementation bug therefore
            # alerts loudly but cannot turn a successful weight write into a process
            # crash that prevents the next scheduled write.
            self.metric_publication_errors.inc()
            self.log.exception(
                "post-submit publication raised; weights remain accepted and the"
                " durable intent stays queued for reconciliation",
                extra=log_fields(
                    intent_id=intent_id,
                    error=f"{type(exc).__name__}: {exc}",
                    submission_action="accepted",
                ),
            )
        # Convergence-health observation (wave 5, observe-only): now that OUR honest
        # vector is on chain, sample peers and surface agreement. Never a gate — it
        # cannot change what we already submitted; a failure here is swallowed.
        try:
            await asyncio.to_thread(self._observe_convergence, vector)
        except Exception:
            self.log.exception("convergence observation failed; submission unaffected")
        return True

    # -- snapshot inputs (wave 5) ------------------------------------------------

    def _epoch_inputs(self) -> EpochInputs | None:
        """The shared provider's epoch inputs, or None for the local provider.

        Feature-detected: a `SharedSnapshotProvider` exposes `epoch_inputs()`
        returning the epoch log's reward state/result/burn_uid + stated u16 vector; the
        local `miner_manager` provider does not, and this returns None (the reward
        state comes from local schema-v14 persistence). Any error the shared
        provider raises propagates to `attempt_once`'s typed HOLD/REFUSE handling.
        """
        getter = getattr(self._snapshots, "epoch_inputs", None)
        if not callable(getter):
            return None
        inputs = getter()
        if inputs is not None and not isinstance(inputs, EpochInputs):
            raise SharedSnapshotError(
                "epoch_inputs() returned a non-EpochInputs value; refusing to compose"
            )
        return inputs

    def _resolved_snapshot_digest(self) -> str | None:
        """Verified shared EpochLog digest for this attempt, or None in local mode."""
        getter = getattr(self._snapshots, "resolved_snapshot_digest", None)
        if not callable(getter):
            return None
        digest = getter()
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise SnapshotDigestMismatch(
                f"shared provider returned invalid snapshot digest {digest!r}"
            )
        return digest

    def _mirror_authority_reward_read_model(self, inputs: EpochInputs) -> None:
        """Best-effort exact EpochLog -> local dashboard state projection.

        The shared vector has already crossed the pointer/store/anchor boundary.
        A local SQLite fault or an immutable-history conflict is an operator signal,
        not permission to hold emissions.  ``crown_store`` supplies the atomic,
        monotonic/idempotent projection; this wrapper supplies the report-only
        failure boundary required by Decision 24.
        """
        try:
            crown_store.mirror_epoch_reward_read_model(
                self._conn,
                result=inputs.competition_result,
                state=inputs.reward_window_state,
            )
        except Exception as exc:
            self.metric_reward_read_model_failures.inc()
            self.log.error(
                "could not mirror the authenticated authority competition/reward "
                "state into the dashboard read model; continuing with the exact "
                "authority-vector submission",
                extra=log_fields(
                    epoch_id=inputs.epoch_id,
                    competition_id=(
                        None
                        if inputs.competition_result is None
                        else inputs.competition_result.competition_id
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                    submission_action="continue",
                ),
            )

    def _shared_epoch_is_safe(self, inputs: EpochInputs) -> bool:
        """Require a current archive boundary and a durable non-regression floor.

        A signed/anchored epoch stays valid historical evidence forever.  It is not
        necessarily the vector a validator should submit *now*.  In live mode the
        shared provider must therefore expose the exact archive-proven boundary it
        matched while resolving ``/epoch/latest``.  Independently, the highest
        shared epoch already admitted to ``weight_intents`` survives process/RPC
        restarts and prevents rollback if both the API and one transient chain view
        regress together.  Same-epoch retries remain valid.
        """
        if self._chain_mode == "bittensor":
            getter = getattr(self._snapshots, "resolved_latest_boundary", None)
            if not callable(getter):
                reason = "snapshot_epoch_boundary_unverified"
                self.metric_chain_state_skips.labels(reason=reason).inc()
                self.log.critical(
                    "shared provider did not expose an archive-proven latest epoch "
                    "boundary in bittensor mode — HOLDING the weight submission",
                    extra=log_fields(
                        reason=reason,
                        epoch_id=inputs.epoch_id,
                        close_block=inputs.close_block,
                    ),
                )
                return False
            try:
                boundary = getter()
            except Exception as exc:
                reason = "snapshot_epoch_boundary_unverified"
                self.metric_chain_state_skips.labels(reason=reason).inc()
                self.log.error(
                    "could not read the provider's archive-boundary proof — HOLDING",
                    extra=log_fields(
                        reason=reason,
                        epoch_id=inputs.epoch_id,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
                return False
            expected = (int(inputs.epoch_id), int(inputs.close_block))
            if boundary is None:
                reason = "snapshot_epoch_boundary_unverified"
                self.metric_chain_state_skips.labels(reason=reason).inc()
                self.log.critical(
                    "shared provider has no archive-proven latest boundary in "
                    "bittensor mode — HOLDING the weight submission",
                    extra=log_fields(
                        reason=reason,
                        epoch_id=inputs.epoch_id,
                        close_block=inputs.close_block,
                    ),
                )
                return False
            if (
                not isinstance(boundary, tuple)
                or len(boundary) != 2
                or isinstance(boundary[0], bool)
                or not isinstance(boundary[0], int)
                or isinstance(boundary[1], bool)
                or not isinstance(boundary[1], int)
                or boundary != expected
            ):
                reason = "snapshot_epoch_boundary_mismatch"
                self.metric_chain_state_skips.labels(reason=reason).inc()
                self.log.critical(
                    "shared epoch does not match its archive-proven latest boundary — "
                    "HOLDING the weight submission",
                    extra=log_fields(
                        reason=reason,
                        epoch_id=inputs.epoch_id,
                        close_block=inputs.close_block,
                        verified_boundary=repr(boundary),
                    ),
                )
                return False

        try:
            durable_floor = intents.latest_snapshot_epoch_id(self._conn)
        except Exception as exc:
            reason = "snapshot_epoch_floor_unavailable"
            self.metric_chain_state_skips.labels(reason=reason).inc()
            self.log.error(
                "could not read the durable authority-epoch floor — HOLDING",
                extra=log_fields(
                    reason=reason,
                    epoch_id=inputs.epoch_id,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            return False
        if durable_floor is not None and inputs.epoch_id < durable_floor:
            reason = "snapshot_epoch_regression"
            self.metric_chain_state_skips.labels(reason=reason).inc()
            self.log.critical(
                "authority latest epoch regressed below this validator's durable "
                "highest-seen epoch — HOLDING the weight submission",
                extra=log_fields(
                    reason=reason,
                    epoch_id=inputs.epoch_id,
                    durable_epoch_floor=durable_floor,
                ),
            )
            return False
        return True

    async def _submission_hotkey_binding(
        self,
        vector: dict[int, float],
        miners: Sequence[MinerSnapshot],
        epoch_inputs: EpochInputs | None,
    ) -> dict[int, str] | None:
        """Bind every positive uid to the identity the vector was earned by.

        Shared authority vectors may pay competition/crown recipients that are
        absent from ``EpochLog.miners``.  Their authenticated identity still
        exists in the complete close-block census; passing only inference miners
        leaves those targets vulnerable to uid recycling before submission.

        The subnet-owner sink is intentionally not a miner-census row.  When it
        is positive, re-resolve the live owner uid and require it to be the uid
        authenticated by the epoch log, then bind that uid to the current cached
        metagraph hotkey.  An unavailable/moved owner is a chain-state HOLD, not
        permission to pay whichever hotkey currently occupies the numeric slot.
        """
        if epoch_inputs is None:
            # Report/local mode is forbidden in production, but its Bittensor
            # adapter still requires a complete binding map. Economic recipients
            # are already represented by ``miners``; the only extra positive uid
            # the local composer may add is the canonical subnet-owner sink. Bind
            # that identity from the refreshed metagraph when available. Report
            # simulators without a real owner neuron retain their historical
            # best-effort map because their adapters do not enforce binding safety.
            binding = {miner.uid: miner.hotkey for miner in miners}
            positive_uids = {int(uid) for uid, value in vector.items() if value > 0.0}
            missing = positive_uids - binding.keys()
            if missing:
                try:
                    current_sink_uid = await asyncio.to_thread(
                        resolve_burn_uid,
                        self._chain,
                        report_fallback=self.config.burn_uid,
                    )
                    matches = [
                        neuron
                        for neuron in await asyncio.to_thread(self._chain.neurons)
                        if neuron.uid == current_sink_uid and neuron.hotkey
                    ]
                except Exception:
                    matches = []
                    current_sink_uid = None
                if (
                    current_sink_uid is not None
                    and missing == {current_sink_uid}
                    and len(matches) == 1
                ):
                    binding[current_sink_uid] = matches[0].hotkey
            return binding

        binding = {
            int(uid): str(hotkey)
            for uid, hotkey in epoch_inputs.miner_census_hotkeys.items()
            if str(hotkey)
        }
        # Pure-model/back-compat epoch fixtures can derive their census from the
        # economic snapshots.  Production logs carry both and validate equality.
        for miner in miners:
            existing = binding.get(miner.uid)
            if existing is not None and existing != miner.hotkey:
                self.metric_chain_state_skips.labels(
                    reason="target_binding_mismatch"
                ).inc()
                self.log.critical(
                    "authenticated epoch target identity is internally inconsistent;"
                    " refusing to submit",
                    extra=log_fields(
                        uid=miner.uid,
                        census_hotkey=existing,
                        miner_hotkey=miner.hotkey,
                    ),
                )
                return None
            binding[miner.uid] = miner.hotkey

        positive_uids = {int(uid) for uid, value in vector.items() if value > 0.0}
        sink_uid = epoch_inputs.burn_uid
        if sink_uid is not None and sink_uid in positive_uids:
            try:
                current_sink_uid = await asyncio.to_thread(
                    resolve_burn_uid,
                    self._chain,
                    report_fallback=self.config.burn_uid,
                )
                current = [
                    neuron
                    for neuron in await asyncio.to_thread(self._chain.neurons)
                    if neuron.uid == current_sink_uid and neuron.hotkey
                ]
            except Exception as exc:
                self.metric_chain_state_skips.labels(
                    reason="burn_identity_unavailable"
                ).inc()
                self.log.error(
                    "cannot bind the authenticated subnet-owner sink to a live"
                    " hotkey; refusing to submit",
                    extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
                )
                return None
            if current_sink_uid != sink_uid:
                self.metric_chain_state_skips.labels(reason="burn_uid_changed").inc()
                self.log.error(
                    "the authenticated epoch sink uid no longer belongs to the"
                    " current subnet owner; refusing to pay a recycled slot",
                    extra=log_fields(
                        epoch_burn_uid=sink_uid,
                        current_burn_uid=current_sink_uid,
                    ),
                )
                return None
            if len(current) != 1:
                self.metric_chain_state_skips.labels(
                    reason="burn_identity_unavailable"
                ).inc()
                self.log.error(
                    "the current subnet-owner uid has no unique cached hotkey"
                    " binding; refusing to submit",
                    extra=log_fields(uid=current_sink_uid, matches=len(current)),
                )
                return None
            binding[sink_uid] = current[0].hotkey

        missing = sorted(positive_uids - binding.keys())
        if missing:
            self.metric_chain_state_skips.labels(reason="target_binding_missing").inc()
            self.log.critical(
                "authenticated authority vector has positive targets without"
                " uid-to-hotkey bindings; refusing an unsafe uid-only submission",
                extra=log_fields(missing_uids=missing),
            )
            return None
        return {uid: binding[uid] for uid in positive_uids}

    # -- convergence-health observation (wave 5) ---------------------------------

    def _convergence_peers(self) -> list[str]:
        """Peer validator hotkeys to sample (never ours), from config."""
        peers = list(self.config.convergence_peer_hotkeys)
        if self.config.convergence_use_metagraph_peers:
            try:
                for neuron in self._chain.neurons():
                    if neuron.is_validator and neuron.hotkey:
                        peers.append(neuron.hotkey)
            except Exception as exc:
                self.log.warning(
                    "could not read the metagraph for convergence peers",
                    extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
                )
        own = self.config.validator_hotkey
        # de-dup, drop ours, preserve order
        seen: set[str] = set()
        ordered: list[str] = []
        for hotkey in peers:
            if hotkey and hotkey != own and hotkey not in seen:
                seen.add(hotkey)
                ordered.append(hotkey)
        return ordered

    def _observe_convergence(self, our_vector: dict[int, float]) -> None:
        """Emit `vidaio_weightsetter_convergence` = fraction of peers matching us.

        Reads each configured/metagraph peer's on-chain vector via the adapter's
        optional `submitted_weights(hotkey)` and compares it to ours with the same
        scale-invariant `intents.weights_match` (±1 u16) the tri-state confirmation
        uses. Observe-only: it does not, and must not, change this submission — this
        validator always submits its own honest vector; the gauge just surfaces
        whether the shared source / pipeline drifted, BEFORE it costs emissions. On
        an honest network every peer validator reading the same finalized epoch
        matches, so the gauge sits at 1.0.
        """
        if not (
            self.config.convergence_observe_enabled
            or self.config.convergence_peer_hotkeys
        ):
            return
        reader = getattr(self._chain, "submitted_weights", None)
        if not callable(reader):
            return
        peers = self._convergence_peers()
        if not peers:
            return
        observed = 0
        agreeing = 0
        for hotkey in peers:
            try:
                report = reader(hotkey)
            except Exception as exc:
                # A peer we cannot read is simply not observed this epoch — never a
                # disagreement (unknown is never "assume divergent").
                self.log.debug(
                    "convergence peer read failed",
                    extra=log_fields(peer=hotkey, error=f"{type(exc).__name__}: {exc}"),
                )
                continue
            if report is None:
                continue  # peer has not submitted yet: not observed
            peer_vector = {
                int(uid): float(w) for uid, w in dict(report.weights or {}).items()
            }
            if not peer_vector:
                continue
            observed += 1
            if intents.weights_match(peer_vector, our_vector):
                agreeing += 1
        if observed == 0:
            return  # nothing to report; leave the gauge as-is
        fraction = agreeing / observed
        self.metric_convergence.set(fraction)
        (self.log.info if agreeing == observed else self.log.warning)(
            "convergence observed across peer validators",
            extra=log_fields(
                peers_observed=observed,
                peers_agreeing=agreeing,
                fraction=fraction,
            ),
        )

    # -- chain submission --------------------------------------------------------

    async def _submit(
        self,
        vector: dict[int, float],
        *,
        attempt_block: int,
        intent_id: int | None = None,
        hotkeys: dict[int, str] | None = None,
    ) -> _Submission:
        """Bounded-retry set_weights (NOT timeout-bounded) that never RE-writes blindly.

        `hotkeys` is the intended uid -> hotkey binding this vector was scored against
        (the composition snapshot's per-uid hotkeys). It is passed straight through to
        the adapter so production refuses the complete write if a uid is recycled
        between scoring and submission. It never pays the new occupant or donates the
        removed target's share to survivors.

        The set_weights call itself is NOT wrapped in a caller timeout (chain #11):
        the adapter serializes the extrinsic on its own socket and the inclusion wait
        must not be abandoned mid-flight. The bounded retry stays safe because the
        chain's tempo gate cannot accept a second write inside the first's window.

        `set_weights` is not idempotent, so a retry after a timed-out attempt is
        only safe because the chain's own tempo gate rejects a second write inside
        the same window. That gives the reconciliation rule: before
        re-submitting after an ambiguous attempt, ASK the chain whether OUR VECTOR
        is already there; if it is, stop — the write succeeded.

        A transport error is treated as ambiguous too: we cannot distinguish a
        request that never left from one whose response was lost.

        Round 2 made that question tri-state and forced a REFRESHED snapshot.
        Round 3 fixed WHAT it asks: the chain must report this intent's own
        vector (`_chain_evidence`), never merely "a write of ours happened".

        * CONFIRMED stops immediately — the write landed, and no second write is
          issued at all.
        * DENIED permits the retry outright: the chain has positively told us
          this vector is not there.
        * UNKNOWN also retries, and that retry is NOT blind: the whole reason the
          envelope is safe is the chain's own tempo gate, which cannot accept a
          second write inside the window the first one would occupy (retries are
          seconds apart, a tempo is ~20 minutes). What UNKNOWN must never do is
          CONCLUDE anything — see the caller: an unknown outcome leaves the
          intent pending, never abandoned.

        The two rejections that a retry can come back with are, deliberately, no
        longer conclusions about the FIRST attempt:

        - a TEMPO rejection after an ambiguous write means the chain already
          holds a write from this window. That write is PROBABLY ours — but
          "probably" is not publishable, so the intent is only reconciled as a
          success when the chain SHOWS us our vector. Otherwise it stays pending.
        - a synchronous credential/server rejection of a RETRY says nothing at
          all about the earlier ambiguous attempt, so it may not bury it.
        """
        cfg = self.config
        ambiguous = False
        for attempt in range(1, cfg.chain_retry_attempts + 1):
            if ambiguous:
                evidence = await self._chain_evidence(
                    vector=vector, attempt_block=attempt_block, intent_id=intent_id
                )
                if evidence.verdict is ChainConfirmation.CONFIRMED:
                    return self._reconciled(
                        evidence.set_block or attempt_block,
                        "chain_confirmed",
                        submitted=evidence.reported_weights,
                    )
                if evidence.verdict is ChainConfirmation.UNKNOWN:
                    self.log.warning(
                        "the chain cannot say whether our write landed; retrying"
                        " under the protection of the tempo gate, which cannot"
                        " accept a second write inside this window",
                        extra=log_fields(
                            attempt=attempt,
                            attempt_block=attempt_block,
                            chain_check=evidence.detail,
                        ),
                    )
            try:
                # NO timeout bound on set_weights (companion, chain #11): the real
                # adapter serializes the extrinsic on its socket internally and the
                # inclusion/finalization wait MUST run to completion — a caller
                # timeout that abandons the awaiting coroutine cannot cancel the
                # worker thread and leaves a live submit behind (the leak that OOMed
                # the pod). The chain's own tempo gate keeps the bounded retry safe;
                # a transport failure still surfaces as OSError (ambiguous) below.
                outcome = await self._chain.set_weights(
                    vector, version_key=cfg.version_key, hotkeys=hotkeys
                )
            except (TimeoutError, OSError) as exc:
                ambiguous = True
                self.log.warning(
                    "set_weights attempt was AMBIGUOUS (the extrinsic may or may not"
                    " have landed); the chain is consulted before any re-submission",
                    extra=log_fields(
                        attempt=attempt, error=f"{type(exc).__name__}: {exc}"
                    ),
                )
                if attempt < cfg.chain_retry_attempts:
                    await asyncio.sleep(
                        cfg.chain_retry_base_delay_seconds * 2 ** (attempt - 1)
                    )
                continue
            if outcome.pending_reveal:
                # The CRv4 commitment itself finalized, but ``Weights`` still
                # carries the previous vector until automatic reveal. This is a
                # known pending state, not a rejection and not permission to
                # publish/retry. The durable intent is reconciled on later passes
                # only after submitted_weights() shows this exact vector.
                self.log.info(
                    "weight commitment finalized and is awaiting CRv4 reveal; "
                    "leaving the intent pending without publishing or retrying",
                    extra=log_fields(block=outcome.block, attempt=attempt),
                )
                return _Submission(
                    success=False,
                    block=outcome.block,
                    resolution="commit_reveal_pending",
                    message=outcome.message,
                    confirmation=ChainConfirmation.UNKNOWN,
                )
            if outcome.success:
                return _Submission(
                    success=True,
                    block=outcome.block,
                    resolution="chain_accepted",
                    submitted=dict(outcome.submitted) or None,
                )
            if _is_tempo(outcome.message):
                if ambiguous:
                    # The chain already holds a write from this window and our own
                    # ambiguous attempt is the only candidate we know of — but a
                    # candidate is not evidence. Only the chain SHOWING us our
                    # vector turns this into an acceptance; anything else leaves
                    # the intent pending, so an unlanded vector is never published
                    #.
                    evidence = await self._chain_evidence(
                        vector=vector, attempt_block=attempt_block, intent_id=intent_id
                    )
                    if evidence.verdict is ChainConfirmation.CONFIRMED:
                        return self._reconciled(
                            evidence.set_block or outcome.block,
                            "tempo_after_ambiguous",
                            submitted=evidence.reported_weights,
                        )
                    self.metric_tempo_gated.inc()
                    self.log.error(
                        "an ambiguous write was followed by a TEMPO rejection, so"
                        " some write occupies this window — but the chain cannot"
                        " show us OUR vector, so this attempt is NOT recorded as a"
                        " success and nothing is published. The intent stays"
                        " pending and is re-checked",
                        extra=log_fields(
                            block=outcome.block,
                            attempt_block=attempt_block,
                            chain_check=evidence.detail,
                            verdict=evidence.verdict.value,
                        ),
                    )
                    return _Submission(
                        success=False,
                        block=outcome.block,
                        resolution="tempo_after_ambiguous_unconfirmed",
                        message=outcome.message,
                        tempo_gated=True,
                        ambiguous=True,
                        confirmation=evidence.verdict,
                    )
                # Normal cadence overlap (72 min attempts vs ~20 min tempo). A
                # synchronous rejection of the ONLY write we issued IS a positive
                # answer about it: nothing of ours reached the chain.
                self.metric_tempo_gated.inc()
                self.log.info(
                    "tempo gate not open yet — rescheduling",
                    extra=log_fields(block=outcome.block, message=outcome.message),
                )
                return _Submission(
                    success=False,
                    block=outcome.block,
                    resolution="tempo_gated",
                    message=outcome.message,
                    tempo_gated=True,
                    confirmation=ChainConfirmation.DENIED,
                )
            if ambiguous:
                # A credential/server rejection of a RETRY. It describes the retry,
                # not the earlier attempt whose fate is unknown — reading it as a
                # denial is what abandoned live intents on the spot, bypassing the
                # age bound entirely.
                evidence = await self._chain_evidence(
                    vector=vector, attempt_block=attempt_block, intent_id=intent_id
                )
                if evidence.verdict is ChainConfirmation.CONFIRMED:
                    return self._reconciled(
                        evidence.set_block or attempt_block,
                        "chain_confirmed_after_rejected_retry",
                        submitted=evidence.reported_weights,
                    )
                self.metric_chain_failures.inc()
                self.log.error(
                    "a RETRY was rejected synchronously after an ambiguous write."
                    " That rejection is about the retry only: the first attempt may"
                    " still be live on chain, so this intent is NOT abandoned here",
                    extra=log_fields(
                        block=outcome.block,
                        message=outcome.message,
                        attempt_block=attempt_block,
                        chain_check=evidence.detail,
                        verdict=evidence.verdict.value,
                    ),
                )
                return _Submission(
                    success=False,
                    block=outcome.block,
                    resolution="rejected_retry_after_ambiguous",
                    message=outcome.message,
                    ambiguous=True,
                    confirmation=evidence.verdict,
                )
            self.metric_chain_failures.inc()
            self.log.warning(
                "chain rejected set_weights",
                extra=log_fields(block=outcome.block, message=outcome.message),
            )
            return _Submission(
                success=False,
                block=outcome.block,
                resolution="chain_rejected",
                message=outcome.message,
                confirmation=ChainConfirmation.DENIED,
            )

        # Every attempt was ambiguous (or we stopped writing into an unreadable
        # chain): one last fresh read decides how this intent may be settled.
        evidence = await self._chain_evidence(
            vector=vector, attempt_block=attempt_block, intent_id=intent_id
        )
        verdict = evidence.verdict
        if verdict is ChainConfirmation.CONFIRMED:
            return self._reconciled(
                evidence.set_block or attempt_block,
                "chain_confirmed_after_exhaustion",
                submitted=evidence.reported_weights,
            )
        self.metric_chain_failures.inc()
        if verdict is ChainConfirmation.DENIED:
            self.log.error(
                "set_weights failed after retries and the chain positively does NOT"
                " hold this vector — the attempt is recorded as denied",
                extra=log_fields(attempt_block=attempt_block),
            )
            return _Submission(
                success=False,
                block=attempt_block,
                resolution="chain_denied_after_retries",
                ambiguous=True,
                confirmation=ChainConfirmation.DENIED,
            )
        self.log.error(
            "set_weights outcome is UNKNOWN after retries: the extrinsic may be"
            " live on chain. The intent STAYS PENDING and is re-checked by later"
            " reconciliation passes — it is never abandoned on a guess, because an"
            " abandoned intent is never published",
            extra=log_fields(attempt_block=attempt_block),
        )
        return _Submission(
            success=False,
            block=attempt_block,
            resolution="unconfirmed_after_retries",
            ambiguous=True,
            confirmation=ChainConfirmation.UNKNOWN,
        )

    def _reconciled(
        self, block: int, resolution: str, *, submitted: dict[int, float] | None = None
    ) -> _Submission:
        self.metric_reconciled.labels(resolution=resolution).inc()
        self.log.warning(
            "set_weights RECONCILED as a chain acceptance after an ambiguous attempt:"
            " the chain already holds this window's write, so the vector was NOT"
            " re-submitted and the attempt is recorded as a success, not a failure",
            extra=log_fields(block=block, resolution=resolution),
        )
        # an internal review: a reconciled acceptance is a recovery-style confirmation —
        # the chain SHOWED us its vector (`weights_match`), and that read is carried
        # here so the caller reconciles the durable row to the exact reported vector
        # (via accept_with_vector), never publishing the pre-quantization float. When
        # the confirming read is a u16 renormalization the ints on the grid are stored;
        # a scale-equivalent read still supersedes the float intent it derived from.
        return _Submission(
            success=True,
            block=block,
            resolution=resolution,
            ambiguous=True,
            submitted=(
                {int(uid): float(w) for uid, w in submitted.items()}
                if submitted
                else None
            ),
        )

    # -- publication (the auditable weight path) ---------------------------------

    def _committed_authority_packet_digests(self) -> list[str] | None:
        """Best-effort, non-verdict capture from the resolved authority epoch.

        The production shared provider has already authenticated the epoch-log bytes
        in order to obtain the vector. This method only copies the SCORE_PACKET refs
        from those same in-memory bytes; it performs no storage, network, scoring, or
        audit work. Any missing surface/exception returns ``None`` so the caller can
        persist an unresolved publication obligation and continue to ``set_weights``.
        The post-submit publisher independently re-resolves the exact epoch and checks
        its manifest root before anything is anchored.
        """
        provider = self._publication_inputs
        capture = getattr(provider, "committed_packet_digests", None)
        if not callable(capture):
            self.log.error(
                "shared publication provider has no committed_packet_digests surface;"
                " exact post-submit recovery will be attempted from the durable epoch"
            )
            return None
        try:
            return [str(digest) for digest in capture()]
        except Exception as exc:
            self.log.error(
                "committed authority packet-ref capture failed (report-only)",
                extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
            )
            return None

    def _publication_digests(self) -> list[str] | None:
        """Score-packet digests backing the vector about to be submitted.

        Returns [] when there genuinely were NO packets (the documented sentinel
        case) and None when the evidence COULD NOT BE READ — two situations the
        old code merged into one empty list, so a broken provider published the
        "no score packets" sentinel and claimed on chain that this cycle had no
        evidence. The caller skips the attempt on None.

        Prefers the provider's optional `recent_packet_digests(since)`, passing
        the WATERMARK of the last published intent — the instant its packet list
        was frozen, not when it settled — so consecutive publications partition
        the evidence with no gap.
        """
        provider = self._publication_inputs
        if provider is None:
            return []  # no provider wired at all: the sentinel is honest here
        recent = getattr(provider, "recent_packet_digests", None)
        if callable(recent) and _accepts_cutoff(recent):
            watermark = intents.publication_watermark(self._conn)
            since: datetime | str = watermark or (
                self._clock()
                - timedelta(seconds=self.config.publication_lookback_seconds)
            )
            try:
                return [str(d) for d in recent(since)]
            except Exception as exc:
                self.log.error(
                    "recent_packet_digests FAILED — the packet evidence for this"
                    " publication is unreadable (this is NOT an empty packet set)",
                    extra=log_fields(
                        since=str(since), error=f"{type(exc).__name__}: {exc}"
                    ),
                )
                return None
        try:
            return [str(d) for d in provider.score_packet_digests()]
        except Exception as exc:
            self.log.error(
                "score_packet_digests FAILED — the packet evidence for this"
                " publication is unreadable (this is NOT an empty packet set)",
                extra=log_fields(error=f"{type(exc).__name__}: {exc}"),
            )
            return None

    async def _intent_publication_digests(self, row: sqlite3.Row) -> list[str]:
        """Return independently validated leaves for this intent's exact epoch.

        Shared authority intents always re-resolve their durable ``snapshot_epoch_id``
        and ``snapshot_digest`` after submission. This both validates a successful
        best-effort capture and repairs a ``null`` capture after a crash. The provider
        must expose an epoch-addressed resolver; using its current/latest epoch is never
        an acceptable fallback because it could anchor a newer packet set under an
        older accepted vector.

        Local/report rows have no authority epoch and continue to use their already
        frozen digest list.
        """
        snapshot_epoch_id = row["snapshot_epoch_id"]
        if snapshot_epoch_id is None:
            return intents.load_packet_digests(row)

        snapshot_digest = row["snapshot_digest"]
        if not isinstance(snapshot_digest, str) or len(snapshot_digest) != 64:
            raise ValueError(
                f"weight intent {int(row['id'])} has an authority epoch but no valid"
                " durable snapshot digest"
            )
        provider = self._publication_inputs
        resolve = getattr(provider, "score_packet_digests_for_epoch", None)
        if not callable(resolve):
            raise RuntimeError(
                "shared publication provider cannot resolve packet digests by exact"
                " epoch; accepted intent remains queued"
            )
        validated = [
            str(digest)
            for digest in await asyncio.to_thread(
                resolve,
                int(snapshot_epoch_id),
                expected_snapshot_digest=snapshot_digest,
            )
        ]
        if intents.packet_digests_resolved(row):
            captured = intents.load_packet_digests(row)
            if sorted(captured) != sorted(validated):
                raise ValueError(
                    f"weight intent {int(row['id'])} pre-submit packet refs disagree"
                    " with the exact post-submit authority epoch"
                )
            return captured

        intents.attach_packet_digests(
            self._conn, int(row["id"]), packet_digests=validated
        )
        return validated

    async def _publish_intent(self, intent_id: int) -> bool:
        """Publish an ACCEPTED intent: store the exact vector, ledger it, anchor it.

        Driven entirely from the durable intent row, so it is safe to call again
        after a crash: the artifact store is content-addressed and write-once, the
        commitment id is pinned back onto the intent so a re-drive re-anchors the
        SAME commitment instead of minting a second one, and an anchor failure
        leaves the commitment at pending_chain for the next reconciliation pass.
        """
        if not self.config.publication_enabled:
            # dev/test only (guarded in __init__): settle the intent so the
            # reconciliation pass does not chase it forever, keeping the
            # acceptance resolution so HOW the chain accepted it is not lost.
            intents.mark_published(self._conn, intent_id, at=self._iso_now())
            self._sync_pending_intents_metric()
            self.log.info(
                "publication is disabled — the accepted vector is NOT audited",
                extra=log_fields(intent_id=intent_id),
            )
            return False
        assert (
            self._store is not None and self._ledger is not None
        )  # guarded in __init__
        row = intents.get_intent(self._conn, intent_id)
        if row["state"] == intents.STATE_PUBLISHED:
            self._sync_pending_intents_metric()
            return False
        # Resolve/validate evidence before storing or anchoring anything. A failure
        # leaves the ACCEPTED row durable and retryable; it cannot affect the vector
        # that was already submitted, and it is never converted to the empty sentinel.
        digests = await self._intent_publication_digests(row)
        vector = intents.load_vector(row)
        document = weight_vector_document(
            vector,
            version_key=int(row["version_key"]),
            block=int(row["accepted_block"]),
        )
        vector_ref = await asyncio.to_thread(
            self._store.put,
            canonical_json_bytes(document),
            ArtifactKind.WEIGHT_VECTOR,
        )
        root = merkle_root(digests) if digests else EMPTY_SCORE_PACKET_SET_ROOT
        payload = build_publication_record(
            PublicationRecord(
                score_packet_merkle_root=root,
                weight_vector_digest=vector_ref.digest,
                snapshot_digest=row["snapshot_digest"],
            )
        )
        # Keep the record's canonical JSON openable from the store (commitments.py
        # convention: the on-chain root is always backed by a stored document).
        await asyncio.to_thread(
            self._store.put, payload.canonical_json, ArtifactKind.MANIFEST
        )
        commitment_id = row["commitment_id"]
        if commitment_id is None:
            commitment_id = self._ledger.record(payload, created_at=self._iso_now())
            intents.attach_commitment(self._conn, intent_id, commitment_id)
        commitment_id = int(commitment_id)
        if not await self._anchor(commitment_id, payload):
            return False
        intents.mark_published(self._conn, intent_id, at=self._iso_now())
        self._sync_pending_intents_metric()
        self.metric_publications.inc()
        self.log.info(
            "weight vector published",
            extra=log_fields(
                intent_id=intent_id,
                commitment_id=commitment_id,
                weight_vector_digest=vector_ref.digest,
                score_packet_merkle_root=root,
                snapshot_digest=row["snapshot_digest"],
                score_packets=len(digests),
                empty_packet_sentinel=not digests,
            ),
        )
        return True

    async def _publish_intent_bounded(self, intent_id: int) -> bool:
        """Start at most one detached publication and wait only for its budget.

        ``anchor_commitment`` may be backed by a synchronous SDK worker which
        deliberately resists cancellation until its ambiguous extrinsic finishes.
        Cancelling this task at the budget boundary would therefore still freeze
        the emissions loop. Shield the retained single-flight task instead: timeout
        returns immediately, the accepted intent stays durable, and later weight
        attempts merely observe the in-flight publication rather than duplicating
        it or waiting again.
        """
        existing = self._publication_task
        if existing is not None:
            if not existing.done():
                return False
            # Retrieve the completed result before releasing the reference. The
            # runner converts ordinary exceptions to False, and this explicit
            # harvest also handles lifecycle cancellation without a task warning.
            if not existing.cancelled():
                existing.result()
            self._publication_task = None
            self._publication_task_intent_id = None

        task = asyncio.create_task(
            self._run_publication_task(intent_id),
            name=f"weight-publication-{intent_id}",
        )
        self._publication_task = task
        self._publication_task_intent_id = intent_id
        try:
            result = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self.config.publication_attempt_timeout_seconds,
            )
            return bool(result)
        except TimeoutError:
            self.metric_publication_errors.inc()
            self.log.error(
                "weight publication exceeded its isolated time budget and remains"
                " detached in single-flight; the chain acceptance stays durable"
                " and later weight attempts continue",
                extra=log_fields(
                    intent_id=intent_id,
                    timeout_seconds=self.config.publication_attempt_timeout_seconds,
                ),
            )
            return False
        finally:
            if task.done() and self._publication_task is task:
                if not task.cancelled():
                    task.result()
                self._publication_task = None
                self._publication_task_intent_id = None

    async def _run_publication_task(self, intent_id: int) -> bool:
        """Exception-contained body for the retained publication task."""
        try:
            return await self._publish_intent(intent_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # durable ACCEPTED row remains the retry boundary
            self.metric_publication_errors.inc()
            self.log.exception(
                "post-submit publication raised; weights remain accepted and the"
                " durable intent stays queued for reconciliation",
                extra=log_fields(
                    intent_id=intent_id,
                    error=f"{type(exc).__name__}: {exc}",
                    submission_action="accepted",
                ),
            )
            return False

    async def _finish_publication_task(self) -> None:
        """Drain the owned single-flight task during an orderly service stop."""
        task = self._publication_task
        if task is None:
            return
        if not task.done():
            self.log.info(
                "waiting for detached weight publication during shutdown",
                extra=log_fields(intent_id=self._publication_task_intent_id),
            )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # A publication task should only be cancelled by event-loop teardown;
            # retrieving that state prevents an unobserved-task diagnostic.
            pass
        finally:
            if task.done() and not task.cancelled():
                task.result()
            if task.done() and self._publication_task is task:
                self._publication_task = None
                self._publication_task_intent_id = None

    async def _drain_one_accepted_publication(self) -> bool:
        """Retry at most one old accepted row after the scheduled weight attempt."""
        for row in intents.unsettled_intents(self._conn):
            if row["state"] == intents.STATE_ACCEPTED:
                published = await self._publish_intent_bounded(int(row["id"]))
                if published:
                    self.metric_redriven.inc()
                return published
        return False

    async def _anchor(self, commitment_id: int, payload: CommitmentPayload) -> bool:
        """Anchor a pending commitment on chain. False leaves it re-drivable."""
        if self._ledger is None:
            return False
        if self._ledger.current_status(commitment_id) != CommitmentStatus.PENDING_CHAIN:
            return True  # already anchored by an earlier attempt
        cfg = self.config
        for attempt in range(1, cfg.chain_retry_attempts + 1):
            try:
                async with anchor_writer_lock(
                    self._anchor_writer_lock_path,
                    timeout_seconds=self._anchor_writer_lock_timeout_seconds,
                ):
                    await require_commitment_capacity(
                        self._chain,
                        netuid=self._anchor_netuid,
                        hotkey=self._anchor_hotkey,
                        payload=payload.payload,
                        operation=f"weight publication {commitment_id} anchor",
                    )
                    tx_id = await with_timeout(
                        self._chain.anchor_commitment(payload.payload),
                        cfg.chain_timeout_seconds,
                        "anchor_commitment",
                    )
            except (CommitmentCapacityError, TimeoutError, OSError) as exc:
                if attempt < cfg.chain_retry_attempts:
                    await asyncio.sleep(
                        cfg.chain_retry_base_delay_seconds * 2 ** (attempt - 1)
                    )
                    continue
                self.metric_chain_failures.inc()
                self.log.error(
                    "publication anchor failed after retries — commitment stays"
                    " pending_chain and IS re-driven by the reconciliation pass",
                    extra=log_fields(
                        commitment_id=commitment_id,
                        root=payload.root,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
                return False
            self._ledger.advance(
                commitment_id, CommitmentStatus.ANCHORED, at=self._iso_now()
            )
            self.log.info(
                "publication anchored",
                extra=log_fields(commitment_id=commitment_id, tx_id=tx_id),
            )
            return True
        return False

    # -- recovery loop -----------------------------------------------

    async def reconcile(self, *, publish_accepted: bool = True) -> int:
        """Finish every half-done attempt. Runs at startup AND before each attempt.

        - `accepted` intents: the chain holds the vector but publication (or its
          anchor) never completed — publish/re-anchor from the intent row.
        - `pending` intents: a crash between writing the intent and learning the
          chain outcome, an attempt whose fate the chain could not decide, or an
          ambiguous attempt whose retry was rejected. Ask the chain about THIS
          INTENT'S OWN VECTOR, with a FRESH snapshot:

          * CONFIRMED -> promote to accepted and publish;
          * UNKNOWN   -> leave PENDING and re-check next pass. Never abandoned,
            at any age: a vector that may be live on chain must stay publishable,
            and an abandoned intent is never published;
          * DENIED    -> only after `abandon_denied_intent_after_seconds` is it
            settled as abandoned, logged CRITICAL with the evidence.

          This is the ONLY place an intent can be buried after a write that might
          have landed, and it takes a positive denial plus the age bound to do it
.

          The whole pending branch is DEFERRED while the chain snapshot is
          unusable — a stale read must not be turned into a verdict.

        ``publish_accepted=False`` still resolves ambiguous chain writes but
        defers all object-store/anchor work; the emissions loop uses that mode
        before each scheduled submission. Returns the number of intents it
        settled (accepted, published, or abandoned).
        """
        settled = 0
        if self._chain_state_reason() is not None:
            # Self-contained entry point: try once to get a usable snapshot before
            # deciding anything about a pending intent.
            await self._refresh_chain_async()
        chain_usable = self._chain_state_reason() is None
        pending = 0
        for row in intents.unsettled_intents(self._conn):
            intent_id = int(row["id"])
            if row["state"] == intents.STATE_ACCEPTED:
                if not publish_accepted:
                    pending += 1
                    continue
                if await self._publish_intent_bounded(intent_id):
                    self.metric_redriven.inc()
                    settled += 1
                else:
                    pending += 1  # publication/anchor still owed
                continue
            if not chain_usable:
                pending += 1
                self.log.info(
                    "deferring a pending weight intent: the chain snapshot cannot"
                    " decide whether its write landed",
                    extra=log_fields(intent_id=intent_id),
                )
                continue
            try:
                pending_vector = intents.load_vector(row)
            except Exception as exc:
                pending += 1
                self.log.error(
                    "a pending weight intent's stored vector is unreadable, so it"
                    " can never be compared against the chain — it stays pending"
                    " for an operator to settle by hand",
                    extra=log_fields(
                        intent_id=intent_id, error=f"{type(exc).__name__}: {exc}"
                    ),
                )
                continue
            evidence = await self._chain_evidence(
                vector=pending_vector,
                attempt_block=int(row["attempt_block"]),
                intent_id=intent_id,
            )
            verdict = evidence.verdict
            intents.note_check(
                self._conn, intent_id, at=self._iso_now(), verdict=verdict.value
            )
            if verdict is ChainConfirmation.CONFIRMED:
                self.log.warning(
                    "a crashed attempt's OWN vector is on chain — completing its"
                    " publication instead of losing the audit trail",
                    extra=log_fields(
                        intent_id=intent_id,
                        block=row["attempt_block"],
                        set_block=evidence.set_block,
                    ),
                )
                accepted_block = (
                    evidence.set_block
                    if evidence.set_block is not None
                    else int(row["attempt_block"])
                )
                # an internal review: reconcile the durable row to the EXACT vector the
                # chain reported, ATOMICALLY with the state change, BEFORE
                # _publish_intent reads it. Confirmation is scale/tolerance-based
                # (weights_match), so the stored FLOAT intent is only scale-equivalent
                # to what the chain holds — publishing it verbatim would anchor a vector
                # the chain never held. When the confirming read carried a vector,
                # accept_with_vector persists it; when it did not (block bookkeeping
                # only), reported_weights is None, the rewrite is skipped, and the
                # intent's own match-proven vector stands — never substitute one.
                if evidence.reported_weights:
                    intents.accept_with_vector(
                        self._conn,
                        intent_id,
                        accepted_block=accepted_block,
                        resolution="chain_confirmed_on_restart",
                        weights=evidence.reported_weights,
                    )
                else:
                    intents.mark_accepted(
                        self._conn,
                        intent_id,
                        accepted_block=accepted_block,
                        resolution="chain_confirmed_on_restart",
                    )
                    self.log.warning(
                        "recovery confirmed this intent by block bookkeeping without a"
                        " vector read — the stored vector is NOT reconciled and stays as"
                        " recorded (residual: it may be the pre-quantization float, but"
                        " weights_match proved the chain holds this vector)",
                        extra=log_fields(
                            intent_id=intent_id, set_block=evidence.set_block
                        ),
                    )
                self.metric_reconciled.labels(
                    resolution="chain_confirmed_on_restart"
                ).inc()
                if publish_accepted:
                    if await self._publish_intent_bounded(intent_id):
                        self.metric_redriven.inc()
                    else:
                        pending += 1
                else:
                    pending += 1
                settled += 1
                continue
            if verdict is ChainConfirmation.UNKNOWN:
                # The one thing this pass must NOT do. Stay pending forever if
                # need be: an unpublishable accepted vector is unrecoverable,
                # whereas a pending intent costs a row and a re-check.
                pending += 1
                self.log.warning(
                    "a weight intent's fate is still UNKNOWN — it stays PENDING"
                    " and will be re-checked. It is never abandoned and never"
                    " re-submitted: the chain cannot tell us either way",
                    extra=log_fields(
                        intent_id=intent_id,
                        block=row["attempt_block"],
                        created_at=row["created_at"],
                        chain_check=evidence.detail,
                    ),
                )
                continue
            # DENIED: a fresh chain read positively lacks THIS vector.
            age = self._intent_age_seconds(row)
            if age < self.config.abandon_denied_intent_after_seconds:
                pending += 1
                self.log.info(
                    "a weight intent is denied by the chain but still young —"
                    " re-checking before settling it",
                    extra=log_fields(intent_id=intent_id, age_seconds=age),
                )
                continue
            self.log.critical(
                "ABANDONING a weight intent: a FRESH chain read positively does not"
                " carry THIS intent's vector and the attempt is older than the"
                " bounded settle window. It will never be published — this is the"
                " only path to that outcome, and it requires positive evidence",
                extra=log_fields(
                    intent_id=intent_id,
                    attempt_block=row["attempt_block"],
                    created_at=row["created_at"],
                    age_seconds=age,
                    vector_digest=row["vector_digest"],
                    vector_fingerprint=intents.vector_fingerprint(pending_vector),
                    chain_set_block=evidence.set_block,
                    validator_hotkey=self.config.validator_hotkey,
                    chain_check=ChainConfirmation.DENIED.value,
                    evidence=evidence.detail,
                ),
            )
            intents.mark_abandoned(
                self._conn,
                intent_id,
                at=self._iso_now(),
                resolution="chain_denied_after_crash",
            )
            self.metric_abandoned.labels(resolution="chain_denied_after_crash").inc()
            settled += 1
        self.metric_pending_intents.set(pending)
        return settled

    def _intent_age_seconds(self, row: sqlite3.Row) -> float:
        """Seconds since the intent was recorded; 0.0 when unparseable.

        An unreadable timestamp must make the intent look YOUNG, never old: age
        is one of the two conditions for the terminal abandon, so a parse failure
        has to withhold that outcome rather than grant it.
        """
        try:
            created = datetime.fromisoformat(str(row["created_at"]))
        except (TypeError, ValueError):
            return 0.0
        if created.tzinfo is None or created.tzinfo.utcoffset(created) is None:
            created = created.replace(tzinfo=timezone.utc)
        now = self._clock()
        if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
            now = now.replace(tzinfo=timezone.utc)
        return max((now - created).total_seconds(), 0.0)
