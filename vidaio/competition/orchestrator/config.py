"""Orchestrator configuration — the `orchestrator:` section of config/default.yaml.

Every boundary the orchestrator crosses (build, batch exec, probe, scoring HTTP)
is bounded here (spec §14: every boundary bounded; an unbounded await is a bug),
and every retry budget is finite — exhaustion HALTS the competition's pipeline
work with a CRITICAL log, it never fails the competition (systemic infra blocker
halts, spec §14 failure-recovery row).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from vidaio.chain.anchor_receipt import (
    DEFAULT_ANCHOR_RECEIPT_POLL_SECONDS,
    DEFAULT_ANCHOR_RECEIPT_TIMEOUT_SECONDS,
)


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Main loop cadence: engine.tick + the running competition's phase work.
    tick_seconds: float = Field(default=5.0, gt=0)

    # ---- bounded timeouts (seconds) -------------------------------------------
    build_timeout_seconds: float = Field(default=600.0, gt=0)
    batch_timeout_seconds: float = Field(default=900.0, gt=0)
    probe_timeout_seconds: float = Field(default=120.0, gt=0)
    scoring_timeout_seconds: float = Field(default=300.0, gt=0)

    # ---- retry budgets (attempts, incl. the first) -----------------------------
    build_retry_attempts: int = Field(default=2, ge=1)
    batch_retry_attempts: int = Field(default=2, ge=1)
    scoring_retry_attempts: int = Field(default=3, ge=1)
    retry_base_delay_seconds: float = Field(default=0.5, gt=0)
    retry_max_delay_seconds: float = Field(default=30.0, gt=0)

    #: A batch may be requeued at most this many times after its in-step retry
    #: budget is exhausted; beyond it the pipeline HALTS (never fails) with a
    #: CRITICAL log and an `orchestrator_halted` event.
    max_batch_requeues: int = Field(default=3, ge=0)

    # ---- contender execution / repository composition -------------------------
    #: The model default remains dependency-free for unit tests and report-mode
    #: callers.  The shipped production YAML explicitly selects ``modal``; both
    #: report overlays explicitly select ``docker``.  Production startup refuses
    #: Docker, and Modal startup refuses report mode, so neither path can silently
    #: drift onto the other execution boundary.
    sandbox_backend: Literal["docker", "modal"] = "docker"

    #: Create-only Modal identity.  All four values stay empty in the schema so
    #: merely importing/constructing config can never contact a remote GPU.  A
    #: Bittensor orchestrator must receive fresh, unique ``vidaio-next-*`` values
    #: plus the exact confirmation before the runtime constructor is called.
    modal_environment_name: str = ""
    modal_app_name: str = ""
    modal_run_label: str = ""
    modal_creation_confirmation: str = ""
    modal_gpu: str = "L4"
    modal_cpu: float = Field(default=2.0, gt=0)
    modal_memory_mb: int = Field(default=8192, ge=256)
    modal_sandbox_lifetime_seconds: int = Field(
        default=23 * 3600 + 30 * 60, ge=60, le=23 * 3600 + 30 * 60
    )
    modal_idle_timeout_seconds: int = Field(default=300, ge=1)
    modal_snapshot_ttl_seconds: int = Field(default=3600, ge=1)
    modal_max_output_entries: int = Field(default=4096, ge=1)

    #: Production materializes each independently enrolled HTTPS repository into
    #: a new private checkout, verifies its exact commit and tree, and removes it
    #: after use.  SecretStr keeps validation/repr paths from exposing the token;
    #: Git receives it only through a child-private askpass environment.
    git_read_only_token: SecretStr = SecretStr("")
    git_username: str = "x-access-token"
    git_allowed_hosts: tuple[str, ...] = ("github.com",)
    git_executable: str = "git"
    git_checkout_timeout_seconds: float = Field(default=180.0, gt=0)
    git_checkout_max_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    git_log_max_bytes: int = Field(default=1024 * 1024, gt=0)
    git_poll_seconds: float = Field(default=0.1, gt=0)

    # ---- docker sandbox resource limits ----------------------------------------
    sandbox_memory: str = "2g"
    sandbox_cpus: float = Field(default=1.0, gt=0)
    sandbox_tmpfs_size: str = "256m"
    sandbox_pids_limit: int = Field(default=256, ge=16)

    # ---- bounded sandbox output ----------------------
    #: Per-output and per-batch byte caps on what a contender may write to /output.
    #: Enforced by a HOST-side watchdog DURING the run (the container is killed the
    #: moment the batch cap is crossed) and re-checked after it — a contender can
    #: never fill the validator's disk. Crossing a cap is a CONTENDER fault: that
    #: contender is zero-scored, the competition continues.
    sandbox_output_max_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    sandbox_batch_output_max_bytes: int = Field(default=2 * 1024 * 1024 * 1024, gt=0)
    #: Cap on captured container stdout/stderr (another host-disk vector).
    sandbox_log_max_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    #: How often the watchdog measures /output while the container runs.
    sandbox_output_poll_seconds: float = Field(default=0.25, gt=0)
    #: Cap on a contender submission tarball (the FINALIZING_SUBMISSIONS backup).
    submission_backup_max_bytes: int = Field(default=512 * 1024 * 1024, gt=0)

    # ---- control API --------------------------------
    #: Bearer token for the control API. EMPTY DISABLES THE API entirely (fail
    #: closed): an unauthenticated competition-control surface would let anyone
    #: create competitions, enroll contenders and anchor commitments.
    control_token: str = ""
    control_host: str = "127.0.0.1"
    control_port: int = 8500

    #: In-step attempts to archive one contender's submission tarball before the
    #: failure is classified (a CONTENDER-fault tree is rejected; an INFRA failure
    #: halts finalization rather than advancing an unarchived contender).
    submission_backup_attempts: int = Field(default=2, ge=1)

    # ---- chain writes (anchoring) ----------------------------------------------
    #: Every anchor goes through the injected ChainAdapter (report mode records it;
    #: the real chain submits it) — the SINGLE anchor path. The write is attempted
    #: once; an ambiguous outcome is resolved by independent read-back, never a
    #: blind resubmission.
    # Live Commitments-pallet writes wait for inclusion + finalization; shorter
    # bounds abandon an in-flight worker and leave the outcome ambiguous.
    chain_timeout_seconds: float = Field(default=180.0, gt=0)
    #: Retained as a backwards-compatible config key. Competition commitment
    #: writes are deliberately never retried; only receipt reads are polled.
    chain_retry_attempts: int = Field(default=3, ge=1)
    anchor_receipt_timeout_seconds: float = Field(
        default=DEFAULT_ANCHOR_RECEIPT_TIMEOUT_SECONDS, gt=0
    )
    anchor_receipt_poll_seconds: float = Field(
        default=DEFAULT_ANCHOR_RECEIPT_POLL_SECONDS, gt=0
    )
    #: How long an unresolved anchor CLAIM blocks further anchor attempts.
    #: Within the window a second request is refused outright (409) — that is what
    #: stops two concurrent requests from both reaching the chain. Past it the
    #: claim is treated as a crashed attempt whose outcome is UNKNOWN: only the
    #: IDENTICAL payload may enter READ-ONLY receipt recovery (never another
    #: write), and a DIFFERENT payload is refused until an operator proves that
    #: nothing landed and resolves the claim with release_anchor_claim.
    anchor_claim_stale_seconds: float = Field(default=900.0, gt=0)

    #: Trusted remote scoring worker (spec §05: whoever controls this endpoint
    #: controls the numbers — must be validator-operated).
    scoring_worker_url: str = "http://127.0.0.1:8201"

    #: Working tree: <work_dir>/inputs (sealed input pool, files named by sha256),
    #: <work_dir>/outputs (collected output pool, same naming), <work_dir>/scratch
    #: (per-batch staging, disposable).
    work_dir: Path = Path("./data/orchestrator")

    #: Read-only schema-v14 baseline registry. Earning competition creation
    #: fails closed unless the manifest names this ledger's exact active row and
    #: the track has no unresolved CROWN promotion latch.
    baseline_registry_db_path: Path | None = None

    metrics_port: int = 9104
