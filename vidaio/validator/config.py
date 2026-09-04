"""Validator configuration (config section: `validator`).

Cadence anchors come from the spec (design spec §01): synthetic cycle sleep 3600–7200 s,
metagraph resync throttled to 30 min. The sleep jitter is DETERMINISTIC given the
rng injected into InferenceValidator (seeded `random.Random` in tests) — config
only carries the bounds.

EWMA decay is deliberately NOT duplicated here: the validator reads it from the
shared `tokenomics` section (TokenomicsConfig.ewma_decay) so the accumulator and
the weight composition can never drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


# A production challenge fetch includes a finalized Bittensor commitment write
# and archive-state readback before any miner dispatch.  Thirty seconds was a
# local HTTP timeout, not a defensible chain-finality budget.
MIN_ANCHORED_CHALLENGE_REQUEST_TIMEOUT_SECONDS = 240.0


class ValidatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # -- round cadence (design spec §01 anchors) --------------------------------------
    #: sleep between synthetic rounds is rng.uniform(min, max) — spec: 3600–7200 s
    cycle_sleep_min_seconds: float = 3600.0
    cycle_sleep_max_seconds: float = 7200.0
    #: metagraph resync throttle (spec: 30 min); refresh() is skipped inside this
    metagraph_refresh_seconds: float = 1800.0
    #: Chain-snapshot staleness gate. A round is SKIPPED with a
    #: structured reason when the cached snapshot is older than this or the
    #: adapter reports it unfresh/unavailable — a failed refresh must never be
    #: recorded as a successful one, and an empty snapshot must never be scored
    #: as a real (empty) round. 0 disables the gate.
    max_chain_snapshot_age_seconds: float = 3600.0

    # -- identity --------------------------------------------------------------
    #: WHO this validator is, as far as its peers are concerned. Sent as `owner`
    #: on every `POST /challenge/next` and used to scope the dispatched-challenge
    #: sweep to OUR OWN challenges: without an owner
    #: boundary, validator B expires validator A's still-live challenge and kills
    #: a round in flight. The natural value is this validator's hotkey.
    #:
    #: Empty DISABLES the orphan sweep entirely (with a warning): a validator that
    #: cannot say which challenges are its own must not expire anybody's.
    identity: str = ""

    # -- miner eligibility -----------------------------------------------------
    #: alpha-stake registration floor: miners below it are not dispatched to
    #: (blueprint §13: "a real stake/registration floor"); 0.0 = no floor.
    min_stake: float = 0.0

    # -- peer endpoints --------------------------------------------------------
    scoring_worker_url: str = "http://127.0.0.1:8201"
    challenge_service_url: str = "http://127.0.0.1:8210"
    #: Fleet-wide miner transport scheme. Bittensor's axon advertisement carries
    #: only IP/port, so every validator and miner edge on one deployment must use
    #: the same explicit HTTP or HTTPS contract. HTTPS retains normal certificate
    #: verification in httpx; local/report stacks use HTTP.
    miner_url_scheme: Literal["http", "https"] = "http"
    #: port the miner service listens on (protocol.py: reference miner 8300);
    #: ChainNeuron.ip carries no port.
    miner_port: int = 8300
    #: Validator-owned landing zone for remote miner output streams. The scoring
    #: worker must be able to read this path (co-locate it or mount this one
    #: directory); miners never see it.
    miner_artifact_dir: Path = Path("./data/validator/miner-artifacts")
    #: Independent caller-side byte caps. The client preflights a regular input
    #: descriptor and cuts a lying/chunked output off while streaming.
    miner_max_input_bytes: int = 2 * 1024 * 1024 * 1024
    miner_max_output_bytes: int = 4 * 1024 * 1024 * 1024
    #: SSRF boundary for chain-provided axon hosts. False (production default)
    #: permits only globally-routable literal IPs. Report/local/compose stacks
    #: must opt in to private IPs or service DNS labels explicitly.
    allow_non_public_miner_addresses: bool = False

    # -- peer credentials ------------------------------------------------------
    #: Bearer token for EVERY challenge-service route (`challenge_service.api_token`
    #: on that service). The challenge service hands out the held-out reference,
    #: so it authenticates unconditionally and fails closed: without this value the
    #: validator gets 401 on /challenge/next and can never run a round. Empty =
    #: send no Authorization header (only usable against an unauthenticated stub).
    challenge_service_token: str = ""
    #: Shared secret presented to miners as `X-Miner-Token` when they are
    #: configured with `miner.api_token` (the same header the organic gateway
    #: sends via `gateway.miner_api_token`). Empty = send no header, which is what
    #: an open loopback miner expects.
    miner_api_token: str = ""

    # -- boundary timeouts (every external await is with_timeout-wrapped) ------
    #: TaskWarrant track probe; a timeout leaves track=None and the miner is
    #: SKIPPED for the round — never defaulted to upscaling (old validator.py:844 bug).
    warrant_probe_timeout_seconds: float = 10.0
    #: Full miner task round-trip. A timeout is recorded and skipped
    #: non-punitively because no public third-party evidence can prove it.
    miner_request_timeout_seconds: float = 300.0
    challenge_request_timeout_seconds: float = (
        MIN_ANCHORED_CHALLENGE_REQUEST_TIMEOUT_SECONDS
    )
    #: POST /challenge/{id}/resolve — the call that releases the checked-out asset
    #: and unblocks the commitment reveal. Every fetched challenge is
    #: resolved on success, failure, timeout, shutdown, and on the next startup.
    challenge_resolve_timeout_seconds: float = 30.0
    #: scoring runs VMAF/PieAPP — allow it real time
    scoring_request_timeout_seconds: float = 600.0

    # -- orphaned-challenge sweep (the lost-RESPONSE blind spot) ---------------
    #: Startup recovery also sweeps GET /challenges?status=dispatched: a
    #: /challenge/next whose RESPONSE was lost leaves a dispatched challenge the
    #: validator never learned the id of, so no in-flight row exists to drain and
    #: the asset stays checked out forever. Any dispatched challenge older than
    #: this with no in-flight record of ours is resolved as `expired`. It must be
    #: comfortably longer than a round (a live round's challenge is younger than
    #: this and is drained through the normal in-flight path). 0 disables the sweep.
    orphan_sweep_age_seconds: float = 3600.0

    # (The retention_window_blocks field was REMOVED with the retention multiplier for v1 —
    # retention removed — owner decision; an internal review.)

    # -- scorer identity -------------------------------------------------------
    #: EXPLICIT OPERATOR PIN of the scorer identity (services.protocol, "THE
    #: SCORER-IDENTITY CONTRACT"). The identity is the worker's full effective
    #: string `<name>+<identity digest[:12]>` published on its GET /healthz —
    #: never a bare configured name.
    #:
    #: Empty (the default) selects PIN ON FIRST CONTACT: the validator discovers
    #: the identity from the worker at startup, pins it in memory, omits
    #: `scorer_version` from its ScoreRequests, and rejects any packet whose
    #: scorer_version is not the pinned one.
    #:
    #: Non-empty means an operator has decided which scorer this validator serves:
    #: the value is ASSERTED in every ScoreRequest (so the worker itself answers
    #: 409 scorer_version_mismatch to a stranger) AND the discovered identity must
    #: equal it, or startup fails loudly with a config error rather than drifting
    #: onto a scorer nobody chose (an internal review: a compromised worker must not be able
    #: to slip a differently-scored packet past us).
    scorer_version: str = ""
    #: How long the scorer-identity discovery call to GET /healthz may take.
    scorer_identity_timeout_seconds: float = 15.0
    #: OPERATOR ACKNOWLEDGEMENT that this validator may bind to a DIFFERENT scorer
    #: than the one its persisted pin names. The pin is
    #: durable precisely so a restart onto another worker cannot silently fold two
    #: scorers' packets into one EWMA accumulator: a disagreement normally REFUSES
    #: to score (CRITICAL log) until a human decides.
    #:
    #: Setting this to true clears the persisted pin at startup so the next
    #: discovery re-pins. It does NOT reset the accumulators built under the old
    #: scorer — by setting it the operator accepts that those accumulators now mix
    #: two scorers, or wipes them deliberately. Leave it false in production and
    #: flip it back after the intended re-pin.
    reset_scorer_pin: bool = False

    # -- parked challenge obligations ----------------------
    #: OPERATOR ACKNOWLEDGEMENT that PARKED in-flight challenges may be retried.
    #: A genuine ownership 403 on /challenge/{id}/resolve is permanent, so the
    #: refused row is parked durably (migration 0005): excluded from every later
    #: drain/recovery pass, but kept visible — the `vidaio_validator_parked_
    #: challenges` gauge, the startup recovery log and
    #: `miner_manager.parked_challenges()` all show it, because it is the only
    #: record that a service-side asset is stranded `in_use`.
    #:
    #: Setting this to true unparks every parked row at startup (the
    #: `InferenceValidator.unpark_challenges()` admin method does the same at
    #: runtime), returning them to the normal drain — do it AFTER fixing the
    #: service-side ownership state (or deciding to let the resolve be refused
    #: again). A row whose refusal still stands is simply re-parked on its next
    #: 403. Leave false in production and flip it back after the retry.
    unpark_challenges: bool = False

    # -- misc ------------------------------------------------------------------
    metrics_port: int = 9101

    @model_validator(mode="after")
    def _validate(self) -> "ValidatorConfig":
        if self.cycle_sleep_min_seconds <= 0:
            raise ValueError("cycle_sleep_min_seconds must be > 0")
        if self.cycle_sleep_max_seconds < self.cycle_sleep_min_seconds:
            raise ValueError(
                "cycle_sleep_max_seconds must be >= cycle_sleep_min_seconds"
            )
        if self.metagraph_refresh_seconds < 0:
            raise ValueError("metagraph_refresh_seconds must be >= 0")
        if self.max_chain_snapshot_age_seconds < 0:
            raise ValueError("max_chain_snapshot_age_seconds must be >= 0")
        if self.min_stake < 0:
            raise ValueError("min_stake must be >= 0")
        for name in (
            "warrant_probe_timeout_seconds",
            "miner_request_timeout_seconds",
            "challenge_request_timeout_seconds",
            "challenge_resolve_timeout_seconds",
            "scoring_request_timeout_seconds",
            "scorer_identity_timeout_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.orphan_sweep_age_seconds < 0:
            raise ValueError("orphan_sweep_age_seconds must be >= 0")
        if self.miner_max_input_bytes <= 0 or self.miner_max_output_bytes <= 0:
            raise ValueError("miner artifact byte bounds must be > 0")
        if not 0 < self.miner_port < 65536 or not 0 < self.metrics_port < 65536:
            raise ValueError("ports must be in (0, 65536)")
        return self
