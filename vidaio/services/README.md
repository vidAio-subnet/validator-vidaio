# vidaio.services — service skeleton and wire contracts

The shared foundation of every long-running process in the stack: `BaseService`
(lifecycle, config, logging, health, the exit-code contract) and
`protocol.py` (the HTTP wire contracts between services, including THE
SCORER-IDENTITY CONTRACT). Everything here is code as it exists today; the
per-service behaviour lives in the service modules that extend this one
(`vidaio/validator`, `vidaio/weightsetter`, `vidaio/chainsim`,
`vidaio/scoring_worker`, `an owner-operated module (not in this repo)`, `vidaio/miner`,
`an owner-operated module (not in this repo)`).

## What it does

- **`base.py`** — [DONE] `BaseService`: loads layered config (YAML +
  `VIDAIO__…` env overrides), pulls the `core` section, sets up structured
  JSON logging, starts a `HealthServer` (`/health` + `/metrics`), installs
  SIGINT/SIGTERM handlers, runs the subclass's `run()` loop, and turns the
  outcome into a process exit code via `run_service()`.
- **`protocol.py`** — [DONE] the request/response models exchanged between
  services (`MinerTaskRequest/Response`, `ScoreRequest/Response`), the
  canonical route list, the service port map, and the scorer-identity
  discovery helpers (`fetch_scorer_identity`, `fetch_scorer_identity_async`,
  `fetch_scorer_runtime_contract[_async]`,
  `scorer_runtime_contract_from_healthz`, `ScorerIdentityUnavailable`,
  `ScorerIdentityMismatch`, `ScorerRuntimeMismatch`). The runtime parser is an
  independent wire-policy implementation: it does not import the worker's
  attestation verifier.
- **`artifact_auth.py`** — [DONE] artifact-v2 canonical request/response signature
  domains, hotkey signer/verifier seams, fresh metagraph validator-permit check, and the
  bounded never-evict-live-nonces replay cache.
- **`miner_artifacts.py`** — [DONE] path-free streaming caller: safe input descriptor,
  signed request, bounded/digest-verified response staging, miner-signature verification,
  atomic publish, and cleanup on every failure/cancellation path.

## Design & decisions

### BaseService lifecycle

```
BaseService.main(config_path)
  -> load_raw_config + setup_logging
  -> run_service(cls(raw))
       -> asyncio.run(service.serve())
            -> health.start(); signal handlers; await self.run()
            -> finally: health.stop()
            -> raise FatalServiceError if fail_fatal() was called
       -> FatalServiceError => SystemExit(exit_code)   # non-zero
```

Subclasses implement `run()` (must return promptly once `self.stopping` is
set) and register health checks/metrics on `self.health`. `request_stop()`
asks for a cooperative stop; the injected signal handlers call it too.

### THE EXIT-CODE CONTRACT

The process exit code is the ONLY channel between a service and the
supervisor (`vidaio/validator/supervisor.py`), so it has exactly two
meanings:

| Exit code | Meaning | Supervisor action |
|---|---|---|
| `0` | Deliberate stop: SIGINT/SIGTERM, or an in-process `request_stop()` | Child stays stopped, never restarted |
| non-zero (incl. `FATAL_EXIT_CODE = 70`, and negative signal deaths) | Crash | Restart with bounded backoff; parked if crash-looping |

`70` (EX_SOFTWARE) is the distinctive value `run_service()` uses for a
declared fatal failure; any non-zero value means "crash, restart me".

**`fail_fatal(reason)` semantics.** A service that discovers it can no longer
do its job (the classic case: its uvicorn task died on its own, leaving a
live process with nobody answering its port) must NOT return normally from
`run()` — that would exit 0 and read as a clean shutdown, so the supervisor
would leave it down forever while `/health` looked fine. Instead it calls
`fail_fatal(reason)`, which:

1. records the reason (the FIRST reason wins — later cascading failures are
   symptoms);
2. lazily registers the `fatal_failure` health check, pinned `False`, so a
   scraper sees the failure before the process is gone (a healthy service's
   `/health` payload is untouched — the check only appears after a failure);
3. logs CRITICAL and calls `request_stop()`.

`serve()` then raises `FatalServiceError` after the loop unwinds, and
`run_service()` converts it into `SystemExit(exc.exit_code)` — THE one place
a fatal failure becomes a process exit code. It is idempotent and never
raises, so it is safe to call from any task.

### The wire protocol (`protocol.py`)

**Remote miner artifact exchange, hotkey-authenticated and digest-verified.** The canonical miner boundary is
path-free: `POST /v1/task/artifact` carries bounded base64url metadata plus raw streamed
input/output bytes. Production artifact v2 has a current validator hotkey sign canonical
task metadata, exact input digest/size, intended chain miner hotkey, timestamp, and
128-bit nonce. The miner verifies fresh validator permit and claims the nonce before body
ingress, then signs the complete request binding plus output digest/size/processing time.
The caller verifies that signature against the chain-attributed miner before publish;
callers never follow a miner path or URL. Replay state has global and per-validator live
entry caps and never evicts a still-valid nonce. It is process-local: one serving hotkey
must terminate at one ingress process, because shared-hotkey load-balanced replicas would
permit a captured request once per replica. A shared replay store is not implemented.
Each process also rejects timestamps at or before its start-plus-future-skew fence with
425 `artifact_auth_starting`; after the short startup blackout, a caller retries using a
fresh timestamp/nonce. This closes replay across an in-memory cache restart.
Miner ingress is additionally bounded by a
server-owned admission-through-receive/fsync clock (60 seconds by default, maximum 300),
which caller metadata cannot extend. Unsigned artifact v1 and the separate legacy JSON
path routes remain explicit report/test compatibility only, are disabled by default, and
are forbidden by the production guard. Within a co-located authority/scoring deployment, challenge and
scoring contracts still use validator-owned absolute paths plus sha256. The scoring worker
verifies each digest through a single symlink-refusing descriptor and scores an immutable
private COPY taken from that descriptor, so a post-request rewrite of the
caller-named path cannot change what was measured (symlinks/fifos/devices/
directories are refused with 422).

**Service port map** (each value is the service's own config default):

| Service | API port | Metrics port |
|---|---|---|
| scoring worker | 8201 | 9103 |
| challenge service | 8210 | 9105 |
| reference miner | 8300 | 9106 |
| organic gateway | 29996 | 9107 |
| chainsim | 8400 | 9108 |
| inference validator | — (no API) | 9101 |
| weight-setter | — (no API) | 9102 |
| competition orchestrator | 8500 (control) | 9104 |
| dashboard | 8600 | 9109 |
| autoupdater | — (loop) | 9110 |
| scoring authority | 8700 | 9111 |
| audit-results API | 8710 | 9112 |
| baseline registry | 8720 | 9123 |
| authority-finalizer | — (loop) | 9120 (Bittensor production) |
| beacon auditor | — (loop) | 9121 (Bittensor production) |
| own-auditor | — (loop) | 9122 (Bittensor production) |

(`core.metrics_port` defaults to 9100; each service overrides it with its
own section's `metrics_port`. The script-owned finalizer/auditor loops use
explicit `local_stack` ports in Bittensor mode and ephemeral port 0 in report mode.)

Production processes are launched from the release image with
`python scripts/service_entrypoint.py <role>`. The default composition is
`config/default.yaml` plus environment overrides, matching production preflight; pass
the same `--config PATH` to both when using an explicit deployment overlay.
`--compose-dev` is exclusively the report/local Compose overlay. The target bundled roles
are `authority-node` (central inference measurement + pointer API + finalizer) and
`thin-validator-node` (weight-setter only). Beacon and full own-audit run as distinct
standalone processes/containers, each with its own durable cursor and pending-report
state, so neither shares the weight process or cgroup; `validator-node` is the existing
combined single-host/Compose unit. Bundles preserve coherent local SQLite relationships
and fail the unit if a critical child permanently parks. Standalone inference-validator,
weight-setter, authority API/finalizer, beacon-auditor, and own-auditor roles exist for explicit one-host
orchestration. See
the project design record for the role/state map. The public miner
runs least-privileged as `python scripts/service_entrypoint.py reference-miner`; it keeps
its own serving hotkey-only wallet for response signing, and its advertisement job uses
that same miner-specific identity. It never receives a validator wallet or S3 secrets.

**Canonical routes** — one path per contract; every client uses these:

| Route | Contract |
|---|---|
| `POST {miner}/v1/task/artifact` | Canonical production artifact v2: validator-hotkey-signed bounded metadata + streamed raw input → miner-hotkey-signed streamed output with version/task/size/sha256/timing bindings. Validator and gateway dispatch here. |
| `POST {miner}/v1/task`, `/task` | DEPRECATED JSON absolute-path routes; absent unless a local test explicitly enables them, and forbidden in production. |
| `GET {miner}/warrant` | `-> {"track": ...}` — the TaskWarrant probe: the one pool that miner identity competes in. |
| `POST {scoring_worker}/score` | `ScoreRequest -> ScoreResponse`. |
| `GET {scoring_worker}/healthz` | `-> {"scorer_version": <identity>, "runtime_commitment": ...}` — scorer plus complete public payout-runtime discovery. |
| `POST {challenge_service}/challenge/next` | `{track, owner} ->` the produced challenge (miner-facing dispatch payload + validator-private reference/input artifacts). |
| `GET {challenge_service}/challenges?status=dispatched&older_than_seconds=N` | the dispatched-challenge sweep list — the validator's blind-spot closer for a lost `/challenge/next` response. |

### THE SCORER-IDENTITY CONTRACT

A score packet is only auditable if everyone agrees on WHICH scorer produced
it. The scoring worker has exactly one name, and it is the worker's to mint:

```
identity = f"{scoring_worker.scorer_version}+{identity_digest[:12]}"
```

`identity_digest` (`vidaio.scoring_worker.service.scorer_identity_digest`)
is a sha256 over every configured scoring lever and the complete payout-runtime
commitment: verified release manifest/image marker, Linux/amd64 and fixed CPU
thread policy, native media tools, pinned PieAPP weights, and exact Python metric
packages. Its full digest is also a required measured-packet backend stamp.
Ports, paths, timeouts and concurrency remain excluded because they cannot
change a packet.

**The identity is the FULL effective string** `<name>+<digest12>` — never the
bare configured name. A bare name (e.g. `"scorer-v1"`) is not an identity and
any caller asserting one is refused. The worker publishes its identity on
`GET /healthz` (field `scorer_version`, plus the public runtime-commitment
preimage) and stamps it into every packet it emits. `ScoreRequest.scorer_version`
is a caller ASSERTION, never an
instruction: absent/empty means "whichever scorer you are", an equal value
means agreement, anything else is `409 scorer_version_mismatch`.

Consumers adopt it as follows:

In Bittensor mode, all three consumers independently derive an expected contract
from their own marker-qualified release image and exact-match the live health
identity, digest, complete attestation, and complete backend map before allowing
earning work. A self-consistent remote claim is insufficient. The only opt-out is
an explicit constructor seam named `allow_noncanonical_runtime_for_report_or_tests`;
production composition never sets it.

1. **Inference validator — PIN ON FIRST CONTACT.** Discovers the identity
   from `GET /healthz`, pins it (durably — see
   [`vidaio/validator/README.md`](../validator/README.md)); every returned
   packet's `scorer_version` must equal the pin. A non-empty
   `validator.scorer_version` is an explicit operator pin: asserted on the
   wire AND checked against discovery, else startup fails loudly.
2. **Competition orchestrator — the manifest commits to the identity.**
   `CompetitionManifest.scoring_version` must be the full effective identity
   (the manifest digest is anchored on chain before enrollment); a
   disagreement is an INFRA HALT, never a FAILED competition. Every measured
   response is then fail-closed against the exact request fields and the
   independently re-hashed health runtime commitment: packet
   `backend_versions` must equal the complete advertised payout-backend map plus
   its derived runtime stamp. A replayed identity/content packet, omitted stamp,
   moved backend, or CUDA scorer packet never reaches persistence/ranking.
3. **Challenge service — the commitment preimage names the scorer.**
   `challenge_service.scorer_version` is bound into every challenge
   commitment's preimage; when `scoring_worker_url` is configured the value
   is verified against the live worker (mismatch: CRITICAL + refuse to
   produce, 503). Bittensor mode requires that URL and exact runtime check;
   the unverified literal remains report-only.

Nobody ever rewrites the worker's stamp: a MEASURED packet's
`scorer_version` is always the value the worker minted — that is what makes
the later cross-check (packet vs bundle vs manifest vs commitment)
meaningful.

**Reserved namespace: `orchestrator-zero/*`.** Exactly one packet kind is
NOT minted by a scoring worker: when a competition item has no measurable
bytes, the orchestrator records a gate-failed ZERO locally
(`gate_passed=False` forces `score=0.0` structurally) attributed honestly as
`scorer_version = "orchestrator-zero/1+<digest12>"`. Consequences:

- such a packet's `scorer_version` legitimately DIFFERS from the manifest's
  `scoring_version` — an orchestrator fact, not scorer drift; its audit
  bundle carries the same orchestrator-zero identity while the bundle's
  manifest still names the committed worker;
- NO scoring worker may advertise or stamp an identity in the
  `orchestrator-zero/` namespace; a worker that does is refused, because it
  would make measured packets indistinguishable from these records.

Helpers: `vidaio.competition.orchestrator.zero_packets`
(`is_orchestrator_zero_identity` / `assert_not_reserved`).

## Public API & endpoints

Python surface (re-exported from `vidaio.services`):

- `BaseService` — subclass with `name`, `run()`; use `self.stopping`,
  `request_stop()`, `fail_fatal(reason)`, `self.health`, `self.log`,
  `self.core`.
- `run_service(service)` — the entrypoint wrapper that applies the exit-code
  contract; every entrypoint (including the local-stack supervisor children)
  goes through it.
- `FatalServiceError`, `FATAL_EXIT_CODE` (70), `FATAL_CHECK_NAME`
  (`"fatal_failure"`, defined in `base.py`).
- From `protocol.py`: `MinerTaskRequest`, `MinerTaskResponse`,
  `ScoreRequest`, `ScoreResponse`, `SHA256_HEX`, `SCORER_IDENTITY_ROUTE`
  (`/healthz`), `SCORER_IDENTITY_FIELD` (`scorer_version`),
  `fetch_scorer_identity[_async]`, `scorer_identity_from_healthz`,
  `ScorerIdentityUnavailable`, `ScorerIdentityMismatch`.
- From `artifact_auth.py`: `ArtifactClientAuth`, `ArtifactServerAuth`,
  `ArtifactRequestClaims`, hotkey signer/validator-registry protocols and adapters,
  `BoundedReplayCache`, canonical signature helpers, and typed auth failures.

This module serves no HTTP endpoints of its own; `HealthServer` (from
`vidaio.core`) serves `/health` and `/metrics` for each service instance.

## Data & invariants

- No database. State is per-process (`_stop` event, `_fatal_reason`).
- The first `fail_fatal` reason wins and is immutable; once the
  `fatal_failure` check is registered it can never pass again.
- A `ScoreResponse` carries the exact packet bytes (`item_score_json`) plus
  `packet_digest` = sha256 of those bytes; receivers recompute and compare.
- `MinerTaskRequest.params` may carry track params (upscale factor, bitrate
  cap) but never seeds/DAG material.
- `ScorerIdentityUnavailable` must be treated as "unknown", never as
  "whatever I had configured" — the worker, not the caller, names the scorer.
  A DEGRADED worker (503) still answers identity discovery: the body is
  parsed regardless of status code.

## Configuration

This module owns no config section. `BaseService` consumes the shared
`core` section (schema `vidaio/core/config.py::CoreConfig`):

| Key | Default | Meaning |
|---|---|---|
| `core.data_dir` | `./data` | Data directory (DB lives here) |
| `core.db_filename` | `vidaio.db` | SQLite filename under `data_dir` |
| `core.log_level` | `INFO` | Logging level |
| `core.metrics_port` | `9100` | Health/metrics port (services override with their own section's `metrics_port`) |
| `core.network` | `finney` | Chain network label |
| `core.netuid` | `85` | Subnet netuid |

Env override pattern (applies to every section in the repo):
`VIDAIO__<SECTION>__<KEY>=<value>` — e.g. `VIDAIO__CORE__LOG_LEVEL=DEBUG`.
Values are YAML-parsed. See `config/default.yaml` and the root
[`README.md`](../../README.md).

## How to test

```sh
python -m pytest the development-tree exit-contract test   # the exit-code contract
python -m pytest tests/core                                 # config/db/metrics/resilience
```

The scorer-identity contract is exercised across
`tests/scoring_worker/test_scorer_version.py`,
`tests/validator/test_scorer_identity.py`,
`tests/challenge_service/test_scorer_identity.py`,
`tests/orchestrator/test_scorer_identity.py` and
`tests/orchestrator/test_zero_packet_identity.py`.

## How to change safely

- Never add a "we cannot continue" path that returns normally from `run()` —
  it must end in `fail_fatal` (non-zero exit), or the supervisor will file it
  as a deliberate stop. `the development-tree exit-contract test` guards
  this.
- Route changes in `protocol.py` are cross-service contract changes: keep one
  canonical path per contract, and treat the deprecated `/task` alias as
  frozen. Update every client and the port table together.
- Anything added to the scorer-identity digest changes every worker's
  effective identity and therefore invalidates pins, manifests and
  commitment preimages — only add levers that genuinely change a measured
  score.
- The `orchestrator-zero/` namespace is reserved; do not mint identities in
  it anywhere except the orchestrator's zero-packet path.

## Status & gaps

- [DONE] BaseService lifecycle, exit-code contract, fail_fatal, health/
  metrics wiring, protocol models, scorer-identity helpers.
- [DONE, needs multi-host validation] Remote miner exchange is streamed, bounded, and
  path-free. Artifact v2 binds both directions to current validator/expected miner
  hotkeys with freshness and bounded replay protection. Validator/gateway publish into
  caller-owned directories only after binding task/digest/size/signature. Both clean
  partial/rejected downloads; the validator cleans accepted round artifacts after
  scoring/archive, while the gateway retains accepted completed results under its job
  policy. Live testnet/multi-host behavior is not yet claimed; unsigned v1 remains
  report/test-only compatibility.
