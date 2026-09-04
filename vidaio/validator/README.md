# vidaio.validator — inference validator (round loop, registry, supervisor)

The synthetic scoring round loop (`inference.py::InferenceValidator`), the
SQLite miner registry + round ledger (`miner_manager.py`), the score-packet
evidence reader (`evidence.py`), and the process-isolation supervisor
(`supervisor.py`). Weights are NOT set here — the weight-setter is its own
supervised process ([`vidaio/weightsetter/README.md`](../weightsetter/README.md));
the database is the only shared state.

## What it does

One round (`run_round`), in order:

1. **Scorer-runtime gate** — ensure a pinned scorer identity exists and, in
   Bittensor mode, exact-match the worker's health identity, digest, complete
   attestation and backend map to a contract independently derived from this
   process's marker-qualified release image. No pin or a latched identity/runtime
   conflict → the round is SKIPPED with a structured reason.
2. **Chain refresh (throttled) + staleness gate** — a FAILED refresh is
   never recorded as a success; a stale/unavailable snapshot SKIPS the round
   (`chain_snapshot_stale` / `chain_snapshot_never_refreshed` /
   `chain_state_unavailable`) instead of reporting a successful empty round.
3. **Filter + dedup miners** — registered identities with
   `alpha_stake >= validator.min_stake`, then first-uid-per-IP and
   per-coldkey, uid ascending (`dedup_miners` — the same deterministic rule
   as `tokenomics.rank_curve.eligible_for_ranking`, minus the
   positive-score gate so new miners still get dispatched to).
   `validator_permit` is a non-exclusive capability and is never a miner filter:
   an earning miner can acquire a permit without disappearing from scoring. A
   permit-holder must have an explicit chain-advertised, specified-IP axon to enter
   dispatch; this keeps no-axon authority/control hotkeys census-only and prevents
   them from consuming a real miner's pre-warrant dedup slot.
4. **Warrant probe** (the fixed TaskWarrant) — miners whose (staged) track
   is unknown are probed via `GET /warrant`; results are STAGED, not
   written.
5. **Per track: challenge fetch** — `POST /challenge/next` stamped with
   `owner=validator.identity`; the in-flight obligation is persisted BEFORE
   anything that can fail, WITH the owner the fetch actually carried.
6. **Dispatch** — streamed artifact-v2 `POST /v1/task/artifact` to every eligible miner
   concurrently, with a VALIDATOR-AUTHORED task id
   (`task_id_for = "{challenge_id}:{uid}"`). The validator wallet signs the canonical
   task metadata, expected chain miner hotkey, timestamp, 128-bit nonce, and exact input
   size; the miner proves current validator permit before accepting the body. A response
   echoing any other id becomes a signed availability zero when the exact request
   evidence is available.
7. **Artifact bind** — stream into the validator-owned `miner_artifact_dir` under
   independent input/output caps; artifact version, task id, size, sha256, and the
   chain-attributed miner's response signature must all match before atomic publish.
   The signature binds the complete signed request plus response digest, size, and
   processing time. No peer path/URL is followed; miner-attributable protocol and
   digest failures become request-bound availability zeros.
8. **Response-time dedup** — byte-identical signed outputs use the
   independently reproducible `anchor_hash_hotkey/1` order
   (`scoring.dedup_responses`, exact output SHA-256 equality). The
   winner is the lowest domain-separated hash of finalized challenge-anchor block
   hash plus miner hotkey, never first arrival. A loser is zeroed only with both
   signed outputs and the content-addressed duplicate witness preserved for audit.
9. **Score** — `POST /score` per kept response (idempotent recompute →
   retried); the returned packet's digest is recomputed, the packet parsed,
   its complete backend map must equal the health-bound local runtime contract,
   and **bound to the request**: `challenge_id`, `item_id`,
   `miner_hotkey`, `track`, `content_digest` and the PINNED
   `scorer_version` must all equal what this validator sent/expects. An
   unbound packet is rejected (replay/MITM guard), counted per field, and
   is NON-punitive — the miner is skipped for the round, not zeroed.
10. **Evidence** — the exact packet bytes + digest are persisted per
    (round, uid, item), and archived as `SCORE_PACKET` artifacts when an
    audit store is configured — FAIL-CLOSED on archive failure. Availability
    zeros persist their canonical validator-signed observation separately.
11. **EWMA accumulate + atomic commit** — one `BEGIN IMMEDIATE`
    transaction applies the whole round (see below).
12. **Resolve challenges** — `finally`: every fetched challenge is resolved
    with the challenge service (success, failure, timeout AND shutdown).

Failure discipline: every external await is `with_timeout`-wrapped;
idempotent reads get `retry_async`. Timeout, transport, protocol, task-id, digest,
receipt, deadline-exhausting restart-fence, and unusable chain-endpoint failures fold
zero only when a canonical observation binds the exact validator-signed request,
finalized challenge anchor, target hotkey/endpoint and deadline. This prevents
response or advertisement cherry-picking from freezing EWMA. The negative network
fact is still a validator observation, not a Byzantine proof; its immutable signature
makes the trust boundary explicit and disputable. Scoring-worker, audit-store, chain,
challenge-service, local-input and validator signing failures remain excused:
validator infrastructure trouble must not punish miners.

## Design & decisions

### The TaskWarrant fix (unknown track = skipped, never defaulted)

A miner's track exists ONLY as an explicit warrant-probe result. A probe
timeout, a missing record or a garbage value leaves `track = NULL` and the
round loop SKIPS the miner with a structured log and the
`vidaio_validator_skipped_unknown_track_total` metric. This deliberately
replaces the old `validator.py:844-849` confirmed bug, where any of those
cases was silently bucketed as upscaling and a real compression miner got
mis-scored. There is structurally no way to write a default:
`record_track` refuses anything outside
`KNOWN_TRACKS = ("compression", "upscaling")`, and `normalize_track` maps
everything else to `None`.

### Round atomicity (staged registry + one transaction + rounds ledger)

A round's whole OBSERVABLE state — registry sync (hotkey-change purges, new
miners, ip/coldkey updates), warrant-probe tracks,
EWMA folds, packet evidence, and the round-ledger stamp — is ONE
`BEGIN IMMEDIATE` transaction (`miner_manager.commit_round`). During the
round, registry effects are STAGED in memory (`RegistryUpdate`) and the loop
reads the post-sync view through `planned_tracks` without writing anything.
The only pre-commit write is the `rounds` marker row (`begin_round`,
`committed_at NULL`), which carries no miner state.

Consequences: the independently-running weight-setter can never read a
half-applied round (e.g. a hotkey reset from a round that then died); a
crashed round is both invisible (rolled back) and DETECTABLE afterwards
(`committed_at` stays NULL); readers — `ScorePacketEvidence`, the
weight-setter — ignore uncommitted rounds. `commit_round` raises
`RoundLedgerError` on a double commit or a commit without `begin_round`.

A hotkey change on a uid is a NEW miner: EWMA accumulator purged, track
reset to NULL (re-probe required) — no phantom history carries over.

### Evidence persistence (packets → PublicationInputs)

Persisting only numeric EWMAs made published weights unreproducible (every
real publication anchored the "no score packets" sentinel — a violation of
the the project design record integrity invariant "every scored metric must be independently
recomputable from the audit store"). Now:

- the exact packet bytes + `packet_digest` land in `score_packets` inside
  the round's transaction, keyed (round_id, uid, item_id), with the request
  bindings they were accepted against;
- with an audit store configured (`store=` injected), each packet plus the challenge
  input, miner output, and reference original is archived. Their refs are committed in
  `score_packets` (migration `0006_audit_media_refs.sql`) so the finalizer can build a
  genuinely recomputable `POST_RETIREMENT` bundle. **Any archive failure fails the ITEM
  closed**: the score is not
  accumulated and no evidence row claims an archive that does not exist
  (`AuditStoreFailure`; non-punitive to the miner). A stored digest that
  differs from the verified one is likewise refused.
- DB-only operation (no store configured at all) is a report/test-only,
  explicitly flagged mode: warned at startup, named in every round log
  (`evidence_mode=db_only`) and exported on
  `vidaio_validator_audit_store_configured`.

`evidence.py::ScorePacketEvidence` is the reader: only COMMITTED rounds,
digests sorted (reproducible merkle roots), and it is structurally a
`vidaio.weightsetter.PublicationInputs` (`score_packet_digests()` over a
lookback window, default 24 h; `recent_packet_digests(since)` for the
watermark path). The weight-setter constructs it over its OWN connection to
this database file — never over the validator's.

### The scorer pin (durable, conflict-latching, operator-reset)

The identity every accepted packet must carry (services.protocol,
THE SCORER-IDENTITY CONTRACT — see
[`vidaio/services/README.md`](../services/README.md)):

- **Pin on first contact**, persisted in the single-row `scorer_pin` table
  (migration 0003) so it survives restarts. `record_scorer_pin` refuses to
  overwrite a different pin. A pin that cannot be persisted refuses to score
  (an unpersisted pin cannot protect the accumulators across a restart).
- A restart reloads the pin but still VERIFIES it against the live worker
  once per process — a silent worker swap must not stay invisible.
- **Disagreement latches** `scorer_pin_conflict`: rounds skip
  (`scorer_pin_conflict` reason), `/health` check `scorer_pin` fails, the
  `vidaio_validator_scorer_pinned` gauge goes 0, and the process stays UP
  (crash-looping would take the health surface down). It is never
  auto-cleared.
- `validator.scorer_version` non-empty is an explicit OPERATOR pin: asserted
  on the wire (worker 409s a stranger) AND checked against discovery —
  disagreement at startup raises (config error, process refuses to start).
  Under pin-on-first-contact, requests OMIT `scorer_version`; the packet's
  own stamp is bound either way.
- `validator.reset_scorer_pin: true` is the operator acknowledgement: clears
  the persisted pin ONCE at startup (logged CRITICAL) so the next discovery
  re-pins. It does NOT reset accumulators built under the old scorer.

### Challenge resolution obligations

Fetching a challenge consumes a pool asset and leaves it `in_use` with its
commitment unrevealed until `POST /challenge/{id}/resolve`. Therefore:

- every fetch writes an `inflight_challenges` row BEFORE anything that can
  fail, with outcome `'expired'` (flipped to `'resolved'` only once the
  track's scoring completed) and the OWNER the fetch actually carried
  (migration 0004);
- the drain runs in the round's `finally`, on shutdown, and in the startup
  recovery pass (`recover_inflight_challenges`); rows survive a failed
  resolve and are retried, so one flaky call can never permanently strand
  the pool. Unknown/already-terminal (404/409) rows are dropped, not
  retried;
- recovery resolves as the RECORDED owner, not the current config — an
  identity rotation between fetch and restart would otherwise be 403'd
  forever. A rotation logs a WARNING; a genuine
  `403 not_owner` (`ChallengeOwnershipRefused`) is PERMANENT: counted on
  `vidaio_validator_challenge_resolve_forbidden_total` and PARKED durably
 — no number of retries can move an
  ownership boundary, so the row is excluded from every later drain and
  restart instead of ringing the same alarm each round. The row is kept
  (it is the only record that an asset is stranded) and stays visible:
  the `vidaio_validator_parked_challenges` gauge,
  `miner_manager.parked_challenges()`, and a WARNING in the startup
  recovery pass listing every parked id + reason. The operator's way out,
  after fixing (or accepting) the service-side ownership state, is
  `validator.unpark_challenges = true` at startup or the
  `unpark_challenges()` admin method — both return every parked row to the
  normal drain; a refusal that still stands re-parks on its next 403;
- the **orphan sweep** closes the one remaining blind spot: a lost
  `/challenge/next` RESPONSE leaves a dispatched challenge we never learned
  the id of. Startup asks
  `GET /challenges?status=dispatched&older_than_seconds=N`
  (`orphan_sweep_age_seconds`, comfortably longer than a round) and expires
  anything with no in-flight row of ours — but ONLY challenges the service
  attributes to `validator.identity`. No identity configured → sweep
  disabled outright; a listed row that does not NAME our owner is never
  swept (`unattributed` / `foreign_owner`), because an unknown query
  parameter is silently ignored by most HTTP frameworks and an unfiltered
  list must never be mistaken for a filtered one.

### Retention windows (REMOVED)

The block-driven retention-window bookkeeping (`observe_retention` /
`latest_full_window` / the `retention_windows` table / the
`retention_window_blocks` config) was REMOVED with the retention multiplier
for v1 — retention removed — owner decision; an internal review. It fed the removed `MinerSnapshot` windowed fields,
which no longer affect weight. `snapshot()` now maps registry state alone onto
`MinerSnapshot`s, excluding unknown-track miners entirely.

### The supervisor (`supervisor.py`)

Process isolation (spec §13, SN44's core safety pattern): every heavy/unsafe
service runs as a SEPARATE OS process (multiprocessing `spawn` context — no
forked locks/loops), so a segfault/OOM in one child can never take down the
weight-setter. The supervisor monitors `Process.is_alive` + exit codes and
applies THE EXIT-CODE CONTRACT
([`vidaio/services/README.md`](../services/README.md)):

- `exitcode 0` = deliberate stop → marked STOPPED, never restarted;
- any non-zero exitcode (signal deaths, unhandled exceptions,
  `FATAL_EXIT_CODE` 70) = crash → restarted with bounded exponential
  backoff (`base 0.5 s, doubling, cap 30 s` by default);
- a child that exhausts its restart budget (default 5 restarts inside a
  600 s window) is PARKED — logged CRITICAL, everything else keeps running;
- shutdown fans out SIGTERM: terminate → join (5 s default) → kill
  stragglers.

Children are generic `(name, "module:callable", config)` specs, resolved at
construction (fail fast on a typo); timing knobs and the clock are
injectable. `the development-tree stack runner` runs the whole stack under this
supervisor.

## Public API & endpoints

No HTTP API — the validator exposes only `/health` + `/metrics` (port
`validator.metrics_port`, default 9101). Health checks: `db`,
`last_round_age` (allowance `2 × cycle_sleep_max + 600 s`), `scorer_pin`.
Health checks run on the HealthServer thread and get their own sqlite
connection (per-thread; `:memory:` falls back to the loop's handle).

Python surface (re-exported in `__init__.py`): `InferenceValidator`,
`ValidatorConfig`, the client Protocols (`ChallengeClient`, `MinerClient`,
`ScoringClient`) and their HTTP implementations (`HttpChallengeClient`,
`HttpMinerClient`, `HttpScoringClient`), `RoundReport`, `PacketEvidence`,
`ScorePacketEvidence`, the miner-manager functions
(`sync_neurons`, `commit_round`, `planned_tracks`, `record_track`,
`snapshot`, the inflight/scorer-pin helpers), and
`Supervisor` / `ChildSpec` / the `STATE_*` constants.

Outbound wire calls (contracts in `vidaio/services/protocol.py`):
`GET {miner}/warrant`, hotkey-authenticated artifact-v2
`POST {miner}/v1/task/artifact` (with optional extra `X-Miner-Token` when
`validator.miner_api_token` is set), `POST {scoring_worker}/score`,
`GET {scoring_worker}/healthz`, and the challenge-service routes
(`/challenge/next`, `/challenge/{id}/resolve`, `/challenges`), all with
`Authorization: Bearer <validator.challenge_service_token>`.

In Bittensor mode `chain.validator_hotkey`/wallet is the artifact request signer,
`validator.identity` must match it, and each `ChainNeuron.hotkey` is the expected miner
response signer. There is no unsigned production fallback. The optional shared miner
token is appropriate only for a controlled fleet; permissionless miners leave it unset
and rely on artifact-v2 hotkey authentication plus a hardened TLS/rate/connection edge.

## Data & invariants

Database: the shared core DB (`core.data_dir/core.db_filename`, default
`./data/vidaio.db`); migrations in `migrations/`:

- `0001_validator.sql` — `miners` (uid PK, hotkey, track NULLable +
  CHECK-constrained, `accumulate_score`, `first_seen_block`). (The
  `retention_windows` table was REMOVED with the retention multiplier —
  retention removed — owner decision; an internal review.)
- `0002_round_ledger_and_evidence.sql` — `rounds` (committed_at NULL until
  the atomic commit), `score_packets` (exact bytes + digest + bindings +
  `audit_ref`), `inflight_challenges`.
- `0003_scorer_pin.sql` — single-row `scorer_pin` (id = 1), never
  rewritten; cleared only by explicit operator acknowledgement.
- `0004_inflight_challenge_owner.sql` — `inflight_challenges.owner`; rows
  predating it have `''` and resolve as the current identity (exactly what
  they meant when written).
- `0005_inflight_challenge_parked.sql` — `inflight_challenges.parked_at` /
  `park_reason`: a genuine ownership 403 parks the row out of the
  drain/recovery selection, durably; NULL = live (the shape every
  pre-migration row already has). Unparked only by explicit operator
  acknowledgement (`validator.unpark_challenges` / `unpark_challenges()`).
- `0009_availability_folds.sql` — canonical signed observations backing explicit
  score-zero EWMA folds for the closed miner-failure taxonomy.
- `0010_availability_endpoint_failures.sql` — extends that closed taxonomy for an
  authenticated restart fence that consumes the whole deadline and an unusable
  chain-advertised endpoint; both now decay instead of freezing a prior EWMA.

Invariants worth defending:

- `scores`/EWMA use `tokenomics.accumulate` verbatim (decay from the shared
  `tokenomics.ewma_decay`) — the validator has no private EWMA math to
  drift from the weight composition; the `-1` exclusion sentinel passes
  through untouched.
- `score_packets.packet_digest` is the merkle leaf a publication commits
  to; `scorer_version` in evidence is the OBSERVED stamp (what actually
  ran), never what was asked for.
- Timestamps are `utc_now_iso()` (`+00:00`), so ISO strings sort as
  instants (the evidence reader compares them as strings).
- All round dependencies are Protocols; a full deterministic round runs
  in-process against `InMemoryChain` + fake clients in tests.

## Configuration

Section: `validator` (schema `config.py::ValidatorConfig`,
`extra="forbid"`). Env override pattern: `VIDAIO__VALIDATOR__<KEY>=<value>`.
EWMA decay deliberately lives in the shared `tokenomics` section, not here.

| Key | Default | Meaning |
|---|---|---|
| `cycle_sleep_min_seconds` / `cycle_sleep_max_seconds` | `3600` / `7200` | Sleep between rounds = `rng.uniform(min, max)` (spec §01) |
| `metagraph_refresh_seconds` | `1800` | Chain refresh throttle |
| `max_chain_snapshot_age_seconds` | `3600` | Staleness gate; 0 disables |
| `identity` | `""` | This validator's identity (its hotkey): `owner` on fetch/resolve, the orphan-sweep boundary. Empty DISABLES the sweep |
| `min_stake` | `0.0` | Alpha-stake dispatch floor |
| `scoring_worker_url` | `http://127.0.0.1:8201` | Scoring worker base URL |
| `challenge_service_url` | `http://127.0.0.1:8210` | Challenge service base URL |
| `miner_url_scheme` | `http` | Fleet-wide `http`/`https` miner transport; axon records contain only IP/port, so advertisement and live preflight must match this setting |
| `miner_port` | `8300` | Fallback miner API port; a valid chain-advertised `ChainNeuron.axon_port` is preferred |
| `challenge_service_token` | `""` | Bearer for EVERY challenge-service route (that service fails closed — required in any multi-process deployment) |
| `miner_api_token` | `""` | Optional controlled-fleet `X-Miner-Token`; permissionless artifact v2 uses hotkey signatures instead |
| `miner_artifact_dir` | `./data/validator/miner-artifacts` | Validator-owned landing zone for verified remote responses; deleted after scoring/archive on every path |
| `miner_max_input_bytes` | `2147483648` (2 GiB) | Hard request-input cap checked before and during streaming |
| `miner_max_output_bytes` | `4294967296` (4 GiB) | Hard miner-output cap checked before and during streaming |
| `allow_non_public_miner_addresses` | `false` | Private/DNS peer opt-in for local/report use; production requires globally routable literal axon addresses |
| `warrant_probe_timeout_seconds` | `10` | Warrant probe; timeout ⇒ track unknown ⇒ SKIP |
| `miner_request_timeout_seconds` | `300` | Full task round-trip; a request-bound timeout folds an explicit availability zero |
| `challenge_request_timeout_seconds` | `30` | Challenge fetch/sweep |
| `challenge_resolve_timeout_seconds` | `30` | `/challenge/{id}/resolve` |
| `scoring_request_timeout_seconds` | `600` | Scoring (VMAF/PieAPP need real time) |
| `orphan_sweep_age_seconds` | `3600` | Sweep threshold; 0 disables |
| `scorer_version` | `""` | Explicit operator pin (full `<name>+<digest12>`); empty = pin on first contact |
| `scorer_identity_timeout_seconds` | `15` | Identity discovery call budget |
| `reset_scorer_pin` | `false` | Operator acknowledgement to clear the durable pin once |
| `unpark_challenges` | `false` | Operator acknowledgement to return PARKED (403'd) in-flight rows to the startup drain once |
| `metrics_port` | `9101` | Health/metrics port |

Construction wiring (not config): `chain=` (a `ChainAdapter`), optional
`store=` (audit store — inject in production), `conn=`/`conn_factory=`,
seeded `rng=` for deterministic sleep jitter. See
`the development-tree stack runner` for the real wiring and
`vidaio/validator/DEPS.md` for the wanted config/deps additions.

## How to test

```sh
python -m pytest tests/validator
```

By concern: `test_round.py` (pipeline + atomicity),
`test_miner_manager.py` (registry, tracks),
`test_packet_binding.py`,
`test_scorer_identity.py` (pin/conflict/reset), `test_evidence.py`
(fail-closed archiving + PublicationInputs), `test_challenge_resolution.py`
(inflight drain, owner recovery, sweep boundaries), `test_chain_state.py`
(staleness gate), `test_supervisor.py` (exit-code contract, backoff,
parking), `test_availability.py` (signed request/observation bindings and closed
reason taxonomy), `test_config.py`, `test_http_clients.py`. Full-loop:
`the development-tree e2e suite` and `the development-tree stack runner`.

## How to change safely

- Anything a round writes must go through the staged
  `RegistryUpdate`/`commit_round` path — a write that commits mid-round
  reintroduces the partial-round reads an internal review fixed.
- Never introduce a default track, and never derive a scored id from
  anything the miner echoed — both are regressions of confirmed bugs with
  dedicated tests.
- Keep the evidence discipline: miner-attributable availability zeros require the
  exact signed request and canonical validator-signed observation. The reason enum is
  closed; never add scorer/storage/chain/challenge failures to it. A duplicate may
  zero a loser only when both miner-signed outputs, receipts, finalized anchor hash,
  byte-exact SHA-256 equality and the deterministic `anchor_hash_hotkey/1` witness
  are public and independently recomputable. Validator-side faults skip accumulation.
- New in-flight obligations must be persisted BEFORE the fallible work and
  drained on every exit path including startup recovery.
- Schema changes are new files in `migrations/`; the weight-setter reads
  this database file directly, so treat `score_packets`/`rounds` columns as
  a cross-process contract.
- Repo-wide: bump the root `VERSION` on any release-worthy change.

## Status & gaps

- [DONE] Round pipeline, TaskWarrant checks, atomic rounds + ledger, packet/media
  binding and fail-closed persistence, close-block miner-state history, durable scorer
  pin, challenge resolution/recovery/sweep with ownership, and supervisor.
- [DONE, needs live-bucket validation] Production composition injects the configured
  S3-compatible store and encrypted holdout envelope. Terminal challenge resolution
  publishes the verified reference at its public released key; DB-only mode remains an
  explicitly flagged report/test posture.
- [DONE, needs testnet validation] Real-chain reads/signing are supplied by
  [`BittensorChainAdapter`](../chain/README.md); deployment must set
  `validator.identity`, wallet identity, and challenge/API tokens consistently.
- [DONE, needs deployed multi-host validation] Miner exchange is bounded streamed bytes,
  mutually hotkey-signed, replay/time-window guarded, task/digest/size bound,
  SSRF-hardened, and cleanup-safe. Production accepts globally routable literal axon
  IPs, brackets IPv6, and preserves/prefers real axon ports; local overlays explicitly
  opt into private/DNS peers. Unsigned artifact v1 and the JSON path routes are explicit
  report/test compatibility modes only.
- Note: `config/default.yaml`'s `validator:` section sets only the
  credential/identity/pin keys; every other default (cadence, timeouts)
  lives in `config.py` — that is the intended layering, not a
  gap.
