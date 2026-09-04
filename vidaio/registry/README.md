# vidaio.registry — schema-v14 executable-baseline ledger

This package owns the durable executable identity for the compression and
upscaling tracks. Each track has exactly one active baseline, starts from an
explicit archived version zero, and advances only from independently verified,
anchored CROWN evidence. The finite reward window and the executable baseline
are separate state machines: an earning window expires, while an active baseline
persists until another verified activation or an append-only rollback.

The executable bytes live in the content-addressed audit store. SQLite contains
their digest, byte count, build-image digest, source repository/git identity,
provenance-manifest identity, and the anchored evidence that activated them.
This is enough for an independent party to fetch the exact public bytes and
rebuild the same implementation; a mutable filename or operational rank is never
an executable identity.

## Shipped components

- `baseline.py` is the schema-v14 ledger model: version-zero seeding, active and
  historical reads, invariant checks, pending-promotion interlocks, append-only
  rollback records, and event queries.
- `baseline_promotion.py` is the verified CROWN pipeline. Its caller supplies a
  snapshot digest only. Winner, track, score, prior baseline, source archive,
  audit matrix, and chain anchor are derived through a trusted epoch source and
  rechecked before a fresh build/rerun can activate anything.
- `crown_source.py` defines the immutable verified-epoch DTO and the internal
  `VerifiedCrownEpochSource` protocol. A source must authenticate canonical epoch
  bytes against the finalized on-chain anchor and independently rederive the
  economic state and winner.
- `config.py` validates the persistent service path, fixed ports, and both
  content-addressed version-zero definitions. Each configured `image_digest`
  must recompute as the `vidaio.competition.logical-build.v1` canonical digest
  of its exact public repository, commit, and tree; an opaque Modal `im-*` id is
  never a registry identity.
- `service.py` owns the SQLite writer connection and exposes the read-only HTTP
  surface. It also defines the internal `BaselinePromotionWatcher` injection
  seam; there is deliberately no request-driven activation path.
- `migrations/0002_schema_v14_baselines.sql` adds `baselines`,
  `baseline_promotion_latches`, and `baseline_events`. Older compatibility
  tables remain isolated and are ignored by the schema-v14 service.

## Ledger invariants

The SQL schema and startup checks jointly enforce:

- exactly the `compression` and `upscaling` tracks;
- exactly one active row per track;
- version zero only from the `genesis` source and later versions strictly
  monotonic per track;
- immutable artifact, git, image, provenance, anchor, competition, winner, and
  score evidence after insertion;
- fresh CROWN reruns must preserve the winner's stable logical image identity;
  their exact provider object id belongs in runner ownership evidence, not the
  promoted baseline row;
- terminal historical rows cannot be reactivated;
- at most one unresolved verified-CROWN latch per track;
- idempotence by `(snapshot_digest, competition_id, track)`;
- append-only event history; and
- one `BEGIN IMMEDIATE` handover for supersede/insert/latch resolution, so a
  crash cannot leave two active rows or silently remove the active floor.

Every timestamp accepted by the persistence layer is explicitly timezone-aware.
Every artifact read is streamed, bounded by the committed byte count, hashed, and
compared with its configured SHA-256 identity.

## Version-zero provisioning

Version zero is not synthesized by the service. Before first startup, provision
one non-empty `submission_archive` and one non-empty provenance `manifest` for
each track in the configured audit store. Record the following exact values in
`registry.genesis_baselines.<track>`:

```yaml
registry:
  db_path: /var/lib/vidaio/registry-state/registry.db
  http_host: 0.0.0.0
  http_port: 8720
  metrics_port: 9123
  automatic_promotion_enabled: false
  allow_disabled_automatic_promotion_for_testnet: true
  genesis_baselines:
    compression:
      artifact_digest: <64 lowercase hex>
      artifact_bytes: <positive integer>
      image_digest: <64 lowercase hex>
      provenance_digest: <64 lowercase hex>
      provenance_bytes: <positive integer>
      repo_url: https://example.org/org/compression-v0.git
      commit_sha: <40 lowercase hex>
      tree_sha: <40 lowercase hex>
    upscaling:
      artifact_digest: <64 lowercase hex>
      artifact_bytes: <positive integer>
      image_digest: <64 lowercase hex>
      provenance_digest: <64 lowercase hex>
      provenance_bytes: <positive integer>
      repo_url: https://example.org/org/upscaling-v0.git
      commit_sha: <40 lowercase hex>
      tree_sha: <40 lowercase hex>
```

Repository URLs must be credential-free HTTPS URLs. Startup verifies all four
objects before inserting either row, publishes the two sealed executable
archives under their released content addresses, and inserts both version-zero
rows in one transaction. An identical restart is a no-op. Partial existing state,
changed identities, absent bytes, digest/size mismatches, or release failure stop
startup.

The registry role therefore needs its own writable local state mount, audit-store
writer credentials, and the sealed-artifact key. Readers such as the orchestrator
and dashboard mount the SQLite directory read-only. SQLite state must stay on a
coherent local filesystem; it is not shared through object storage or NFS.

## Read-only service API

Run the production role with:

```sh
python scripts/service_entrypoint.py baseline-registry
```

The only application routes are:

- `GET /healthz` — schema version, active version per track, archive-verification
  status, explicit automatic-promotion posture, and `503` on any live ledger
  invariant violation.
- `GET /v1/baselines` — active baseline records and unresolved internal promotion
  latches plus the same promotion posture. The response kind is
  `vidaio.baseline-registry.v1` and the schema version is `14`.

OpenAPI/docs are disabled. `POST`, `PUT`, `PATCH`, and `DELETE` are absent. The
separate health/Prometheus server uses port `9123`; the read API uses port `8720`.
The testnet Compose service is the sole writable owner of `registry-state`, while
its consumers receive read-only mounts and wait for its health check.

## Verified CROWN activation

The promotion pipeline does not accept a caller-selected winner or a boolean
claim that an anchor was verified. Given one snapshot digest, its trusted source
must prove and return a canonical current-schema CROWN epoch containing:

- the finalized chain anchor and canonical snapshot digest;
- competition id, track, cycle, half-open reward window, machine winner, score,
  baseline score, and recomputed relative margin;
- the exact currently active baseline version/digest used for comparison;
- the winner's sealed source archive, build-image identity, repository, commit,
  and tree; and
- a complete, ordered score-packet/audit-bundle matrix.

The pipeline repeats schema, state, anchor, track, margin, and active-baseline
binding checks; verifies every referenced object; durably latches the result;
rebuilds and reruns the exact archive in a fresh trusted environment; checks score
parity and stable logical build-identity equality; publishes the source and
provenance; then atomically activates the next
version. A build or verification failure leaves the prior baseline active and the
latch pending, which blocks the next competition for that track until the same
evidence is safely retried. Replaying an already applied evidence key returns the
same row.

There is no public or unverified/manual activation shortcut. An explicit internal
rollback appends a new version that references a known historical executable and
requires a non-empty reason; it never edits or reactivates an old row.

## Current integration boundary

The persistent service, version-zero bootstrap, API, migrations, Compose role,
and verified promotion pipeline are implemented. Automatic post-CROWN activation
still requires a concrete chain-backed `VerifiedCrownEpochSource`, fresh
build/rerun adapter, and watcher scheduler to be injected into
`BaselineRegistryService`. Until those adapters are wired, the service correctly
serves and verifies version zero but does not advance executable state. That gap
must not be filled with a direct database or HTTP write.

This state is never implicit. `automatic_promotion_enabled=true` without an
injected watcher fails startup, as does injecting a watcher while the flag is
false. Production preflight rejects a disabled flag unless the deployment sets
the narrowly named `allow_disabled_automatic_promotion_for_testnet=true` exception.
The tracked testnet stack uses that temporary exception and reports status
`disabled_testnet_exception` in both APIs and Prometheus; mainnet must enable and
wire the verified adapter rather than reuse the exception.

Gateway routing is intentionally outside this service. Consumers may read the
active ledger or its API, but this package neither proxies inference traffic nor
publishes a mutation endpoint.

## Tests

```sh
python -m pytest tests/registry
python -m pytest tests/integration/test_testnet_compose.py \
  tests/integration/test_service_entrypoint_roles.py \
  tests/integration/test_production_guard.py
```

The focused suite covers complete/atomic/idempotent version-zero seeding,
content verification and publication, SQL invariants, read-only routes, visible
runtime corruption, verified CROWN proof refusal, retryable latches, fresh
build/rerun parity, promotion idempotence, rollback append semantics, role-secret
boundaries, and the writable-owner/read-only-consumer deployment topology.
