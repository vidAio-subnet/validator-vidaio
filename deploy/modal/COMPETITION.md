# Fresh Modal GPU competition sandboxes

This path is exercised in production. It runs contender code on a GPU and
returns ordinary output bytes to
the existing competition orchestrator. The trusted scoring worker and every
auditor stay CPU-only:

```text
pinned contender checkout
  -> fresh Modal image
  -> fresh network-blocked GPU Sandbox for one sealed batch
  -> immutable short-lived output snapshot
  -> fresh network-blocked CPU collector
  -> existing CPU scoring worker
  -> epoch log + audit bundle + S3 audit store
  -> independent CPU auditor recomputation
```

The implementation follows Modal's current
[Sandbox](https://modal.com/docs/guide/sandboxes),
[Sandbox filesystem](https://modal.com/docs/guide/sandbox-files),
[custom image](https://modal.com/docs/guide/existing-images), and
[GPU](https://modal.com/docs/guide/gpu) APIs. The tested SDK pin is `modal==1.5.4`.

## Non-negotiable ownership boundary

Never list, inspect, stop, update, resolve, or reuse any unrelated/pre-existing Modal
App, Sandbox, Volume, Secret, Function, image id, or instance. The runner contains no
inventory/name lookup. The sole restart exception is `Image.from_id` for an immutable
Image created by this exact competition and durably bound to its pinned source/digest in
append-only DB evidence. It never restores an App, Sandbox, or instance. Every execution
Sandbox is created in-process and terminated. All names start with `vidaio-next-`.

Create a new Environment for each smoke test and another new Environment for the
real testnet soak. An Environment used by a completed smoke test is not reused for
the soak. A name collision is a HOLD: mint a new run id without inventorying the
collision.

No Modal Secret, Volume, NFS, bucket mount, inbound port, OIDC token, or
environment variable is attached to contender sandboxes. Modal credentials stay
on the trusted orchestrator control host and authorize control-plane calls only.
The shipped Bittensor configuration explicitly selects Modal, while its identity
and confirmation fields stay empty so it fails before the first cloud call. The
report/local overlays explicitly select Docker. Cloud creation requires the exact
code-level confirmation
`CREATE_FRESH_VIDAIO_NEXT_MODAL_RESOURCES` and the CLI additionally requires
`--authorize-create-fresh-resources`.

## Static checks (safe now; zero cloud calls)

From the repository root:

```sh
python -m pytest -q \
  tests/orchestrator/test_modal_runner.py \
  tests/orchestrator/test_modal_restart_recovery.py \
  tests/orchestrator/test_git_repo_provider.py \
  tests/orchestrator/test_gpu_manifest_policy.py \
  tests/orchestrator/test_competition_contender_examples.py \
  tests/integration/test_modal_orchestrator_composition.py \
  tests/integration/test_release_modal_dependencies.py
ruff check \
  vidaio/competition/runners/repo.py \
  vidaio/competition/runners/modal_runner.py \
  vidaio/competition/orchestrator/service.py \
  tests/orchestrator/test_modal_runner.py \
  tests/orchestrator/test_modal_restart_recovery.py \
  tests/orchestrator/test_git_repo_provider.py \
  tests/orchestrator/test_gpu_manifest_policy.py \
  tests/orchestrator/test_competition_contender_examples.py \
  tests/integration/test_modal_orchestrator_composition.py
python -m py_compile \
  vidaio/competition/runners/repo.py \
  vidaio/competition/runners/modal_runner.py \
  examples/competition_contenders/materialize.py
python scripts/verify_release_dependencies.py
```

The tests inject a fake runtime. They assert fresh names/handles, GPU-only
contender execution, CPU-only collection, no persistent lookup APIs, immutable
per-batch input contents, exact owned-Image restart restoration, probe-before-run,
forced rollover, byte/log caps,
symlink refusal, build timeout poisoning, and termination/detach.

They also model a real process boundary with runner A and runner B: the second
runner has no executable handles from the first, may rehydrate only exact
competition-owned immutable Image ids, reprobes them, resets and reruns the full
effective matrix, and halts before GPU work on binding/digest drift or reprobe failure.

## Inputs required before a live smoke test

- Operator approval to create new Modal resources.
- Modal client authentication on the trusted control host. No credential is
  passed to a contender.
- One new run id per track, preferably UTC timestamp plus random suffix.
- A local Git checkout of a real contender solution with its exact 40-character
  commit SHA and tree SHA. It must contain `Dockerfile` and `/app/run.sh` and must
  be compatible with a Modal L4 Sandbox.
- For production enrollment: a credential-free HTTPS repository URL on an exact
  allowlisted host and a server-side read-only token. The token is supplied only
  through a child-private Git askpass environment, never through the URL/argv.
- At least two independently committed solutions per track. Runnable CUDA
  `quality`, `balanced`, and `compact` baselines are materialized from
  [`examples/competition_contenders`](../../examples/competition_contenders/README.md).
- One real sealed evaluation input for that track. For upscaling, record its committed
  factor (2 or 4) and pass it explicitly to the smoke command. The command retains one
  new output path and refuses to overwrite it.
- For the pre-testnet CPU-audit gate: run the control harness, canonical CPU
  scorer, and
  independent recomputer from the same digest-pinned Linux/amd64 release image;
  a native host process is rejected before Modal access. Supply the exact YAML
  that produced its config-and-runtime scorer identity, a new evidence directory,
  and (for upscaling) the sealed pristine reference. The opt-in gate verifies the
  worker's full runtime-commitment preimage, persists a normal local bundle,
  resolves it through the public-store role, and requires strict independent CPU
  recomputation plus a positive gate-passing score. The PieAPP tolerance remains
  `1e-5`; runtime drift is never hidden by widening or quantizing it.
- Testnet still adds the full contender/item matrix, fresh S3 audit bucket, finalized
  epoch, chain anchor, ranking/dedup/crown, and emissions. The single-item local
  report does not claim those boundaries.

## Explicit live smoke tests (do not run until authorized)

Live smoke tests are driven by a create-only control harness run from the
development tree; the harness itself is not part of this release, so what
follows documents the procedure it automates, one run per track.

Create-only provisioning uses no inventory command. Generate three independent
names locally (Environment, App, and run label) per track, for example:

```sh
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)"
MODAL_ENV="vidaio-next-comp-compress-${RUN_ID}"
MODAL_APP="vidaio-next-comp-compress-app-${RUN_ID}"
RUN_LABEL="vidaio-next-comp-compress-run-${RUN_ID}"

uvx --from modal==1.5.4 modal environment create "${MODAL_ENV}"
```

If `environment create` reports a collision, stop and mint a new run id.
Upscaling uses a second new Environment/App/run and a track-specific contender,
plus its committed factor (2 or 4) and exact target geometry.

For the CPU-audit form, arrange all inputs and new output/evidence parents
below one absolute `PREFLIGHT_ROOT`, and run the control process from the exact
qualified `linux/amd64` release-image OCI digest with host networking so it
reaches the separately running scorer at loopback; both containers must use
that same digest and mount `PREFLIGHT_ROOT` at the same `/preflight` path
because the scoring protocol binds shared snapshot paths. The two Modal token
variables are forwarded by name so neither value lands in argv. Per track, the
harness receives the fresh Environment/App/run-label names, the explicit
fresh-resource authorization, the pinned contender checkout with its exact
40-character commit and tree SHAs and credential-free allowlisted repository
URL, the sealed evaluation input (and, for upscaling, the sealed pristine
reference plus factor/geometry), a new output path, a new CPU-audit-evidence
directory, the scoring-worker URL and the exact scoring YAML, the enrolling
miner hotkey, and the GPU type (`L4`).

The harness verifies the git pin and track/factor contract before Modal creation,
refuses an existing output path, runs build/probe/batch, verifies the retained
digest/size, and closes the runner in `finally`. With the CPU-audit arguments it
also checks the live CPU scorer identity and backend stamps, stores the canonical
packet and bundle, proves an upscaling reference is hidden until release, reloads
through `StoredBundleSource(public)`, and invokes `RealScoreRecomputer` in strict
mode. `Ctrl-C` also takes the close path. A hard-killed control process cannot run
cleanup; every Sandbox still has a bounded idle timeout and a maximum lifetime
below 23h30m.

## Real testnet service wiring

The production entrypoint now composes `GitRepoProvider` plus
`ModalSandboxRunner`; report/local composes `LocalRepoProvider` plus Docker.
There is no fallback between them. Install the exact locked release image, mint
three fresh names, create only the new Environment, and inject these secrets and
settings through the orchestrator's least-privilege environment:

```sh
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)"
export MODAL_ENV="vidaio-next-env-${RUN_ID}"
export MODAL_APP="vidaio-next-app-${RUN_ID}"
export MODAL_RUN="vidaio-next-run-${RUN_ID}"

# MODAL_TOKEN_ID / MODAL_TOKEN_SECRET are injected by the deployment secret store.
# This is a create command, not an inventory lookup.
uvx --from modal==1.5.4 modal environment create "${MODAL_ENV}"

export VIDAIO__ORCHESTRATOR__SANDBOX_BACKEND=modal
export VIDAIO__ORCHESTRATOR__MODAL_ENVIRONMENT_NAME="${MODAL_ENV}"
export VIDAIO__ORCHESTRATOR__MODAL_APP_NAME="${MODAL_APP}"
export VIDAIO__ORCHESTRATOR__MODAL_RUN_LABEL="${MODAL_RUN}"
export VIDAIO__ORCHESTRATOR__MODAL_CREATION_CONFIRMATION=CREATE_FRESH_VIDAIO_NEXT_MODAL_RESOURCES
export VIDAIO__ORCHESTRATOR__MODAL_GPU=L4
export VIDAIO__ORCHESTRATOR__GIT_READ_ONLY_TOKEN='<secret-manager-reference>'
export VIDAIO__ORCHESTRATOR__GIT_USERNAME=x-access-token
export VIDAIO__ORCHESTRATOR__GIT_ALLOWED_HOSTS='["github.com"]'
export VIDAIO__ORCHESTRATOR__CONTROL_TOKEN='<secret-manager-reference>'
export VIDAIO__ORCHESTRATOR__SCORING_WORKER_URL=http://scoring-worker:8201
export VIDAIO__ORCHESTRATOR__WORK_DIR=/var/lib/vidaio/orchestrator
```

The competition orchestrator is owner-operated; it is not one of the roles the
public service entrypoint serves. With those values injected, the owner
deployment first passes its static production preflight inside the exact
release image, then starts the orchestrator role against the production
config.

The [Modal configuration contract](https://modal.com/docs/sdk/py/latest/config)
uses `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`; never place them in YAML, Git, or
contender configuration. Do not run a separate `modal deploy`: the orchestrator
opens one freshly named ephemeral App and creates only owned Sandboxes beneath
it. On process restart, mint a new Environment/App/run identity. The orchestrator may
rehydrate only exact immutable Image ids from this competition's append-only ownership
bindings; it must never resolve by name, inventory resources, or restore a Sandbox.

`image_digest` is not a Modal object id hash. It is the stable
`vidaio.competition.logical-build.v1` digest over the exact pinned repository,
commit, and tree coordinates, so qualification must supply the registry's
credential-free public URL even though it builds an already verified local checkout.
The later public clone then recomputes the same identity. Every newly built `im-*` remains separately
bound to that logical digest, source, fresh runtime session, and resource label in
the typed append-only `modal_image_bindings` ledger and matching event. Restoration
uses only the exact newest role-matching bound id and verifies Modal returns that
same id; no provider listing/name lookup is permitted.

### Controlled orchestrator restart protocol

Each `ModalSandboxRunner` construction mints an opaque random session id in
addition to the operator-supplied fresh run label. Before the first BUILDING
action, the orchestrator appends `modal_runtime_bound`. A replacement process:

1. creates a wholly new Environment/App/run namespace (never inventory or reuse);
2. reads only its own database evidence and pinned repo/commit/tree values;
3. rehydrates each persisted `BUILT` contender only from its exact logical-source
   digest plus provider-id ownership binding, then verifies the id readback and
   reprobes it through the new runtime;
   and
4. when the competition is EVALUATING, atomically appends
   `modal_evaluation_reset`, returns every batch to PENDING, and appends the new
   runtime binding. Batch-output and requeue readers accept only events above the
   latest reset fence, so the full matrix is produced by runner B.

Earlier batch events and bytes are retained for incident review; they are not
selected for scoring. An image-restore/ownership error, digest mismatch, unavailable probe, or
failed probe records an orchestrator halt before runner B executes any batch.
Never clear such a halt merely to “resume”: first prove a newly created runtime
can satisfy the exact image/probe contract. This protocol also detects accidental
reuse of the same configured run label because the per-process session id still
changes.

Every live manifest must include the exact configured value (`L4` above) in
`allowed_gpus`. Creation rejects a mismatch before persistence; build and
evaluation recheck the persisted manifest and HALT before another GPU action if
the runtime/config drifts. Docker/report fakes deliberately bypass this live
provider-attestation rule.

## Isolation, rollover and limits

- `block_network=True`, `secrets=[]`, `env={}`,
  `include_oidc_identity_token=False`, no Volumes/NFS/bucket mounts and no ports
  are set on every Sandbox creation.
- The input mount is a force-built anonymous image containing only that batch's
  digest-named media files and, for upscaling, one hidden committed-task sidecar per
  digest. Modal exposes a sandbox-local writable overlay for the mount, but its
  content-bound Image base is immutable: the live probe destroys its writer and
  remounts the same Image in a fresh CPU Sandbox, which must prove the write did not
  persist. Each batch receives a newly built Image and a fresh Sandbox. It supports
  mixed 2x/4x batches with exact target geometry, never contains `index.json`, and never exposes a pristine
  reference digest or bytes.
- A fresh GPU Sandbox is used for every batch. This intentionally gives up warm
  reuse: a warm untrusted container can leave a background process/state that
  observes the next batch. Per-batch termination is the rollover policy.
- Stdout/stderr and the remote output tree are watched while `run.sh` executes.
  The watchdog continues while output is frozen. A cap breach terminates the
  contender and becomes a typed contender fault.
- The frozen output snapshot has a one-hour TTL. A fresh network-blocked,
  secret-free CPU collector mounts its content-bound Image; symlinks/directories
  are refused and ordinary bytes are copied into the orchestrator's
  content-addressed pool.
- Modal controls GPU execution only. The collector performs file transport, not
  media scoring. All score packets still come from the canonical CPU scorer.

## Testnet acceptance and observability

The pre-testnet live preflight is green only when both track commands use the
CPU-audit mode and return `status=PASS` with `cpu_recompute=PASS`, a complete CPU
backend/canonicalization attestation, and a gate-passing score at or above the
launch payout floor.
The actual testnet soak is green only after all of the following are observed:

- at least two distinct pinned contender solutions per track build and pass their
  isolation probes;
- every batch log contains a unique fresh Sandbox id/name and a termination event;
- compression and upscaling each produce non-empty outputs and observable score
  ordering/digest dedup through the ordinary orchestrator metrics;
- Modal's fresh run dashboard shows L4 allocation only on `probe`/`contender`
  roles; `collector` is CPU-only;
- the orchestrator's existing build/batch/fault/halt metrics remain healthy;
- every measured score packet references the exact collected output digest in its
  persisted audit bundle and finalized epoch log; and
- an independent CPU-only auditor downloads those same bytes from the fresh S3
  audit store, recomputes every committed score in uncapped all-items mode, and
  returns PASS.

## Delete only completed fresh runs

Normal preflight and orchestrator shutdown close the exact ephemeral App context
and terminate/detach every owned Sandbox. Keep the Environment while an unfinished
competition may still need its append-only image ownership bindings for controlled
restart recovery. After the smoke/competition is complete, CPU audit evidence is
retained, and the control process is closed, delete only the exact Environment from
that run's receipt:

```sh
test -n "${MODAL_ENV:-}" || { echo 'MODAL_ENV is unset' >&2; exit 64; }
case "${MODAL_ENV}" in
  vidaio-next-comp-*|vidaio-next-env-*) ;;
  *) echo 'refusing non-vidaio-next competition Environment' >&2; exit 64 ;;
esac
uvx --from modal==1.5.4 modal environment delete --yes "${MODAL_ENV}"
```

Do not list environments, reuse a stopped App, or clean by prefix/glob. A hard-killed
controller can leave a Sandbox until its bounded idle/lifetime timeout; delete the
recorded fresh Environment after the process is gone rather than targeting Sandbox
ids discovered from workspace inventory.

Runner lifecycle events are structured local JSON logs keyed by `run_label`,
`sandbox_id`, role, image digest and batch index. The trusted control process logs
are sufficient to correlate Modal dashboard GPU/container metrics without any
inventory/list call.

## Provider limitations to keep visible during the soak

1. Modal's public Sandbox API accepts the isolation controls but does not expose a
   public endpoint that reads the full applied Sandbox configuration back. The
   report records the exact host-side request plus an advisory-negative probe; it
   cannot claim a Docker-style server config readback. Any evidence that requested
   controls were not applied is an immediate testnet HOLD.
2. Modal Sandbox creation does not expose a custom ephemeral-disk quota. The
   runner enforces business output caps with polling and final immutable-snapshot
   checks; Modal's provider disk ceiling remains the outer bound between polls.
   A quota/disk exhaustion event is a contender fault and testnet investigation,
   never a scoreable output.
3. Modal `Image.build` is synchronous and has no per-build cancel method. On the
   configured deadline the runner closes its entire fresh ephemeral App and
   poisons itself; the server-side build may take a short time to observe that
   cancellation. Start a new run rather than reusing the poisoned one.
4. Anonymous force-built contender/input/collector images and short-lived output
   snapshots are never resolved or reused. Snapshot TTL is one hour; Modal may
   retain other image layers under its provider-managed cache after active GPU
   resources are gone.
5. `GitRepoProvider` creates a new private `vidaio-next-checkout-*` directory for
   every validation, archive, Docker build, and Modal build. It verifies exact
   commit/tree identities, strips `.git`, caps time/bytes/logs, and releases each
   checkout only after its consumer has completed. Shutdown closes any remaining
   owned session; no existing checkout is discovered or reused.

These limitations do not introduce a GPU-only score. The release gate remains
the CPU auditor reproducing each GPU-contender output's score from immutable
audit bytes.
