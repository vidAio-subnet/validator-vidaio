# Fresh Modal GPU miners

Competition contender execution has a separate create-only Sandbox runbook:
[`COMPETITION.md`](COMPETITION.md). It uses Modal GPU only for untrusted
contender execution; collection, scoring, and auditing remain CPU-only.

This deployment path is exercised in production. It creates GPU media-transform
workers for both inference tracks while
keeping the public Bittensor miner protocol on a CPU ingress. The worker emits a
normal H.264 MP4; the canonical scoring worker and every auditor remain CPU-only.
The acceptance test is therefore:

```text
GPU worker produces output bytes
  -> CPU scoring worker creates the score packet
  -> same output bytes + audit bundle reach an independent CPU auditor
  -> auditor recomputes the same packet/score and returns PASS
```

There is no GPU-only score or metric in this design. Compression uses a small
CUDA detail transform followed by an H.264 rate/quality profile; upscaling uses
CUDA bicubic/bilinear interpolation plus a profile-specific detail transform.
The `quality`, `balanced`, and `compact` profiles deliberately produce different
bytes and rate/quality trade-offs so score ranking and digest dedup are visible.

The implementation follows Modal's current APIs for
[GPU Functions](https://modal.com/docs/guide/gpu),
[ASGI web Functions](https://modal.com/docs/guide/webhooks),
[Secrets](https://modal.com/docs/guide/secrets),
[Environments](https://modal.com/docs/guide/environments), and
[app logs](https://modal.com/docs/cli/latest/app). Modal's own GPU utilization,
memory, power and temperature metrics remain available in its dashboard; the app
also exposes authenticated `/healthz` and Prometheus `/metrics` routes and emits
one structured JSON event per completed/failed task.

## Resource-isolation rule

Never reuse, update, stop, inspect or deploy over any existing Modal app, secret,
environment, Volume, Dict, Queue, Function or container. This app performs no
persistent resource lookups besides its explicitly named auth secret. Every
resource created for this test starts with `vidaio-next-`. Do not use `--force`
when creating a secret and do not use a rolling deploy against a prior name: a
name collision is a HOLD, and the operator chooses a new run ID.

The commands below intentionally do not call `modal app list`, `modal secret
list`, or any inventory command. They address only names minted for this run.

## Static checks (safe now; no cloud call)

From the repository root:

```sh
python -m pytest -q \
  tests/miner/test_remote_gpu_backend.py \
  tests/miner/test_gpu_worker.py \
  tests/miner/test_modal_deployment_contract.py
python -m py_compile \
  vidaio/miner/remote_gpu.py \
  vidaio/miner/gpu_worker.py \
  deploy/modal/vidaio_next_gpu_miner.py
```

The miner's default remains `backend_mode: ffmpeg`; nothing silently activates
Modal or spends GPU. Fake-network tests cover auth, response binding, deadlines,
caps, atomic output writes and both track bindings without cloud access.

## Provision only after the deployment test is authorized

Authenticate the Modal CLI to your own workspace (`modal setup`). This
procedure creates a fresh isolated Environment.
Generate a unique run ID locally and record it in the soak log:

```sh
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)"
MODAL_ENV="vidaio-next-testnet-${RUN_ID}"
uvx --from modal==1.5.4 modal environment create "${MODAL_ENV}"
```

If environment creation reports that the name exists, stop and mint a new run ID.
Do not inspect or reuse the collision.

Create one new app and one new secret per solution profile. Put secret material in
a mode-0600 file outside the repository so it is not exposed in shell arguments
or Git. Repeat the block with `quality`, `balanced`, and `compact` (and, for the premium serving app, `premium` — its compression runs the baked ab-av1/libvmaf toolchain on CPU and attests `abav1:<encoder>`):

```sh
VARIANT=quality
APP_NAME="vidaio-next-testnet-${VARIANT}-${RUN_ID}"
SECRET_NAME="vidaio-next-testnet-${VARIANT}-auth-${RUN_ID}"
SECRET_FILE="$(mktemp)"
chmod 600 "${SECRET_FILE}"
trap 'rm -f -- "${SECRET_FILE}"' EXIT HUP INT TERM
VIDAIO_NEXT_GPU_AUTH_TOKEN="$(openssl rand -hex 32)"
printf 'VIDAIO_NEXT_GPU_AUTH_TOKEN=%s\n' \
  "${VIDAIO_NEXT_GPU_AUTH_TOKEN}" >"${SECRET_FILE}"

# Modal's dotenv-file parser is an optional CLI dependency. Pin it explicitly:
# without it, Modal 1.5.4 can print an import error yet return success without
# creating the secret.
uvx --from modal==1.5.4 --with python-dotenv==1.2.3 \
  modal secret create --env "${MODAL_ENV}" \
    --from-dotenv "${SECRET_FILE}" "${SECRET_NAME}"

VIDAIO_NEXT_MODAL_SECRET_NAME="${SECRET_NAME}" \
VIDAIO_NEXT_DEPLOYMENT_LABEL="${APP_NAME}" \
VIDAIO_NEXT_SOLUTION_VARIANT="${VARIANT}" \
VIDAIO_NEXT_FRESH_CREATION_CONFIRMATION=CREATE_FRESH_VIDAIO_NEXT_MODAL_RESOURCES \
uvx --from modal==1.5.4 modal deploy --env "${MODAL_ENV}" \
  --name "${APP_NAME}" --stream-logs \
  deploy/modal/vidaio_next_gpu_miner.py

rm -f -- "${SECRET_FILE}"
trap - EXIT HUP INT TERM
```

The deploy module has no reusable app or secret identity and refuses to construct
deployment-side Modal objects without the exact fresh-creation confirmation above.
Modal also imports the source inside each remote Function container; that second import
is not a deployment operation and does not receive the local acknowledgement. The module
uses Modal's documented `modal.is_local()` boundary there, constructs only inert remote
placeholders, performs no named lookup, and receives the already-bound deployment label
and solution variant through the Function environment. The successfully created,
never-reused Environment is the create-only namespace barrier: it contains no pre-existing
app that `modal deploy` could update. Record
the HTTPS web URL printed by the deploy command beside the matching token, then
securely delete the temporary file. Retain the token in the deployment secret
manager used by the corresponding CPU ingress; do not paste it into YAML, a
ticket, logs or this repository.

The app uses an L4 by default, one request per container, zero warm containers,
at most four containers, and a 120-second scale-down window. Override these only
at deploy time with `VIDAIO_NEXT_MODAL_GPU`,
`VIDAIO_NEXT_MODAL_MAX_CONTAINERS`, and
`VIDAIO_NEXT_MODAL_SCALEDOWN_WINDOW_SECONDS`. Modal currently supports L4 and
other named GPU types through the `gpu=` Function argument. A worker that cannot
see CUDA exits; it never falls back to CPU while claiming GPU execution.

## Wire fresh CPU miner ingresses

One hotkey belongs to one inference pool. For each solution profile, run two new
CPU ingress hosts — each an ordinary miner-service deployment (the `miner` role
of `scripts/service_entrypoint.py` with its own wallet, work directory and
public HTTP(S) edge): a compression hotkey and an upscaling hotkey. The positive
soak therefore has six
fresh hosts and six distinct public IPv4 addresses. Both
may delegate to the same profile-specific fresh Modal app, but each keeps its own
wallet, replay cache, public endpoint and work directory. Do not load-balance one
hotkey across replicas.

Add these environment values to each new ingress in addition to the existing
artifact-v2 wallet/chain settings. The secret value must match that profile's
fresh Modal secret:

```text
VIDAIO__MINER__BACKEND_MODE=remote_gpu
VIDAIO__MINER__REMOTE_GPU_URL=<fresh Modal HTTPS URL>
VIDAIO__MINER__REMOTE_GPU_AUTH_TOKEN=<fresh profile bearer>
VIDAIO__MINER__REMOTE_GPU_SOLUTION_VARIANT=<quality|balanced|compact>
VIDAIO__MINER__REMOTE_GPU_CONNECT_TIMEOUT_SECONDS=10
VIDAIO__MINER__WARRANT_TRACK=<compression|upscaling>
VIDAIO__MINER__ARTIFACT_HOTKEY=<that ingress's registered hotkey>
VIDAIO__MINER__ALLOW_UNSIGNED_ARTIFACT_V1=false
VIDAIO__MINER__ENABLE_LEGACY_PATH_ROUTES=false
```

The local miner's passive `/health` checks only that the remote configuration is
present; it intentionally does not cold-start and bill Modal on every scrape.
The explicit preflight below proves the remote GPU live. Existing local miner
task counters and latency histograms include the remote call.

## Per-track GPU -> CPU preflight

First prove the deployed worker live through its authenticated health route.
The response must report `gpu_available: true` and a real CUDA device. Keep a
short socket-connect budget but a generous overall request deadline, because a
newly deployed or scaled-to-zero GPU Function may need a real cold start:

```sh
export VIDAIO_NEXT_GPU_AUTH_TOKEN='<load from the fresh secret manager>'
curl --fail --connect-timeout 10 --max-time 300 \
  -H "Authorization: Bearer ${VIDAIO_NEXT_GPU_AUTH_TOKEN}" \
  '<fresh Modal URL>/healthz'
```

Then run one real challenge end-to-end per track through your own stack: point
a CPU ingress at the fresh Modal URL (the `remote_gpu` settings above), submit
a real challenge's pristine reference and miner input with its real scoring
parameters, and have your local CPU scoring worker score the returned bytes.
Run it from a path shared with the CPU scoring worker so all three immutable
files (reference, miner input, GPU output) are visible to `/score`, and write
the GPU output to a never-existing filename that nothing overwrites.

A PASS requires the worker's live CUDA health assertion, response bindings,
output digest and size cap, a complete CPU backend/canonicalization attestation,
every score-packet request binding and packet digest to match, shipping gates to
pass, an absolute score at or above the launch payout floor, and byte-identical
score packet JSON/digest from two CPU scoring runs. Never lower a challenge gate
for preflight; copy the exact real challenge parameters (compression VMAF is 90 at
launch). Passing this is not the final audit test: retain the resulting normal
inference evidence, finalize its epoch, and require an independent CPU auditor PASS
before calling the testnet soak healthy.

A health-probe timeout is still a HOLD; do not turn on a CPU fallback or weaken
the CUDA assertion to make the probe pass.

## Observe the soak

For each freshly minted app name only:

```sh
uvx --from modal==1.5.4 modal app logs --env "${MODAL_ENV}" \
  --timestamps --show-container-id --follow "${APP_NAME}"
```

Scrape authenticated `<fresh URL>/metrics` from the deployment monitoring plane
and alert on failed transforms, latency, input/output bytes and CUDA memory.
Use the Modal dashboard for its platform GPU utilization/temperature/power and
container/input metrics. A completed request log contains track, profile,
geometry, byte counts, timing and 12-character digest prefixes; it never contains
the bearer, input bytes or a full wallet identity.

Testnet acceptance for each profile/track is:

- at least one scored, gate-passing real challenge;
- distinct output digests between profiles on the same input;
- observable within-track score ordering (ties are allowed only if the media
  genuinely collapses the profiles to the same score, and must be investigated);
- no GPU worker 5xx/timeout/oom and no sustained queue saturation;
- the finalized audit bundle contains the exact GPU output digest; and
- an independent ordinary CPU auditor recomputes it and returns PASS.

## Delete only this run when it is finished

First stop the six independent CPU miner-edge projects that hold these fresh URLs/tokens and retain
the finalized epoch/audit evidence required for the soak record. Then delete the
one exact Environment name minted at the start of this run. In Modal 1.5.4,
Environment deletion also deletes the apps and secrets inside that Environment:

```sh
test -n "${MODAL_ENV:-}" || { echo 'MODAL_ENV is unset' >&2; exit 64; }
case "${MODAL_ENV}" in
  vidaio-next-testnet-*) ;;
  *) echo 'refusing non-vidaio-next testnet Environment' >&2; exit 64 ;;
esac
uvx --from modal==1.5.4 modal environment delete --yes "${MODAL_ENV}"
```

Run that command once, from the recorded run receipt; do not list environments or
substitute a guessed name. It targets only the fresh namespace that contained this
test's three profile apps/secrets. Delete the fresh CPU ingress hosts at your
hosting provider separately, by their recorded provider resource IDs, only
after their logs and state have been retained.

## Inputs still needed from the operator

Beyond Modal client authentication in your own workspace, provisioning the
real testnet fleet still needs:

- approval to create the fresh Modal Environment, three fresh apps and three
  fresh auth secrets named by this procedure;
- six new registered testnet miner hotkeys (three profiles times two tracks),
  with funded registration/transaction wallets;
- six fresh CPU ingress deployments whose public HTTP(S) edges exactly match the
  fleet-wide `validator.miner_url_scheme` (prefer HTTPS with IP-valid certificates),
  plus a secure place to inject each Modal bearer and each hotkey wallet;
- the fresh S3 audit bucket/policy and archive RPC configuration used by the
  central deployment;
- real pristine/miner-input challenge pairs and their committed params for both
  preflight commands; and
- the monitoring destination/alert receiver for service health, Prometheus and
  Modal GPU metrics.

No existing Modal instance/app is required or permitted.
