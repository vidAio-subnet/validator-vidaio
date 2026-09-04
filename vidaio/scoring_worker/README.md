# vidaio.scoring_worker — HTTP `/score` over the pure engine + real ffmpeg/libvmaf

The scoring worker is the only place where the pure scoring module
([`vidaio/scoring`](../scoring/README.md)) meets subprocesses and the wire
protocol ([`vidaio/services/protocol.py`](../services/protocol.py)): one
`POST /score` executes the whole honest pipeline — verify-then-snapshot the
inputs, probe, canonicalize to raw y4m, measure VMAF (two models) / PieAPP, run
the gates, compose the audit-grade `ItemScore` — and returns the exact packet
bytes plus their sha256. Backends are injected Protocols: deterministic fakes for
tests; ffmpeg/ffprobe/libvmaf, CPU PIQ PieAPP, and deterministic CPU/OpenCV
manipulation checks for real runs. A genuinely unavailable dependency still surfaces as
a typed refusal, never a substituted score.

## What it does

- **`service.py`** — [DONE] the FastAPI app (`create_app`: `POST /score`,
  `GET /healthz`), the synchronous pipeline (`_score_sync`, run on a worker
  thread under a `MediaProcessScope`), scorer-identity minting
  (`effective_scorer_version` / `scorer_identity_digest` /
  `check_scorer_version`), backend composition (`real_backends`), the
  expansion-budget reservation (`_reserve_expansion`), metrics
  (`WorkerMetrics`), health checks (`build_health_checks`), and the
  `ScoringWorker` BaseService (embedded uvicorn, `fail_fatal` on API death).
- **`inputs.py`** — [DONE] verify-then-snapshot (`snapshot_request_inputs` /
  `snapshot_input`), the scratch byte budgets (`ByteLimits`, `ScratchBudget`,
  `ScratchLease`), the y4m expansion projection (`projected_canonical_bytes`,
  `projected_frame_count`, `y4m_frame_bytes`, `projected_metric_log_bytes`),
  the typed refusal carrier (`ScoreRejected`), and the startup reclamation
  sweep (`sweep_work_dir`) with its residual accounting
  (`measure_scratch_entries`, `ScratchBudget.charge_residual` /
  `retry_residual_sweep`).
- **`config.py`** — [DONE] `ScoringWorkerConfig` (the `scoring_worker:`
  section), including the nesting validator over the four scratch ceilings.
- **`runtime_identity.py`** — [DONE] the complete payout-runtime attestation:
  verified release manifest/image marker, Linux/amd64 and deterministic thread
  policy, native media tools, Python metric packages, full packet backend stamp,
  and the canonical-runtime fail-closed check used by release rehearsals.

The real subprocess backends themselves (`FfprobeBackend`, `FfmpegVmafBackend`,
`PieAppTorchBackend`, `CpuPerceptualCheckBackend`, `CanonicalizeExecutor`, and
`MediaProcessScope`) live in
[`vidaio/scoring/backends_real.py`](../scoring/backends_real.py); this service
composes and drives them.

## Design & decisions

### Verify-then-snapshot (the TOCTOU story)

A `/score` request names files on the scoring worker's trusted co-located filesystem
that the upstream producer may still be able to write. Remote miners never choose these
paths: the validator first streams, bounds, and digest-verifies their bytes into its
owner-only `miner_artifact_dir`. Hashing a path and then
re-opening that same path for canonicalization is a time-of-check/time-of-use
hole: swap the bytes in between and the packet's `content_digest` names one
file while ffmpeg measured another — and the digest is the only thing binding a
score to a submission. The fix (`inputs.py`), in this exact order per input:

1. `os.open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK)` — ONE descriptor, taken
   once. `O_NOFOLLOW` refuses a symlinked final component (a miner cannot point
   the worker at the held-out reference); `O_NONBLOCK` makes a FIFO open
   instead of hanging forever.
2. `os.fstat` on THAT descriptor: anything not a regular file (fifo, device,
   directory, socket) is a 422, not an unbounded read. The fstat size is also
   the byte-budget reservation.
3. Hash AND copy in a single pass from that same descriptor into the request's
   private directory, mode `0400` (`SNAPSHOT_MODE`), opened
   `O_CREAT | O_EXCL | O_NOFOLLOW`. Both the hashed bytes and the kept bytes
   come from one open file description, so no path swap can separate them.
4. Compare the streamed digest against the request's claim (mismatch → 422,
   copy discarded), then re-hash the private copy as cheap insurance against a
   truncated write (mismatch → 500 `snapshot_corrupt`).

Everything downstream reads ONLY the private copies; the packet stamps the
digest of the copy that was actually measured (`snapshot.output.digest`), never
a re-read of the caller-named path. The copies live inside the per-request
`TemporaryDirectory` (`score-` prefix under `work_dir`) and die with it.

### The scratch budget (per file / per request / worker-wide)

Snapshotting is a COPY and canonicalization is an EXPANSION, so every byte a
request puts on the volume is reserved BEFORE it exists, against one shared
`ScratchBudget` per app:

- **per file** (`max_input_bytes`, 2 GiB — deliberately the reference miner's
  own ingress ceiling): decided from the fstat of the descriptor that will be
  read, so an oversize file never writes a byte (422 `input_too_large`). The
  copy loop carries the same number as a hard stop: a source that GROWS after
  its fstat is cut off at its reservation (422 `input_grew_during_snapshot`).
- **per request, inputs** (`max_request_bytes`, 4 GiB): the three inputs
  together (413 `request_inputs_too_large`). Not 3× the file cap on purpose — a
  genuine item is one lossless reference plus two smaller files.
- **per request, ALL scratch** (`max_request_scratch_bytes`, 16 GiB): snapshots
  PLUS the canonicalized y4m of both sides PLUS the libvmaf logs (413
  `request_scratch_too_large`). 413 rather than 503 on purpose: a request that
  cannot fit inside one request's allowance can NEVER fit, so shedding it would
  shed it forever. Clamped to the worker budget
  (`ByteLimits.request_scratch_ceiling`).
- **worker-wide** (`max_scratch_bytes`, 32 GiB): every live request's held
  bytes summed (503 `scratch_budget_unavailable` + `Retry-After: 5`) — N
  requests that each fit cannot fill the volume between them. The default is
  exactly `max_concurrent × max_request_scratch_bytes`, so a fully loaded
  worker never sheds its own legitimate load.

The four ceilings must nest, widest last
(`max_scratch_bytes ≥ max_request_scratch_bytes ≥ max_request_bytes ≥
max_input_bytes`); `ScoringWorkerConfig` refuses any inversion at load time. A
lease is held for the LIFETIME of the request's scratch directory (released
after the directory is gone, never when copying ends); an aborted copy discards
its partial file and refunds its reservation, and the projection's surplus is
refunded once the real y4m size is known.

### The expansion projection (why input caps alone are not a scratch bound)

Every input cap measures the ENCODED file; scoring measures the DECODED one —
both sides are canonicalized to raw y4m before any metric runs, and raw video
is three to four orders of magnitude larger than its encoding (a 30 MB
ten-minute 4K clip passes every input cap and decodes to ~450 GB, twice). So
the raw size is computed from the PROBE, before ffmpeg starts, and reserved as
ONE claim covering both sides plus the metric logs (a partial reservation is
not a reservation):

```
bytes = Y4M_HEADER_BYTES + (frames + CANONICAL_SLACK_FRAMES) × (6 + frame_planes)
```

- `frames = max(nb_frames, ceil(duration × fps))` (`projected_frame_count`) —
  the max of the container's two independent statements, because `-fps_mode
  cfr` DUPLICATES frames when duration × rate implies more than are stored;
  trusting `nb_frames` alone would under-project exactly the file that expands
  the most.
- `frame_planes = y4m_frame_bytes(w, h, pix_fmt)` from the closed
  `_PIX_FMT_PLANES` table (yuv420p = 1.5 B/px: luma + two quarter-resolution
  chroma planes, chroma rows rounded UP); an unknown format is a 422, never a
  heuristic. The pix_fmt is the CANONICAL one, which is what makes the
  projection knowable in advance.
- Constants: `Y4M_HEADER_BYTES = 128` (measured 58 B real; bound covers the
  widest plausible fields), `Y4M_FRAME_HEADER_BYTES = 6` (`b"FRAME\n"`),
  `CANONICAL_SLACK_FRAMES = 2` (CFR rounding only — a WRONG projection is
  caught by the hard cap, not by padding).
- Logs: `projected_metric_log_bytes = runs × max(VMAF_LOG_FLOOR_BYTES = 64 KiB,
  frames × VMAF_LOG_BYTES_PER_FRAME = 1024)` — measured ~718 B/frame, rounded
  up. The per-run amount (`runs=1`) is also the HARD bound each libvmaf run's
  JSON log is held to (`use_metric_log_limit` + the output-size watchdog; 413
  `metric_log_too_large` past it), so the reservation is enforced, not just
  taken.

A stream whose geometry cannot be bounded (zero dims, no frame count and no
usable duration × rate) is a 422 `unprojectable_stream`. The projection is
verified exact against real ffmpeg output
(`tests/scoring_worker/test_expansion_budget.py`) — and it is still a
prediction about a miner-produced file, so `CanonicalizeExecutor` takes the
reserved cap as a HARD bound and kills the process group the moment the output
passes it (413 `canonicalized_output_too_large`, partial y4m removed).

### Real backends and the NEG secondary model

`real_backends(config)` composes `FfprobeBackend`, two `FfmpegVmafBackend`
instances (primary `version=vmaf_v0.6.1`, secondary `version=vmaf_v0.6.1neg`),
`CanonicalizeExecutor`, `PieAppTorchBackend` (PIQ/PyTorch on configured CPU or CUDA),
and `CpuPerceptualCheckBackend`, stamping probed tool/algorithm versions into every
packet. Production selects CPU. VMAF is deterministic by construction
(`n_subsample=1`, `n_threads=1`, `pool=mean`, pinned models — no randomness to seed).

The secondary model feeds the `vmaf_model_delta` gate, and NEG ("no
enhancement gain") is the point: it clips exactly the sharpening/contrast
"enhancement" tricks that inflate the default model, so a large primary-vs-NEG
delta is precisely the model-gaming signal the gate hunts. The 4k model was
rejected as secondary — calibrated for a different viewing distance, it
legitimately diverges at small resolutions, which would make the delta gate
noisy instead of adversarial-sensitive
([`vidaio/scoring/backends_real.py`](../scoring/backends_real.py) docstring).

Both model runs compare the candidate with the **miner input**, not the pristine
reference. This locked basis avoids rejecting honest transforms whose challenge input
was already degraded. Evidence records `vmaf_model_delta_basis: miner_input` and the
canonical input-plan digest.

### Scorer identity (minting + the 409 contract)

THE SCORER-IDENTITY CONTRACT ([`vidaio/services/README.md`](../services/README.md))
starts here — the worker mints exactly one name:

```
effective_scorer_version = f"{scoring_worker.scorer_version}+{scorer_identity_digest[:12]}"
```

`scorer_identity_digest` is a sha256 over every configured lever that can
change a measured score plus the full commitment from `runtime_identity.py`:
the verified release runtime/image marker, Linux/amd64 + one-thread CPU policy,
ffmpeg/ffprobe/libvmaf and exact PIQ/PyTorch/torchvision/OpenCV/NumPy/Python
versions, including the pinned PieAPP weights. The full runtime digest is also
stamped in measured packets as `backend_versions.runtime`. Ports, paths,
timeouts and concurrency are deliberately NOT in it — they cannot change a
packet, so two identically-scoring deployments must not refuse each other.

`ScoringWorker` also enforces that attestation in its own constructor whenever
`chain.mode: bittensor`; full-stack and service-scoped preflights are additional early
diagnostics, not the sole safety boundary. `RealScoreRecomputer` is strict by default as
well. Its explicitly named noncanonical opt-out exists only for the Docker smoke before
the release marker is created and for injected-backend tests, never for a production or
third-party auditor.

The identity and full public runtime-commitment preimage are published on
`GET /healthz` (served even when degraded — a 503 body still names the scorer)
and the identity is stamped into every packet. `ScoreRequest.scorer_version` is
a caller ASSERTION, never an
instruction: absent/empty means "whichever scorer you are", an equal value
means agreement, anything else is refused `409 scorer_version_mismatch`
BEFORE any work (`check_scorer_version` runs before the concurrency slot is
even requested). The worker never stamps a caller-supplied version.

### Bounded work: slots, deadlines, process-group cancellation

- `max_concurrent` is an `asyncio.Semaphore` and a HARD bound on real load:
  the slot is released by the executor future's done-callback — when the
  scoring thread has genuinely finished or died — not when the coroutine stops
  awaiting it.
- A request that cannot win a slot within `queue_wait_timeout_seconds` is SHED:
  503 `queue_saturated` + `Retry-After`. 503 means "we never started, come
  back"; 504 stays reserved for "we started and gave up".
- `request_timeout` starts when the request WINS a slot and bounds the real
  work: on timeout (or client disconnect) the request's `MediaProcessScope` is
  cancelled, which SIGKILLs every registered ffmpeg PROCESS GROUP (children
  are started with `start_new_session`) and makes the next subprocess refuse to
  start, so the worker thread unwinds with `MediaWorkCancelled` instead of
  burning CPU behind a caller that already got its 504. The cancellation
  callback also reaches the snapshot copy loop (`SnapshotCancelled`) — copying
  a multi-GB submission is real work the deadline must bound too.
- Everything one request writes lives under ONE directory — snapshots,
  y4m, and the libvmaf temp dirs (`use_media_scratch` installs the request's
  scratch as a thread-local for the metric backends). One directory is what
  makes the accounting checkable, the cleanup total, and the startup sweep
  sufficient: `sweep_work_dir` reclaims every prefix this worker can create
  (`score-`, `vmaf-`, `vmafver-`, `.healthz-probe-`) and touches nothing else.
- What the sweep CANNOT delete (permissions, a busy mount) is not forgotten:
  `measure_scratch_entries` sizes the survivors and `create_app` pre-charges
  them into the `ScratchBudget` (`charge_residual`), so admission shrinks by
  exactly the leftover bytes instead of overcommitting the volume. Every scored
  request retries the deletion (`retry_residual_sweep`) and releases what is
  genuinely gone; residuals larger than the whole `max_scratch_bytes` fail the
  worker fatally at startup — an operator must reclaim the work dir by hand.

### `perceptual_checks: required | skip`

The three manipulation gates (tone / color-grayscale / chroma-UV) use a deterministic
CPU/OpenCV backend with pinned sampling and threshold configuration. They compare
the candidate to the canonical **miner input at candidate geometry**, matching the
VMAF-delta anti-gaming basis; the pristine holdout remains the quality-reference
basis only. Packets record the basis and input-plan digest. Two sanctioned modes exist:

- **`required`** (default and production): all checks run. Their sufficient statistics
  use fixed evenly spaced frames/pixels and deterministic integer reductions. A missing
  OpenCV/backend dependency refuses the request instead of silently passing it.
- **`skip`**: the three gates are CONSCIOUSLY not run and each records a
  `GateSkip` in the ItemScore packet naming this flag — the same mechanism
  `require_secondary_vmaf=False` uses. Nothing is faked: a skip-mode packet is
  permanently distinguishable from one that genuinely passed. This is what
  preserves a diagnostic/test escape hatch but is not accepted by the production guard.

PieAPP has no skip mode. `PieAppTorchBackend` runs PIQ on CPU by default and auditors
always force CPU. The backend retains a CUDA option for explicit development/calibration,
but the production guard rejects it: launch scoring and independent audit both use the
pinned CPU model. The `1e-5` audit tolerance is a same-model CPU numerical allowance,
not permission for an uncalibrated CUDA score. Modal GPUs are used for miner inference;
the resulting media is scored on CPU and independently reproduced by a CPU-only auditor.
Release weights are digest-verified and preloaded; missing packages/weights or an
unavailable requested device produce a typed refusal.

### Recomputability

Every packet field is a pure function of the request and the pinned tool
versions — no timestamps, no host paths (the recorded canonicalization digest
is the path-independent *template* digest, `plan_template_digest`), no
randomness (the PieAPP start frame is derived from the held-out reference
digest + challenge id) — so re-scoring the same artifacts yields byte-identical
packet JSON and digest (`tests/scoring_worker/test_service_real.py::
test_same_pair_scores_byte_identical_packets`).

## Public API & endpoints

HTTP (port `scoring_worker.port`, default 8201):

| Route | Contract |
|---|---|
| `POST /score` | `ScoreRequest -> ScoreResponse` (exact `item_score_json` bytes + `packet_digest`) |
| `GET /healthz` | `{service, status, checks, scorer_version}` — 200/503; the scorer-identity discovery route |

The only 200s are genuinely measured packets — gate-failed included (a zero
with reasons is a measurement). Typed refusals (`{"detail": {"error": ...}}`):

| Status | `error` values | Meaning |
|---|---|---|
| 409 | `scorer_version_mismatch` | request asserts a scorer this worker is not |
| 422 | `unsupported_track`, `invalid_param`, `file_missing`, `symlink_rejected`, `not_a_regular_file`, `unreadable_input`, `input_too_large`, `input_grew_during_snapshot`, `digest_mismatch`, `unprojectable_stream`, `unsupported_canonical_pix_fmt` | refused on its merits, before/during snapshot or projection |
| 413 | `request_inputs_too_large`, `request_scratch_too_large`, `canonicalized_output_too_large` | can NEVER fit one request's allowance — deterministic, do not retry |
| 503 | `queue_saturated`, `scratch_budget_unavailable` (+ `Retry-After`) | shed — frees itself, come back |
| 501 | `backend_not_configured` | required metric/check dependency or selected device is unavailable — honest refusal |
| 502 | `media_tool_failed` | ffmpeg/ffprobe failed (argv + stderr in the detail) |
| 504 | `scoring_timeout` | started and ran out of `request_timeout` (process groups killed) |
| 500 | `snapshot_corrupt` | our fault: the private copy no longer hashes to the verified digest |

Python surface (re-exported in `__init__.py`): `ScoringWorker`,
`ScoringWorkerConfig`, `ScoringBackends`, `create_app`, `real_backends`,
`effective_scorer_version`, `scorer_identity_digest`, `check_scorer_version`,
`build_health_checks`, `WorkerMetrics`, and from `inputs`: `ByteLimits`,
`ScratchBudget`, `ScratchLease`, `ScoreRejected`, `SnapshotCancelled`,
`InputSnapshot`, `VerifiedInput`, `snapshot_request_inputs`, `sweep_work_dir`,
`projected_canonical_bytes`, `projected_frame_count`,
`projected_metric_log_bytes`, `y4m_frame_bytes`.

Health checks (HealthServer on `metrics_port` AND `/healthz`, both stateless by
design — they read config and filesystem only, never another thread's state):
`work_dir_writable` (unique probe file per call), `media_tools_present`
(skipped in fake mode), plus `http_api_serving` on the service's `/health`.

## Data & invariants

No database. State is the filesystem (`work_dir` scratch, all of it reclaimed
by the startup sweep) and the in-process `ScratchBudget` counters (guarded by a
lock — accounting happens on executor threads).

Invariants worth defending:

- Nothing unverified is ever scored, and nothing but the private 0400 copies is
  ever read after verification; the packet's `content_digest` is the snapshot's
  digest, by construction equal to the request's claim.
- Every byte is reserved before it is written (fstat for copies, probe
  projection for expansions); every reservation is refunded on every exit path
  (`test_expansion_budget.py::test_the_budget_returns_to_zero_on_every_path`).
- `backend: fake` without injected backends is refused at construction — the
  worker never invents a fake on its own.
- The concurrency slot is held for the TRUE lifetime of the work; a timed-out
  request's dying subprocesses still occupy it until they are dead.
- The uvicorn task is monitored: an API that dies on its own flips
  `http_api_serving` and ends the process via `fail_fatal` (non-zero exit → the
  supervisor restarts it; a normal return would read as a deliberate stop).

## Configuration

Section: `scoring_worker` (schema `config.py::ScoringWorkerConfig`). Env
override pattern: `VIDAIO__SCORING_WORKER__<KEY>=<value>`. The shared scoring
levers (`scoring:` section, `ScoringConfig`) are consumed too — they are part
of the scorer identity.

| Key | Default | Meaning |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `8201` | HTTP bind (protocol port map) |
| `metrics_port` | `9103` | Health/metrics port |
| `work_dir` | `./data/scoring-work` | Scratch root (per-request dirs, libvmaf logs); swept at startup |
| `backend` | `real` | `real` shells out to ffmpeg/ffprobe; `fake` REQUIRES injected backends |
| `request_timeout` | `300.0` | Whole-request budget from slot win; overrun kills the process groups → 504 |
| `subprocess_timeout` | `120.0` | Per-ffmpeg/ffprobe invocation budget |
| `max_concurrent` | `2` | Hard bound on scorings in flight (slot held for the work's true lifetime) |
| `queue_wait_timeout_seconds` | `30.0` | Max wait for a slot before 503 + Retry-After |
| `max_input_bytes` | `2147483648` (2 GiB) | Per-file cap (422), decided from the read descriptor's fstat |
| `max_request_bytes` | `4294967296` (4 GiB) | One request's three inputs together (413) |
| `max_request_scratch_bytes` | `17179869184` (16 GiB) | One request's inputs + generated y4m + logs (413); clamped to the worker budget |
| `max_scratch_bytes` | `34359738368` (32 GiB) | Worker-wide live scratch (503 + Retry-After); keep ≥ `max_concurrent × max_request_scratch_bytes` |
| `perceptual_checks` | `required` | `required` = deterministic CPU gates run; `skip` = auditable diagnostic GateSkip records (production guard rejects it) |
| `pieapp_device` | `cpu` | `cpu` is mandatory for the launch scorer/auditor path; `cuda` is development-only until a fresh Modal scorer proves exact CPU packet parity and a versioned production guard permits it |
| `perceptual_cpu` | pinned defaults | Sampling geometry and tone/grayscale/chroma thresholds; included in scorer identity |
| `scorer_version` | `vidaio-scorer/1` | The identity NAME; the stamped identity is `<name>+<digest12>` |
| `ffmpeg_path` / `ffprobe_path` | `ffmpeg` / `ffprobe` | Media tools (PATH or absolute) |
| `vmaf_model_primary` | `version=vmaf_v0.6.1` | Scores; part of the identity digest |
| `vmaf_model_secondary` | `version=vmaf_v0.6.1neg` | Feeds the model-delta gate (NEG rationale above); part of the identity digest |

Construction wiring (not config): `backends=` (a `ScoringBackends` — tests
inject deterministic fakes), `scoring_config=`/`registry=` on `create_app`.
[`config/default.yaml`](../../config/default.yaml), the local stack, and compose all run
`perceptual_checks: required`; production additionally enforces `pieapp_device: cpu`.

## How to test

```sh
python -m pytest tests/scoring_worker
```

By concern: `test_inputs.py` (source-swap/TOCTOU, symlink/fifo/device/dir
422s, grow-during-copy, budgets, the sweep), `test_expansion_budget.py` (the
projection vs real ffmpeg y4m, plane geometry, reservation/shedding/refunds,
the hard cap killing an over-projection, ceiling nesting), `test_scorer_version.py`
(identity minting, what moves it and what must not, the 409, /healthz),
`test_concurrency.py` (queue shed vs timeout, slot accounting under a timeout
burst, process-group kill), `test_service_fake.py` (typed errors, skip-mode
audit records, required-mode 501), `test_service_real.py` (real
ffmpeg/libvmaf end-to-end, byte-identical recompute — skips without media
tools), `test_backends_real.py` (probe/VMAF/canonicalize/tool-version
backends). Full-loop: `the development-tree e2e suite` and `the development-tree stack runner`.

## How to change safely

- Never read a caller-named path after verification — every downstream stage
  must take snapshot paths only; the TOCTOU tests encode this.
- Anything added to `scorer_identity_digest` changes every worker's effective
  identity and invalidates validator pins, competition manifests and challenge
  commitments — add only levers that genuinely change a measured score, and
  never remove the caller-influence property (`check_scorer_version`).
- Keep the 413-vs-503 split exact: 413 is "can never fit" (deterministic),
  503 is "cannot fit now" (self-freeing) — flipping one turns a permanent
  refusal into an eternal retry or vice versa.
- New scratch shapes must be reserved before they exist, refunded on every
  exit path, AND added to `SCRATCH_PREFIXES` so the startup sweep reclaims
  them — an unswept prefix is how the scorer eventually fills its own disk.
- A new refusal is a `ScoreRejected` with a stable `error` string (they are
  asserted by name across the suites), not an ad-hoc HTTPException.
- Never let an unconfigured backend produce a value: the typed-refusal path is the
  integrity invariant (the project design record) — typed refusal, never
  a substituted pass.
- Repo-wide: bump the root `VERSION` on any release-worthy change.

## Status & gaps

- [DONE] Verify-then-snapshot, the four-level scratch budget + expansion
  projection + hard cap, real ffprobe/libvmaf backends (both models),
  canonicalization under budget, scorer-identity minting + 409, bounded
  concurrency with real cancellation, the typed-error surface, startup sweep,
  BaseService lifecycle with fatal API-death handling.
- [DONE] PIQ PieAPP on production-guarded CPU (CUDA retained only for explicit
  development/calibration), deterministic CPU perceptual manipulation checks, pinned
  weight digest/preload, CPU-only auditor composition, and miner-input VMAF model-delta
  evidence.
- [DONE at miner boundary, needs multi-host validation] Remote miners stream bounded
  bytes into the validator-owned `miner_artifact_dir`; the co-located scoring worker is
  given that verified local landing path and still applies its symlink-refusing
  verify-then-snapshot contract. No miner/scorer shared filesystem or peer URL is used.
- [DONE] The VMAF model-delta basis is miner input (owner decision); changing that
  semantics in the future must move the scorer identity and fleet fence.
