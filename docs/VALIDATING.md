# VALIDATING — how to run a VidAIO validator

What a validator operates, what it must protect, and what every failure mode
means. Companion guides: [ARCHITECTURE.md](ARCHITECTURE.md) (topology and trust
boundaries) and the project design record (convergence
math).

> **Status.** The production default is the REAL bittensor chain
> (`chain.mode: bittensor`), and **mainnet netuid 85 is LIVE** — validators run
> against it today. The `BittensorChainAdapter` carried the full public-testnet
> soak: the retained schema-v15 run covered metagraph/archive reads, anchors,
> and authority-vector submissions against the real substrate path, and the
> release receipt contains a clean correction epoch on the exact final image
> plus the final soak. Both modes still drive the *identical* service code —
> only the `ChainAdapter` behind the Protocol swaps; `chain.mode: report`
> remains test/dev/local only.

---

## What a validator identity operates now

The rework made the validator **thin**.
It no longer measures miners for its own weights. The **honest measurement is
central** — an owner-run **Scoring Authority** dispatches challenges, scores with
real ffmpeg/libvmaf, folds the EWMA, runs competitions, and each epoch writes one
immutable **epoch log** to a shared object store and anchors its digest on chain.
A validator identity operates three isolated workloads against that:

1. **Converges** ([`vidaio/weightsetter/README.md`](../vidaio/weightsetter/README.md)):
   fetch the epoch pointer → mirror the epoch-log bytes → verify
   `sha256(bytes) == pointer digest == on-chain anchor` → take the authenticated
   authority-published u16 vector → submit it through its own `set_weights`. This thin
   process is weight-setter only. Two validators reading the same finalized epoch emit
   byte-identical vectors, so Yuma sees one agreed vector.
2. **Runs a full own-auditor**
   ([`vidaio/auditor/README.md`](../vidaio/auditor/README.md)): in a separate OS
   container/cgroup, CPU-recompute every committed item, re-fold earning state, re-derive
   the vector, and POST a signed `audit_mode=own_audit` report.
3. **Runs a beacon auditor**: independently repeat the CPU all-items audit in another
   container/cgroup and POST a signed `audit_mode=beacon` report bound to the
   post-finalization beacon. Each auditor has its own durable cursor and pending-report
   outbox. Neither audit workload, verdict, or delivery failure can delay weight-setting;
   findings flow to the central API and operator alerting for manual remediation.

### What a validator runs

"The validator" is a fleet of supervised processes. The two central APIs and their
epoch-close driver are **owner-run infra** — a third-party validator points at the
owner's `authority_url` / `results_api_url` and runs the thin submit loop plus the two
isolated auditors. The **owner's** validator host additionally runs the Scoring Authority
stack. Wire contracts and the port map are owned by
[`vidaio/services/protocol.py`](../vidaio/services/protocol.py):

> **Getting access.** The owner's authority pointer API URL, the Audit Results
> API URL, and the auditor read tokens (the values behind
> `VIDAIO_AUTHORITY_READ_TOKEN` / `VIDAIO_AUDIT_RESULTS_TOKEN`) are published
> in the VidAIO Discord (invite in the README). Auditing the public S3
> evidence itself needs NO token or key — anonymous read is the design.

| Process | What it does | API port | Metrics port |
|---|---|---|---|
| **weight-setter** (thin validator) | fetch → authenticate → submit the authority vector unchanged; no audit/scoring client ([`vidaio/weightsetter/README.md`](../vidaio/weightsetter/README.md)) | — (no API) | 9102 |
| **beacon auditor** (`auditor`) | CPU all-items recompute + re-fold + signed `audit_mode=beacon` reports ([`vidaio/auditor/README.md`](../vidaio/auditor/README.md)) | — (loop) | 9121 (Bittensor production) |
| **full own-auditor** (`own-auditor`) | independently repeats CPU all-items recompute + re-fold + signed `audit_mode=own_audit` reports | — (loop) | 9122 (Bittensor production) |
| scoring authority (owner) | the THIN POINTER API over the epoch object store ([`vidaio/authority/README.md`](../vidaio/authority/README.md)) | 8700 | 9111 |
| authority-finalizer (owner) | the epoch-CLOSE driver: snapshot → manifest → `_FINALIZED` log → anchor | — (loop) | 9120 (Bittensor production) |
| audit-results API (owner) | the honesty surface auditors POST to ([`vidaio/audit_api/README.md`](../vidaio/audit_api/README.md)) | 8710 | 9112 |
| inference validator (owner) | central scorer's dispatch+measure+fold feeder ([`vidaio/validator/README.md`](../vidaio/validator/README.md)) | — (loop) | 9101 |
| scoring worker (owner) | real ffmpeg/libvmaf measurement over HTTP (`vidaio/scoring_worker/`) | 8201 | 9103 |
| challenge service (owner) | ingest + degradation DAGs; holds the held-out reference (owner-operated, not in this repo) | 8210 | 9105 |
| competition orchestrator (owner) | drives competitions, Docker sandboxes ([`vidaio/competition/README.md`](../vidaio/competition/README.md)) | 8500 (control) | 9104 |
| organic gateway (owner, optional) | customer ingress + champion routing (owner-operated, not in this repo) | 29996 | 9107 |
| chainsim (report mode only) | the simulated chain ([`vidaio/chainsim/README.md`](../vidaio/chainsim/README.md)) | 8400 | 9108 |

All extend `BaseService` ([`vidaio/services/README.md`](../vidaio/services/README.md)):
JSON logging, `/health` + `/metrics` on the metrics port, graceful shutdown, and
**the exit-code contract** — exit 0 = deliberate stop (never restarted),
non-zero = crash (restarted with backoff). The shipped **supervisor**
(`vidaio/validator/supervisor.py`) runs each service as a separate OS process
(spawn context), restarts crashes with exponential backoff, and PARKS a child
that crash-loops (default 5 restarts / 600 s) while everything else keeps
running. The reference wiring of the whole fleet (all central APIs + a thin
`provider: shared` weight-setter + two isolated auditors) runs under that
supervisor on the owner's side; a third-party validator wires up only the
weight-setter and the two auditors.

---

## Requirements

- **A chain wallet (hotkey).** A validator now MUST carry an on-chain identity: in
  bittensor mode a hotkey-only wallet — env seed (`chain.hotkey_seed_env`, default
  `VIDAIO_HOTKEY_SEED`) or an on-disk btcli wallet (`chain.wallet_name` /
  `wallet_hotkey`); coldkey never touched at runtime; registration via
  `btcli subnet register` on netuid 85. The pod **fails fast at startup** if
  neither a wallet nor a seed is present in bittensor mode. In report mode the
  chainsim stand-in is `chain.validator_hotkey` + a bearer `chain.auth_token`.
- **ffmpeg + libvmaf** on `PATH` — both AUDITORS genuinely recompute over real
  ffmpeg/libvmaf (`RealScoreRecomputer`, `vidaio/auditor/recomputer.py`), and the
  owner's scoring worker/challenge service execute them too. Nothing is mocked in
  the shipped composition.
- **The shared object store is load-bearing.** The epoch logs AND the audit files
  the validator mirrors and the auditors recompute over live in the object store
  (`audit.local_root` for `LocalFsStore`; S3/Hippius in production). Each auditor uses a
  separate unsigned, anonymous, read-only client. The thin weight-setter uses scoped credentials
  (or workload IAM) to read public evidence and publish only `weight_vector`/`manifest`;
  its store rejects sealed-kind operations and never loads the live holdout AES key. A
  weight-setter that cannot reach the store HOLDs (it never improvises a vector). An
  auditor outage is independently reported/alerted and does not gate that submit path.
- **Docker**: required only for the owner's competition sandboxes; not needed by the
  thin submit loop or either auditor's recomputation logic (the deployment still uses
  containers to enforce isolation).
- **Python ≥ 3.11** (developed on 3.13) + `uv`; see the root README's Getting
  started.
- **Disk** — the audit store (`audit.local_root`) grows monotonically (write-once,
  `retention_days: 0` = retain forever); each auditor's recompute scratch is
  transient. Audit downloads are streamed to verified files (2 GiB input/reference,
  4 GiB output; 16 MiB metadata, 1 MiB bundle, 64 MiB epoch-log caps). Budget scoring
  scratch on the owner's worker host and each CPU auditor per its
  `scoring_worker` budgets ([`config/default.yaml`](../config/default.yaml)).
- **GPU: not required for any validator audit.** Compression recomputes with CPU
  libvmaf; upscaling recomputes with PIQ PieAPP on CPU; tone, grayscale, and chroma
  checks use the deterministic CPU/OpenCV backend. The release dependency check
  rejects CUDA-bearing PyTorch wheels and preloads the pinned PieAPP weights. A GPU
  may accelerate miner inference or a separately configured scorer, but every score
  must remain reproducible by an independent CPU-only auditor.

---

## Keys, identity, and tokens

### Registered-hotkey authentication (P2, `hotkey_auth:` section)

Every miner/validator-facing surface verifies the caller is a REGISTERED hotkey
on this subnet, and validator-only routes (the authority pointer API, the
challenge service's `/challenge/next` and `/resolve`) additionally require a
validator permit. Two ways to authenticate, both signed with the wallet the
validator already holds — no new secret is provisioned:

- **Signed request (Scheme A)**: send `X-Vidaio-Hotkey`, `X-Vidaio-Timestamp`
  (unix seconds), `X-Vidaio-Nonce` (128-bit lowercase hex, single-use) and
  `X-Vidaio-Signature` — the sr25519 signature over the domain-separated
  canonical digest of `(method, path, sha256(body), timestamp, nonce)`.
  `vidaio.services.hotkey_auth.sign_request_headers(...)` builds these.
- **Session token (Scheme B, for polling)**: `POST /auth/challenge` on the
  authority API → sign the returned nonce (Scheme A envelope over
  `POST /auth/token` with the nonce as body) → receive a short-lived opaque
  token; use it as `Authorization: Bearer vk1....`. Registration is re-checked
  from a cached registry on every use, so DEREGISTRATION REVOKES within ~45 s.

Rollout: `hotkey_auth.mode: log` observes (refusals are logged, nothing is
refused — an observe-only migration posture); `enforce` refuses with 401/403/503. Production
preflight requires `enforce` on mainnet. The static bearer tokens remain a
second factor during migration; operator tokens are permanent (operators are
not hotkeys).

- **Chain identity (the writing and artifact-request hotkey).**
  `chain.validator_hotkey` is the ss58 the vector is submitted and read back under;
  the inference validator and organic gateway also sign artifact-v2 requests with
  this wallet. In bittensor mode it must be a real, registered validator-permit
  hotkey; in report mode it identifies the process on the sim (claimed with
  `chain.auth_token`). `validator.identity` must name the same hotkey.
- **`weightsetter.validator_hotkey`** — this validator's REAL hotkey ss58, used to
  read its own vector back after an ambiguous submit AND as the identity the
  convergence gauge compares peers against. **Empty ⇒ fail-fast in bittensor mode**
  (it is a convergence-critical misconfig, not a quiet degradation).
- **`weightsetter.authority_url` / `authority_token`** — where the shared provider
  fetches the epoch pointer from the Scoring Authority (its `authority.api_token`).
  Empty `authority_url` ⇒ fail-fast in bittensor mode.
- **`local_stack.authority_url` / `audit_api_url`** — explicit remote endpoints for
  both auditor processes; production may not rely on loopback defaults.
- **`auditor.authority_api_token_env` / `results_api_token_env`** — names of the
  client-only raw bearer variables read by both auditor processes (defaults:
  `VIDAIO_AUTHORITY_READ_TOKEN` and `VIDAIO_AUDIT_RESULTS_TOKEN`). Bittensor auditors
  never receive or fall back to either API server's nested config token.
- **`local_stack.auditor_media_sample_rate` /
  `own_auditor_media_sample_rate`** — both are exactly `1.0` in production, selecting
  uncapped `all_items` CPU coverage. `auditor_cursor_db_path` and
  `own_auditor_cursor_db_path` must resolve to distinct durable volumes; pending-report
  stores are likewise process-private.
- **`local_stack.auditor_cursor_floor`** — one positive, future runtime
  `SubnetEpochIndex` chosen before launch as the authority genesis (normally latest
  archive-proven closed index + 1). Every process uses the same value; a fresh/lost
  cursor walks from it and never trusts a pointer 404 as a pruning boundary. Sharing the
  genesis value does not mean sharing a cursor database.
- **`auditor.auditor_hotkey`** + the report signer — every `AuditReport` is
  attributed to and hotkey-signed under it. In production the report signer is the
  validator's hotkey keypair; the Audit Results API verifies it with a real
  `HotkeySignatureVerifier` (an sr25519/ed25519 signature over the report's
  canonical bytes). See "The audit obligations" below.
- **`audit_api.api_token`** — gates `POST /audit/report`; only the API server carries
  this nested config key. Auditors carry the matching client-only raw token; READS are
  open (the honesty surface is public).
- **`dashboard.api_token`** — gates `/operator` and `/api/operator`. Bittensor
  full-stack and dashboard-role startup refuse an empty/whitespace value; the
  public and miner views remain open.
- **`challenge_service.api_token` / `challenge_service.operator_token` /
  `chainsim.operator_token` / `orchestrator.control_token` / `miner.api_token`** — owner-side service tokens
  (unchanged). Miners must never reach the
  challenge service (it holds the held-out reference).

The miner token is only an extra shared-secret option for an operator-controlled
reference fleet. The canonical artifact-v2 exchange is already authenticated in both
directions: the current validator hotkey signs the task metadata, intended miner,
timestamp, 128-bit nonce, and exact input size; the chain-attributed miner hotkey signs
the bound response digest and size. The miner verifies current validator permit before
reading the body, keeps bounded global/per-validator live replay state, and applies a
short start-plus-skew timestamp fence after restart. HTTP 425
`artifact_auth_starting` means retry with a fresh timestamp/nonce after that bounded
startup window. A permissionless subnet must
leave the optional fleet-wide bearer unset and still enforce TLS plus request-size,
rate, connection, concurrency, and timeout protections at the public edge.

The dev/local stack pins dev-only token values; none is a key to anything
real. Production values are env-injected (`VIDAIO__<SECTION>__<KEY>`), never
committed.

### The scorer-identity pin (owner-side)

The scoring worker mints ONE identity —
`<scorer_version>+<config-and-runtime digest[:12]>`, a sha256 over every scoring
lever and the complete canonical payout runtime — published on its
`GET /healthz` (THE SCORER-IDENTITY CONTRACT,
[`vidaio/services/protocol.py`](../vidaio/services/protocol.py)). The central
scorer pins it; the finalizer stamps it into every epoch log (`scorer_version`);
the auditor re-derives weights with the SAME tokenomics levers or an honest log
would false-flag `WEIGHT_DERIVATION_MISMATCH`. A `version_key` (below) fences a
mixed-version fleet so two schema versions never submit two different "correct"
vectors.

Launch scoring and auditing run only inside the digest-pinned Linux/amd64 release
image. Its verified runtime manifest, image-only marker, one-thread CPU policy,
ffmpeg/ffprobe/libvmaf stamps and Python/PIQ/PyTorch/OpenCV/NumPy versions form a
full runtime commitment. Every score packet carries that commitment digest and a
strict auditor independently produces the same stamp. A native checkout, another
OS/architecture, a stale manifest or a different backend build therefore has a
different identity and is not launch acceptance evidence; the `1e-5` PieAPP
tolerance is unchanged.

---

## Operations

### Outage gaps (epoch schema v16) — what a spine does after downtime

An epoch can only be anchored inside its un-grindable window (`close_block + K`
blocks, K = the confirmation depth — minutes, not hours). Any authority outage
longer than that leaves epochs that can NEVER be finalized: publishing their
log late would make the already-known beacon grindable, so the finalizer
refuses forever. Before v16 this permanently wedged the spine. Now:

- The finalizer SKIPS the un-anchorable epochs and DECLARES them as
  `gap_epochs` in the next anchorable epoch's log — the full contiguous range,
  validated at the schema boundary, signed and anchored like every other log
  field. The outage becomes an auditable on-chain fact.
- The digest chain and the earning carry-in continue from the last REAL
  predecessor (`prior_epoch_id` is gap-aware). Nobody's accumulated earnings
  reset; the skipped epochs simply never fold new scores.
- Gaps up to `authority.max_auto_gap_epochs` (default 48) self-heal on
  restart. A larger gap fails loudly until the operator sets
  `VIDAIO__AUTHORITY__GAP_ACK_THROUGH_EPOCH=<last gap epoch>` — an outage that
  long deserves a human before earnings resume. Remove the ack after recovery.
- The rare crash-orphan (an epoch indexed but not yet anchored when the outage
  hit) has the same named remediation: with the ack set, the orphaned row is
  TOMBSTONED in the authority index (kept forever as audit trail, never served
  again) and the spine resumes from the previous anchored epoch.
- **Auditors** advance their contiguous cursor over a gap ONLY when the next
  anchored log's declaration matches the withheld range exactly and verifies
  against the on-chain anchor. A silent 404 still HOLDs forever — a gap
  declaration is proof-carrying, never an unauthenticated skip.
- A FRESH deployment never auto-gaps: the preflight floor rule
  (`auditor_cursor_floor` = latest closed + 1, exactly) stays strict.

### The convergence attempt (what "working" looks like)

One weight-setter attempt (tempo-gated; attempts before the chain window opens are
normal). The shared provider
([`vidaio/weightsetter/README.md`](../vidaio/weightsetter/README.md),
`SharedSnapshotProvider`) does:

1. `GET {authority_url}/epoch/latest` → the pointer `{snapshot_key,
   snapshot_digest, weight_vector_digest, anchor{txid,digest}}`.
2. mirror the epoch-log BYTES from the object store by `snapshot_key` (never a
   half-written set — the `_FINALIZED` guard).
3. **three-way verify:** `sha256(bytes) == pointer digest == on-chain anchored
   digest`. The on-chain leg is **mandatory** in bittensor mode
   (`verify_anchor: true`): a `ChainAdapterAnchorReader` reads the digest straight
   off the chain adapter (sim `/state` in report mode; Commitments pallet in
   production). A finalized-but-not-yet-anchored epoch HOLDs; any inequality is
   `SnapshotDigestMismatch` → REFUSE (CRITICAL), never a quiet submit.
4. read the authenticated log's stated `weight_u16` as the SDK-input sum-grid.
5. `set_weights` (own hotkey, tempo-gated) → the pinned SDK emits the deterministic max-grid;
   read own vector back
   (CONFIRMED / DENIED / UNKNOWN) → on CONFIRMED, publish + anchor a
   `PublicationRecord`.
6. **convergence gauge** (observe-only): read PEER peer validators' on-chain
   vectors and emit `vidaio_weightsetter_convergence` = fraction agreeing. It NEVER
   changes what this validator submits; it surfaces divergence before it costs
   emissions.

**Unallocated fixed pools ⇒ canonical sink convergence**:
a genuinely empty epoch is the
trivially-identical 100%-sink vector. A non-empty epoch may name the same UID beside
earners when an empty/below-floor inference track, incomplete fresh podium, or
deregistered crowned hotkey leaves a fixed share unallocated. This prevents chain
normalization from donating it to another pool. An ops/DB/archive failure is never a
sink event: the finalizer HOLDs.

### The two isolated auditor loops

On independent cadences ([`vidaio/auditor/README.md`](../vidaio/auditor/README.md)), the
beacon auditor and full own-auditor each fetch the newest finalized pointer, walk history
from their own durable contiguous cursor, mirror the epoch log, and:

- **select every committed item on CPU in production** (`all_items`, rate `1.0`). The
  beacon auditor's general selection algorithm is seeded from
  `(beacon, epoch_id, auditor_hotkey)` — the beacon is the finalized hash of the
  fixed future block `close_block + K`, not an authority-selected log digest or
  rerollable anchor value. The anchor must land before that beacon becomes knowable,
  so the authority cannot grind invalid item ids outside a reduced third-party sample
  (`vidaio/auditor/sampling.py` and `vidaio/auditor/beacon.py`). The own-auditor does not
  depend on a sampling draw;
- **recomputes** each selected item over the REAL engine
  (`RealScoreRecomputer` + `verify_bundle`, strict merkle inclusion); a tampered
  score is `SCORE_MISMATCH`, an out-of-committed-set packet is `MERKLE_EXCLUSION`,
  an item it cannot honestly recompute (for example, required CPU media/model bytes
  are unavailable) is **SKIP** (never a false PASS);
- **re-folds the EARNING STATE**: every inference earner's `accumulate_score`
  re-folded from the audited packet scores + the chained prior-epoch carry-in
  (`prior_log_digest` back to genesis) — a substituted earning state published
  alongside honest packets is `EARNING_STATE_MISMATCH`;
- **rebuilds competition economics** from each subject's exact score packets and paired
  audit bundles, using packet means and stable score/hotkey/uid ordering rather than
  stored human ranks/review state; then reconstructs the archived-baseline-relative
  result and predecessor-folded PODIUM/CROWN window. Any mismatch is a provable
  competition/window failure;
- **re-derives** the weight vector and compares digests
  (`WEIGHT_DERIVATION_MISMATCH` otherwise);
- **POSTs a hotkey-signed `AuditReport`** to the Audit Results API through its own durable
  pending-report outbox. Its `overall`
  is DERIVED (never a free field): **DISPUTED** if any item/weight/earning FAILs,
  **INCONCLUSIVE** if the selected media items all SKIP (nothing recomputed — not
  clean, needs attention), else **CLEAN**. The dashboard reads the aggregate; a
  DISPUTED epoch is publicly visible. The two immutable report identities use
  `audit_mode=beacon` and `audit_mode=own_audit`. Both are investigation and alerting
  surfaces only; neither worker, verdict, nor delivery failure is consumed as a
  `set_weights` gate.

### What the logs / metrics / health tell you

CRv4 commitment finalization is not yet an active weight vector. Its durable
intent remains `commit_reveal_pending`; reconciliation polls every 300 seconds
between scheduled submissions (`weightsetter.reconciliation_interval_seconds`),
**only while such a pending row exists**. Ordinary ambiguous or accepted rows alone
do not enable chain/metagraph polling. Accepted-only publication queues get
separate bounded retry wakeups without those reads. The production read pins all pending-commit pages,
`Uids`, `Weights`, and `LastUpdate` to the **same finalized hash**. Only a dated,
matching vector accepts the intent and updates the success counter/clock once.
No commit plus old/empty weights is a finality gap, not rejection: a known CR
intent HOLDS at any age and blocks another submission until confirmed.
Startup/scheduled reconciliation also covers legacy `pending` rows with NULL
resolution: a strictly boolean `commit_reveal_enabled() == True` durably labels
them `commit_reveal_pending` before reading old/empty weights. This is a
conservative obligation, **not proof the old commitment landed**. Unreadable or
non-boolean mode leaves resolution NULL, records UNKNOWN, and blocks scheduled
resubmission/abandonment; a positive False preserves ordinary non-CR recovery.
Already-labeled CR rows remain protected even if the current mode later changes.
Legacy adapters without a CR-mode surface keep their ordinary non-CR behavior;
the production Bittensor adapter always exposes this fail-closed read. Its
readback now refuses transports lacking the coherent finalized-read surface.

Polling never calls `set_weights` or advances its fixed cadence, but **can WRITE
publication anchors** after confirmation. A poll drains at most one accepted row,
only when its publication timeout budget fits before the next submission. All
publication paths (including the scheduled drain and explicit reconciliation)
share at most **three starts per rolling attempt interval**, including the first
attempt, across all intents. Each start makes at most one anchor call. Failures
back off exponentially from 300 seconds (300, 600, 1,200, ...), capped at the
attempt interval; scheduled drains cannot bypass the backoff. A timed-out worker
stays single-flight. When the last pending CR row clears, failed publications
still get backoff-aware wakeups without scanning the metagraph. An exhausted
budget or in-flight task adds no publication-only wakeups before the next
eligible time; the scheduled drain uses the same gate. Here the retry "tempo"
means the configured submission interval, not a newly queried subnet epoch.
Migration `0010_publication_retry_budget.sql` durably reserves each publication
start before launching it, with its intent id, UTC start/completion timestamps,
failure count and retry deadline. Restart cannot reset either the rolling cap or
exponential backoff. An interrupted reservation still consumes a start and waits
at least the publication timeout plus backoff before recovery. Within a running
process, even a worker outliving that timeout remains single-flight. The
once-per-intent INFO reveal-wait claim is durable too (a crash between claiming
and emitting can omit that informational message rather than duplicate it).
Persistent deadlines assume a sane host UTC clock; submission cadence still uses
monotonic time. This does not promise exactly-once external anchors after abrupt
process death with an ambiguous remote outcome, or support overlapping service
processes beyond the existing writer lock. Accepted rows drain oldest-first;
a persistently failing old publication can delay younger rows.
Only `PendingWeightReveal`, raised on a positive finalized pending-commit read,
uses the once-only INFO path. Generic `ChainStateUnavailable` RPC/storage errors
still warn on every failed read, including for known CR intents, while HOLDing.
Unconfirmed CR intents can HOLD
indefinitely and require investigation rather than automatic abandonment.

The effective `last_success_age` limit clamps `max_last_success_age_seconds`
(default **8,640 seconds / 2.4 hours**, previously 4.8 hours) between
`attempt_interval_seconds + reveal_grace_seconds` and
`2 * attempt_interval_seconds + reveal_grace_seconds`. The defaults are 4,320
seconds for both the attempt interval and reveal grace; even an oversized health
override cannot mask more than one missed submission tempo beyond that allowance.
Set the reveal allowance for the subnet's actual tempo/reveal period plus
finalization margin; it is a health allowance, not permission to publish early.
A pending commitment alone never refreshes success health.

- **Structured JSON logs** from every service; every HOLD, refusal and transition
  carries a machine-readable reason. Each audit worker emits a local CRITICAL
  (`DISPUTED`) or WARNING (`INCONCLUSIVE`) before central delivery; the Audit Results API
  repeats the appropriate operator signal when accepting a finding and logs every
  divergent report conflict. All name manual remediation and leave weights unchanged.
- **`/health`** per service (metrics port): the weight-setter checks
  `last_success_age`, `db`; the beacon and own-auditor expose separately visible health
  on 9121 and 9122; the scoring-authority checks `http_api`; every service
  grows a pinned-red `fatal_failure` check after `fail_fatal` fires.
- **`/metrics`** (Prometheus): `vidaio_weightsetter_convergence` (peer agreement
  fraction), `weightsetter_chain_state_skips_total`, `weightsetter_pending_intents`,
  each audit endpoint's accepted `auditor_reports_total{status}`,
  `auditor_report_delivery_attempts_total{status}`, and
  `auditor_report_delivery_failures_total{status}`; the central API's
  `vidaio_audit_reports_received_total{verdict}`,
  `vidaio_audit_reports_received_by_mode_total{audit_mode,verdict}`,
  `vidaio_audit_report_conflicts_total`,
  `vidaio_audit_report_conflicts_by_mode_total{audit_mode}`, and
  `vidaio_audit_disputed_epochs`; plus `vidaio_authority_epochs_finalized_total` and
  `vidaio_authority_pointer_reads_total`. Each service's `/metrics` endpoint
  exposes its full list.
- **The Audit Results dashboard** (`GET /audit/status?epoch_id=…`,
  `/audit/feed`, `/audit/epochs`): the honesty surface. `CLEAN` /
  `DISPUTED` / `INCONCLUSIVE` / `UNAUDITED` per epoch, disputed items + their
  reason codes, and per-auditor reports.
- **Report files** (report mode): `report-<ts>.json/.md` under
  `chainsim.report_dir` — registered neurons, weight-vector history, anchored
  commitments, emission summary.

### Common failure modes, their reason codes, and what you do

| Signal | Meaning | Operator action |
|---|---|---|
| **fail-fast at startup** (`SystemExit`: "bittensor-mode weight-setter is misconfigured — convergence would break") | In bittensor mode, `weightsetter.provider != shared`, an empty `validator_hotkey`, `verify_anchor: false`, or an empty `authority_url`. A misconfig that would silently break convergence must be LOUD. | Fix `weightsetter:` — `provider: shared`, the real hotkey ss58, the authority URL, `verify_anchor: true` — and restart. |
| `snapshot_unavailable` (attempt SKIPS; `weightsetter_chain_state_skips_total`) | The Scoring Authority API / object store is unreachable, the epoch set is not `_FINALIZED`, the bytes could not be mirrored, or the epoch is finalized-but-not-yet-anchored (the mandatory third leg cannot be verified). HOLD is deliberate — the last confirmed vector persists; the validator NEVER falls back to local sampling. | Restore the authority / object store / chain endpoint; the validator re-converges as soon as it can mirror + verify. |
| **anchor-mismatch HOLD / `SnapshotDigestMismatch`** (REFUSE, CRITICAL) | `sha256(bytes)` ≠ pointer digest, the pointer's anchor field disagrees, the on-chain anchored digest disagrees, the pointer claims a txid but the chain holds no anchor, or the mirrored bytes fail their own epoch-log invariants. Someone tampered. A mutated log is never quantized into a submission. | Investigate the tamper: compare the anchored digest on chain with the pointer. Do not force a submit. |
| **isolated own-auditor finding** (`DISPUTED` / `INCONCLUSIVE`, logged, metered, and centrally reported) | The standalone own-auditor found a `WEIGHT_DERIVATION_MISMATCH`, an `EARNING_STATE_MISMATCH`, another provable fault, or could not recompute evidence. The authenticated authority vector is still submitted on schedule by the separate thin validator. Signed `audit_mode=own_audit` and `audit_mode=beacon` reports coexist for the hotkey and epoch. | Investigate the epoch and its Audit Results API/operator alerts. Any remediation is manual; do not expect or add an automatic weight HOLD. |
| `EARNING_STATE_MISMATCH` (auditor FAIL → epoch DISPUTED) | An inference uid's stated `accumulate_score` is NOT the EWMA fold of its audited packet scores + the chained carry-in, or its cycle scores are unbacked. The authority cannot publish honest packets with a substituted earning state. | Treat it as a provable, anchored incident, investigate, and choose remediation manually. Validators continue submitting the authenticated authority vector meanwhile. |
| competition/window mismatch (auditor FAIL → epoch DISPUTED) | The exact competition packet/bundle evidence does not reproduce the machine-score ordering, archived-baseline margin, `CompetitionResult`, predecessor-folded PODIUM/CROWN window, or resulting pool weights. Human review fields cannot make this pass. | Inspect the committed packets/bundles and authority bridge, alert operators, and remediate manually; the audit verdict itself does not interrupt weight-setting. |
| `WEIGHT_DERIVATION_MISMATCH` (auditor/own-audit FAIL) | The published weight vector does not follow from the log's stated inputs (`build_weight_vector` + `quantize_u16` yields a different digest). | Report and alert the provable, anchored mismatch. Submit the authenticated authority-published vector unchanged until an operator takes explicit remedial action. |
| **convergence divergence** (`vidaio_weightsetter_convergence` < 1.0) | Peer peer validators submitted different on-chain vectors this epoch. Root cause is a log/pipeline-version mismatch (there is no per-validator sampling difference anymore) — often a mixed `version_key` / epoch-log schema fleet, or a peer on a stale epoch. | Fence the fleet on `weightsetter.version_key` (bump it on any quantizer/epoch-log schema change); confirm all peer validators run the same build and read the same authority. Observe-only — it never changes your own submission. |
| `weightsetter_pending_intents` not coming down | Weight intents stuck UNKNOWN — the chain cannot confirm or deny our vector (no `validator_hotkey`, or the read path down). Held deliberately: "we could not find out" is not evidence of absence. | Set `weightsetter.validator_hotkey`; restore the read path. Never force-settle by hand. |
| Child PARKED by the supervisor (CRITICAL) | Crash loop exhausted the restart budget. | Read the child's last logs, fix, restart the stack (or the child). |

---

## The audit obligations (why you run this, not just what)

Everything the Scoring Authority publishes must be independently recomputable, and
the validator's SECOND hat is to prove it
([`vidaio/auditor/README.md`](../vidaio/auditor/README.md),
[`vidaio/audit/README.md`](../vidaio/audit/README.md)):

- **The schema-v15 epoch log is the ONE artifact both hats consume.** It carries
  the weight vector, exact close-block `miner_census`, `prior_log_digest` (chaining
  epochs back to genesis), the complete total `fold_cursors` map (including null
  first-fold entries and deregistration tombstones), the audit
  manifest (per-uid backing files + score-packet **merkle root + per-item inclusion
  proofs**), `earning_inputs` (the verifiable inference earning-state derivation), and
  `competition_input` / `competition_bundles` plus the stated result and chained
  `reward_window_state`. It is
  content-addressed (`sha256 == log_digest`) and
  that digest is **anchored on chain** — the tamper-evidence root
  ([`vidaio/epoch/README.md`](../vidaio/epoch/README.md)).
- **The media is durable and CPU-recomputable.** Challenge input, miner output,
  manifest, packet, DAG reveal, and encrypted live reference are persisted in the
  S3-compatible store. After terminal single-use retirement, verified reference
  plaintext is published under `released/reference_original/…`; the canonical
  `reference_original/` remains private. The public evidence set is an exact
  prefix whitelist enforced by bucket policy; everything else stays private.
- **The Audit Results API is fail-closed and tamper-evident**
  ([`vidaio/audit_api/README.md`](../vidaio/audit_api/README.md)): every report is
  hotkey-signed over its canonical bytes and verified against the claimed
  `auditor_hotkey` by a real `HotkeySignatureVerifier`. Where no verifier is
  configured the default is `RejectingVerifier` (refuse every report) — a
  misconfigured deployment is loud, never spoofable. The insecure `Sha256Verifier`
  double is opt-in ONLY via `audit_api.dev_insecure_verifier: true` for chainless
  runs. The `overall` roll-up is DERIVED at both ends, so a CLEAN report can never
  carry a FAIL, and an all-SKIP sample is never washed to CLEAN. Signed
  `audit_mode=own_audit` and `audit_mode=beacon` reports coexist under the immutable
  `(auditor_hotkey, epoch_id, audit_mode)` key; modes and findings are visible for
  investigation but never automatically control weight submission.
- **The aggregate cannot be out-voted.** `GET /audit/status` recomputes each
  report's verdict and asserts DISPUTED if ANY auditor reported a provable FAIL — an
  honest majority cannot bury one auditor's provable fault.
- **The integrity invariants** (enforced in review):
  scores and weights come from one uniform code path for every hotkey; the calibration
  baseline is non-earning; outputs are measured from the miner's own submission;
  timestamps and logs are append-only and auditable. Where a measurement cannot be
  made, the answer is a typed refusal — recorded as such — never a substituted pass.
  Run the published scoring/weights/quantizer code unmodified; the whole system exists
  to make that verifiable.

---

## Mainnet (live)

Validators run against mainnet netuid 85 today. The `BittensorChainAdapter`
is the production chain path (single long-lived socket, condemn-on-timeout
`set_weights`, CRv4 `TimelockedWeightCommits`-aware `submitted_weights`,
`anchor_commitment`, the shared `quantize_u16`). `chain.mode: bittensor` is the
production default and builds it (lazily importing the optional `.[chain]`
bittensor deps — a missing dep or wallet fails fast with `NotConfiguredError`
pointing at the extra). ONLY the ChainAdapter swaps; every service, token,
provider, isolated own-auditor reporting path, and audit duty above is
identical between report and bittensor mode. The live-testnet run exercised
metagraph sync, weight cadence, and anchor commitments against the real chain.
Historical testnet epoch 62 remains an immutable DISPUTED audit because the
auditor service adapter omitted already-committed upscaling geometry from its
recompute context; release acceptance required — and the release receipt
contains — a clean competition-bearing correction epoch under the exact final
image, plus resource stability, failure recovery, and pending-intent behavior
proven during the final soak.
Adapter-specific operational notes: `chain_timeout_seconds` must stay ≥180 s;
a commit-reveal submit can end UNKNOWN and must be read back, never assumed
landed; `blocks_since_last_update` is the "am I actually weight-setting"
gauge.

Competition earnings are ON. Schema v15 commits the
pre-enrollment manifest/commitment root, exact archived-baseline artifact, provenance,
git and image identity, every contender's sealed source archive and execution identity,
the full item/packet/bundle matrix, the global cycle ordinal, and the chain-derived epoch
application time. Auditors rebuild the result, inclusive 5% state transition, half-open
window, and final vector. CROWN publication additionally requires the exact winning
source archive to be public. Findings remain report-and-alert only for
weight-setting; remediation is manual.
