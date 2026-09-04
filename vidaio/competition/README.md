# vidaio.competition — competition core: manifest, lifecycle, persistence, review, orchestrator, sandbox runners

The compression and upscaling competition system (spec design spec §04–§06, §14): a strictly
validated manifest, a guarded phase state machine over an auditability-first SQLite
schema, packet-bound score persistence, hash-chained human review — plus two
subpackages: `orchestrator/` (the service that drives it) and `runners/` (the
sandbox implementations). Locked choices bind throughout:
the project design record #1 (baseline = non-earning calibration),
#3/#7 (local-first, fresh instances only for GPU), #8 (real Bittensor is the shipping
default; report mode is the local/test overlay and only the ChainAdapter swaps). Downstream: a COMPLETED
competition becomes a tokenomics `CompetitionResult`
([vidaio/tokenomics](../tokenomics/README.md)); every persisted score is linked to
an audit bundle ([vidaio/audit](../audit/README.md)).

> **Competition/crown is live and earning on testnet**
> (the project design record #13). The shipping config sets
> `tokenomics.competition_emissions_enabled: true`; this is an emergency off switch.
> Schema v15 derives economic order only from the complete, exact score-packet matrix and
> paired audit bundles. A CPU-only auditor independently rebuilds the packet means,
> result, global seven-day PODIUM/CROWN window, and vector. Human review remains append-only operational context
> but never changes emissions.

## What it does

**Manifest** (`manifest.py`) — `CompetitionManifest` is frozen,
`extra="forbid"`-strict, with ordered timezone-aware lifecycle times (normalized to
UTC so the digest is representation-independent), convex `ScoringFactors`
(sum exactly 1.0; comp-01 uses 0.6/0.0/0.4), sealed per-item VMAF variants, an
`allowed_gpus` list, batch-size bounds, a `scoring_seed_commitment` (sha256 of the
dataset-variant RNG seed — the hash, never the seed), and `manifest_digest()` =
sha256 over canonical JSON (pre-committed on chain by the audit module before
enrollment). The **baseline block carries pinned executable provenance**: version,
source-archive digest/size, provenance digest/size, execution-image digest, and
`repo_url + commit_sha + tree_sha`. It has no hotkey and **no payout field of any
kind is possible** — `extra="forbid"` rejects unknown keys and a before-validator
turns any payout-shaped key (`hotkey`, `coldkey`, `payout`, `wallet`, `reward`,
`emission` substrings) into a pointed error citing decision #1.
`validate_against_config` checks the schema-valid manifest against the
operator-configured envelope (`CompetitionConfig` bounds) before a row is created.
Upscaling manifests additionally carry ordered `allowed_upscale_factors` and one
`evaluation_item_commitments` entry per item. New entries are sha256 over the canonical
`vidaio.competition.evaluation-item.v2` preimage containing `competition_id`,
`item_index`, the distinct pristine-reference and low-resolution-input digests,
`upscale_factor`, and exact `target_width`/`target_height`. The list is inside the
anchored manifest digest; mutable database columns are never the trust root. A v1
preimage without geometry remains verifiable/deserializable for already-anchored rows;
new rows require v2. Absent upscaling-only fields are omitted from legacy compression
canonical bytes so already-anchored digests do not change on migration.

**Phase state machine** (`states.py` + `engine.py`) —
`SCHEDULED → ENROLLING → FINALIZING_SUBMISSIONS → VALIDATING → BUILDING →
EVALUATING → SCORING → AWAITING_END_TIME → COMPLETED`, plus the spec's
FAILED/CANCELLED edges. `TRANSITIONS` is the single source of truth (edge → guard
name); the engine refuses anything else (`IllegalTransition`), every transition is
idempotent, appends to the append-only event log, and emits one structured log
line. Guards that matter:
- **commitment claimed before enrolling**: `SCHEDULED → ENROLLING` requires
  `commitment_root` to be set (`mark_commitment_anchored`, only while SCHEDULED,
  idempotent for the same root, never re-bindable) — enforced in the tick *and*
  as a structural backstop inside `_apply`, so enrollment can never open before
  the pre-commitment claim is stored, even via a direct call. The orchestrator also
  persists its exact inclusion block/hash/finalized head and verifies the same payload
  from archive state before enrollment;
- evidence-carrying guards: `mark_submissions_backed_up` requires a non-empty
  `backup_ref` persisted on the event; `mark_validation_complete` requires ≥1
  ACCEPTED real contender (the baseline alone cannot carry a competition) and no
  pending validation; `mark_evaluation_complete` requires every batch terminal;
  `mark_scores_persisted` requires a score row for every (accepted real
  contender × item) and commits the transition + review deadline + initial
  ranking in ONE transaction;
- **completion gates**: `AWAITING_END_TIME → COMPLETED` fires at
  `max(end_time, human_review_deadline)` (the review window is never truncated)
  and, with `require_audit_linkage=True` (the default; `False` is tests/dev only
  and warn-logged on every bypass), requires **audit linkage** (every
  `performance_history` row — baseline calibration rows included — carries its
  `audit_bundle_digest`) **and** a complete baseline calibration matrix
  (`count_missing_calibration_rows == 0` — a baseline with *zero* rows would
  otherwise bypass a gate that only sees rows that exist). It also revalidates exact
  manifest item count/order/digests/factors. For upscaling, every sealed
  `REFERENCE_ORIGINAL` must be published to the released/keyless prefix before the
  tick may complete; release failure halts in AWAITING_END_TIME;
- **TOCTOU re-checks in-transaction**: the batch-terminal, score-matrix, and
  completion-gate checks all run twice — a cheap pre-check outside the write
  lock, and again INSIDE the transition's `BEGIN IMMEDIATE` transaction, so a row
  landing between check and commit still blocks (and rolls back) the transition.

The **single-running-competition invariant** is enforced twice: engine pre-check
and, authoritatively, a partial UNIQUE index over the running statuses in SQL.

**Packet-bound score persistence** (`repository.record_item_score`) — the ONLY
score source is the scorer's verbatim `ItemScore` JSON bytes: `item_score` is the
packet's top-level `score`, `valid` is its `gate_passed`, and
`score_packet_digest` is sha256 over the exact bytes. Callers cannot supply
score/valid/digest out of band — there are no free supply parameters.
`ScorePacketPayload` (local shape; deliberately not imported from scoring)
rejects non-finite/out-of-range scores at parse (`ScorePacketError`), enforces the
packet's own gates-first invariant, and the packet's identity (miner_hotkey vs the
contender, challenge_id + item_id vs the evaluation item's stored pair) must match
what is being recorded. `set_audit_bundle_digest` is format-checked and
**write-once** (idempotent re-link of the same value only) — in Python and via a
SQL trigger.

The production boundary is stricter than the repository's final backstop.
`HttpScoringClient` independently validates the scorer health runtime preimage and
digest and exact-matches identity, digest, complete attestation, and backend map
to a contract derived locally from the same pinned release image. It then accepts
a self-hashed packet only when its
track/challenge/item/miner/content/scorer fields equal the exact request and its
complete `backend_versions` map equals health `payout_backends` plus the derived
runtime stamp. `_persist_scored_item` rechecks the locally knowable identity,
output digest/size, scoring-config digest, and health backend map **before** writing
a score packet, audit bundle, or economic row. These are systemic INFRA halts,
never contender zeroes; later CPU audit is defense in depth, not the first place a
moved/CUDA/replayed scorer response is discovered. Orchestrator-minted empty-output
zeros retain their reserved derived identity and empty backend map, and pass the
same local request/config/output binding without impersonating a measured worker.

**Track-bound item persistence** (`repository.add_evaluation_item`) — compression
normalizes `reference=input`. Upscaling requires distinct pristine and miner-input
digests plus a manifest-allowed factor, re-hashes the canonical item preimage, and
requires the exact commitment at `item_index`. The same full-matrix validation runs at
scoring, completion, and epoch-evidence construction so direct SQL cannot substitute
metadata. Input/reference bytes are single-use across competitions, including
cross-kind reuse; repository checks plus SQL INSERT/UPDATE triggers prevent a released
holdout from becoming a later hidden input or reference.

**Hash-chained human reviews** (`review.py` + `repository.py`) — reviews
(DISQUALIFY / REINSTATE / TIE_BREAK) are append-only rows chained per competition:
`integrity_hash = sha256(prev_hash || canonical_row_json)` with genesis
`sha256("vidaio-review-chain:<competition_id>")`; corrections happen via a
superseding row, never mutation; `verify_review_chain` recomputes end to end.
Reviews are allowed only in AWAITING_END_TIME at or before
`human_review_deadline`; reviewing the calibration contender is rejected.
`recalculate_ranks` re-ranks strictly from persisted rows (media is never re-run):
gate-failed/missing items enter every aggregate as zero (never vanish), a
contender with no rows is ranked last (never dropped), item lengths come only from
`evaluation_items`, both quality aggregates (length-weighted mean and
worst-decile) are always computed and stored with the manifest's
`use_worst_decile` flag choosing which one `final_score` ranks on, and **cost
efficiency is anchored per item**: each item's anchor is the cheapest valid
positive cost recorded on THAT item across all contenders (calibration included —
the baseline legitimately anchors the cost bar while earning nothing);
`min(1, anchor/own_cost)` per item, zeros for failures, so a cheap item can never
suppress efficiency earned on an expensive one.

`final_rank`, `manual_disqualified`, `eligible`, tie-break reviews, and the aggregate
above remain useful for operator/registry views. They are deliberately **not** the
economic result. `epoch_evidence.py` selects the completed competition's registered,
machine-built contenders plus exactly one baseline, requires the exact subject-by-item
packet/bundle matrix, and constructs schema-v15 `CompetitionInput`. `economic_result.py`
then derives each subject's arithmetic packet-score mean and stable
`(-score, hotkey, uid)` ordering. Mutating review/ranking columns cannot change payout.
Every `BUILT` contender must resolve in the close-block census; otherwise evidence
construction aborts the epoch rather than silently shrinking the ranked field.

The authority places those exact packets and bundles in every earning epoch log. The
CPU-only auditor recomputes the matrix and re-derives the baseline-relative result,
predecessor-folded reward window, and final weights. Schema v15 additionally commits the
global cycle ordinal, epoch-close application time, archived baseline provenance, each
contender's exact sealed source/git/image identity, and total evaluation matrix. Missing,
malformed, or inconsistent evidence fails closed. A CROWN result is publishable only
after the winning source archive becomes publicly readable.

## Design & decisions

- **Auditability-first schema** (`migrations/0001_schema.sql`): every logically
  required relation is a real FK, all `ON DELETE RESTRICT` (competition data is
  audit data). Child tables reference contenders/items/batches/sandboxes through
  **composite `(competition_id, id)` FKs**, so a row can never point at an entity
  of a different competition — cross-competition score/review tampering is a schema
  error, not a code-review hope (a review in competition B can never silence one
  in competition A).
- **Baseline non-earning by construction** (decision #1): calibration rows have
  `hotkey NULL` + `is_calibration=1` (CHECK pairs them), at most one per
  competition (partial UNIQUE), and `CHECK (is_calibration = 0 OR final_rank IS
  NULL)` — a calibration row can never carry a rank, so it can never reach
  podium/payout; `ranking()`/`podium()` exclude it by construction and only its
  score crosses to tokenomics.
- **No wall-clock, no hidden state**: every engine/repository entry point takes a
  timezone-aware `now`; events and statuses are the recovery source of truth.
- review-review-driven hardening that is structural: the anchored-before-enrolling
  backstop (#4), evidence-carrying guards and DB-verified marks (#11a–c), the
  never-truncated review window (#11d), in-transaction TOCTOU re-checks (#11),
  packet-bound persistence (#1), per-item cost anchors and never-dropped
  contenders (#8), per-(contender,item) audit linkage with the baseline-zero-rows
  gate (#12b).

## Public API (`vidaio/competition/__init__.py`)

Manifest
- `CompetitionManifest`, `ArchivedBaseline`, `ScoringFactors`,
  `EvaluationBatchSizeBounds`, `ManifestBoundsError`, `validate_against_config`.

Lifecycle
- `Phase`, `TRANSITIONS`, `RUNNING_PHASES`, `TERMINAL_PHASES`, `is_allowed`.
- `LifecycleEngine` (`create_competition`, `tick`, `mark_commitment_anchored`,
  `mark_submissions_backed_up`, `mark_validation_complete`,
  `mark_builds_complete`, `mark_evaluation_complete`, `mark_scores_persisted`,
  `audit_linkage_gaps`, `fail`, `cancel`), `IllegalTransition`.
- `CompetitionConfig` — tick cadence, review window, `require_audit_linkage`,
  manifest validation bounds.

Persistence
- `migrate(conn)`, `MIGRATIONS_DIR`, `CompetitionRecord`, `ContenderRecord`.
- `EnrollmentError` (phase/deadline/stake-gated `enroll_contender` lives in
  `repository`), `ScorePacketError`, `ScorePacketPayload`, `verify_review_chain`.
- `epoch_evidence.build_competition_epoch_evidence`, `CompetitionEpochEvidence`,
  `CompetitionEvidenceError` — completed DB state to complete schema-v15 audit/economic
  input, requiring the complete root/payload-bound finalized pre-enrollment receipt and
  explicitly ignoring human review fields.
- `anchor_evidence.verify_competition_anchor_on_chain` — authority/auditor independent
  exact-block raw commitment, block-hash, finality, and pre-enrollment chronology proof.
- `economic_result.derive_competition_result`, `derive_competition_economics`,
  `CompetitionEconomicResultError` — pure exact-coverage packet-mean derivation.

Review
- `submit_review`, `recalculate_ranks`, `ReviewError`, `ReviewWindowClosed`.

Execution seams (`interfaces.py`)
- `SandboxRunner`, `CompetitionScoringClient` (Protocols), `ContenderSpec`,
  `BatchItem`, `BatchOutput`, `IsolationProbeReport`, `ScorePacket`.

(The full repository surface — `enroll_contender`, `insert_calibration_contender`,
`add_evaluation_item`, `record_item_score`, `set_audit_bundle_digest`,
`ranking`/`podium`, event log — is imported as `vidaio.competition.repository`.)

## Data & invariants

`migrations/0001_schema.sql` creates tables `competitions`, `contenders`,
`evaluation_items`, `sandboxes`, `batches`, `performance_history`, `events`,
`human_reviews`; `0002_upscaling_item_bindings.sql` adds reference/factor/item
bindings and cross-competition media-single-use triggers. The invariants a maintainer must not break (all tested in
`tests/competition/test_schema_integrity.py` / `test_migrations.py`):

- one running competition (partial UNIQUE on `running_guard`);
- composite same-competition FKs everywhere;
- calibration CHECKs (hotkey-null pairing, one per competition, no rank);
- gates-first in SQL: `CHECK (valid = 1 OR item_score = 0)`;
- `score_packet_digest` NOT NULL with a `typeof`+length+GLOB hex check (a BLOB
  with a NUL would pass length+GLOB alone);
- `audit_bundle_digest` format CHECK + write-once trigger;
- append-only `events` and `human_reviews` (triggers);
- `UNIQUE (contender_id, item_id)` on performance rows,
  `UNIQUE (competition_id, hotkey)`, `UNIQUE (contender_id, batch_index)`.

## How to test

```
.venv/bin/pytest tests/competition tests/orchestrator
```

Notable tests — competition core: `test_transitions.py` (transition table matches
the spec diagram; every illegal edge raises), `test_commitment.py`
(anchored-before-enrolling incl. the direct-`_apply` backstop; anchor idempotent
but never re-bindable), `test_guards.py` (evidence-carrying guards, single
transaction, TOCTOU probes for late-inserted items/batches, review window never
truncated), `test_audit_linkage_gate.py` (linkage gaps block completion; a baseline
with no rows stalls completion; digest format/write-once in Python **and** SQL;
TOCTOU deferral), `test_score_packets.py` (score/valid/digest derived from bytes;
no free supply parameters; tampered identity/hotkey/item rejected; non-finite
rejected), `test_ranking.py` (zeros drag, per-item cost anchors, worst-decile
flag, no-rows contenders ranked last), `test_review.py` (chain verifies and
detects tampering; calibration unrankable at schema level; supersedes flow),
`test_invariant.py` (partial-unique backstop), `test_schema_integrity.py`.

Orchestrator/runners: see the sections below — `test_anchor_claim.py`,
`test_fault_classification.py`, `test_sandbox_safety.py`, `test_safeio.py`,
`test_zero_packet_identity.py`, `test_scorer_identity.py`,
`test_resumability.py`, `test_submission_backup.py`, `test_control_api.py`,
`test_phase_driver.py`, `test_e2e_docker.py` (needs a local Docker daemon).

## How to change safely

- **Schema changes are new migration files**; the composite-FK pattern, CHECKs
  and triggers above are load-bearing for the audit story — never weaken them for
  convenience. `manifest_digest()` inputs are frozen the moment a manifest is
  anchored: schema additions to the manifest change digests of *new* manifests
  only, and must default sanely.
- New phases/edges go through `states.TRANSITIONS` (and the spec diagram test);
  new guards should be evidence-carrying and re-checked in-transaction if their
  inputs can move concurrently.
- Score persistence must stay packet-bound: never add a parameter that lets a
  caller supply a score, validity flag, or digest directly.
- The review chain hashes `_review_content_json` — changing its field set breaks
  verification of existing chains.
- Aggregation changes (cost anchoring, zero semantics) change ranking outcomes:
  they are spec changes; update `test_ranking.py` deliberately.

---

## orchestrator/ — the bulletproof driver service

`Orchestrator` (a `BaseService`) drives `LifecycleEngine.tick` on an interval plus
the phase work between ticks: submission backup, validation intake, sandbox builds
+ isolation probes, batched evaluation, scoring + audit-bundle linkage. It
composes over the engine/repository/review modules — it never modifies them.
Every stage is **idempotent re-entry**: persisted work is read from the DB
(contender status, batch rows, `batch_outputs` events, performance rows), so a
crash loses at most the operation in flight (`tests/orchestrator/test_resumability.py`).
Batch membership is deterministic and derived, not stored (items ordered by
`item_index`, sliced by the manifest's batch size). Modal is intentionally stricter:
an image digest is evidence, not a reusable SDK handle. Each runner construction has
a unique session fence persisted before its first build. Each built image also gets an
append-only binding from pinned source/digest to the exact immutable provider Image id
created by this competition. A replacement fresh runtime may rehydrate only that id and
must reprobe every `BUILT` image; any ownership/digest/probe uncertainty halts before it
executes a batch. If the crash occurred in
EVALUATING, an append-only `modal_evaluation_reset` event atomically returns every
batch to PENDING and fences readers above all prior output/requeue events, so the full
matrix is rerun by the replacement runtime instead of mixing runtimes. Old events stay
inspectable. No inventory/name lookup or attachment to an old App/Sandbox occurs; the
only provider-id operation rehydrates the exact competition-owned immutable Image
(`tests/orchestrator/test_modal_restart_recovery.py`). Scores enter the DB only as verbatim scorer packets, and every
persisted score is audit-linked (`build_bundle → set_audit_bundle_digest`) in the
same transaction — the engine's completion gate can never see a half-recorded
score. All boundaries are bounded (`OrchestratorConfig` timeouts + finite retry
budgets); exhaustion **halts** the pipeline (CRITICAL log + `orchestrator_halted`
event, cleared through authenticated `POST /competitions/{id}/halt/clear` with an
operator and reason recorded in the append-only event log) — an infra blocker never
marks a competition FAILED.

**Control API** (`control.py`, token-authed, started only when a control token is
configured; the serve task is health-monitored — if it dies the service flips
unhealthy and stops rather than running with no control plane):
`POST /competitions` (create), `POST /competitions/{id}/items` (safe item ingest),
`POST /competitions/{id}/contenders` (enroll),
`POST /competitions/{id}/anchor` and `/anchor/release`,
`POST /competitions/{id}/halt/clear` (audited operator recovery),
`GET /competitions/{id}` (status incl. halted/podium),
`POST /competitions/{id}/review`, `GET /competitions/{id}/result` (the auditable
packet-derived economic payload). Status `podium`/`final_rank` is explicitly labeled
operational human-review ranking and is non-earning. This is the single seam both
chainless report mode and the real chain drive (decision #8) — the manifest digest and seed commitment are read from
the *persisted* manifest, never caller-substituted.

Operator sequence is strict:

1. Put each media file in the dedicated 0700 `<work_dir>/ingest` directory. The API
   accepts a basename only, never an arbitrary or remote host path.
2. `POST /competitions`, then one authenticated `POST /competitions/{id}/items` per
   manifest index. Compression sends `input_name`; upscaling sends `input_name`, a
   distinct `reference_name`, `upscale_factor`, `target_width`, and `target_height`.
   Files must be bounded regular
   files; `lstat` + `O_NOFOLLOW` + inode checks close symlink/swap races.
3. `POST /competitions/{id}/anchor` with the canonical active
   `reward_param_digest`. When the manifest declares a baseline, omit
   `baseline_image_digest`: the orchestrator builds it in this same fresh runtime,
   anchors the learned digest, and returns it. Supplying one is only a strict match
   assertion. The endpoint refuses an empty/incomplete or commitment-mismatched
   item matrix before any chain write.
4. After the anchored start time, enroll contenders and let the phase driver run.

Item ingestion is SCHEDULED/pre-anchor only. Runner-facing `BatchItem` contains the
low-resolution input digest plus its manifest-committed factor and exact output geometry.
For upscaling, the content-bound batch Image contains the digest-named media file plus
hidden `.vidaio-next-upscale-task-<digest>` canonical JSON, for example
`{"target_height":1080,"target_width":1920,"upscale_factor":2}\n`; mixed 2x/4x
batches remain supported. Modal exposes a private writable overlay inside the
fresh Sandbox; a second fresh CPU Sandbox must prove those writes did not change the
base Image. It never contains the pristine reference digest, path, or bytes.
Compression items have no task sidecar.

**Anchor claim protocol** (`anchor_competition`; an internal review — the chain
write used to precede the guarded DB transition, so two concurrent requests could
both anchor):
1. `BEGIN IMMEDIATE`: verify SCHEDULED, no root, no open claim; record the claim
   with the exact payload digest; COMMIT — a second concurrent request is refused
   (409, machine-readable `code`, guaranteed nothing was written to the chain by
   that request) *before* it can touch the chain.
2. Submit once through the injected ChainAdapter, then independently poll the raw
   v1 payload through finality and exact inclusion-block archive read-back. A lost
   response is recovered by reads; the payload is never blindly resubmitted.
3. Atomically record the root + complete `commitment_anchored_onchain` receipt
   (block, block hash, finalized head, archive proof), resolving the claim.

A crash between 2 and 3 leaves an open claim that is **ambiguous** (the extrinsic
may or may not have landed). While fresh, every request is refused; once stale
(`anchor_claim_stale_seconds`) only the *identical* payload may be checked again in
READ-ONLY recovery mode; it is never resubmitted. A different/new write stays
refused until an operator independently proves nothing landed and calls
`release_anchor_claim` — deliberately manual, since auto-release would restore the
double-anchor hazard. (`tests/orchestrator/test_anchor_claim.py`.)

**Fault classification** (`failures.py` — the single place that answers "whose
fault was that?"; every stage routes through `classify_failure`, none hard-codes a
verdict):
- `Fault.CONTENDER` — attributable to the submission: non-zero exit, per-contender
  timeout, unsafe/oversize output, no output, a Dockerfile that does not build, an
  isolation probe that RAN and failed, or the trusted scorer rejecting the
  contender's own output bytes. Outcome: that contender's batch/items are
  zero-scored with a reason code; the competition continues.
- `Fault.INFRA` — attributable to us: docker down, image/sealed-input missing,
  isolation contract not holding on a container *we* launched, a probe that could
  not be run, checkout failures, DB errors, scorer 5xx/transport, scorer-identity
  disagreement, and any scorer rejection naming OUR half of the request. Outcome:
  bounded requeue/retry, then halt — never a FAILED competition, never a
  substituted score.
- **Default is INFRA**: zeroing a contender for our own bug would silently corrupt
  the result; halting is loud and recoverable. A scorer 422 is contender fault
  only when the worker's typed error names the *output* field
  (`_CONTENDER_STATUSES` is an allowlist of {400, 415}, not a 4xx catch-all).
  A missing/empty output is never sent to the scorer: it is zero-scored locally.

**Zero-packet reserved identity** (`zero_packets.py`) — the orchestrator mints a
packet itself in exactly one case: an item with no measurable bytes (no output, or
an output the worker rejected as its own). Such packets carry
`scorer_version = "orchestrator-zero/1+<digest12>"` — the same `<name>+<digest12>`
shape as a worker identity, under a **reserved** name whose digest covers
{convention, scoring-config digest, the manifest's committed worker identity,
track}, so it is recomputable from the anchored manifest and records which worker
the competition committed to *without claiming to be it*. The packet asserts no
measurement (`gate_passed=False` forces score 0.0 structurally, reason code in the
violation, canonical empty digest as content). Impersonation is refused in both
directions: the orchestrator never stamps a worker identity on its own packets,
and a manifest or live worker claiming `orchestrator-zero/*` is a
`ReservedScorerIdentity` → INFRA halt. Its audit bundle is built with the same
identity so packet and bundle agree under
[audit recompute](../audit/README.md). Relatedly, `scoring_version` in the
manifest is the full effective scorer identity: `_check_scorer_identity` compares
the live worker's advertised identity to the persisted manifest at competition
start and again before SCORING; disagreement is an INFRA halt
(`tests/orchestrator/test_scorer_identity.py`).

**Submission archive invariant**: every contender that can
still win has an archived submission — a contender-fault tree (symlinks,
oversize) is rejected before it can advance; an infra failure halts finalization
instead of certifying a partial archive set; per-contender evidence lives in the
append-only `contender_submission_archived` events
(`tests/orchestrator/test_submission_backup.py`).

**Legacy operational results adapter** (`results.py`) — a pure read converting a COMPLETED
competition's repository ranking into a control/registry `CompetitionResult`: `cycle` is the 1-based
ordinal of the append-only terminal COMPLETED transition (`events.event_id` order), so
creation/start order can never outrank a result that actually completed later; `completed_at` comes from the event
log; `contenders` from `ranking()` (baseline can never leak in — schema CHECK);
`margin` is the business-native paired improvement computed per item against the
calibration baseline's persisted compression rates (length-weighted; a failed item
contributes 0.0 — never dropped; `None` when no baseline or, on upscaling, where
no paired business metric is persisted — the honest answer, not a composite score
in disguise); `baseline_score` is the calibration row's final_score;
the baseline score is diagnostic only in this adapter (the serving-baseline registry is
the authoritative executable source); `uid` resolves through a caller-supplied hotkey→uid snapshot,
`UNKNOWN_UID = -1` for unregistered hotkeys. This helper is not served as `/result`
and is not the earning source;
schema-v15 economics uses `epoch_evidence.py` + `economic_result.py` and exact packet
means, so review-driven repository ranking cannot affect emissions.

`Orchestrator.build_result` and control `GET /result` use the auditable packet-economic
path directly: every packet/bundle must resolve, all BUILT contenders must be present in
the current census, and no `UNKNOWN_UID` fallback exists. The endpoint labels this a
`current_unpinned_chain_head` **preview**, not the authoritative already-emitted result;
only finalized schema-v15 epoch evidence has the exact close-block census. Missing
census/evidence returns a fail-closed `unauditable_result`, not a partial or human-ranked
payout object. The control handler performs its synchronous chain census RPC off the
event loop, then reads the thread-bound SQLite connection on the owning thread.

Public API: see `orchestrator/__init__.py` — `Orchestrator`,
`OrchestratorConfig`, `build_docker_runner`, `HttpScoringClient` /
`ScoringClientError`, `AnchorClaimRefused` / `AnchorError` / `AnchorResult`,
`EMPTY_SHA256`, `Fault` / `classify_failure` / `fault_code`,
`ORCHESTRATOR_ZERO_SCORER_NAME` / `ReservedScorerIdentity` /
`is_orchestrator_zero_identity` / `orchestrator_zero_identity`,
`build_competition_result` / `result_payload` / `ResultNotReady` / `UNKNOWN_UID`.
The economic/evidence APIs above are imported from their explicit modules rather than
the package root.

---

## runners/ — sandbox implementations of `SandboxRunner`

**`DockerSandboxRunner`** [DONE] — the local-first real implementation
(decision #7). Isolation contract on every solution run: `--network none`,
`--read-only` rootfs (only a bounded `/tmp` tmpfs and `/output` writable),
`--cap-drop ALL`, `--security-opt no-new-privileges`, pids/memory/cpu limits,
inputs mounted read-only at `/evaluation-inputs` as a **per-batch staging subdir**
(never the whole sealed pool, never `index.json`), no env injected beyond the
image's own. Solution image contract: `/bin/sh /app/run.sh <input_dir>
<output_dir>`, one output per input under the same digest-named filename, plain
regular files only, exit 0. Build binding: `image_digest` is
`sha256(canonical_json({scheme, repo_url, commit_sha, tree_sha}))` under the
versioned `vidaio.competition.logical-build.v1` domain, with a digest-derived
local tag so builds are resumable across restarts. It identifies the pinned
source claim; qualification supplies the same public repository locator while
building its verified local checkout. Provider object ids are separate execution
bindings.

- **Host-inspect-authoritative isolation**: the probe's verdict used
  to come from commands executed *by the image* — an untrusted image can fake all
  of that. Now only `docker inspect` of the actually-launched container can make
  a probe PASS (network mode, mounts + RW flags + sources, tmpfs, ReadonlyRootfs,
  Privileged/Cap*/SecurityOpt, effective env); the in-container script is
  advisory-**negative**-only (it can turn a True into a False, never the
  reverse), and the same host verification runs on every *evaluation* container
  too — a run whose flags did not take effect raises `SandboxIsolationError`
  (INFRA), it is never scored.
- **Bounded output**: `/output` is a host dir, bounded by a host-side
  watchdog polling the `docker run` child — on-disk output size (lstat, links
  included) plus captured logs vs `max_output_bytes`/`max_batch_output_bytes`;
  crossing the cap force-removes the container → `OversizeOutputError`
  (CONTENDER fault). Fast writers that exit between polls are still caught: the
  watchdog measures on the same iteration it observes the exit, and a final
  post-run check covers both /output and the logs. A tmpfs at /output was
  rejected because extraction would require the untrusted image's cooperation —
  "a bound that depends on the thing being bounded is not a bound".
- **Symlink-safe IO**: both the output
  dir and the submission checkout are adversary-chosen trees, and stdlib helpers
  follow symlinks (`<name> -> /proc/self/environ` would make the host archive its
  own secrets). Rules: `lstat` first and reject anything that is not a plain
  regular file; open with `O_NOFOLLOW` then `fstat` and require the same
  (st_dev, st_ino) — closing the swap-after-check race; realpath-prefix
  containment; tarballs built only from explicit `TarInfo` records over regular
  files (never a directory walk, so no symlink/hardlink/device members). Unsafe
  trees are **rejected, not filtered** — a backup that quietly drops files is not
  evidence of the submitted tree.
- Every failure from `run_batch` is typed by fault class (`errors.py`):
  `ContenderFaultError` subtypes (`SolutionExitError`, `BatchTimeout`,
  `OversizeOutputError`, `UnsafePathError`, `ContenderBuildError`, …) vs INFRA
  (`BuildError`, `SandboxIsolationError`, `SandboxProbeUnavailableError`,
  `RunnerUnavailableError`, `UnknownImageError`, `InputStagingError`, …).

**`ModalSandboxRunner`** [IMPLEMENTED, live smoke pending] — create-only GPU
execution behind an injected runtime, so unit tests never import Modal or contact
the cloud. `ModalSdkRuntime.start_fresh` requires the exact creation confirmation,
accepts only unique `vidaio-next-*` Environment/App/run names, and contains no
inventory/name lookup or Sandbox/instance reuse path. Restart may call `Image.from_id`
only with an exact append-only competition-owned binding. Every batch gets a fresh network-blocked,
secret/env/OIDC/port-free GPU Sandbox with a fresh content-bound batch-only input Image;
Modal's sandbox-local writable overlay is explicitly reported, and a second fresh
CPU Sandbox remount proves the writer could not mutate the Image base;
the untrusted Sandbox is terminated before a separate fresh CPU-only collector
copies the frozen output snapshot through the existing safeio/size/log caps. The
ordinary trusted CPU scoring/audit path consumes those bytes — Modal never creates
a GPU-only score. Requested isolation is host-attested plus advisory-negative probe
because Modal does not expose Docker-style full applied-config readback. Detailed
operator contract: [`deploy/modal/COMPETITION.md`](../../deploy/modal/COMPETITION.md).
The runner also exposes only a local live-handle check plus a random session identity;
the orchestrator's persisted restart fence forces exact owned-image restore/reprobe and
full-matrix reset as described above, never cloud inventory discovery.
Modal's fresh `im-*` object id is deliberately excluded from `image_digest`: it
changes on every fresh build. The exact id is instead stored with the logical
digest, pinned source, runtime session, and resource label in the typed append-only
`modal_image_bindings` ledger plus the chronological `modal_image_bound` event.

Repo access: `repo.py` provides `RepoProvider` — `LocalRepoProvider` (tests) and
`GitRepoProvider` (read-only clone at a pinned commit).

Runner tests: `tests/orchestrator/test_sandbox_safety.py` (symlinked output
rejected and the host file never archived; output floods killed by the host
watchdog, including fast log floods that exit between polls; probe verdict comes
from the host, not the image; batch runs host-verified too; `exit 1` a typed
contender fault) and `tests/orchestrator/test_safeio.py`;
`tests/orchestrator/test_e2e_docker.py` exercises the real Docker path end to end;
`tests/orchestrator/test_modal_runner.py` proves create-only ownership, fresh-per-batch
GPU execution, CPU collection, rollover/termination, isolation requests, and caps with
an injected runtime and zero cloud calls; `test_modal_restart_recovery.py` proves a
runner-A/runner-B restart cannot use a lost handle, mix effective batch outputs, accept
digest drift, or proceed after a failed reprobe.

---

## Status & gaps

- [DONE] Manifest, state machine + guards, schema, packet-bound persistence,
  review chain + re-ranking, orchestrator (control API, anchor claim, fault
  classification, zero packets, resumability), DockerSandboxRunner, safeio,
  HttpScoringClient.
- [DONE, needs authorized live smoke] `ModalSandboxRunner` + `ModalSdkRuntime` implement
  fresh create-only GPU contender execution and CPU-only collection. No Modal resource
  has been created or reused by the repository tests.
- [DONE] Schema-v14 economic bridge: complete exact packet/bundle matrix and sealed
  source provenance, machine packet-mean ordering that ignores human review, live
  result/reward-window emission inputs, and
  CPU auditor re-derivation.
- [DONE] Both competition tracks. Upscaling binds distinct pristine/input bytes,
  factor, and exact target geometry inside the anchored manifest, exposes only the
  low-resolution input plus its task-contract sidecar to GPU contenders, scores with
  CPU PieAPP, releases the pristine reference before
  completion, and strictly recomputes through a public keyless store.
- [DONE] The active archived executable is rerun as the non-earning baseline on every
  hidden matrix. Version zero is explicitly seeded per track; verified CROWN promotion
  appends the next serving version (see [vidaio/registry](../registry/README.md)).
- [KNOWN LIMIT] The legacy operational paired-business-margin view has no upscaling
  margin and honestly returns `None`; earning order does not use it and is derived
  from committed packet-score means for both tracks.
- [PENDING DECISION]-adjacent: container image size on Modal is unattested (spec
  §04 "acknowledged-but-unmeasured") until the trusted-builder proof exists.
- [DONE] Reward application is a one-time global cycle fold at the epoch-close block
  timestamp; exact packet/result/window/payout recomputation is mandatory. Human review
  and repository rank never cross the economic seam.
- `sandboxes.expires_at` / warm-rollover bookkeeping exists in the schema; the
  local Docker runner does not need it (no lifetime cap) — it becomes live with
  the Modal implementation.
