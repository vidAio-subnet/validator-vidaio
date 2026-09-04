# vidaio.weightsetter — compose, submit, and PUBLISH the exact weight vector

The weight-setter loop (`service.py::WeightSetter`): compose the weight
vector, submit it through the `ChainAdapter`
([`vidaio/chain/README.md`](../chain/README.md)) on a tempo-gated cadence,
and publish the EXACT submitted vector through the audit store + commitment
ledger so third parties can reproduce the chain weights. Runs as its own
supervised process.

The vector's SOURCE is config-selected (`provider`, wave 5):

- **`shared`** (PRODUCTION — the convergence path,
  the project design record rules 9-11): the
  `SharedSnapshotProvider` fetches the Scoring Authority's per-epoch pointer,
  mirrors the immutable [`EpochLog`](../epoch/README.md) bytes, verifies the
  three-way tamper-evidence chain, validates a narrow chain-safe submission view,
  and hands the loop the epoch log's exact authority-published `weight_u16` plus
  its close-block uid/hotkey census and owner-sink identity. The strict
  `MinerSnapshot` / competition / reward-window model remains diagnostic input for auditors,
  not a submit gate. Two validators reading the SAME finalized epoch
  pass that authenticated sum-grid unchanged to the pinned SDK; Bittensor 10.5 applies the same
  max-normalization to both, so their emitted runtime vectors are byte-identical and Yuma sees
  one agreed vector, including live testnet
  competition reward-window emissions and any conditional sink residual.
- **`local`** (report-mode / dryrun / third-party recompute): the existing
  per-validator `miner_manager` fold — kept unchanged under rule 8; both modes drive
  identical service code, only the provider swaps.

This process is deliberately only the thin weight path. It constructs no scorer,
recomputer, audit reviewer, or Audit Results client. The same validator identity runs
the beacon auditor and full own-auditor as separate OS containers/cgroups, each with its
own durable cursor and pending-report outbox; those workers can report and alert without
sharing a failure domain with `set_weights`.

## What it does

- **`service.py`** — [DONE] the loop (`attempt_once` every
  `attempt_interval_seconds`, default 72 min), the tri-state chain
  confirmation (`ChainConfirmation`), the publication path, and the
  recovery pass (`reconcile()`, at startup AND before every attempt).
- **`intents.py`** — [DONE] the weight-submission intent ledger (durability
  across the non-idempotent `set_weights`), plus the u16 vector-matching
  helpers (`quantize_weights`, `weights_match`, `vector_fingerprint`) and
  the publication watermark.
- **reward-state persistence** — [DONE, local/report compatibility] additive schema-v14
  SQLite persistence for the singleton `RewardWindowState` and immutable per-cycle
  `CompetitionResult` rows (idempotent; conflicting cycle or competition-id re-ingest
  raises `ResultConflictError`). Economics use only chain-bound `applied_at`. Legacy v1
  tables remain inert/readable and are never used to derive the v2 state. Production
  shared convergence deliberately does not trust validator-local economic state: the
  exact vector comes from the anchored schema-v15 epoch log and is independently audited.
- **`shared_snapshot.py`** — [DONE] `SharedSnapshotProvider` (the convergence
  source: fetch pointer → mirror bytes → 3-way verify → hand over inputs), the
  `ScoringAuthorityClient` / `EpochLogStore` / `EpochAnchorReader` seams (+ the
  `Http` / `ChainAdapterAnchorReader` implementations), `EpochInputs`, the typed
  `SnapshotUnavailable` / `SnapshotDigestMismatch` failures, and
  `make_snapshot_provider` (the `shared` | `local` selector).
- **`own_audit.py`** — [LEGACY / TEST-ONLY] `OwnAuditGate` / `OwnAuditVerdict`, retained
  for focused regression and backward-compatibility tests. Production `WeightSetter`
  neither imports nor constructs it; live own-audit uses the standalone auditor spine.
- **`config.py`** — [DONE] `WeightSetterConfig` (the `weightsetter:`
  section).

## Design & decisions

### One attempt (`attempt_once`)

1. `reconcile()` first — an owed publication must not sit behind a chain
   that happens to be tempo-gated for hours.
2. Chain-state gate: stale/unavailable snapshot ⇒ SKIP with a structured
   reason (`weightsetter_chain_state_skips_total`); an empty or partial
   vector is never submitted because a chain read failed.
3. Resolve the vector. A local provider composes with `tokenomics.build_weight_vector` over
   `SnapshotProvider.miner_snapshots()` (the validator's miner manager in
   report/dryrun wiring) plus the canonical `burn_uid`, persisted
   `reward_window_state`, and chain-bound composition time. The production shared
   provider instead uses
   the authenticated epoch log's exact `weight_u16`; it never substitutes a validator-local
   derivation or persistence state. Re-derivation is performed by the standalone
   auditors and remains an investigation signal only.
4. Freeze packet evidence. In shared production mode this is best-effort: an
   unreadable/malformed set stays durable as unresolved `null`, the exact authority
   vector submits, and post-submit reconciliation retries from its frozen
   `(epoch_id, snapshot_digest)`. It is never replaced with the false empty sentinel.
   Local/report composition still skips when its unfrozen evidence source is unreadable.
5. Write the INTENT row (vector, digest, packet digests, attempt block,
   `packets_frozen_at`) BEFORE the first `set_weights` call.
6. Submit (`_submit`): timeout-guarded, bounded retries — but never a blind
   re-write (below).
7. On acceptance: `mark_accepted` → `_publish_intent` (store the canonical
   `WEIGHT_VECTOR` document, ledger a `PublicationRecord`, anchor it). The process then
   returns to its chain cadence; no audit hook runs before or after submission.

In the `shared` provider the vector step (3) reads the epoch log's stated u16 vector.
Digest/anchor authenticity remains a hard gate. Economic re-derivation and media
recomputation belong to the separate auditor processes described below.

### The shared convergence source (`SharedSnapshotProvider`)

Local per-validator EWMA folds never converge, so production reads a SHARED
authoritative source instead. Every `miner_snapshots()` / `epoch_inputs()` call
RE-RESOLVES the latest finalized epoch: fetch the Scoring Authority pointer
(`GET /epoch/latest`), mirror the epoch-log bytes from the object store by
`snapshot_key` (behind the `_FINALIZED` half-write guard), and verify the
**three-way tamper-evidence chain**:

    sha256(mirrored bytes) == pointer.snapshot_digest == on-chain anchored digest

The third leg is MANDATORY: with `verify_anchor` on, an anchor READER must be
wired or the provider HOLDS — verifying only bytes==API==the pointer's own
anchor field trusts three values all supplied by the same (possibly untrusted)
authority. The reader is one `ChainAdapterAnchorReader` for every mode (it wraps
the `HttpChainAdapter` sim read in report mode and `BittensorChainAdapter.read_anchor`
in production). Any inequality → `SnapshotDigestMismatch` (REFUSE, CRITICAL,
submit nothing). An unreachable authority / object store / anchor or a finalized-
but-not-yet-anchored epoch → `SnapshotUnavailable` (HOLD — the last confirmed vector
stays live on chain). After authentication, the narrow parser still requires current
schema/canonical bytes, an exact non-empty u16 sum-grid/digest, pointer epoch/close
identity, and a unique close-block hotkey binding for every positive non-owner uid.
Failure of one of those submission-safety properties refuses the write. A strict
economic/evidence model failure instead produces a CRITICAL local finding and the
standalone auditors durably outbox a signed `EPOCH_LOG_INVALID` (or
`EPOCH_LOG_UNVERIFIED`) central report; the safe authority vector continues unchanged.
The provider NEVER falls back to local sampling — a locally-improvised vector is
exactly what diverges. The authenticated stated u16 vector is submitted unchanged;
the standalone auditors independently re-derive the economics and centrally report any
finding for manual remediation.
The provider binds the pointer's `epoch_id` / `close_block` /
`weight_vector_digest` to the canonical raw log fields. The v14-introduced/current-v15 competition
result, reward window, total fold cursors, earning folds, shares, and evidence relationships are
independently verified and reported by the standalone auditors.

Production wiring uses `make_unsealed_writer_store()`, not the authority's private store
and not the auditor's anonymous read-only view. It uses signed, narrowly scoped S3/Hippius
credentials (or workload IAM) to read public finalized evidence and write only this
validator's `WEIGHT_VECTOR`/`MANIFEST` publication. The store rejects all sealed-holdout
operations before transport and never loads the AES key; bucket IAM must independently
deny canonical `reference_original/`. The beacon and own-auditor containers are not
co-bundled with this process: each constructs an unsigned client and must not receive
these credentials. Epoch-log reads are capped at 64 MiB.

### The audit isolation boundary

A self-consistent but invalid epoch log can reproduce its own stated vector. That is why
the validator identity also operates two services over the
[`vidaio.auditor`](../auditor/README.md) spine: `own-auditor` emits signed
`audit_mode=own_audit` reports and the beacon auditor emits signed
`audit_mode=beacon` reports. In production each uses CPU `all_items` coverage and performs
the media recompute, earning-state re-fold, complete competition/result/reward-window
reconstruction, and full weight re-derivation.

These are separate OS processes/containers and cgroups from `WeightSetter` and from one
another. Each owns a durable contiguous cursor and pending-report outbox. Local
DISPUTED/INCONCLUSIVE findings produce CRITICAL/WARNING operator logs before delivery;
the central API logs acceptance/conflicts and exposes per-mode metrics. Findings,
recomputer exceptions, and API delivery failures cannot change the authority vector,
mark a weight-setter chain-state skip, or return a submit HOLD. Remediation is manual.

### The dropped `set_weights` timeout bound

`set_weights` is now called WITHOUT a `with_timeout` wrapper (companion to
chain #11): the real adapter serializes the extrinsic on its socket internally,
and the inclusion/finalization wait MUST run to completion — a caller timeout
that abandons the awaiting coroutine cannot cancel the worker thread and leaves
a live submit behind (the block-subscription leak that OOMed a prior
production validator's pod). The
chain's own tempo gate keeps the bounded retry safe; a transport failure still
surfaces as `OSError`/`TimeoutError` (ambiguous) and is reconciled tri-state as
before. `chain_timeout_seconds` still bounds `anchor_commitment`; for the real
adapter it must be ≥ 180 s regardless.

### Convergence-health observation (observe-only)

When `convergence_observe_enabled` is set, after submitting the authenticated authority
vector the loop reads peer validators' on-chain vectors (explicit
`convergence_peer_hotkeys` and/or every metagraph validator when
`convergence_use_metagraph_peers`) and emits
`vidaio_weightsetter_convergence` = the fraction agreeing with us on the u16
grid. On an honest network all peer validators reading the same epoch converge
to 1.0; a dip surfaces divergence for the dashboard/operator BEFORE it costs
emissions. It NEVER changes what this validator submits — it always submits the
authority vector. The dashboard scrapes this gauge for its convergence panel.

### The intent ledger (why it exists)

`set_weights` is a NON-IDEMPOTENT chain write behind a retry envelope: if
the extrinsic lands but the response is lost, the retry is tempo-rejected —
the old shape recorded the whole attempt as FAILED even though the chain had
changed, and a crash between acceptance and publication left an accepted
vector permanently unaudited. Every attempt now writes an intent first;
every later step is driven FROM the row:

```
pending    written, chain outcome not yet known
accepted   the chain holds it (directly observed or confirmed); publication owed
published  artifacts stored, PublicationRecord ledgered and anchored
abandoned  the attempt provably did not change the chain (terminal)
```

`resolution` records HOW each state was reached (`chain_accepted`,
`chain_confirmed`, `tempo_after_ambiguous`, `chain_confirmed_on_restart`,
`chain_denied_after_crash`, …) so an inferred reconciliation is never
indistinguishable from a directly-observed one.

### The tri-state chain check (`ChainConfirmation`)

An ambiguous attempt (timeout / transport error — a request that never left
is indistinguishable from a lost response) consults the chain before any
re-submission, and the question is about THIS INTENT'S OWN VECTOR, via the
optional `SubmittedWeightsReader.submitted_weights(hotkey)` read. Block
bookkeeping ("our `last_update` advanced") only proves SOME write landed —
it is what used to publish vectors that had never landed.

- `CONFIRMED` — the chain reports OUR vector (same uid set, per-uid within
  1 step on the u16 grid), recorded at/after this attempt's block when the
  adapter dates it, and no other intent whose vector ALSO matches the chain
  report under that same tolerance could equally be its author
  (`_identical_twin_exists` — one equivalence relation everywhere, an internal review; settled intents count too; a twin with no set-block to compare
  against makes the answer a guess ⇒ UNKNOWN).
- `DENIED` — the chain POSITIVELY holds no weights for our hotkey
  (`submitted_weights -> None`), or its current vector predates this
  attempt. Nothing of ours landed since we tried.
- `UNKNOWN` — everything else: no `validator_hotkey` configured, no fresh
  snapshot, an adapter without the read surface, a failed read, a differing
  vector that postdates our attempt (ours may have landed and been
  overwritten), or the identical-twin ambiguity.

The check ALWAYS refreshes first — the pre-write snapshot answers a
question about the pre-write world.

**The publication rule** (only publish what can be shown to have landed):

| Verdict | Action |
|---|---|
| CONFIRMED | publish: store the vector, ledger, anchor |
| UNKNOWN | HOLD — intent stays pending, re-checked every pass; never abandoned at any age, never re-driven blindly |
| DENIED | never published; abandoned only via `reconcile()`'s age-bounded path (`abandon_denied_intent_after_seconds`, logged CRITICAL with evidence) — except a synchronous rejection of the ONLY write issued, which may settle on the spot. A retry's rejection describes the retry, never the write before it |

Retrying under UNKNOWN is safe only because the chain's own tempo gate
cannot accept a second write inside the same window; a tempo rejection after
an ambiguous write is reconciled as a success ONLY when the chain shows us
our vector.

### Vector matching on the chain's u16 grid (`intents.py`)

The chain MAX-normalizes submitted vectors onto u16 (largest weight →
65535 — `WEIGHT_QUANTIZATION_SCALE`), so submitted floats never come back.
`quantize_weights` drops non-positive entries, mirrors the SDK's binary32
max-normalization, rounds to u16 and drops zero-rounded entries. Raw u16s,
sum-normalized floats and the untouched submission compare equal within the
one-step tolerance below; exact bytes can move one step under positive rescaling
because the SDK casts to binary32 first.
`weights_match` requires the same uid set and per-uid difference ≤
`WEIGHT_MATCH_TOLERANCE_STEPS` (1); two EMPTY vectors never match.
`weights_match` is THE one equivalence relation: it ties a chain report to
an intent AND decides twin ambiguity (whenever the report matches more than
one candidate intent under the tolerance, everyone's verdict is UNKNOWN
unless block dating positively disambiguates).
`vector_fingerprint` (digest of the quantized vector) is an audit/log label
only; a stricter exact-digest twin test is exactly the hole that let a
one-step-away later vector confirm an earlier, unlanded intent.

### Publication (the auditable weight path)

After acceptance, driven entirely from the intent row (crash-safe to
re-run): the exact SDK-emitted runtime max-grid is serialized as canonical JSON
(`weight_vector_document`, domain `vidaio.weight_vector.v1`, carrying the
chain's accepted block) and stored as a `WEIGHT_VECTOR` artifact; a
`PublicationRecord {score_packet_merkle_root, weight_vector_digest}` is
recorded in the `CommitmentLedger` (`pending_chain`) — the commitment id is
pinned back onto the intent so a re-drive re-anchors the SAME commitment —
and anchored via `ChainAdapter.anchor_commitment` (→ `anchored`). An anchor
failure leaves the commitment `pending_chain` for the next reconcile pass.

In shared production mode, publication evidence is deliberately downstream of
`set_weights`. The intent best-effort copies packet refs from the already-authenticated
in-memory epoch log, but a copy failure is persisted as JSON `null` together with the
exact `snapshot_epoch_id` and `snapshot_digest`; the authority vector is still submitted.
After acceptance, publication re-fetches that specific historical epoch (never
`latest_pointer()`), verifies the anchored log and its packet Merkle root, then fills the
digest list exactly once. An unavailable or conflicting result leaves the accepted intent
queued and alerts; it can never become the empty-set sentinel or gate a later scheduled
weight write. The legacy local/report provider has no immutable epoch recovery key and
therefore retains its pre-submit `None` (failed) versus `[]` (genuinely empty) distinction.

Empty-packet sentinel: `audit.merkle_root` requires ≥ 1 leaf, so a
publication with no packets uses the root over the single leaf
`sha256(EMPTY_SCORE_PACKET_MARKER)`
(`b"vidaio.weightsetter.no-score-packets.v1"` →
`EMPTY_SCORE_PACKET_SET_ROOT`) — a versioned, public convention a third
party can recompute. It is a SENTINEL, not a default: with the validator's
`ScorePacketEvidence` wired as `PublicationInputs`, real publications carry
the real merkle set.

Evidence windows partition via the WATERMARK: the lower bound of the next
publication is the previous published intent's `packets_frozen_at` — when
its packet list was CAPTURED, not when its anchor finally settled
(`settled_at` is only the pre-fix fallback for old rows). Otherwise every
packet created while a slow/failed anchor was re-driven would belong to no
publication. Overlap is harmless; a gap is not. First-ever publication falls
back to `publication_lookback_seconds`.

### `reconcile()` — the recovery loop

Runs at startup and before each attempt. `accepted` intents get their
publication/anchor re-driven (`weightsetter_redriven_publications_total`).
`pending` intents are asked about with a FRESH snapshot: CONFIRMED →
promote + publish; UNKNOWN → stays pending (logged, gauged on
`weightsetter_pending_intents` — a number that does not come back down
means the chain cannot be read at all); DENIED → abandoned ONLY after
`abandon_denied_intent_after_seconds`, logged CRITICAL with the full
evidence — the single path to that terminal outcome. The whole pending
branch is deferred while the chain snapshot is unusable, and an intent
whose stored vector is unreadable stays pending for an operator.
`note_check` records every verdict on the row (`last_checked_at` /
`last_check`) so an operator can see how long a fate has been unknown.
The pending gauge is synchronized at every durable intent mutation, including
immediately before the write and immediately after an ambiguous outcome; it does
not wait for the next 72-minute reconciliation pass to expose a pending row.

## Public API & endpoints

No HTTP API — only `/health` + `/metrics` on `weightsetter.metrics_port`
(default 9102). Health checks: `last_success_age` (fails past
`max_last_success_age_seconds`) and `db` (own per-thread connection; skipped
for `:memory:`).

Python surface (`__init__.py`): `WeightSetter`, `WeightSetterConfig`,
`SnapshotProvider` / `PublicationInputs` (Protocols; the validator's miner
manager and `ScorePacketEvidence` are the real implementations), the shared
provider (`SharedSnapshotProvider`, `EpochInputs`, `ScoringAuthorityClient`,
`HttpScoringAuthorityClient`, `EpochLogStore`, `EpochAnchorReader`,
`InMemoryChainAnchorReader`, `SharedSnapshotError`, `SnapshotUnavailable`,
`SnapshotDigestMismatch`, `make_snapshot_provider`), `ChainConfirmation`,
`weight_vector_document`, `WEIGHT_VECTOR_DOMAIN`, `EMPTY_SCORE_PACKET_MARKER` /
`EMPTY_SCORE_PACKET_SET_ROOT`, the local persistence functions
(`migrate`, `load_reward_window`, `save_reward_window`, `ingest_competition_result`, `latest_result`,
`ResultConflictError` — exported but not an authority for the shared compose path) and the `intents`
module. `OwnAuditGate` / `OwnAuditVerdict` remain directly importable from the legacy
`own_audit.py` module for focused tests only; they are not exported by this package or
accepted by `WeightSetter`.

## Data & invariants

Migrations (`migrations/`, applied by `migrate` at
construction):

- `0001` — legacy-local v1 state/result tables. They stay inert/readable after the
  additive migration and are not a v2 economic source.
- `0002_weight_intents.sql` — `weight_intents` (canonical `vector_json` +
  `vector_digest`, `packet_digests_json`, state machine, `resolution`,
  `accepted_block`, `commitment_id`, `settled_at`).
- `0003_intent_watermark.sql` — `packets_frozen_at` (the publication
  watermark) and `last_checked_at`/`last_check` (the UNKNOWN audit trail).
- `0006_publication_snapshot_digest.sql` — the exact authenticated epoch-log digest
  carried into each shared publication.
- `0007_publication_snapshot_epoch.sql` — the historical epoch id needed to recover
  unresolved post-submit packet leaves without ever falling forward to a newer epoch.
- `0008_reward_window_state_v2.sql` — singleton `reward_window_state` and immutable
  `competition_results_v2`. Contenders persist as exact `{hotkey, uid, score}` rows;
  baseline version/artifact identity and chain-bound `applied_at` round-trip exactly.

Invariants:

- `abandoned` is terminal and reachable only from a POSITIVE denial; an
  intent whose fate is unknown stays `pending` indefinitely. "We could not
  find out" is not evidence of absence.
- `publication_enabled: true` (the default) REQUIRES both an audit store
  and a commitment ledger at construction (`ValueError` otherwise);
  disabling it settles intents unaudited and is dev/test only.
- V2 `competition_results_v2` rows are immutable inputs; ingest re-runs are idempotent
  by cycle and competition id; the result row and resolved reward window are written in
  ONE transaction. They do not override an anchored shared epoch.
- All timestamps come from an injected clock (tz-aware); no logic path
  reads the wall clock directly (the epoch-seconds `wall_clock` exists only
  for the chain adapter's freshness surface). An unparseable intent
  timestamp makes the intent look YOUNG, never old — age is a condition for
  the terminal abandon.
- Reward-window activity is evaluated at the epoch log's chain-bound `created_at`; local
  ingest order and database clocks cannot extend or shorten it. Successfully applied
  cycles are replay-safe through `last_applied_cycle`.
- A result whose executable-baseline score is missing/non-positive (or whose contender
  set is empty) is incomplete evidence, not PODIUM. Its fold is a retryable no-op that
  preserves the prior window and does not consume the cycle.
- Missing/deregistered podium identities do not block shared submission. Their fixed
  shares remain unallocated and route to the authenticated subnet-owner sink rather than
  being redistributed to present miners.

## Configuration

Section: `weightsetter` (schema `config.py::WeightSetterConfig`,
`extra="forbid"`). Env override pattern:
`VIDAIO__WEIGHTSETTER__<KEY>=<value>`.

| Key | Default | Meaning |
|---|---|---|
| `attempt_interval_seconds` | `4320` (72 min) | Attempt cadence (spec §01, against a ~20 min tempo gate — most attempts tempo-gate, which is metered as reschedule, not failure) |
| `chain_timeout_seconds` | `180` | Timeout around `anchor_commitment` (does NOT bound `set_weights` — the real adapter's inclusion wait must not be caller-cancelled) |
| `chain_retry_attempts` | `3` | Bounded retry envelope for chain writes |
| `chain_retry_base_delay_seconds` | `1.0` | Retry backoff base (exponential) |
| `version_key` | `15` | Explicit fleet fence synchronized with epoch-log schema v15 (report/test may override to 0) |
| `validator_hotkey` | `""` | OUR hotkey, used only to read our vector back after an ambiguous attempt. Empty ⇒ every ambiguous attempt stays UNKNOWN: never abandoned AND never published. Set it in any deployment that publishes |
| `max_chain_snapshot_age_seconds` | `3600` | Staleness gate; 0 disables |
| `abandon_denied_intent_after_seconds` | `3600` | Minimum age before a POSITIVELY DENIED intent may be abandoned |
| `publication_lookback_seconds` | `86400` | Evidence window when no publication watermark exists yet |
| `publication_enabled` | `true` | The auditable weight path; disable ONLY in dev/test without an audit store |
| `max_last_success_age_seconds` | `17280` (4 intervals) | Health: degraded past this since the last accepted set |
| `metrics_port` | `9102` | Health/metrics port |
| `provider` | `local` | Snapshot SOURCE: `shared` (the `SharedSnapshotProvider` — convergence, PRODUCTION) or `local` (the per-validator `miner_manager` fold — report/dryrun). `config/default.yaml` ships `shared` |
| `authority_url` | `""` | (shared) The Scoring Authority pointer API base URL; carries no credentials |
| `authority_token` | `""` | (shared) Bearer presented to the authority (its `authority.api_token`) |
| `authority_netuid` | `85` | (shared) The subnet id the anchor reader keys off when verifying the third digest leg |
| `authority_timeout_seconds` | `10` | (shared) Timeout around every pointer fetch |
| `verify_anchor` | `true` | (shared) Verify the on-chain anchored digest as the third, independent leg. When on, an anchor reader MUST be wired (HOLD otherwise); a not-yet-anchored epoch HOLDs; a mismatch REFUSES. Disable ONLY in overlays with genuinely no chain anchor |
| `convergence_observe_enabled` | `false` | Observe-only: emit `vidaio_weightsetter_convergence` from peer validators' on-chain vectors (never a submit gate) |
| `convergence_peer_hotkeys` | `[]` | Explicit peer validator hotkeys to sample |
| `convergence_use_metagraph_peers` | `false` | Also sample every metagraph validator (excluding ours) as a peer |

Construction wiring (not config): `chain=`, `snapshots=` (SnapshotProvider),
`conn=` (its own connection to the validator's DB file in the real stack —
see `the development-tree stack runner`), `store=`, `ledger=`,
`publication_inputs=` (the validator's `ScorePacketEvidence`), `clock=`, and
`wall_clock=`. There is intentionally no audit reviewer or reporter constructor seam.

## How to test

```sh
python -m pytest tests/weightsetter
```

By concern: `test_service.py` (attempt loop, tri-state confirmation,
publication rule), `test_durability.py` (intent ledger across
crash/ambiguity/reconcile), `test_publication.py` (artifacts, sentinel,
watermark, re-drive), the reward-state persistence tests (replay-safe resolution,
exact immutable results, and legacy-table isolation), `test_config.py`. Cross-process evidence flow:
`tests/validator/test_evidence.py` and `the development-tree e2e suite`.
The `test_own_audit*.py` and `test_own_audit_ledger.py` files cover the legacy helper;
production audit-process coverage lives under `tests/auditor` and integration/E2E tests.

## How to change safely

- Never let any path publish an intent that is not CONFIRMED/accepted, and
  never let UNKNOWN settle anything — the truth table in the `service.py`
  module docstring is the contract, and the durability tests encode it.
- Tempo classification string-matches `"tempo"` in the failure message
  (`_is_tempo`); a real adapter must map the chain's rate-limit rejections
  into a `"tempo"`-containing message or reschedules will be counted as
  chain failures.
- Any change to `weight_vector_document`'s shape must bump
  `WEIGHT_VECTOR_DOMAIN` — it invalidates recorded digests.
- Keep `_publication_digests`'s `None` vs `[]` distinction: flattening a
  read failure into an empty list anchors a false "no evidence" claim on
  chain.
- The watermark must remain the EARLIER of the candidate bounds
  (`packets_frozen_at` over `settled_at`): overlap between consecutive
  evidence windows is harmless, a gap is permanent.
- Feature detection of `recent_packet_digests(since)` is by SIGNATURE, not
  by catching `TypeError` — a provider's internal `TypeError` must not
  silently widen the evidence window.
- Schema changes are new migration files; `weight_intents` rows are audit
  history — never rewrite them in place.

## Status & gaps

- [DONE] Composition, tempo-gated submission, intent ledger, tri-state
  vector-specific confirmation, publication + sentinel + watermark,
  reconcile recovery loop, schema-v14 reward-state persistence, live
  competition reward-window convergence, the `SharedSnapshotProvider`
  convergence source (pointer → mirror → 3-way verify → HOLD-on-mismatch),
  the weight-setter-only runtime boundary, and the convergence-health gauge. Beacon and
  own-audit reporting run in isolated auditor services.
- [DONE, needs testnet validation] Real-chain submission via
  `BittensorChainAdapter` ([`vidaio/chain`](../chain/README.md)): the commit-
  reveal caveat is honoured (a pending timelocked commit answers UNKNOWN, so the
  DENIED path never abandons an intent awaiting its reveal window), and the
  `set_weights` timeout bound is dropped so the inclusion wait runs to
  completion.
- [DONE, needs testnet validation] `anchor_commitment` records its inclusion block and
  shared readers verify the digest through an exact historical read at that block. The
  Commitments-pallet cadence/finality and archive availability have no SN85 live result
  yet; anchor failure remains a HOLD.
- [DONE, needs live-bucket validation] S3-compatible audit storage, AES-GCM holdouts,
  and post-retirement public evidence release are wired. Each deployment must still set
  its bucket/role-scoped IAM, authority-only AES key, authority URL/token, validator
  hotkey, and public-prefix policy before shared publications are meaningful.
