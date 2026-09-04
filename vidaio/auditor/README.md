# vidaio.auditor — isolated CPU honesty workers

The other half of the project design record rule 10
(the project design record §1c): a validator
CONVERGES on the central Scoring Authority's vector AND independently AUDITS that
authority. Each validator identity runs a beacon auditor and full own-auditor in separate
OS containers/cgroups from the thin weight-setter and from one another. Each epoch both
workers read the epoch log's [audit manifest](../epoch/README.md) and, in production,
RECOMPUTE every committed inference and competition item over the REAL CPU
scoring engine, RE-DERIVE the log's earning state, machine competition result, chained
reward window, and weight vector from committed inputs, and aggregates a signed,
deterministic [`AuditReport`](#verdicts-and-the-derived-overall-7-8) it submits to the central Audit Results
API ([`vidaio/audit_api`](../audit_api/README.md)). A substituted score, competition
rank, reward window, weight, or earning state cannot survive an independent recompute — that is what makes the central
authority tamper-evident in public.

`Auditor` is a plain, injectable component (not a `BaseService`); production entrypoints
wrap it in two standalone service loops. It is never run inline with or constructed by
`WeightSetter`, and it never imports the authority package. Each wrapper owns its own
durable contiguous cursor and pending-report outbox.

## What it does

- **`recomputer.py`** — [DONE] `RealScoreRecomputer`: the crux. Reruns the honest
  scoring pipeline over an item's integrity-verified artifacts and returns the
  `RecomputedScore` that `verify_bundle` compares against the RECORDED packet.
- **`service.py`** — [DONE] `Auditor` (`audit_epoch` / `audit_and_submit`), the
  `BundleSource` seam (`StoredBundleSource` / `InMemoryBundleSource`), `persist_bundle`,
  the earning-state re-fold, and schema-v15 competition/reward-window derivation.
- **`sampling.py`** — [DONE] deterministic, beacon-hardened, stratified sampling
  (`sample_items`, `manifest_items`, `AuditItem`).
- **`report.py`** — [DONE] `AuditReport` + `ItemVerdict` / `WeightVerdict` /
  `ItemVerdictKind` / `AuditStatus`, `overall_status`, the `ReportSigner` seam +
  `Sha256Signer` double, and the stable codes.
- **`config.py`** — [DONE] `AuditorConfig` + `SamplePolicy` (the `auditor:` section).
- **`client.py`** — [DONE] the `AuditResultsClient` submit seam +
  `RecordingAuditResultsClient` test double.

## Design & decisions

### The real recompute (`RealScoreRecomputer`) — reuse, never re-implement

The recomputer composes the audit artifacts back into a
[`ScoreRequest`](../services/README.md) and runs the scoring worker's OWN pipeline
function, `vidaio.scoring_worker.service._score_sync`, verbatim — same
verify-then-snapshot of every input, same canonicalization, same VMAF models, same gate
pipeline, same gates-first composition. So an honest packet recomputes EQUAL within the
audit tolerances because it is the identical code path that produced it; an injected or
substituted score fails `SCORE_MISMATCH`. It stamps and enforces the SAME
`scorer_version` the worker would for its config
(`effective_scorer_version`), so recompute parity is a property of running the identical
scorer, not a coincidence.

**Committed params, not the packet's (#9).** The gate parameters come from a source the
authority cannot dial: the recompute passes NO `vmaf_threshold` (the worker falls back
to the auditor's OWN locked `ScoringConfig` threshold for the track) and NO
`content_length` (the worker re-measures it from the committed reference bytes). A real
VMAF 86 packaged with `vmaf_threshold=0` + a positive score cannot let the auditor reuse
0 and reproduce the misreport. An honest packet — produced with the same config threshold —
still recomputes EQUAL.

**Two-track CPU recompute.** Compression uses deterministic CPU libvmaf. Upscaling uses
PIQ PieAPP on CPU, and tone/grayscale/chroma manipulation gates use the deterministic
CPU/OpenCV backend. The auditor forces `pieapp_device="cpu"` regardless of a scorer's
deployment device, so a validator never needs a GPU. The locked image preloads the
digest-pinned PieAPP weights. If a required package, model, or artifact is genuinely
unavailable, `RecomputeUnavailable` remains an honest SKIP/INCONCLUSIVE boundary rather
than a false CLEAN. `unsupported_reason(...)` checks the committed track before decoding,
so substituting `track=upscaling` cannot create an audit bypass.

### The earning-state re-fold (`_earning_verdicts`, #1)

A self-consistent but INVALID log — one whose EARNING STATE was substituted (an
inflated `accumulate_score`) even though every published score packet is honest —
reproduces the authority's weight vector, so a pure weight cross-check passes. The
auditor closes that gap CHEAPLY over the FULL vector (no media): for each nonzero-weight
uid it re-folds `accumulate_score` from the manifest's `EarningInput` (prior carry-in +
ordered cycle scores), checks that every nonzero cycle score is backed by one of the
uid's audited SCORE_PACKETs, and CHAINS the carry-in against the prior epoch's log (back
to genesis). A mismatch is `EARNING_STATE_MISMATCH` (a provable FAIL); an unverifiable
one (no earning input carried, or a nonzero carry-in with no prior log supplied) is
`EARNING_STATE_UNVERIFIED` (a SKIP, surfaced, never washed to PASS). An earning FAIL is
reflected THROUGH the weight verdict so the roll-up (and the Audit Results API, which
reads only `item_verdicts` + `weight_verdict`) disputes on it without a signature change.

### Competition result and reward-window re-derivation (schema v15)

An earning competition carries `CompetitionInput` plus an exact score-packet/audit-bundle
pair for every subject/item. The auditor CPU-recomputes that complete matrix, requires
exact digest and identity coverage, derives each subject's arithmetic mean, orders
contenders by `(-score, hotkey, uid)`, derives the one executable baseline, and
exact-compares the resulting `CompetitionResult`. It first independently archive-reads
the committed pre-enrollment raw anchor at its exact inclusion block, validates the
normalized block hash/finality, and proves both inclusion and finality precede enrollment
and epoch close. Missing RPC is `COMPETITION_UNVERIFIED`; readable mismatch is
`COMPETITION_MISMATCH`. The result's `applied_at` must equal
`EpochLog.created_at`, the close-block chain time; committed `completed_at` is accepted
only when no later than that time and never drives economics. It then folds only the
new result into the predecessor via `resolve_reward_window` and exact-compares
`RewardWindowState` at the log time. Stored `final_rank`,
`manual_disqualified`, `eligible`, and human reviews are absent from this derivation.

Every contender commits a positive-size sealed submission archive, archive digest,
execution-image digest, repository URL, commit SHA, and tree SHA. The baseline commits
its execution image, while `CompetitionInput` additionally commits exact baseline
artifact/provenance sizes and digests. The auditor requires every resolved bundle image
to match its subject. A CROWN winner's sealed archive must also exist publicly and match
the committed size/digest, so the anchored result identifies a releasable executable.

Missing/unreadable evidence is `COMPETITION_UNVERIFIED`; a score/result/order/baseline
disagreement is `COMPETITION_MISMATCH`; a chained reward-state disagreement is
`REWARD_WINDOW_MISMATCH`. A positive competition-only payee is exempt from the
inference EWMA fold only when committed competition and reward-window identities back it.

**Total replay boundary (schema v15).** `AuditManifest.fold_cursors` is the complete,
anchored `uid -> int | null` map, not merely a summary of the current epoch. It must equal
the whole predecessor map including tombstones, plus every current census uid inserted as
`null` if never observed, advanced only by this epoch's committed maxima. `null` means the
identity was observed but never folded; its first fold is allowed. An integer cursor must
strictly advance. Entries survive idle epochs, exclusion, deregistration, uid/hotkey reuse,
and empty/burn epochs. A missing/regressed/invented boundary is
`FOLD_CURSOR_MISMATCH`; a cycle at or below the carried boundary is
`EARNING_PACKET_REPLAY`. Both are conclusive failures.

**Independent registered census.** `EpochLog.miner_census` carries every registered
subnet identity's `uid`/`hotkey`/`coldkey`/`ip` at the exact close block, separately from
`log.miners`, which remains the eligible/economic subset and requires a known protocol
track. The auditor reads `neurons_at(close_block)`, exact-compares all census identities,
requires every economic row to be a matching member, and still rejects committed earning
evidence omitted from the economic set. This keeps offline/new/unknown-track registrations
visible without forcing them into tokenomics. Validator permit is a capability rather than
an exclusive role, so permit-holding serving miners remain in both the census and economic
path. An unavailable metagraph is INCONCLUSIVE even
for an empty census; it never proves an empty subnet.

### Selection policy and the post-finalization beacon

`sample_items` is a pure function of `(manifest, beacon, epoch_id, auditor_hotkey,
policy)` — no wall clock, no PRNG state — so a run is REPRODUCIBLE (anyone can check
which items the auditor should have recomputed). The seed mixes in an UNPREDICTABLE
BEACON the authority cannot know when it BUILDS the manifest: the epoch log's own
fixed future chain hash, `block_hash(close_block + K)`, read only after that block is
finalized (#10). Previously the seed was
only `sha256(epoch_id || auditor_hotkey)` — public and fixed before finalization — so
the authority, controlling the manifest item keys, could grind invalid IDs to land
outside every known auditor's sample. Anchoring the seed to a chain-chosen value AFTER
the manifest is fixed removes that freedom. Selection is stratified by source
(competition vs inference) so both tracks always get coverage; within a stratum items
are ordered by `sha256(seed || item_key)` and the first `SamplePolicy.target_count`
taken. `NO_BEACON` (report/dev, un-anchored epochs) keeps determinism but drops
unpredictability — production always passes the finalized future-block beacon, and the
epoch anchor must land before it becomes knowable.

The two launch workers override the generic sampling defaults with uncapped `all_items`
coverage at rate `1.0`, so they recompute the complete committed population. The beacon
still binds `audit_mode=beacon`; the selection algorithm remains available for additional
third-party auditors that intentionally choose reduced coverage.

### Verdicts, and the derived overall (#7, #8)

`verify_bundle` PROVES a sampled item honest (PASS) or invalid (a typed FAIL:
`SCORE_MISMATCH`, `MERKLE_EXCLUSION`, `IDENTITY_MISMATCH`, `REVEAL_INVALID`, …); an
unreachable/un-recomputable item is a SKIP, never a PASS-in-disguise. With the v2
manifest carrying the committed `score_packet_merkle_root` + per-item inclusion proofs,
the auditor runs STRICT merkle inclusion by default (`config.strict=True`): a sampled
item not provably in the committed root fails `MERKLE_EXCLUSION`. The `AuditReport.overall`
is a DERIVED, validated property — always recomputed at construction from
`item_verdicts` + `weight_verdict` via `overall_status`, so a report can NEVER claim
CLEAN while carrying a fault. `overall_status`: **DISPUTED** if any item/weight FAILs;
else **INCONCLUSIVE** if there WERE sampled media items but every one SKIPped (nothing
recomputed — not clean, needs attention); else **CLEAN**. A SKIP never disputes, but an
all-SKIP media sample is never washed to CLEAN.

### Bundle resolution + the submit seam

The manifest names a `bundle_digest`, but an `AuditBundle` is not a standalone blob, so
the auditor resolves it through an injected `BundleSource`. **`StoredBundleSource`** is
the real one: `persist_bundle(store, bundle)` writes each epoch's bundle
content-addressed (the store digest IS the `bundle_digest()`), and `bundle_for` fetches
it back, verifies `sha256(bytes) == ref.digest` (verify-on-read), parses, and re-checks
`bundle_digest()` — a corrupt/unreachable/malformed bundle raises `BundleUnavailable` →
SKIP. The object store IS the bundle store. `Auditor.over_store(...)` wires it in
production; the standalone service supplies `make_public_store()`, an unsigned,
read-only S3/Hippius view that loads neither write credentials nor the live holdout AES
key. Tests inject `InMemoryBundleSource`. Production bundles carry real challenge
input, miner output, reference original, manifest, score packet, and DAG reveal refs.
Live holdouts are AES-GCM sealed; terminal resolution publishes verified reference
plaintext under `released/reference_original/…` while its canonical holdout key remains
private. The public view resolves that ref only after release and never probes or decrypts
the canonical object. Public recomputation also requires the non-secret evidence prefixes listed in
the project design record. The finished beacon report is signed with
`audit_mode=beacon` and POSTed through the
`AuditResultsClient` seam (`vidaio.audit_api.HttpAuditResultsClient` in production, the
recording fake in tests). The standalone own-auditor signs and POSTs its separate
`audit_mode=own_audit` report. Both modes are report-and-alert only; neither worker,
verdict, nor delivery failure gates `set_weights`. Swapping the transport changes
nothing above it.

## Public API (`vidaio/auditor/__init__.py`)

`Auditor`, `BundleSource`, `BundleUnavailable`, `InMemoryBundleSource`,
`StoredBundleSource`, `persist_bundle`; `RealScoreRecomputer`, `RecomputeUnavailable`;
`AuditItem`, `ManifestIncomplete`, `manifest_items`, `sample_items`; `AuditReport`,
`AuditMode`, `AuditStatus`, `ItemVerdict`, `ItemVerdictKind`, `WeightVerdict`, `overall_status`,
`ReportSigner`, `Sha256Signer`, `WEIGHT_DERIVATION_MISMATCH`, `EARNING_STATE_MISMATCH`,
`EARNING_STATE_UNVERIFIED`, `EARNING_PACKET_REPLAY`, `FOLD_CURSOR_MISMATCH`,
`COMPETITION_MISMATCH`, `COMPETITION_UNVERIFIED`, `REWARD_WINDOW_MISMATCH`,
`CENSUS_MISMATCH`; `AuditorConfig`, `SamplePolicy`; `AuditResultsClient`,
`RecordingAuditResultsClient`, `SubmitAck`. No HTTP endpoints — the auditor is a
component, not a service.

## Data & invariants

- The reusable `Auditor` component has no database. Each standalone service wrapper has
  its own durable cursor and pending-report SQLite store on a distinct volume. It reads
  the epoch log (mirrored bytes), the object store (bundles + packets, verify-on-read),
  the exact close-block metagraph, and prior logs for the carry-in/replay chain. Signed
  report bytes enter its outbox before central delivery and are retried byte-identically.
- Each worker exposes accepted `auditor_reports_total{status}` plus
  `auditor_report_delivery_attempts_total{status}` and
  `auditor_report_delivery_failures_total{status}`. It logs local DISPUTED findings at
  CRITICAL and INCONCLUSIVE findings at WARNING before delivery. The central API repeats
  the signal on acceptance or conflict and exports received/by-mode/conflict metrics.
  Every signal calls for manual remediation; none is automatic enforcement.
- Media is streamed to digest/size-verified temporary files with 2 GiB caps for
  challenge input/reference and 4 GiB for miner output. Metadata is capped at 16 MiB,
  each bundle at 1 MiB, and each mirrored epoch log at 64 MiB. Size the auditor's local
  scratch for those files and scorer intermediates; these limits prevent a valid 4 GiB
  output from first being accumulated in memory.
- The report is FROZEN and deterministic: `canonical_bytes()` is canonical JSON over
  every field except the signature, so the same audit run yields byte-identical report
  bytes on any machine — the bytes a hotkey signs and `report_digest()` addresses.
  `audit_mode` is `beacon` or `own_audit`; non-default `own_audit` is included in those
  signed canonical bytes, while the default `beacon` encoding preserves historical
  signatures and report digests.
- `overall` is never a free field (re-derived at construction); the media coverage floor
  counts only `competition`/`inference` items (the synthetic `earning`/`weight` rows do
  not).
- Recompute reason codes are FORWARDED from `vidaio.audit.recompute`, never invented;
  the auditor adds stable whole-log codes for weight, earning, competition, reward state,
  census, sink, and replay checks.

## Configuration

`AuditorConfig` (schema `config.py`) is the reusable component's settings. The standalone
`auditor` and `own-auditor` entrypoints construct it independently from the shared
launcher configuration; neither is wired into the weight-setter. The knobs:

| Key | Default | Meaning |
|---|---|---|
| `auditor_hotkey` | `""` | The on-chain identity every report is attributed to and (with a signer wired) signed under |
| `audit_mode` | `beacon` | Signed report identity: the `auditor` entrypoint selects `beacon`; the separate `own-auditor` entrypoint selects `own_audit`, so both paths coexist per hotkey+epoch |
| `sample_policy.sample_rate` | `0.10` | Fraction of the item population recomputed, applied PER SOURCE |
| `sample_policy.min_samples` / `max_samples` | `1` / `50` | Per-source clamp on the sampled count |
| `backend` | `real` | `real` = CPU ffmpeg/libvmaf + PIQ PieAPP + perceptual recompute; `fake` = REQUIRES an injected recomputer (tests) |
| `strict` | `true` | Strict `verify_bundle`: absent anchors count as failures — the v2 manifest carries the merkle root + per-item proofs, so strict merkle inclusion is proved for every sampled item |
| `results_api_url` | `""` | The Audit Results API pointer (wave 7) |
| `results_api_token_env` | `VIDAIO_AUDIT_RESULTS_TOKEN` | NAME of the env var the bearer token is read from (never the token itself) |
| `authority_api_token_env` | `VIDAIO_AUTHORITY_READ_TOKEN` | NAME of the env var holding the authority read bearer. Production auditors never receive or fall back to `authority.api_token` |
| `tokenomics` | (locked levers) | The tokenomics config the auditor re-derives weights with; MUST match the authority's (the project design record #5) or an honest log false-flags `WEIGHT_DERIVATION_MISMATCH` |

The service wiring also requires a production launch boundary. Before first start,
select a future runtime `SubnetEpochIndex`—normally the latest archive-proven closed
index plus one—and configure it as `local_stack.auditor_cursor_floor` on every process;
it is also the authority genesis. A fresh/lost cursor audits contiguously from this
trusted positive floor. It never discovers a floor from pointer 404s; a missing epoch at
or above the floor HOLDs so an authority cannot hide earlier misreporting. Retain the
chosen value with release evidence. Production also requires
`auditor_beacon_confirmation_depth >= 20` and only treats the future beacon block as
available once the chain reports it finalized. The service wrapper separately counts
consecutive HOLD/REFUSE passes on the same required epoch. One transient remains healthy;
health degrades halfway through `local_stack.auditor_max_consecutive_stalls` and the
process exits fatally at the limit (default 30, about 10 minutes at the production
20-second cadence). Completing a cursor walk, finding no work, or moving to a different
required epoch resets/starts the counter, so a permanent contiguous hold cannot look
healthy forever.

Production also sets `local_stack.auditor_media_sample_rate` and
`local_stack.own_auditor_media_sample_rate` to exactly `1.0`. Configure
`auditor_cursor_db_path` and `own_auditor_cursor_db_path` on distinct durable volumes;
the same process-private database also backs each worker's pending-report outbox. Metrics
are separate on 9121 and 9122.

## How to test

```sh
python -m pytest tests/auditor
```

By concern: `test_recomputer.py` (real CPU recompute parity + the #9 committed-param
refusal), `test_recomputer_refusal.py` (honest unavailable-backend refusal),
`test_sampling.py` (determinism, stratification, beacon un-steerability),
`test_earning_state.py` (the re-fold, packet backing, carry-in chaining, cumulative
replay/omission/hotkey regressions), `test_competition_cycle.py` (chained cycle
monotonicity), `test_census.py` (exact close-block registration binding and outage
behavior), `test_stored_bundle_source.py` (verify-on-read, corruption → SKIP), and
`test_service.py` (competition/result/reward-window coverage plus the full audit → derived
overall → submit path, including INCONCLUSIVE).

## How to change safely

- Never let an item the backend cannot honestly recompute become a PASS or a spurious
  FAIL — it is a SKIP (the `RecomputeUnavailable` / `unsupported_reason` path), and the
  media coverage floor turns an all-SKIP sample INCONCLUSIVE, never CLEAN.
- Recompute must keep running the worker's OWN `_score_sync` and the auditor's OWN
  locked gate params (#9) — reusing the packet's `vmaf_threshold`/`content_length`
  re-opens the misreporting channel.
- `overall` must stay derived at construction; never accept a caller-supplied roll-up.
- Keep sampling a pure function seeded from the post-finalization beacon (#10); a seed
  the authority can precompute lets it grind items out of every sample.
- The auditor's `tokenomics` levers must track the authority's exactly, or honest logs
  false-dispute.

## Status & gaps

- [DONE] The real CPU recomputer for both tracks, deterministic perceptual checks, the
  miner-input basis for every anti-gaming/perceptual check, numeric gate-boundary
  hysteresis constrained by the declared per-metric audit tolerances,
  earning-state re-fold, complete competition packet/result/reward-window re-derivation,
  weight re-derivation, total replay boundary, exact registered census,
  beacon-hardened sampling, the stored
  bundle source, the derived-overall report + submit seam.
- [DONE, needs testnet validation] Bittensor mode signs with the loaded wallet through
  `BittensorHotkeySigner`; the Audit Results API verifies the signature and live subnet
  registration. `Sha256Signer` remains a deterministic report/test double.
- [LOCKED FOR LAUNCH] Both production workers use CPU `all_items` at rate `1.0`. The
  remaining policy question is only whether additional third-party/public auditors may
  choose reduced coverage and what confidence target governs it.
- [DONE] Exact score/result/reward-window/payout recomputation is live for the committed
  evidence. Schema v15 binds result application to the epoch close-block time and commits
  exact baseline artifact/provenance plus sealed contender release identities.
- [DONE] Findings remain report-and-alert only. CLEAN, DISPUTED, INCONCLUSIVE, or report
  delivery failure never interrupts the authenticated authority-vector submission path;
  remediation is manual.
