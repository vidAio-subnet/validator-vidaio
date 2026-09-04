# vidaio.epoch — the shared per-epoch EpochLog: the ONE artifact both sides read

The dependency-light data model (pydantic + stdlib only) that the two-API rework is
built around. ONE `EpochLog` per epoch drives BOTH sides of the subnet: the central
Scoring Authority ([`vidaio/authority`](../authority/README.md)) PRODUCES it each
epoch, the thin validator ([`vidaio/weightsetter`](../weightsetter/README.md))
CONVERGES from its weight vector, and the auditor ([`vidaio/auditor`](../auditor/README.md))
VERIFIES from its audit manifest. Spec: the project design record
§3.1, the project design record §2.2.

It reuses exactly two designated shared primitives and nothing heavier, so importing
it never drags a service/SDK tree in:
[`vidaio.tokenomics.quantize.quantize_u16`](../tokenomics/README.md) (THE deterministic
float→u16 grid) and [`vidaio.audit.canonical`](../audit/README.md)
(`canonical_json_bytes` / `sha256_hex` — sorted keys, no whitespace). Per-uid inference
inputs and the live `CompetitionResult` / `RewardWindowState` come from
`vidaio.tokenomics.state`. Schema v15 commits total replay cursors, chain-time reward
application, and the exact executable provenance needed for CPU-only recomputation.

## What it does

- **`log.py`** — [DONE] the whole model:
  - **`EpochLog`** — the frozen, content-addressed per-epoch artifact. Carries
    identity/pinning (`epoch_id`, `close_block`, `scorer_version`, `schema_version`,
    `created_at`), the chaining `prior_log_digest`, the optional residual `burn_uid`,
    the complete registered subnet identity set (`miner_census`), the eligible
    economic inputs (`miners`), the optional machine-derived `competition_result`, the
    replay-safe `reward_window_state`, the canonical vector as BOTH `weight_shares`
    (float) AND `weight_u16` (its `quantize_u16`) + `weight_vector_digest`, and the
    `audit_manifest`.
    `to_json()` / `log_digest()` / `from_json()` are the
    byte-identity + digest contract; construction VALIDATES the convergence invariants
    (below) and raises `EpochLogInvalid`.
  - **`AuditManifest`** — `per_uid[uid] -> tuple[AuditFileRef, ...]` (the auditor's
    literal inference worklist), `baseline_bundles`, the committed
    `score_packet_merkle_root`, `earning_inputs[uid]`, the cumulative
    `fold_cursors[uid]` replay boundary, and the live `competition_input` /
    `competition_bundles` worklists.
    `refs_for(uid)` / `earning_for(uid)` are the readers.
  - **`MinerCensusEntry`** — one close-block registered subnet identity
    (`uid`/`hotkey`/`coldkey`/`ip`). It deliberately has no track or earning state, so an
    offline, new, or unknown-track registration remains bindable without entering tokenomics.
  - **`AuditFileRef`** — one audit file backing a weight: `kind` (`AuditFileKind`,
    `SCORE_PACKET` | `AUDIT_BUNDLE`), the store `digest`, the `challenge_id` / `item_id`
    / `source` pins, the per-item merkle `inclusion_proof` (SCORE_PACKET only), and the
    COMMITTED `committed_track`.
  - **`EarningInput`** — the verifiable derivation of one uid's earning state:
    `prior_accumulate_score` (the chained carry-in) + the ordered `cycle_scores` that
    EWMA-fold into it.
  - **`CompetitionAuditItem` / `CompetitionAuditSubject` / `CompetitionInput`** — the
    complete hidden item matrix; each subject's exact packet/bundle pairs, sealed
    contender archive/git/tree/image identity; executable-baseline artifact, image, and
    provenance digests/sizes; manifest and commitment roots; aggregation version; cycle;
    operational completion time; chain-bound application time; and the exact pre-enrollment
    anchor subnet/raw payload/digest/inclusion hash/finality receipt. There is exactly one
    non-earning baseline subject and at least one contender. Human review fields do not
    exist in this economic structure.
  - **`EpochLogInputs`** — the plain struct the finalizer assembles a log from.
  - **`weight_vector_digest(weight_u16)`** — the sha256 over the canonical authority
    sum-grid pairs a validator cross-checks. The post-submit weight publication separately
    records the SDK-emitted runtime max-grid.
  - **`EPOCH_LOG_SCHEMA_VERSION`** (currently **15**) — the `version_key` convergence
    fence: a validator on an old schema must not converge with one on a new schema
    (their bytes/digests differ). See `the project design record` §8.

## Design & decisions

### Byte-identity is the whole point

`to_json()` normalizes EVERY collection to a deterministic order (registered census and
economic miners by uid, u16
as uid-ascending pairs, manifest refs by a stable sort key, dict keys canonicalized by
`canonical_json_bytes`) and delegates key ordering to the canonical-JSON contract, so
the SAME epoch state yields byte-identical `EpochLog` bytes and `log_digest` on any
machine, regardless of input dict/list order (a tested property). Two validators that
fetch the same finalized log agree on the u16 vector with zero coordination — the
mechanism behind convergence (the project design record rule 9).

`cycle_scores` inside an `EarningInput` are the deliberate exception: EWMA is
history-dependent, so they are kept in fold order, NOT sorted.

### Construction invariants (`EpochLogInvalid`, not `ValueError`)

`EpochLog(...)` rejects, up front:

- **the convergence cross-check** — `weight_u16 == quantize_u16(weight_shares)`; the
  u16 vector must be the deterministic quantization of the float vector, or nobody
  converges;
- **digest binding** — `weight_vector_digest` must bind the u16 vector;
- **census structure** — `miner_census` has one row per uid, and every economic `miners`
  row must be present there with exactly matching `hotkey`/`coldkey`/`ip`;
- **manifest coverage** — every positive inference earner must have current evidence or
  a verifiable predecessor carry; a competition-only payee must be bound by the committed
  competition subjects and reward window. The canonical sink is evidence-exempt;
- **competition binding** — a stated result requires exact `competition_input`, subjects
  and `competition_bundles`; cycle/time/identities/provenance must agree and the baseline
  cannot earn. `CompetitionResult.applied_at` must equal `EpochLog.created_at`, while the
  committed operational `completed_at` must be no later;
- **reward identity** — every newly applied contender must match the complete registered
  close-block census. A registered contender absent from the narrower economic snapshot
  remains in the auditable result while any unpayable podium share burns. During a carried
  window the committed hotkey is the recipient identity (the source-result uid remains
  provenance); an absent hotkey's fixed share burns instead of being redistributed;
- **the sink path** — a set `burn_uid` must receive a positive residual and cannot overlap
  the miner census, inference evidence, or competition subjects. It may be the sole
  positive uid in an empty epoch or coexist with earners in a partially allocated epoch.

`EpochLogInvalid` is deliberately NOT a `ValueError` subclass: pydantic would wrap a
`ValueError` raised inside a validator into its own `ValidationError` and hide the
domain error. As a plain `Exception` it propagates from construction unchanged, so the
finalizer and the shared-snapshot provider catch one meaningful type.

### Schema-version history (each bump changes every `log_digest`)

- **v2–v5** introduced Merkle membership, predecessor-chained earning inputs, committed
  tracks, and evidence-bound ordered EWMA cycles.
- **v6–v10** record the historical retention and first competition-evidence designs,
  including their later removal. These wire shapes are rejected by current readers.
- **v11** introduced close-block census binding and a cumulative replay boundary.
- **v12** restored machine-score-only competition economics and CPU recomputation.
- **v13** added committed duplicate exclusions and per-subject execution-image identity.
- **v14** replaces the partial replay map with total
  `fold_cursors: uid -> int | null`: every current census uid is present, `null` means
  observed but never folded, the complete predecessor map persists as tombstones, the
  first fold after `null` is allowed, and subsequent folds strictly advance. It also
  makes `RewardWindowState` the sole reward-state DTO, binds result application to
  `EpochLog.created_at`, commits exact executable-baseline provenance, and commits sealed
  contender archive/git/tree/image identities.
- **v15 (CURRENT)** makes validator permit explicitly non-exclusive: the exact census
  contains every registered subnet identity, and serving permit-holders remain scoreable
  and payable. The JSON field shape is unchanged, but old v14 producers/auditors disagree
  on metagraph membership, so v14 logs are intentionally rejected at this fleet fence.

## Public API (`vidaio/epoch/__init__.py`)

`EpochLog`, `EpochLogInputs`, `EpochLogInvalid`, `MinerCensusEntry`, `AuditFileRef`,
`AuditFileKind`, `AuditManifest`, `CycleScore`, `EarningInput`,
`CompetitionAuditItem`, `CompetitionAuditSubject`, `CompetitionInput`,
`EPOCH_LOG_SCHEMA_VERSION`, `weight_vector_digest`.

No service, no config section, no I/O — a pure model shared by the producer/consumer
packages.

## Data & invariants

- The log is **frozen** and **content-addressed**: `log_digest()` == sha256 of
  `to_json()` == the value anchored on chain and verified against fetched bytes by every
  validator. `from_json(...)` re-runs every construction invariant, so a mirrored log
  that violates its own convergence rules is rejected on parse, not quantized.
- `weight_shares` / `weight_u16` are the SAME authority vector in float and deterministic
  sum-grid representations; the adapter's confirmed publication records the separate
  runtime max-grid produced by Bittensor 10.5.
- `prior_log_digest` chains epochs: an auditor holding the prior log checks
  `prior_log_digest == prior_log.log_digest()` before trusting a nonzero carry-in.
- `audit_manifest.fold_cursors` is a cumulative, anchored replay boundary. Entries are
  tombstones rather than census rows and therefore are never removed on deregistration.
- Competition economics is derived from exact packet scores and bundles; no human-review
  field is serialized into the earning input.
- Every schema-shape change bumps `EPOCH_LOG_SCHEMA_VERSION` — it is the mixed-version
  fence.

## How to test

```sh
python -m pytest tests/epoch
```

`test_log.py` covers byte-identity across input orderings, the digest contract, the
`quantize_u16`/digest/coverage/burn invariants, the merkle-proof and earning-input
round trips, and `from_json` re-validation.

## How to change safely

- ANY change to the canonical-JSON shape (a new field, a reordering, a serialization
  tweak) MUST bump `EPOCH_LOG_SCHEMA_VERSION` — it changes every recorded `log_digest`,
  so a mixed-version fleet fences on it. This is the convergence version key.
- Keep the two shared primitives the ONLY heavy imports (`quantize_u16`, canonical
  JSON); pulling a service/SDK dependency in here would poison both the validator and
  the auditor that import this model.
- New collections must serialize in a deterministic order in `_canonical_obj` (sort, or
  preserve a documented order like `cycle_scores`) or byte-identity — and convergence —
  breaks.
- Keep `EpochLogInvalid` a plain `Exception`; do not make it a `ValueError`.

## Status & gaps

- [DONE] The full model through schema v15: total fold cursors, complete registered census,
  executable-provenance-complete competition evidence, chain-time result application,
  independently re-readable pre-enrollment anchor receipts, reward-window economics,
  the byte-identity + digest contract, and
  every construction invariant — pure logic, fully tested.
- [DONE] Result economics use the epoch close-block chain time. Operational completion
  time is committed for chronology but never drives reward-window duration.
- [DONE] Every competition subject has an exact packet/bundle matrix and execution image;
  contenders additionally bind their sealed submission archive size/digest and immutable
  repository commit/tree identity. The executable baseline binds artifact and provenance
  sizes/digests plus its image.
- [PENDING DECISION] The sampling policy the manifest's items are later drawn against
  lives in [`vidaio/auditor`](../auditor/README.md) / `the project design record` §8,
  not here — this model only carries the manifest.
