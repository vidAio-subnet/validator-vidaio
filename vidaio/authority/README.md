# vidaio.authority — the Scoring Authority: finalize, anchor, publish thin pointers

The central, owner-run measurement side of the two-API rework
(the project design record rule 10,
the project design record §1(a)). The existing
challenge/scoring/orchestrator/tokenomics stack becomes the Scoring Authority's stack;
this package is its epoch boundary: at each epoch close it assembles the immutable
[`EpochLog`](../epoch/README.md), writes it to the object store behind a `_FINALIZED`
marker, records a thin pointer in its epoch index, anchors the `log_digest` on chain,
and serves that pointer over a small HTTP API (`:8700`). Validators FETCH the pointer,
mirror the bytes themselves from the object store, and verify the digest against the
on-chain anchor before submitting. **The API never serves epoch-log bytes — only keys
and digests** — so it stays cheap, cacheable, and unable to become a per-request
tampering surface.

## What it does

- **`finalizer.py`** — [DONE] `EpochFinalizer` + `build_audit_manifest`. Turns the
  folded snapshots + the epoch's scored items into ONE validated `EpochLog` and writes it
  as a `_FINALIZED` set (member first, marker LAST). It validates the exact
  packet-score-derived `CompetitionResult`, folds the replay-safe `RewardWindowState`,
  and composes live testnet competition emissions at the epoch close-block time.
  `FinalizedEpoch` is the returned pointer (`snapshot_key` + `log_digest` +
  `weight_vector_digest`). `ScoredItem` is one (uid, item) row carrying its REAL stored
  `bundle_digest` / `packet_digest` (+ optional `score`, `committed_track`, `baseline`).
- **`api.py`** — [DONE] the POINTER payloads: `EpochPointer` (epoch id + `snapshot_key`
  + `snapshot_digest` + `weight_vector_digest` + `anchor`), `AnchorPointer`,
  `AnchorRecord`, and the `pointer_from_record` / `anchor_from_record` projections. NO
  body here ever carries the epoch-log bytes.
- **`index.py`** — [DONE] `EpochIndex`: an append-only, immutable SQLite index of which
  epochs are finalized (epoch_id → pointer + anchor). `record_finalized` (idempotent,
  raises `EpochIndexConflict` on a divergent re-finalize), `set_anchor` (fills the
  anchor columns once), `get` / `latest`. In-database triggers enforce immutability
  against direct SQL.
- **`anchoring.py`** — [DONE] `anchor_epoch` + `anchor_payload` + `ANCHOR_DOMAIN`
  (`vidaio.epoch.anchor.v1`): the ≤128-byte, domain-tagged commitment
  `<domain>:<netuid>:<epoch_id>:<log_digest>` submitted via `ChainAdapter.anchor_commitment`;
  idempotent per epoch.
- **`service.py`** — [DONE] `ScoringAuthority` (`BaseService`): the pointer API on
  `:8700` plus `finalize_and_anchor` (the composed epoch-close sequence).
- **`config.py`** — [DONE] `AuthorityConfig` (the `authority:` section).
- **`migrations/`** — [DONE] `0001_epoch_index.sql` (the `authority_epochs` table +
  immutability triggers).

## Design & decisions

### The finalizer: honest by construction

`build_audit_manifest(scored_items, store=..., competition_input=...)` groups each
inference earning row under its uid (two refs: an AUDIT_BUNDLE binding + the SCORE_PACKET
blob) and collects calibration rows into `baseline_bundles`. When a store is supplied it
PROBES every score-packet AND every bundle
digest and raises `AuditFileMissingError` on any miss — the manifest must be a
followable index, never a dead link (a weight is never backed by an audit file that was
never stored; #8). The score-packet digests of all rows are the leaves of a merkle
tree; the manifest carries its `score_packet_merkle_root` and each SCORE_PACKET ref its
own inclusion proof, both DETERMINISTIC from the sorted leaves so the manifest — and the
`EpochLog` it feeds — stays byte-identical across machines. `earning_inputs` carries,
per earning uid, the ordered cycle scores + the chained prior-epoch carry-in. Competition
rows are additionally namespaced under `competition_bundles` by the exact subject set in
`CompetitionInput`, so the same score-packet/bundle pairs are part of the epoch merkle and
the CPU auditor's literal worklist. The input also carries the exact pre-enrollment raw
anchor receipt; the authority refuses incomplete persisted receipts and independently
archive-reads its payload, inclusion hash, finality, and chronology before publishing.
The total `fold_cursors` map starts from the exact
predecessor map (including tombstones), adds every current census uid as `null` when it
has never folded, and advances only to current committed maxima. A first fold after
`null` is valid; a fold after an integer must strictly advance.

`EpochFinalizer.build_log` composes the vector via
[`build_weight_vector`](../tokenomics/README.md) (with the canonical residual `burn_uid`),
quantizes via `quantize_u16`, and runs an **earning-consistency self-check**
(`_check_earning_consistency`): every manifest `EarningInput` must EWMA-fold to its
snapshot's stated `accumulate_score`, or `EpochLogInvalid` is raised BEFORE publication
— the producer refuses to ship a log whose earning derivation does not reproduce its own
weights' inputs (the auditor re-checks the same thing against the audited packets; this
catches a producer bug first, #1). When a competition result is supplied, the finalizer
requires its complete `CompetitionInput` and exact packet-score map, derives the result
again from subject means, rejects any mismatch, and calls `resolve_reward_window` exactly
once against the predecessor `reward_window_state`. `CompetitionResult.applied_at` and
the input's `applied_at` must equal `EpochLog.created_at`; database-local `completed_at`
is committed operational evidence only and must be no later. An epoch with no newly
completed result carries the predecessor reward state without reapplying it, while active
versus expired emissions are derived from the new close time. It
sets `burn_uid` whenever a positive fixed-pool residual is routed to the canonical sink;
that UID may coexist with genuine inference/competition earners but may never overlap a
scored miner identity.

### The `_FINALIZED` half-write set (immutable)

`finalize(...)` writes the log object first, then the `_FINALIZED` marker LAST (the
production-proven half-write guard from [`vidaio.audit.store`](../audit/README.md)), so no
validator can mirror a half-written epoch. It is **idempotent**: re-finalizing an
already-`_FINALIZED` epoch is a NO-OP that reads the stored bytes back and returns the
SAME key/digest — and, crucially (#16), the recovered pointer is bound to the STORED
log's OWN fields (`epoch_id` / `close_block` / `weight_vector_digest`), never the
caller's arguments, so a crash-after-`_FINALIZED`-before-indexing that changed epoch
parameters cannot index a pointer whose metadata contradicts its anchored bytes.

For a newly derived CROWN, finalization has one additional pre-write interlock: it
resolves rank one to the exact committed `submission_archive` digest/size and verifies
that plaintext through the separate keyless public-store role. Missing, corrupt,
credential-only, or anonymously inaccessible winner source refuses finalization before
the epoch member exists. PODIUM and all non-winning source archives remain sealed.

### The pointer API (`:8700`) and the tamper-evidence chain

Three read routes serve pointers, never bytes:

    sha256(mirrored bytes) == snapshot_digest == on-chain anchored digest

A validator fetches a pointer here, pulls the bytes from the object store by
`snapshot_key`, and verifies that equality (the third leg on chain) before trusting or
submitting (`the project design record` §5). Every pointer route is gated on
`authority.api_token` (401 missing/malformed bearer, 403 wrong; open only on a
loopback/dev bind — PRODUCTION MUST SET IT); `/healthz` is always open. Unknown and
not-yet-finalized epochs are the SAME 404 (`epoch_not_found`), so an in-progress epoch
cannot leak. The API's fatal-on-death lifecycle follows the exit-code contract
([`vidaio/services`](../services/README.md)).

### `finalize_and_anchor` — the epoch-close sequence

`finalize` (write the `_FINALIZED` log) → `index.record_finalized` (record the pointer)
→ `anchor_epoch` (anchor the `log_digest`). Each step is idempotent, so the whole
sequence is idempotent per epoch: a re-run returns the same pointer and never
double-writes the store or the chain.

### Modes (rule 8): one code path

Both report and bittensor drive this exact service code — only the injected `AuditStore`
/ `ChainAdapter` differ. Tests wire a `LocalFsStore` + `InMemoryChain`; production
builds them from the shared `audit:` / `chain:` sections via `make_store` /
`make_chain_adapter`. The store backend and chain mode are NEVER selected here.

## Public API & endpoints

HTTP (port `authority.http_port`, default 8700; metrics 9111):

| Route | Contract |
|---|---|
| `GET /epoch/latest` | pointer to the newest finalized epoch — keys + digests + anchor, NO bytes. Bearer-gated. 404 `epoch_not_found` if none |
| `GET /epoch/{epoch_id}` | pointer for a specific epoch. Bearer-gated. 404 for unknown OR not-yet-finalized (never distinguished) |
| `GET /epoch/{epoch_id}/anchor` | the standalone `AnchorRecord` (epoch, anchored `digest`, `txid`, `block`) for independent on-chain checks. Bearer-gated |
| `GET /healthz` | liveness, unauthenticated |

Python surface (`__init__.py`): `EpochFinalizer`, `FinalizedEpoch`, `ScoredItem`,
`build_audit_manifest`, `AuditFileMissingError`, `epoch_prefix`, `EPOCH_LOG_MEMBER`,
`AuthorityConfig`, `EpochIndex`, `EpochRecord`, `EpochIndexConflict`, `anchor_epoch`,
`anchor_payload`, `ANCHOR_DOMAIN`, `ScoringAuthority`, `EpochPointer`, `AnchorPointer`,
`AnchorRecord`, `pointer_from_record`, `anchor_from_record`. Construction seams:
`store=`, `chain=`, `index=`, `finalizer=`, `now=`.

## Data & invariants

- **`authority_epochs`** (`migrations/0001_epoch_index.sql`) — one immutable row per
  finalized epoch (`epoch_id`, `close_block`, `snapshot_key`, `log_digest`,
  `weight_vector_digest`, `anchor_txid`, `anchor_block`, `finalized_at`). A finalized
  epoch's pointer fields can never change; the anchor columns are the only fields that
  transition (NULL → value, once). Triggers enforce this against direct SQL.
- The object store holds the bytes (`finalized/epoch={N}/log.json` inside a `_FINALIZED`
  set); the index holds only the pointer. The API is a pure index over both.
- `log_digest` (index/pointer) == sha256 of the stored member bytes == the anchored
  digest. Any drift means tampering, and the validator's three-way check refuses.

## Configuration

Section: `authority` (schema `config.py::AuthorityConfig`, `extra="forbid"`). Env
override pattern: `VIDAIO__AUTHORITY__<KEY>=<value>`.

| Key | Default | Meaning |
|---|---|---|
| `http_host` / `http_port` | `0.0.0.0` / `8700` | Pointer API bind |
| `metrics_port` | `9111` | Health/metrics port |
| `api_token` | `null` | Bearer gating EVERY pointer route (validators/operators carry it). Null = OPEN — loopback/dev only; PRODUCTION MUST SET IT |
| `db_path` | `./data/authority.db` | The append-only epoch index (own migrations) |
| `netuid` | `85` | The subnet this authority scores; stamped on the anchor |
| `blocks_per_epoch` | `360` | Report-mode fixed-grid fallback only; Bittensor mode uses archive-proven runtime epoch boundaries |
| `burn_uid` | `null` | Report/local fallback for the canonical residual sink; production derives the subnet-owner uid from chain and requires this to remain null |
| `scorer_version` | `""` | The scorer identity (`<name>+<digest12>`) stamped into each log; supplied by the central scorer, empty only in tests |

The object-store backend (`audit:`) and chain mode (`chain:`) come from the shared
sections — this service reuses `make_store` / `make_chain_adapter`, so a report-mode
overlay drives the exact same code.

## How to test

```sh
python -m pytest tests/authority
```

By concern: `test_finalizer.py` (inference/competition manifest assembly + missing-file
refusal, packet-derived result/reward-window and earning-consistency self-checks, total
fold-cursor continuity, conditional sink,
`_FINALIZED` member-first/marker-last, idempotent
re-finalize + the #16 stored-fields pointer), `test_index.py` (append-only immutability,
conflict on divergent re-finalize, anchor-once), `test_anchoring.py` (domain-tagged
payload, idempotent anchor), `test_service.py` (the pointer routes, bearer gate, merged
404, `finalize_and_anchor` end to end).

## How to change safely

- The API MUST stay a thin pointer index: never add a route that returns epoch-log
  bytes — trust comes from the digest + the on-chain anchor, not from the API.
- Keep `finalize` writing the member before the `_FINALIZED` marker, and keep the
  idempotent recovery reading the pointer off the STORED log's own fields (#16).
- Never publish a manifest an auditor cannot follow: `build_audit_manifest` must keep
  probing the store when one is supplied, and keep the merkle root + per-item proofs
  deterministic (they fold into `log_digest` and get anchored).
- Keep every epoch-close step idempotent — `finalize_and_anchor` composes them and must
  stay a full no-op on re-run.
- Schema changes are new migration files; `authority_epochs` rows are immutable audit
  history. Repo-wide: bump the root `VERSION` on any release-worthy change.

## Status & gaps

- [DONE] Finalizer (+ honest inference/competition manifest, earning/result/reward-window
  self-checks, fixed-pool sink, `_FINALIZED` write,
  idempotent recovery), epoch index, anchoring, the pointer API, `finalize_and_anchor` —
  fully tested against `LocalFsStore` + `InMemoryChain`.
- [DONE, needs live validation] The boto3 S3/S3-compatible content plane, AES-GCM
  live-holdout sealing, public post-retirement release, real Bittensor commitment
  anchor, inclusion-block pointer, and exact archive read are wired. No real testnet
  anchor or unauthenticated bucket-policy round trip is claimed yet.
- [DONE, needs live validation] Production derives exact epoch ids/close blocks from
  consensus-finalized, archive-proven `SubnetEpochIndex` transitions plus
  `LastEpochBlock`, and exact census from `neurons_at(close_block)`. Testnet must still
  verify SN85 runtime scheduling and archive behavior. The sampling-rate policy remains
  an owner decision in `the project design record` §8.
- [DONE] Schema-v14 competition evidence commits one executable baseline, exact artifact
  and provenance bytes/digests, every subject image, and each contender's sealed archive
  bytes/digest plus immutable git commit/tree identity. Result application is bound to
  the exact close-block chain time, and v13 logs are rejected by the mixed-schema fence.
- [DONE] A newly payable podium must match the current eligible census. Carried reward
  provenance may outlive registration; missing podium shares are sent to the canonical
  sink rather than reassigned.
