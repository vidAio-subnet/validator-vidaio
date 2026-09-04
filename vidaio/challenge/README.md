# vidaio.challenge — degrade→restore challenge factory: DAG, commit-reveal, pool, scheduler

Challenges are made by taking a pristine held-out clip, corrupting it with a
seed-determined procedural degradation DAG, and asking miners to restore it — the
task type is public, the parameters and seeds are private (spec design spec §18). This
module is pure logic + SQLite persistence: ffmpeg is **never executed here** —
operators and the ingest contract emit command *plans* for an external executor
(the challenge service, see
the owner-operated service, executes them; the
[vidaio/scoring](../scoring/README.md) engine scores the restorations).

## What it does

**Versioned procedural DAG** (`dag.py`) — all randomness flows through one injected
`random.Random`; same seed + `dag_version` produce a byte-identical canonical JSON,
digest, and ffmpeg plan. Operators are grouped into pipeline stages modeling how
real footage degrades — `capture` (Exposure, MotionBlur, GaussianBlur, Noise) →
`edit` (ToneShift, ColorPipeline, ArtifactMask) → `delivery` (Downscale,
CodecCompress) — with fixed stage order, rng-shuffled order within a stage, and two
hard delivery constraints: CodecCompress (when present) is always final, so
Downscale always precedes it. DAG v7's launch rules are explicit single-op pools:
compression gets a minimally-lossy all-intra H.264 input, and upscaling gets a
2x downscale-only input. Broader operators remain registered only so historical
documents deserialize; re-enabling one requires real-media production-gate
calibration plus a version bump. `to_ffmpeg_plan` chains commands through
lossless FFV1/MKV intermediates when a historical DAG has multiple stages.

`DAG_VERSION` history (the structure is versioned in code, **not** config-tunable,
so a config edit can never silently change what a digest means):

1. initial structure (continuous Downscale factor on every track);
2. upscaling-track Downscale restricted to the **discrete factors
   `UPSCALE_FACTORS = (2, 4)`** — exactly the keys of scoring's `file_size_caps`,
   so the file-size gate is always enforceable (compression keeps the continuous
   range);
3. **FrameDrop excluded from every track's pool** (P1 scope cut, kept in the
   registry so historical documents deserialize). Why: FrameDrop renumbers the
   surviving frames at the source frame rate, so the degraded clip carries only
   `keep/cycle` of the reference's frames and duration; an honest restoration
   necessarily has the same shortened temporal shape, so the validity gates
   (FrameCountGate, `STREAM_DURATION_MISMATCH`) zero every honest miner.
   fps-normalization cannot repair it structurally — the durations genuinely
   differ, and for `keep ≥ 2` the kept frames are non-uniform in source time while
   `fps=` resamples uniformly, so full-reference metrics (VMAF/PieAPP) become
   temporally undefined, not just gated. Reintroduction requires a
   temporal-alignment-aware scoring path and a DAG_VERSION bump;
4. **ArtifactMask actually masks.** Its `drawbox` realization put `t` in the x/y
   expressions meaning "time", but drawbox reads `t` there as the box
   **thickness** — and the same expression set `t=fill`, so every box was
   positioned at the fill thickness, far off frame: the operator was a **silent
   no-op** (verified byte-identical by frame-md5 for every sampled velocity
   except exactly 0.0). Rebuilt on split → lutyuv-darken → moving crop → overlay
   (filters that *do* have a real time variable), chroma-safe (even-offset snap
   for 4:2:0) and pix_fmt-preserving. Sampled parameters and operator pools are
   unchanged — only the realization — but a version-3 document now renders
   differently, so the version (and every digest) moved.
5. Challenge commitments fence the scorer contract in which the VMAF primary/NEG
   anti-gaming pair moved to the miner-input basis; operator sampling stayed v4.
6. **Launch-winnable task pools and consistent perceptual basis.** Compression
   excludes GaussianBlur and Downscale: empirical calibration showed these severe
   draws erased enough detail to put honest outputs below pristine-reference VMAF
   eligibility irrespective of compression merit. Upscaling retains them as its
   explicit restoration task. Tone/grayscale/chroma checks now compare candidate
   output to the canonical miner input, so DAG-applied tone/color changes cannot
   falsely zero an honest miner. The scorer pipeline identity moves independently
   too, preventing a v5 commitment from silently entering v6 semantics.
7. **Production-calibrated launch pools.** Compression is codec-only H.264
   4:2:0/8-bit at private CRF 8/10/12 with an all-intra GOP; the all-intra
   representation leaves a normal inter-frame CRF-22 baseline enough headroom
   to clear the strict `<0.80` size gate even on low-entropy clips. Upscaling is
   downscale-only and live generation is restricted to
   `LAUNCH_UPSCALE_FACTORS = (2,)`, because real-media calibration found honest
   4x draws below the production floor. `UPSCALE_FACTORS = (2, 4)` remains the
   protocol/scorer surface for historical competition items and future v8+
   calibration.

**Seeds and key derivation** (`dag.py` + `scheduler.py`) — private seeds MUST come
from a CSPRNG (`secrets.randbits(256)` or better); `make_challenge` rejects
anything below `MIN_SEED_BITS = 128` with `WeakSeedError` (config may only raise
the floor, `ge=128`). The Mersenne Twister is never seeded with the bare seed:
`dag_rng_from_seed` seeds it from `sha256(b"dag" || seed_bytes)`, and every public
derivation uses a *different* sha256 domain tag (`challenge_id` comes from
`sha256(b"challenge-id" || seed_bytes || asset_id)`), so no raw MT output tied to
the bare seed is ever observable — a miner cannot brute-force public dispatch
material back to the private DAG stream.

**Commit-before-dispatch** (`commitment.py` + migrations) — the validator commits
`sha256({asset_id, dag_digest, seed, scorer_version})` (canonical JSON preimage)
BEFORE dispatching, so it can never cherry-pick a favorable corruption after seeing
miner outputs. This is **schema, not convention**:
- `challenges.commit_hash` is a NOT NULL FK to `challenge_commitments` — a
  challenge row cannot exist before its commitment row;
- a BEFORE INSERT trigger (`challenges_match_commitment`) forces the challenge's
  `(asset_id, dag_digest)` to equal the commitment's — a challenge can never claim
  a commitment binding different private material;
- `commit_hash` is UNIQUE on challenges — one challenge per commitment, no reuse;
- a BEFORE UPDATE trigger (`challenges_identity_immutable`) freezes every identity
  column after insert; only the resolution lifecycle (`status`, `resolved_at`) may
  change.
`record_challenge` additionally integrity-checks the in-memory object first
(`ChallengeIntegrityError`: the DAG must re-hash to the committed digest, the
commitment must bind this asset, the commit hash must hash its own preimage) — an
inconsistent object never persists, not even its commitment row.

**Reveal gating** — `reveal_commitment` is allowed only once the clean asset is
**retired AND** no challenge on that asset is still `dispatched`
(`RevealBeforeRetireError` / `RevealBeforeResolutionError`); it is idempotent and
keeps the first reveal timestamp. `verify_reveal` re-hashes the preimage;
`verify_reveal_deep` additionally rebuilds the DAG from the revealed seed via the
sanctioned derived-key path and requires the rebuilt canonical digest to match —
the check the audit layer injects to prove the corruption was seed-determined
rather than hand-picked and merely hashed.

**Content pool** (`pool.py` + migrations) — lifecycle
`ingesting → fresh → in_use → (fresh …) → retired`:
- assets are registered as `ingesting` and become `fresh` (the sole checkoutable
  status) only when every planned ingest step (fetch, transcode+metadata-strip,
  segment) is confirmed via `confirm_ingest_step`; the whole confirm runs in one
  `BEGIN IMMEDIATE` transaction and duplicate confirms are rejected inside SQLite
  itself (partial UNIQUE index on the confirmation events);
- `checkout_asset` issues only fresh challenge-split assets (tag preference is a
  4:1 rng weighting) and can be restricted to an exact prevalidated eligible-ID
  set, with an atomic guarded-UPDATE claim so concurrent checkouts can never
  double-issue a single-use asset. The executable service derives that set from
  the append-only per-clip name/SHA-256/duration manifest and current files;
- `release_asset` retires at `retire_after_uses` (default 1: single-use);
  `retire_asset` is the admin force path (leak suspicion, license withdrawal) —
  it requires `force=True` while challenges are still dispatched, and even then
  reveal stays blocked until resolution;
- the **provenance log is append-only at the DB layer** (UPDATE/DELETE abort via
  triggers) — the rights/derivation trail of every asset;
- leakage controls: splits (`challenge` vs `holdout`) are deterministic per
  **source group** (`sha256(source_key || split_salt)` over the configured
  `split_key_fields`, default creator+source), never per clip, so no clip of a
  holdout source ever circulates; `check_near_duplicate` screens perceptual
  fingerprints against known public corpora (`FingerprintIndex` seam) before an
  asset enters the pool.

**Scheduler / ingest-lite** (`scheduler.py`) — `make_challenge(track, asset, seed,
scorer_version)` produces the Challenge (private DAG + commitment + miner-facing
`DispatchPayload`); `register_asset` is the pure backend half of the
content-ingestion admin (admission, split assignment, pool insert, provenance
entries, and *plans* for fetch/yt-dlp, pristine-FFV1-transcode with
`-map_metadata -1` strip, and segmentation).

**Leak probes** — `DispatchPayload` carries exactly `{challenge_id, task_type,
input_ref}` (frozen model); `_assert_payload_clean` structurally rejects unexpected
fields and substring-probes the serialized payload for every private value (seed,
asset id, clean content digest, source URL, dag digest, DAG JSON) ≥16 chars,
raising `PayloadLeakError` — every new payload field must survive this probe.

## Design & decisions

- The philosophy is spec design spec §18 verbatim: degradation must be a versioned
  procedural DAG, not a preset list — randomized operator order/kernels/subpixel
  phase/codec settings/noise/tone/masks, with task type public and parameters +
  seeds private. Commit-reveal is the §18 "validator fairness" mechanism.
- Structure-in-code versioning ties every digest to `DAG_VERSION`; the operator →
  ffmpeg mapping table in the `dag.py` docstring is the reference for
  reimplementers.
- The v3 (FrameDrop) and v4 (ArtifactMask drawbox-`t` bug) entries are honest
  records of measured behavior — v4 in particular came out of frame-md5
  verification that the old realization degraded nothing, and the fix is pinned by
  a snapshot test plus `test_artifact_mask_never_uses_drawbox_again`.
- All timestamps are caller-supplied strings and all randomness is injected —
  nothing in this module reads a clock or global rng, which is what makes
  challenge production reproducible and testable.
- Binding decisions: the project design record integrity invariants
  (no substituted material, everything recomputable) drive the commitment schema;
  the `min_seed_bits` floor and derived-key rng are review-hardening driven, as is
  the SQL-level double-confirm rejection.

## Public API (`vidaio/challenge/__init__.py`)

DAG
- `DAG_VERSION`, `OPERATOR_REGISTRY`, `TRACK_RULES`, `UPSCALE_FACTORS` — the
  versioned structure (registry maps op name → class; rules per track).
- `DegradationDag`, `DegradationOp` — frozen models; `canonical_json` /
  `canonical_digest` on the DAG.
- `build_dag(task_type, rng, dag_version=…)` — sample a DAG from a private rng.
- `dag_rng_from_seed(seed)` / `seed_to_bytes(seed)` — the only sanctioned
  seed→rng derivation.
- `to_ffmpeg_plan(dag, input, output)` — ordered argv plans (pure builder).
- `canonical_json_dumps(payload)` — the shared canonical-JSON helper.

Commitment
- `ChallengeCommitment` (`create`, `compute_hash`, `preimage_payload`,
  `preimage_bytes`), `RevealedCommitment`.
- `record_commitment`, `reveal_commitment`, `verify_reveal`, `verify_reveal_deep`.
- `RevealBeforeRetireError`, `RevealBeforeResolutionError`.

Pool
- `Asset` (frozen; defaults to `ingesting`), `FingerprintIndex` /
  `StaticFingerprintIndex`.
- `add_asset`, `get_asset`, `checkout_asset`, `release_asset`, `retire_asset`.
- `append_provenance`, `provenance_log`, `assign_split`, `source_key`,
  `check_near_duplicate`.
- `NoFreshAssetError`, `NearDuplicateError`, `UnresolvedChallengeError`.

Scheduler
- `make_challenge`, `record_challenge`, `resolve_challenge` — produce / persist /
  terminate (`resolved` | `expired`) a challenge.
- `register_asset`, `confirm_ingest_step`, `IngestResult` — ingest-lite contract.
- `Challenge`, `DispatchPayload`, `MIN_SEED_BITS`.
- `WeakSeedError`, `PayloadLeakError`, `ChallengeIntegrityError`.

Also exported: `ChallengeConfig` (section `challenge`) and `MIGRATIONS_DIR`.

## Data & invariants

Migrations (`vidaio/challenge/migrations/`, applied via
`vidaio.core.apply_migrations`; designed to co-apply with the audit module's
ledger on one connection — table names are namespaced accordingly):

- `0001_challenge.sql` — `assets` (status/split CHECKs, UNIQUE content_digest),
  append-only `provenance_log` (no-update/no-delete triggers + the partial UNIQUE
  ingest-confirmation index), `challenge_commitments` (seed stored as TEXT so
  arbitrary-precision ints round-trip exactly), and the initial `challenges`.
- `0002_challenge_binding.sql` — rebuilds `challenges` with the commit-binding
  trigger, UNIQUE `commit_hash`, the `dispatched → resolved | expired` lifecycle,
  and the identity-immutability trigger.
- `0003_commitment_dispatch_binding.sql` through
  `0005_external_commitment_anchor.sql` bind track/order, allocate monotonic
  dispatch order, and persist finalized-chain anchor receipts.
- `0006_dag_version.sql` persists/backfills `challenges.dag_version`, indexes it
  for recovery/operations, and prevents the scalar from diverging from committed
  `dag_json` or changing after dispatch.

Invariants a maintainer must not break: commitment-before-challenge (FK),
challenge↔commitment agreement (trigger), one-challenge-per-commitment (UNIQUE),
identity immutability (trigger), append-only provenance, ingest-confirm-once
(partial UNIQUE index), only-fresh-checkoutable, holdout-never-issued, and
reveal-only-after-retire-and-resolve (enforced in `reveal_commitment`).

## How to test

```
.venv/bin/pytest tests/challenge
```

Notable tests:
- `test_dag.py` — same seed ⇒ identical digest and plan; stage/order constraints;
  discrete upscale factors matching scoring's caps
  (`test_upscale_factors_match_scoring_caps`); the derived-key property
  (`test_dag_rng_is_derived_not_bare_seed`); FrameDrop excluded from every pool
  and never sampled; full ffmpeg-plan snapshots for a fixed seed including the
  ArtifactMask graph, plus `test_artifact_mask_never_uses_drawbox_again`.
- `test_commitment.py` — round trip, reveal-before-retire raising, tampered
  reveals failing, idempotent reveal keeping the first timestamp, and the deep
  check accepting seed-generated DAGs while rejecting hand-picked ones.
- `test_pool.py` — group-deterministic splits (salt moves them, clip fields do
  not), retire-after-use lifecycle, holdout/ingesting never issued, atomic
  checkout claims under race, append-only provenance enforced in DB, forced
  retirement still blocking reveal.
- `test_scheduler.py` — deterministic `make_challenge`, weak seeds rejected,
  challenge_id never MT output, the leak probes
  (`test_dispatch_payload_contains_no_private_material`), the full ingest flow
  with crash-rollback proofs (`test_confirm_crash_between_append_and_flip_rolls_
  back_fully`), SQL-layer duplicate-confirm rejection, commitment-first
  persistence, cross-commitment mismatch rejection, identity-column immutability,
  and commitment non-reuse.
- `test_migrations.py` — clean/once application and challenge+audit migrations
  co-applying on one database.

## How to change safely

- **Any change to DAG structure or realization is a `DAG_VERSION` bump** — new
  operator, changed ranges, changed stage rules, changed filter graph (v4 is the
  precedent: same parameters, different rendering, version moved). Old versions'
  documents must keep deserializing (keep registry entries), but `build_dag` only
  supports the current version.
- **Never change** `canonical_json_dumps`, `seed_to_bytes`,
  `dag_rng_from_seed`'s domain tag, or `ChallengeCommitment.preimage_payload`:
  they define every recorded digest and commitment. The preimage bytes are also
  the exact content of the audit store's DAG_REVEAL artifact
  ([vidaio/audit/bundle.py](../audit/bundle.py)) — publishing anything else would
  never match during audit recompute.
- Keep `UPSCALE_FACTORS` and scoring's `file_size_caps` keys in lockstep (a
  cross-module test pins it). `LAUNCH_UPSCALE_FACTORS` must remain a subset and
  may expand only with calibration plus a DAG version bump.
- New `DispatchPayload` fields must be added to `_ALLOWED_PAYLOAD_KEYS` *and*
  survive the leak probe — think before exposing anything derived from private
  material.
- Schema changes are new migration files only (core's runner applies once);
  respect the co-application constraint with the audit migrations (shared DB,
  namespaced tables).
- `min_seed_bits` can be raised, never lowered (config floor `ge=128`).

## Status & gaps

- [DONE] DAG (v7), commitments + reveal gating, pool lifecycle + provenance,
  scheduler/ingest-lite, leak probes — implemented and tested.
- [DONE IN CHALLENGE SERVICE] The pure `FingerprintIndex` seam retains its exact-
  match test fake; production ingest uses `CpuVideoPhash` plus a digest-pinned
  public-corpus Hamming-distance index before any asset admission.
- [NOT BUILT] Plan execution: fetch/transcode/segment and the DAG's ffmpeg plans
  are executed by the challenge service, not here; this module only emits argv
  plans and records confirmations.
- [NOT BUILT] FrameDrop stays excluded until a temporal-alignment-aware scoring
  path exists (documented in `dag.py`; would be a DAG_VERSION bump).
- Known limitation: `checkout_asset` loads all fresh assets per call (fine at
  current pool sizes); tag preference is a weighting, not a guarantee.
