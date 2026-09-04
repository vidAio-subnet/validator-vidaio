# vidaio.audit — content-addressed store, bundles, commitments, independent recompute

The honesty backbone (spec design spec §08/§14/§15): every scored metric must be
independently recomputable by third parties from what this module preserves —
inputs, outputs, manifests, score packets, weight vectors — pinned by digests that
are committed before dispatch/finalization and anchored through the chain adapter, then published after
evaluation. Config section: `audit` (`AuditConfig`). Producers are the competition
orchestrator ([vidaio/competition](../competition/README.md)) and the validator
side; the reveal verifier it injects comes from
[vidaio/challenge](../challenge/README.md).

## What it does

**Content-addressed store** (`store.py`) — artifacts are keyed by
`(kind, sha256(plaintext))`; the sharded backend key is
`<kind>/<aa>/<bb>/<digest>`. Kinds: `challenge_input`, `miner_output`,
`reference_original` (the sealed holdout), `manifest`, `score_packet`,
`weight_vector`, `dag_reveal`, and — added by the two-API rework — `epoch_log`
(the finalized [`EpochLog`](../epoch/README.md) bytes) and `audit_bundle` (a
persisted `AuditBundle` so an auditor can resolve one by digest). Properties
enforced:
- **write-once**: putting bytes whose digest already exists is a no-op returning
  the same ref — artifacts can never be replaced in place;
- **verify-on-read**: `get()` recomputes the sha256 (and size) and raises
  `IntegrityError` on any mismatch — silent tampering/corruption cannot pass;
- **path derivation from (kind, digest) only**: `ArtifactRef.backend_key` is
  informational and never trusted for lookup, so a doctored ref cannot address
  outside the store layout;
- atomic publication for local content-addressed dev writes; real S3/S3-compatible
  writes use `PutObject` with `If-None-Match: *`. A 412
  precondition loser becomes a typed conflict and never overwrites the first writer;
  an exact byte-for-byte repeat remains idempotent at the store layer.

**The object store is now the shared content plane.** The two-API rework makes
this store the single content layer BOTH the Scoring Authority and the
validator/auditor read: the authority's private `make_store()` view writes each epoch's
`EPOCH_LOG` and its `AUDIT_BUNDLE`s here; a thin weight-setter uses the sealed-kind-disabled
writer described below; and its auditor uses unsigned, read-only `make_public_store()` to
mirror epoch-log bytes and resolve released bundles straight back by digest
(`persist_bundle` / `StoredBundleSource` in
[`vidaio/auditor`](../auditor/README.md); the store digest of a persisted bundle
IS its `bundle_digest()`). No separate bundle store exists.

**The `_FINALIZED` half-write set convention** (`_SetConventionMixin`, a
production-proven guard) — an epoch's artifacts are written under a set prefix
(`finalized/epoch={N}/`) and become readable only once an empty `_FINALIZED`
marker object is written LAST. `is_finalized(prefix)` probes it;
`put_set_member(prefix, name, data, kind)` writes a member (refused after the set
is finalized), `finalize_set(prefix)` writes the marker,
`get_set_member(prefix, name, *, expected_digest=…)` reads a member back (raising
`SetNotFinalizedError` before the marker exists, verifying the digest when given).
`set_member_key` / `finalized_marker_key` name the objects. A reader can therefore
never mirror a half-written epoch — the [finalizer](../authority/README.md) writes
the log member first, the marker last; the [shared-snapshot provider](../weightsetter/README.md)
gates its mirror on `is_finalized`. The convention is identical across LocalFs / S3
/ Hippius (it lives in the mixin once).

`LocalFsStore` is the report/dev backend. **`S3Store`** is the production content
layer: `_RealS3Transport` uses boto3 for AWS or any S3-compatible endpoint, including
streaming non-sealed media uploads. Every bytes/file upload is a conditional create
(`If-None-Match: *`), including finalized members and markers. `HippiusStore` uses the same transport with the
Hippius S3 gateway endpoint and path-style SigV4. All three share content addressing,
verify-on-read, write-once, release, and `_FINALIZED` behavior. The decision logic is
unit-tested against an in-memory transport, and the real transport's request/conflict
contract is tested with an injected client. The chosen provider must still pass the
live first-writer collision probe and anonymous/private policy checks; no live bucket is
claimed here. Credentials are never in config — only the *names* of the environment
variables the transport reads. `make_store` selects `local` | `s3` | `hippius`.
`make_public_store` selects the same locator without signing requests, refuses every
write/release operation, and never reads the credential or holdout-key environment
variables. That is the production view used by the standalone auditor. The shared-source
weight-setter instead uses `make_unsealed_writer_store()`: scoped signed credentials (or
workload IAM) may read public evidence and write its `weight_vector`/`manifest`, while the
store rejects every sealed-kind operation before transport and never loads the AES key.

**Sealed holdout and public release** — `reference_original` artifacts pass through an
`Envelope`. Production's `AesGcmEnvelope` uses a 32-byte environment-provided AES-256-GCM
key; authentication failure is an integrity error. Digests always cover the **plaintext**, so
pre-enrollment commitments made while sealed remain verifiable against the
post-retirement release. `make_store()` **refuses to build a store whose holdout
envelope is the no-op `PassthroughEnvelope`** unless
`audit.allow_plaintext_holdout: true` is set explicitly (dev/test opt-in) —
otherwise the "sealed" ground truth would be readable by anyone with storage
access mid-competition. After terminal challenge resolution, `release(ref)` verifies and
publishes the plaintext at `released/<canonical-key>`; the authority's inference-validator
resolution path calls it for retired references. A due upscaling competition likewise
releases every manifest-bound pristine reference before it can transition from
AWAITING_END_TIME to COMPLETED; a failed release halts and leaves it non-earning. A private store view may prefer the
released copy, while a public view resolves a sealed ref **only** at
`released/reference_original/...` and never falls back to the canonical key. Both verify
the same plaintext digest and size.

Third-party recomputation requires more than the released reference. A public bucket
reader needs the non-secret evidence prefixes `challenge_input/`, `miner_output/`,
`manifest/`, `score_packet/`, `weight_vector/`, `dag_reveal/`, `epoch_log/`,
`audit_bundle/`, and `finalized/` (as present), plus
`released/reference_original/`. Deny the canonical `reference_original/` prefix and all
other keys. No unauthenticated live-bucket policy test has been run yet.

**Bounded streaming reads.** Audit media is materialized to digest/size-verified temporary
files rather than assembled as one Python byte string: challenge input and reference are
each capped at 2 GiB, and miner output at 4 GiB. Audit metadata reads are capped at 16 MiB,
one serialized bundle at 1 MiB, and a mirrored epoch log at 64 MiB. The AES-GCM archive and
post-retirement release paths stream through disk as well. These are memory-safety bounds,
not storage-budget promises: the auditor host still needs enough local scratch for the
bounded media plus the scoring worker's canonicalization intermediates.

**Bundles** (`bundle.py`) — `AuditBundle` binds one challenge item's artifacts
(challenge_input, miner_output, manifest, score_packet, optionally
reference_original + dag_reveal), the commitment hash, the miner hotkey, and
scorer/backend version pins under a single stable `bundle_digest()` (sha256 over
canonical JSON). Stage rules are validated on construction:
- `PRE_REVEAL`: the reference original and DAG reveal **must be absent** —
  publishing either mid-competition would leak the holdout / the private seeds;
- `COMPETITION_SEALED`: an upscaling competition bundle carries a sealed
  `reference_original`, no inference DAG, and a `CompetitionItemBinding` whose
  reference/input/factor preimage re-hashes to the indexed commitment in the canonical
  manifest. It becomes publicly recomputable only after completion releases the ref;
- `POST_RETIREMENT`: everything must be present — the only stage that can pass
  full inference verification.
Slot kinds are checked (a `miner_output` ref cannot sit in the manifest slot).
The DAG_REVEAL artifact must be the challenge commitment **preimage JSON**
(`ChallengeCommitment.preimage_bytes()`), not the raw DAG JSON — anything else
would never match the commitment during recompute.

**Commitment payloads + merkle + ledger** (`commitments.py`) —
- *Before enrollment*: `CompetitionCommitment` pins {manifest digest, archived-baseline
  version/artifact/provenance/tree/image identities, dataset-selection seed commitment, reward-param
  digest}; the chain payload is the domain-tagged
  `"vidaio.commitment.v1:<kind>:<root>"` bytes (asserted ≤128 bytes) with
  `root = sha256(canonical_json)`; the JSON itself is kept off-chain as a store
  artifact so the root is always openable.
- *After evaluation*: `PublicationRecord` pins {merkle root of all score-packet
  digests, weight-vector digest} with inclusion proofs.
- `pin_git_sha(sha)` is THE canonical adapter from a git object id (sha1 or
  sha256 repo, any case) to the sha256 hex these commitments require — defined
  once so every caller produces the same bytes.
- **Merkle construction (documented for reimplementers)**: leaves are the sha256
  digests as 32 raw bytes, sorted ascending as bytes, duplicates kept (root is
  order-independent); `leaf_hash = sha256(0x00 || leaf)`,
  `node_hash = sha256(0x01 || left || right)` (domain separation prevents
  leaf/node second-preimage splices); an odd node at any level is promoted
  unchanged. `merkle_root` / `merkle_proof` / `verify_merkle_proof`.
- **CommitmentLedger** (SQLite, `migrations/0001_commitment_ledger.sql`):
  append-only rows (UPDATE/DELETE abort via triggers); `record()` **re-derives
  the payload's internal relationships** (root over canonical JSON, domain-tagged
  bytes, allowed kind) before insert — the ledger never trusts the caller
  (`LedgerIntegrityError`) — and writes row + initial `pending_chain` status in
  one `BEGIN IMMEDIATE` transaction. Status advances **forward only, one step at
  a time** (`pending_chain → anchored → published`; skipping `anchored` is
  rejected) as appended event rows, never edits. Timestamps are compared **as
  instants**: caller values must be timezone-aware ISO-8601 (naive rejected),
  are normalized to canonical UTC before storage, and monotonicity is checked
  after UTC-normalization so an ISO offset cannot disguise a backdate
  (`'09:00+05:00'` is 04:00Z and is rejected after `'08:00+00:00'`). Every one of
  these invariants is *also* enforced in-database by triggers (comparisons via
  `julianday()`, which parses offsets), so direct SQL cannot backdate, regress,
  skip, or restart a history either.

**Independent recompute** (`recompute.py`) — `verify_bundle(bundle, store,
recomputer, …)` runs every audit check and reports each pass/fail with a stable
failure code:

1. stage must be POST_RETIREMENT for inference, PRE_REVEAL for normalized-reference
   compression competitions, or COMPETITION_SEALED for manifest-bound upscaling
   competitions (`INCOMPLETE_BUNDLE` otherwise);
2. `bundle_digest()` vs the published/anchored digest (`DIGEST_MISMATCH`) — edited
   metadata (timestamps, versions, refs) fails here;
3. every referenced artifact fetched via the store's verify-on-read
   (`ARTIFACT_MISSING`/`ARTIFACT_CORRUPT`);
4. commit-reveal: revealed DAG bytes must hash to the committed hash
   (`COMMITMENT_MISMATCH`), and the injected `reveal_verifier` (the challenge
   module's deep check) must confirm the bytes regenerate the committed DAG
   (`REVEAL_INVALID`; a verifier crash is itself a finding);
5. **full-shape strict packet parsing**: `ScorePacketShape` is defined locally
   (audit never imports scoring) as the REQUIRED authoritative field set
   mirroring `ItemScore` — extras ignored so scoring can grow the packet, but a
   missing key is `MALFORMED_SCORE_PACKET`, as is a non-finite or out-of-[0,1]
   top-level score, *at parse time, before any recompute*. Then:
   - **identity**: challenge_id, item_id and (when the bundle pins one)
     miner_hotkey must match the bundle (`IDENTITY_MISMATCH` — a packet minted
     for another challenge/item/miner);
   - **scorer pinning**: packet `scorer_version` vs bundle
     (`SCORER_VERSION_MISMATCH`);
   - **backend pinning**: the bundle and packet must carry the same complete
     backend-version map (`BACKEND_VERSION_MISMATCH`; omissions and packet-only
     extras both fail);
   - **internal consistency**: gates-first (`gate_passed=False` ⇒ score 0.0; a
     passing packet may not carry violations or a null breakdown) —
     `PACKET_INCONSISTENT`;
6. merkle inclusion of the score packet in the published set
   (`MERKLE_EXCLUSION` — packets injected outside the committed set fail here);
7. recompute via the `ScoreRecomputer` Protocol and compare: the recomputer must
   run the pinned scorer version and exact backend-version map, the metric *set*
   must match
   (`METRIC_SET_MISMATCH` — injected extra metrics fail), each metric compares
   under per-metric absolute tolerances (`DEFAULT_TOLERANCES = {"vmaf": 0.05}`;
   anything unlisted, e.g. byte ratios, compares exactly at 0.0), **the top-level
   score itself is compared** (`SCORE_MISMATCH` — honest metrics with an edited
   top-level score fail here, not slip through), and the recomputed gate outcome
   must agree (a doctored `gate_passed` flag fails). The only exception is narrow
   **numeric-boundary hysteresis**: when the independently recomputed raw metric
   crosses a whitelisted gate/formula boundary by no more than that metric's
   existing tolerance, the verifier honors the committed outcome. This covers the
   discontinuous compression VMAF threshold as well as VMAF floor/model-delta and
   CPU perceptual boundaries. It does not widen tolerances: a beyond-band metric,
   structural/identity gate, missing numeric limit, or unrelated violation remains
   `SCORE_MISMATCH`.

   The release tolerance table is an acceptance ceiling, not an operator tuning
   surface. `verify_bundle(..., tolerances=...)` may only make a listed tolerance
   smaller (or leave it equal); it rejects wider, negative/non-finite, and positive
   unlisted-metric values before reading evidence. In particular, PieAPP stays at
   the strict shipped `1e-5` ceiling and no deployment can widen or round around a
   runtime-parity defect.

Schema-v14 earning competition evidence uses this exact verifier for every
subject-by-item packet/bundle pair, including the non-earning archived-baseline rerun.
The auditor then derives each subject's packet-score mean, stable machine ordering,
baseline margin, global PODIUM/CROWN reward window, and final vector; human review and
stored ranking fields never enter the economic seam. The epoch input commits the exact
baseline archive/provenance/image identity, each contender's sealed source archive and
git identity, the full evaluation matrix, and the chain-derived application time. A
CROWN result additionally requires the winning source archive to be publicly readable.
`is_released(ref)` is a bounded content/size-verified read of the released plaintext,
not an existence/HEAD claim. Through `make_public_store()` that read is explicitly
unsigned, so authority finalization proves the exact CROWN archive is anonymously
retrievable before it can write `_FINALIZED`; a private writer's successful release
alone is insufficient. The canonical `submission_archive/` and `reference_original/`
prefixes remain private, and PODIUM/non-winning contender source is never released.

**Strictness**: full third-party verification needs the external anchors
(published bundle digest, published merkle root, the deep reveal verifier). When
one is absent the check records an explicit SKIPPED result — and under
`strict=True` (the default) **a skipped anchor is a failure** with code
`MISSING_ANCHOR`/`REVEAL_UNVERIFIED`; `strict=False` is for partial/diagnostic
audits only.

## Design & decisions

- Implements the design spec §15 review fixes "anchor commitments on chain" and "archive
  the bytes": digests-first design so a third party holding only on-chain roots
  can fetch exact bytes and recompute — the
  the project design record integrity invariant ("every scored metric
  must be independently recomputable from the audit store") is this module's
  contract. Chain submission itself is a later-phase adapter; this module
  produces exact payload bytes and ledgers them locally.
- `canonical.py` pins the determinism contract (UTF-8, sorted keys, no
  whitespace, non-ASCII preserved, NaN/Infinity rejected); every digest in the
  module is computed over bytes produced there, versioned through
  `COMMITMENT_DOMAIN = "vidaio.commitment.v1"`.
- review-driven hardening that is structural here: the plaintext-holdout refusal in
  `make_store`, the ledger's payload re-derivation and instant-normalized
  timestamp triggers, the single-step status advancement, the top-level-score
  comparison (a review probe showed honest metrics + tampered top-level score passing
  before), and strict-mode skipped-anchor failures.
- Audit deliberately does not import scoring: `ScorePacketShape` is an
  independent, minimal re-statement of the packet contract, so a compromised or
  drifted scoring module cannot weaken verification by redefinition.

## Public API (`vidaio/audit/__init__.py`)

Store
- `AuditStore` (Protocol), `LocalFsStore`, `S3Store`, `HippiusStore`,
  `make_store(config, envelope=None)` — backends + the guarded factory.
- `ArtifactKind` (incl. `EPOCH_LOG`, `AUDIT_BUNDLE`), `ArtifactRef`,
  `SEALED_KINDS`, `Envelope`, `PassthroughEnvelope`, `AesGcmEnvelope`,
  `released_backend_key`, `release()` / `is_released()`, `IntegrityError`, and
  `NotConfiguredError`.
- The `_FINALIZED` set convention: `backend_key`, `set_member_key`
  (`put_set_member` / `finalize_set` / `is_finalized` / `get_set_member` on every
  backend), `FINALIZED_MARKER`, `SetNotFinalizedError`, `SetAlreadyFinalizedError`.

Bundle
- `AuditBundle`, `LifecycleStage`, `build_bundle(...)` — stage-validated bundles
  with `bundle_digest()`.

Commitments
- `CompetitionCommitment`, `PublicationRecord`, `CommitmentPayload`,
  `build_competition_commitment`, `build_publication_record`,
  `COMMITMENT_DOMAIN`, `ALLOWED_COMMITMENT_KINDS`, `pin_git_sha`.
- `merkle_root`, `merkle_proof`, `verify_merkle_proof`.
- `CommitmentLedger` (`record`/`advance`/`current_status`/`history`/`get`),
  `CommitmentStatus`, `LedgerIntegrityError`.

Recompute
- `verify_bundle(...)` → `VerificationReport` (`passed`, `failures()`, `skips()`),
  `CheckResult`.
- `ScoreRecomputer` (Protocol), `RecomputedScore`, `StaticRecomputer` (test
  double), `ScorePacketShape`, `DEFAULT_TOLERANCES`.
- The failure-code vocabulary (stable strings for dashboards/alerts/tests):
  `SCORE_MISMATCH`, `ARTIFACT_CORRUPT`, `ARTIFACT_MISSING`,
  `COMMITMENT_MISMATCH`, `DIGEST_MISMATCH`, `IDENTITY_MISMATCH`,
  `INCOMPLETE_BUNDLE`, `MALFORMED_SCORE_PACKET`, `MERKLE_EXCLUSION`,
  `METRIC_SET_MISMATCH`, `MISSING_ANCHOR`, `PACKET_INCONSISTENT`,
  `RECOMPUTE_ERROR`, `REVEAL_INVALID`, `REVEAL_UNVERIFIED`,
  `SCORER_VERSION_MISMATCH`, `BACKEND_VERSION_MISMATCH`.

Canonical helpers
- `canonical_json_bytes`, `sha256_hex`.

## Data & invariants

`migrations/0001_commitment_ledger.sql` — `commitment_ledger` (kind CHECK,
64-char root CHECK) + `commitment_ledger_status`, both append-only via triggers;
the `commitment_ledger_status_forward_only` trigger enforces first-status
`pending_chain`, strictly-forward single-step advancement, no skipping
`anchored`, parseable ISO-8601 timestamps, and instant-monotonicity via
`julianday()`. Tables are namespaced `commitment_ledger*` so the migration can
share a database with the challenge module's own `challenge_commitments` table
(co-application is tested).

Do not break: write-once + verify-on-read in the store; plaintext-digest
addressing under sealing; stage completeness rules on bundles; the merkle
domain-separation constants; the `COMMITMENT_DOMAIN` tag (bump it on any
canonical-JSON contract change); ledger append-only + single-step + instant
monotonicity; and the strict-by-default skip semantics of `verify_bundle`.

## How to test

```
.venv/bin/pytest tests/audit
```

Notable tests:
- `test_store.py` — local and fake-transport round trips, sharded layout, streaming
  uploads, write-once behavior, corruption detection, AES-GCM holdout sealing,
  post-retirement release, S3/Hippius transport configuration, and plaintext refusal.
- `test_bundle.py` — pre-reveal bundles rejecting holdout refs, post-retirement
  completeness, slot-kind mismatches, digest stability (dict-order independent)
  and any metadata change moving the digest.
- `test_commitments.py` — deterministic ≤128-byte payloads, order-independent
  merkle roots, inclusion proofs for every leaf, tampered leaves failing, tampered
  root/payload/canonical-JSON rejected by the ledger, append-only + forward-only
  enforced both in Python and via **direct SQL** (regression, skip,
  offset-disguised backdate, naive/unparseable timestamps, status-before-creation
  all rejected at the trigger level), UTC canonicalization on write, and atomic
  record (row + status together).
- `test_recompute.py` — the honest path passing all checks; the tampered paths each
  failing with their code: tampered metrics (`SCORE_MISMATCH`), the review probe of
  honest metrics + tampered top-level score, wrong recomputer version, packets for
  another challenge/item/miner (`IDENTITY_MISMATCH`), backend-pin conflicts and
  missing pinned backends, gates-first inconsistencies, doctored `gate_passed`
  caught by gate recompute, Infinity/NaN/out-of-range packets malformed at parse,
  missing required keys malformed, injected extra metrics, merkle exclusion,
  tampered bundle metadata, corrupted artifacts, reveal-verifier rejection and
  crash-as-finding, and strict vs non-strict skip semantics.

## How to change safely

- **Never change recorded-digest inputs**: `canonical_json_bytes`, the merkle
  leaf/node domain bytes, `backend_key` sharding, or the payload byte format —
  any such change invalidates recorded digests and must come with a
  `COMMITMENT_DOMAIN` version bump and a migration story for old roots.
- **Failure codes are a stable API** — add codes; never rename or reuse them.
  Every code is re-exported from `__init__.py`; keep new ones exported too.
- New required packet fields go into `ScorePacketShape` only when every historic
  packet carries them; otherwise default them (the `skips` field is the
  precedent: defaulted so older packets parse, carried so newer ones are
  visible).
- New artifact kinds: extend `ArtifactKind`, decide sealing (`SEALED_KINDS`), and
  add a bundle slot with the right stage rule if bundles carry them.
- Ledger schema changes are new migration files; keep the trigger semantics —
  Python-level checks are convenience, the SQL is the authority.
- Tolerances: byte-exact metrics stay at 0.0; only add an epsilon for a metric
  with a measured cross-environment float-accumulation story (VMAF's 0.05 is the
  precedent).

## Status & gaps

- [DONE] LocalFsStore, AES-GCM envelope/release policy, the `_FINALIZED` set convention, the
  `EPOCH_LOG` / `AUDIT_BUNDLE` kinds (the shared content plane), bundles,
  commitments/merkle/ledger, recompute verifier — implemented and tested.
- [DONE, needs live-bucket validation] `S3Store` and `HippiusStore` use the real boto3
  S3-compatible transport with atomic `If-None-Match: *` creates for bytes and files;
  precondition conflicts preserve the existing object. The request/conflict contract is
  covered through an injected client. Provider conditional-write behavior, credentials,
  public-prefix policy, overwrite-denial/Object-Lock equivalent, and lifecycle must still
  be exercised and retained as evidence on the selected test bucket.
- [DONE in composition, needs testnet validation] Epoch and publication anchors are
  submitted/read through `BittensorChainAdapter`; this module remains deliberately
  chain-agnostic.
- [DONE] `AesGcmEnvelope` is the production holdout envelope. Plaintext is explicit
  dev/test opt-in only; terminal resolution publishes a verified released copy.
- [DONE] `RealScoreRecomputer` lives in `vidaio.auditor` and drives the scoring worker's
  CPU pipeline; `StaticRecomputer` remains this module's deterministic test double.
- `AuditConfig.retention_days` is stored config; no pruning job exists (0 =
  retain forever is the default and the intent).
