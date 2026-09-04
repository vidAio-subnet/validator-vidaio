# vidaio.scoring — pure, deterministic, auditable score composition, gating, aggregation

The scoring engine: exact spec formulas, validity gates that run first, fail-closed
handling of anything non-finite, and an audit-grade per-item record. Nothing in this
package performs I/O, shells out, or uses randomness — expensive metric computation
lives behind injectable Protocols (`backends.py`), with the real subprocess-backed
implementations in `backends_real.py`. Spec authority: `the design spec` §02
(formulas), §08 (recomputability), §18 (anti-gaming). The service that drives this
engine over HTTP is the scoring worker (see
[vidaio/scoring_worker](../scoring_worker/README.md)).

## What it does

**Gates first** (`gates.py`) — validity gates run before any formula and are
absolute: any failure forces score 0 with a machine-readable `ReasonCode`
(`ENCODING_NOT_ALLOWED`, `FRAME_COUNT_MISMATCH`, `FILE_SIZE_CAP_EXCEEDED`,
`COMPRESSION_RATE_TOO_HIGH`, `VMAF_BELOW_FLOOR`, `VMAF_BELOW_THRESHOLD`,
`VMAF_MODEL_DELTA_EXCEEDED`, `TONE_MANIPULATION`, `COLOR_GRAYSCALE`,
`CHROMA_UV_MANIPULATION`, the `STREAM_*` codes from stream validation,
`REPLAY_DUPLICATE`, `METRIC_MISSING`, `METRIC_NON_FINITE`,
`UNSUPPORTED_SCALE_FACTOR`). The pipeline collects **all** violations (no
short-circuit) — the full failure picture is part of the audit record. The only
sanctioned way to *not* run a check is an explicit config flag, which records an
informational `GateSkip` (`require_secondary_vmaf=False` on the model-delta gate;
`pipeline_without_perceptual_checks(reason=…)` for the three perceptual gates) —
skips are persisted on the `ItemScore` packet, so a reader can always tell a packet
that passed a check from one that never ran it.

**Fail-closed non-finite handling** (`finite.py` + everywhere) — NaN compares False
against every threshold, so a non-finite metric would silently pass a gate or
compose into a score. Formulas and aggregation raise `ValueError` at the boundary
(`require_finite`); gates translate to `METRIC_MISSING`/`METRIC_NON_FINITE`
violations; `ScoringConfig` rejects non-finite levers at construction (a NaN
`compression_norm` once sailed past a `<= 0` check and composed to score 1.0).

**Compression formula** (`compression.py`, spec constants in `config.py`):

```
rate >= 0.80 (compression_rate_max)   -> 0   (needs >= 1.25x shrink)
vmaf < threshold - 5                  -> 0   (VMAF_BELOW_FLOOR)
vmaf < threshold                      -> 0   (VMAF_BELOW_THRESHOLD — documented near-miss band)
else final = min(1, (0.7*comp + 0.3*vmaf/100) / 1.12)
comp = clamp(1 - rate, 0, 1)
```

The `threshold−5 ≤ vmaf < threshold` band is a documented spec ambiguity: the spec
defines no formula there, so the case list is treated as
exhaustive-with-fallthrough-to-zero (no linear ramp); the two zero cases keep
distinct reason codes for auditability. With default weights the theoretical max is
(0.7+0.3)/1.12 ≈ 0.893; `min(1, ·)` is the spec's clamp for non-default weights.
Bitrate is never a score term — only a gate constraint.

**Upscaling formula** (`upscaling.py`):

```
s_q   = 1 / (1 + max(0, pieapp))                 (documented mapping; monotone decreasing,
                                                  perfect output -> exactly 1.0, no tunables)
s_l   = log(1 + content_length) / log(321)        clamped to 1 (saturates at length 320)
s_pre = 0.5*s_q + 0.5*s_l
final = clamp(0.1 * exp(6.979 * (s_pre - 0.5)), 0, 1)
```

Anchor points: `s_pre = 0.5 → final = 0.1` exactly; `s_pre = 1 → 0.1·e^3.4895 ≈
3.27`, clamped to 1.0. VMAF on this track is only a pass/fail gate
(`vmaf/100 < 0.5 → 0` via `VmafFloorGate`).

**Deterministic PieAPP frame derivation** (`backends.derive_pieapp_start_frame`) —
legacy PieAPP sampled 4 consecutive frames from a *random* start, making live scores
unreproducible. Here the start frame is
`sha256(content_digest || 0x00 || challenge_id) mod usable_frames` — recomputable by
any verifier, unpredictable to a miner before the challenge_id is assigned, with a
NUL join so no (digest, id) pair is ambiguous.

**Aggregation** (`aggregate.py`) — spec §18 worst-decile/bottleneck: the round score
is the mean of the worst `ceil(n · 0.1)` item scores (at least one), so one
excellent item never offsets a failed one (gate-failed items enter as 0.0). Plus
`length_weighted_mean` and the competition aggregate
`final_score = 0.6·quality + 0.25·cost_efficiency + 0.15·length_coverage` with
manifest-injectable convex weights (`AggregateWeights` must partition 1.0).

**Cross-miner dedup** (`dedup.py`, `duplicate_evidence.py`) — economically
attributable replay/collusion detection uses exact `content_digest` equality only.
The `anchor_hash_hotkey/1` order hashes the finalized challenge-anchor block hash
with each miner hotkey, so network arrival cannot choose the winner and miners could
not grind the salt before the block existed. A losing zero carries both signed
receipts, content-addressed output bytes and an exact-digest witness for independent
replay. Perceptual similarity is deliberately non-economic.

**Canonicalization** (`canonicalize.py`) — both reference and candidate are
normalized before any metric (container/colorspace/timebase/pix_fmt must never be a
scoring vector). `build_canonicalization_plan` is a **pure argv builder** (first
video stream only, PTS rebased, CFR, `yuv420p`, `bt709`, timescale 90000);
`plan_digest` (sha256 over the NUL-joined argv) identifies the exact per-item
invocation and `plan_template_digest` (paths replaced by `{input}`/`{output}`
tokens) identifies the *recipe* across items. The per-item digest is stored on every
`ItemScore`. `validate_stream` post-checks dims/pix_fmt/frame-count/duration/PTS
consistency, emitting the shared `ReasonCode`s; `cross_check_decoders` is the
two-decoder hook against untrusted bitstreams.

**ItemScore** (`result.py`) — the audit-grade per-item packet: identity
(item/challenge/track/hotkey/content digest), the bounded outcome, all violations
and skips, the full formula `Breakdown` (every term), raw metric inputs, and
provenance (`scorer_version`, `backend_versions`, `canonicalization_plan_digest`,
`pieapp_start_frame`, `scoring_config_digest` = sha256 of the config in force).
`score` is **bounded by construction**: finite and in [0, 1], enforced by a
validator so an Infinity/NaN packet is unconstructible *and* unparseable
(`from_json` rejects it too); `Breakdown.final` carries the same bound.
`compose_item_score` enforces gates-first structurally: any violation forces 0.0
regardless of the metrics.

## Design & decisions

- **Honest-rebuild boundary** (module docstring): scores come only from measured
  metric inputs — there is no substitution path and no per-hotkey special case
  anywhere in this package. This implements the
  the project design record integrity invariants (one uniform
  scoring path; every metric independently recomputable).
- **Backends behind Protocols** (`backends.py`): `VmafBackend`, `PieAppBackend`,
  `ProbeBackend`, `PerceptualCheckBackend`, `PerceptualHashBackend`.
  `DeterministicFakeBackend` implements them all from supplied mappings (missing
  keys raise — a fake never invents a metric) and is what tests and golden
  dry-runs compose with.
- **Real backends** (`backends_real.py`): argv-only subprocesses (never
  `shell=True`), explicit timeout on every call, children in their own
  session/process group so a timeout or a `MediaProcessScope.cancel()` kills the
  whole group; typed errors carry argv + captured stderr, never a silent default.
  `CanonicalizeExecutor` enforces an optional `max_output_bytes` bound with a
  host-side output watchdog (decode is an expansion; an untrusted file must cost the
  cap, not the disk). VMAF determinism: `n_subsample=1`, `n_threads=1`,
  `pool=mean`, pinned model — no randomness exists to seed.
- **The NEG-model choice** (`backends_real.py` docstring): the secondary model for
  the delta gate is `vmaf_v0.6.1neg` — the "no enhancement gain" variant clips
  exactly the sharpening/contrast tricks that inflate the default model, so a
  large primary-vs-NEG delta is precisely the model-gaming signal the gate hunts.
  The 4k model was rejected as secondary: calibrated for a different viewing
  distance, it legitimately diverges at small resolutions and would make the gate
  noisy instead of adversarial-sensitive.
- **review-review-driven hardening baked in structurally**: config-level finiteness
  (fail closed at construction), the `require_secondary_vmaf` fail-closed default
  with auditable skip, `FileSizeCapGate` failing closed on absent/unsupported
  upscale factors (scoring never trusts that the challenge DAG only issues
  supported factors), and the bounded-by-construction score packets.

## Public API (`vidaio/scoring/__init__.py`)

Config
- `ScoringConfig` — every lever with spec defaults; `vmaf_floor(track)` /
  `vmaf_threshold(track)` helpers. `CompressionWeights`, `AggregateWeights`.
- `TRACK_COMPRESSION`, `TRACK_UPSCALING` — the track name constants.

Gates
- `ReasonCode`, `ValidityViolation`, `GateSkip`, `GateContext`, `Gate`, `GatePipeline`.
- Gate classes: `EncodingGate`, `FrameCountGate`, `FileSizeCapGate`,
  `CompressionRateGate`, `VmafFloorGate`, `VmafModelDeltaGate`,
  `ToneManipulationGate`, `ColorGrayscaleGate`, `ChromaUvGate`, `SkippedGate`.
- `default_pipeline(perceptual_backend)` / `pipeline_without_perceptual_checks(reason=…)`
  and `PERCEPTUAL_GATE_NAMES`.

Formulas
- Compression: `compression_rate`, `compression_score_from_rate`,
  `score_compression` → `CompressionBreakdown`.
- Upscaling: `quality_from_pieapp`, `length_score`, `final_from_pre`,
  `score_upscaling` → `UpscalingBreakdown`.

Aggregation
- `worst_decile_score` / `worst_decile_from_config`, `length_weighted_mean`,
  `competition_final_score`.

Dedup
- `DedupEntry`, `DedupVerdict`, `PerceptualHashComparator`, `dedup_responses`.

Canonicalization
- `build_canonicalization_plan`, `plan_digest`, `plan_template_digest`,
  `validate_stream`, `cross_check_decoders`, `SecondaryDecoderBackend`.

Backends (Protocol seams + fake)
- `MediaInfo`, `PerceptualCheckResult`, `VmafBackend`, `PieAppBackend`,
  `ProbeBackend`, `PerceptualCheckBackend`, `PerceptualHashBackend`,
  `DeterministicFakeBackend`, `derive_pieapp_start_frame`, `usable_frames`.

Result
- `ItemScore`, `compose_item_score`, `config_digest`, `require_finite`.

Real implementations (imported from `vidaio.scoring.backends_real`, deliberately
not re-exported from the package): `FfprobeBackend`, `FfmpegVmafBackend`
(`DEFAULT_VMAF_MODEL` / `SECONDARY_VMAF_MODEL`), `CanonicalizeExecutor`,
`MediaProcessScope` / `use_process_scope`, `use_media_scratch`,
`detect_tool_versions`, the typed error tree (`MediaToolError`,
`MediaToolTimeout`, `CanonicalizationTooLarge`, `MediaWorkCancelled`, …), and the
CPU/CUDA-selectable `PieAppTorchBackend`, deterministic
`CpuPerceptualCheckBackend`, and explicit refusal-only
`UnconfiguredPerceptualCheckBackend`.

## Data & invariants

No SQL — persistence of packets belongs to the callers
([vidaio/competition](../competition/README.md) `record_item_score`,
[vidaio/audit](../audit/README.md) bundles). Pydantic-enforced invariants:

- `ItemScore.score` and `Breakdown.final` are finite and in [0, 1] — enforced on
  construction and on JSON parse; an unbounded score cannot exist as a packet.
- `score == 0.0` whenever `gate_passed == False` (`compose_item_score` structural
  zeroing); the audit layer re-checks this as `PACKET_INCONSISTENT`.
- `ScoringConfig` rejects non-finite levers and nonsense ranges at construction;
  `AggregateWeights` must partition 1.0 (tolerance 1e-9, float representation
  only); `CompressionWeights` deliberately need not sum to 1 (the `min(1,·)`
  clamp covers that).
- `ReasonCode` strings and the metric names inside `ItemScore.metrics` are stable
  identifiers stored in the audit record — renaming one is a breaking change for
  every recorded packet.
- `UPSCALE_FACTORS`-alignment: `file_size_caps` keys {2: 8.0, 4: 20.0} must stay
  in sync with the challenge DAG's discrete upscale factors
  (`tests/challenge/test_dag.py::test_upscale_factors_match_scoring_caps` pins it).

## How to test

```
.venv/bin/pytest tests/scoring
```

Notable tests:
- `test_compression.py` / `test_upscaling.py` — golden mid-cases against the spec
  formulas, exact-boundary behavior (rate exactly 0.80 → 0; vmaf exactly at
  threshold → full formula; the anchor `s_pre=0.5 → 0.1`), non-finite inputs
  raising fail-closed, breakdowns recording every term.
- `test_gates.py` — a failing gate zeroing a perfect metric score, each gate's
  boundary, missing/NaN/inf primary+secondary VMAF as violations, secondary-VMAF
  absence failing closed by default vs recording an informational skip with the
  flag off, absent/unsupported upscale factors failing closed, pipelines
  collecting all violations plus `extra_violations`.
- `test_result.py` — lossless JSON round-trips, config digest tracking changes,
  skips persisting through packet JSON, out-of-range/non-finite scores being
  unconstructible AND unparseable (including Infinity inside a breakdown).
- `test_aggregate.py` — worst-decile punishing one failure a mean would hide,
  ceil sizing, order independence, non-finite rejection.
- `test_backends.py` — start-frame determinism, per-challenge/content variation,
  range and separator-ambiguity properties.
- `test_canonicalize.py` / `test_dedup.py` — digest stability across paths,
  stream-mismatch flagging, deterministic dedup regardless of arrival order.

The subprocess-backed code in `backends_real.py` is exercised by the scoring
worker's suite (`tests/scoring_worker`, which needs ffmpeg/ffprobe) rather than
`tests/scoring`.

## How to change safely

- **Formula/threshold changes change what a score means**: `scoring_config_digest`
  moves automatically with any `ScoringConfig` change, and the audit recompute
  compares recorded vs recomputed values — recorded packets stay verifiable only
  when the recompute runs the same config and `scorer_version`. Treat constants as
  spec changes: update the golden tests deliberately.
- **`ReasonCode` and metric-key names are a stable vocabulary** consumed by the
  audit layer and dashboards; add codes, never rename/reuse them.
- **Canonicalization targets** (`CANONICAL_PIX_FMT`/`CANONICAL_COLOR`/
  `CANONICAL_TIMESCALE` and the argv shape) change `plan_digest` for every future
  item — fine going forward, but never "fix up" old digests.
- **`derive_pieapp_start_frame` is frozen**: changing it breaks recomputability of
  every recorded upscaling score.
- New gates: return violations with an existing-or-new `ReasonCode`; skipping must
  go through `GateSkip`, never a silent early return.
- Keep `file_size_caps` keys and `challenge.dag.UPSCALE_FACTORS` in lockstep (the
  cross-module test enforces it).

## Status & gaps

- [DONE] All pure logic (formulas, gates, aggregation, dedup, canonicalization
  plans, packets) and the real ffprobe/ffmpeg-libvmaf/canonicalization backends.
- [DONE] `PieAppTorchBackend` supports explicit CPU execution through PIQ PieAPP,
  while `CpuPerceptualCheckBackend` implements deterministic CPU/OpenCV tone,
  grayscale, and chroma checks. The production worker wires these real backends;
  `UnconfiguredPerceptualCheckBackend` remains only an explicit refusal seam for
  incomplete/test compositions. `SecondaryDecoderBackend` remains a deferred
  defense-in-depth hook.
- [DONE] The VMAF model-delta gate uses the **miner input** as the basis for both
  primary and NEG comparisons. Evidence records that basis and its canonical input
  plan digest. Calibration and the adopted decision are documented in
  the project design record.
- [DONE] Tone, grayscale, and chroma manipulation gates use that same canonical
  **miner-input basis**, not the pristine holdout. This prevents a DAG-applied
  tone/exposure/color transform from being attributed to an honest miner. Packets
  record `perceptual_gate_basis: miner_input` plus the canonical input-plan digest;
  scorer pipeline version 4 fences the semantic change.
- Perceptual-hash *computation* has no real backend yet (only the comparator seam
  used by dedup); `DeterministicFakeBackend` covers tests.
