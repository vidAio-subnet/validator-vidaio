# vidaio.tokenomics — deterministic inference, competition, crown, and sink economics

This package is the I/O-free emission engine. Chain and database adapters construct the
frozen inputs; tokenomics folds scores, ranks miners, resolves the competition crown,
composes float shares, and exposes the one deterministic u16 quantizer used by every
validator. It does not read a clock, chain, or database. Locked launch choices are in
the project design record #1, #2, #5, #11–#13.

Competition/crown is **live and earning on testnet**. The shipping
[`config/default.yaml`](../../config/default.yaml) sets
`tokenomics.competition_emissions_enabled: true`. The model default remains false so
isolated library callers must explicitly supply the additional competition inputs; the
shipping flag is an emergency testnet off switch, not the launch target.

## Pool composition

`weights.build_weight_vector` composes fixed pools without cross-subsidy:

| State | Inference | Competition | Canonical sink |
| --- | ---: | ---: | ---: |
| IDLE | 0.80 | 0.00 | 0.20 |
| PODIUM window | 0.60 | 0.40 | unpayable shares only |
| CROWN window | 0.10 | 0.90 | unpayable shares only |

Configuration validates each state to sum exactly to one and launch guards pin these
protocol values. Disabling competition emissions forces the IDLE allocation; it does
not donate the state-derived 20% sink share to inference. The inference portion keeps
its fixed per-track split in every state.

### Inference tracks

The current inference portion always splits **0.8 compression / 0.2 upscaling**. An
empty or below-floor track does not donate its share to the other track.

Within each track, miners rank by accumulated EWMA score with uid as the deterministic
tie-break. The first five eligible miners take the graded `5:4:3:2:1` curve; rank six and
below receive zero. A miner must have an absolute accumulated score of at least `0.10`,
must not be excluded, and must survive deterministic IP/coldkey deduplication. Unspecified
addresses (`0.0.0.0` and `::`) do not form dedup groups.

The EWMA is `new = 0.75 * old + 0.25 * score`. `EXCLUDED_SCORE = -1.0` latches until a
genuine score arrives, at which point re-entry starts from zero. Persisted, signed
miner-attributable availability failures contribute real zero observations so selective
non-response cannot freeze an old standing; authority-side failures remain excused.

### Competition podium and crown

Competition economics consumes the `CompetitionResult` that the schema-v15 evidence
bridge and auditor independently derive from exact packet scores. For every committed
subject, the economic score is the arithmetic mean of its complete item packet set;
contenders then sort by `(-score, hotkey, uid)`. Stored `final_rank`,
`manual_disqualified`, `eligible`, and human-review preferences are not tokenomics inputs.

The archived executable baseline is a non-earning audit subject with no payout identity.
Its source archive, provenance, git tree and execution image are committed before
enrollment and it runs over exactly the same hidden item matrix as every contender.
Version zero is seeded from the public reference implementation; each audit-clean CROWN
winner becomes the next serving baseline. Serving persists until replacement or explicit
rollback, independently of the finite payout window.

The latest successfully applied global result replaces any earlier window and starts a
half-open seven-day interval `[applied_at, applied_at + 168h)`. `applied_at` is the epoch
close-block timestamp committed in the log, never a database or process clock. The
global cycle is the 1-based append order of actual terminal COMPLETED events, with the
persisted event id breaking equal completion timestamps; creation and scheduled-start
order never select the latest result. Each global competition cycle can apply only once:

- **IDLE** (no active interval): inference 80%, competition 0%, canonical sink 20%.
- **PODIUM** (winner is below the inclusive 5% improvement floor): inference 60%,
  competition 40%.
- **CROWN** (winner meets or exceeds that floor): inference 10%, competition 90%.

A missing/non-positive executable-baseline score or absent contender is incomplete
economic evidence, not a PODIUM outcome. Resolution fails closed as a retryable no-op:
the prior window is preserved and the cycle is not consumed, so repaired evidence for
that same cycle can still be applied.

The first three contenders receive `0.70 / 0.20 / 0.10` of the active competition
pool. Missing ranks and contenders absent from the current snapshot are not
redistributed; their fixed shares go to the canonical sink. Every new contender must
match the exact close-block census and duplicate suppression is independently derived
from coldkey, advertised IP (excluding `0.0.0.0`), and exact output matrix. A carried
window pays the podium hotkey at its current eligible uid even if that numeric uid changed
after the result; a now-absent hotkey receives nothing and its fixed share burns.

### Canonical sink/burn UID

`burn_proportion` remains locked at `0.0`; there is no discretionary percentage burn.
The production caller nevertheless supplies the chain-derived canonical sink/burn UID.
Every unallocated fixed share goes there: an empty or below-floor inference track, a
missing active podium rank, IDLE's fixed 20%, or a rewarded hotkey no longer present in
the close-block snapshot. A fully empty epoch therefore becomes the
trivially-identical `{burn_uid: 1.0}` vector. A non-empty epoch may contain both earners
and the sink.

This residual entry is required because Bittensor normalizes submitted weights. Simply
omitting an unallocated pool would donate it to the remaining earners and violate the
fixed pool declarations. An operational/database/archive failure is never converted into
a sink allocation; the caller HOLDs instead.

## Deterministic quantization

`quantize_u16(weights)` is the canonical float-to-u16 grid used by the authority,
epoch-log model, auditor, and weight-setter. It converts deciding computations to exact
`Fraction` arithmetic, drops non-positive entries, assigns every positive share at least
one unit, distributes the remainder by largest fractional remainder with uid tie-break,
and water-fills overflow. The result sums to exactly `U16_MAX = 65535`. Two validators
holding the same float vector therefore emit byte-identical u16 vectors.

## Public API

Configuration and state:

- `TokenomicsConfig` — locked pool/rank/window settings and the emergency competition flag.
- `MinerSnapshot` — one close-block inference/economic identity.
- `ContenderResult`, `CompetitionResult`, `RewardWindowState`, `EmissionState` — the
  recomputable result and predecessor-folded window state.

Inference:

- `EXCLUDED_SCORE`, `is_excluded`, `accumulate`.
- `eligible_for_ranking`, `track_shares`, `inference_shares`.

Competition windows:

- `contender_margin`, `qualifies_for_crown`, `winner`, `PODIUM_SPLIT`.
- `resolve_reward_window`, `window_active`, `active_emission_state`,
  `emission_shares`, `podium_hotkey_shares`.

Composition:

- `build_weight_vector(config, miners, *, burn_uid, reward_state, now)` — deterministic
  pool composition. `now` is required when competition emissions are enabled.
- `ensure_locked_levers`, `ensure_alpha_stake_factor_disabled`.
- `quantize_u16`, `U16_MAX`.

## Invariants

- The shipping config enables competition emissions; disabling it is an explicit
  emergency/test override.
- Every state allocation is exact: IDLE `0.80 + 0.20`, PODIUM `0.60 + 0.40`,
  CROWN `0.10 + 0.90`.
- Inference track allocations remain 0.8/0.2 even when one track is empty.
- The inference cutoff is top five, graded `5:4:3:2:1`, with absolute score floor 0.10.
- `burn_proportion == 0.0` and `alpha_stake_weigh_factor == 0.0` hard-fail if changed;
  the IDLE sink share is state-derived, not discretionary burn configuration.
- `empty_pool_policy == "withhold"` is locked for launch; the production burn UID makes
  withheld shares explicit on chain.
- Every newly applied competition contender must match the complete close-block registered
  census. A contender absent from the narrower economic snapshot stays in the result, but
  an unpayable podium share goes to the sink. The archived baseline has no payout identity
  and cannot rank or earn.
- Window resolution is deterministic and time-based from the result's committed
  `applied_at` epoch clock, never a local wall clock at ingestion.
- Human review state never crosses the economic seam.

The removed retention multiplier remains removed. `emission_liquidation_weigh_factor`
and `retention_full_window_required` survive as validated compatibility knobs, but no
retention value reshapes a miner's rank share.

## Testing

```sh
.venv/bin/pytest tests/tokenomics tests/competition/test_economic_result.py \
  tests/integration/test_competition_epoch_evidence.py
```

Coverage includes the graded rank curve and absolute floor, fixed 0.8/0.2 track
allocations, conditional sink allocations, global result replacement, uid/hotkey
binding, exact half-open seven-day window boundaries, inclusive 5% qualification, exact
quantization, and proof that human-review mutations cannot change packet-derived
competition emissions.

## Change safety

- Any pool, rank, margin, or crown change is an economics change: update
  the project design record, config, golden scenarios, authority derivation,
  auditor derivation, and the epoch schema/version fence together.
- Never feed repository ranking/review fields directly into tokenomics. Only the
  independently recomputed schema-v15 packet derivation may construct the economic
  `CompetitionResult`.
- Keep reward-window state separate from the serving-baseline registry: expiry returns
  emissions to IDLE but never rolls executable quality back.
- A CROWN epoch is publishable only after the exact winner submission archive is public;
  baseline promotion additionally requires finalized-anchor verification and a fresh
  rebuild/rerun against the committed matrix.
