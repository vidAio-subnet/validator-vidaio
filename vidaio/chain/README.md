# vidaio.chain — the single chain boundary (ChainAdapter)

Every service talks to the chain ONLY through the `ChainAdapter` Protocol in
`adapter.py`. Only ONE module — `bittensor_adapter.py`, and only its thin
transport seam — ever imports bittensor, and it imports it LAZILY, so
`import vidaio.chain` and the whole test suite run without the optional
`.[chain]` dependency group. Mode selection is config-only:
[`config/default.yaml`](../../config/default.yaml) ships `chain.mode:
bittensor` as the PRODUCTION default (real metagraph sync, real weight sets,
real anchors — the project design record rules 8 & 9);
`chain.mode: report` (the chainsim / embedded-journal path) is available and
is the default ONLY in test/dev/local overlays.

## What it does

- **`adapter.py`** — [DONE] the `ChainAdapter` Protocol, the
  `ChainNeuron` / `EpochBoundary` / `SetWeightsResult` / `SubmittedWeights` datatypes, the
  `ChainStateUnavailable` error, the optional `SubmittedWeightsReader`
  plus finalized-block/epoch-boundary/archive extensions, `parse_anchor_digest`, and
  `InMemoryChain` (the deterministic
  in-process fake).
- **`client.py`** — [DONE] the report-mode implementations:
  `HttpChainAdapter` (over a running chainsim,
  [`vidaio/chainsim/README.md`](../chainsim/README.md)) and
  `EmbeddedReportingChain` (JSONL-journaled `InMemoryChain` for
  single-process runs).
- **`bittensor_adapter.py`** — [DONE, needs testnet validation]
  `BittensorChainAdapter` (+ `BittensorAdapterConfig`, `MetagraphView`): the
  REAL adapter, built from prior production-subnet experience. All decision
  logic — u16 quantization, metagraph→neuron mapping,
  set_weights result classification, freshness/cache,
  submitted-weights readback + commit-reveal awareness, hotkey↔uid
  reconciliation, one-socket/reconnect discipline — is FULLY unit-tested
  against a fake transport and imports no bittensor; only the `_RealSubtensorTransport`
  seam touches the SDK, and must be validated on testnet rather than inferred from unit tests.
- **`factory.py`** — [DONE] `ChainConfig` (the `chain:` config section) and
  `make_chain_adapter(raw_config)` plus the wallet-free
  `make_read_only_chain_adapter(raw_config)` used for public registration reads;
  mode selection is config-only.
- **`quantize_u16` / `U16_MAX`** — the canonical deterministic quantizer now
  lives in the dependency-free shared home
  [`vidaio/tokenomics/quantize.py`](../tokenomics/README.md) (wave-1
  CONSOLIDATE); it is re-exported from `bittensor_adapter` so the historical
  import site keeps working, but the core math is single-sourced there and
  reused by the epoch-log finalizer, the publication document, and the
  auditor.

## Design & decisions

### The Protocol

```python
class ChainAdapter(Protocol):
    def current_block(self) -> int: ...
    def neurons(self) -> list[ChainNeuron]: ...
    def refresh(self) -> None: ...                                   # never raises
    def has_fresh_snapshot(self, now: float, max_age_seconds: float) -> bool: ...
    def snapshot_age(self, now: float) -> float | None: ...
    async def set_weights(self, weights: dict[int, float], *, version_key: int) -> SetWeightsResult: ...
    async def anchor_commitment(self, payload: bytes) -> str: ...    # <= 128 bytes
```

- **Reads are synchronous snapshots** of the adapter's cached state;
  refreshing (and throttling refreshes) is the adapter's own concern.
- **Writes are async and MUST be wrapped in
  `vidaio.core.resilience.with_timeout` by callers** — an adapter never
  guards for you.
- **Freshness is explicit.** `refresh()` never raises on transport/decoding
  failure (a flaky chain must not crash a service loop); it keeps the
  previous snapshot and reports through `has_fresh_snapshot(now, max_age)` /
  `snapshot_age(now)` (`now` is wall-clock epoch seconds — the one clock
  family this surface speaks, since the timestamp crosses a process
  boundary). Both the validator and the weight-setter gate their rounds on
  this surface and skip with a structured reason when it is stale.
- **`ChainStateUnavailable`:** `neurons()` on an adapter that has NEVER
  successfully refreshed raises instead of returning `[]`. An empty list
  means "no neurons registered"; it must never be the way "we could not
  reach the chain" is expressed — that ambiguity is exactly how a startup
  race turns into "successful" empty rounds and silently omitted weights.

### The optional `SubmittedWeightsReader` extension

`submitted_weights(hotkey) -> SubmittedWeights | None` — the weight vector
the chain currently records for that hotkey. It exists because it is the
only evidence that can prove a SPECIFIC weight write landed: "our
`last_update` advanced" merely says somebody's write did, which is a
different claim and is exactly how an unconfirmed vector used to get
published. The weight-setter feature-detects it
(`isinstance(chain, SubmittedWeightsReader)`); an adapter that cannot answer
makes every confirmation UNKNOWN — never CONFIRMED and never DENIED (see
[`vidaio/weightsetter/README.md`](../weightsetter/README.md)).

Contract:

- returns the CURRENT record (latest accepted vector) for `hotkey`;
- returns `None` to say POSITIVELY that the chain holds no weights for it —
  the ONE answer that can deny an intent on its own. An EMPTY mapping is
  different: "a record exists but carries no positive weight";
- RAISES (`ChainStateUnavailable` or any transport/decoding error) when the
  answer cannot be read. Never substitute `None` for a failed read: `None`
  denies, and a denied intent is eventually abandoned unpublished.

`SubmittedWeights.weights` may be reported in whatever scale is cheapest
(raw chain u16s, sum-normalized floats, untouched floats): the comparison
the weight-setter runs is scale-invariant (max-normalized onto the chain's
u16 grid — `vidaio.weightsetter.intents.quantize_weights`). What must NOT be
done is reporting a re-weighted, filtered or padded vector: that is a
different vector and will read as one. `block` (when reportable) is what
disambiguates two attempts carrying an identical vector.

### The four shipped adapters

| Adapter | Use | Notes |
|---|---|---|
| `InMemoryChain` | tests, in-process harnesses | Deterministic; time moves only via `advance_blocks()`; records every accepted `set_weights` (`weight_calls`) and anchor (`anchored`); tempo gate (`block <= last + tempo`, tempo 100) mirrors chainsim; `fail_next_set_weights` fault injection; trivially fresh (it IS the chain); implements `submitted_weights` (single writing identity — answers for any hotkey asked) and `read_anchor` (reads its recorded anchor payloads back — the shared-snapshot anchor reader in report mode). |
| `EmbeddedReportingChain` | single-process report-mode runs (`chain.chainsim_url: embedded`) | `InMemoryChain` + append-only JSONL journal of every `set_weights` call (accepted AND rejected, with outcome) and every anchor — the journal shows what was ATTEMPTED, `weight_calls` keeps only what was accepted. Default journal: `<chain.report_dir>/embedded-chain.jsonl`. |
| `HttpChainAdapter` | multi-process report mode (the test/dev default wiring) | The Protocol over a running chainsim. `refresh()` pulls `GET /neurons` into the cached snapshot (both fetch AND decode guarded; failure keeps the previous snapshot and sets `last_refresh_error`); `neurons()` raises `ChainStateUnavailable` until the first successful refresh; writes `POST /weights` / `POST /anchor` carry the identity's bearer token (`Authorization: Bearer`); `submitted_weights(hotkey)` is a LIVE `GET /weights/{hotkey}` (a question about the chain right now, not a snapshot). Transport failures on writes surface as `OSError`/`TimeoutError` so callers' retry envelopes handle them; a 401/403 on `set_weights` becomes a failed `SetWeightsResult` (not retryable — wrong identity), a 401/403 on anchor raises `PermissionError` (still an `OSError` for retry envelopes, but the message says "credential"). `register()` claims the hotkey on the sim and CAPTURES the issued token, so a self-registering process needs no configured secret; a 409/403 there means the hotkey belongs to somebody else's token — do NOT continue unauthenticated. |
| `BittensorChainAdapter` | PRODUCTION (`chain.mode: bittensor`) | The REAL adapter over one long-lived `bt.Subtensor` socket. See below — this is the whole real-adapter blueprint made real. |
| `BittensorReadOnlyChainAdapter` | Audit Results API registration reads | Uses the same bounded/reconnecting read transport without loading a wallet, seed, or signer identity; every sign/weight/commitment attempt raises `ReadOnlyChainError`. |

The bearer token in report mode sits exactly where the real adapter's signing
keypair sits: mutations are authenticated, reads are public — the same shape as
real chain extrinsics vs storage queries.

### `BittensorChainAdapter` — the real adapter (needs testnet validation)

Construction FAILS FAST: with no injected transport it opens the real socket
NOW (loading the wallet + connecting), so a pod that forgot its
`hotkey_seed_env` seed or cannot reach the endpoint crashes at startup rather
than idling without submitting (risk 13). The load-bearing behaviours:

- **v10 result classification (`_parse_chain_result`).** bittensor v10 (pinned
  `bittensor==10.5.0` + `async-substrate-interface==2.2.1`, co-pinned) returns
  an `ExtrinsicResponse`: the adapter reads `.success` / `.message` /
  `.extrinsic_receipt` and DELIBERATELY does NOT consult `bool(response)` — an
  `ExtrinsicResponse` can be truthy for a REJECTION, which is exactly how
  `bool(response)` used to "publish" weights that never landed. An unrecognized
  shape RAISES (a failure), never an implicit success. The txid is the
  extrinsic RECEIPT hash, not the response message.
- **Mutex-serialized, never-timeout-bounded `set_weights`.** The
  inclusion/finalization wait is intentionally long and MUST NOT be
  timeout-bounded by the caller: `asyncio.to_thread` cannot be cancelled, so a
  caller `with_timeout` that fires does NOT stop the worker thread — it keeps
  the SDK's per-submit block subscription open (the leak that OOMed a prior
  production validator's pod).
  A socket-level `RLock` is held across the whole extrinsic so a
  caller-abandoned submit thread is never run over, and a cancelled attempt
  CONDEMNS the socket so the next call reconnects only AFTER that thread
  finishes. The weight-setter must therefore NOT wrap this in `with_timeout`
  and uses the shipped `chain_timeout_seconds=180` s default for anchors.
- **Commit-reveal awareness (`submitted_weights`).** While a v10 timelocked
  (CRv4) commit is pending for the hotkey, `Weights` storage still holds the
  PREVIOUS vector — read literally that is the weight-setter's one positive
  DENIAL, which would bury an intent whose commit is merely awaiting its reveal
  window. So a pending commit RAISES (`ChainStateUnavailable` = UNKNOWN = HOLD)
  rather than answering. The pending-commit probe pins one best-head hash and
  exhausts every page and every live epoch bucket under
  `TimelockedWeightCommits[netuid_index]`. Do not use v10.5's convenience getter
  here: it asks for a one-record page and then reads only `result.records[0]`, so
  it can miss our commit across an epoch transition. The legacy per-hotkey
  `WeightCommits` storage shape is not CRv4 state. A registered
  hotkey with no positive weight returns `None` (the one answer that can deny an
  intent); a failed read RAISES, never `None`.
- **One SDK write attempt per durable intent.** `set_weights(max_attempts=1)`
  disables the SDK's default five immediate retries. VIDAIO's intent ledger owns
  retry and readback, so a rejected/ambiguous CRv4 write remains attributable to
  one durable attempt.
- **Hotkey↔uid exact-target verification (`_reconcile_targets`).** A uid can be
  recycled to a DIFFERENT hotkey between scoring and submission; checking only
  uid liveness would pay the new occupant the old miner's weight. When the
  weight-setter supplies the intended per-uid `hotkeys`, every positive target
  must still bind that uid to the SAME hotkey. An orphan or recycled uid rejects
  the complete attempt before any transport write. The adapter never drops and
  re-normalizes a subset, because that would donate the missing fixed share to
  surviving targets and mutate the authenticated authority vector.
- **`quantize_u16`** is reused from `vidaio/tokenomics/quantize.py`, so two
  validators holding the same float vector emit byte-identical u16 output
  (summing to exactly 65535) — the convergence crux.
- **`anchor_commitment` / `read_anchor` / `read_anchor_at`.** Anchors the ≤128-byte epoch-log
  commitment on the Commitments pallet (`set_commitment`) and reads it back for
  the authority's account (`anchor_hotkey`, falling back to our own) as the
  independent third tamper-evidence leg. Historical reads pin
  `get_commitment_metadata(..., block=pointer.anchor_block)` and verify the
  record's stored inclusion block, so later single-slot overwrites do not erase
  backfill verification. Same socket/lock/classification discipline. UNPROVEN:
  no production precedent in either subnet, so validate on testnet/archive.
- **Finalized, chain-native epoch reads.** `finalized_block()` resolves GRANDPA
  finality. `latest_closed_epoch()` and `epoch_close_block()` binary-search historical
  `SubnetEpochIndex` and accept only an exact `E-1 -> E` transition whose close state
  reports the same `LastEpochBlock`. Missing/pruned/inconsistent history raises/HOLDs;
  production never synthesizes a boundary from best head or `head // tempo`.
- **`tempo(netuid)`** remains a live chain input for submission/rate-limit context and
  report compatibility; it is not the authority for Bittensor epoch boundaries.
- **One socket, reconnect after 3 RAISED failures.** A clean submit resets the
  counter; a clean read does not (only a submit proves the write path healthy);
  a condemned socket (fired timeout) reconnects before the next call.
  `refresh()` NEVER raises — a transient RPC failure keeps the cached snapshot.

### The factory and `chain.mode`

`make_chain_adapter(raw_config)` reads the `chain:` section:

- `mode: bittensor` (the PRODUCTION default in `config/default.yaml`): builds
  `BittensorChainAdapter`, lazily importing the optional `.[chain]` bittensor
  deps inside `bittensor_adapter`; a missing dep or wallet fails fast with a
  clear message (`NotConfiguredError` points at the extra).
- `mode: report`: the chainless path — the whole stack runs end-to-end WITHOUT
  bittensor, producing reports of scores and weight vectors instead of chain
  pushes. `chainsim_url` selects `HttpChainAdapter` (a URL) or
  `EmbeddedReportingChain` (the sentinel string `"embedded"`). This is the
  default in test/dev/local overlays.

`make_read_only_chain_adapter(raw_config)` is the Audit Results API exception. In
Bittensor mode it retains public chain locators, timeout bounds, and reconnect policy,
but deliberately discards `validator_hotkey`, wallet paths/names, and the seed-env
locator before construction. It exposes all bounded reads and fails every signing or
mutation attempt locally. Its startup therefore needs the SDK and reachable RPC, but no
wallet file or signing secret.

Note the `ChainConfig.mode` MODEL default is deliberately `report` (not
`bittensor`) so a bare `make_chain_adapter({})` — a testing convenience with no
YAML — stays chainless; production never passes an empty dict, it loads
`config/default.yaml` whose `chain.mode: bittensor` is the real default. Only
the adapter implementation swaps — never service code.

### The real-adapter design (from prior production experience)

The blueprint distills lessons from prior production-subnet deployments, and
`BittensorChainAdapter` implements it (unit-tested through a fake transport, not yet
validated on SN85 testnet). The load-bearing lessons it honours:

- **Serialized sockets with an isolated commitment lane.** Ordinary sync
  `bt.Subtensor` calls share one long-lived socket/lock. Commitment writes use a
  second long-lived socket/lock so a cancelled, non-cancellable SDK finalization
  worker cannot starve archive/metagraph/readback RPCs. Reconnect only after 3
  consecutive RAISED failures (chain rejections over a healthy socket must not
  churn the socket). Per-error reconnects and abandoned worker threads leaked
  block-event subscriptions and OOMed a production pod in minutes.
- **Never hard-timeout `set_weights`.** Short RPCs get bounded worker
  threads; the `set_weights` inclusion+finalization wait is deliberately NOT
  bounded (or is condemn-socket-on-timeout) — abandoning the thread
  mid-submit is the subscription leak above. The weight-setter's
  anchor `chain_timeout_seconds` defaults to 180 s for the real adapter.
- **Poll the independent finalized proof, never the write.** A non-CR SDK
  success gets five exact `Uids`/`Weights`/`LastUpdate` observations, spaced by
  one expected block (12 s), because the archive read can briefly trail the
  submit socket. The adapter emits no second extrinsic during that wait and
  still returns UNKNOWN after exhaustion unless the exact runtime bytes appear
  at a strictly newer `LastUpdate`.
- **Keep rejections legible without dumping SDK objects.** Structured dispatch
  errors are rendered from a bounded public-field allow-list, which preserves
  rate-limit names while ignoring opaque payloads. The pinned SDK's empty false
  result is treated as tempo only inside `set_weights`, where v10.5 uses that
  exact shape when its internal rate-limit precheck makes no attempt.
- **u16 quantization.** The chain MAX-normalizes submitted vectors (largest
  weight → 65535); submitted floats never come back as floats. The
  weight-setter already compares on the chain's own grid
  (`intents.quantize_weights`, tolerance 1 u16 step) — the adapter must
  report the chain's vector AS GIVEN, filtered/re-weighted by nothing, and
  should own its outbound quantization so the chain's rescale is a no-op.
- **A non-CR SDK success still needs storage proof.** The v10
  `ExtrinsicResponse` is tuple-like and truthy even on rejection, so the adapter
  reads `.success` explicitly; even a true value remains only a claim until raw
  `Uids`, `Weights`, and `LastUpdate` from one GRANDPA-finalized hash show the
  exact emitted max-grid strictly after the submit snapshot. A missing, stale,
  mismatched, or unreadable proof is UNKNOWN and keeps the durable intent
  pending. This closes a false-success failure observed in production.
- **Commit-reveal is explicit on write and read.** A clean CRv4 submission is a
  finalized timelocked commit, not an active weight vector, so the adapter reports
  `pending_reveal=True`/`success=False` and publication remains forbidden. While
  any bucket in the block-pinned, fully paginated
  `TimelockedWeightCommits[netuid_index]` prefix still contains this hotkey,
  `submitted_weights` raises UNKNOWN and the weight-setter emits no duplicate
  intent. Only exact active `Weights` readback after reveal confirms publication.
- `anchor_commitment` has NO production precedent in either subnet
  (Commitments pallet is the natural target) — testnet prototype required
  before relying on it. A failed attempt remains `pending_chain` and is re-driven,
  but an anchor included after `close_block + K` cannot be repaired safely; retain
  live attempt/rejection/acceptance/inclusion blocks and prove the deadline.

## Public API & endpoints

Re-exported from `vidaio.chain`: `ChainAdapter`, `ChainNeuron`,
`ChainStateUnavailable`, `SetWeightsResult`, `SubmittedWeights`,
`SubmittedWeightsReader`, `InMemoryChain`, `EmbeddedReportingChain`,
`HttpChainAdapter`, `BittensorChainAdapter`, `BittensorAdapterConfig`,
`ChainConfig`, `make_chain_adapter`. (`quantize_u16` / `U16_MAX` are re-exported
from `vidaio.chain.bittensor_adapter` for the historical import site but are
single-sourced in [`vidaio/tokenomics/quantize.py`](../tokenomics/README.md).)

No HTTP endpoints of its own — `HttpChainAdapter` is a CLIENT of the
chainsim's endpoints (`GET /neurons`, `GET /weights/{hotkey}`,
`POST /register`, `POST /weights`, `POST /anchor`; see
[`vidaio/chainsim/README.md`](../chainsim/README.md)).

## Data & invariants

- `ChainNeuron`: `uid, hotkey, coldkey, ip, alpha_stake, emission,
  is_validator, last_update` (frozen dataclass). `emission` is treated by
  consumers as a per-block rate at observation time.
- `anchor_commitment` payloads are ≤ 128 bytes (enforced by every
  implementation); returns a transaction/extrinsic id.
- `InMemoryChain.advance_blocks(n)` refuses negative `n` — blocks only move
  forward.
- `HttpChainAdapter.current_block()` is 0 until the first successful
  refresh; `last_successful_refresh` / `last_refresh_error` expose refresh
  state. On `InMemoryChain`, `last_successful_refresh` is always `None` for
  surface uniformity — do NOT read that as "never refreshed";
  `has_fresh_snapshot()` (always `True` there) is the freshness contract.
- The embedded journal is append-only, one sorted-key JSON object per line.

## Configuration

Section: `chain` (schema `factory.py::ChainConfig`,
`extra="forbid"`). Env override pattern: `VIDAIO__CHAIN__<KEY>=<value>`.

| Key | Default | Meaning |
|---|---|---|
| `mode` | `report` (model) / `bittensor` (`config/default.yaml`) | `bittensor` (the REAL adapter — production default) or `report` (chainless sim/embedded — the test/dev overlay default). The model default stays `report` so a bare `make_chain_adapter({})` is chainless |
| `chainsim_url` | `http://127.0.0.1:8400` | (report) Chainsim base URL, or the sentinel `embedded` for an in-process `EmbeddedReportingChain` |
| `validator_hotkey` | `local-validator` | Our validator identity (report: sim register+weights+anchors; bittensor: the ss58 that writes weights and is read back under) |
| `auth_token` | `""` | (report) Bearer proving ownership of `validator_hotkey` on the sim — the stand-in for the signing wallet. Empty when the process self-registers via `HttpChainAdapter.register()`; set it (`VIDAIO__CHAIN__AUTH_TOKEN`) when another process registered. Ignored by the `embedded` chain |
| `report_dir` | `./data/chain-reports` | (report) Report artifacts (embedded-chain journal; chainsim report output) |
| `anchor_hotkey` | `""` | (bittensor) The Scoring Authority ss58 read as the third verification leg. The adapter fallback is useful for report/tests, but production authority/challenge writers require it explicitly equal to their loaded signer; thin readers explicitly name the central signer and may use a different own wallet. |
| `anchor_writer_lock_path` | `null` (model), `/var/lib/vidaio/state/anchor-writer.lock` (release config) | Coherent local/POSIX lock shared by processes writing one wallet's mutable commitment slot. Production writer roles require an absolute path; never use NFS or independent container files. |
| `anchor_writer_lock_timeout_seconds` | `30` | Bounded wait for the one-slot writer lane; `(0, 60]`. Contention fails closed for retry/HOLD. |
| `network` | `finney` | (bittensor) Named network (`finney` prod / `test` testnet / `archive`); an explicit `endpoint` beats it |
| `netuid` | `85` | (bittensor) The subnet this validator writes weights on (VidAIO) |
| `endpoint` | `""` (model), `wss://archive.chain.opentensor.ai:443` (mainnet release config) | (bittensor) Explicit public `wss://` endpoint; overrides `network`. The testnet template requires operator-supplied `VIDAIO_TESTNET_CHAIN_ENDPOINT` with no default and shows `wss://test.finney.opentensor.ai:443` only as an example. Production refuses the empty model default and live preflight proves historical retention rather than trusting an endpoint label. |
| `fallback_endpoints` | `[]` | (bittensor) Ordered additional public archive RPCs. Empty for initial launch; a later second endpoint is tried in operator preference order and must pass the same deep-history probe. |
| `wallet_name` / `wallet_hotkey` / `wallet_path` | `""` | (bittensor) On-disk btcli wallet; takes precedence over the seed env when both exist |
| `hotkey_seed_env` | `VIDAIO_HOTKEY_SEED` | (bittensor) NAME of the env var holding the hotkey seed/mnemonic (never the value); the pod crashes at startup if absent |
| `version_key` | `15` | (bittensor) Explicit fleet fence synchronized with epoch-log schema v15; report/test may opt into 0 |
| `connect_timeout_seconds` / `rpc_timeout_seconds` | `30` / `30` | (bittensor) Per-attempt connect + short-RPC timeouts (daemon-thread bounded; NEVER bounds `set_weights`) |
| `weight_readback_attempts` / `weight_readback_delay_seconds` | `5` / `12` | (bittensor, non-CR) Exact finalized-state observations after an SDK success claim: immediate + four one-block waits; no resubmission occurs inside this poll |
| `metagraph_ttl_seconds` | `120` | (bittensor) Metagraph snapshot TTL — `refresh()` skips the RPC inside this window |
| `reconnect_after_consecutive_failures` | `3` | (bittensor) Reconnect the socket only after this many CONSECUTIVE raised failures |

## How to test

```sh
python -m pytest tests/chain/test_bittensor_adapter.py    # the real adapter vs a fake transport
python -m pytest tests/chain/test_config_default_flip.py  # the report/bittensor default flip
python -m pytest tests/chainsim/test_adapter.py           # HttpChainAdapter vs a live sim app
python -m pytest tests/chainsim/test_embedded_chain.py    # EmbeddedReportingChain journal
python -m pytest tests/chainsim/test_freshness.py         # freshness contract / ChainStateUnavailable
python -m pytest tests/validator/test_chain_state.py      # consumer-side gating
```

`InMemoryChain` is used pervasively across `tests/validator`,
`tests/weightsetter` and `the development-tree e2e suite` as the deterministic chain.

## How to change safely

- The Protocol is a cross-module contract consumed by the validator, the
  weight-setter, the orchestrator and the gateway wiring: adding a REQUIRED
  method breaks every fake; prefer optional, feature-detected extensions
  (the `SubmittedWeightsReader` / `has_fresh_snapshot` pattern — consumers
  probe with `getattr`/`isinstance` and degrade to UNKNOWN/skip).
- Never let `refresh()` raise, and never let a read substitute an empty
  snapshot — both halves of the freshness contract have dedicated tests.
- In `submitted_weights`, never flatten a failed read into `None`.
- The real adapter is built: keep ALL decision logic in
  `BittensorChainAdapter` (unit-tested against a fake transport) and confine SDK
  calls to `_RealSubtensorTransport` — the one class that imports bittensor,
  lazily, so report/test paths never load it. Keep `bittensor` co-pinned with
  `async-substrate-interface`, and NEVER timeout-bound the `set_weights`
  inclusion/finalization wait (the subscription-leak class of bug).

## Status & gaps

- [DONE] Protocol, freshness contract, `SubmittedWeightsReader`,
  `InMemoryChain`, `EmbeddedReportingChain`, `HttpChainAdapter`, factory.
- [DONE, needs testnet validation] `BittensorChainAdapter` — the full
  real-adapter blueprint made real: v10 `ExtrinsicResponse`
  classification, mutex-serialized never-timeout-bounded `set_weights`,
  timelocked-commit-aware `submitted_weights`, hotkey↔uid recycle
  reconciliation, one-socket/reconnect discipline, current/exact-block anchors,
  archive `neurons_at`, finalized height, exact runtime-epoch transitions/close blocks,
  wallet signing, block hash/time, chain-derived burn UID, axon IP/port, and live `tempo`.
  All decision logic is unit-tested against a fake
  transport; only the `_RealSubtensorTransport` SDK seam awaits testnet.
- [DONE, locked image] The `.[chain]` SDK family is optional for report installs but
  co-pinned in `uv.lock`/the release image (`bittensor==10.5.0`,
  `async-substrate-interface==2.2.1`, drand, and wallet). The image probe verifies the
  exact method signatures used by the adapter; lazy imports keep report mode chainless.
- [DONE semantics, needs testnet observation] `ChainNeuron.ip` comes from the metagraph
  axon. Unspecified `0.0.0.0`/`::` values are exempt from IP dedup; specified duplicates
  remain grouped. The Commitments-pallet anchor and archive retention/availability are
  still unproven live and must be exercised on testnet.
