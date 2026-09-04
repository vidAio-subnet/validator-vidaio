# vidaio.chainsim — the simulated chain (default runtime, report mode)

An HTTP chain SIMULATOR — not a test-only mock. In the chainless default
(the project design record rule 8) this is the actual
runtime chain the whole stack talks to: validators register here, the
weight-setter submits vectors here (same tempo semantics as
`vidaio.chain.InMemoryChain`), publications anchor here — and the outcome is
a report (JSON + markdown) of scores and weight vectors instead of a real
chain push. Services reach it through `vidaio.chain.HttpChainAdapter`
([`vidaio/chain/README.md`](../chain/README.md)); real-chain mode is a
separate, explicit opt-in behind the same `ChainAdapter` Protocol.

## What it does

- **`service.py`** — [DONE] `ChainSim(BaseService)`: FastAPI app on port
  8400, SQLite-backed state (registration/uids, block clock, weight calls,
  anchors, emission), token authorization, health checks, metrics.
- **`report.py`** — [DONE] `build_report` / `render_markdown` /
  `write_report` / `decode_anchor_payload`: the report generator over a full
  sim state.
- **`config.py`** — [DONE] `ChainSimConfig` (the `chainsim:` section).
- **`migrations/`** — [DONE] `0001_init.sql` (meta, neurons, weight_calls,
  anchors), `0002_auth.sql` (per-identity `token_sha256`).

## Design & decisions

Simulation model — deliberately simple, every simplification explicit:

- **Blocks — lazy wall-clock production, no background task.**
  `block = 1 + floor(max(0, now - start_time) / block_seconds) +
  advance_offset`. `start_time` and `advance_offset` persist in SQLite, so a
  restart resumes the block clock. `POST /advance` bumps the offset for
  deterministic tests (inject a frozen `now` and drive blocks entirely via
  `/advance`).
- **Registration → uid.** Uids are assigned sequentially from 0.
  Re-registering a known hotkey is idempotent: same uid, fields updated
  (`alpha_stake` only when the request carries one). A NEW hotkey taking
  over an EXISTING uid slot (real-chain deregistration/recycling) is NOT
  modeled — every new hotkey gets a fresh uid.
- **Tempo gate — `InMemoryChain` parity.** A validator hotkey's
  `set_weights` fails with `"tempo gate: too soon"` while
  `block <= last_accepted + tempo`; rejected calls are not recorded.
  Multi-validator stake-weighted consensus is NOT modeled: the last recorded
  vector (any hotkey) IS the emission director. Every accepted vector is
  kept; `GET /weights/{hotkey}` reads the latest one back (vector + the
  block it was recorded at, or `vector: null` when that hotkey never set
  weights) — the evidence a validator needs to prove its OWN vector landed
  before publishing it.
- **Proportional emission crediting.** Each block mints
  `emission_per_block`, distributed proportionally to the last recorded
  weight vector: a vector recorded at block B directs the emission of
  blocks strictly AFTER B, until the next recorded vector takes over.
  Blocks with no prior vector, and vector share pointing at unregistered
  uids, are undistributed ("burned", reported in `/state`). Crediting is
  lazy but purely a function of (call history, credited watermark
  `emission_credited_until`, current block), so WHEN settlement runs never
  changes the outcome. `GET /neurons` reports `emission` as the CURRENT
  per-block rate under the latest vector; cumulative credited emission
  lives in `/state` and the report.
- **Alpha stake — static.** Whatever (re-)registration set; no stake
  dynamics. Move stake by re-registering with a new `alpha_stake` (the
  lever that feeds `alpha_stake_delta` into the validator's retention
  windows).
- **Restart-safety.** Everything the sim knows lives in SQLite
  (`chainsim.db_path`): blocks, uids, tokens (hashed), weight history,
  anchors and the emission watermark all resume across restarts.
  `POST /reset` wipes history and rewinds the clock (identities and their
  tokens go with it); the OPERATOR credential survives a reset — it belongs
  to the node, not to the run.

### Authorization — the stand-in for chain signatures

On a real chain a mutation is an extrinsic SIGNED by a hotkey's keypair,
authorized by who signed it and what that neuron may do. The sim has no
keypairs, so it uses bearer tokens as the signature analogue:

```
participant token  ~  a signature by that hotkey   ("I am this neuron")
operator token     ~  the node/sudo key            ("I run this chain")
```

- Every identity gets a token at registration: the server generates one (or
  adopts a client-supplied `auth_token`, so a local fleet can share one dev
  secret from config — `chain.auth_token`) and returns it ONCE in the
  `/register` response. Only its SHA-256 is stored
  (`neurons.token_sha256`, constant-time compare): the sim can verify a
  token but can never hand it back out.
- Participant mutations (`/weights`, `/anchor`) require
  `Authorization: Bearer <token>` matching the hotkey NAMED IN THE BODY —
  a caller can only act as itself. `/weights` additionally requires the
  hotkey's role to be `validator` (a miner submitting weights gets 403) —
  the runtime's validator-permit check.
- **Takeover-blocked re-registration:** re-registering an already-claimed
  hotkey requires that hotkey's token — 409 (`auth_hotkey_claimed`) with no
  credential, 403 (`auth_token_invalid`) with a wrong one — so nobody can
  seize an identity by re-registering it. Rows migrated from a pre-auth
  database land with `token_sha256 = ''` ("unclaimed"): they can
  authenticate nothing, and the next `/register` for that hotkey claims
  them.
- Operator powers (`POST /advance`, `POST /reset`) require the OPERATOR
  token: producing blocks and destroying history are node powers, not
  participant powers. `POST /anchor` with no `hotkey` in the body
  (unattributed) is also operator-only. `POST /report/write` accepts either
  the operator or any registered identity (it writes files, so it is not an
  open read).
- **Open reads** — no auth, because the sim's whole point is an observable
  local chain and real chain state is public too: `GET /healthz`,
  `GET /neurons`, `GET /weights/{hotkey}`, `GET /state`, `GET /report`.
- Operator token provisioning: `chainsim.operator_token` in config wins and
  is re-applied on every start (the recovery path for a lost token).
  Otherwise one is generated ONCE per sim database, written to
  `<report_dir>/operator-token.txt` (mode 0600) and logged once.
- Auth failures are typed (`auth_token_missing`, `auth_token_invalid`,
  `auth_unknown_hotkey`, `auth_wrong_role`, `auth_hotkey_claimed`) in every
  401/403 detail and counted on
  `vidaio_chainsim_auth_failures_total{error}`.

The real adapter will sign extrinsics with a wallet instead; no service code
changes — `HttpChainAdapter` carries the token exactly where the bittensor
adapter will carry the keypair.

### The report generator

`build_report(state)` turns one full sim state (`GET /state`) into a
self-contained JSON document (`kind: vidaio.chainsim.report.v1`):

- registered neurons with role, alpha stake, current emission rate and
  cumulative credited emission;
- the weight-vector history per validator hotkey, each entry with per-uid
  deltas against that validator's previous vector;
- the latest vector overall as a RANKED table (rank, uid, hotkey, weight,
  share %);
- every anchored commitment with its best-effort DECODED payload —
  `vidaio.audit` anchors are ascii `domain:kind:sha256root` and decode into
  `{domain, kind, root}`; other printable ascii decodes to `{text}`; opaque
  bytes stay raw;
- the emission summary (minted / distributed / undistributed through the
  credited watermark).

`render_markdown` renders the same document with GFM tables;
`write_report(state, dir)` persists `report-<ts>.json` + `report-<ts>.md`.
`POST /report/write` does this server-side into `chainsim.report_dir`.

### Health — two checks that can actually say NO

- `sim_db` — opens its OWN short-lived connection per check (health runs on
  the HealthServer thread; sharing the service connection would test the
  wrong thing, and sqlite happily serves a deleted file's inode).
- `http_api` — flips false once the uvicorn task exits without a stop being
  requested (classically a bind failure). That path calls `fail_fatal`, so
  the process exits NON-ZERO and a supervisor restarts it — otherwise the
  worst outcome available: a process reporting "ok" with no API answering,
  leaving the whole stack without a chain.

## Public API & endpoints

| Method & path | Auth | Purpose |
|---|---|---|
| `GET /healthz` | open | Liveness + current block |
| `POST /register` | none for a new hotkey; the hotkey's token to re-register a claimed one | Register/update a neuron; returns `{uid, new, block, auth_token}` — `auth_token` only at the claiming call |
| `GET /neurons` | open | Current block + neuron list (settles emission first) |
| `POST /weights` | hotkey token + `validator` role | Tempo-gated `set_weights`; `{success, block, message}` |
| `GET /weights/{hotkey}` | open | Latest recorded vector for a hotkey (`vector: null` = none — a positive answer) |
| `POST /anchor` | hotkey token (or operator when unattributed) | Record a ≤128-byte commitment; `{txid, block}` (txid = `0x` + sha256(payload)[:16], `InMemoryChain` parity) |
| `POST /advance` | operator | Move the block clock forward |
| `GET /state` | open | Full sim state (the input contract of `build_report`) |
| `GET /report` | open | `build_report(state)` as JSON |
| `POST /report/write` | operator or any registered identity | Write `report-<ts>.json/.md` under `report_dir` |
| `POST /reset` | operator (and `enable_reset: true`) | Wipe history, rewind the clock |

Python surface: `ChainSim`, `ChainSimConfig`, `build_report`,
`render_markdown`, `write_report`, `decode_anchor_payload`. The
`register` / `submit_weights` / `anchor` / `advance` / `reset` methods on
`ChainSim` are TRUSTED in-process entry points (no auth check) — HTTP
callers only reach them through the authorizing handlers.

## Data & invariants

SQLite tables (WAL, `foreign_keys=ON`, `check_same_thread=False` — access is
serialized by the single event loop):

- `meta` — `start_time`, `advance_offset`, `emission_credited_until`,
  `operator_token_sha256` (+ source).
- `neurons` — `uid PK`, `hotkey UNIQUE`, coldkey, ip,
  `role IN ('miner','validator')`, `alpha_stake`, `emission_credited`,
  `registered_block`, `token_sha256`.
- `weight_calls` — accepted calls ONLY (seq, hotkey, block, version_key,
  sorted-key `vector_json`, created_at); tempo-rejected calls are not
  recorded.
- `anchors` — seq, txid, payload_hex, optional hotkey attribution, block.

Invariants:

- Weight vectors are validated finite and ≥ 0 per uid before acceptance.
- Emission settlement is deterministic and monotone: block *b* is governed
  by the latest recorded vector with record-block < *b*; history behind the
  credited watermark can never change.
- Anchor payloads are ≤ 128 bytes of hex-decodable data.
- A participant token is returned exactly once and never re-readable
  (hash-only storage); losing it means resetting the sim or claiming a new
  hotkey.

## Configuration

Section: `chainsim` (schema `config.py::ChainSimConfig`, `extra="forbid"`).
Env override pattern: `VIDAIO__CHAINSIM__<KEY>=<value>` (e.g.
`VIDAIO__CHAINSIM__OPERATOR_TOKEN=...`).

| Key | Default | Meaning |
|---|---|---|
| `port` | `8400` | HTTP API port (service port map: `vidaio/services/protocol.py`) |
| `metrics_port` | `9108` | Health/metrics port |
| `db_path` | `./data/chainsim.db` | SQLite state (restart-safe) |
| `block_seconds` | `1.0` | Wall-clock seconds per block (lazy clock) |
| `tempo` | `100` | Tempo gate: `set_weights` fails while `block <= last + tempo` |
| `emission_per_block` | `1.0` | Emission minted per block (proportional model above) |
| `enable_reset` | `true` | Allow `POST /reset` (disable for long-lived sims) |
| `operator_token` | `""` | Node credential for `/advance`, `/reset`, `/report/write`; empty = generated at first start, written to `<report_dir>/operator-token.txt` and logged once |
| `report_dir` | `./data/chain-reports` | Where `POST /report/write` drops reports (and the operator-token file) |

Defaults mirror the real chain where an analogue exists (tempo 100);
everything else is chosen for fast deterministic local runs (1 s blocks,
unit emission).

## How to test

```sh
python -m pytest tests/chainsim
```

Coverage by file: `test_registration.py` (uids, idempotent re-register),
`test_auth.py` (token model, takeover guard, roles, operator),
`test_blocks_and_tempo.py` (lazy clock, `/advance`, tempo parity),
`test_emission.py` (proportional crediting, burn, settlement determinism),
`test_adapter.py` (`HttpChainAdapter` end-to-end), `test_freshness.py`,
`test_report.py` (report content + decoding), `test_reset.py`,
`test_health_supervision.py` (`http_api` fail-fatal path),
`test_embedded_chain.py`. The full-stack path runs in `the development-tree e2e suite` and via
`the development-tree stack runner`.

## How to change safely

- Keep write semantics in lockstep with `vidaio.chain.InMemoryChain`
  (tempo message text, txid shape, 128-byte anchor cap): consumers are
  tested against both and the weight-setter string-matches `"tempo"` in
  failure messages to classify reschedules.
- Never record a rejected weight call, and never answer
  `GET /weights/{hotkey}` with `{}` where `null` is meant — an empty vector
  and a missing record are different chain states, and only the missing one
  can deny a weight intent.
- Emission model changes must preserve settlement determinism (a pure
  function of history + watermark + current block) or restarts change
  history.
- New mutations must be authorized in the HTTP handler (the service methods
  stay trusted); reads should stay open.
- Schema changes go through a new file in `migrations/` (applied by
  `vidaio.core.apply_migrations` at startup), never by editing shipped ones.

## Status & gaps

- [DONE] Registration/uids, lazy block clock, tempo gate, proportional
  emission, weights read-back, anchors, auth model, reports, restart-safety,
  health/fatal supervision, metrics.
- [NOT BUILT] (deliberately out of scope for the sim): uid recycling /
  deregistration, multi-validator stake-weighted consensus, stake dynamics,
  commit-reveal semantics, per-neuron incentive/dividend split. Model hotkey
  churn by registering new neurons; move stake by re-registering.
- [PENDING DECISION] None open in this module; real-chain behaviours the sim
  omits are covered by the real adapter ([`vidaio/chain`](../chain/README.md)).
