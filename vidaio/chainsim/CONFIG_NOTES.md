# chainsim and chain-factory configuration notes

The shipped production configuration selects the real adapter:

```yaml
chain:
  mode: bittensor
  network: finney
  netuid: 85
  version_key: 15
```

`the development-tree config overlay`, `the development-tree config overlay`, and test/e2e overlays select
`chain.mode: report` explicitly. The `ChainConfig` model also defaults to report mode so
a bare test construction with no loaded YAML remains chainless; that model convenience
does not override the shipped production file.

## Report-mode configuration

```yaml
chain:
  mode: report
  chainsim_url: http://127.0.0.1:8400   # or "embedded"
  validator_hotkey: local-validator
  anchor_hotkey: ""                    # empty => validator_hotkey
  auth_token: ""
  report_dir: ./data/chain-reports

chainsim:
  port: 8400
  metrics_port: 9108
  db_path: ./data/chainsim.db
  block_seconds: 1.0
  tempo: 100
  emission_per_block: 1.0
  enable_reset: true
  operator_token: ""
  report_dir: ./data/chain-reports
```

Report mode drives the same service code as Bittensor mode. Only the `ChainAdapter`
implementation changes: an HTTP chainsim adapter, or an embedded append-only journal for
single-process harnesses.

## Authorization

Reads are open in the local simulator; mutations require identity or operator authority:

- `chain.auth_token` proves ownership of `chain.validator_hotkey` for weights and
  anchors. If a process registers itself, `HttpChainAdapter.register()` captures the
  issued token. If another process performs fleet registration, configure the same token
  in both places.
- `chainsim.operator_token` gates `/advance`, `/reset`, and `/report/write`. If empty, the
  simulator creates it at first start, writes it owner-only below `report_dir`, and logs it
  once. Pin it for a repeatable long-lived simulator.

The local stack and compose overlays contain dev-only credentials. They must never be
copied into a Bittensor deployment.

## Historical behavior

The simulator persists blocks, neurons, weights, anchors, emission, and historical
metagraph views in SQLite. It implements the current schema-v15 production seams (including
the v14-introduced total replay structure) used by the
shared code, including:

- `neurons_at(close_block)` for exact registered-census binding;
- chain-derived tempo and report-mode burn UID behavior;
- anchor reads plus the recorded anchor block used by archive-style verification; and
- authenticated registration, weight, and anchor writes.

This makes report mode useful for deterministic integration tests, but it does not prove
the live SDK, archive-node behavior, finality, runtime epoch schedule, or commitment
pallet. Report mode intentionally retains its fixed tempo grid; Bittensor production
instead accepts only GRANDPA-finalized, archive-proven `SubnetEpochIndex` transitions
with matching `LastEpochBlock`. Those require the testnet ladder in
the project design record.

All keys are environment-overridable with `VIDAIO__<SECTION>__<KEY>`, for example
`VIDAIO__CHAIN__MODE=report` or `VIDAIO__CHAINSIM__TEMPO=20`.
