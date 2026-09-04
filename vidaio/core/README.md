# vidaio.core — shared foundation: config, DB, logging, metrics, resilience

The one module every service imports and the only cross-cutting dependency in the
tree. Module-specific configuration models live in each module (e.g.
`vidaio/tokenomics/config.py`); core provides the loading/layering machinery, the
SQLite discipline, structured logging, the health/metrics server, and the bounded
retry/timeout primitives. See the [root README](../../README.md) for how the
services compose on top of it.

## What it does

- **Layered configuration** (`config.py`): `load_raw_config(path)` reads the YAML
  file (`config/default.yaml`) if present, then applies environment overrides on
  top. Any key at any depth is overridable as `VIDAIO__SECTION__KEY=value` — the
  double-underscore path is split, lowercased, and the value is parsed with
  `yaml.safe_load` (so `VIDAIO__TOKENOMICS__EWMA_DECAY=0.5` arrives as a float,
  `VIDAIO__SCORING__REQUIRE_SECONDARY_VMAF=false` as a bool). Each service then
  pulls exactly its own section with `section(raw, "tokenomics", TokenomicsConfig)`,
  which pydantic-validates it against the module-owned model. `CoreConfig` holds
  only the process-level settings every service shares (data_dir, db_filename,
  log_level, metrics_port, network, netuid) [DONE].
- **SQLite + atomic migration runner** (`db.py`): `connect()` opens every database
  with WAL, `synchronous=NORMAL`, `foreign_keys=ON`, and a 5s busy timeout.
  `apply_migrations(conn, dir)` applies `*.sql` files in sorted order, recording
  each in `schema_migrations`. Two properties matter:
  - **Statement splitting via `sqlite3.complete_statement`** — semicolons inside
    trigger bodies (`BEGIN ... ; ... END;`), strings and comments do not split a
    statement. This exists because `executescript()` auto-commits, which would
    break per-file transactional application, and several modules (challenge,
    audit, competition) rely heavily on triggers.
  - **Per-file atomicity** — each migration's statements AND its
    `schema_migrations` row commit together inside one `BEGIN IMMEDIATE`; any
    failure rolls the whole file back, so a crash leaves neither partial schema
    nor a phantom ledger row. [DONE]
- **Structured JSON logging** (`logging.py`): one JSON object per line to stdout,
  never a file. `bound(**fields)` binds contextvar fields (round/contender/batch
  ids) to every line in a block; `log_fields(...)` attaches per-call fields via
  `extra=`. The module docstring states the rule the whole repo follows:
  **secrets and PATs must never be passed as field values.** [DONE]
- **HealthServer** (`metrics.py`): a small threaded HTTP server every service
  runs, exposing `GET /health` (JSON `{service, status, checks}`; 200 when all
  registered checks pass, 503 otherwise — a check that raises counts as failed)
  and `GET /metrics` (Prometheus exposition from the service's own
  `CollectorRegistry`). `port=0` + `bound_port` supports tests. [DONE]
- **Resilience** (`resilience.py`): `retry_async` (bounded exponential backoff
  with jitter in [0.5, 1.0)x, raises `RetriesExhausted` with the last error as
  cause) and `with_timeout(coro, seconds, name)` (raises a named `TimeoutError`).
  The stated rule: every network/subprocess boundary goes through one of these —
  an unbounded await is a bug. [DONE]

## Design & decisions

- Core is deliberately tiny and dependency-light; the docstring of
  `vidaio/core/__init__.py` pins the rule that services import from here and
  nowhere else in core, and that config *schemas* live with their owning module —
  so a config edit is reviewed next to the code it drives.
- The migration runner's transaction/splitting design exists to support the
  trigger-heavy, append-only schemas mandated by the auditability requirements in
  the project design record (integrity invariants: every scored metric
  independently recomputable) — see [vidaio/audit](../audit/README.md) and
  [vidaio/competition](../competition/README.md) for the schemas that depend on it.
- No wall-clock or randomness policy: core does not enforce it, but the modules
  built on top (tokenomics, challenge, competition) all take `now` and `rng` as
  explicit arguments; core keeps itself compatible by never hiding a clock in the
  DB layer (timestamps in `schema_migrations` are the only `datetime('now')`).

## Public API (`vidaio/core/__init__.py`)

Config
- `CoreConfig` — process-level settings shared by every service.
- `load_raw_config(path)` — YAML file → dict, with `VIDAIO__…` env overrides applied.
- `section(raw, name, Model)` — validate one named section into a module's model.

Database
- `connect(path)` — SQLite connection with WAL/FK/busy-timeout pragmas.
- `apply_migrations(conn, migrations_dir)` — ordered, atomic, ledgered migrations.

Logging
- `setup_logging(level)` — install the JSON stdout handler on the root logger.
- `get_logger(name)` — standard named logger.
- `bound(**fields)` — context manager binding fields to every line in the block.
- `log_fields(**fields)` — helper for `logger.info("msg", extra=log_fields(...))`.

Observability
- `HealthServer` — threaded `/health` + `/metrics` server with registrable checks.

Resilience
- `retry_async(fn, attempts=…, …)` / `RetriesExhausted` — bounded jittered backoff.
- `with_timeout(coro, seconds, name)` — named timeout wrapper.

## Data & invariants

- `schema_migrations(name PRIMARY KEY, applied_at)` is the only table core owns.
  Migration files are **applied once and never re-executed**; a recorded file is
  skipped forever, so fixing a bad migration means writing a new file, never
  editing an applied one (editing an unapplied/failed one is fine — the failed
  run rolled back completely).
- Migration files must be complete SQL: a trailing incomplete statement raises
  `ValueError` before anything runs.
- Env overrides only merge into dict-shaped nodes; an override path that collides
  with a non-dict YAML value is silently skipped (the YAML value wins).

## How to test

```
.venv/bin/pytest tests/core
```

Notable tests:
- `test_db.py` — proves WAL/pragma setup, apply-once semantics, full rollback of
  a failed migration (`test_failed_migration_is_fully_rolled_back`), that trigger
  bodies survive statement splitting, and that the migration commit is atomic
  with its ledger row.
- `test_config.py` — YAML + section loading and `VIDAIO__…` env override parsing.
- `test_metrics.py` — live `/health` + `/metrics` endpoints; a failing check
  degrades to 503.
- `test_resilience.py` — retry success after failures, exhaustion raising with
  the cause chained, and timeouts naming the operation.
- `test_service_exit_contract.py` also lives in `tests/core` — it covers the
  service base-class exit contract (`vidaio/services/base.py`, documented in that
  package's README): cooperative stop exits 0, a fatal failure flips health and
  exits non-zero, the first fatal reason wins.

## How to change safely

- **Adding a config key**: add it to the owning module's config model (not here);
  it becomes env-overridable automatically. Only process-level settings belong in
  `CoreConfig`.
- **Migration discipline (load-bearing for every module)**: new schema changes are
  new `NNNN_*.sql` files, ordered by filename; never edit an applied file. Write
  files to be self-contained — the runner gives you atomicity per file, nothing
  across files.
- **`_split_statements`** is load-bearing for every trigger-bearing migration in
  the repo (challenge, audit, competition). Changing it requires the trigger-body
  splitting test to keep passing.
- **JSON log shape** (`ts`/`level`/`logger`/`message` + bound/extra fields) is what
  a log store will index; treat field names as an interface.
- `HealthServer` responses (`/health` JSON shape, 200/503 semantics) are probed by
  service health checks; keep the payload keys stable.

## Status & gaps

- [DONE] Everything in this module is implemented and tested; there are no stubs.
- Known limitation: env overrides cannot replace a YAML scalar with a mapping
  (non-dict nodes stop the merge), and list-valued keys can only be overridden
  wholesale with YAML syntax (e.g. `VIDAIO__CHALLENGE__TRACKS='[compression]'`).
- `HealthServer` binds `0.0.0.0` by default; nothing in core authenticates these
  endpoints — deployment-level network policy is assumed.
