# vidaio.autoupdater — CI-gated version shipping for your deployment

CI-gated deploys ("tests + golden dry-run gate
deploys"), local-first: a small service that polls a VERSION source and, on a
change, runs the deployment's own update command — gated on a recorded pass of
the release CI gate (run in the development tree). It keys off the repo
`VERSION` file exactly like the legacy subnet's autoupdater did, but it never
edits code itself and it ships closed (report-only) by default. An armed production
updater activates only a separately staged tree whose complete CI marker, runtime
manifest, and exact runtime bytes all verify immediately before execution.

## What it does

- **`config.py`** — [DONE] `AutoupdaterConfig` (the `autoupdater:` section):
  version source (`file` | `http`), poll cadence, the update command, the
  CI-pass/runtime-manifest gate, separately staged source root, downgrade policy,
  state file, metrics port.
- **`integrity.py`** — [DONE] deterministic full-source and shipped-runtime digests,
  a manifest of every exact runtime path/size/sha256, CI-marker verification, and
  fail-closed re-hashing of the staged tree.
- **`service.py`** — [DONE] `Autoupdater(BaseService)`: the poll loop,
  the version ordering rule (`version_key` / `compare_versions`), the ci-pass
  gate, subprocess execution of the update command with timeout + captured
  output + bounded retry, persistent `update_pending` / `update_failed` health
  signals, restart-safe state.

## Design & decisions

- **The autoupdater ships CODE versions. Champions go through the registry.**
  The spec's §20 diagram puts "champion promotion" in the same CI/CD box; the
  two are deliberately SEPARATE here and must stay that way:
  - *This service* watches the repo `VERSION` and triggers the fleet's own
    pull/restart mechanism — it moves the SERVICE CODE every node runs.
  - *Champion promotion* is `vidaio.registry.promotion.PromotionPipeline`: it
    derives the holdout winner from the competition database, verifies every
    packet/bundle/artifact against the audit store, and only then installs the
    champion executable as the reference/quality floor.

  Conflating them would let a version push slip in a serving model that
  never passed the promotion gates — the exact re-attribution §15's integrity
  invariants forbid. A champion is *data with provenance*; a version is *code with a CI
  pass*. Different evidence, different gate, different service.
- **Never triggers on first sight.** The baseline is the persisted state file,
  else the local `version_file` (what this deployment runs), else the first
  observed source version — adopted and persisted, never acted on. Only a
  subsequent CHANGE triggers.
- **The ordering rule (semver-ish, deliberately small).** Only the leading
  dotted numeric spine orders: `v1.2.10-rc1 -> (1, 2, 10)`, shorter spines are
  zero-padded (`1.2 == 1.2.0`). Above the baseline = upgrade; below = DOWNGRADE
  — refused with a CRITICAL log and a counted metric unless
  `allow_downgrade: true` (a rollback must be a human decision); equal spine
  with a different string (`0.1.0-rc1` -> `0.1.0`) = lateral, applied like an
  upgrade. No digits at all = empty spine, every change lateral.
- **The CI gate binds source identity and proves exact runtime bytes, not mtime.** A full
  run of the release CI gate (in the development tree) emits
  `runtime-release-manifest.json` plus `data/ci-pass`;
  partial/subset runs emit neither. The marker names the target
  `VERSION`, full-source sha256, shipped-runtime sha256, and manifest sha256. The
  manifest records `VERSION`, `pyproject.toml`, `uv.lock`, and every shipped path
  below `vidaio/`, `scripts/`, and `config/`, including each size and sha256. With
  `require_ci_pass` (the default), the updater requires the marker and manifest to
  agree, verifies the manifest itself, rejects missing/added/symlinked/changed runtime
  inputs, non-regular entries, bytecode, and ignored cache directories, and repeats the
  full verification immediately before **every** activation attempt. CI separately
  rechecks the complete source tree; the lean image and checkout produce byte-identical
  manifests, so `manifest-sha256` also matches the shipped image. Refusals remain
  pending; replacing the stage with the correct artifact can unblock a later poll.
- **Report-only default.** `update_command: []` means a change is logged,
  counted (`vidaio_autoupdater_version_changes_total`) and surfaced as
  `update_pending` on /health — and nothing executes. Arming the autoupdater is
  an explicit config act naming the deployment's own mechanism.
- **The command is the deployment's, with verified identity in its env.**
  It runs via subprocess (`capture_output`, `update_timeout_seconds`) with
  `VIDAIO_AUTOUPDATER_TARGET_VERSION`,
  `VIDAIO_AUTOUPDATER_TARGET_SOURCE_SHA256`,
  `VIDAIO_AUTOUPDATER_TARGET_RUNTIME_SHA256`, and
  `VIDAIO_AUTOUPDATER_STAGED_ROOT`. The command must switch the deployment to that
  already-staged root; the updater never pulls, copies, or edits release bytes. Exit 0 = applied: state is
  persisted, pending/failed clear. Non-zero/timeout = bounded retry
  (`update_retry_attempts` × `update_retry_delay_seconds`), then a PERSISTENT
  `update_failed` health signal; that version is not retried (a later version
  may trigger; success clears the flag). A command that restarts this very
  process should arrange the restart and exit 0, so state lands first.
- **Both sources are implemented.** `file` reads `version_file`; `http` GETs
  `version_url` (plain text, first line wins) — implemented and tested today so
  the fleet-push endpoint later is a config flip, not a code change. Under
  `http`, `version_file` still names the RUNNING version (the startup
  baseline). Active Bittensor production is narrower: it requires the HTTP source
  over a valid credential-free `https://` URL. File polling remains report/local use.
- **Health tells the truth.** `version_source` false while the source is
  unreadable; `update_pending` false while a detected change is unapplied
  (report-only included — "the fleet is behind" is the signal that mode
  exists to surface); `update_failed` false after an abandoned update, until a
  success.

## Public API

`Autoupdater` (BaseService; `poll_once()` is the whole policy as one awaitable
step, used directly by tests), `AutoupdaterConfig`, `version_key`,
`compare_versions`, `VersionSourceError`, `TARGET_VERSION_ENV`,
`TARGET_SOURCE_DIGEST_ENV`, `TARGET_RUNTIME_DIGEST_ENV`, and
`TARGET_STAGED_ROOT_ENV`.

Metrics (registry on the health server, port 9110):
`vidaio_autoupdater_polls_total`, `..._poll_failures_total`,
`..._version_changes_total`, `..._updates_applied_total`,
`..._updates_failed_total`, `..._downgrades_refused_total`,
`..._ci_gate_refusals_total`, `..._update_pending` (gauge).

## Configuration

Section: `autoupdater` (schema `config.py::AutoupdaterConfig`,
`extra="forbid"`). Env override pattern: `VIDAIO__AUTOUPDATER__<KEY>=<value>`.

| Key | Default | Meaning |
|---|---|---|
| `version_source` | `file` | `file` reads `version_file`; `http` GETs `version_url` |
| `version_file` | `./VERSION` | The source under `file`; the RUNNING version (startup baseline) under `http` |
| `version_url` | `""` | GET endpoint returning the version string; active Bittensor production requires credential-free HTTPS |
| `http_timeout_seconds` | `10.0` | Timeout for one version GET |
| `poll_seconds` | `60.0` | Poll cadence |
| `update_command` | `[]` | The deployment's own updater argv; `[]` = REPORT-ONLY. Active production requires an absolute, existing executable as argv[0] |
| `update_timeout_seconds` | `600.0` | Wall-clock budget per command run |
| `update_retry_attempts` | `3` | Command retry budget before `update_failed` |
| `update_retry_delay_seconds` | `5.0` | Delay between retries |
| `require_ci_pass` | `true` | Refuse unless marker, manifest, source identity, and exact staged runtime bytes verify |
| `source_root` | `.` | Runtime tree to verify; active production requires a separate absolute, existing non-symlink staging directory |
| `ci_pass_marker` | `./data/ci-pass` | Full-gate marker; a relative path resolves inside `source_root` |
| `runtime_manifest_file` | `runtime-release-manifest.json` | Exact staged-runtime manifest; a relative path resolves inside `source_root` |
| `allow_downgrade` | `false` | Permit a lower-ordered version (otherwise CRITICAL + refuse) |
| `state_file` | `./data/autoupdater-state.json` | Last applied version (restart safety) |
| `metrics_port` | `9110` | Health/metrics port |

## Production activation contract

The shipped default is intentionally inert:

```sh
python scripts/service_entrypoint.py autoupdater
```

With `autoupdater.update_command: []`, that role reports a discovered change but executes
nothing and needs no wallet or S3 secret. To arm it for Bittensor production:

1. Run the **full** release CI gate (in the development tree) against the target
   checkout. Retain the emitted
   `data/ci-pass` and `runtime-release-manifest.json` with the exact target tree.
2. Atomically install that target as a separate, read-only staging directory. It must be
   an absolute existing non-symlink path, must not be the currently running `/app`, and
   must contain distinct non-symlink marker/manifest files. Relative configured paths
   resolve beneath this root.
3. Configure `version_source: http`, a credential-free HTTPS `version_url`,
   `require_ci_pass: true`, `allow_downgrade: false`, the readable current-running
   `version_file`, and a non-empty `update_command` whose executable is absolute,
   existing, and executable.
4. Start the role and pass its role-scoped production guard. Only after the complete
   staged identity verifies should the operator publish the target version at the HTTPS
   source.
5. The activation command must switch service orchestration to
   `VIDAIO_AUTOUPDATER_STAGED_ROOT` and may independently record/compare the three
   supplied version/source/runtime identity variables. The updater re-verifies every
   staged byte before each attempt but does not implement that deployment switch itself.

Never build or mutate the staged tree in place after publication. Prepare a new complete
tree and atomically replace the staging target so readers see either the old artifact or
the new one, not a mixed release.

## How to test

```sh
python -m pytest tests/autoupdater
```

Coverage by file: `test_versioning.py` (the ordering rule), `test_integrity.py`
(manifest/marker agreement, exact paths and bytes, tamper/add/remove/symlink refusal),
and `test_service.py` (file + HTTP change detection, never-on-first-sight,
report-only mode, CI gating and just-in-time re-verification, command environment,
failure retry + persistent unhealthy, downgrade refusal + `allow_downgrade`, state
persistence across restarts, source-failure health). Production-guard integration tests
exercise the separate-stage, HTTPS, executable, and downgrade constraints.

## How to change safely

- Keep `poll_once()` the single policy path — every trigger decision must flow
  through the baseline compare, downgrade check and ci-pass gate in that order,
  or a test loses its meaning.
- Never make the autoupdater write code, config or champions; its only
  mutation is running the configured command and writing its own state file.
- The CI marker and runtime-manifest schemas are shared with the release CI gate
  (run in the development tree), the
  release Docker build, and `verify_release_dependencies.py`; change them together.
- The ordering rule is documented behavior — if it must grow (pre-release
  ordering, build metadata), update the module docstring, this README and
  `test_versioning.py` together.

## Status & gaps

- [DONE] File + HTTP sources, change detection, ordering rule, marker/manifest/runtime
  integrity gate, report-only mode, guarded command execution with verified identity
  environment and retry/timeout, downgrade refusal, persistent health signals,
  restart-safe state, metrics, and role-scoped production startup.
- [DEPLOYMENT-SPECIFIC] Each node runs its own updater against the shared HTTPS source.
  Operators must supply the absolute activation command and atomically stage the target
  tree plus its exact marker/manifest before publishing the new version. The source
  endpoint is HTTPS-authenticated but its plaintext version value has no separate
  application signature.
- [OUT OF SCOPE BY DESIGN] Champion shipping remains the registry
  `PromotionPipeline`; it never travels through this code-version updater.
