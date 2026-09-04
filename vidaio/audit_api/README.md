# vidaio.audit_api — the Audit Results API: the public honesty surface

The SECOND central surface of the two-API rework (the project design record
rule 10, the project design record §3.2). Where
the [Scoring Authority API](../authority/README.md) PUBLISHES the epoch log, this API
RECEIVES the auditors' independent verdicts on it. Each validator deployment POSTs both
its [full CPU `audit_mode=own_audit` review](../auditor/README.md) and its independent
[`audit_mode=beacon` duty](../auditor/README.md) as hotkey-signed
`AuditReport`s here (`:8710`); the service
verifies the signature, persists
it append-only (one per auditor+epoch+`audit_mode`; a divergent same-mode resubmission is a logged conflict),
and publishes the AGGREGATE across auditors. That aggregate is the investigation surface:
any misreporting by the central Scoring Authority becomes publicly visible as a DISPUTED
epoch, and an honest majority cannot out-vote one auditor's provable FAIL. Response
means public evidence, operator alerting, and manual remediation only: neither this
API nor any verdict is a weight-setting control plane.

It is the server side of the auditor's `AuditResultsClient` seam: report JSON in,
`{report_id, accepted}` out. Reads are OPEN — the honesty surface is meant to be public
(the owner-operated service reads it).

## What it does

- **`service.py`** — [DONE] `AuditResultsService` (`BaseService`): the FastAPI app
  (`POST /audit/report`, the read routes, `/healthz`), the bearer gate + signature
  verification, the disputed-epochs gauge, and the fatal-on-death lifecycle.
- **`verify.py`** — [DONE] the signature-verify seam: `ReportVerifier` (Protocol),
  `HotkeySignatureVerifier` (production), `RejectingVerifier` (the fail-closed default),
  `Sha256Verifier` (the explicit test/dev double); plus the registration seam
  `RegisteredHotkeys` (Protocol) with `FrozenRegisteredHotkeys` (fixed set) and
  `NoRegisteredHotkeys` (fail-closed default).
- **`store.py`** — [DONE] `AuditResultsStore`: append-only SQLite persistence of received
  reports + the conflict ledger (`RecordOutcome`, `RecordResult`, `StoredReport`).
- **`aggregate.py`** — [DONE] the pure aggregate functions (`epoch_status`,
  `epoch_rollup`, `feed_entry`) that RECOMPUTE the verdict from the stored reports.
- **`config.py`** — [DONE] `AuditResultsConfig` (the `audit_api:` section).
- **`client.py`** — [DONE] `HttpAuditResultsClient`: the auditor's real POST path
  (`AuditResultsConflict` / `AuditResultsUnavailable`).
- **`migrations/`** — [DONE] `0001_audit_reports.sql`, `0002_conflict_rejected_verdict.sql`,
  `0003_inconclusive_overall.sql` / `0004_conflict_inconclusive_verdict.sql` (admit
  `INCONCLUSIVE`), and `0005_audit_report_modes.sql` (data-preserving historical
  `beacon` backfill plus immutable `(auditor_hotkey, epoch_id, audit_mode)` reports).

## Design & decisions

### Fail-closed signature verification (`verify.py`)

Every `AuditReport` arrives hotkey-signed over `canonical_bytes()` (the report WITHOUT
its signature). The service re-derives those bytes from the parsed report and asks a
`ReportVerifier` whether the presented `auditor_signature` is valid FOR THE CLAIMED
`auditor_hotkey` — so an unsigned (401), badly-signed (403), or misattributed report is
rejected before it is ever persisted. Verifier selection is **FAIL-CLOSED**: an
explicitly injected verifier wins (a real `HotkeySignatureVerifier` in production);
otherwise the ONLY way to get the insecure `Sha256Verifier` double is to opt in via
`dev_insecure_verifier`; with neither, the default is `RejectingVerifier` — a
misconfigured deployment refuses every report rather than silently trusting tampered reports.

`HotkeySignatureVerifier` is the production contract, and authenticity is **two facts,
both required**: (1) the claimed `auditor_hotkey` is a **registered neuron on the subnet
metagraph**, and (2) an sr25519/ed25519 signature verifies against that hotkey's ss58. A
valid signature alone is **not** enough — anyone can mint a keypair and sign, so without
the registration check an unregistered key could sign a report claiming to be an auditor.
**Only a subnet-registered validator/auditor may submit**; a valid signature from an
unregistered hotkey is rejected (403). Registration is answered by an injected
`RegisteredHotkeys` seam — production backs it with the chain adapter's metagraph; tests
use `FrozenRegisteredHotkeys` (a fixed set). It defaults to the fail-closed
`NoRegisteredHotkeys` (reject everyone) when no provider is injected, and an unreadable
metagraph fails closed (rejected, never a 500). The bittensor `Keypair.verify` call is
deploy-time and lives behind an injectable `verify_fn` seam (the same isolation the chain
adapter uses), so the class imports and unit-tests without the SDK. In the shipped
Bittensor composition this is no longer an unwired deployment seam:
`the development-tree stack runner` constructs a wallet-free read-only chain
adapter, injects its live metagraph-backed registration provider, and lets the verifier use the lazy
`bittensor.Keypair.verify` implementation. The auditor side signs the same canonical
bytes with the wallet-backed `BittensorHotkeySigner`. The API reader never loads an
on-disk wallet or seed and rejects signing/extrinsic calls locally. Report mode deliberately swaps
only the signing primitive for its simulator equivalent while retaining the registration
check.

`AuditReport.audit_mode` binds the audit path to the signature: `beacon` is the default
independent beacon duty and `own_audit` is the separate full CPU review. Neither worker
runs inside or controls the weight-setter.
The non-default `own_audit` value is included in canonical/signature bytes; default
`beacon` deliberately retains the historical canonical bytes and report digests. Thus
the two modes from one hotkey can coexist for one epoch without being confused or
rewritten, and neither can impersonate the other.

### The aggregate RECOMPUTES verdicts — it never trusts `overall`

`GET /audit/status` returns the aggregate across every auditor that reported an epoch:
how many reported, how many CLEAN vs DISPUTED vs INCONCLUSIVE, the union of disputed
items + reason codes, and one epoch verdict. `auditors_reporting` counts distinct
hotkeys, while `reports_received` and `reports_by_mode` show the two mode reports.
Each report's effective verdict is
RECOMPUTED from its `item_verdicts` + `weight_verdict` via the same `overall_status` the
auditor uses (`_effective_verdict`), never read from the report's self-reported
`overall` — a report claiming CLEAN while carrying a FAIL item aggregates as DISPUTED.
**One provable fault ⇒ DISPUTED**, conclusive: it cannot be out-voted or buried by a
CLEAN first report, and even a DISPUTED *divergent* report (one that lost a conflict and
was not persisted) still flips the epoch. Epoch verdicts:

    UNAUDITED     — no auditor has reported this epoch yet
    DISPUTED      — a persisted OR a divergent report proved a fault (conclusive)
    CLEAN         — at least one report achieved recompute coverage and none disputed
    INCONCLUSIVE  — reports exist, none disputed, none clean → every one was
                    INCONCLUSIVE (nothing recomputed): a needs-attention state, NOT
                    washed to CLEAN (#8)

The same rule is applied at WRITE time too (`store._recomputed_overall`), so the
persisted `overall` column, the disputed-epochs gauge, and the feed cannot be fooled by
an inconsistent report either.

### Append-only persistence + the conflict ledger

`record` writes one immutable row per `(auditor_hotkey, epoch_id, audit_mode)`. A re-post of the
IDENTICAL report (same `report_id` = report digest) is idempotent (200). A DIFFERENT
report for an already-reported triplet is a CONFLICT (409): the FIRST report is KEPT and the
divergence is logged in `audit_report_conflicts` — the conflict is itself a signal,
surfaced alongside the verdict, never folded into it or silently overwritten. The
rejected report's recomputed verdict is stored on the conflict row so a divergent
DISPUTED report still marks the epoch DISPUTED even though it was not persisted. Reports
and conflicts are append-only; in-database triggers enforce it against direct SQL. The
full report JSON is stored, so any read can reconstruct the exact `AuditReport` and
re-verify its signature / re-derive the aggregate.

### Boundary

`POST /audit/report` is bearer-gated on `audit_api.api_token` (validators carry it: 401
missing/malformed, 403 wrong) AND signature-verified. All READS are open. `/healthz` is
open. The API's fatal-on-death lifecycle follows the exit-code contract. Modes (rule 8):
the same service code runs everywhere — only the injected `ReportVerifier` differs.

## Public API & endpoints

HTTP (port `audit_api.http_port`, default 8710; metrics 9112):

| Route | Contract |
|---|---|
| `POST /audit/report` | a signed `AuditReport` (`audit_mode=beacon|own_audit`) → `{report_id, accepted}`. 201 new, 200 idempotent re-post, 409 same-mode conflict (first kept), 401 unsigned / no bearer, 403 bad signature / unregistered auditor / wrong bearer. Bearer-gated |
| `GET /audit/status?epoch_id=` | the aggregate honesty status for one epoch (verdict, distinct-hotkey count, total/mode report counts, disputed items + reason codes with `audit_mode`, snapshot digests, conflicts). Open |
| `GET /audit/epochs` | a compact per-epoch rollup row for every reported epoch, newest first. Open |
| `GET /audit/feed?limit=` | the newest reports as dashboard rows (summary + failures). Open |
| `GET /healthz` | liveness, unauthenticated |

Python surface (`__init__.py`): `AuditResultsService`, `AuditResultsConfig`,
`AuditResultsStore`, `StoredReport`, `RecordResult`, `RecordOutcome`, `epoch_status`,
`epoch_rollup`, `feed_entry`, `EPOCH_CLEAN`, `EPOCH_DISPUTED`, `EPOCH_INCONCLUSIVE`,
`EPOCH_UNAUDITED`, `ReportVerifier`, `HotkeySignatureVerifier`, `Sha256Verifier`,
`RejectingVerifier`, `RegisteredHotkeys`, `FrozenRegisteredHotkeys`,
`NoRegisteredHotkeys`, `HttpAuditResultsClient`, `AuditResultsConflict`,
`AuditResultsUnavailable`. Construction seams: `store=`, `verifier=`, `now=`.

## Data & invariants

- **`audit_reports`** — one immutable row per `(auditor_hotkey, epoch_id, audit_mode)` (`report_id`,
  the digests, the recomputed `overall`, coverage counts, the full `report_json`,
  timestamps). **`audit_report_conflicts`** — the divergence ledger, carrying the
  rejected report's recomputed verdict. Both append-only via triggers.
- The persisted `overall` is ALWAYS the recomputed verdict, never the report's
  self-reported field — the gauge, feed, and `/audit/status` all agree by construction.
- `report_id` == the auditor's `report_digest()`, so it matches the `SubmitAck` the
  auditor already holds.
- Status and feed rows expose `audit_mode`; `auditors_reporting` remains the count of
  distinct hotkeys, not the number of mode reports.
- A report is never accepted unverified (fail-closed); a conflict never overwrites the
  first verdict.

## Configuration

Section: `audit_api` (schema `config.py::AuditResultsConfig`, `extra="forbid"`). Env
override pattern: `VIDAIO__AUDIT_API__<KEY>=<value>`.

| Key | Default | Meaning |
|---|---|---|
| `http_host` / `http_port` | `0.0.0.0` / `8710` | Report-ingest + honesty-read API bind |
| `metrics_port` | `9112` | Health/metrics port |
| `db_path` | `./data/audit_results.db` | Append-only report store + conflict ledger (own migrations) |
| `api_token` | `null` | Bearer gating `POST /audit/report`. Null = OPEN — loopback/dev only; PRODUCTION MUST SET IT. Reads always open |
| `dev_insecure_verifier` | `false` | FAIL-CLOSED default: with no verifier injected the service uses `RejectingVerifier`. Set true ONLY for chainless/dev runs to opt into the `Sha256Verifier` double. PRODUCTION never sets this (it injects a real `HotkeySignatureVerifier`) |
| `verifier_secret` | `""` | The shared secret the `Sha256Verifier` double checks — used ONLY when `dev_insecure_verifier` is set; ignored in production |
| `feed_default_limit` / `feed_max_limit` | `50` / `500` | `GET /audit/feed` sizing |

## How to test

```sh
python -m pytest tests/audit_api
```

By concern: `test_verify.py` (fail-closed default, the three verifiers, misattribution
rejected), `test_service.py` (bearer gate, unsigned/bad-signature rejection, the derived
`overall` at ingest, the fatal lifecycle), `test_status.py` (aggregate recompute, one
FAIL ⇒ DISPUTED, divergent-DISPUTED conflicts flipping the epoch, INCONCLUSIVE handling),
`test_client_roundtrip.py` (the `HttpAuditResultsClient` round trip incl. 409).

## How to change safely

- Verifier selection MUST stay fail-closed: never make the absence of a verifier accept.
- The aggregate MUST keep RECOMPUTING each report's verdict from its item + weight
  verdicts — never trust `overall` off the wire, at read time OR write time.
- Keep "one provable fault ⇒ DISPUTED" absolute, including for divergent (rejected)
  reports — a CLEAN first report must never bury a later DISPUTED one.
- Reports and conflicts are append-only; a conflict keeps the first verdict and records
  the divergence, never overwrites. Schema changes are new migration files.
- INCONCLUSIVE is its own state — never collapse it into CLEAN or DISPUTED.

## Status & gaps

- [DONE] Signature-verified ingest, append-only store + conflict ledger, the aggregate
  investigation/alerting surface, the disputed-epochs gauge, the HTTP client — fully tested.
- [DONE, needs testnet validation] Bittensor-mode signing and verification are wired:
  reports are signed by the loaded validator hotkey, verified with
  `bittensor.Keypair.verify`, and accepted only when the same hotkey is present in the
  wallet-free read-only chain adapter's metagraph. Missing SDK/metagraph state fails
  closed; the Audit Results API does not require or load a wallet. This path
  has unit/integration coverage but has not yet received a report from a live testnet
  wallet.
- [PENDING DECISION] On-chain / IPFS mirroring of reports for a fully trustless surface
  (the `ReportVerifier` Protocol already leaves room for an anchored check);
  `the project design record` §8.
