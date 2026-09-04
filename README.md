# validator-vidaio

Validation code for the **VidAIO subnet** (Bittensor **SN85**). A VidAIO
validator is deliberately **thin and verifying**: it mirrors the finalized
epoch log, re-verifies it against the on-chain anchor, and submits the
authenticated weight vector — while the auditors it runs alongside re-compute
the scoring on CPU from published, keyless evidence. Everything the validator
trusts, this repository lets you re-derive.

> **Release mirror.** Each VidAIO release lands here as one snapshot commit.
> Deep documentation: [docs/VALIDATING.md](docs/VALIDATING.md) (operations and
> failure modes), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 1. What a validator runs

Three cooperating roles (one box is fine to start):

| Role | What it does |
|---|---|
| **thin-validator** (weight-setter) | mirrors the epoch pointer → verifies `sha256(bytes) == pointer digest == on-chain anchor` → submits the authenticated `weight_u16` vector under your hotkey |
| **beacon-auditor** | walks every finalized epoch contiguously, re-computes scores from published evidence, submits signed CLEAN/DISPUTED verdicts |
| **own-auditor** | audits your own vector path end-to-end before you follow it |

The three-way digest verification is mandatory; a validator here never
"trusts an API" — it trusts bytes it re-hashed against the chain.

## 2. Requirements

- Linux x86-64, Python 3.12+, Docker recommended; a few CPU cores (all
  auditing is CPU-only by design) and ~50 GB disk for mirrored evidence.
- A Bittensor wallet with a **validator permit** on netuid 85 (enough stake to
  hold one) — `btcli` to create/register it (fresh hotkey; coldkey offline).
- An archive-capable chain endpoint for preflight and anchor reads.

## 3. Set up

```sh
pip install -e .

# every field documented in the per-section config models (vidaio/*/config.py)
export VIDAIO__CHAIN__NETWORK=finney         # or 'test' first
export VIDAIO__CHAIN__NETUID=85
export VIDAIO__CHAIN__VALIDATOR_HOTKEY=<your validator ss58>
export VIDAIO__WEIGHTSETTER__VALIDATOR_HOTKEY=<the same ss58>
export VIDAIO__WEIGHTSETTER__AUTHORITY_URL=<the authority pointer API>
export VIDAIO_HOTKEY_SEED=<hotkey seed — file/secret store, never argv>
```

**Run preflight before first start** — it fails closed on every
misconfiguration that matters (floor rule, anchor capacity, archive depth,
storage semantics). Preflight (and, in this build generation, the validator
runners themselves) run from the **canonical release image**,
`ghcr.io/vidaio-subnet/vidaio-next`, always pinned by `@sha256` digest — audit
identity binds the canonical runtime, so this is also the posture that lets
your verdicts count. Each release's exact digest is published in the GitHub
Release notes of this repository and must match what the fleet advertises;
never run a floating tag:

```sh
IMAGE=ghcr.io/vidaio-subnet/vidaio-next@sha256:<digest from the release notes>

docker run --rm --env-file <your role env> "$IMAGE" \
  python scripts/production_preflight.py --config config/default.yaml --live
```

One rule preflight enforces that deserves calling out: on a fresh start your
`auditor_cursor_floor` must be **exactly the latest closed runtime epoch + 1**
— preflight prints the exact value to configure.

Then:

```sh
docker run -d --env-file <role env> "$IMAGE" \
  python scripts/service_entrypoint.py thin-validator-node   # + auditor, own-auditor
```

(The source tree here is the full audit surface — read, diff, and verify it;
running the roles straight from source is on the roadmap as the runners are
extracted from the development tree's orchestration module.)

## 4. Authentication (registered-hotkey auth)

Validator-facing APIs verify you are a **registered hotkey with a validator
permit** — signed with the wallet you already hold, no extra secret:

- **Signed requests**: `X-Vidaio-Hotkey/-Timestamp/-Nonce/-Signature` headers
  (`vidaio.services.hotkey_auth.sign_request_headers` builds them).
- **Session tokens** for polling: `POST /auth/challenge` → sign the nonce →
  short-lived bearer token. Deregistration revokes within ~45 s.

Keep the box NTP-synced: the signed-request window is ±120 s.

## 5. What to monitor (and what each state means)

- **Weight submissions land**: your hotkey's on-chain vector updates each
  tempo, matching the authority vector (`docs/VALIDATING.md` convergence
  attempt).
- **Auditor cursor advances** one epoch at a time. A cursor that STOPS is
  meaningful, never cosmetic:
  - `HOLD` — something is temporarily unverifiable (unanchored yet,
    unreadable store). It retries; investigate if it persists.
  - `REFUSE` — bytes disagree with the on-chain anchor: tampering alarm.
    This should page a human.
  - **Outage gap** (schema v16): the authority declares epochs it could
    never anchor after downtime; your auditor verifies the declaration
    against the anchor and advances. A silent gap (404 with NO anchored
    declaration) holds forever — that distinction is the security property.
- Health endpoints: every role serves `/healthz` + Prometheus metrics.

## 6. Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| Preflight refuses the cursor floor | You copied a stale floor: set exactly `latest_closed + 1` (the error names it) and rerun |
| `commit-reveal`/submit never confirms | Check the netuid's commit-reveal posture; the adapter reports CR state at submit — see logs |
| Auditor HOLDs on one epoch for hours | Store/anchor genuinely unavailable, or the authority is withholding — check `/epoch/<id>` and the object store; a verified outage-gap declaration advances on its own |
| `403 hotkey_no_validator_permit` | Your hotkey lost its permit (stake churn) — check the metagraph |
| `503 hotkey_registry_unavailable` | The service's chain view is stale beyond its bound — check your chain endpoint |
| Weights submitted but "not matching" alarms | You may be on a different release than the fleet: `VERSION`/`version_key` is a lockstep fence — upgrade to the current release |
| Clock-skew signature refusals | NTP-sync; the window is ±120 s |

## 7. Upgrades

Releases are snapshot commits here; `VERSION` + `version_key` fence the fleet.
**Epoch-schema releases are lockstep** — release notes will say when every
validator must move in the same window. The mixed-version fence means a
foreign-schema log is refused rather than mis-followed; being behind fails
safe but fails visible.

## 8. Getting help

- **Discord** — the VidAIO server (invite via [vidaio.io](https://vidaio.io)):
  validator questions in the validators channel; the core team reads it daily.
- **GitHub issues on this repo** — include role, epoch id, and the exact HOLD/
  REFUSE log lines; every verdict is reproducible from public evidence.

## License

MIT — see [LICENSE](LICENSE).
