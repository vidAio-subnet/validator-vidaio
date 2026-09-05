# validator-vidaio

Thin-validator runtime and public verification source for **VidAIO, Bittensor
SN85 (finney)**. The validator fetches the authenticated epoch pointer, verifies
`sha256(epoch bytes) == pointer digest == on-chain anchor`, and submits the exact
authenticated `weight_u16` vector under its own registered validator hotkey.

> **Release mirror.** Each release is exported as a snapshot. See
> [docs/VALIDATING.md](docs/VALIDATING.md) for protocol details and
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the verification model.

## 1. Packaged roles

The public image **ghcr.io/vidaio-subnet/validator-vidaio** is built from this
repository's Dockerfile. It runs `thin-validator-node` (alias `weight-setter`).
It does not contain the private operational scripts or product services.

Auditor/scorer implementation is included for inspection, but **auditor launchers
are not packaged in this release**. This image has its own runtime/source digest;
it is not qualified to recompute the authority's scoring identity. A thin
validator verifies signed, anchored bytes; that is not an independent scoring
audit verdict. Do not launch an auditor using this quickstart.

## 2. Requirements

- Linux x86-64, Docker, a few CPU cores and persistent local disk.
- Your registered SN85 hotkey with a validator permit and transaction fees.
  Keep coldkeys offline. Never run two weight setters for the same hotkey.
- An archive-capable WSS chain endpoint and an NTP-synchronized clock.
- Per-validator S3 publication credentials supplied through subnet operations:
  read public evidence and write only `manifest/` and `weight_vector/`, with
  sealed prefixes explicitly denied. The live probe verifies a public round trip;
  operators must also enforce the IAM policy. No holdout decryption key is needed
  or permitted. The mainnet bucket is not anonymously writable.

## 3. Thin-validator quickstart

Pin the **public image digest from this repository's GitHub Release notes**.
Do not use the private monorepo image or a floating tag.

```sh
IMAGE=ghcr.io/vidaio-subnet/validator-vidaio@sha256:<release-digest>
docker pull "$IMAGE"
cp deploy/public/thin-validator.env.example .env.thin-validator
chmod 600 .env.thin-validator
```

Edit every `REPLACE_*` value in that role file. Both validator-hotkey fields must
be your own ss58. Keep the authority anchor hotkey as supplied; it is not your
wallet. Set your wallet name/hotkey, and stage only its hotkey JSON and public
wallet metadata beneath a dedicated absolute `HOTKEY_ROOT`. Do not mount your
entire local wallet directory if it contains coldkeys. Give container UID10001
read access to the staged hotkey, not broad filesystem permissions.

Create a dedicated absolute `STATE_ROOT` owned by UID/GID10001. Never change
permissions on a home directory, `.ssh` or `authorized_keys`. Keep both paths
consistent across recreations. The state directory includes SQLite journals and
the commitment writer lock; all writers using this hotkey must share that lock.

```sh
docker run --rm --read-only --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g --env-file .env.thin-validator "$IMAGE" \
  python scripts/production_preflight.py --print-floor
```

Set `VIDAIO__LOCAL_STACK__AUDITOR_CURSOR_FLOOR` to the returned
`auditor_cursor_floor`, exactly latest closed runtime epoch+1 on first deployment.
Then run the same image's full preflight:

```sh
docker run --rm --read-only --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g --env-file .env.thin-validator \
  -v "$HOTKEY_ROOT:/wallets:ro" -v "$STATE_ROOT:/var/lib/vidaio/state" "$IMAGE" \
  python scripts/production_preflight.py --config config/default.yaml --live
```

Require `MAINNET_THIN_VALIDATOR_PREFLIGHT_PASS`. The check verifies runtime
dependencies/manifest, finalized registration+permit, 7200-block archive access,
local hotkey signature, commitment capacity, fresh floor, authenticated pointer /
public bytes / independent anchor agreement, and a tiny public S3 publication
round trip. **It submits no chain transaction.** Its immutable probe object stays
in the publication prefix. A racing epoch close returns HOLD: refresh the floor
and retry, never disable anchor verification.

Only after a passing preflight and after stopping any old setter for this hotkey:

```sh
docker run -d --name vidaio-thin-validator --restart on-failure --init \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g --env-file .env.thin-validator \
  -v "$HOTKEY_ROOT:/wallets:ro" -v "$STATE_ROOT:/var/lib/vidaio/state" \
  -p 127.0.0.1:9102:9102 "$IMAGE" \
  python scripts/service_entrypoint.py thin-validator-node
```

The first-start floor is a preflight check, not a setting to advance on every
restart. Preserve the same durable state when restarting or upgrading.

## 4. Authentication and monitoring

Pointer requests are signed with your registered validator hotkey; no shared
authority bearer is required. Keep `verify_anchor=true` and `provider=shared`.
Follow the service logs, localhost9102 `/healthz` and `/metrics`, and the actual
on-chain weight vector. A healthy process alone does not prove a submitted vector.

## 5. Troubleshooting

| Symptom | Action |
|---|---|
| Preflight floor HOLD | Discover latest closed epoch again; use exactly+1 for fresh deployment |
| 403 / no validator permit | Verify the hotkey and current SN85 validator permit |
| Missing / mismatched anchor | HOLD; never disable the independent anchor check |
| Public storage probe fails | Check scoped IAM, region/endpoint and public prefix policy |
| Signature/clock refusals | Verify the wallet matches both ss58 fields and NTP is synchronized |
| Runtime/manifest check fails | Pull the exact public digest, not a locally altered runtime |
| Auditor service rejected | This release packages the thin runtime, not auditor launchers |

## 6. Upgrades and help

Keep durable state and the writer lock; upgrade to the published digest.
`VERSION` and epoch `version_key=16` fence incompatible releases. Do not change
tokenomics or sample/scoring knobs to bypass a failure.

Questions: GitHub issues or the validator channel via [vidaio.io](https://vidaio.io).
Include image digest, role, epoch and redacted HOLD/REFUSE logs, never secrets.

## License

MIT — see [LICENSE](LICENSE).
