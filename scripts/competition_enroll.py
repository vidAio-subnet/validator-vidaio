#!/usr/bin/env python3
"""Enter a VidAIO competition with your registered miner hotkey.

    python scripts/competition_enroll.py list   --url https://<competitions-host>
    python scripts/competition_enroll.py enroll --url https://<competitions-host> \
        --competition-id <id> --repo-url https://github.com/you/solution.git \
        --commit <40-hex commit sha> --tree <40-hex tree sha> \
        --wallet-name <coldkey name> --wallet-hotkey <hotkey name>

What happens on `enroll`:
  * the request body is only {repo_url, commit_sha, tree_sha};
  * it is signed with your HOTKEY (sr25519) over method, path, body hash, a timestamp
    and a nonce — your coldkey is never touched and nothing secret leaves this machine;
  * the validator checks the signature, that the hotkey is registered on the subnet,
    and that its alpha stake (read from the chain, not from you) clears the floor shown
    by `list`.

Pin exactly what you want evaluated:
    git rev-parse HEAD 'HEAD^{tree}'
The repository must be reachable over https on an allowed host (see `list`). A private
repository works when the competition's read-only reader account has been granted
access; ask in the subnet Discord for that account name.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _WalletSigner:
    """Adapts a bittensor hotkey keypair to the request-signing helper."""

    def __init__(self, keypair: object) -> None:
        self._keypair = keypair
        self.hotkey = str(keypair.ss58_address)  # type: ignore[attr-defined]

    def sign(self, payload: bytes) -> str:
        return self._keypair.sign(payload).hex()  # type: ignore[attr-defined]


def _request(method: str, url: str, *, body: bytes | None = None, headers: dict | None = None):
    request = urllib.request.Request(url, data=body, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"raw": raw[:500]}


def cmd_list(args: argparse.Namespace) -> int:
    status, body = _request("GET", f"{args.url.rstrip('/')}/v1/competitions")
    if status != 200:
        print(json.dumps(body, indent=1))
        return 1
    print(f"enrollment stake floor: {body.get('min_enroll_alpha_stake')} alpha")
    print(f"allowed repository hosts: {', '.join(body.get('allowed_repo_hosts', []))}")
    for comp in body.get("competitions", []):
        manifest = comp.get("manifest", {})
        print(
            f"\n{comp['competition_id']}  [{comp['track']}]  {comp['status']}"
            f"{'  <- enrollment OPEN' if comp.get('enrollment_open') else ''}\n"
            f"  enrollment: {comp['start_time']} -> {comp['enrollment_deadline']}\n"
            f"  evaluation ends: {comp['end_time']}\n"
            f"  stake floor: {manifest.get('minimum_alpha_stake')}  "
            f"vmaf threshold: {manifest.get('vmaf_threshold')}  "
            f"clips: {len(manifest.get('evaluation_item_commitments') or []) or '?'}\n"
            f"  sandbox: {manifest.get('sandbox_resources') or 'operator defaults'}  "
            f"gpus: {manifest.get('allowed_gpus')}\n"
            f"  result rules: {manifest.get('result_rules') or 'protocol defaults'}\n"
            f"  enrolled: {len(comp.get('enrolled', []))}"
        )
    return 0


def cmd_enroll(args: argparse.Namespace) -> int:
    from vidaio.services.hotkey_auth import sign_request_headers

    try:
        import bittensor
    except ImportError:
        raise SystemExit("bittensor is required to sign with your hotkey") from None
    wallet_kwargs = {"name": args.wallet_name, "hotkey": args.wallet_hotkey}
    if args.wallet_path:
        wallet_kwargs["path"] = args.wallet_path
    wallet = bittensor.wallet(**wallet_kwargs)
    signer = _WalletSigner(wallet.hotkey)

    path = f"/v1/competitions/{args.competition_id}/enroll"
    body = json.dumps(
        {"repo_url": args.repo_url, "commit_sha": args.commit, "tree_sha": args.tree},
        separators=(",", ":"),
    ).encode()
    headers = sign_request_headers(signer, method="POST", path=path, body=body)
    headers["Content-Type"] = "application/json"
    print(f"enrolling hotkey {signer.hotkey} in {args.competition_id}")
    status, response = _request("POST", f"{args.url.rstrip('/')}{path}", body=body, headers=headers)
    print(json.dumps(response, indent=1))
    return 0 if status == 201 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="show competitions, floors and resources")
    listing.add_argument("--url", required=True)
    listing.set_defaults(func=cmd_list)
    enroll = sub.add_parser("enroll", help="enroll your hotkey with a pinned repository")
    enroll.add_argument("--url", required=True)
    enroll.add_argument("--competition-id", required=True)
    enroll.add_argument("--repo-url", required=True)
    enroll.add_argument("--commit", required=True)
    enroll.add_argument("--tree", required=True)
    enroll.add_argument("--wallet-name", required=True)
    enroll.add_argument("--wallet-hotkey", required=True)
    enroll.add_argument("--wallet-path")
    enroll.set_defaults(func=cmd_enroll)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
