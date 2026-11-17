#!/usr/bin/env python3
"""Enter a VidAIO competition with your registered miner hotkey, change or withdraw the
entry before the deadline, and check it.

    # what is open, with floors, resources and what the entry checks enforce
    python scripts/competition_enroll.py list --url https://95.133.254.122/competitions

    # enroll the commit checked out in this clone (HEAD and HEAD^{tree} by default)
    python scripts/competition_enroll.py enroll \
        --url https://95.133.254.122/competitions --competition-id <id> \
        --repo-url https://github.com/you/solution.git \
        --wallet-name <coldkey name> --wallet-hotkey <hotkey name>

    # a better commit before the deadline: push it, then the same command with
    python scripts/competition_enroll.py enroll ... --replace

    # your entry: status, full reasons, the pre-deadline fetch check and trial build
    python scripts/competition_enroll.py status \
        --url https://95.133.254.122/competitions --competition-id <id> \
        --wallet-name <coldkey name> --wallet-hotkey <hotkey name>
    # or the public summary of any hotkey, no wallet needed
    python scripts/competition_enroll.py status \
        --url https://95.133.254.122/competitions --competition-id <id> --hotkey <ss58>

    # take the entry out before the deadline (final for this hotkey in this competition)
    python scripts/competition_enroll.py withdraw \
        --url https://95.133.254.122/competitions --competition-id <id> \
        --wallet-name <coldkey name> --wallet-hotkey <hotkey name> --yes

What `enroll` does:
  * pins the commit and tree from your clone (--repo, default "."): --commit defaults to
    HEAD and --tree to that commit's tree; values you pass are checked against the clone;
  * runs local checks and prints them: the tree belongs to the commit, the commit is on
    --repo-url (git ls-remote; a warning when it cannot tell, e.g. a private repo),
    a Dockerfile at the root, LICENSE and LICENSES.md at the root when the competition
    requires them, no symlinks or submodules, and the source tree within the cap.
    A failed check stops here, before anything is signed (--skip-local-checks overrides);
  * sends {repo_url, commit_sha, tree_sha, replace} signed with your HOTKEY (sr25519)
    over method, path, body hash, a timestamp and a nonce. Your coldkey is never touched
    and nothing secret leaves this machine;
  * the validator checks the signature, that the hotkey is registered on the subnet,
    and that its alpha stake (read from the chain, not from you) clears the floor shown
    by `list`. The stake is read again when entries are finalized (the manifest's
    finalization_time, after the enrollment deadline): an entry whose hotkey is below
    the floor then is rejected (`status` warns about it before). Keep it staked until then.

One entry per hotkey per competition. Before the enrollment deadline --replace changes
it in place (same contender id), and `withdraw` takes it out for good. After enrolling,
the validator fetches your entry ahead of the deadline with its real reader account and
reports problems in `status` (it never rejects you for them before the deadline): fix
them and --replace in time. `status` also shows the entry commitment, sha256 of
"<competition_id>\\n<hotkey>\\n<repo_url>\\n<commit_sha>\\n<tree_sha>", which the public
listing shows too, so you can confirm it carries your latest replacement.

The repository must be reachable over https on an allowed host (see `list`). A private
repository works when the competition's read-only reader account has been granted
access; ask in the subnet Discord for that account name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Defaults when the competition cannot be read (the validator's own values win).
DEFAULT_MAX_WORKTREE_BYTES = 512 * 1024 * 1024
LICENSE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")
LICENSES_MANIFEST = "LICENSES.md"
GIT_TIMEOUT_SECONDS = 60


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


def _load_signer(args: argparse.Namespace) -> _WalletSigner:
    if not args.wallet_name or not args.wallet_hotkey:
        raise SystemExit("--wallet-name and --wallet-hotkey are required to sign")
    try:
        import bittensor
    except ImportError:
        raise SystemExit("bittensor is required to sign with your hotkey") from None
    wallet_kwargs = {"name": args.wallet_name, "hotkey": args.wallet_hotkey}
    if args.wallet_path:
        wallet_kwargs["path"] = args.wallet_path
    # bittensor >= 10 exposes ``Wallet``; older releases only the lowercase factory.
    wallet_factory = getattr(bittensor, "Wallet", None) or getattr(bittensor, "wallet")
    wallet = wallet_factory(**wallet_kwargs)
    return _WalletSigner(wallet.hotkey)


def _signed(args: argparse.Namespace, method: str, action: str, body: bytes = b""):
    from vidaio.services.hotkey_auth import sign_request_headers

    signer = _load_signer(args)
    path = f"/v1/competitions/{args.competition_id}/{action}"
    headers = sign_request_headers(signer, method=method, path=path, body=body)
    if body:
        headers["Content-Type"] = "application/json"
    status, response = _request(
        method, f"{args.url.rstrip('/')}{path}", body=body or None, headers=headers
    )
    return signer, status, response


def _competition(args: argparse.Namespace) -> tuple[int, dict]:
    return _request("GET", f"{args.url.rstrip('/')}/v1/competitions/{args.competition_id}")


def entry_commitment(competition_id: str, hotkey: str, repo_url: str, commit: str, tree: str) -> str:
    preimage = "\n".join((competition_id, hotkey, repo_url, commit, tree))
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


# ---- list ------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    status, body = _request("GET", f"{args.url.rstrip('/')}/v1/competitions")
    if status != 200:
        print(json.dumps(body, indent=1))
        return 1
    print(f"enrollment stake floor: {body.get('min_enroll_alpha_stake')} alpha")
    print(f"allowed repository hosts: {', '.join(body.get('allowed_repo_hosts', []))}")
    for comp in body.get("competitions", []):
        manifest = comp.get("manifest", {})
        checks = comp.get("entry_checks") or {}
        print(
            f"\n{comp['competition_id']}  [{comp['track']}]  {comp['status']}"
            f"{'  <- enrollment OPEN' if comp.get('enrollment_open') else ''}"
            f"{'  (PAUSED)' if comp.get('paused') else ''}\n"
            f"  enrollment: {comp['start_time']} -> {comp['enrollment_deadline']}\n"
            f"  evaluation ends: {comp['end_time']}\n"
            f"  stake floor: {manifest.get('minimum_alpha_stake')}  "
            f"vmaf threshold: {manifest.get('vmaf_threshold')}  "
            f"clips: {len(manifest.get('evaluation_item_commitments') or []) or '?'}\n"
            f"  sandbox: {manifest.get('sandbox_resources') or 'operator defaults'}  "
            f"gpus: {manifest.get('allowed_gpus')}\n"
            f"  result rules: {manifest.get('result_rules') or 'protocol defaults'}\n"
            f"  entry checks: LICENSE + LICENSES.md "
            f"{'required' if checks.get('licenses_required') else 'not required'}, "
            f"source tree <= {checks.get('max_worktree_bytes', DEFAULT_MAX_WORKTREE_BYTES)} "
            f"bytes, trial build {'on' if checks.get('trial_build') else 'off'}\n"
            f"  enrolled: {len(comp.get('enrolled', []))}"
        )
    return 0


# ---- pins and local checks -------------------------------------------------------

_SHA = re.compile(r"[0-9a-f]{40}")


def check_shas(commit: str, tree: str, repo: str | None) -> None:
    """Refuse the mistakes that lose an entry at finalization, before anything is signed:
    a malformed SHA, the commit SHA given as the tree, or (with --repo) a tree that is not
    the commit's."""
    for name, value in (("--commit", commit), ("--tree", tree)):
        if not _SHA.fullmatch(value):
            raise SystemExit(f"{name} must be a full 40-character lowercase SHA, got {value!r}")
    if commit == tree:
        raise SystemExit(
            "--tree is the commit SHA; it must be the commit's TREE: git rev-parse <commit>^{tree}"
        )
    if repo is None:
        return
    actual = _rev_parse(repo, f"{commit}^{{tree}}")
    if actual != tree:
        raise SystemExit(f"--tree {tree} is not the tree of commit {commit} in {repo} (it is {actual})")


def _git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    done = subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True,
        timeout=GIT_TIMEOUT_SECONDS, env=env,
    )
    if check and done.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {repo}: {done.stderr.strip()}")
    return done


def _rev_parse(repo: str, spec: str) -> str:
    done = _git(repo, "rev-parse", "--verify", "--quiet", spec, check=False)
    if done.returncode != 0:
        raise SystemExit(f"git rev-parse {spec} failed in {repo}: {done.stderr.strip() or 'unknown revision'}")
    return done.stdout.strip()


def resolve_pins(repo: str | None, commit: str | None, tree: str | None) -> tuple[str, str]:
    """``(commit, tree)``: HEAD and its tree by default, anything given verified."""
    if repo is None:
        if not commit or not tree:
            raise SystemExit("without a local clone, give both --commit and --tree")
        check_shas(commit, tree, None)
        return commit, tree
    if _git(repo, "rev-parse", "--git-dir", check=False).returncode != 0:
        raise SystemExit(
            f"--repo {repo} is not a Git clone; run this from your clone or pass --repo"
        )
    if commit is not None and not _SHA.fullmatch(commit):
        raise SystemExit(f"--commit must be a full 40-character lowercase SHA, got {commit!r}")
    resolved = _rev_parse(repo, f"{commit or 'HEAD'}^{{commit}}")
    actual_tree = _rev_parse(repo, f"{resolved}^{{tree}}")
    if tree is None:
        return resolved, actual_tree
    check_shas(resolved, tree, repo)
    return resolved, actual_tree


class LocalCheck(NamedTuple):
    status: str  # ok | warn | fail | skip
    name: str
    message: str


def _normalise_remote(url: str) -> str:
    url = url.strip().rstrip("/")
    return url[:-4] if url.endswith(".git") else url


def commit_on_remote(repo: str, commit: str, repo_url: str) -> LocalCheck:
    """Is ``commit`` reachable from a branch or tag of ``repo_url``?"""
    name = "commit pushed"
    try:
        listed = _git(repo, "ls-remote", repo_url, check=False)
    except subprocess.TimeoutExpired:
        listed = None
    if listed is None or listed.returncode != 0:
        return LocalCheck(
            "warn", name,
            f"could not list {repo_url} (private repository or no network); make sure "
            f"{commit[:12]} is pushed there",
        )
    tips: dict[str, str] = {}
    for line in listed.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if _SHA.fullmatch(sha) and ref:
            tips.setdefault(sha, ref.removesuffix("^{}"))
    if not tips:
        return LocalCheck("fail", name, f"{repo_url} has no branches or tags; push {commit[:12]}")
    if commit in tips:
        return LocalCheck("ok", name, f"{commit[:12]} is the tip of {tips[commit]}")
    unknown = 0
    for sha, ref in tips.items():
        if _git(repo, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode != 0:
            unknown += 1
            continue
        if _git(repo, "merge-base", "--is-ancestor", commit, sha, check=False).returncode == 0:
            return LocalCheck("ok", name, f"{commit[:12]} is contained in {ref}")
    if unknown:
        return LocalCheck(
            "warn", name,
            f"could not prove {commit[:12]} is on {repo_url}: fetch from it (git fetch) "
            "and run again, or push the commit",
        )
    return LocalCheck("fail", name, f"{commit[:12]} is not on {repo_url}: push it first")


def tree_checks(
    repo: str, commit: str, *, licenses_required: bool, max_bytes: int
) -> list[LocalCheck]:
    """Root files, symlinks/submodules and size, read from the commit itself."""
    raw = _git(repo, "ls-tree", "-r", "-l", "-z", commit).stdout
    entries: list[tuple[str, str, str, str]] = []  # mode, type, size, path
    for record in raw.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        parts = meta.split()
        if len(parts) == 4:
            entries.append((parts[0], parts[1], parts[3], path))
    root = {path: (mode, kind) for mode, kind, _size, path in entries if "/" not in path}
    checks: list[LocalCheck] = []

    def regular(path: str) -> bool:
        mode, kind = root.get(path, ("", ""))
        return kind == "blob" and mode in ("100644", "100755")

    checks.append(
        LocalCheck("ok", "Dockerfile", "Dockerfile at the repository root")
        if regular("Dockerfile")
        else LocalCheck("fail", "Dockerfile", "no regular file named Dockerfile at the repository root")
    )
    if licenses_required:
        found = [name for name in LICENSE_FILES if regular(name)]
        checks.append(
            LocalCheck("ok", "LICENSE", f"{found[0]} at the repository root")
            if found
            else LocalCheck(
                "fail", "LICENSE",
                "no LICENSE (or LICENSE.md, LICENSE.txt, COPYING) at the repository root",
            )
        )
        checks.append(
            LocalCheck("ok", "LICENSES.md", "LICENSES.md at the repository root")
            if regular(LICENSES_MANIFEST)
            else LocalCheck("fail", "LICENSES.md", "no LICENSES.md at the repository root")
        )
    else:
        checks.append(LocalCheck("skip", "LICENSE", "not required by this competition"))
    links = [path for mode, _k, _s, path in entries if mode == "120000"]
    gitlinks = [path for mode, _k, _s, path in entries if mode == "160000"]
    bad = []
    if links:
        bad.append(f"symlinks: {', '.join(links[:5])}{' ...' if len(links) > 5 else ''}")
    if gitlinks or ".gitmodules" in root:
        bad.append(f"submodules: {', '.join(gitlinks[:5]) or '.gitmodules'}")
    checks.append(
        LocalCheck("fail", "regular files only", "; ".join(bad) + " (not accepted)")
        if bad
        else LocalCheck("ok", "regular files only", "no symlinks or submodules")
    )
    size = sum(int(s) for _m, kind, s, _p in entries if kind == "blob" and s.isdigit())
    checks.append(
        LocalCheck("ok", "size", f"source tree {size} bytes (cap {max_bytes})")
        if size <= max_bytes
        else LocalCheck(
            "fail", "size",
            f"source tree {size} bytes, over the cap of {max_bytes}: download large "
            "weights in the Dockerfile instead",
        )
    )
    return checks


def _print_checks(checks: list[LocalCheck]) -> None:
    labels = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL", "skip": "skip"}
    for check in checks:
        print(f"  [{labels.get(check.status, check.status)}] {check.name}: {check.message}")


# ---- enroll / withdraw / status ----------------------------------------------------


def cmd_enroll(args: argparse.Namespace) -> int:
    repo = None if args.skip_local_checks and args.commit and args.tree else args.repo
    commit, tree = resolve_pins(repo, args.commit, args.tree)
    status, comp = _competition(args)
    entry_rules: dict = {}
    if status == 200 and isinstance(comp, dict):
        entry_rules = comp.get("entry_checks") or {}
        if not comp.get("enrollment_open"):
            print(f"note: enrollment for {args.competition_id} is not open "
                  f"(status {comp.get('status')}, deadline {comp.get('enrollment_deadline')})")
    elif status == 404:
        raise SystemExit(f"unknown competition {args.competition_id!r} (see `list`)")
    else:
        print("note: could not read the competition; checking against the defaults")
    licenses_required = bool(entry_rules.get("licenses_required", True))
    max_bytes = int(entry_rules.get("max_worktree_bytes", DEFAULT_MAX_WORKTREE_BYTES))

    print(f"entry: {args.repo_url} commit {commit} tree {tree}")
    if args.skip_local_checks:
        print("local checks skipped (--skip-local-checks)")
    else:
        checks = [LocalCheck("ok", "tree", f"{tree[:12]} is the tree of commit {commit[:12]}")]
        checks.append(commit_on_remote(args.repo, commit, args.repo_url))
        checks.extend(
            tree_checks(args.repo, commit, licenses_required=licenses_required, max_bytes=max_bytes)
        )
        print("local checks:")
        _print_checks(checks)
        failed = [check for check in checks if check.status == "fail"]
        if failed:
            print(f"{len(failed)} check(s) failed: nothing was signed or sent. Fix them, or "
                  "pass --skip-local-checks if you are sure.")
            return 1

    body = json.dumps(
        {"repo_url": args.repo_url, "commit_sha": commit, "tree_sha": tree,
         "replace": bool(args.replace)},
        separators=(",", ":"),
    ).encode()
    signer, status, response = _signed(args, "POST", "enroll", body)
    verb = "replacing the entry of" if args.replace else "enrolling"
    print(f"{verb} hotkey {signer.hotkey} in {args.competition_id}")
    print(json.dumps(response, indent=1))
    if status in (200, 201):
        print(
            f"entry commitment {entry_commitment(args.competition_id, signer.hotkey, args.repo_url, commit, tree)}"
            "\nthe validator checks the entry before the deadline; see `status`."
        )
        return 0
    detail = response.get("detail") if isinstance(response, dict) else None
    if isinstance(detail, dict) and detail.get("code") == "already_enrolled":
        print("this hotkey already has an entry: add --replace to change it before the deadline")
    return 1


def cmd_withdraw(args: argparse.Namespace) -> int:
    if not args.yes:
        raise SystemExit(
            "withdrawing is final: this hotkey cannot enroll again in this competition. "
            "Pass --yes to confirm."
        )
    body = json.dumps({"confirm": True}, separators=(",", ":")).encode()
    signer, status, response = _signed(args, "POST", "withdraw", body)
    print(f"withdrawing hotkey {signer.hotkey} from {args.competition_id}")
    print(json.dumps(response, indent=1))
    return 0 if status == 200 else 1


def cmd_status(args: argparse.Namespace) -> int:
    if args.hotkey:
        status, comp = _competition(args)
        if status != 200:
            print(json.dumps(comp, indent=1))
            return 1
        mine = [e for e in comp.get("enrolled", []) if e.get("hotkey") == args.hotkey]
        if not mine:
            print(f"{args.hotkey} is not enrolled in {args.competition_id}")
            return 1
        print(json.dumps({
            "competition_id": args.competition_id,
            "status": comp.get("status"),
            "enrollment_open": comp.get("enrollment_open"),
            "paused": comp.get("paused"),
            "entry": mine[0],
        }, indent=1))
        return 0
    signer, status, response = _signed(args, "GET", "entry")
    print(f"entry of hotkey {signer.hotkey} in {args.competition_id}")
    print(json.dumps(response, indent=1))
    return 0 if status == 200 else 1


def _wallet_args(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--wallet-name", required=required)
    parser.add_argument("--wallet-hotkey", required=required)
    parser.add_argument("--wallet-path")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="show competitions, floors, resources and entry checks")
    listing.add_argument("--url", required=True)
    listing.set_defaults(func=cmd_list)

    enroll = sub.add_parser("enroll", help="enroll (or --replace) your hotkey's pinned entry")
    enroll.add_argument("--url", required=True)
    enroll.add_argument("--competition-id", required=True)
    enroll.add_argument("--repo-url", required=True, help="https URL the validator fetches")
    enroll.add_argument("--repo", default=".", help="your local clone (default: .)")
    enroll.add_argument("--commit", help="40-hex commit (default: HEAD of --repo)")
    enroll.add_argument("--tree", help="40-hex tree of the commit (default: its tree)")
    enroll.add_argument("--replace", action="store_true",
                        help="change this hotkey's entry in place (before the deadline)")
    enroll.add_argument("--skip-local-checks", action="store_true",
                        help="send even when a local check fails")
    _wallet_args(enroll, required=True)
    enroll.set_defaults(func=cmd_enroll)

    withdraw = sub.add_parser("withdraw", help="take your entry out (final) before the deadline")
    withdraw.add_argument("--url", required=True)
    withdraw.add_argument("--competition-id", required=True)
    withdraw.add_argument("--yes", action="store_true", help="confirm: withdrawing is final")
    _wallet_args(withdraw, required=True)
    withdraw.set_defaults(func=cmd_withdraw)

    status = sub.add_parser("status", help="your entry (signed), or any hotkey's public summary")
    status.add_argument("--url", required=True)
    status.add_argument("--competition-id", required=True)
    status.add_argument("--hotkey", help="public summary for this ss58 hotkey (no wallet)")
    _wallet_args(status, required=False)
    status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
