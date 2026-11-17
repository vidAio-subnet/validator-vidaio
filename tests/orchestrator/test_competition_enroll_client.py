"""The enroll client refuses commit/tree mistakes before anything is signed."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "competition_enroll", ROOT / "scripts" / "competition_enroll.py"
)
enroll = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(enroll)

COMMIT = "efb37f1e2adcfd4797fdd43aacab72156eb8aa8e"
TREE = "5e1e3477837805f2350891bdb88f6b7c71659d58"


def test_commit_given_as_tree_is_refused() -> None:
    with pytest.raises(SystemExit, match=r"\^\{tree\}"):
        enroll.check_shas(COMMIT, COMMIT, None)


@pytest.mark.parametrize(
    ("commit", "tree"), [(COMMIT[:12], TREE), (COMMIT, TREE.upper()), (COMMIT, "HEAD")]
)
def test_malformed_shas_are_refused(commit: str, tree: str) -> None:
    with pytest.raises(SystemExit, match="40-character"):
        enroll.check_shas(commit, tree, None)


def test_distinct_full_shas_pass_without_a_repo() -> None:
    enroll.check_shas(COMMIT, TREE, None)


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(tmp_path), *a], check=True,
                              capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    (tmp_path / "run.sh").write_text("#!/bin/sh\n")
    git("add", "run.sh")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "entry")
    return tmp_path, git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")


def test_repo_check_accepts_the_commits_own_tree(tmp_path: Path) -> None:
    repo, commit, tree = _repo(tmp_path)
    enroll.check_shas(commit, tree, str(repo))


def test_repo_check_refuses_another_tree(tmp_path: Path) -> None:
    repo, commit, _tree = _repo(tmp_path)
    with pytest.raises(SystemExit, match="is not the tree of commit"):
        enroll.check_shas(commit, TREE, str(repo))


# ---- pins from the clone, local checks, and the signed commands ---------------------

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def _git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *a], check=True,
                          capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, message: str = "entry") -> str:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _entry_repo(tmp_path: Path, *, licenses: bool = True) -> Path:
    repo = tmp_path / "clone"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "Dockerfile").write_text("FROM alpine:3.20\nCOPY run.sh /app/run.sh\n")
    (repo / "run.sh").write_text("#!/bin/sh\n")
    if licenses:
        (repo / "LICENSE").write_text("MIT\n")
        (repo / "LICENSES.md").write_text("none\n")
    _commit(repo)
    return repo


def test_pins_default_to_head_and_its_tree(tmp_path: Path) -> None:
    repo = _entry_repo(tmp_path)
    head, tree = _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD^{tree}")
    assert enroll.resolve_pins(str(repo), None, None) == (head, tree)
    assert enroll.resolve_pins(str(repo), head, None) == (head, tree)
    assert enroll.resolve_pins(str(repo), head, tree) == (head, tree)
    with pytest.raises(SystemExit, match="is not the tree of commit"):
        enroll.resolve_pins(str(repo), head, TREE)
    with pytest.raises(SystemExit, match="40-character"):
        enroll.resolve_pins(str(repo), head[:12], None)
    with pytest.raises(SystemExit, match=r"\^\{tree\}"):
        enroll.resolve_pins(str(repo), head, head)
    with pytest.raises(SystemExit, match="not a Git clone"):
        enroll.resolve_pins(str(tmp_path / "nowhere-dir-missing"), None, None)
    assert enroll.resolve_pins(None, COMMIT, TREE) == (COMMIT, TREE)
    with pytest.raises(SystemExit, match="both --commit and --tree"):
        enroll.resolve_pins(None, COMMIT, None)


def test_tree_checks_read_the_commit_itself(tmp_path: Path) -> None:
    repo = _entry_repo(tmp_path, licenses=False)
    head = _git(repo, "rev-parse", "HEAD")
    checks = {c.name: c for c in enroll.tree_checks(
        str(repo), head, licenses_required=False, max_bytes=1 << 20)}
    assert {n: c.status for n, c in checks.items()} == {
        "Dockerfile": "ok", "LICENSE": "skip", "regular files only": "ok", "size": "ok",
    }
    checks = {c.name: c.status for c in enroll.tree_checks(
        str(repo), head, licenses_required=True, max_bytes=10)}
    assert checks["LICENSE"] == checks["LICENSES.md"] == checks["size"] == "fail"

    (repo / "COPYING").write_text("Apache-2.0\n")
    (repo / "LICENSES.md").write_text("none\n")
    (repo / "leak").symlink_to("/etc/passwd")
    (repo / "Dockerfile").unlink()
    _git(repo, "add", "-A")
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{head},vendored")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", "broken")
    head = _git(repo, "rev-parse", "HEAD")
    checks = {c.name: c for c in enroll.tree_checks(
        str(repo), head, licenses_required=True, max_bytes=1 << 20)}
    assert checks["LICENSE"].status == "ok" and "COPYING" in checks["LICENSE"].message
    assert checks["Dockerfile"].status == "fail"
    assert checks["regular files only"].status == "fail"
    assert "leak" in checks["regular files only"].message
    assert "vendored" in checks["regular files only"].message


def test_pushed_check_uses_the_remote(tmp_path: Path) -> None:
    repo = _entry_repo(tmp_path)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    _git(repo, "push", "-q", str(remote), "HEAD:refs/heads/main")
    first = _git(repo, "rev-parse", "HEAD")
    assert enroll.commit_on_remote(str(repo), first, str(remote)).status == "ok"
    (repo / "run.sh").write_text("#!/bin/sh\necho v2\n")
    second = _commit(repo, "v2")
    check = enroll.commit_on_remote(str(repo), second, str(remote))
    assert check.status == "fail" and "push it first" in check.message
    _git(repo, "push", "-q", str(remote), "HEAD:refs/heads/dev")
    (repo / "run.sh").write_text("#!/bin/sh\necho v3\n")
    _commit(repo, "v3")
    _git(repo, "push", "-q", str(remote), "HEAD:refs/heads/dev")
    contained = enroll.commit_on_remote(str(repo), second, str(remote))
    assert contained.status == "ok" and "contained in refs/heads/dev" in contained.message
    unreachable = enroll.commit_on_remote(str(repo), second, str(tmp_path / "missing.git"))
    assert unreachable.status == "warn"


class _Signer:
    hotkey = "5" + "S" * 47

    def sign(self, payload: bytes) -> str:
        return "ab" * 64


def _args(repo: Path, **overrides) -> SimpleNamespace:
    base = dict(
        url="https://competitions.example", competition_id="comp-x",
        repo_url="https://github.com/you/solution.git", repo=str(repo), commit=None,
        tree=None, replace=False, skip_local_checks=False, wallet_name="w",
        wallet_hotkey="h", wallet_path=None, yes=False, hotkey=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _wire(monkeypatch, *, competition: dict | None = None, reply=(201, {"contender_id": 7})):
    sent: list[tuple[str, str, bytes | None, dict]] = []
    monkeypatch.setattr(enroll, "_load_signer", lambda args: _Signer())

    def fake_request(method, url, *, body=None, headers=None):
        sent.append((method, url, body, dict(headers or {})))
        if method == "GET" and url.endswith("/v1/competitions/comp-x"):
            return 200, competition or {
                "status": "ENROLLING", "enrollment_open": True,
                "entry_checks": {"licenses_required": True, "max_worktree_bytes": 1 << 20},
                "enrolled": [],
            }
        return reply

    monkeypatch.setattr(enroll, "_request", fake_request)
    monkeypatch.setattr(
        enroll, "commit_on_remote",
        lambda repo, commit, url: enroll.LocalCheck("ok", "commit pushed", "stubbed"),
    )
    return sent


def test_enroll_signs_head_and_sends_replace(tmp_path: Path, monkeypatch) -> None:
    repo = _entry_repo(tmp_path)
    sent = _wire(monkeypatch, reply=(200, {"contender_id": 7, "replaced": True}))
    assert enroll.cmd_enroll(_args(repo, replace=True)) == 0
    method, url, body, headers = sent[-1]
    assert (method, url) == ("POST", "https://competitions.example/v1/competitions/comp-x/enroll")
    assert json.loads(body) == {
        "repo_url": "https://github.com/you/solution.git",
        "commit_sha": _git(repo, "rev-parse", "HEAD"),
        "tree_sha": _git(repo, "rev-parse", "HEAD^{tree}"),
        "replace": True,
    }
    assert headers["x-vidaio-hotkey"] == _Signer.hotkey
    assert headers["Content-Type"] == "application/json"


def test_a_failed_local_check_stops_before_signing(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _entry_repo(tmp_path, licenses=False)
    sent = _wire(monkeypatch)
    monkeypatch.setattr(enroll, "_load_signer", lambda args: pytest.fail("signed"))
    assert enroll.cmd_enroll(_args(repo)) == 1
    assert [m for m, *_ in sent] == ["GET"]
    out = capsys.readouterr().out
    assert "[FAIL] LICENSE" in out and "nothing was signed" in out

    sent = _wire(monkeypatch)
    assert enroll.cmd_enroll(_args(repo, skip_local_checks=True)) == 0
    assert sent[-1][0] == "POST"


def test_withdraw_needs_yes_and_status_reads_both_views(tmp_path: Path, monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, reply=(200, {"withdrawn": True}))
    with pytest.raises(SystemExit, match="final"):
        enroll.cmd_withdraw(_args(tmp_path))
    assert sent == []
    assert enroll.cmd_withdraw(_args(tmp_path, yes=True)) == 0
    method, url, body, _headers = sent[-1]
    assert url.endswith("/v1/competitions/comp-x/withdraw") and json.loads(body) == {"confirm": True}

    sent = _wire(monkeypatch, reply=(200, {"status": "ENROLLED"}))
    assert enroll.cmd_status(_args(tmp_path)) == 0
    method, url, body, headers = sent[-1]
    assert (method, url, body) == (
        "GET", "https://competitions.example/v1/competitions/comp-x/entry", None
    )
    assert "Content-Type" not in headers

    listing = {"status": "ENROLLING", "enrolled": [
        {"hotkey": "5Pub", "status": "ENROLLED", "checks": {"fetch": "ok", "build": "disabled"}}
    ]}
    _wire(monkeypatch, competition=listing)
    monkeypatch.setattr(enroll, "_load_signer", lambda args: pytest.fail("no wallet needed"))
    assert enroll.cmd_status(_args(tmp_path, hotkey="5Pub", wallet_name=None)) == 0
    assert '"fetch": "ok"' in capsys.readouterr().out
    assert enroll.cmd_status(_args(tmp_path, hotkey="5Other", wallet_name=None)) == 1


def test_cli_wires_the_new_commands() -> None:
    with pytest.raises(SystemExit):
        enroll.main(["withdraw", "--url", "u", "--competition-id", "c"])  # wallet required
    parser_help = enroll.__doc__ or ""
    assert "--replace" in parser_help and "withdraw" in parser_help and "status" in parser_help
