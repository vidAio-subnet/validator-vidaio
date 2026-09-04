"""Fail-closed production GitRepoProvider tests with a local fake Git binary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from vidaio.competition.orchestrator.failures import Fault, classify_failure
from vidaio.competition.interfaces import ContenderSpec
from vidaio.competition.runners.docker_runner import DockerSandboxRunner
from vidaio.competition.runners.errors import CheckoutError, CheckoutRejectedError
from vidaio.competition.runners.repo import (
    GitRepoProvider,
    LocalRepoProvider,
    checkout_pinned,
)

COMMIT = "a" * 40
TREE = "b" * 40


def _fake_git(tmp_path: Path, *, mode: str = "ok") -> tuple[Path, Path]:
    executable = tmp_path / f"fake-git-{mode}"
    audit = tmp_path / f"fake-git-{mode}.jsonl"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys
import time

args = sys.argv[1:]
if {mode!r} == "slow_each":
    time.sleep(0.02)
audit = pathlib.Path({str(audit)!r})
with audit.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({{
        "argv": args,
        "token_present": bool(os.environ.get("VIDAIO_NEXT_GIT_TOKEN")),
        "askpass_present": bool(os.environ.get("GIT_ASKPASS")),
        "secret_canary_present": "VIDAIO_SECRET_CANARY" in os.environ,
    }}) + "\\n")

repo = None
if args and args[0] == "init":
    repo = pathlib.Path(args[-1])
elif "-C" in args:
    repo = pathlib.Path(args[args.index("-C") + 1])

if args and args[0] == "init":
    (repo / ".git").mkdir(parents=True, exist_ok=True)
elif "fetch" in args:
    if {mode!r} == "timeout":
        time.sleep(2)
    (repo / ".git" / "fetched").write_bytes(b"objects")
elif "rev-parse" in args:
    target = args[-1]
    if "tree" in target:
        print({TREE!r})
    else:
        print({COMMIT!r})
elif "checkout" in args:
    (repo / "Dockerfile").write_text("FROM scratch\\n", encoding="utf-8")
    (repo / "app").mkdir(exist_ok=True)
    (repo / "app" / "run.sh").write_text("#!/bin/sh\\n", encoding="utf-8")
    if {mode!r} == "symlink":
        os.symlink("/etc/passwd", repo / "leak")
    elif {mode!r} == "oversize":
        (repo / "large.bin").write_bytes(b"x" * 4096)
    elif {mode!r} == "submodule":
        (repo / ".gitmodules").write_text("[submodule \'x\']\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, audit


def _provider(tmp_path: Path, fake_git: Path, **kwargs: object) -> GitRepoProvider:
    return GitRepoProvider(
        scratch_root=tmp_path / "checkouts",
        read_only_token="read-only-token-value",
        allowed_hosts=("git.example.test",),
        git_path=str(fake_git),
        poll_seconds=0.005,
        **kwargs,
    )


def test_fresh_checkout_verifies_both_pins_strips_git_and_never_reuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake, audit = _fake_git(tmp_path)
    monkeypatch.setenv("VIDAIO_SECRET_CANARY", "must-not-reach-git")
    provider = _provider(tmp_path, fake)
    first = provider.checkout_pinned(
        "https://git.example.test/org/solution.git", COMMIT, TREE
    )
    second = provider.checkout_pinned(
        "https://git.example.test/org/solution.git", COMMIT, TREE
    )
    assert first != second
    assert first.name == second.name == "repo"
    assert (first / "Dockerfile").read_text() == "FROM scratch\n"
    assert not (first / ".git").exists()
    assert not any("token" in path.name.lower() for path in first.rglob("*"))

    calls = [json.loads(line) for line in audit.read_text().splitlines()]
    assert calls
    assert all(call["token_present"] and call["askpass_present"] for call in calls)
    assert all(not call["secret_canary_present"] for call in calls)
    serialized_argv = json.dumps([call["argv"] for call in calls])
    assert "read-only-token-value" not in serialized_argv
    assert "https://git.example.test/org/solution.git" in serialized_argv

    first_session = first.parent
    second_session = second.parent
    provider.release(first)
    assert not first_session.exists()
    assert second.exists()
    provider.close()
    assert not second_session.exists()


def test_tree_mismatch_is_contender_rejection_and_failed_session_is_removed(
    tmp_path: Path,
):
    fake, _audit = _fake_git(tmp_path)
    provider = _provider(tmp_path, fake)
    with pytest.raises(
        CheckoutRejectedError, match="does not match enrolled"
    ) as caught:
        provider.checkout_pinned(
            "https://git.example.test/org/solution.git", COMMIT, "c" * 40
        )
    assert classify_failure(caught.value) is Fault.CONTENDER
    assert list((tmp_path / "checkouts").iterdir()) == []


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("symlink", "unsafe checkout tree"),
        ("oversize", "checkout cap"),
        ("submodule", "unpinned submodules"),
    ],
)
def test_unsafe_oversize_and_submodule_trees_fail_closed(
    tmp_path: Path, mode: str, match: str
):
    fake, _audit = _fake_git(tmp_path, mode=mode)
    provider = _provider(tmp_path, fake, max_checkout_bytes=1024)
    with pytest.raises(CheckoutRejectedError, match=match):
        provider.checkout_pinned(
            "https://git.example.test/org/solution.git", COMMIT, TREE
        )


def test_timeout_is_bounded_infrastructure_failure_and_kills_child(tmp_path: Path):
    fake, _audit = _fake_git(tmp_path, mode="timeout")
    provider = _provider(tmp_path, fake, timeout_seconds=0.05)
    with pytest.raises(CheckoutError, match="bounded") as caught:
        provider.checkout_pinned(
            "https://git.example.test/org/solution.git", COMMIT, TREE
        )
    assert not isinstance(caught.value, CheckoutRejectedError)
    assert classify_failure(caught.value) is Fault.INFRA


def test_timeout_bounds_the_whole_checkout_not_each_git_command(tmp_path: Path):
    fake, _audit = _fake_git(tmp_path, mode="slow_each")
    provider = _provider(tmp_path, fake, timeout_seconds=0.07)
    with pytest.raises(CheckoutError, match="total Git timeout"):
        provider.checkout_pinned(
            "https://git.example.test/org/solution.git", COMMIT, TREE
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://git.example.test/org/repo.git",
        "https://user:password@git.example.test/org/repo.git",
        "https://other.example.test/org/repo.git",
        "https://git.example.test:444/org/repo.git",
        "https://git.example.test/org/repo.git?token=oops",
        "file:///tmp/repo",
    ],
)
def test_url_boundary_rejects_credentials_non_https_ports_and_unknown_hosts(
    tmp_path: Path, url: str
):
    fake, audit = _fake_git(tmp_path)
    provider = _provider(tmp_path, fake)
    with pytest.raises(CheckoutRejectedError, match="credential-free HTTPS"):
        provider.checkout_pinned(url, COMMIT, TREE)
    assert not audit.exists()  # rejected before any subprocess


def test_malformed_url_port_is_a_contender_rejection_before_subprocess(tmp_path: Path):
    fake, audit = _fake_git(tmp_path)
    provider = _provider(tmp_path, fake)
    with pytest.raises(CheckoutRejectedError, match="valid HTTPS port"):
        provider.checkout_pinned(
            "https://git.example.test:not-a-port/org/repo.git", COMMIT, TREE
        )
    assert not audit.exists()


def test_git_provider_refuses_commit_only_interface(tmp_path: Path):
    fake, _audit = _fake_git(tmp_path)
    provider = _provider(tmp_path, fake)
    with pytest.raises(CheckoutError, match="requires the enrolled tree SHA"):
        provider.checkout("https://git.example.test/org/repo.git", COMMIT)


def test_checkout_pinned_preserves_non_git_local_fixture_provider(tmp_path: Path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    local = LocalRepoProvider({"local://fixture": fixture})
    assert checkout_pinned(local, "local://fixture", COMMIT, TREE) == fixture


def test_source_never_uses_shell_or_embeds_token_in_git_argv():
    source = Path("vidaio/competition/runners/repo.py").read_text()
    assert "shell=True" not in source
    assert "https://{self._" not in source
    assert "VIDAIO_NEXT_GIT_TOKEN" in source
    assert "http.followRedirects=false" in source
    assert "GIT_CONFIG_NOSYSTEM" in source


def test_docker_build_releases_checkout_after_build_context_is_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "Dockerfile").write_text("FROM scratch\n")
    events: list[str] = []

    class Provider:
        def checkout(self, repo_url: str, commit_sha: str) -> Path:
            events.append("checkout")
            return checkout

        def release(self, path: str | Path) -> None:
            assert Path(path) == checkout
            events.append("release")

    def fake_run_docker(self, args, **_kwargs):
        if args[0] == "version":
            return "27.0"
        if args[0] == "ps":
            return ""
        if args[0] == "build":
            assert Path(args[-1]) == checkout
            assert (checkout / "Dockerfile").is_file()
            events.append("docker-build-complete")
            return ""
        if args[:2] == ["image", "inspect"]:
            events.append("docker-inspect")
            return "sha256:fresh-image-id"
        if args[0] == "tag":
            events.append("docker-tag")
            return ""
        pytest.fail(f"unexpected docker call: {args}")

    monkeypatch.setattr(DockerSandboxRunner, "_run_docker", fake_run_docker)
    runner = DockerSandboxRunner(
        Provider(),
        inputs_dir=tmp_path / "inputs",
        outputs_dir=tmp_path / "outputs",
        scratch_dir=tmp_path / "scratch",
    )
    runner.build(
        ContenderSpec(
            contender_id=7,
            repo_url="https://example.invalid/solution.git",
            commit_sha=COMMIT,
            tree_sha=TREE,
        )
    )
    assert events == [
        "checkout",
        "docker-build-complete",
        "docker-inspect",
        "docker-tag",
        "release",
    ]
