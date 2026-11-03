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
