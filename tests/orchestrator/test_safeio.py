"""Symlink-safety of contender-controlled trees and outputs.

Nothing here needs docker: the risk is a filesystem one, and the defence is
a filesystem defence. The docker half (a container that EMITS a symlink) lives in
test_sandbox_safety.py.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from vidaio.competition import repository as repo
from vidaio.competition.runners import safeio
from vidaio.competition.runners.errors import OversizeOutputError, UnsafePathError
from vidaio.competition.states import Phase

from orchestrator_support import (
    FINALIZATION,
    M,
    build_manifest,
    enroll,
    phase,
    repo_url,
    start_and_enroll,
)

HOST_SECRET = "/etc/passwd"


def _tree(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "Dockerfile").write_text("FROM alpine:3.20\n")
    (root / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    sub = root / "src"
    sub.mkdir(exist_ok=True)
    (sub / "main.py").write_text("print('hi')\n")
    return root


# ---- tree scanning --------------------------------------------------------------


def test_safe_tree_accepts_regular_files_only(tmp_path):
    root = _tree(tmp_path / "clean")
    assert safeio.assert_safe_tree(root) == 3
    names = [rel for rel, _p, _st in safeio.iter_tree_regular_files(root)]
    assert names == ["Dockerfile", "run.sh", "src/main.py"]


def test_symlink_in_tree_is_rejected_not_followed(tmp_path):
    root = _tree(tmp_path / "untrusted")
    (root / "stolen").symlink_to(HOST_SECRET)
    with pytest.raises(UnsafePathError) as excinfo:
        safeio.assert_safe_tree(root)
    assert "stolen" in str(excinfo.value)


def test_symlinked_directory_in_tree_is_rejected(tmp_path):
    root = _tree(tmp_path / "untrusted-dir")
    (root / "escape").symlink_to("/etc", target_is_directory=True)
    with pytest.raises(UnsafePathError):
        safeio.assert_safe_tree(root)


def test_fifo_in_tree_is_rejected(tmp_path):
    root = _tree(tmp_path / "fifo")
    os.mkfifo(root / "pipe")
    with pytest.raises(UnsafePathError):
        safeio.assert_safe_tree(root)


# ---- tarball --------------------------------------------------------------------


def test_tarball_is_deterministic_and_holds_only_regular_files(tmp_path):
    root = _tree(tmp_path / "clean")
    first = safeio.deterministic_tarball(root, max_bytes=1 << 20)
    second = safeio.deterministic_tarball(root, max_bytes=1 << 20)
    assert first == second
    with tarfile.open(fileobj=BytesIO(first)) as tar:
        members = tar.getmembers()
    assert [m.name for m in members] == ["Dockerfile", "run.sh", "src/main.py"]
    assert all(m.isreg() for m in members)
    assert not any(m.issym() or m.islnk() or m.isdev() for m in members)


def test_tarball_never_archives_a_symlink_target(tmp_path):
    """The whole point of #3: /etc/passwd must not end up in the audit store."""
    root = _tree(tmp_path / "untrusted")
    (root / "stolen").symlink_to(HOST_SECRET)
    with pytest.raises(UnsafePathError):
        safeio.deterministic_tarball(root, max_bytes=1 << 20)


def test_tarball_rejects_an_oversize_tree(tmp_path):
    root = _tree(tmp_path / "big")
    (root / "blob.bin").write_bytes(b"x" * 4096)
    with pytest.raises(OversizeOutputError):
        safeio.deterministic_tarball(root, max_bytes=1024)


# ---- pooling --------------------------------------------------------------------


def test_hash_into_pool_stores_exactly_the_bytes_it_hashed(tmp_path):
    src = tmp_path / "out"
    src.mkdir()
    payload = b"video-bytes" * 100
    produced = src / "artifact"
    produced.write_bytes(payload)
    pool = tmp_path / "pool"
    st = safeio.lstat_regular(produced, what="sandbox output")
    digest, size = safeio.hash_into_pool(
        produced, st, pool, max_bytes=1 << 20, what="sandbox output"
    )
    assert digest == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)
    assert (pool / digest).read_bytes() == payload


def test_pooling_a_symlink_is_refused_before_any_read(tmp_path):
    src = tmp_path / "out"
    src.mkdir()
    link = src / "artifact"
    link.symlink_to(HOST_SECRET)
    with pytest.raises(UnsafePathError):
        safeio.lstat_regular(link, what="sandbox output")
    pool = tmp_path / "pool"
    pool.mkdir()
    assert list(pool.iterdir()) == []


def test_open_regular_nofollow_detects_a_swap(tmp_path):
    target = tmp_path / "artifact"
    target.write_bytes(b"original")
    st = safeio.lstat_regular(target, what="sandbox output")
    target.unlink()
    target.write_bytes(b"swapped")  # different inode
    with pytest.raises(UnsafePathError):
        safeio.open_regular_nofollow(target, st, what="sandbox output").close()


def test_tree_bytes_counts_without_following_links(tmp_path):
    root = tmp_path / "out"
    (root / "nested").mkdir(parents=True)
    (root / "a").write_bytes(b"x" * 100)
    (root / "nested" / "b").write_bytes(b"y" * 50)
    (root / "link").symlink_to(HOST_SECRET)
    counted = safeio.tree_bytes(root)
    assert counted >= 150
    # /etc/passwd is far bigger than the symlink entry itself; following it would
    # blow past this bound.
    assert counted < 150 + 1024


# ---- orchestrator integration ---------------------------------------------------


async def test_symlinked_submission_is_rejected_and_never_archived(
    orchestrator_factory, fixture_repos, tmp_path
):
    """A repo shipping `stolen-credentials -> /etc/passwd`:

    - its backup tarball is skipped (nothing of the host is archived),
    - validation REJECTS it with a reason,
    - the honest contender is unaffected and the competition proceeds.
    """
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    enroll(orch, cid, "hk-link")

    await orch.step(FINALIZATION)  # backup + -> VALIDATING
    await orch.step(FINALIZATION + 2 * M)  # validation
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    assert by_hotkey["hk-a"].status == "ACCEPTED"
    assert by_hotkey["hk-link"].status == "REJECTED"

    events = [
        e for e in repo.list_events(orch.conn, cid) if e["event_type"] == "contender_validated"
    ]
    rejected = [e for e in events if "REJECTED" in (e["payload_json"] or "")]
    assert rejected and "unsafe submission tree" in rejected[0]["payload_json"]

    # Nothing resembling the host secret reached the audit store.
    secret_head = Path(HOST_SECRET).read_bytes()[:32]
    audit_root = Path(orch.raw_config["audit"]["local_root"])
    for blob in audit_root.rglob("*"):
        if blob.is_file():
            assert secret_head not in blob.read_bytes()

    assert phase(orch, cid) is Phase.BUILDING


def test_repo_symlink_does_not_break_the_honest_backup(
    orchestrator_factory, fixture_repos
):
    """The rejection is per-contender: the honest tree still produced a backup."""
    orchestrator_factory(repos=fixture_repos)
    checkout = fixture_repos[repo_url("hk-a")]
    tarball = safeio.deterministic_tarball(checkout, max_bytes=1 << 20)
    assert tarball  # builds fine, no exception
