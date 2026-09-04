"""Symlink-safe filesystem primitives for ADVERSARY-CONTROLLED trees.

review service-review #3 (critical): both the sandbox output directory and the
contender's submission checkout are bytes an adversary chose. Every stdlib path
helper that "just works" (``Path.is_file``, ``open``, ``shutil.copy2``,
``tarfile.add``, ``Path.rglob``) FOLLOWS symlinks, so a container that emits
``<expected-name> -> /proc/self/environ`` (or a repo that ships
``creds -> /etc/passwd``) makes the host read — and archive into the audit store —
its own secrets.

The rules enforced here, applied to every contender-controlled path:

1. ``os.lstat`` first, never ``stat``: the verdict is about the entry itself, not
   about whatever it points at. Anything that is not a regular file (symlink,
   fifo, socket, device, directory) is REJECTED with a typed error.
2. Open with ``O_NOFOLLOW`` so the kernel — not our earlier lstat — guarantees no
   symlink is traversed at open time, then ``fstat`` the descriptor and require
   the SAME (st_dev, st_ino) we inspected. That closes the swap-after-check race.
3. Verify the resolved path stays inside the expected root (``realpath`` prefix
   check) before anything is read or copied.
4. Hard links are covered by the same rule set at the only place they matter:
   nothing is ever archived by reference, only by bytes we read through a
   verified descriptor, and a link count is irrelevant once the bytes are ours.
   Tarballs are built from explicit ``TarInfo`` records over regular files only —
   ``tarfile`` is never handed a directory to walk, so it can never emit a
   symlink, hardlink or device member (``dereference`` is moot).

Trees containing an unsafe entry are REJECTED, not silently filtered: a backup
that quietly drops files is not evidence of the submitted tree. The caller turns
that rejection into a contender-level outcome (submission REJECTED / item
zero-scored), never into a competition failure.
"""

from __future__ import annotations

import hashlib
import io
import os
import stat
import tarfile
import uuid
from pathlib import Path
from typing import IO, Iterator

from vidaio.competition.runners.errors import (
    OversizeOutputError,
    UnsafePathError,
)

CHUNK = 1 << 20


def lstat_regular(path: Path, *, what: str) -> os.stat_result:
    """``os.lstat`` the entry and require a plain regular file.

    Raises FileNotFoundError when absent (the caller decides whether an absent
    entry is fatal) and UnsafePathError for symlinks/fifos/devices/dirs.
    """
    st = os.lstat(path)  # never follows
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise UnsafePathError(
            f"{what} {path.name!r} is {_describe(st.st_mode)}, not a regular file — "
            "symlinks, hardlink targets outside the root, fifos, sockets, devices "
            "and directories are never read or archived"
        )
    return st


def _describe(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "a symlink"
    if stat.S_ISDIR(mode):
        return "a directory"
    if stat.S_ISFIFO(mode):
        return "a fifo"
    if stat.S_ISSOCK(mode):
        return "a socket"
    if stat.S_ISBLK(mode) or stat.S_ISCHR(mode):
        return "a device node"
    return f"mode {stat.filemode(mode)}"


def assert_within(path: Path, root: Path, *, what: str) -> None:
    """Require ``path`` to resolve inside ``root`` (realpath prefix check)."""
    resolved = os.path.realpath(path)
    real_root = os.path.realpath(root)
    if resolved != real_root and not resolved.startswith(real_root + os.sep):
        raise UnsafePathError(
            f"{what} {path.name!r} resolves to {resolved!r}, outside its expected "
            f"root {real_root!r}"
        )


def open_regular_nofollow(path: Path, expected: os.stat_result, *, what: str) -> IO[bytes]:
    """Open a verified regular file without ever traversing a symlink.

    ``expected`` is the ``lstat_regular`` result: the descriptor must be the very
    same inode, otherwise the entry was swapped between check and open.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:  # ELOOP == someone made it a symlink after the lstat
        raise UnsafePathError(
            f"{what} {path.name!r} could not be opened without following a link: {exc}"
        ) from exc
    handle = os.fdopen(fd, "rb")
    try:
        got = os.fstat(handle.fileno())
        if not stat.S_ISREG(got.st_mode):
            raise UnsafePathError(f"{what} {path.name!r} is not a regular file at open time")
        if (got.st_dev, got.st_ino) != (expected.st_dev, expected.st_ino):
            raise UnsafePathError(
                f"{what} {path.name!r} was replaced between inspection and read "
                "(inode changed) — refusing to hash or archive it"
            )
    except BaseException:
        handle.close()
        raise
    return handle


def hash_into_pool(
    path: Path,
    expected: os.stat_result,
    pool_dir: Path,
    *,
    max_bytes: int,
    what: str,
) -> tuple[str, int]:
    """Stream a verified regular file into the content-addressed pool.

    ONE pass: the bytes that are hashed are exactly the bytes that are stored (no
    hash-then-copy TOCTOU, no ``copy2`` that would follow a link). Returns
    ``(sha256_hex, byte_size)``. Raises OversizeOutputError past ``max_bytes``.
    """
    pool_dir.mkdir(parents=True, exist_ok=True)
    tmp = pool_dir / f".incoming-{uuid.uuid4().hex}"
    digest = hashlib.sha256()
    size = 0
    try:
        with open_regular_nofollow(path, expected, what=what) as src:
            tmp_fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(tmp_fd, "wb") as dst:
                while chunk := src.read(CHUNK):
                    size += len(chunk)
                    if size > max_bytes:
                        raise OversizeOutputError(
                            f"{what} {path.name!r} exceeds the per-output cap of "
                            f"{max_bytes} bytes"
                        )
                    digest.update(chunk)
                    dst.write(chunk)
    except BaseException:
        with _suppress_os_error():
            tmp.unlink()
        raise
    hexdigest = digest.hexdigest()
    pooled = pool_dir / hexdigest
    if pooled.exists():
        with _suppress_os_error():
            tmp.unlink()
    else:
        os.replace(tmp, pooled)
    return hexdigest, size


class _suppress_os_error:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return isinstance(exc, OSError)


def tree_bytes(root: Path) -> int:
    """On-disk bytes under ``root``, counting entries WITHOUT following links.

    Used by the sandbox output watchdog: it must charge a contender for whatever
    it wrote, including symlinks and special files it created.
    """
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode):
                stack.append(Path(entry.path))
            total += st.st_size
    return total


def iter_tree_regular_files(root: Path) -> Iterator[tuple[str, Path, os.stat_result]]:
    """Yield ``(relative_posix_path, path, lstat)`` for every regular file under
    ``root``, in sorted order, REJECTING any tree that holds an unsafe entry.

    Directories are descended without following symlinked directories; a symlink
    anywhere in the tree raises UnsafePathError (the whole tree is rejected — see
    the module docstring on why filtering silently is not acceptable here).
    """
    real_root = os.path.realpath(root)
    if not os.path.isdir(real_root):
        raise UnsafePathError(f"tree root {root} is not a directory")
    for dirpath, dirnames, filenames in os.walk(real_root, followlinks=False):
        dirnames.sort()
        for name in sorted(dirnames):
            entry = Path(dirpath) / name
            st = os.lstat(entry)
            if stat.S_ISLNK(st.st_mode):
                raise UnsafePathError(
                    f"submission tree contains a symlinked directory "
                    f"{os.path.relpath(entry, real_root)!r} — rejected"
                )
        for name in sorted(filenames):
            entry = Path(dirpath) / name
            st = os.lstat(entry)
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                raise UnsafePathError(
                    f"submission tree entry {os.path.relpath(entry, real_root)!r} is "
                    f"{_describe(st.st_mode)}, not a regular file — rejected "
                    "(a tree that needs filtering is not the tree that was submitted)"
                )
            assert_within(entry, Path(real_root), what="submission tree entry")
            yield os.path.relpath(entry, real_root).replace(os.sep, "/"), entry, st


def assert_safe_tree(root: Path) -> int:
    """Validate a contender checkout; returns the number of regular files."""
    return sum(1 for _ in iter_tree_regular_files(root))


def deterministic_tarball(root: Path, *, max_bytes: int) -> bytes:
    """Deterministic tarball of a checkout: sorted paths, zeroed metadata, REGULAR
    FILES ONLY (the same tree always produces the same digest).

    Members are synthesized as explicit ``TarInfo`` records from bytes read
    through ``O_NOFOLLOW`` descriptors, so no symlink, hardlink or device member
    can exist in the archive and no host file outside ``root`` can be read. An
    unsafe tree raises UnsafePathError; an oversize tree raises
    OversizeOutputError — both are contender-level rejections.
    """
    buf = io.BytesIO()
    total = 0
    with tarfile.open(fileobj=buf, mode="w", dereference=False) as tar:
        for relative, path, st in iter_tree_regular_files(root):
            total += st.st_size
            if total > max_bytes:
                raise OversizeOutputError(
                    f"submission tree {root} exceeds the backup cap of {max_bytes} bytes"
                )
            with open_regular_nofollow(path, st, what="submission tree entry") as handle:
                data = handle.read(st.st_size + 1)
            if len(data) > st.st_size:
                raise OversizeOutputError(
                    f"submission tree entry {relative!r} grew while being archived"
                )
            info = tarfile.TarInfo(name=relative)
            info.type = tarfile.REGTYPE
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()
