"""Cross-process serialization for Bittensor's one-slot commitment writer.

The Commitments pallet exposes one mutable slot per ``(netuid, hotkey)``.  A
process-local ``asyncio.Lock`` cannot stop a challenge anchor and an epoch anchor
running in separate processes from landing in the same block and invalidating one
another's historical receipt.  Production writers therefore share this small
POSIX advisory lock and hold it until their finalized/archive read-back is done.

The lock is deliberately filesystem-only: release deployments already require
the authority writer processes to share one coherent local host/volume for their
SQLite state.  It is not safe on NFS or independently mounted container filesystems.
"""

from __future__ import annotations

import asyncio
import contextvars
import errno
import fcntl
import os
import stat
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path


class AnchorWriterLockError(OSError):
    """The one-slot writer lane could not be acquired safely."""


_held_paths: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "vidaio_anchor_writer_lock_paths", default=frozenset()
)


def _open_lock_file(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise AnchorWriterLockError(
            f"cannot open anchor writer lock {path}: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise AnchorWriterLockError(
                f"anchor writer lock {path} is not a regular file"
            )
        # Existing files may have been created under a permissive umask.  The file
        # carries no secret, but other users must not be able to replace/truncate a
        # production coordination primitive.
        os.fchmod(fd, 0o600)
        return fd
    except Exception:
        os.close(fd)
        raise


@asynccontextmanager
async def anchor_writer_lock(
    path: str | Path | None, *, timeout_seconds: float
) -> AsyncIterator[None]:
    """Acquire a cancellation-safe, re-entrant cross-process writer lane.

    Nested callers in the same async task are a no-op.  This lets a high-level
    challenge/epoch transaction hold the lane through finalized read-back while
    the adapter's lower-level ``anchor_commitment`` applies the same protection to
    every other commitment caller.
    """
    if path is None or not str(path).strip():
        yield
        return
    if timeout_seconds <= 0:
        raise ValueError("anchor writer lock timeout must be positive")

    # ``Path.resolve`` would follow the final symlink before O_NOFOLLOW sees it.
    # Normalize lexically instead so a substituted lock-file symlink is refused.
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    held = _held_paths.get()
    if resolved in held:
        yield
        return

    lock_path = Path(resolved)
    fd = _open_lock_file(lock_path)
    acquired = False
    token: contextvars.Token[frozenset[str]] | None = None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + float(timeout_seconds)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise AnchorWriterLockError(
                        f"cannot acquire anchor writer lock {lock_path}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise AnchorWriterLockError(
                        f"timed out after {timeout_seconds:g}s waiting for anchor "
                        f"writer lock {lock_path}"
                    ) from exc
                await asyncio.sleep(min(0.05, remaining))

        token = _held_paths.set(held | {resolved})
        yield
    finally:
        if token is not None:
            _held_paths.reset(token)
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


__all__ = ["AnchorWriterLockError", "anchor_writer_lock"]
