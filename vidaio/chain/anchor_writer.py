"""Cross-process serialization for Bittensor's one-slot commitment writer.

The Commitments pallet exposes one mutable slot per ``(netuid, hotkey)``.  A
process-local ``asyncio.Lock`` cannot stop a challenge anchor and an epoch anchor
running in separate processes from landing in the same block and invalidating one
another's historical receipt.  Production writers therefore share this small
POSIX advisory lock and hold it until their finalized/archive read-back is done.
An unresolved authority transaction also leaves a fsynced sidecar fence, so a
process exit cannot release the hotkey for an unrelated commitment overwrite.

The lock is deliberately filesystem-only: release deployments already require
the authority writer processes to share one coherent local host/volume for their
SQLite state.  It is not safe on NFS or independently mounted container filesystems.
"""

from __future__ import annotations

import asyncio
import contextvars
import errno
import fcntl
import json
import logging
import os
import re
import stat
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any


class AnchorWriterLockError(OSError):
    """The one-slot writer lane could not be acquired safely."""


class AnchorWriterPendingError(AnchorWriterLockError):
    """An earlier authority submission must be resolved before another write."""


_pending_warning_at: dict[str, float] = {}


def _pending_path(path: str | Path) -> Path:
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    return Path(resolved + ".pending-authority.json")


def read_anchor_writer_pending(path: str | Path | None) -> dict[str, Any] | None:
    """Read the atomic durable fence, including enough evidence for recovery."""
    if path is None or not str(path).strip():
        return None
    pending_path = _pending_path(path)
    try:
        fd = os.open(
            pending_path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise AnchorWriterPendingError(f"cannot read pending authority marker {pending_path}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1 << 20:
            raise ValueError("pending marker must be a small regular file")
        with os.fdopen(fd, "rb", closefd=False) as source:
            marker = json.load(source)
        if not isinstance(marker, dict) or re.fullmatch(
            r"0x[0-9a-f]{64}", str(marker.get("extrinsic_hash", ""))
        ) is None:
            raise ValueError("pending marker lacks a canonical extrinsic hash")
        return marker
    except (ValueError, TypeError) as exc:
        raise AnchorWriterPendingError(f"invalid pending authority marker {pending_path}") from exc
    finally:
        os.close(fd)


def _require_held(path: str | Path) -> None:
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if (resolved, asyncio.current_task()) not in _held_paths.get():
        raise AnchorWriterLockError("pending authority evidence requires the writer lane")


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mark_anchor_writer_pending(path: str | Path | None, marker: dict[str, Any]) -> None:
    """Fsync an atomic fence before dispatch; unrelated writers must then HOLD."""
    if path is None or not str(path).strip():
        return
    _require_held(path)
    existing = read_anchor_writer_pending(path)
    if existing is not None:
        if existing["extrinsic_hash"] != marker["extrinsic_hash"]:
            raise AnchorWriterPendingError("another authority hash already holds the writer lane")
        return
    pending_path = _pending_path(path)
    payload = json.dumps(marker, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    fd, temporary = tempfile.mkstemp(prefix=pending_path.name + ".", dir=pending_path.parent)
    try:
        with os.fdopen(fd, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, pending_path)
        _sync_directory(pending_path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def clear_anchor_writer_pending(path: str | Path | None, *, extrinsic_hash: str) -> None:
    """Release only the matching, durably terminal authority submission."""
    if path is None or not str(path).strip():
        return
    _require_held(path)
    marker = read_anchor_writer_pending(path)
    if marker is None:
        return
    if marker["extrinsic_hash"] != extrinsic_hash:
        raise AnchorWriterPendingError("refusing to clear a different authority hash")
    pending_path = _pending_path(path)
    pending_path.unlink()
    _sync_directory(pending_path.parent)


_held_paths: contextvars.ContextVar[frozenset[tuple[str, asyncio.Task | None]]] = (
    contextvars.ContextVar("vidaio_anchor_writer_lock_paths", default=frozenset())
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
    path: str | Path | None, *, timeout_seconds: float,
    pending_extrinsic_hash: str | None = None,
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
    owner = (resolved, asyncio.current_task())
    if owner in held:
        yield
        return

    lock_path = Path(resolved)
    fd = _open_lock_file(lock_path)
    acquired = False
    token: contextvars.Token | None = None
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

        marker = read_anchor_writer_pending(lock_path)
        if marker is not None and marker["extrinsic_hash"] != pending_extrinsic_hash:
            now = time.monotonic()
            if now - _pending_warning_at.get(resolved, float("-inf")) >= 60:
                _pending_warning_at[resolved] = now
                logging.getLogger(__name__).warning(
                    "authority submission pending; holding commitment writer",
                    extra={"fields": {"marker": marker}},
                )
            raise AnchorWriterPendingError(
                f"authority submission {marker['extrinsic_hash']} is pending; writer HOLD"
            )
        token = _held_paths.set(held | {owner})
        yield
    finally:
        if token is not None:
            _held_paths.reset(token)
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


__all__ = [
    "AnchorWriterLockError", "AnchorWriterPendingError", "anchor_writer_lock",
    "read_anchor_writer_pending", "mark_anchor_writer_pending", "clear_anchor_writer_pending",
]
