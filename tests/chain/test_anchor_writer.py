"""One-slot commitment writer serialization regressions."""

from __future__ import annotations

import asyncio
import fcntl
import os
from pathlib import Path

import pytest

from vidaio.chain.anchor_writer import (
    AnchorWriterLockError, AnchorWriterPendingError, anchor_writer_lock,
    clear_anchor_writer_pending, mark_anchor_writer_pending, read_anchor_writer_pending,
)


@pytest.mark.asyncio
async def test_writer_lane_is_reentrant_and_serializes_other_tasks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "anchor.lock"
    entered = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []

    async def first() -> None:
        async with anchor_writer_lock(path, timeout_seconds=1):
            async with anchor_writer_lock(path, timeout_seconds=1):
                order.append("first")
                entered.set()
                await release.wait()

    async def second() -> None:
        await entered.wait()
        async with anchor_writer_lock(path, timeout_seconds=1):
            order.append("second")

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())
    await entered.wait()
    await asyncio.sleep(0.05)
    assert order == ["first"]
    release.set()
    await asyncio.gather(first_task, second_task)
    assert order == ["first", "second"]
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_writer_lane_times_out_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "anchor.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(AnchorWriterLockError, match="timed out"):
            async with anchor_writer_lock(path, timeout_seconds=0.01):
                pytest.fail("contended writer lane must not open")
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@pytest.mark.asyncio
async def test_writer_lane_refuses_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.lock"
    target.touch()
    link = tmp_path / "anchor.lock"
    link.symlink_to(target)
    with pytest.raises(AnchorWriterLockError, match="cannot open"):
        async with anchor_writer_lock(link, timeout_seconds=1):
            pytest.fail("symlink lock must not open")


async def test_pending_marker_survives_lane_release_and_warns_once_per_minute(
    tmp_path: Path, caplog,
) -> None:
    path = tmp_path / "anchor.lock"
    txid = "0x" + "a" * 64
    marker = {"extrinsic_hash": txid, "epoch_id": 42, "submit_block": 100,
              "pid": 123, "time": 1.0}
    async with anchor_writer_lock(path, timeout_seconds=1):
        mark_anchor_writer_pending(path, marker)
    assert path.read_bytes() == b""
    assert read_anchor_writer_pending(path) == marker
    assert Path(str(path) + ".pending-authority.json").stat().st_mode & 0o777 == 0o600
    for _ in range(2):
        with pytest.raises(AnchorWriterPendingError, match=txid):
            async with anchor_writer_lock(path, timeout_seconds=1):
                pytest.fail("pending marker must hold another writer")
    assert sum("holding commitment writer" in item.message for item in caplog.records) == 1
    async with anchor_writer_lock(path, timeout_seconds=1, pending_extrinsic_hash=txid):
        clear_anchor_writer_pending(path, extrinsic_hash=txid)
    async with anchor_writer_lock(path, timeout_seconds=1):
        assert read_anchor_writer_pending(path) is None


async def test_child_task_cannot_inherit_reentrancy_past_pending_writer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "anchor.lock"
    parent_released = asyncio.Event()
    txid = "0x" + "b" * 64

    async def child():
        await parent_released.wait()
        with pytest.raises(AnchorWriterPendingError):
            async with anchor_writer_lock(path, timeout_seconds=1):
                pytest.fail("inherited context must not bypass the durable fence")

    async with anchor_writer_lock(path, timeout_seconds=1):
        child_task = asyncio.create_task(child())
        mark_anchor_writer_pending(path, {"extrinsic_hash": txid})
    parent_released.set()
    await child_task
