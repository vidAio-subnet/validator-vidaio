"""One-slot commitment writer serialization regressions."""

from __future__ import annotations

import asyncio
import fcntl
import os
from pathlib import Path

import pytest

from vidaio.chain.anchor_writer import AnchorWriterLockError, anchor_writer_lock


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
