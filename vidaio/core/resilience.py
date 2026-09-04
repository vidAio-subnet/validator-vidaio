"""Bounded exponential backoff and named timeouts — the SN44-grade edge discipline.

Every network/subprocess boundary in the system must go through one of these; an
unbounded await is a bug.
"""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class RetriesExhausted(Exception):
    pass


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except retry_on as exc:  # noqa: PERF203
            last = exc
            if attempt == attempts:
                break
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay *= 0.5 + random.random() / 2  # jitter in [0.5, 1.0)x
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            await asyncio.sleep(delay)
    raise RetriesExhausted(f"failed after {attempts} attempts") from last


async def with_timeout(coro: Awaitable[T], seconds: float, name: str = "operation") -> T:
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{name} timed out after {seconds}s") from exc
