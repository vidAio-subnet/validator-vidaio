import asyncio

import pytest

from vidaio.core.resilience import RetriesExhausted, retry_async, with_timeout


async def test_retry_succeeds_after_failures() -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    result = await retry_async(flaky, attempts=5, base_delay=0.001, max_delay=0.002)
    assert result == "ok"
    assert calls["n"] == 3


async def test_retry_exhaustion_raises_with_cause() -> None:
    async def always_fails() -> None:
        raise ValueError("nope")

    with pytest.raises(RetriesExhausted) as exc_info:
        await retry_async(always_fails, attempts=2, base_delay=0.001)
    assert isinstance(exc_info.value.__cause__, ValueError)


async def test_with_timeout_names_the_operation() -> None:
    with pytest.raises(TimeoutError, match="slow-op timed out"):
        await with_timeout(asyncio.sleep(5), 0.01, name="slow-op")
