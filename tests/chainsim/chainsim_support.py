"""Test doubles/helpers for the chainsim suite (importable under importlib mode)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Sequence

import httpx

from vidaio.tokenomics import MinerSnapshot

T0 = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)

#: Operator credential every test sim is pinned to (chainsim.operator_token).
OPERATOR_TOKEN = "test-operator-token"


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def register(client, hotkey: str, **fields) -> tuple[int, str]:
    """POST /register a NEW hotkey; returns (uid, the token issued once)."""
    response = await client.post("/register", json={"hotkey": hotkey, **fields})
    response.raise_for_status()
    body = response.json()
    assert body["auth_token"], f"no token issued for {hotkey!r}: {body}"
    return int(body["uid"]), str(body["auth_token"])


class FakeTime:
    """Frozen wall clock for the sim's lazy block production (advance explicitly)."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class Clock:
    """Deterministic tz-aware datetime clock (WeightSetter injection)."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeSnapshots:
    """Test SnapshotProvider: a fixed miner list."""

    def __init__(self, miners: Sequence[MinerSnapshot]) -> None:
        self.miners = list(miners)

    def miner_snapshots(self) -> Sequence[MinerSnapshot]:
        return list(self.miners)


def mk_miner(
    uid: int, *, track: str = "compression", score: float = 0.5, hotkey: str | None = None
) -> MinerSnapshot:
    return MinerSnapshot(
        uid=uid,
        hotkey=hotkey if hotkey is not None else f"hk{uid}",
        coldkey=f"ck{uid}",
        ip=f"10.0.0.{uid}",
        track=track,
        accumulate_score=score,
    )


class SyncASGITransport(httpx.BaseTransport):
    """Sync httpx transport over an ASGI app.

    HttpChainAdapter.refresh() is synchronous (ChainAdapter contract: reads are
    cached snapshots, refresh blocks briefly) but may be called from inside a
    running event loop (WeightSetter.attempt_once). Each request is executed on
    a worker thread with its own asyncio loop via httpx.ASGITransport, so no
    sleeps and no port binding are needed in tests.
    """

    def __init__(self, app: object) -> None:
        self._app = app

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        content = request.read()

        def run() -> tuple[httpx.Response, bytes]:
            async def go() -> tuple[httpx.Response, bytes]:
                transport = httpx.ASGITransport(app=self._app)  # type: ignore[arg-type]
                try:
                    inner = httpx.Request(
                        request.method, request.url, headers=request.headers, content=content
                    )
                    response = await transport.handle_async_request(inner)
                    body = await response.aread()
                    return response, body
                finally:
                    await transport.aclose()

            return asyncio.run(go())

        with ThreadPoolExecutor(max_workers=1) as pool:
            response, body = pool.submit(run).result()
        return httpx.Response(response.status_code, headers=response.headers, content=body)
