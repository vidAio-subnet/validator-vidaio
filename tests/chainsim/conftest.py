from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

import httpx
import pytest

# pytest runs with --import-mode=importlib; make the local helper importable.
sys.path.insert(0, str(Path(__file__).parent))

from chainsim_support import OPERATOR_TOKEN, FakeTime, SyncASGITransport, bearer

from vidaio.chain.client import HttpChainAdapter
from vidaio.chainsim import ChainSim


@pytest.fixture
def fake_time() -> FakeTime:
    return FakeTime()


@pytest.fixture
def make_sim(tmp_path: Path, fake_time: FakeTime) -> Iterator[Callable[..., ChainSim]]:
    """ChainSim factory: frozen clock, tmp SQLite, ports 0. Overrides patch `chainsim`.

    The operator credential is PINNED (chainsim.operator_token) so tests can drive
    /advance, /reset and /report/write without scraping the generated token file.
    """
    sims: list[ChainSim] = []

    def _mk(**overrides: Any) -> ChainSim:
        raw = {
            "core": {"metrics_port": 0},
            "chainsim": {
                "port": 0,
                "metrics_port": 0,
                "db_path": str(tmp_path / "chainsim.db"),
                "report_dir": str(tmp_path / "reports"),
                "operator_token": OPERATOR_TOKEN,
                **overrides,
            },
        }
        sim = ChainSim(raw, now=fake_time)
        sims.append(sim)
        return sim

    yield _mk
    for sim in sims:
        sim.close()


@pytest.fixture
def operator() -> dict[str, str]:
    """Authorization header for the pinned operator token."""
    return bearer(OPERATOR_TOKEN)


@pytest.fixture
def sim(make_sim: Callable[..., ChainSim]) -> ChainSim:
    return make_sim()


@pytest.fixture
async def client(sim: ChainSim) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=sim.app), base_url="http://sim"
    ) as c:
        yield c


@pytest.fixture
def make_adapter(sim: ChainSim) -> Iterator[Callable[..., HttpChainAdapter]]:
    """HttpChainAdapter wired to the sim's ASGI app (sync + async transports)."""
    adapters: list[HttpChainAdapter] = []

    def _mk(hotkey: str = "local-validator", auth_token: str | None = None) -> HttpChainAdapter:
        adapter = HttpChainAdapter(
            "http://sim",
            validator_hotkey=hotkey,
            auth_token=auth_token,
            client=httpx.Client(transport=SyncASGITransport(sim.app)),
            async_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=sim.app)),
        )
        adapters.append(adapter)
        return adapter

    yield _mk
    for adapter in adapters:
        adapter.close()
