"""Fixtures for the reference-miner suite (ASGI app over real ffmpeg)."""

import sys
from pathlib import Path

import httpx
import pytest

from vidaio.miner import Miner

# Make sibling test helpers (miner_support.py) importable regardless of the
# pytest import mode in use.
_HERE = str(Path(__file__).parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from miner_support import FFMPEG, FFPROBE  # noqa: E402

_skip_without_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe not installed"
)


def pytest_collection_modifyitems(items) -> None:
    for item in items:
        item.add_marker(_skip_without_ffmpeg)


@pytest.fixture
def miner(tmp_path: Path) -> Miner:
    raw = {
        "core": {"data_dir": str(tmp_path / "data")},
        "miner": {
            "work_dir": str(tmp_path / "miner_work"),
            "ffmpeg_path": FFMPEG,
            "ffmpeg_timeout_seconds": 120,
            "enable_legacy_path_routes": True,
            # Most legacy transport tests exercise byte bounds/cleanup in the
            # deprecated wire shape. Dedicated artifact-v2 tests below cover
            # the production default and hotkey identity contract.
            "allow_unsigned_artifact_v1": True,
        },
    }
    return Miner(raw)


@pytest.fixture
async def miner_client(miner: Miner):
    transport = httpx.ASGITransport(app=miner.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://miner", timeout=180
    ) as c:
        yield c
