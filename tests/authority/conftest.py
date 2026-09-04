"""Scoring Authority suite fixtures. Machinery lives in authority_support.

Nothing binds a port (tests/conftest.py port guard) and nothing sleeps: the
service under test is driven through an in-process ASGI transport.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterator, Iterator

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from authority_support import Authority  # noqa: E402


@pytest.fixture
def authority(tmp_path: Path) -> Iterator[Authority]:
    a = Authority(tmp_path)
    yield a
    a.close()


@pytest.fixture
def authority_authed(tmp_path: Path) -> Iterator[Authority]:
    a = Authority(tmp_path, api_token="s3cr3t-validator-token")
    yield a
    a.close()


async def _client(a: Authority) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=a.service.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://authority.test") as c:
        yield c


@pytest.fixture
async def client(authority: Authority) -> AsyncIterator[httpx.AsyncClient]:
    async for c in _client(authority):
        yield c
