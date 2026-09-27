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


# ---- emission profile for these suites ---------------------------------------------
# Tokenomics v3 (2026-09-23) makes inference earn nothing by default. The mechanisms
# under test here (publication, rewritten-vector refusal, own audit, stake floors,
# recompute proofs) use inference weights as their vehicle, so they run on the
# supported v2 launch profile; v3 behaviour is covered by tests/tokenomics.
import pytest as _pytest
from vidaio.tokenomics import EMISSION_PROFILES as _PROFILES, TokenomicsConfig as _TC


@_pytest.fixture(autouse=True)
def _v2_emission_profile_defaults():
    saved = {k: _TC.model_fields[k].default for k in _PROFILES["v2"]}
    for k, v in _PROFILES["v2"].items():
        _TC.model_fields[k].default = v
    _TC.model_rebuild(force=True)
    try:
        yield
    finally:
        for k, v in saved.items():
            _TC.model_fields[k].default = v
        _TC.model_rebuild(force=True)
