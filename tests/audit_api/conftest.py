"""Audit Results API suite fixtures. Machinery lives in audit_api_support.

Nothing binds a port (tests/conftest.py port guard) and nothing sleeps: the service
under test is driven through an in-process ASGI transport.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterator, Iterator

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from audit_api_support import AuditApi, AUDIT_BASE_URL  # noqa: E402


@pytest.fixture
def api() -> Iterator[AuditApi]:
    a = AuditApi()
    yield a
    a.close()


@pytest.fixture
def api_authed() -> Iterator[AuditApi]:
    a = AuditApi(api_token="s3cr3t-validator-token")
    yield a
    a.close()


@pytest.fixture
async def client(api: AuditApi) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=api.service.app)
    async with httpx.AsyncClient(transport=transport, base_url=AUDIT_BASE_URL) as c:
        yield c
