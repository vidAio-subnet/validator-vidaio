from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# pytest 9 uses importlib import mode; make the shared `support` module importable
# from the test modules in this directory.
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from vidaio.core.db import connect
from vidaio.competition import LifecycleEngine, migrate

from support import Driver


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = connect(":memory:")
    migrate(c)
    yield c
    c.close()


@pytest.fixture
def engine() -> LifecycleEngine:
    return LifecycleEngine()


@pytest.fixture
def driver(conn: sqlite3.Connection, engine: LifecycleEngine) -> Driver:
    return Driver(conn, engine)
