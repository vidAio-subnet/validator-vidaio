from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# pytest 9 uses importlib import mode; make the shared `registry_support` module
# importable from the test modules in this directory.
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from vidaio.audit.store import LocalFsStore
from vidaio.core.db import connect
from vidaio.registry import migrate

from registry_support import competition_db


@pytest.fixture
def conn() -> sqlite3.Connection:
    """The REGISTRY database (champions + registry events)."""
    c = connect(":memory:")
    migrate(c)
    yield c
    c.close()


@pytest.fixture
def comp_conn() -> sqlite3.Connection:
    """The COMPETITION database — the promotion's authority, migrated but empty.

    Tests seed it with `registry_support.seed_competition`; a test that seeds
    nothing is a test of the substituted-evidence path.
    """
    c = competition_db()
    yield c
    c.close()


@pytest.fixture
def store(tmp_path: Path) -> LocalFsStore:
    return LocalFsStore(tmp_path / "audit")
