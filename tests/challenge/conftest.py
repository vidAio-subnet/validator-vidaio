import sqlite3
import sys
from pathlib import Path

import pytest

from vidaio.challenge import MIGRATIONS_DIR
from vidaio.core.db import apply_migrations, connect

# Make sibling test helpers (asset_factories.py) importable regardless of the
# pytest import mode in use.
_HERE = str(Path(__file__).parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "challenge.db")
    apply_migrations(c, MIGRATIONS_DIR)
    return c
