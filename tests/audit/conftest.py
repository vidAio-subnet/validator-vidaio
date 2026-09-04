import sys
from pathlib import Path

import pytest

# Local helper modules (audit_helpers) must stay importable under
# --import-mode=importlib, which does not add test dirs to sys.path.
sys.path.insert(0, str(Path(__file__).parent))

from vidaio.audit.store import LocalFsStore


@pytest.fixture
def store(tmp_path: Path) -> LocalFsStore:
    return LocalFsStore(tmp_path / "audit")
