from __future__ import annotations

import sys
from pathlib import Path

import pytest

# pytest runs with --import-mode=importlib, which does not add test dirs to sys.path;
# make the local helper module (integration_support) importable. The name is unique
# across the test tree on purpose — a generic `support` would collide in sys.modules
# with tests/competition/support.py when the full suite runs.
sys.path.insert(0, str(Path(__file__).parent))

from integration_support import GoldenWorld, build_golden_world


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> GoldenWorld:
    """One golden end-to-end run shared by a test module (read-only in tests)."""
    return build_golden_world(tmp_path_factory.mktemp("golden"))


@pytest.fixture
def fresh_world(tmp_path: Path) -> GoldenWorld:
    """A private golden run for tests that mutate or spoof state."""
    return build_golden_world(tmp_path)
