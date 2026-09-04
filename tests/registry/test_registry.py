"""Legacy direct registry writes are sealed for schema v14."""

from __future__ import annotations

import sqlite3

import pytest

from vidaio.registry import registry
from vidaio.registry.registry import LegacyRegistryWriteDisabledError

from registry_support import NOW, candidate


@pytest.mark.parametrize("operation", ["promote", "rollback"])
def test_legacy_direct_write_cannot_activate_executable_state(
    conn: sqlite3.Connection, operation: str
) -> None:
    with pytest.raises(LegacyRegistryWriteDisabledError):
        if operation == "promote":
            registry.promote(conn, "compression", candidate(score=0.9), NOW)
        else:
            registry.rollback(conn, "compression", 1, "caller request", NOW)

    assert conn.execute("SELECT COUNT(*) FROM champions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM registry_events").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM baselines").fetchone()[0] == 0


def test_legacy_direct_write_functions_are_not_package_exports() -> None:
    import vidaio.registry as public

    assert not hasattr(public, "promote")
    assert not hasattr(public, "rollback")
    assert "promote" not in public.__all__
    assert "rollback" not in public.__all__
