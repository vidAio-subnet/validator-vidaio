"""Legacy writes fail before opening a transaction or mutating either schema."""

from __future__ import annotations

import sqlite3

import pytest

from vidaio.registry import registry
from vidaio.registry.registry import LegacyRegistryWriteDisabledError

from registry_support import NOW, candidate


def test_disabled_legacy_promotion_preserves_an_open_caller_transaction(
    conn: sqlite3.Connection,
) -> None:
    conn.execute("CREATE TABLE caller_state (value TEXT NOT NULL)")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT INTO caller_state VALUES ('before')")
    assert conn.in_transaction

    with pytest.raises(LegacyRegistryWriteDisabledError):
        registry.promote(conn, "compression", candidate(score=1.0), NOW)

    assert conn.in_transaction
    assert conn.execute("SELECT value FROM caller_state").fetchone()[0] == "before"
    assert conn.execute("SELECT COUNT(*) FROM champions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM baselines").fetchone()[0] == 0
