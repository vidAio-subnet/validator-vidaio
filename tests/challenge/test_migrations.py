from pathlib import Path

from vidaio.challenge import MIGRATIONS_DIR
from vidaio.core.db import apply_migrations, connect


def test_migrations_apply_cleanly_and_once(tmp_path: Path) -> None:
    conn = connect(tmp_path / "m.db")
    ran = apply_migrations(conn, MIGRATIONS_DIR)
    assert ran == [
        "0001_challenge.sql",
        "0002_challenge_binding.sql",
        "0003_commitment_dispatch_binding.sql",
        "0004_dispatch_order_allocator.sql",
        "0005_external_commitment_anchor.sql",
        "0006_dag_version.sql",
    ]
    assert apply_migrations(conn, MIGRATIONS_DIR) == []  # idempotent via ledger

    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {
        "assets",
        "provenance_log",
        "challenge_commitments",
        "challenge_commitment_anchors",
        "challenges",
    } <= tables
    columns = {
        row["name"]: row for row in conn.execute("PRAGMA table_info(challenges)")
    }
    assert columns["dag_version"]["notnull"] == 1
    assert columns["dag_version"]["dflt_value"] == "6"


def test_challenge_and_audit_migrations_co_apply_on_one_db(tmp_path: Path) -> None:
    """Both modules share the configured core database: their migrations must apply
    on ONE connection without any table collision (challenge's commitment table is
    challenge_commitments precisely so the audit ledger can keep its own name)."""
    import vidaio.audit as audit

    audit_migrations = getattr(
        audit, "MIGRATIONS_DIR", Path(audit.__file__).parent / "migrations"
    )
    conn = connect(tmp_path / "core.db")
    assert apply_migrations(conn, MIGRATIONS_DIR)
    assert apply_migrations(conn, audit_migrations)  # must not raise

    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    # challenge module's tables and the audit module's tables coexist
    assert {"assets", "provenance_log", "challenge_commitments", "challenges"} <= tables
    audit_tables = tables - {
        "assets",
        "provenance_log",
        "challenge_commitments",
        "challenges",
        "schema_migrations",
        "sqlite_sequence",
    }
    assert audit_tables  # the audit schema actually landed alongside ours
