import sqlite3
from pathlib import Path

import pytest

from vidaio.core.db import apply_migrations, connect, connect_read_only


def test_wal_and_pragmas(tmp_path: Path) -> None:
    conn = connect(tmp_path / "t.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_read_only_connection_observes_live_wal_but_cannot_write(tmp_path: Path) -> None:
    db = tmp_path / "directory with spaces" / "source.db"
    writer = connect(db)
    writer.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY, value TEXT)")
    writer.execute("INSERT INTO evidence(value) VALUES ('first')")

    reader = connect_read_only(db)
    assert reader.execute("SELECT value FROM evidence").fetchone()[0] == "first"
    assert reader.execute("PRAGMA query_only").fetchone()[0] == 1
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.execute("INSERT INTO evidence(value) VALUES ('forbidden')")

    writer.execute("INSERT INTO evidence(value) VALUES ('second')")
    assert reader.execute("SELECT count(*) FROM evidence").fetchone()[0] == 2


def test_read_only_connection_never_creates_a_missing_database(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        connect_read_only(missing)
    assert not missing.exists()


def test_migrations_apply_once(tmp_path: Path) -> None:
    mig = tmp_path / "migrations"
    mig.mkdir()
    (mig / "0001_init.sql").write_text("CREATE TABLE a (id INTEGER PRIMARY KEY);")
    (mig / "0002_more.sql").write_text("CREATE TABLE b (id INTEGER PRIMARY KEY);")
    conn = connect(tmp_path / "t.db")

    ran = apply_migrations(conn, mig)
    assert ran == ["0001_init.sql", "0002_more.sql"]
    assert apply_migrations(conn, mig) == []  # idempotent via ledger

    (mig / "0003_late.sql").write_text("ALTER TABLE a ADD COLUMN v TEXT;")
    assert apply_migrations(conn, mig) == ["0003_late.sql"]
    names = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations")}
    assert names == {"0001_init.sql", "0002_more.sql", "0003_late.sql"}


def test_failed_migration_is_fully_rolled_back(tmp_path: Path) -> None:
    """A mid-script failure must leave NO schema change and NO ledger row."""
    mig = tmp_path / "migrations"
    mig.mkdir()
    (mig / "0001_bad.sql").write_text(
        "CREATE TABLE first_ok (id INTEGER PRIMARY KEY);\n"
        "CREATE TABLE nope (id INTEGER PRIMARY KEY, FOREIGN KEY (id) REFERENCES);\n"
    )
    conn = connect(tmp_path / "t.db")
    with pytest.raises(sqlite3.OperationalError):
        apply_migrations(conn, mig)
    # neither the first statement's table nor a schema_migrations row survives
    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert "first_ok" not in tables
    assert conn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 0
    # the corrected file can simply be re-applied — nothing partial blocks it
    (mig / "0001_bad.sql").write_text(
        "CREATE TABLE first_ok (id INTEGER PRIMARY KEY);\n"
        "CREATE TABLE second_ok (id INTEGER PRIMARY KEY);\n"
    )
    assert apply_migrations(conn, mig) == ["0001_bad.sql"]
    conn.execute("INSERT INTO second_ok (id) VALUES (1)")


def test_trigger_bodies_survive_statement_splitting(tmp_path: Path) -> None:
    """Semicolons inside BEGIN...END trigger bodies must not split statements."""
    mig = tmp_path / "migrations"
    mig.mkdir()
    (mig / "0001_trigger.sql").write_text(
        "-- a leading comment; with a semicolon\n"
        "CREATE TABLE guarded (id INTEGER PRIMARY KEY, note TEXT);\n"
        "CREATE TRIGGER guarded_no_update\n"
        "BEFORE UPDATE ON guarded\n"
        "BEGIN\n"
        "    SELECT RAISE(ABORT, 'guarded is append-only');\n"
        "    SELECT RAISE(ABORT, 'unreachable; kept to exercise multi-statement bodies');\n"
        "END;\n"
        "CREATE TABLE after_trigger (id INTEGER PRIMARY KEY);\n"
    )
    conn = connect(tmp_path / "t.db")
    assert apply_migrations(conn, mig) == ["0001_trigger.sql"]
    conn.execute("INSERT INTO guarded (id, note) VALUES (1, 'x')")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE guarded SET note = 'y' WHERE id = 1")
    conn.execute("INSERT INTO after_trigger (id) VALUES (1)")  # statement after END ran


def test_migration_commit_is_atomic_with_ledger_row(tmp_path: Path) -> None:
    """The schema change and its schema_migrations row land in one transaction."""
    mig = tmp_path / "migrations"
    mig.mkdir()
    (mig / "0001_init.sql").write_text("CREATE TABLE t (id INTEGER PRIMARY KEY);")
    db = tmp_path / "t.db"
    apply_migrations(connect(db), mig)
    observer = connect(db)  # separate connection sees only committed state
    assert observer.execute(
        "SELECT count(*) FROM sqlite_master WHERE name = 't'"
    ).fetchone()[0] == 1
    assert observer.execute(
        "SELECT count(*) FROM schema_migrations WHERE name = '0001_init.sql'"
    ).fetchone()[0] == 1
