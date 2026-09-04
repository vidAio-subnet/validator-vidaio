"""SQLite with WAL, foreign keys, busy timeout, and a simple ordered migration runner.

Each service module owns its migrations directory (e.g. vidaio/competition/migrations/)
and calls apply_migrations() at startup. Migration files are applied in sorted order and
recorded in schema_migrations; write them to be safe to re-run only via that ledger
(they are NOT re-executed once recorded).

Each migration file is atomic: its statements AND its schema_migrations row are
committed together in one explicit transaction, so a mid-script failure or crash
leaves neither partial schema nor a phantom ledger row — the corrected file can
simply be re-applied.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def connect_read_only(path: str | Path) -> sqlite3.Connection:
    """Open an existing SQLite database through the engine's read-only URI.

    This is for cross-service observers.  Unlike :func:`connect`, it never
    creates a missing file, changes journal mode, or grants an accidental write
    path.  It intentionally does *not* use ``immutable=1``: the source may be a
    live WAL database and readers must see committed WAL frames.
    """
    p = Path(path).expanduser().resolve()
    conn = sqlite3.connect(
        f"{p.as_uri()}?mode=ro", uri=True, timeout=30, isolation_level=None
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA query_only=ON")
    return conn


def _split_statements(script: str) -> list[str]:
    """Split a migration script into individually executable statements.

    Uses sqlite3.complete_statement so semicolons inside trigger bodies
    (BEGIN ... ; ... END;), strings, and comments do not split a statement.
    Needed because executescript() auto-commits and would break per-file
    transactional application.
    """
    statements: list[str] = []
    buf = ""
    for piece in script.split(";"):
        buf += piece + ";"
        if sqlite3.complete_statement(buf):
            stmt = buf.strip()
            buf = ""
            # Drop fragments with no actual SQL (pure whitespace/comments).
            lines = [ln.strip() for ln in stmt.splitlines()]
            if any(ln and not ln.startswith("--") for ln in lines) and stmt != ";":
                statements.append(stmt)
    trailing = buf.strip()
    if trailing and any(
        ln.strip() and not ln.strip().startswith("--") for ln in trailing.splitlines()
    ):
        raise ValueError(f"migration script ends with an incomplete statement: {trailing!r}")
    return statements


def apply_migrations(conn: sqlite3.Connection, migrations_dir: str | Path) -> list[str]:
    """Apply pending *.sql files from migrations_dir in sorted order. Returns names run.

    Each file runs inside BEGIN IMMEDIATE ... COMMIT together with its
    schema_migrations insert; on any failure the whole file is rolled back.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " name TEXT PRIMARY KEY,"
        " applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
    )
    applied = {row["name"] for row in conn.execute("SELECT name FROM schema_migrations")}
    ran: list[str] = []
    for f in sorted(Path(migrations_dir).glob("*.sql")):
        if f.name in applied:
            continue
        statements = _split_statements(f.read_text())
        conn.execute("BEGIN IMMEDIATE")
        try:
            for stmt in statements:
                conn.execute(stmt)
            conn.execute("INSERT INTO schema_migrations(name) VALUES (?)", (f.name,))
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        ran.append(f.name)
    return ran
