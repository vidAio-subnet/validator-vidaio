"""Migrations apply cleanly; FKs and CHECKs enforce; append-only tables are append-only."""

import sqlite3

import pytest

from vidaio.core.db import connect
from vidaio.competition import (
    BUILD_IDENTITY_SCHEME,
    MIGRATIONS_DIR,
    Phase,
    logical_build_identity,
    migrate,
)
from vidaio.competition import repository as repo

from support import START, T0, Driver, build_manifest

EXPECTED_TABLES = {
    "competitions",
    "contenders",
    "evaluation_items",
    "sandboxes",
    "batches",
    "performance_history",
    "events",
    "human_reviews",
    "modal_image_bindings",
}


def test_migrations_apply_cleanly_and_once(tmp_path) -> None:
    c = connect(tmp_path / "comp.db")
    ran = migrate(c)
    assert ran == [
        "0001_schema.sql",
        "0002_upscaling_item_bindings.sql",
        "0003_upscaling_item_geometry.sql",
        "0004_terminal_completion_order.sql",
        "0005_stable_modal_build_identity.sql",
    ]
    assert migrate(c) == []  # ledgered, not re-run
    tables = {
        r["name"]
        for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        if not r["name"].startswith("sqlite_")
    }
    assert EXPECTED_TABLES <= tables
    assert MIGRATIONS_DIR.name == "migrations"
    c.close()


def test_deleting_competition_with_contenders_is_restricted(driver: Driver) -> None:
    # Chosen semantics: RESTRICT everywhere — competition data is audit data; a
    # competition with any dependent rows can never be deleted in place.
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    driver.anchor(cid)
    driver.engine.tick(driver.conn, START)
    driver.enroll(cid, "hk-1")
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute("DELETE FROM competitions WHERE competition_id = ?", (cid,))
    # Contender rows referenced by performance history are protected the same way.
    item_ids = driver.seed_items(cid, 1)
    contender = repo.list_contenders(driver.conn, cid)[0]
    driver.score_contender(cid, contender.contender_id, item_ids, 0.9)
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute(
            "DELETE FROM contenders WHERE contender_id = ?", (contender.contender_id,)
        )


def test_contender_fk_requires_existing_competition(conn) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO contenders
               (competition_id, hotkey, is_calibration, repo_url, commit_sha, tree_sha,
                created_at, updated_at)
               VALUES ('ghost-01', 'hk', 0, 'r', 'c', 't', '2026', '2026')"""
        )


def test_calibration_check_constraints(driver: Driver) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)

    # Calibration row with a hotkey: forbidden.
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute(
            """INSERT INTO contenders
               (competition_id, hotkey, is_calibration, repo_url, commit_sha, tree_sha,
                created_at, updated_at)
               VALUES (?, 'baseline-hotkey', 1, 'r', 'c', 't', '2026', '2026')""",
            (cid,),
        )
    # Real contender without a hotkey: forbidden.
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute(
            """INSERT INTO contenders
               (competition_id, hotkey, is_calibration, repo_url, commit_sha, tree_sha,
                created_at, updated_at)
               VALUES (?, NULL, 0, 'r', 'c', 't', '2026', '2026')""",
            (cid,),
        )
    # Second calibration contender for the same competition: forbidden.
    for _ in range(1):
        driver.conn.execute(
            """INSERT INTO contenders
               (competition_id, hotkey, is_calibration, repo_url, commit_sha, tree_sha,
                created_at, updated_at)
               VALUES (?, NULL, 1, 'r', 'c', 't', '2026', '2026')""",
            (cid,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute(
            """INSERT INTO contenders
               (competition_id, hotkey, is_calibration, repo_url, commit_sha, tree_sha,
                created_at, updated_at)
               VALUES (?, NULL, 1, 'r2', 'c2', 't2', '2026', '2026')""",
            (cid,),
        )


def test_events_table_is_append_only(driver: Driver) -> None:
    manifest = build_manifest()
    driver.engine.create_competition(driver.conn, manifest, T0)
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute("UPDATE events SET event_type = 'tampered'")
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute("DELETE FROM events")


def test_modal_image_binding_is_typed_and_append_only(driver: Driver) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    repo_url = "https://example.invalid/contender.git"
    commit_sha = "a" * 40
    tree_sha = "b" * 40
    image_digest = logical_build_identity(
        repo_url=repo_url,
        commit_sha=commit_sha,
        tree_sha=tree_sha,
    )
    driver.conn.execute(
        """INSERT INTO modal_image_bindings
           (competition_id, contender_id, is_calibration, repo_url, commit_sha,
            tree_sha, build_identity_scheme, image_digest, provider,
            image_object_id, runtime_session_id, runtime_label, created_at)
           VALUES (?, 1, 0, ?, ?, ?, ?, ?, 'modal', 'im-exact-object', ?,
                   'vidaio-next-test-runtime', ?)""",
        (
            cid,
            repo_url,
            commit_sha,
            tree_sha,
            BUILD_IDENTITY_SCHEME,
            image_digest,
            "c" * 64,
            T0.isoformat(),
        ),
    )

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        driver.conn.execute(
            "UPDATE modal_image_bindings SET image_object_id = 'im-tampered'"
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        driver.conn.execute("DELETE FROM modal_image_bindings")


def test_terminal_completion_event_is_unique(driver: Driver) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    repo.set_status(driver.conn, cid, Phase.COMPLETED, START)
    repo.record_event(
        driver.conn,
        cid,
        "transition",
        START,
        from_phase=Phase.AWAITING_END_TIME,
        to_phase=Phase.COMPLETED,
        guard="test-terminal-event",
    )

    with pytest.raises(sqlite3.IntegrityError):
        repo.record_event(
            driver.conn,
            cid,
            "transition",
            START,
            from_phase=Phase.AWAITING_END_TIME,
            to_phase=Phase.COMPLETED,
            guard="duplicate-terminal-event",
        )


def test_invalid_status_rejected(driver: Driver) -> None:
    manifest = build_manifest()
    driver.engine.create_competition(driver.conn, manifest, T0)
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute(
            "UPDATE competitions SET status = 'LIMBO' WHERE competition_id = ?",
            (manifest.competition_id,),
        )
