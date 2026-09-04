"""Schema-v14 reward-window/result persistence and replay safety."""

from __future__ import annotations

import dataclasses
import sqlite3
from datetime import timedelta

import pytest
from weightsetter_support import T0

from vidaio.core.db import connect
from vidaio.tokenomics import (
    ContenderResult,
    EmissionState,
    RewardWindowState,
    TokenomicsConfig,
    resolve_reward_window,
    window_active,
)
from vidaio.weightsetter import (
    ResultConflictError,
    ingest_competition_result,
    latest_result,
    load_reward_window,
    migrate,
    save_reward_window,
)


def _contender(score: float, hotkey: str = "hk1", uid: int = 1) -> ContenderResult:
    return ContenderResult(hotkey=hotkey, uid=uid, score=score)


def test_pristine_store_has_idle_window_and_no_result(conn):
    assert load_reward_window(conn) == RewardWindowState()
    assert latest_result(conn) is None
    row = conn.execute("SELECT * FROM reward_window_state WHERE id = 1").fetchone()
    assert row is not None
    assert dict(row) == {
        "id": 1,
        "kind": "IDLE",
        "starts_at": None,
        "ends_at": None,
        "podium_hotkeys_json": "[]",
        "winner_hotkey": None,
        "winner_uid": None,
        "winner_score": None,
        "winner_margin": None,
        "baseline_score": None,
        "baseline_version": None,
        "baseline_artifact_digest": None,
        "source_competition_id": None,
        "source_track": None,
        "source_cycle": None,
        "last_applied_cycle": None,
    }

    # Startup migrations are replay-safe and never replace an earned state.
    assert migrate(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM reward_window_state").fetchone()[0] == 1


def test_seed_migration_repairs_preseed_v2_database(conn):
    """The additive migration also covers a DB that already applied schema v2."""
    migration = "0009_seed_reward_window_state.sql"
    conn.execute("DELETE FROM reward_window_state")
    conn.execute("DELETE FROM schema_migrations WHERE name = ?", (migration,))

    assert migrate(conn) == [migration]
    assert load_reward_window(conn) == RewardWindowState()
    row = conn.execute(
        "SELECT kind, podium_hotkeys_json FROM reward_window_state WHERE id = 1"
    ).fetchone()
    assert dict(row) == {"kind": "IDLE", "podium_hotkeys_json": "[]"}


def test_reward_window_roundtrip_and_expiry_across_restart(tmp_path, mk_result):
    cfg = TokenomicsConfig()
    db_path = tmp_path / "weightsetter.db"
    conn = connect(db_path)
    migrate(conn)
    result = mk_result(cycle=1, contenders=[_contender(0.65)])
    assert ingest_competition_result(conn, result, cfg) is True
    state = load_reward_window(conn)
    assert state.kind is EmissionState.CROWN
    assert state.winner_hotkey == "hk1"
    assert state.winner_uid == 1
    assert state.starts_at == T0
    assert state.podium_hotkeys == ("hk1",)
    assert state.last_applied_cycle == 1
    conn.close()

    conn2 = connect(db_path)
    migrate(conn2)
    reloaded = load_reward_window(conn2)
    assert reloaded == state
    assert latest_result(conn2) == result
    assert window_active(reloaded, T0 + timedelta(hours=167)) is True
    assert window_active(reloaded, T0 + timedelta(hours=169)) is False
    conn2.close()


def test_ingest_is_idempotent_by_cycle_and_id(conn, mk_result):
    cfg = TokenomicsConfig()
    result = mk_result(cycle=1, contenders=[_contender(0.65)])
    assert ingest_competition_result(conn, result, cfg) is True
    state = load_reward_window(conn)

    assert ingest_competition_result(conn, result, cfg) is False
    assert load_reward_window(conn) == state
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM competition_results_v2"
    ).fetchone()["n"]
    assert count == 1


def test_ingest_rejects_conflicting_content_for_same_cycle(conn, mk_result):
    cfg = TokenomicsConfig()
    result = mk_result(cycle=1, contenders=[_contender(0.65)])
    ingest_competition_result(conn, result, cfg)

    tampered = dataclasses.replace(result, baseline_score=0.9)
    with pytest.raises(ResultConflictError):
        ingest_competition_result(conn, tampered, cfg)
    assert latest_result(conn) == result


def test_schema_makes_v2_results_immutable(conn, mk_result):
    ingest_competition_result(
        conn,
        mk_result(cycle=1, contenders=[_contender(0.65)]),
        TokenomicsConfig(),
    )

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE competition_results_v2 SET baseline_score = 0.9 WHERE cycle = 1"
        )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM competition_results_v2 WHERE cycle = 1")


def test_below_breakthrough_margin_creates_podium_window(conn, mk_result):
    cfg = TokenomicsConfig()
    result = mk_result(cycle=3, contenders=[_contender(0.51)])
    assert ingest_competition_result(conn, result, cfg) is True
    state = load_reward_window(conn)
    assert state.kind is EmissionState.PODIUM
    assert state.winner_hotkey == "hk1"
    assert state.last_applied_cycle == 3


def test_save_reward_window_upserts_singleton(conn, mk_result):
    cfg = TokenomicsConfig()
    first = resolve_reward_window(
        cfg, RewardWindowState(), mk_result(cycle=1, contenders=[_contender(0.65)])
    )
    save_reward_window(conn, first)
    assert load_reward_window(conn) == first

    second = resolve_reward_window(
        cfg,
        first,
        mk_result(
            cycle=2,
            applied_at=T0 + timedelta(hours=1),
            contenders=[_contender(0.7, hotkey="hk2", uid=2)],
        ),
    )
    save_reward_window(conn, second)
    assert load_reward_window(conn) == second
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM reward_window_state"
    ).fetchone()["n"]
    assert count == 1


def test_latest_result_returns_highest_cycle(conn, mk_result):
    cfg = TokenomicsConfig()
    for cycle in (1, 3, 2):
        ingest_competition_result(
            conn,
            mk_result(
                cycle=cycle,
                applied_at=T0 + timedelta(hours=cycle),
                contenders=[_contender(0.6)],
            ),
            cfg,
        )
    latest = latest_result(conn)
    assert latest is not None
    assert latest.cycle == 3
