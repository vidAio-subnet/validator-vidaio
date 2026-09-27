"""The weight-setter read model stores results of competitions without a baseline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from vidaio.core.db import apply_migrations, connect
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import (
    CompetitionResult,
    CompetitionRules,
    ContenderResult,
    EmissionState,
)
from vidaio.weightsetter import crown_store

MIGRATIONS = Path(crown_store.__file__).parent / "migrations"
AT = datetime(2026, 10, 4, 13, 0, tzinfo=timezone.utc)


def _result(cycle: int, *, baseline: bool) -> CompetitionResult:
    return CompetitionResult(
        competition_id=f"comp-{cycle}",
        track="removal" if not baseline else "compression",
        cycle=cycle,
        applied_at=AT,
        contenders=(ContenderResult(hotkey="hk", uid=7, score=0.8),),
        baseline_score=0.5 if baseline else None,
        baseline_version=0 if baseline else None,
        baseline_artifact_digest=("a" * 64) if baseline else None,
        rules=CompetitionRules(crown_margin=0.05, crown_min_score=0.7, podium_min_score=0.3),
    )


def test_migration_keeps_rows_and_accepts_a_result_without_baseline(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ws.db")
    old = tmp_path / "old"
    old.mkdir()
    for f in sorted(MIGRATIONS.glob("*.sql")):
        if f.name < "0012":
            (old / f.name).write_text(f.read_text())
    apply_migrations(conn, old)
    conn.execute(
        "INSERT INTO competition_results_v2 (cycle, competition_id, track, applied_at,"
        " contenders_json, baseline_score, baseline_version, baseline_artifact_digest)"
        " VALUES (1, 'comp-1', 'compression', ?, '[]', 0.25, 0, ?)",
        (AT.isoformat(), "b" * 64),
    )
    conn.commit()
    before = [tuple(r) for r in conn.execute("SELECT * FROM competition_results_v2")]

    crown_store.migrate(conn)

    assert [tuple(r) for r in conn.execute("SELECT * FROM competition_results_v2")] == before
    config = TokenomicsConfig(competition_emissions_enabled=True)
    assert crown_store.ingest_competition_result(conn, _result(2, baseline=False), config)
    latest = crown_store.latest_result(conn)
    assert latest is not None and latest.baseline_version is None
    window = crown_store.load_reward_window(conn)
    assert window.kind is EmissionState.CROWN and window.baseline_version is None
