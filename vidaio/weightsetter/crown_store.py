"""SQLite persistence for schema-v14 reward windows and competition results.

The filename is retained so deployed imports and migration ownership remain stable;
legacy v1 tables are left inert and readable. New code exclusively uses the additive
v2 tables created by migration 0008.
All economic timestamps come from chain-bound ``CompetitionResult.applied_at``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from vidaio.core.db import apply_migrations
from vidaio.tokenomics import (
    CompetitionResult,
    ContenderResult,
    RewardWindowState,
    TokenomicsConfig,
    resolve_reward_window,
)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class ResultConflictError(ValueError):
    """A cycle or competition id was re-ingested with different immutable content."""


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Apply weight-setter migrations; safe to call at every startup."""
    return apply_migrations(conn, MIGRATIONS_DIR)


def _contenders_json(result: CompetitionResult) -> str:
    return json.dumps(
        [
            {"hotkey": contender.hotkey, "uid": contender.uid, "score": contender.score}
            for contender in result.contenders
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def _result_from_row(row: sqlite3.Row) -> CompetitionResult:
    return CompetitionResult(
        competition_id=row["competition_id"],
        track=row["track"],
        cycle=int(row["cycle"]),
        applied_at=datetime.fromisoformat(row["applied_at"]),
        contenders=tuple(
            ContenderResult(
                hotkey=contender["hotkey"],
                uid=int(contender["uid"]),
                score=float(contender["score"]),
            )
            for contender in json.loads(row["contenders_json"])
        ),
        baseline_score=(
            None if row["baseline_score"] is None else float(row["baseline_score"])
        ),
        baseline_version=int(row["baseline_version"]),
        baseline_artifact_digest=row["baseline_artifact_digest"],
    )


def load_reward_window(conn: sqlite3.Connection) -> RewardWindowState:
    """Return the persisted reward window, or pristine IDLE before the first save."""
    row = conn.execute("SELECT * FROM reward_window_state WHERE id = 1").fetchone()
    if row is None:
        return RewardWindowState()
    return RewardWindowState(
        kind=row["kind"],
        starts_at=(
            None if row["starts_at"] is None else datetime.fromisoformat(row["starts_at"])
        ),
        ends_at=(
            None if row["ends_at"] is None else datetime.fromisoformat(row["ends_at"])
        ),
        podium_hotkeys=tuple(json.loads(row["podium_hotkeys_json"])),
        winner_hotkey=row["winner_hotkey"],
        winner_uid=row["winner_uid"],
        winner_score=row["winner_score"],
        winner_margin=row["winner_margin"],
        baseline_score=row["baseline_score"],
        baseline_version=row["baseline_version"],
        baseline_artifact_digest=row["baseline_artifact_digest"],
        source_competition_id=row["source_competition_id"],
        source_track=row["source_track"],
        source_cycle=row["source_cycle"],
        last_applied_cycle=row["last_applied_cycle"],
    )


def save_reward_window(conn: sqlite3.Connection, state: RewardWindowState) -> None:
    """Upsert the schema-v14 singleton reward-window row."""
    conn.execute(
        "INSERT INTO reward_window_state"
        " (id, kind, starts_at, ends_at, podium_hotkeys_json, winner_hotkey,"
        "  winner_uid, winner_score, winner_margin, baseline_score, baseline_version,"
        "  baseline_artifact_digest, source_competition_id, source_track, source_cycle,"
        "  last_applied_cycle)"
        " VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (id) DO UPDATE SET"
        "  kind = excluded.kind,"
        "  starts_at = excluded.starts_at,"
        "  ends_at = excluded.ends_at,"
        "  podium_hotkeys_json = excluded.podium_hotkeys_json,"
        "  winner_hotkey = excluded.winner_hotkey,"
        "  winner_uid = excluded.winner_uid,"
        "  winner_score = excluded.winner_score,"
        "  winner_margin = excluded.winner_margin,"
        "  baseline_score = excluded.baseline_score,"
        "  baseline_version = excluded.baseline_version,"
        "  baseline_artifact_digest = excluded.baseline_artifact_digest,"
        "  source_competition_id = excluded.source_competition_id,"
        "  source_track = excluded.source_track,"
        "  source_cycle = excluded.source_cycle,"
        "  last_applied_cycle = excluded.last_applied_cycle",
        (
            state.kind.value,
            state.starts_at.isoformat() if state.starts_at is not None else None,
            state.ends_at.isoformat() if state.ends_at is not None else None,
            json.dumps(list(state.podium_hotkeys), separators=(",", ":")),
            state.winner_hotkey,
            state.winner_uid,
            state.winner_score,
            state.winner_margin,
            state.baseline_score,
            state.baseline_version,
            state.baseline_artifact_digest,
            state.source_competition_id,
            state.source_track,
            state.source_cycle,
            state.last_applied_cycle,
        ),
    )


def ingest_competition_result(
    conn: sqlite3.Connection,
    result: CompetitionResult,
    config: TokenomicsConfig,
) -> bool:
    """Atomically persist one immutable result and replay-safe reward-window fold."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = conn.execute(
            "SELECT * FROM competition_results_v2 WHERE cycle = ? OR competition_id = ?",
            (result.cycle, result.competition_id),
        ).fetchone()
        if existing is not None:
            same = _result_from_row(existing) == result
            conn.execute("ROLLBACK")
            if same:
                return False
            raise ResultConflictError(
                f"competition cycle/id ({result.cycle}, {result.competition_id!r}) was "
                "already ingested with different immutable content"
            )
        state = resolve_reward_window(config, load_reward_window(conn), result)
        conn.execute(
            "INSERT INTO competition_results_v2"
            " (cycle, competition_id, track, applied_at, contenders_json, baseline_score,"
            "  baseline_version, baseline_artifact_digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result.cycle,
                result.competition_id,
                result.track,
                result.applied_at.isoformat(),
                _contenders_json(result),
                result.baseline_score,
                result.baseline_version,
                result.baseline_artifact_digest,
            ),
        )
        save_reward_window(conn, state)
    except ResultConflictError:
        raise
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return True


def mirror_epoch_reward_read_model(
    conn: sqlite3.Connection,
    *,
    result: CompetitionResult | None,
    state: RewardWindowState,
) -> bool:
    """Atomically mirror the exact authenticated EpochLog reward read model.

    Production/shared weight setting follows the authority log's already-derived
    vector; this local state is presentation data for the dashboard, never an input
    to that submission.  Do not re-fold ``result`` here: doing so from a validator's
    potentially incomplete local history could disagree with the predecessor-folded
    ``state`` committed by the EpochLog.  Instead, retain an immutable copy of any
    result carried by the log and mirror that log's exact window state.

    Replays are idempotent.  A different result under an existing cycle/id, or a
    reward-state regression/conflict, aborts the whole transaction so the dashboard
    cannot silently display a locally mixed history.
    """
    if not isinstance(state, RewardWindowState):
        raise TypeError("state must be a RewardWindowState")
    if result is not None and not isinstance(result, CompetitionResult):
        raise TypeError("result must be a CompetitionResult or None")

    conn.execute("BEGIN IMMEDIATE")
    try:
        changed = False
        if result is not None:
            existing_rows = conn.execute(
                "SELECT * FROM competition_results_v2"
                " WHERE cycle = ? OR competition_id = ? ORDER BY cycle",
                (result.cycle, result.competition_id),
            ).fetchall()
            if existing_rows:
                if (
                    len(existing_rows) != 1
                    or _result_from_row(existing_rows[0]) != result
                ):
                    raise ResultConflictError(
                        f"competition cycle/id ({result.cycle}, "
                        f"{result.competition_id!r}) conflicts with the immutable "
                        "authority-epoch read model"
                    )
            else:
                conn.execute(
                    "INSERT INTO competition_results_v2"
                    " (cycle, competition_id, track, applied_at, contenders_json,"
                    "  baseline_score, baseline_version, baseline_artifact_digest)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        result.cycle,
                        result.competition_id,
                        result.track,
                        result.applied_at.isoformat(),
                        _contenders_json(result),
                        result.baseline_score,
                        result.baseline_version,
                        result.baseline_artifact_digest,
                    ),
                )
                changed = True

        row_exists = (
            conn.execute("SELECT 1 FROM reward_window_state WHERE id = 1").fetchone()
            is not None
        )
        current = load_reward_window(conn)
        current_cycle = current.last_applied_cycle
        incoming_cycle = state.last_applied_cycle
        if current_cycle is not None and (
            incoming_cycle is None or incoming_cycle < current_cycle
        ):
            raise ResultConflictError(
                "authenticated authority reward-window state would regress the local "
                f"read model from cycle {current_cycle} to {incoming_cycle}"
            )
        if current_cycle == incoming_cycle and current != state:
            raise ResultConflictError(
                "authenticated authority reward-window state conflicts with the "
                f"already mirrored cycle {incoming_cycle}"
            )
        if not row_exists or current != state:
            save_reward_window(conn, state)
            changed = True
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return changed


def latest_result(conn: sqlite3.Connection) -> CompetitionResult | None:
    """Return the highest-cycle immutable schema-v14 result."""
    row = conn.execute(
        "SELECT * FROM competition_results_v2 ORDER BY cycle DESC LIMIT 1"
    ).fetchone()
    return _result_from_row(row) if row is not None else None
