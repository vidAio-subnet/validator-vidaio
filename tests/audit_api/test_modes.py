"""Beacon and own-audit reports share one API without identity collisions."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from audit_api_support import AuditApi, NOW, make_report

from vidaio.audit_api.store import AuditResultsStore
from vidaio.auditor.report import AuditMode

POST = "/audit/report"


async def test_same_hotkey_epoch_modes_coexist_and_count_one_auditor(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    beacon = make_report(auditor_hotkey="hk-one", epoch_id=900)
    own = make_report(
        auditor_hotkey="hk-one", epoch_id=900, audit_mode=AuditMode.OWN_AUDIT
    )
    assert beacon.report_digest() != own.report_digest()

    assert (await client.post(POST, json=beacon.model_dump(mode="json"))).status_code == 201
    assert (await client.post(POST, json=own.model_dump(mode="json"))).status_code == 201

    stored = api.store.for_epoch(900)
    assert len(stored) == 2
    assert api.store.get_for_pair("hk-one", 900).report_id == beacon.report_digest()
    assert (
        api.store.get_for_pair("hk-one", 900, AuditMode.OWN_AUDIT).report_id
        == own.report_digest()
    )
    assert api.store.conflicts_for_epoch(900) == 0

    status = (await client.get("/audit/status", params={"epoch_id": 900})).json()
    assert status["auditors_reporting"] == 1
    assert status["reports_received"] == 2
    assert status["reports_by_mode"] == {"beacon": 1, "own_audit": 1}

    feed = (await client.get("/audit/feed", params={"limit": 10})).json()
    assert {row["audit_mode"] for row in feed["reports"]} == {
        "beacon",
        "own_audit",
    }
    # The historical metric stays intact; the additive metric supplies mode
    # visibility without breaking existing dashboards or alerts.
    assert api.metric("vidaio_audit_reports_received_total", verdict="CLEAN") == 2.0
    assert (
        api.metric(
            "vidaio_audit_reports_received_by_mode_total",
            audit_mode="beacon",
            verdict="CLEAN",
        )
        == 1.0
    )
    assert (
        api.metric(
            "vidaio_audit_reports_received_by_mode_total",
            audit_mode="own_audit",
            verdict="CLEAN",
        )
        == 1.0
    )


async def test_conflicts_are_scoped_to_the_report_mode(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    beacon = make_report(auditor_hotkey="hk-one", epoch_id=901)
    own = make_report(
        auditor_hotkey="hk-one", epoch_id=901, audit_mode=AuditMode.OWN_AUDIT
    )
    own_divergent = make_report(
        auditor_hotkey="hk-one",
        epoch_id=901,
        audit_mode=AuditMode.OWN_AUDIT,
        snapshot_digest="c" * 64,
    )

    assert (await client.post(POST, json=beacon.model_dump(mode="json"))).status_code == 201
    assert (await client.post(POST, json=own.model_dump(mode="json"))).status_code == 201
    assert (
        await client.post(POST, json=own_divergent.model_dump(mode="json"))
    ).status_code == 409

    assert api.store.conflicts_for_epoch(901) == 1
    assert api.store.conflicts_for_epoch(901, AuditMode.BEACON) == 0
    assert api.store.conflicts_for_epoch(901, AuditMode.OWN_AUDIT) == 1
    assert api.store.total_conflicts(AuditMode.BEACON) == 0
    assert api.store.total_conflicts(AuditMode.OWN_AUDIT) == 1
    assert (
        api.metric(
            "vidaio_audit_report_conflicts_by_mode_total",
            audit_mode="own_audit",
        )
        == 1.0
    )


def test_migration_backfills_historical_rows_as_beacon() -> None:
    """A v4 database upgrades without rewriting historical signed JSON."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE audit_reports (
            report_id TEXT PRIMARY KEY,
            auditor_hotkey TEXT NOT NULL,
            epoch_id INTEGER NOT NULL,
            snapshot_digest TEXT NOT NULL,
            pipeline_version TEXT NOT NULL,
            overall TEXT NOT NULL,
            competition_n INTEGER NOT NULL,
            inference_n INTEGER NOT NULL,
            sampled_at TEXT NOT NULL,
            report_json TEXT NOT NULL,
            received_at TEXT NOT NULL,
            UNIQUE (auditor_hotkey, epoch_id)
        );
        CREATE TABLE audit_report_conflicts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            auditor_hotkey TEXT NOT NULL,
            epoch_id INTEGER NOT NULL,
            kept_report_id TEXT NOT NULL,
            kept_snapshot_digest TEXT NOT NULL,
            rejected_report_id TEXT NOT NULL,
            rejected_snapshot_digest TEXT NOT NULL,
            rejected_overall TEXT NOT NULL DEFAULT 'CLEAN',
            recorded_at TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        "INSERT INTO schema_migrations(name) VALUES (?)",
        [(f"000{i}_{name}.sql",) for i, name in (
            (1, "audit_reports"),
            (2, "conflict_rejected_verdict"),
            (3, "inconclusive_overall"),
            (4, "conflict_inconclusive_verdict"),
        )],
    )
    historical = make_report(auditor_hotkey="hk-old", epoch_id=902)
    body = historical.model_dump(mode="json")
    body.pop("audit_mode")
    legacy_json = json.dumps(body, sort_keys=True)
    conn.execute(
        "INSERT INTO audit_reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            historical.report_digest(),
            historical.auditor_hotkey,
            historical.epoch_id,
            historical.snapshot_digest,
            historical.pipeline_version,
            historical.overall.value,
            historical.competition_n,
            historical.inference_n,
            historical.sampled_at.isoformat(),
            legacy_json,
            NOW.isoformat(),
        ),
    )

    store = AuditResultsStore(conn)
    stored = store.get(historical.report_digest())
    assert stored is not None
    assert stored.audit_mode is AuditMode.BEACON
    assert stored.report.audit_mode is AuditMode.BEACON
    assert stored.report.report_digest() == historical.report_digest()
    raw_json = conn.execute(
        "SELECT report_json FROM audit_reports WHERE report_id = ?",
        (historical.report_digest(),),
    ).fetchone()["report_json"]
    assert raw_json == legacy_json
