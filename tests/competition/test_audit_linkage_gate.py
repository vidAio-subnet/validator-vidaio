"""Audit-linkage completion gate:
the AWAITING_END_TIME -> COMPLETED tick transition requires every
performance_history row — the baseline calibration rows included — to carry its
audit_bundle_digest (engine.audit_linkage_gaps == []) AND, when a baseline calibration
contender exists, a baseline score row for every evaluation item
(repository.count_missing_calibration_rows == 0). The digest itself is a validated
64-char lowercase sha256 hex string and write-once — in Python AND at the SQL level
(CHECK + trigger). Gated by CompetitionConfig.require_audit_linkage (default True;
False is tests/dev only and every bypassed completion is logged)."""

from __future__ import annotations

import sqlite3

import pytest

from vidaio.competition import CompetitionConfig, LifecycleEngine, Phase
from vidaio.competition import repository as repo

from support import BASELINE, END, Driver, build_manifest


def _fields(record) -> dict:
    return getattr(record, "fields", None) or {}


def test_completion_blocked_while_linkage_gaps_exist(
    driver: Driver, caplog: pytest.LogCaptureFixture
) -> None:
    cid, ids = driver.run_to_awaiting(
        build_manifest(), {"hk-1": 0.9, "hk-2": 0.5}, link_audit=False
    )
    gaps = driver.engine.audit_linkage_gaps(driver.conn, cid)
    assert len(gaps) == 4  # 2 contenders x 2 items, none linked yet

    with caplog.at_level("INFO", logger="vidaio.competition.engine"):
        assert driver.engine.tick(driver.conn, END) == []
    assert driver.phase(cid) is Phase.AWAITING_END_TIME  # deferred, not failed
    deferred = [
        r for r in caplog.records if _fields(r).get("reason") == "audit_linkage_gaps"
    ]
    assert deferred, "deferral must log a structured reason"
    assert _fields(deferred[0])["gap_count"] == 4
    assert _fields(deferred[0])["competition_id"] == cid

    # The audit runner links every bundle -> the next tick completes.
    assert driver.link_audit_bundles(cid) == 4
    assert driver.engine.audit_linkage_gaps(driver.conn, cid) == []
    driver.engine.tick(driver.conn, END)
    assert driver.phase(cid) is Phase.COMPLETED


def test_calibration_row_gap_alone_blocks_completion(driver: Driver) -> None:
    """Baseline recomputability: the baseline score drives the ratchet/crown, so a baseline
    performance row without its audit bundle digest blocks completion exactly like
    a contender's row."""
    cid, _ = driver.run_to_awaiting(
        build_manifest(baseline=BASELINE), {"hk-1": 0.9}, baseline_score=0.5, link_audit=False
    )
    baseline = next(c for c in repo.list_contenders(driver.conn, cid) if c.is_calibration)
    # Link every row EXCEPT the baseline calibration rows.
    for row in driver.conn.execute(
        "SELECT performance_id, contender_id FROM performance_history"
        " WHERE competition_id = ?",
        (cid,),
    ).fetchall():
        if row["contender_id"] != baseline.contender_id:
            repo.set_audit_bundle_digest(driver.conn, row["performance_id"], "ab" * 32)

    gaps = driver.engine.audit_linkage_gaps(driver.conn, cid)
    assert gaps and all(cid_ == baseline.contender_id for cid_, _ in gaps)

    driver.engine.tick(driver.conn, END)
    assert driver.phase(cid) is Phase.AWAITING_END_TIME  # baseline gap alone blocks

    driver.link_audit_bundles(cid)  # fills the baseline rows too
    driver.engine.tick(driver.conn, END)
    assert driver.phase(cid) is Phase.COMPLETED


def test_require_audit_linkage_false_bypasses_and_logs(
    conn, caplog: pytest.LogCaptureFixture
) -> None:
    engine = LifecycleEngine(CompetitionConfig(require_audit_linkage=False))
    driver = Driver(conn, engine)
    cid, _ = driver.run_to_awaiting(build_manifest(), {"hk-1": 0.9}, link_audit=False)
    assert engine.audit_linkage_gaps(conn, cid) != []  # gaps exist ...

    with caplog.at_level("WARNING", logger="vidaio.competition.engine"):
        engine.tick(conn, END)
    assert driver.phase(cid) is Phase.COMPLETED  # ... but the flag bypasses the gate
    bypassed = [r for r in caplog.records if "BYPASSED" in r.getMessage()]
    assert bypassed, "every bypassed completion must be logged"
    assert _fields(bypassed[0])["require_audit_linkage"] is False
    assert _fields(bypassed[0])["competition_id"] == cid


def test_toctou_unlinked_row_after_precheck_defers_completion(
    driver: Driver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A score row landing between the linkage pre-check and the transition's
    BEGIN IMMEDIATE transaction is caught by the in-transaction re-check: the
    competition stays in AWAITING_END_TIME instead of completing unlinked."""
    cid, _ = driver.run_to_awaiting(build_manifest(), {"hk-1": 0.9}, link_audit=False)
    real = type(driver.engine).audit_linkage_gaps
    calls = {"n": 0}

    def racy(conn, competition_id):
        calls["n"] += 1
        if calls["n"] == 1:
            return []  # pre-check sees full linkage; the unlinked rows land after
        return real(driver.engine, conn, competition_id)

    monkeypatch.setattr(driver.engine, "audit_linkage_gaps", racy)
    assert driver.engine.tick(driver.conn, END) == []  # logged + deferred, no raise
    assert calls["n"] >= 2  # the in-txn re-check ran and caught it
    assert driver.phase(cid) is Phase.AWAITING_END_TIME
    assert all(e["to_phase"] != "COMPLETED" for e in driver.events(cid))


# ---- round-3 NEW-1: the digest is validated + write-once (Python AND SQL) ----


def _one_performance_row(driver: Driver) -> tuple[str, int]:
    cid, _ = driver.run_to_awaiting(build_manifest(), {"hk-1": 0.9}, link_audit=False)
    row = driver.conn.execute(
        "SELECT performance_id FROM performance_history WHERE competition_id = ?"
        " ORDER BY performance_id LIMIT 1",
        (cid,),
    ).fetchone()
    return cid, int(row["performance_id"])


def test_digest_format_enforced_in_python_and_sql(driver: Driver) -> None:
    """An empty or malformed digest can never 'link' a row: the setter rejects it
    (ValueError) and the schema CHECK independently rejects it at the SQL level."""
    cid, pid = _one_performance_row(driver)
    for bad in ("", "abc", "A" * 64, "g" * 64, "ab" * 33, None):
        with pytest.raises(ValueError, match="64-char"):
            repo.set_audit_bundle_digest(driver.conn, pid, bad)  # type: ignore[arg-type]
    for bad_sql in ("", "abc", "Z" * 64, "ab" * 33):
        with pytest.raises(sqlite3.IntegrityError):
            driver.conn.execute(
                "UPDATE performance_history SET audit_bundle_digest = ?"
                " WHERE performance_id = ?",
                (bad_sql, pid),
            )
    # Nothing slipped through: the row is still an open linkage gap.
    assert len(driver.engine.audit_linkage_gaps(driver.conn, cid)) == 2
    # An unknown performance row raises instead of silently updating nothing.
    with pytest.raises(ValueError, match="unknown performance row"):
        repo.set_audit_bundle_digest(driver.conn, 999_999, "a" * 64)


def test_digest_is_write_once_in_python_and_sql(driver: Driver) -> None:
    cid, pid = _one_performance_row(driver)
    first = "a" * 64
    other = "b" * 64
    repo.set_audit_bundle_digest(driver.conn, pid, first)
    # Same value again: idempotent no-op.
    repo.set_audit_bundle_digest(driver.conn, pid, first)
    # A different value is refused by the setter ...
    with pytest.raises(ValueError, match="write-once"):
        repo.set_audit_bundle_digest(driver.conn, pid, other)
    # ... and by the SQL trigger, for overwrite AND clearing alike.
    with pytest.raises(sqlite3.IntegrityError, match="write-once"):
        driver.conn.execute(
            "UPDATE performance_history SET audit_bundle_digest = ?"
            " WHERE performance_id = ?",
            (other, pid),
        )
    with pytest.raises(sqlite3.IntegrityError, match="write-once"):
        driver.conn.execute(
            "UPDATE performance_history SET audit_bundle_digest = NULL"
            " WHERE performance_id = ?",
            (pid,),
        )
    # A same-value SQL rewrite is allowed (idempotent), and the value survived it all.
    driver.conn.execute(
        "UPDATE performance_history SET audit_bundle_digest = ? WHERE performance_id = ?",
        (first, pid),
    )
    row = driver.conn.execute(
        "SELECT audit_bundle_digest FROM performance_history WHERE performance_id = ?",
        (pid,),
    ).fetchone()
    assert row["audit_bundle_digest"] == first


def test_invalid_stored_digest_still_counts_as_gap(driver: Driver) -> None:
    """Robustness: even against a database that lost the CHECK constraint, a
    present-but-invalid digest is a GAP, not linkage — presence of a string is
    never enough to unlock completion."""
    cid, pid = _one_performance_row(driver)
    driver.conn.execute("PRAGMA ignore_check_constraints = ON")
    driver.conn.execute(
        "UPDATE performance_history SET audit_bundle_digest = '' WHERE performance_id = ?",
        (pid,),
    )
    driver.conn.execute("PRAGMA ignore_check_constraints = OFF")
    # Both rows are still gaps: one NULL, one empty-string.
    assert len(driver.engine.audit_linkage_gaps(driver.conn, cid)) == 2
    driver.engine.tick(driver.conn, END)
    assert driver.phase(cid) is Phase.AWAITING_END_TIME


def test_blob_digest_rejected_by_check_and_counts_as_gap(driver: Driver) -> None:
    """review round-4 blocker: a 64-byte BLOB containing a NUL passes length()=64
    and GLOB (which stops at NUL). The CHECK must reject non-text values, and the
    gap predicate must treat them as gaps even on a constraint-less database."""
    cid, pid = _one_performance_row(driver)
    blob = b"a\x00" + b"Z" * 62
    assert len(blob) == 64
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute(
            "UPDATE performance_history SET audit_bundle_digest = ? WHERE performance_id = ?",
            (blob, pid),
        )
    # Constraint-less database: the engine predicate must still see a gap.
    driver.conn.execute("PRAGMA ignore_check_constraints = ON")
    driver.conn.execute(
        "UPDATE performance_history SET audit_bundle_digest = ? WHERE performance_id = ?",
        (blob, pid),
    )
    driver.conn.execute("PRAGMA ignore_check_constraints = OFF")
    assert len(driver.engine.audit_linkage_gaps(driver.conn, cid)) == 2
    driver.engine.tick(driver.conn, END)
    assert driver.phase(cid) is Phase.AWAITING_END_TIME


# ---- round-3 NEW-2: a baseline with NO score rows must not bypass the gate ----


def _baseline_and_items(driver: Driver, cid: str) -> tuple[int, list[int]]:
    baseline = next(c for c in repo.list_contenders(driver.conn, cid) if c.is_calibration)
    item_ids = [
        r["item_id"]
        for r in driver.conn.execute(
            "SELECT item_id FROM evaluation_items WHERE competition_id = ?"
            " ORDER BY item_index",
            (cid,),
        )
    ]
    return baseline.contender_id, item_ids


def test_baseline_with_no_score_rows_stalls_completion(
    driver: Driver, caplog: pytest.LogCaptureFixture
) -> None:
    """A configured baseline whose score rows were never written must stall completion:
    audit_linkage_gaps only sees rows that EXIST, so without the calibration-rows
    check the baseline's absence would unlock the gate."""
    cid, _ = driver.run_to_awaiting(build_manifest(baseline=BASELINE), {"hk-1": 0.9})
    # The real contender's rows exist and are fully linked; the baseline has NONE.
    assert driver.engine.audit_linkage_gaps(driver.conn, cid) == []
    assert repo.count_missing_calibration_rows(driver.conn, cid) == 2

    with caplog.at_level("INFO", logger="vidaio.competition.engine"):
        assert driver.engine.tick(driver.conn, END) == []
    assert driver.phase(cid) is Phase.AWAITING_END_TIME  # stalled, not completed
    stalled = [
        r for r in caplog.records if _fields(r).get("reason") == "calibration_rows_missing"
    ]
    assert stalled, "the stall must log a structured reason"
    assert _fields(stalled[0])["missing_calibration_rows"] == 2
    assert _fields(stalled[0])["competition_id"] == cid

    # The pipeline scores the baseline on every item and links the rows -> completes.
    baseline_id, item_ids = _baseline_and_items(driver, cid)
    driver.score_contender(cid, baseline_id, item_ids, 0.5)
    driver.link_audit_bundles(cid)
    assert repo.count_missing_calibration_rows(driver.conn, cid) == 0
    driver.engine.tick(driver.conn, END)
    assert driver.phase(cid) is Phase.COMPLETED


def test_toctou_baseline_rows_missing_after_precheck_defers_completion(
    driver: Driver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline-matrix check is re-run INSIDE the transition's BEGIN IMMEDIATE
    transaction: a pre-check that (racily) saw a complete baseline matrix cannot
    complete a competition whose baseline rows are actually missing."""
    cid, _ = driver.run_to_awaiting(build_manifest(baseline=BASELINE), {"hk-1": 0.9})
    real = repo.count_missing_calibration_rows
    calls = {"n": 0}

    def racy(conn, competition_id):
        calls["n"] += 1
        if calls["n"] == 1:
            return 0  # the pre-check believes the baseline matrix is complete
        return real(conn, competition_id)

    monkeypatch.setattr(repo, "count_missing_calibration_rows", racy)
    assert driver.engine.tick(driver.conn, END) == []  # logged + deferred, no raise
    assert calls["n"] >= 2  # the in-txn re-check ran and caught it
    assert driver.phase(cid) is Phase.AWAITING_END_TIME
    assert all(e["to_phase"] != "COMPLETED" for e in driver.events(cid))


def test_baseline_with_full_rows_and_linkage_completes(driver: Driver) -> None:
    cid, _ = driver.run_to_awaiting(
        build_manifest(baseline=BASELINE), {"hk-1": 0.9}, baseline_score=0.5
    )
    assert repo.count_missing_calibration_rows(driver.conn, cid) == 0
    assert driver.engine.audit_linkage_gaps(driver.conn, cid) == []
    driver.engine.tick(driver.conn, END)
    assert driver.phase(cid) is Phase.COMPLETED
