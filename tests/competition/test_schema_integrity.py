"""Cross-competition tampering is a schema error: composite
(competition_id, id) FKs pin sandboxes/batches/performance_history/human_reviews to
entities of the SAME competition; audit linkage is per-(contender, item)."""

import hashlib
import sqlite3
from datetime import timedelta

import pytest

from vidaio.competition import repository as repo

from support import START, T0, Driver, build_manifest

TS = "2026-09-01T00:00:00+00:00"
DIGEST = "9" * 64


def _two_competitions(driver: Driver) -> tuple[str, str, int, int]:
    """comp-01 ENROLLING with a real contender; comp-02 SCHEDULED with a raw-inserted
    contender. Returns (cid_a, cid_b, contender_a, contender_b)."""
    a = build_manifest("comp-01")
    b = build_manifest("comp-02")
    driver.engine.create_competition(driver.conn, a, T0)
    driver.engine.create_competition(driver.conn, b, T0)
    driver.anchor("comp-01")
    driver.engine.tick(driver.conn, START)
    contender_a = driver.enroll("comp-01", "hk-a")
    cur = driver.conn.execute(
        """INSERT INTO contenders
           (competition_id, hotkey, is_calibration, repo_url, commit_sha, tree_sha,
            created_at, updated_at)
           VALUES ('comp-02', 'hk-b', 0, 'r', 'c', 't', ?, ?)""",
        (TS, TS),
    )
    return "comp-01", "comp-02", contender_a, int(cur.lastrowid)


def _add_item(driver: Driver, cid: str, index: int = 0) -> int:
    return repo.add_evaluation_item(
        driver.conn,
        cid,
        item_index=index,
        input_sha256=hashlib.sha256(
            f"{cid}:{index}".encode("utf-8")
        ).hexdigest(),
        input_bytes=1,
        threshold_commitment="f" * 64,
        challenge_id=f"chal-{cid}",
        now=T0 + timedelta(hours=4),
    )


def test_sandbox_must_reference_same_competition_contender(driver: Driver) -> None:
    cid_a, cid_b, contender_a, contender_b = _two_competitions(driver)
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute(
            "INSERT INTO sandboxes (competition_id, contender_id, created_at) VALUES (?, ?, ?)",
            (cid_b, contender_a, TS),
        )
    # Same competition: fine.
    driver.conn.execute(
        "INSERT INTO sandboxes (competition_id, contender_id, created_at) VALUES (?, ?, ?)",
        (cid_a, contender_a, TS),
    )


def test_batch_must_reference_same_competition_contender_and_sandbox(driver: Driver) -> None:
    cid_a, cid_b, contender_a, contender_b = _two_competitions(driver)
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute(
            "INSERT INTO batches (competition_id, contender_id, batch_index, created_at)"
            " VALUES (?, ?, 0, ?)",
            (cid_b, contender_a, TS),
        )
    # A sandbox belonging to comp-02 cannot host a comp-01 batch.
    cur = driver.conn.execute(
        "INSERT INTO sandboxes (competition_id, contender_id, created_at) VALUES (?, ?, ?)",
        (cid_b, contender_b, TS),
    )
    sandbox_b = cur.lastrowid
    with pytest.raises(sqlite3.IntegrityError):
        driver.conn.execute(
            "INSERT INTO batches (competition_id, contender_id, sandbox_id, batch_index,"
            " created_at) VALUES (?, ?, ?, 0, ?)",
            (cid_a, contender_a, sandbox_b, TS),
        )


def test_performance_row_must_be_same_competition_everywhere(driver: Driver) -> None:
    cid_a, cid_b, contender_a, contender_b = _two_competitions(driver)
    item_a = _add_item(driver, cid_a)
    item_b = _add_item(driver, cid_b)

    def insert(cid: str, contender_id: int, item_id: int, batch_id: int | None = None):
        driver.conn.execute(
            """INSERT INTO performance_history
               (competition_id, contender_id, item_id, batch_id, valid, item_score,
                score_packet_digest, created_at)
               VALUES (?, ?, ?, ?, 1, 0.5, ?, ?)""",
            (cid, contender_id, item_id, batch_id, DIGEST, TS),
        )

    with pytest.raises(sqlite3.IntegrityError):
        insert(cid_b, contender_a, item_b)  # contender from another competition
    with pytest.raises(sqlite3.IntegrityError):
        insert(cid_a, contender_a, item_b)  # item from another competition
    cur = driver.conn.execute(
        "INSERT INTO batches (competition_id, contender_id, batch_index, created_at)"
        " VALUES (?, ?, 0, ?)",
        (cid_b, contender_b, TS),
    )
    with pytest.raises(sqlite3.IntegrityError):
        insert(cid_a, contender_a, item_a, batch_id=cur.lastrowid)  # foreign batch
    insert(cid_a, contender_a, item_a)  # all same-competition: fine


def test_score_packet_digest_not_null_and_gates_first_in_sql(driver: Driver) -> None:
    cid_a, _, contender_a, _ = _two_competitions(driver)
    item_a = _add_item(driver, cid_a)
    with pytest.raises(sqlite3.IntegrityError):  # digest is NOT NULL
        driver.conn.execute(
            """INSERT INTO performance_history
               (competition_id, contender_id, item_id, valid, item_score, created_at)
               VALUES (?, ?, ?, 1, 0.5, ?)""",
            (cid_a, contender_a, item_a, TS),
        )
    with pytest.raises(sqlite3.IntegrityError):  # invalid rows must score zero
        driver.conn.execute(
            """INSERT INTO performance_history
               (competition_id, contender_id, item_id, valid, item_score,
                score_packet_digest, created_at)
               VALUES (?, ?, ?, 0, 0.35, ?, ?)""",
            (cid_a, contender_a, item_a, DIGEST, TS),
        )


def test_review_supersession_cannot_cross_competitions(driver: Driver) -> None:
    cid_a, cid_b, contender_a, contender_b = _two_competitions(driver)

    def insert_review(cid: str, contender_id: int, supersedes: int | None = None):
        return driver.conn.execute(
            """INSERT INTO human_reviews
               (competition_id, contender_id, action, reviewer, reason,
                supersedes_review_id, prev_hash, integrity_hash, created_at)
               VALUES (?, ?, 'DISQUALIFY', 'owner', 'r', ?, 'p', 'i', ?)""",
            (cid, contender_id, supersedes, TS),
        ).lastrowid

    # A review may not reference a contender of another competition...
    with pytest.raises(sqlite3.IntegrityError):
        insert_review(cid_b, contender_a)
    review_a = insert_review(cid_a, contender_a)
    # ...and a comp-02 review may not supersede (silence) a comp-01 review.
    with pytest.raises(sqlite3.IntegrityError):
        insert_review(cid_b, contender_b, supersedes=review_a)
    # Same-competition supersession works, and effective_reviews resolves it.
    insert_review(cid_a, contender_a, supersedes=review_a)
    effective = repo.effective_reviews(driver.conn, cid_a)
    assert [r["supersedes_review_id"] for r in effective] == [review_a]


def test_audit_bundle_digest_lives_per_contender_item(driver: Driver) -> None:
    columns = {
        r["name"]
        for r in driver.conn.execute("PRAGMA table_info(performance_history)")
    }
    assert "audit_bundle_digest" in columns
    item_columns = {
        r["name"] for r in driver.conn.execute("PRAGMA table_info(evaluation_items)")
    }
    assert "audit_bundle_digest" not in item_columns

    cid, ids = driver.run_to_awaiting(build_manifest(), {"hk-1": 0.9}, link_audit=False)
    gaps = driver.engine.audit_linkage_gaps(driver.conn, cid)
    assert len(gaps) == 2  # one per (contender, item), none linked yet
    for row in driver.conn.execute(
        "SELECT performance_id FROM performance_history WHERE competition_id = ?", (cid,)
    ).fetchall():
        repo.set_audit_bundle_digest(driver.conn, row["performance_id"], "a" * 64)
    assert driver.engine.audit_linkage_gaps(driver.conn, cid) == []
