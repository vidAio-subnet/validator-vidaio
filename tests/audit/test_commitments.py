import sqlite3
from pathlib import Path

import pytest

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.commitments import (
    COMMITMENT_DOMAIN,
    REWARD_POLICY_DOMAIN,
    CommitmentLedger,
    CommitmentPayload,
    CommitmentStatus,
    CompetitionCommitment,
    LedgerIntegrityError,
    PublicationRecord,
    build_competition_commitment,
    build_publication_record,
    merkle_proof,
    merkle_root,
    reward_parameter_digest,
    verify_merkle_proof,
)
from vidaio.tokenomics import TokenomicsConfig

T0 = "2026-08-20T12:00:00+00:00"
T1 = "2026-08-20T13:00:00+00:00"
T2 = "2026-08-20T14:00:00+00:00"


def _competition_commitment() -> CompetitionCommitment:
    return CompetitionCommitment(
        manifest_digest=sha256_hex(b"manifest"),
        baseline_version=0,
        baseline_artifact_digest=sha256_hex(b"baseline archive"),
        baseline_provenance_digest=sha256_hex(b"baseline provenance"),
        baseline_tree_digest=sha256_hex(b"baseline tree"),
        baseline_image_digest=sha256_hex(b"baseline image"),
        dataset_selection_seed_commitment=sha256_hex(b"seed commitment"),
        reward_param_digest=sha256_hex(b"reward params"),
    )


def test_competition_payload_deterministic_and_small() -> None:
    a = build_competition_commitment(_competition_commitment())
    b = build_competition_commitment(_competition_commitment())
    assert a == b
    assert len(a.payload) <= 128
    assert a.payload.decode().startswith(f"{COMMITMENT_DOMAIN}:competition:")
    assert sha256_hex(a.canonical_json) == a.root
    # any field change changes the root
    other = _competition_commitment().model_copy(
        update={"manifest_digest": sha256_hex(b"other manifest")}
    )
    assert build_competition_commitment(other).root != a.root


def test_reward_parameter_digest_binds_hard_coded_podium_split() -> None:
    config = TokenomicsConfig()
    expected_policy = {
        "domain": REWARD_POLICY_DOMAIN,
        "tokenomics": config.model_dump(mode="json"),
        "competition_podium_split": [0.70, 0.20, 0.10],
    }
    assert reward_parameter_digest(config) == sha256_hex(
        canonical_json_bytes(expected_policy)
    )


def test_reward_parameter_digest_binds_result_window_duration() -> None:
    production = TokenomicsConfig(result_window_hours=168.0)
    testnet_acceptance = TokenomicsConfig(result_window_hours=2.0)

    assert reward_parameter_digest(production) != reward_parameter_digest(
        testnet_acceptance
    )


def test_publication_payload() -> None:
    record = PublicationRecord(
        score_packet_merkle_root=merkle_root([sha256_hex(b"packet")]),
        weight_vector_digest=sha256_hex(b"weights"),
    )
    payload = build_publication_record(record)
    assert len(payload.payload) <= 128
    assert payload.kind == "publication"
    assert sha256_hex(payload.canonical_json) == payload.root


def test_merkle_root_deterministic_and_order_independent() -> None:
    digests = [sha256_hex(bytes([i])) for i in range(7)]  # odd count
    root = merkle_root(digests)
    assert root == merkle_root(list(reversed(digests)))
    assert root == merkle_root(digests)  # stable across calls
    assert merkle_root(digests[:1]) != root


def test_merkle_single_leaf() -> None:
    leaf = sha256_hex(b"only")
    root = merkle_root([leaf])
    assert verify_merkle_proof(leaf, merkle_proof([leaf], leaf), root)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_merkle_inclusion_proofs_all_leaves(n: int) -> None:
    digests = [sha256_hex(f"packet-{i}".encode()) for i in range(n)]
    root = merkle_root(digests)
    for d in digests:
        assert verify_merkle_proof(d, merkle_proof(digests, d), root)


def test_merkle_tampered_leaf_fails() -> None:
    digests = [sha256_hex(f"packet-{i}".encode()) for i in range(5)]
    root = merkle_root(digests)
    proof = merkle_proof(digests, digests[2])
    tampered = sha256_hex(b"injected packet with inflated score")
    assert not verify_merkle_proof(tampered, proof, root)
    # tampering the root fails too
    assert not verify_merkle_proof(digests[2], proof, merkle_root(digests[:4]))
    # a digest outside the tree has no proof
    with pytest.raises(ValueError, match="not a leaf"):
        merkle_proof(digests, tampered)


@pytest.fixture
def ledger(tmp_path: Path) -> CommitmentLedger:
    return CommitmentLedger.open(tmp_path / "ledger.db")


def test_ledger_record_and_advance(ledger: CommitmentLedger) -> None:
    payload = build_competition_commitment(_competition_commitment())
    cid = ledger.record(payload, created_at=T0)
    assert ledger.current_status(cid) is CommitmentStatus.PENDING_CHAIN
    assert ledger.get(cid)["root_digest"] == payload.root

    ledger.advance(cid, CommitmentStatus.ANCHORED, at=T1)
    ledger.advance(cid, CommitmentStatus.PUBLISHED, at=T2)
    assert ledger.current_status(cid) is CommitmentStatus.PUBLISHED
    assert ledger.history(cid) == [
        ("pending_chain", T0),
        ("anchored", T1),
        ("published", T2),
    ]


def test_ledger_status_forward_only(ledger: CommitmentLedger) -> None:
    cid = ledger.record(build_competition_commitment(_competition_commitment()), created_at=T0)
    ledger.advance(cid, CommitmentStatus.ANCHORED, at=T1)
    with pytest.raises(ValueError, match="forward"):
        ledger.advance(cid, CommitmentStatus.PENDING_CHAIN, at=T2)
    with pytest.raises(ValueError, match="forward"):
        ledger.advance(cid, CommitmentStatus.ANCHORED, at=T2)


def test_ledger_append_only_enforced_in_db(ledger: CommitmentLedger) -> None:
    cid = ledger.record(build_competition_commitment(_competition_commitment()), created_at=T0)
    conn = ledger._conn
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute(
            "UPDATE commitment_ledger SET root_digest = ? WHERE id = ?", ("f" * 64, cid)
        )
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM commitment_ledger WHERE id = ?", (cid,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE commitment_ledger_status SET status = 'published'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM commitment_ledger_status")


def test_ledger_rejects_tampered_root(ledger: CommitmentLedger) -> None:
    honest = build_competition_commitment(_competition_commitment())
    tampered = honest.model_copy(update={"root": sha256_hex(b"unrelated document")})
    with pytest.raises(LedgerIntegrityError, match="root"):
        ledger.record(tampered, created_at=T0)


def test_ledger_rejects_tampered_payload_bytes(ledger: CommitmentLedger) -> None:
    honest = build_competition_commitment(_competition_commitment())
    tampered = honest.model_copy(
        update={"payload": f"{COMMITMENT_DOMAIN}:competition:{'a' * 64}".encode("ascii")}
    )
    with pytest.raises(LedgerIntegrityError, match="domain-tagged"):
        ledger.record(tampered, created_at=T0)


def test_ledger_rejects_swapped_canonical_json(ledger: CommitmentLedger) -> None:
    honest = build_competition_commitment(_competition_commitment())
    tampered = honest.model_copy(
        update={"canonical_json": canonical_json_bytes({"different": "document"})}
    )
    with pytest.raises(LedgerIntegrityError, match="root"):
        ledger.record(tampered, created_at=T0)


def test_ledger_rejects_unknown_kind(ledger: CommitmentLedger) -> None:
    canonical = canonical_json_bytes({"anything": 1})
    root = sha256_hex(canonical)
    bogus = CommitmentPayload(
        kind="bogus",
        root=root,
        payload=f"{COMMITMENT_DOMAIN}:bogus:{root}".encode("ascii"),
        canonical_json=canonical,
    )
    with pytest.raises(LedgerIntegrityError, match="kind"):
        ledger.record(bogus, created_at=T0)


def test_direct_sql_first_status_must_be_pending_chain(ledger: CommitmentLedger) -> None:
    payload = build_competition_commitment(_competition_commitment())
    conn = ledger._conn
    # bypass record(): insert a bare commitment row with no status history
    cur = conn.execute(
        "INSERT INTO commitment_ledger (kind, root_digest, payload, canonical_json,"
        " created_at) VALUES (?, ?, ?, ?, ?)",
        (payload.kind, payload.root, payload.payload, payload.canonical_json.decode(), T0),
    )
    cid = cur.lastrowid
    for status in ("anchored", "published"):
        with pytest.raises(sqlite3.DatabaseError, match="pending_chain"):
            conn.execute(
                "INSERT INTO commitment_ledger_status (commitment_id, status, recorded_at)"
                " VALUES (?, ?, ?)",
                (cid, status, T0),
            )


def test_direct_sql_status_regression_rejected(ledger: CommitmentLedger) -> None:
    cid = ledger.record(build_competition_commitment(_competition_commitment()), created_at=T0)
    ledger.advance(cid, CommitmentStatus.ANCHORED, at=T1)
    conn = ledger._conn
    # regressions and repeats are rejected by the trigger, not just Python
    for status in ("pending_chain", "anchored"):
        with pytest.raises(sqlite3.DatabaseError, match="forward"):
            conn.execute(
                "INSERT INTO commitment_ledger_status (commitment_id, status, recorded_at)"
                " VALUES (?, ?, ?)",
                (cid, status, T2),
            )
    ledger.advance(cid, CommitmentStatus.PUBLISHED, at=T2)
    with pytest.raises(sqlite3.DatabaseError, match="forward"):
        conn.execute(
            "INSERT INTO commitment_ledger_status (commitment_id, status, recorded_at)"
            " VALUES (?, ?, ?)",
            (cid, "anchored", T2),
        )


def test_ledger_unknown_id(ledger: CommitmentLedger) -> None:
    with pytest.raises(KeyError):
        ledger.current_status(999)
    with pytest.raises(KeyError):
        ledger.get(999)
    with pytest.raises(KeyError):
        ledger.advance(999, CommitmentStatus.ANCHORED, at=T1)


def test_python_rejects_pending_to_published_skip(ledger: CommitmentLedger) -> None:
    cid = ledger.record(build_competition_commitment(_competition_commitment()), created_at=T0)
    with pytest.raises(ValueError, match="skip"):
        ledger.advance(cid, CommitmentStatus.PUBLISHED, at=T1)
    # the failed skip appended nothing; the legal path still works
    assert ledger.history(cid) == [("pending_chain", T0)]
    ledger.advance(cid, CommitmentStatus.ANCHORED, at=T1)
    ledger.advance(cid, CommitmentStatus.PUBLISHED, at=T2)


def test_direct_sql_rejects_pending_to_published_skip(ledger: CommitmentLedger) -> None:
    cid = ledger.record(build_competition_commitment(_competition_commitment()), created_at=T0)
    with pytest.raises(sqlite3.DatabaseError, match="skip"):
        ledger._conn.execute(
            "INSERT INTO commitment_ledger_status (commitment_id, status, recorded_at)"
            " VALUES (?, 'published', ?)",
            (cid, T1),
        )


def test_python_rejects_non_monotonic_timestamp(ledger: CommitmentLedger) -> None:
    cid = ledger.record(build_competition_commitment(_competition_commitment()), created_at=T1)
    with pytest.raises(ValueError, match="monoton"):
        ledger.advance(cid, CommitmentStatus.ANCHORED, at=T0)  # before creation
    ledger.advance(cid, CommitmentStatus.ANCHORED, at=T2)
    with pytest.raises(ValueError, match="monoton"):
        ledger.advance(cid, CommitmentStatus.PUBLISHED, at=T1)  # before anchored
    # equal timestamps are allowed (non-decreasing, not strictly increasing)
    ledger.advance(cid, CommitmentStatus.PUBLISHED, at=T2)


def test_direct_sql_rejects_non_monotonic_timestamp(ledger: CommitmentLedger) -> None:
    cid = ledger.record(build_competition_commitment(_competition_commitment()), created_at=T0)
    ledger.advance(cid, CommitmentStatus.ANCHORED, at=T2)
    with pytest.raises(sqlite3.DatabaseError, match="monoton"):
        ledger._conn.execute(
            "INSERT INTO commitment_ledger_status (commitment_id, status, recorded_at)"
            " VALUES (?, 'published', ?)",
            (cid, T1),  # after creation but before the anchored event
        )


def test_direct_sql_rejects_status_before_creation(ledger: CommitmentLedger) -> None:
    payload = build_competition_commitment(_competition_commitment())
    conn = ledger._conn
    cur = conn.execute(
        "INSERT INTO commitment_ledger (kind, root_digest, payload, canonical_json,"
        " created_at) VALUES (?, ?, ?, ?, ?)",
        (payload.kind, payload.root, payload.payload, payload.canonical_json.decode(), T1),
    )
    with pytest.raises(sqlite3.DatabaseError, match="precede"):
        conn.execute(
            "INSERT INTO commitment_ledger_status (commitment_id, status, recorded_at)"
            " VALUES (?, 'pending_chain', ?)",
            (cur.lastrowid, T0),  # backdated before the ledger row's created_at
        )


# ---- timestamps are instants: offsets can't disguise ----
# ---- a backdate, naive timestamps are rejected, stored form is canonical UTC


def test_python_rejects_offset_disguised_backdate(ledger: CommitmentLedger) -> None:
    """The exact review probe: created 08:00+00:00, anchored '09:00+05:00'
    (actually 04:00Z) — sorts later as a string but is an EARLIER instant."""
    cid = ledger.record(
        build_competition_commitment(_competition_commitment()),
        created_at="2026-08-20T08:00:00+00:00",
    )
    with pytest.raises(ValueError, match="monoton"):
        ledger.advance(cid, CommitmentStatus.ANCHORED, at="2026-08-20T09:00:00+05:00")
    assert ledger.history(cid) == [("pending_chain", "2026-08-20T08:00:00+00:00")]


def test_direct_sql_rejects_offset_disguised_backdate(ledger: CommitmentLedger) -> None:
    """The same probe against the trigger: julianday() normalizes '+05:00'."""
    cid = ledger.record(
        build_competition_commitment(_competition_commitment()),
        created_at="2026-08-20T08:00:00+00:00",
    )
    with pytest.raises(sqlite3.DatabaseError, match="precede|monoton"):
        ledger._conn.execute(
            "INSERT INTO commitment_ledger_status (commitment_id, status, recorded_at)"
            " VALUES (?, 'anchored', ?)",
            (cid, "2026-08-20T09:00:00+05:00"),  # = 04:00Z, before creation
        )
    assert ledger.current_status(cid) is CommitmentStatus.PENDING_CHAIN


def test_python_rejects_naive_timestamps(ledger: CommitmentLedger) -> None:
    payload = build_competition_commitment(_competition_commitment())
    with pytest.raises(ValueError, match="timezone-naive"):
        ledger.record(payload, created_at="2026-08-20T12:00:00")
    # the rejected record left nothing behind
    assert ledger._conn.execute("SELECT COUNT(*) FROM commitment_ledger").fetchone()[0] == 0
    cid = ledger.record(payload, created_at=T0)
    with pytest.raises(ValueError, match="timezone-naive"):
        ledger.advance(cid, CommitmentStatus.ANCHORED, at="2026-08-20T13:00:00")
    assert ledger.history(cid) == [("pending_chain", T0)]


def test_python_rejects_unparseable_timestamps(ledger: CommitmentLedger) -> None:
    payload = build_competition_commitment(_competition_commitment())
    with pytest.raises(ValueError, match="ISO-8601"):
        ledger.record(payload, created_at="yesterday-ish")
    cid = ledger.record(payload, created_at=T0)
    with pytest.raises(ValueError, match="ISO-8601"):
        ledger.advance(cid, CommitmentStatus.ANCHORED, at="not a timestamp")


def test_equal_instants_across_offsets_allowed(ledger: CommitmentLedger) -> None:
    """Non-decreasing means equal INSTANTS pass regardless of offset spelling,
    and the stored form is the canonical UTC string in both cases."""
    cid = ledger.record(build_competition_commitment(_competition_commitment()), created_at=T0)
    ledger.advance(cid, CommitmentStatus.ANCHORED, at="2026-08-20T17:00:00+05:00")  # == T0
    assert ledger.history(cid) == [("pending_chain", T0), ("anchored", T0)]


def test_timestamps_canonicalized_to_utc_on_write(ledger: CommitmentLedger) -> None:
    cid = ledger.record(
        build_competition_commitment(_competition_commitment()),
        created_at="2026-08-20T14:30:00+02:00",  # 12:30Z
    )
    assert ledger.get(cid)["created_at"] == "2026-08-20T12:30:00+00:00"
    ledger.advance(cid, CommitmentStatus.ANCHORED, at="2026-08-20T13:00:00Z")
    assert ledger.history(cid) == [
        ("pending_chain", "2026-08-20T12:30:00+00:00"),
        ("anchored", "2026-08-20T13:00:00+00:00"),
    ]


def test_direct_sql_rejects_unparseable_timestamps(ledger: CommitmentLedger) -> None:
    payload = build_competition_commitment(_competition_commitment())
    conn = ledger._conn
    with pytest.raises(sqlite3.DatabaseError, match="parseable ISO-8601"):
        conn.execute(
            "INSERT INTO commitment_ledger (kind, root_digest, payload, canonical_json,"
            " created_at) VALUES (?, ?, ?, ?, ?)",
            (payload.kind, payload.root, payload.payload, payload.canonical_json.decode(), "junk"),
        )
    cid = ledger.record(payload, created_at=T0)
    with pytest.raises(sqlite3.DatabaseError, match="parseable ISO-8601"):
        conn.execute(
            "INSERT INTO commitment_ledger_status (commitment_id, status, recorded_at)"
            " VALUES (?, 'anchored', ?)",
            (cid, "junk"),
        )


class _CrashBeforeStatusInsert:
    """Connection proxy simulating a crash between the two record() writes."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._real = conn

    def execute(self, sql: str, *args):
        if "INSERT INTO commitment_ledger_status" in sql:
            raise RuntimeError("simulated crash between ledger row and status row")
        return self._real.execute(sql, *args)

    def __getattr__(self, name: str):
        return getattr(self._real, name)


def test_record_is_atomic_across_row_and_status(ledger: CommitmentLedger) -> None:
    payload = build_competition_commitment(_competition_commitment())
    crashy = CommitmentLedger(_CrashBeforeStatusInsert(ledger._conn))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="simulated crash"):
        crashy.record(payload, created_at=T0)
    # the transaction rolled back: no orphan ledger row, no status row
    conn = ledger._conn
    assert conn.execute("SELECT COUNT(*) FROM commitment_ledger").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM commitment_ledger_status").fetchone()[0] == 0
    assert not conn.in_transaction
    # the connection is fully usable afterwards
    cid = ledger.record(payload, created_at=T0)
    assert ledger.history(cid) == [("pending_chain", T0)]
