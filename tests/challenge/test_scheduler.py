import hashlib
import json
import random
import sqlite3
import uuid

import pytest

from vidaio.challenge import (
    ChallengeConfig,
    ChallengeIntegrityError,
    NearDuplicateError,
    NoFreshAssetError,
    RevealBeforeResolutionError,
    StaticFingerprintIndex,
    WeakSeedError,
    build_dag,
    checkout_asset,
    confirm_ingest_step,
    dag_rng_from_seed,
    get_asset,
    make_challenge,
    provenance_log,
    record_challenge,
    register_asset,
    release_asset,
    resolve_challenge,
    reveal_commitment,
    seed_to_bytes,
    verify_reveal,
    verify_reveal_deep,
)
from asset_factories import add_assets, mk_asset

AT = "2026-08-20T12:00:00Z"
# >= 128 bits (WeakSeedError floor). Production seeds come from secrets.randbits(256);
# tests only need a fixed value with real length so substring leak probes are meaningful.
SEED = 0xA3D8F1E64C2B905A7E118F0D3C5B2A4968D7E6F5


def _different_dag(track: str, original):
    """Find a deterministic distinct draw without assuming a large launch pool.

    DAG v7 deliberately has only three compression CRFs, so adjacent private seeds may
    honestly collide on the canonical DAG while still yielding distinct challenge ids.
    """

    for offset in range(1, 256):
        candidate = build_dag(track, dag_rng_from_seed(SEED + offset))
        if candidate.canonical_digest() != original.canonical_digest():
            return candidate
    pytest.fail(f"no distinct {track} DAG found across 255 deterministic seed draws")


def test_make_challenge_deterministic() -> None:
    asset = mk_asset(1)
    a = make_challenge("compression", asset, SEED, "scorer-v1")
    b = make_challenge("compression", asset, SEED, "scorer-v1")
    assert a == b
    assert a.challenge_id == b.challenge_id
    assert a.dag.canonical_digest() == b.dag.canonical_digest()
    assert a.commitment.commit_hash == b.commitment.commit_hash
    c = make_challenge("compression", asset, SEED + 1, "scorer-v1")
    assert c.challenge_id != a.challenge_id
    # Adjacent seeds need not select different members of the intentionally tiny v7
    # launch pool. The private seed still controls the DAG, and the pool has multiple
    # reachable canonical draws.
    assert _different_dag("compression", a.dag).canonical_digest() != (
        a.dag.canonical_digest()
    )


def test_weak_seeds_rejected() -> None:
    asset = mk_asset(1)
    for weak in (5, 20260820, 987654321987654321, (1 << 127) - 1):
        with pytest.raises(WeakSeedError):
            make_challenge("compression", asset, weak, "scorer-v1")
    # exactly at the floor passes
    make_challenge("compression", asset, 1 << 127, "scorer-v1", min_seed_bits=128)
    with pytest.raises(WeakSeedError):
        make_challenge("compression", asset, SEED, "scorer-v1", min_seed_bits=256)


def test_challenge_id_is_sha256_derived_never_mt_output() -> None:
    """The public challenge_id must come from sha256, NEVER from the Mersenne
    Twister stream that also feeds the private DAG — otherwise a brute-forced
    UUID reconstructs every private parameter."""
    asset = mk_asset(1)
    ch = make_challenge("compression", asset, SEED, "scorer-v1")

    expected = uuid.UUID(
        bytes=hashlib.sha256(
            b"challenge-id" + seed_to_bytes(SEED) + asset.id.encode()
        ).digest()[:16],
        version=4,
    )
    assert ch.challenge_id == str(expected)
    # not the first 128-bit draw of an MT seeded with the bare seed
    leaked = uuid.UUID(int=random.Random(SEED).getrandbits(128), version=4)
    assert ch.challenge_id != str(leaked)
    # and the DAG is not built from the bare-seed MT stream either
    assert ch.dag != build_dag("compression", random.Random(SEED))


def test_dispatch_payload_contains_no_private_material() -> None:
    asset = mk_asset(1)
    ch = make_challenge("upscaling", asset, SEED, "scorer-v1")
    text = ch.dispatch.model_dump_json()

    assert set(json.loads(text)) == {"challenge_id", "task_type", "input_ref"}
    assert str(SEED) not in text
    assert asset.id not in text
    assert asset.content_digest not in text
    assert asset.source_url not in text
    assert ch.commitment.dag_digest not in text
    # no DAG operator params leak either
    for op in ch.dag.ops:
        for value in op.model_dump(mode="json").values():
            if isinstance(value, float):
                assert str(value) not in text
    # what the miner DOES see: the public task type and an input ref derived
    # only from the challenge id
    payload = json.loads(text)
    assert payload["task_type"] == "upscaling"
    assert payload["input_ref"] == f"challenges/{ch.challenge_id}/input.mp4"


def test_register_asset_full_flow(conn) -> None:
    cfg = ChallengeConfig(holdout_fraction=0.0)  # keep the asset issuable
    result = register_asset(
        conn,
        cfg,
        source_url="https://example.com/street.mp4",
        license_basis="owner-captured",
        creator="alice",
        source="shoot-1",
        subject="street",
        scene="day",
        content_digest="ab" * 32,
        perceptual_fingerprint="fp_new",
        resolution_tag="4k",
        motion_tag="high",
        content_type_tag="sports",
        ingested_at=AT,
        duplicate_index=StaticFingerprintIndex({"fp_public_benchmark"}),
    )
    asset = result.asset
    # registration records PLANS only: the asset is 'ingesting', not yet issuable
    assert asset.status == "ingesting"
    assert asset.split == "challenge"
    # registration only PLANNED the transcode; nothing has been stripped yet
    assert asset.metadata_stripped is False

    # command plans: fetch, pristine transcode with metadata strip, segmentation
    assert result.fetch_plan[0] == "curl"
    assert "-map_metadata" in result.transcode_plan
    assert result.transcode_plan[result.transcode_plan.index("-map_metadata") + 1] == "-1"
    assert "segment" in result.segment_plan
    assert str(cfg.max_clip_seconds) in result.segment_plan

    events = [e["event"] for e in provenance_log(conn, asset.id)]
    assert events == [
        "ingested", "fetch_planned", "transcode_planned", "segment_planned",
        "fingerprinted", "split_assigned",
    ]

    # nothing has run yet -> the asset must not be checkoutable
    with pytest.raises(NoFreshAssetError):
        checkout_asset(conn, random.Random(5), AT)

    # the executor ran the transcode plan -> confirm flips the flag + logs completion
    confirm_ingest_step(conn, asset.id, "transcode", AT)
    assert get_asset(conn, asset.id).metadata_stripped is True
    events = [e["event"] for e in provenance_log(conn, asset.id)]
    assert events[-2:] == ["transcode_completed", "metadata_stripped"]
    # ...but transcode alone does not make the asset issuable
    assert get_asset(conn, asset.id).status == "ingesting"

    # remaining confirmations land -> the LAST one flips ingesting -> fresh
    confirm_ingest_step(conn, asset.id, "fetch", AT)
    assert get_asset(conn, asset.id).status == "ingesting"
    confirm_ingest_step(conn, asset.id, "segment", AT)
    assert get_asset(conn, asset.id).status == "fresh"
    events = [e["event"] for e in provenance_log(conn, asset.id)]
    assert events[-2:] == ["segment_completed", "ingest_confirmed"]

    # end-to-end: checkout -> challenge -> record -> retire -> resolve -> reveal
    issued = checkout_asset(conn, random.Random(5), AT)
    ch = make_challenge("compression", issued, SEED, "scorer-v1")
    record_challenge(conn, ch, AT)
    release_asset(conn, issued.id, cfg.retire_after_uses, AT)
    # asset retired, but the challenge is still dispatched: reveal must refuse
    with pytest.raises(RevealBeforeResolutionError):
        reveal_commitment(conn, ch.commitment.commit_hash, AT)
    resolve_challenge(conn, ch.challenge_id, AT)
    revealed = reveal_commitment(conn, ch.commitment.commit_hash, AT)
    assert verify_reveal(revealed)
    assert verify_reveal_deep(revealed)
    assert revealed.dag_digest == ch.dag.canonical_digest()


def test_confirm_ingest_step_rejects_unknown(conn) -> None:
    add_assets(conn, mk_asset(1))
    with pytest.raises(ValueError):
        confirm_ingest_step(conn, "asset_0001", "upload", AT)
    with pytest.raises(KeyError):
        confirm_ingest_step(conn, "asset_nope", "transcode", AT)


def _register_simple(conn, **overrides):
    """Register one challenge-split asset via the real ingest contract."""
    kwargs = dict(
        source_url="https://example.com/clip.mp4",
        license_basis="cc0",
        creator="alice",
        source="shoot-1",
        content_digest="ab" * 32,
        perceptual_fingerprint="fp_x",
        resolution_tag="1080p",
        motion_tag="medium",
        content_type_tag="sports",
        ingested_at=AT,
    )
    kwargs.update(overrides)
    return register_asset(conn, ChallengeConfig(holdout_fraction=0.0), **kwargs)


def test_asset_not_checkoutable_until_all_ingest_steps_confirmed(conn) -> None:
    """ingesting -> fresh happens only when the LAST of fetch/transcode/segment is
    confirmed; every partially-confirmed state stays 'ingesting' and is never
    issued by checkout_asset."""
    asset_id = _register_simple(conn).asset.id
    for step in ("segment", "fetch"):  # any order; transcode last here
        confirm_ingest_step(conn, asset_id, step, AT)
        assert get_asset(conn, asset_id).status == "ingesting"
        with pytest.raises(NoFreshAssetError):
            checkout_asset(conn, random.Random(0), AT)
    confirm_ingest_step(conn, asset_id, "transcode", AT)
    a = get_asset(conn, asset_id)
    assert a.status == "fresh"
    assert a.metadata_stripped is True  # strip flag still flips on transcode confirm
    assert checkout_asset(conn, random.Random(0), AT).id == asset_id


def test_confirm_ingest_step_rejects_double_confirmation(conn) -> None:
    asset_id = _register_simple(conn).asset.id
    confirm_ingest_step(conn, asset_id, "fetch", AT)
    with pytest.raises(ValueError, match="already confirmed"):
        confirm_ingest_step(conn, asset_id, "fetch", AT)
    # the duplicate neither flipped status nor tampered extra provenance
    assert get_asset(conn, asset_id).status == "ingesting"
    events = [e["event"] for e in provenance_log(conn, asset_id)]
    assert events.count("fetch_completed") == 1


class _CrashingConn:
    """Connection proxy: raises on the first execute whose SQL contains `needle`
    — crash injection at an exact statement inside confirm_ingest_step. Every
    other statement (including ROLLBACK) passes through to the real connection."""

    def __init__(self, real: sqlite3.Connection, needle: str) -> None:
        self._real = real
        self._needle = needle

    def execute(self, sql: str, *args):
        if self._needle in sql:
            raise RuntimeError(f"injected crash at {self._needle!r}")
        return self._real.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _BlindReadConn:
    """Connection proxy whose duplicate-check SELECT returns no rows: simulates a
    concurrent confirm committed AFTER our read — only the UNIQUE index on
    provenance confirmation events can catch the duplicate then."""

    def __init__(self, real: sqlite3.Connection) -> None:
        self._real = real

    def execute(self, sql: str, *args):
        if "SELECT DISTINCT event" in sql:
            return self._real.execute("SELECT event FROM provenance_log WHERE 1 = 0")
        return self._real.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_confirm_crash_between_append_and_flip_rolls_back_fully(conn) -> None:
    """A crash after appending the final *_completed event but before the
    ingesting->fresh flip must roll back the WHOLE confirmation. The failure mode
    guarded against: the event commits without the flip, so the retry says
    'already confirmed' while the asset is stranded at 'ingesting' forever."""
    asset_id = _register_simple(conn).asset.id
    confirm_ingest_step(conn, asset_id, "fetch", AT)
    confirm_ingest_step(conn, asset_id, "transcode", AT)

    crashing = _CrashingConn(conn, "SET status = 'fresh'")
    with pytest.raises(RuntimeError, match="injected crash"):
        confirm_ingest_step(crashing, asset_id, "segment", AT)

    assert not conn.in_transaction  # rolled back, not left open
    # NOTHING of the failed confirm persisted: the append rolled back with the flip
    events = [e["event"] for e in provenance_log(conn, asset_id)]
    assert "segment_completed" not in events
    assert "ingest_confirmed" not in events
    assert get_asset(conn, asset_id).status == "ingesting"
    with pytest.raises(NoFreshAssetError):
        checkout_asset(conn, random.Random(0), AT)

    # retry after the rollback succeeds cleanly — no 'already confirmed' stranding
    confirm_ingest_step(conn, asset_id, "segment", AT)
    assert get_asset(conn, asset_id).status == "fresh"
    events = [e["event"] for e in provenance_log(conn, asset_id)]
    assert events[-2:] == ["segment_completed", "ingest_confirmed"]
    assert checkout_asset(conn, random.Random(0), AT).id == asset_id


def test_confirm_crash_on_confirmed_event_rolls_back_flip_too(conn, monkeypatch) -> None:
    """Atomicity from the other side: a crash while appending 'ingest_confirmed'
    (i.e. AFTER the status flip) must undo the flip and the *_completed append."""
    import vidaio.challenge.scheduler as scheduler_mod

    asset_id = _register_simple(conn).asset.id
    confirm_ingest_step(conn, asset_id, "fetch", AT)
    confirm_ingest_step(conn, asset_id, "transcode", AT)

    real_append = scheduler_mod.append_provenance

    def crashing_append(c, aid, event, detail, at):
        if event == "ingest_confirmed":
            raise RuntimeError("injected crash on ingest_confirmed")
        return real_append(c, aid, event, detail, at)

    monkeypatch.setattr(scheduler_mod, "append_provenance", crashing_append)
    with pytest.raises(RuntimeError, match="injected crash"):
        confirm_ingest_step(conn, asset_id, "segment", AT)
    monkeypatch.undo()

    assert not conn.in_transaction
    assert get_asset(conn, asset_id).status == "ingesting"  # flip rolled back too
    events = [e["event"] for e in provenance_log(conn, asset_id)]
    assert "segment_completed" not in events

    confirm_ingest_step(conn, asset_id, "segment", AT)
    assert get_asset(conn, asset_id).status == "fresh"


def test_duplicate_confirmation_rejected_at_sql_layer(conn) -> None:
    """Confirmation events are UNIQUE(asset_id, event) via a partial index: a
    duplicate insert fails inside SQLite even when it bypasses Python entirely."""
    asset_id = _register_simple(conn).asset.id
    confirm_ingest_step(conn, asset_id, "fetch", AT)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO provenance_log (asset_id, event, detail, recorded_at)"
            " VALUES (?, 'fetch_completed', '{}', ?)",
            (asset_id, AT),
        )
    # the index is scoped: ordinary provenance events still repeat freely
    for _ in range(2):
        conn.execute(
            "INSERT INTO provenance_log (asset_id, event, detail, recorded_at)"
            " VALUES (?, 'checked_out', '{}', ?)",
            (asset_id, AT),
        )


def test_concurrent_duplicate_confirm_caught_by_sql_not_python(conn) -> None:
    """Race simulation: the pre-insert duplicate read sees nothing (as if another
    connection committed the same confirm after our SELECT); the UNIQUE index
    still rejects the append and the whole confirm rolls back cleanly."""
    asset_id = _register_simple(conn).asset.id
    confirm_ingest_step(conn, asset_id, "fetch", AT)
    with pytest.raises(ValueError, match="already confirmed"):
        confirm_ingest_step(_BlindReadConn(conn), asset_id, "fetch", AT)
    assert not conn.in_transaction
    events = [e["event"] for e in provenance_log(conn, asset_id)]
    assert events.count("fetch_completed") == 1
    assert get_asset(conn, asset_id).status == "ingesting"


def test_register_asset_rejects_near_duplicates(conn) -> None:
    cfg = ChallengeConfig()
    with pytest.raises(NearDuplicateError):
        register_asset(
            conn,
            cfg,
            source_url="https://example.com/vid.mp4",
            license_basis="cc0",
            creator="bob",
            source="stock",
            content_digest="cd" * 32,
            perceptual_fingerprint="fp_public_benchmark",
            resolution_tag="1080p",
            motion_tag="low",
            content_type_tag="nature",
            ingested_at=AT,
            duplicate_index=StaticFingerprintIndex({"fp_public_benchmark"}),
        )
    # nothing entered the pool
    assert conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0] == 0


def test_register_asset_youtube_uses_ytdlp(conn) -> None:
    result = register_asset(
        conn,
        ChallengeConfig(),
        source_url="https://www.youtube.com/watch?v=abc123",
        license_basis="creator-permission",
        creator="carol",
        source="yt-channel-1",
        content_digest="ef" * 32,
        perceptual_fingerprint="fp_yt",
        resolution_tag="1080p",
        motion_tag="medium",
        content_type_tag="vlog",
        ingested_at=AT,
    )
    assert result.fetch_plan[0] == "yt-dlp"


def test_record_challenge_requires_commitment_first(conn) -> None:
    # record_challenge inserts the commitment before the challenge row; the FK and
    # binding trigger make a commitment-less challenge row impossible.
    add_assets(conn, mk_asset(1))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO challenges"
            " (challenge_id, track, asset_id, commit_hash, dag_digest, dag_json, created_at)"
            " VALUES ('c1', 'compression', 'asset_0001', 'missing', 'd', '{}', ?)",
            (AT,),
        )


def test_challenge_must_match_its_commitment(conn) -> None:
    """The binding trigger rejects a challenge row whose asset_id or dag_digest
    differs from the referenced commitment's."""
    asset, other = mk_asset(1), mk_asset(2)
    add_assets(conn, asset, other)
    ch = make_challenge("compression", asset, SEED, "scorer-v1")
    record_challenge(conn, ch, AT)

    for bad_asset, bad_digest in [
        (other.id, ch.commitment.dag_digest),  # wrong asset
        (asset.id, "0" * 64),  # wrong dag digest
    ]:
        with pytest.raises(sqlite3.IntegrityError, match="does not match its commitment"):
            conn.execute(
                "INSERT INTO challenges"
                " (challenge_id, track, asset_id, commit_hash, dag_digest, dag_json, created_at)"
                " VALUES ('tampered', 'compression', ?, ?, ?, '{}', ?)",
                (bad_asset, ch.commitment.commit_hash, bad_digest, AT),
            )


def test_record_challenge_rejects_inconsistent_challenge_object(conn) -> None:
    """An internally inconsistent Challenge (DAG that does not hash to the
    committed dag_digest, or wrong asset binding) must never persist — not even
    its commitment row."""
    asset, other = mk_asset(1), mk_asset(2)
    add_assets(conn, asset, other)
    ch = make_challenge("compression", asset, SEED, "scorer-v1")

    # DAG swapped after commitment: recomputed digest no longer matches
    foreign_dag = _different_dag("compression", ch.dag)
    with pytest.raises(ChallengeIntegrityError, match="dag_digest"):
        record_challenge(conn, ch.model_copy(update={"dag": foreign_dag}), AT)

    # asset identity swapped after commitment
    with pytest.raises(ChallengeIntegrityError, match="clean_asset_id"):
        record_challenge(conn, ch.model_copy(update={"asset_id": other.id}), AT)

    # commit_hash that does not hash its own preimage
    tampered_commitment = ch.commitment.model_copy(update={"commit_hash": "f" * 64})
    with pytest.raises(ChallengeIntegrityError, match="preimage"):
        record_challenge(conn, ch.model_copy(update={"commitment": tampered_commitment}), AT)

    # nothing persisted from any rejected attempt
    assert conn.execute("SELECT COUNT(*) FROM challenge_commitments").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM challenges").fetchone()[0] == 0

    # the untampered object still records fine afterwards
    record_challenge(conn, ch, AT)


def test_challenge_identity_columns_are_immutable(conn) -> None:
    """After insert, every identity column is frozen at the DB layer; only the
    resolution lifecycle (status, resolved_at) may change."""
    asset, other = mk_asset(1), mk_asset(2)
    add_assets(conn, asset, other)
    ch = make_challenge("compression", asset, SEED, "scorer-v1")
    record_challenge(conn, ch, AT)
    original = dict(
        conn.execute(
            "SELECT * FROM challenges WHERE challenge_id = ?", (ch.challenge_id,)
        ).fetchone()
    )

    tampered_fields = {
        "challenge_id": "tampered-id",
        "track": "upscaling",
        "asset_id": other.id,
        "commit_hash": "e" * 64,
        "dag_digest": "0" * 64,
        "dag_json": "{}",
        "created_at": "1999-01-01T00:00:00Z",
    }
    for column, value in tampered_fields.items():
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                f"UPDATE challenges SET {column} = ? WHERE challenge_id = ?",
                (value, ch.challenge_id),
            )
    unchanged = dict(
        conn.execute(
            "SELECT * FROM challenges WHERE challenge_id = ?", (ch.challenge_id,)
        ).fetchone()
    )
    assert unchanged == original

    # the resolution lifecycle stays updatable through the same trigger
    conn.execute(
        "UPDATE challenges SET status = 'resolved', resolved_at = ? WHERE challenge_id = ?",
        (AT, ch.challenge_id),
    )
    row = conn.execute(
        "SELECT status, resolved_at FROM challenges WHERE challenge_id = ?",
        (ch.challenge_id,),
    ).fetchone()
    assert (row["status"], row["resolved_at"]) == ("resolved", AT)


def test_commitment_cannot_be_reused_across_challenges(conn) -> None:
    asset = mk_asset(1)
    add_assets(conn, asset)
    ch = make_challenge("compression", asset, SEED, "scorer-v1")
    record_challenge(conn, ch, AT)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO challenges"
            " (challenge_id, track, asset_id, commit_hash, dag_digest, dag_json, created_at)"
            " VALUES ('second-dispatch', 'compression', ?, ?, ?, '{}', ?)",
            (asset.id, ch.commitment.commit_hash, ch.commitment.dag_digest, AT),
        )


def test_resolve_challenge_transitions(conn) -> None:
    asset = mk_asset(1)
    add_assets(conn, asset)
    ch = make_challenge("compression", asset, SEED, "scorer-v1")
    record_challenge(conn, ch, AT)

    with pytest.raises(KeyError):
        resolve_challenge(conn, "nope", AT)
    with pytest.raises(ValueError):
        resolve_challenge(conn, ch.challenge_id, AT, outcome="abandoned")

    resolve_challenge(conn, ch.challenge_id, AT)
    row = conn.execute(
        "SELECT status, resolved_at FROM challenges WHERE challenge_id = ?",
        (ch.challenge_id,),
    ).fetchone()
    assert (row["status"], row["resolved_at"]) == ("resolved", AT)
    # terminal states are final
    with pytest.raises(ValueError):
        resolve_challenge(conn, ch.challenge_id, AT, outcome="expired")


def test_expire_challenge(conn) -> None:
    asset = mk_asset(1)
    add_assets(conn, asset)
    ch = make_challenge("compression", asset, SEED, "scorer-v1")
    record_challenge(conn, ch, AT)
    resolve_challenge(conn, ch.challenge_id, AT, outcome="expired")
    row = conn.execute(
        "SELECT status FROM challenges WHERE challenge_id = ?", (ch.challenge_id,)
    ).fetchone()
    assert row["status"] == "expired"
