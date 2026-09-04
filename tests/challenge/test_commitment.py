import random
import sqlite3

import pytest

from vidaio.challenge import (
    ChallengeCommitment,
    RevealBeforeRetireError,
    RevealedCommitment,
    build_dag,
    dag_rng_from_seed,
    record_commitment,
    record_commitment_anchor,
    retire_asset,
    reveal_commitment,
    verify_reveal,
    verify_reveal_deep,
)
from asset_factories import add_assets, mk_asset

AT = "2026-08-20T12:00:00Z"
SEED = 0xA3D8F1E64C2B905A7E118F0D3C5B2A4968D7E6F5


def _commit(conn, asset, seed=SEED, track="compression", derived=True):
    rng = dag_rng_from_seed(seed) if derived else random.Random(seed)
    dag = build_dag(track, rng)
    c = ChallengeCommitment.create(asset.id, dag, seed, "scorer-v1", track)
    record_commitment(conn, c, AT)
    return c, dag


def test_external_anchor_receipt_is_bound_append_only_and_idempotent(conn) -> None:
    asset = mk_asset(99)
    add_assets(conn, asset)
    dag = build_dag("compression", dag_rng_from_seed(SEED))
    commitment = ChallengeCommitment.create(
        asset.id,
        dag,
        SEED,
        "scorer-v1",
        "compression",
        dispatch_ordering_key=7,
    )
    record_commitment(conn, commitment, AT)

    receipt = record_commitment_anchor(
        conn,
        commit_hash=commitment.commit_hash,
        netuid=85,
        dispatch_ordering_key=7,
        block=101,
        block_hash="a" * 64,
        txid="0xanchor",
        anchored_at=AT,
    )
    assert receipt.payload().decode() == (
        f"vidaio.challenge.anchor.v1:85:7:{commitment.commit_hash}"
    )
    assert record_commitment_anchor(
        conn,
        commit_hash=commitment.commit_hash,
        netuid=85,
        dispatch_ordering_key=7,
        block=101,
        block_hash="a" * 64,
        txid="0xanchor",
        anchored_at="2026-08-20T13:00:00Z",
    ) == receipt

    with pytest.raises(sqlite3.IntegrityError, match="different chain anchor"):
        record_commitment_anchor(
            conn,
            commit_hash=commitment.commit_hash,
            netuid=85,
            dispatch_ordering_key=7,
            block=102,
            block_hash="b" * 64,
            txid="0xother",
            anchored_at=AT,
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(
            "UPDATE challenge_commitment_anchors SET anchor_block = 102"
            " WHERE commit_hash = ?",
            (commitment.commit_hash,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(
            "DELETE FROM challenge_commitment_anchors WHERE commit_hash = ?",
            (commitment.commit_hash,),
        )


def test_create_verify_round_trip(conn) -> None:
    asset = mk_asset(1)
    add_assets(conn, asset)
    c, dag = _commit(conn, asset)
    assert c.dag_digest == dag.canonical_digest()

    retire_asset(conn, asset.id, AT)
    revealed = reveal_commitment(conn, c.commit_hash, AT)
    assert revealed.commit_hash == c.commit_hash
    assert revealed.seed == c.seed
    assert verify_reveal(revealed)


def test_reveal_before_retire_raises(conn) -> None:
    asset = mk_asset(2)
    add_assets(conn, asset)
    c, _ = _commit(conn, asset)
    with pytest.raises(RevealBeforeRetireError):
        reveal_commitment(conn, c.commit_hash, AT)
    # still unrevealed in the store
    row = conn.execute(
        "SELECT revealed_at FROM challenge_commitments WHERE commit_hash = ?",
        (c.commit_hash,),
    ).fetchone()
    assert row["revealed_at"] is None


def test_tampered_reveal_fails_verification(conn) -> None:
    asset = mk_asset(3)
    add_assets(conn, asset)
    c, _ = _commit(conn, asset)
    retire_asset(conn, asset.id, AT)
    revealed = reveal_commitment(conn, c.commit_hash, AT)
    tampered = RevealedCommitment.model_validate(
        {**revealed.model_dump(), "seed": revealed.seed + 1}
    )
    assert not verify_reveal(tampered)


def test_reveal_is_idempotent_and_keeps_first_timestamp(conn) -> None:
    asset = mk_asset(4)
    add_assets(conn, asset)
    c, _ = _commit(conn, asset)
    retire_asset(conn, asset.id, AT)
    first = reveal_commitment(conn, c.commit_hash, AT)
    second = reveal_commitment(conn, c.commit_hash, "2026-08-21T00:00:00Z")
    assert second.revealed_at == first.revealed_at == AT


def test_unknown_commitment_raises(conn) -> None:
    with pytest.raises(KeyError):
        reveal_commitment(conn, "deadbeef", AT)


@pytest.mark.parametrize("track", ["compression", "upscaling"])
def test_verify_reveal_deep_accepts_seed_generated_dag(conn, track) -> None:
    asset = mk_asset(5)
    add_assets(conn, asset)
    c, _ = _commit(conn, asset, track=track)
    retire_asset(conn, asset.id, AT)
    revealed = reveal_commitment(conn, c.commit_hash, AT)
    assert verify_reveal_deep(revealed)


def test_verify_reveal_deep_rejects_hand_picked_dag(conn) -> None:
    """A commitment over a DAG that was NOT generated from the seed via the
    sanctioned derived-key path hashes fine (shallow verify passes) but must fail
    the deep check — the corruption was hand-picked, not seed-determined."""
    asset = mk_asset(6)
    add_assets(conn, asset)
    # bare-seed MT stream != sanctioned dag_rng_from_seed derivation
    c, _ = _commit(conn, asset, derived=False)
    retire_asset(conn, asset.id, AT)
    revealed = reveal_commitment(conn, c.commit_hash, AT)
    assert verify_reveal(revealed)
    assert not verify_reveal_deep(revealed)


def test_verify_reveal_deep_rejects_bad_hash() -> None:
    dag = build_dag("compression", dag_rng_from_seed(SEED))
    revealed = RevealedCommitment(
        clean_asset_id="asset_0001",
        dag_digest=dag.canonical_digest(),
        seed=SEED,
        scorer_version="scorer-v1",
        track="compression",
        dispatch_ordering_key=0,
        commit_hash="0" * 64,  # does not match the preimage
        revealed_at=AT,
    )
    assert not verify_reveal_deep(revealed)
