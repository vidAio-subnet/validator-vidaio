"""Evaluation media is single-use across competitions, including cross-kind reuse."""

from __future__ import annotations

import sqlite3

import pytest

from vidaio.competition import LifecycleEngine, evaluation_item_commitment
from vidaio.competition import repository as repo
from vidaio.competition.states import Phase

from support import END, T0, build_manifest

OLD_INPUT = "1" * 64
OLD_REFERENCE = "2" * 64
FRESH_INPUT = "3" * 64
FRESH_REFERENCE = "4" * 64
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def _upscaling_manifest(
    competition_id: str, *, input_digest: str, reference_digest: str
):
    commitment = evaluation_item_commitment(
        competition_id=competition_id,
        item_index=0,
        reference_sha256=reference_digest,
        input_sha256=input_digest,
        upscale_factor=2,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
    )
    return build_manifest(
        competition_id,
        track="upscaling",
        allowed_upscale_factors=[2, 4],
        evaluation_item_commitments=[commitment],
    )


def _add_upscaling(conn, manifest, *, input_digest: str, reference_digest: str) -> int:
    return repo.add_evaluation_item(
        conn,
        manifest.competition_id,
        item_index=0,
        input_sha256=input_digest,
        input_bytes=100,
        reference_sha256=reference_digest,
        reference_bytes=400,
        upscale_factor=2,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        threshold_commitment="f" * 64,
        challenge_id=f"chal-{manifest.competition_id}",
        now=T0,
    )


@pytest.mark.parametrize(
    ("new_input", "new_reference"),
    [
        (OLD_INPUT, FRESH_REFERENCE),  # old input -> new input
        (FRESH_INPUT, OLD_INPUT),  # old input -> new reference
        (OLD_REFERENCE, FRESH_REFERENCE),  # old reference -> new input
        (FRESH_INPUT, OLD_REFERENCE),  # old reference -> new reference
    ],
)
def test_repository_rejects_every_cross_competition_cross_kind_digest_reuse(
    conn, new_input: str, new_reference: str
) -> None:
    engine = LifecycleEngine()
    old = _upscaling_manifest(
        "comp-reuse-old",
        input_digest=OLD_INPUT,
        reference_digest=OLD_REFERENCE,
    )
    new = _upscaling_manifest(
        "comp-reuse-new",
        input_digest=new_input,
        reference_digest=new_reference,
    )
    engine.create_competition(conn, old, T0)
    engine.create_competition(conn, new, T0)
    _add_upscaling(
        conn, old, input_digest=OLD_INPUT, reference_digest=OLD_REFERENCE
    )

    with pytest.raises(repo.EvaluationItemReuseError, match="single-use"):
        _add_upscaling(
            conn,
            new,
            input_digest=new_input,
            reference_digest=new_reference,
        )


def _raw_insert(
    conn,
    competition_id: str,
    *,
    input_digest: str,
    reference_digest: str,
    item_commitment: str,
) -> None:
    conn.execute(
        """INSERT INTO evaluation_items
           (competition_id, item_index, input_sha256, input_bytes, length_seconds,
            threshold_commitment, sealed_vmaf_threshold, challenge_id,
            scoring_item_id, created_at, reference_sha256, reference_bytes,
            upscale_factor, item_commitment)
           VALUES (?, 0, ?, 100, 1.0, ?, NULL, ?, ?, ?, ?, 400, 2, ?)""",
        (
            competition_id,
            input_digest,
            "f" * 64,
            f"chal-{competition_id}",
            input_digest,
            repo.iso(T0),
            reference_digest,
            item_commitment,
        ),
    )


@pytest.mark.parametrize(
    ("new_input", "new_reference"),
    [
        (OLD_REFERENCE, FRESH_REFERENCE),
        (FRESH_INPUT, OLD_INPUT),
    ],
)
def test_direct_sql_insert_trigger_rejects_cross_kind_reuse(
    conn, new_input: str, new_reference: str
) -> None:
    engine = LifecycleEngine()
    old = _upscaling_manifest(
        "comp-trigger-old",
        input_digest=OLD_INPUT,
        reference_digest=OLD_REFERENCE,
    )
    new = _upscaling_manifest(
        "comp-trigger-new",
        input_digest=new_input,
        reference_digest=new_reference,
    )
    engine.create_competition(conn, old, T0)
    engine.create_competition(conn, new, T0)
    _add_upscaling(
        conn, old, input_digest=OLD_INPUT, reference_digest=OLD_REFERENCE
    )
    commitment = new.evaluation_item_commitments[0]

    with pytest.raises(sqlite3.IntegrityError, match="single-use"):
        _raw_insert(
            conn,
            new.competition_id,
            input_digest=new_input,
            reference_digest=new_reference,
            item_commitment=commitment,
        )


@pytest.mark.parametrize(
    ("column", "digest"),
    [("input_sha256", OLD_REFERENCE), ("reference_sha256", OLD_INPUT)],
)
def test_direct_sql_update_trigger_rejects_cross_kind_reuse(
    conn, column: str, digest: str
) -> None:
    engine = LifecycleEngine()
    old = _upscaling_manifest(
        "comp-update-old",
        input_digest=OLD_INPUT,
        reference_digest=OLD_REFERENCE,
    )
    new = _upscaling_manifest(
        "comp-update-new",
        input_digest=FRESH_INPUT,
        reference_digest=FRESH_REFERENCE,
    )
    engine.create_competition(conn, old, T0)
    engine.create_competition(conn, new, T0)
    _add_upscaling(
        conn, old, input_digest=OLD_INPUT, reference_digest=OLD_REFERENCE
    )
    new_item_id = _add_upscaling(
        conn,
        new,
        input_digest=FRESH_INPUT,
        reference_digest=FRESH_REFERENCE,
    )

    assert column in {"input_sha256", "reference_sha256"}
    with pytest.raises(sqlite3.IntegrityError, match="single-use"):
        conn.execute(
            f"UPDATE evaluation_items SET {column} = ? WHERE item_id = ?",
            (digest, new_item_id),
        )


def test_completion_revalidates_and_blocks_migrated_dirty_reuse(conn) -> None:
    """Even a DB imported without its triggers cannot turn reused media economic."""
    engine = LifecycleEngine()
    old = build_manifest("comp-dirty-old")
    dirty = build_manifest("comp-dirty-new")
    engine.create_competition(conn, old, T0)
    engine.create_competition(conn, dirty, T0)
    repo.add_evaluation_item(
        conn,
        old.competition_id,
        item_index=0,
        input_sha256=OLD_INPUT,
        input_bytes=100,
        threshold_commitment="f" * 64,
        challenge_id="chal-dirty-old",
        now=T0,
    )

    # Model a legacy/corrupt import that predated the migration backstop.
    conn.execute("DROP TRIGGER evaluation_media_single_competition_insert")
    conn.execute("DROP TRIGGER evaluation_media_single_competition_update")
    _raw_insert(
        conn,
        dirty.competition_id,
        input_digest=OLD_INPUT,
        reference_digest=OLD_INPUT,
        item_commitment=None,
    )
    with pytest.raises(repo.EvaluationItemReuseError, match="single-use"):
        repo.validate_evaluation_item_bindings(conn, dirty.competition_id)

    conn.execute(
        "UPDATE competitions SET status = 'AWAITING_END_TIME', "
        "human_review_deadline = ?, updated_at = ? WHERE competition_id = ?",
        (repo.iso(T0), repo.iso(T0), dirty.competition_id),
    )
    engine.tick(conn, END)
    assert repo.get_competition(conn, dirty.competition_id).status is Phase.AWAITING_END_TIME
