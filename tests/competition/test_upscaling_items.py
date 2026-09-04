"""Committed upscaling evaluation-item contract.

An upscaling contender may see only the low-resolution miner input plus the
discrete scale factor needed to produce correctly sized output. The pristine
reference and factor are committed by the manifest before enrollment; only the
factor crosses into the runner-facing ``BatchItem``.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from vidaio.audit import CompetitionItemBinding
from vidaio.competition import LifecycleEngine, evaluation_item_commitment
from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import persistence as pers

from support import T0, build_manifest


REFERENCE_DIGEST = "a" * 64
INPUT_DIGEST = "b" * 64
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def _commitment(*, factor: int = 2) -> str:
    return evaluation_item_commitment(
        competition_id="comp-upscale",
        item_index=0,
        reference_sha256=REFERENCE_DIGEST,
        input_sha256=INPUT_DIGEST,
        upscale_factor=factor,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
    )


def _upscaling_manifest(*, commitment: str | None = None):
    return build_manifest(
        "comp-upscale",
        track="upscaling",
        allowed_upscale_factors=[2, 4],
        evaluation_item_commitments=[commitment or _commitment()],
    )


def test_evaluation_item_commitment_has_an_exact_canonical_preimage() -> None:
    preimage = json.dumps(
        {
            "domain": "vidaio.competition.evaluation-item.v2",
            "competition_id": "comp-upscale",
            "item_index": 0,
            "reference_sha256": REFERENCE_DIGEST,
            "input_sha256": INPUT_DIGEST,
            "upscale_factor": 2,
            "target_width": TARGET_WIDTH,
            "target_height": TARGET_HEIGHT,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    assert _commitment() == hashlib.sha256(preimage).hexdigest()


def test_historical_v1_commitment_and_bundle_shape_remain_deserializable() -> None:
    """A geometry-v2 release must not orphan an already-anchored v1 digest."""
    v1 = evaluation_item_commitment(
        competition_id="comp-upscale",
        item_index=0,
        reference_sha256=REFERENCE_DIGEST,
        input_sha256=INPUT_DIGEST,
        upscale_factor=2,
    )
    expected = json.dumps(
        {
            "competition_id": "comp-upscale",
            "domain": "vidaio.competition.evaluation-item.v1",
            "input_sha256": INPUT_DIGEST,
            "item_index": 0,
            "reference_sha256": REFERENCE_DIGEST,
            "upscale_factor": 2,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    assert v1 == hashlib.sha256(expected).hexdigest()
    binding = CompetitionItemBinding(
        item_index=0,
        input_sha256=INPUT_DIGEST,
        reference_sha256=REFERENCE_DIGEST,
        upscale_factor=2,
        item_commitment=v1,
    )
    dumped = binding.model_dump(mode="json")
    assert "target_width" not in dumped and "target_height" not in dumped


def test_upscaling_manifest_commits_factors_and_every_evaluation_item() -> None:
    manifest = _upscaling_manifest()

    assert manifest.track == "upscaling"
    assert manifest.allowed_upscale_factors == [2, 4]
    assert manifest.evaluation_item_commitments == [_commitment()]
    # These fields are part of the manifest's canonical, pre-enrollment digest.
    canonical = json.loads(manifest.canonical_json())
    assert canonical["allowed_upscale_factors"] == [2, 4]
    assert canonical["evaluation_item_commitments"] == [_commitment()]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"allowed_upscale_factors": []}, "allowed_upscale_factors"),
        ({"evaluation_item_commitments": []}, "precommitted evaluation"),
    ],
)
def test_upscaling_manifest_rejects_missing_commitment_material(
    overrides: dict[str, object], message: str
) -> None:
    kwargs = {
        "track": "upscaling",
        "allowed_upscale_factors": [2, 4],
        "evaluation_item_commitments": [_commitment()],
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        build_manifest("comp-upscale", **kwargs)


def test_repo_persists_committed_upscaling_pair_but_runner_gets_only_low_res(conn) -> None:
    manifest = _upscaling_manifest()
    LifecycleEngine().create_competition(conn, manifest, T0)

    item_id = repo.add_evaluation_item(
        conn,
        manifest.competition_id,
        item_index=0,
        input_sha256=INPUT_DIGEST,
        input_bytes=123,
        reference_sha256=REFERENCE_DIGEST,
        reference_bytes=456,
        upscale_factor=2,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        threshold_commitment="f" * 64,
        challenge_id="chal-upscale",
        now=T0,
    )
    row = conn.execute(
        "SELECT * FROM evaluation_items WHERE item_id = ?", (item_id,)
    ).fetchone()
    assert row is not None
    assert row["input_sha256"] == INPUT_DIGEST
    assert row["reference_sha256"] == REFERENCE_DIGEST
    assert row["reference_bytes"] == 456
    assert row["upscale_factor"] == 2
    assert row["target_width"] == TARGET_WIDTH
    assert row["target_height"] == TARGET_HEIGHT
    assert row["item_commitment"] == _commitment()

    [runner_item] = pers.batch_items_for([row], batch_index=0, batch_size=1)
    assert runner_item.input_sha256 == INPUT_DIGEST
    assert runner_item.input_bytes == 123
    # The sandbox receives the committed task factor, but has no path to the
    # pristine reference; that remains trusted scorer/auditor-only material.
    assert runner_item.upscale_factor == 2
    assert (runner_item.target_width, runner_item.target_height) == (
        TARGET_WIDTH,
        TARGET_HEIGHT,
    )
    assert not hasattr(runner_item, "reference_sha256")
    assert not hasattr(runner_item, "reference_bytes")


@pytest.mark.parametrize(
    ("reference_sha256", "factor"),
    [(REFERENCE_DIGEST, 4), ("c" * 64, 2)],
)
def test_repo_rejects_upscaling_item_that_does_not_match_manifest_commitment(
    conn, reference_sha256: str, factor: int
) -> None:
    manifest = _upscaling_manifest()
    LifecycleEngine().create_competition(conn, manifest, T0)

    with pytest.raises(ValueError, match="commit"):
        repo.add_evaluation_item(
            conn,
            manifest.competition_id,
            item_index=0,
            input_sha256=INPUT_DIGEST,
            input_bytes=123,
            reference_sha256=reference_sha256,
            reference_bytes=456,
            upscale_factor=factor,
            target_width=TARGET_WIDTH,
            target_height=TARGET_HEIGHT,
            threshold_commitment="f" * 64,
            challenge_id="chal-upscale",
            now=T0,
        )


def test_repo_add_evaluation_item_remains_backward_compatible_for_compression(conn) -> None:
    manifest = build_manifest("comp-compress")
    LifecycleEngine().create_competition(conn, manifest, T0)

    item_id = repo.add_evaluation_item(
        conn,
        manifest.competition_id,
        item_index=0,
        input_sha256=INPUT_DIGEST,
        input_bytes=123,
        threshold_commitment="f" * 64,
        challenge_id="chal-compress",
        now=T0,
    )
    row = conn.execute(
        "SELECT * FROM evaluation_items WHERE item_id = ?", (item_id,)
    ).fetchone()
    assert row is not None
    assert row["reference_sha256"] == INPUT_DIGEST
    assert row["reference_bytes"] == 123
    assert row["upscale_factor"] is None
    [runner_item] = pers.batch_items_for([row], batch_index=0, batch_size=1)
    assert runner_item.upscale_factor is None
