"""Upscaling orchestration keeps holdouts out of the miner sandbox."""

from __future__ import annotations

import hashlib

import pytest

from vidaio.audit import ArtifactKind, ArtifactRef
from vidaio.audit.config import AuditConfig
from vidaio.audit.store import backend_key, make_public_store
from vidaio.competition import evaluation_item_commitment
from vidaio.competition import repository as repo
from vidaio.competition.interfaces import BatchItem, upscale_task_sidecar_name
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.runners.docker_runner import DockerSandboxRunner
from vidaio.competition.states import Phase

from orchestrator_support import END, M, T0, build_manifest, phase

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(reference: bytes, miner_input: bytes):
    return build_manifest(
        "comp-upscale-orch",
        track="upscaling",
        allowed_upscale_factors=[2, 4],
        evaluation_item_commitments=[
            evaluation_item_commitment(
                competition_id="comp-upscale-orch",
                item_index=0,
                reference_sha256=_digest(reference),
                input_sha256=_digest(miner_input),
                upscale_factor=2,
                target_width=TARGET_WIDTH,
                target_height=TARGET_HEIGHT,
            )
        ],
    )


def _seed(orchestrator_factory, tmp_path):
    reference = b"pristine-high-resolution-reference"
    miner_input = b"low-resolution-miner-input"
    reference_path = tmp_path / "reference.bin"
    input_path = tmp_path / "input.bin"
    reference_path.write_bytes(reference)
    input_path.write_bytes(miner_input)
    orch = orchestrator_factory()
    manifest = _manifest(reference, miner_input)
    orch.create_competition(manifest, T0)
    item_id = orch.add_evaluation_item(
        manifest.competition_id,
        input_path=input_path,
        reference_path=reference_path,
        upscale_factor=2,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        item_index=0,
        threshold_commitment="f" * 64,
        challenge_id="chal-upscale",
        now=T0,
    )
    return orch, manifest, item_id, reference, miner_input


def _reference_ref(row) -> ArtifactRef:
    return ArtifactRef(
        digest=row["reference_sha256"],
        kind=ArtifactKind.REFERENCE_ORIGINAL,
        byte_size=row["reference_bytes"],
        backend_key=backend_key(ArtifactKind.REFERENCE_ORIGINAL, row["reference_sha256"]),
    )


def test_add_upscaling_item_stages_distinct_private_reference_and_runner_input(
    orchestrator_factory, tmp_path
) -> None:
    orch, manifest, item_id, reference, miner_input = _seed(
        orchestrator_factory, tmp_path
    )
    row = orch.conn.execute(
        "SELECT * FROM evaluation_items WHERE item_id = ?", (item_id,)
    ).fetchone()
    assert row is not None
    assert row["reference_sha256"] == _digest(reference)
    assert row["input_sha256"] == _digest(miner_input)
    assert row["reference_sha256"] != row["input_sha256"]
    assert (orch.inputs_dir / row["reference_sha256"]).read_bytes() == reference
    assert (orch.inputs_dir / row["input_sha256"]).read_bytes() == miner_input

    input_ref = ArtifactRef(
        digest=row["input_sha256"],
        kind=ArtifactKind.CHALLENGE_INPUT,
        byte_size=row["input_bytes"],
        backend_key=backend_key(ArtifactKind.CHALLENGE_INPUT, row["input_sha256"]),
    )
    reference_ref = _reference_ref(row)
    assert orch.store.get(input_ref) == miner_input
    assert orch.store.get(reference_ref) == reference

    public = make_public_store(
        AuditConfig(backend="local", local_root=tmp_path / "audit")
    )
    with pytest.raises(FileNotFoundError, match="released"):
        public.get(reference_ref)

    [runner_item] = pers.batch_items_for([row], 0, 1)
    assert runner_item.input_sha256 == _digest(miner_input)
    assert runner_item.upscale_factor == 2
    assert (runner_item.target_width, runner_item.target_height) == (
        TARGET_WIDTH,
        TARGET_HEIGHT,
    )
    assert not hasattr(runner_item, "reference_sha256")


def test_docker_staging_exposes_only_digest_bound_factor_sidecars(tmp_path) -> None:
    pool = tmp_path / "pool"
    staged = tmp_path / "staged"
    pool.mkdir()
    staged.mkdir()
    runner = object.__new__(DockerSandboxRunner)
    runner._inputs_dir = pool

    items: list[BatchItem] = []
    for item_id, (payload, factor) in enumerate(
        ((b"compression", None), (b"two-x", 2), (b"four-x", 4)), start=1
    ):
        digest = _digest(payload)
        (pool / digest).write_bytes(payload)
        item = BatchItem(
            item_id=item_id,
            item_index=item_id - 1,
            input_sha256=digest,
            input_bytes=len(payload),
            upscale_factor=factor,
            target_width=None if factor is None else TARGET_WIDTH,
            target_height=None if factor is None else TARGET_HEIGHT,
        )
        runner._stage_input(item, staged)
        items.append(item)

    assert (staged / items[0].input_sha256).read_bytes() == b"compression"
    assert not (staged / upscale_task_sidecar_name(items[0].input_sha256)).exists()
    assert (
        staged / upscale_task_sidecar_name(items[1].input_sha256)
    ).read_bytes() == (
        b'{"target_height":1080,"target_width":1920,"upscale_factor":2}\n'
    )
    assert (
        staged / upscale_task_sidecar_name(items[2].input_sha256)
    ).read_bytes() == (
        b'{"target_height":1080,"target_width":1920,"upscale_factor":4}\n'
    )
    assert all("reference" not in path.name for path in staged.iterdir())


@pytest.mark.asyncio
async def test_evaluation_halts_before_exposing_a_factor_changed_after_commitment(
    orchestrator_factory, tmp_path
) -> None:
    orch, manifest, item_id, _reference, _miner_input = _seed(
        orchestrator_factory, tmp_path
    )
    orch.conn.execute(
        "UPDATE evaluation_items SET upscale_factor = 4 WHERE item_id = ?",
        (item_id,),
    )

    await orch._stage_evaluating(manifest.competition_id, T0)

    assert pers.is_halted(orch.conn, manifest.competition_id)
    halted = [
        event
        for event in repo.list_events(orch.conn, manifest.competition_id)
        if event["event_type"] == pers.EVENT_HALTED
    ]
    assert len(halted) == 1
    assert "committed item binding verification failed" in halted[0]["payload_json"]


@pytest.mark.asyncio
async def test_completion_releases_upscaling_reference_to_keyless_public_reader(
    orchestrator_factory, tmp_path
) -> None:
    orch, manifest, item_id, reference, _ = _seed(orchestrator_factory, tmp_path)
    row = orch.conn.execute(
        "SELECT * FROM evaluation_items WHERE item_id = ?", (item_id,)
    ).fetchone()
    assert row is not None
    reference_ref = _reference_ref(row)
    public = make_public_store(
        AuditConfig(backend="local", local_root=tmp_path / "audit")
    )
    with pytest.raises(FileNotFoundError):
        public.get(reference_ref)

    orch.conn.execute(
        "UPDATE competitions SET status = 'AWAITING_END_TIME', "
        "human_review_deadline = ?, updated_at = ? WHERE competition_id = ?",
        (T0.isoformat(), T0.isoformat(), manifest.competition_id),
    )
    await orch.step(END + M)

    assert phase(orch, manifest.competition_id) is Phase.COMPLETED
    assert public.get(reference_ref) == reference


@pytest.mark.asyncio
async def test_completion_fails_closed_when_reference_release_fails(
    orchestrator_factory, tmp_path, monkeypatch
) -> None:
    orch, manifest, _item_id, _reference, _ = _seed(orchestrator_factory, tmp_path)
    orch.conn.execute(
        "UPDATE competitions SET status = 'AWAITING_END_TIME', "
        "human_review_deadline = ?, updated_at = ? WHERE competition_id = ?",
        (T0.isoformat(), T0.isoformat(), manifest.competition_id),
    )

    def fail_release(_ref: ArtifactRef) -> None:
        raise OSError("object store unavailable")

    monkeypatch.setattr(orch.store, "release", fail_release)
    await orch.step(END + M)

    assert phase(orch, manifest.competition_id) is Phase.AWAITING_END_TIME
    assert repo.get_competition(orch.conn, manifest.competition_id) is not None
