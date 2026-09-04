"""Validator ownership/lifecycle of outputs downloaded from remote miners."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from validator_support import FakeMinerClient, FakeScoringClient, mk_neuron


def _remote_miner(root: Path) -> FakeMinerClient:
    outdir = root / "task-files"
    outdir.mkdir(parents=True)
    return FakeMinerClient(outdir)


def _files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()] if root.exists() else []


async def test_every_terminal_outcome_discards_validator_downloads(
    make_validator, chain, scoring_client, tmp_path: Path
) -> None:
    artifact_root = tmp_path / "remote-artifacts"
    miner = _remote_miner(artifact_root)
    validator = make_validator(
        miner_client=miner,
        config={"miner_artifact_dir": str(artifact_root)},
    )
    chain.set_neurons([mk_neuron(uid) for uid in (1, 2, 3, 4, 5)])
    miner.tracks = {uid: "compression" for uid in (1, 2, 3, 4, 5)}
    miner.outputs = {1: b"duplicate", 2: b"duplicate"}
    miner.bad_digest_uids = {3}
    miner.swap_task_ids = {4: "some-other-task"}
    scoring_client.fail_hotkeys = {"hk5"}

    report = await validator.run_round()

    assert report.scored == {1: 0.8}
    assert report.zeroed[2] == "duplicate"
    assert report.zeroed[3] == "availability:output_digest_mismatch"
    assert report.zeroed[4] == "availability:task_id_mismatch"
    assert report.non_punitive_skips == {}
    assert report.scoring_failed == [5]
    assert _files(artifact_root) == []


class _BlockingScorer(FakeScoringClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def score(self, request):  # type: ignore[no-untyped-def]
        self.started.set()
        await self.release.wait()
        return await super().score(request)


async def test_round_cancellation_discards_already_downloaded_output(
    make_validator, chain, tmp_path: Path
) -> None:
    artifact_root = tmp_path / "remote-artifacts"
    miner = _remote_miner(artifact_root)
    scorer = _BlockingScorer()
    validator = make_validator(
        miner_client=miner,
        scoring_client=scorer,
        config={"miner_artifact_dir": str(artifact_root)},
    )
    chain.set_neurons([mk_neuron(1)])
    miner.tracks = {1: "compression"}

    round_task = asyncio.create_task(validator.run_round())
    await asyncio.wait_for(scorer.started.wait(), timeout=2)
    assert _files(artifact_root), "the cancellation test never reached scoring"
    round_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await round_task

    assert _files(artifact_root) == []


async def test_fake_or_legacy_path_outside_owned_root_is_never_deleted(
    make_validator, chain, tmp_path: Path
) -> None:
    owned_root = tmp_path / "owned-remote-artifacts"
    local_dir = tmp_path / "fake-local-output"
    local_dir.mkdir()
    miner = FakeMinerClient(local_dir)
    validator = make_validator(
        miner_client=miner,
        config={"miner_artifact_dir": str(owned_root)},
    )
    chain.set_neurons([mk_neuron(1)])
    miner.tracks = {1: "compression"}

    report = await validator.run_round()

    assert report.scored == {1: 0.8}
    assert list(local_dir.iterdir()), "cleanup escaped the validator-owned root"
    assert not owned_root.exists()
