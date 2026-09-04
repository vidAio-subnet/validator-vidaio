"""Verify-then-snapshot: the TOCTOU probe and the non-regular-input refusals.

The property under test is the one the whole audit trail rests on: the bytes named
by ``content_digest`` are the bytes that were measured. A miner that can rewrite
its output file after the digest check must not be able to make those two differ —
so these tests deliberately swap/truncate the source *between* verification and
measurement and assert the score reflects the snapshot.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import httpx
import pytest

from tests.scoring_worker.conftest import (
    FFMPEG,
    FFPROBE,
    ClipPair,
    RoleKeyedBackend,
    requires_media_tools,
    score_request_body,
    sha256_file,
)
from vidaio.scoring import DeterministicFakeBackend, ItemScore, MediaInfo
from vidaio.scoring.backends_real import (
    VMAF_SCRATCH_PREFIX,
    VMAF_VERSION_SCRATCH_PREFIX,
    CanonicalizeExecutor,
    FfmpegVmafBackend,
    FfprobeBackend,
    SECONDARY_VMAF_MODEL,
    detect_tool_versions,
)
from vidaio.scoring_worker import (
    ScoringBackends,
    ScoringWorker,
    ScoringWorkerConfig,
    create_app,
    measure_scratch_entries,
    sweep_work_dir,
)
from vidaio.scoring_worker.inputs import (
    HASH_CHUNK,
    HEALTH_PROBE_PREFIX,
    WORK_PREFIX,
    ByteLimits,
    ScoreRejected,
    ScratchBudget,
    SnapshotCancelled,
    snapshot_input,
)

_GARBAGE = b"swapped-after-verification" * 64


def _media(byte_size: int, *, width: int = 320, height: int = 240) -> MediaInfo:
    return MediaInfo(
        codec="h264",
        width=width,
        height=height,
        fps=30.0,
        frame_count=60,
        duration=2.0,
        byte_size=byte_size,
    )


def _write(path: Path, data: bytes) -> tuple[str, str]:
    path.write_bytes(data)
    return str(path), hashlib.sha256(data).hexdigest()


class _FaultInjectingProbe:
    """Probe backend that rewrites the caller-named sources on its first call.

    The first probe happens strictly after verification and snapshotting, so this
    is the exact window a miner could use: the request's digests have been
    accepted and ffmpeg has not measured yet.
    """

    def __init__(self, inner, targets: Sequence[Path]) -> None:
        self._inner = inner
        self._targets = list(targets)
        self.fault_injected = False

    def probe(self, path: str) -> MediaInfo:
        if not self.fault_injected:
            self.fault_injected = True
            for target in self._targets:
                target.write_bytes(_GARBAGE)
        return self._inner.probe(path)


class _FaultInjectingCanonicalizer(CanonicalizeExecutor):
    """Real canonicalizer that swaps the sources before the FIRST plan runs."""

    def __init__(self, ffmpeg: str, targets: Sequence[Path], **kwargs) -> None:
        super().__init__(ffmpeg, **kwargs)
        self._targets = list(targets)
        self.fault_injected = False

    def run(self, plan, timeout=None, **kwargs) -> None:
        if not self.fault_injected:
            self.fault_injected = True
            for target in self._targets:
                target.write_bytes(_GARBAGE)
        super().run(plan, timeout=timeout, **kwargs)


def _fake_world(tmp_path: Path):
    reference, reference_digest = _write(tmp_path / "ref.bin", b"R" * 10_000)
    output, output_digest = _write(tmp_path / "out.bin", b"O" * 5_000)
    fake = RoleKeyedBackend(
        vmaf={("reference", "output"): 93.0},
        media={
            "reference": _media(10_000),
            "output": _media(5_000),
            "miner_input": _media(10_000),
        },
    )
    config = ScoringWorkerConfig(
        backend="fake", work_dir=tmp_path / "work", request_timeout=10.0
    )
    return config, fake, (reference, reference_digest), (output, output_digest)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker"
    )


# --- the TOCTOU probe ------------------------------------------------------------------


async def test_source_swap_after_verification_does_not_reach_the_score(
    tmp_path: Path,
) -> None:
    """Swap both sources between verify and measure: the snapshot is scored."""
    config, fake, (reference, ref_digest), (output, out_digest) = _fake_world(tmp_path)
    injector = _FaultInjectingProbe(fake, [Path(reference), Path(output)])
    backends = ScoringBackends(
        probe=injector,
        vmaf_primary=fake,
        vmaf_secondary=fake,
        pieapp=fake.pieapp,
        perceptual=fake,
        canonicalizer=None,
        versions=fake.versions(),
    )
    body = score_request_body(
        track="compression",
        reference=reference,
        reference_digest=ref_digest,
        output=output,
        output_digest=out_digest,
    )
    async with _client(create_app(config, backends)) as client:
        resp = await client.post("/score", json=body)

    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert injector.fault_injected
    # The swap really happened on disk...
    assert Path(output).read_bytes() == _GARBAGE
    assert sha256_file(output) != out_digest
    # ...and the packet still names — and was measured from — the verified bytes.
    assert item.content_digest == out_digest
    assert item.metrics["vmaf"] == 93.0
    assert item.gate_passed


async def test_measurement_reads_only_private_snapshot_paths(tmp_path: Path) -> None:
    config, fake, (reference, ref_digest), (output, out_digest) = _fake_world(tmp_path)
    backends = ScoringBackends(
        probe=fake,
        vmaf_primary=fake,
        vmaf_secondary=fake,
        pieapp=fake.pieapp,
        perceptual=fake,
        canonicalizer=None,
        versions=fake.versions(),
    )
    body = score_request_body(
        track="compression",
        reference=reference,
        reference_digest=ref_digest,
        output=output,
        output_digest=out_digest,
    )
    async with _client(create_app(config, backends)) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 200, resp.text

    assert fake.probed_paths, "nothing was probed"
    for probed in fake.probed_paths:
        assert probed not in (reference, output)
        assert config.work_dir in Path(probed).parents


@requires_media_tools
async def test_real_pipeline_scores_the_snapshot_after_a_source_swap(
    clips: ClipPair, tmp_path: Path
) -> None:
    """Same probe, real ffmpeg/libvmaf: the swapped bytes never reach the metric."""
    reference = tmp_path / "ref.mp4"
    candidate = tmp_path / "cand.mp4"
    reference.write_bytes(Path(clips.reference).read_bytes())
    candidate.write_bytes(Path(clips.candidate).read_bytes())
    ref_digest, cand_digest = sha256_file(reference), sha256_file(candidate)

    config = ScoringWorkerConfig(
        work_dir=tmp_path / "work",
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        request_timeout=120.0,
        subprocess_timeout=60.0,
        perceptual_checks="skip",
    )
    config.work_dir.mkdir(parents=True, exist_ok=True)
    primary = FfmpegVmafBackend(FFMPEG, work_dir=config.work_dir, timeout=60.0)
    secondary = FfmpegVmafBackend(
        FFMPEG, model=SECONDARY_VMAF_MODEL, work_dir=config.work_dir, timeout=60.0
    )
    canonicalizer = _FaultInjectingCanonicalizer(
        FFMPEG, [reference, candidate], timeout=60.0
    )
    backends = ScoringBackends(
        probe=FfprobeBackend(FFPROBE, timeout=60.0),
        vmaf_primary=primary,
        vmaf_secondary=secondary,
        pieapp=DeterministicFakeBackend().pieapp,
        perceptual=DeterministicFakeBackend(),
        canonicalizer=canonicalizer,
        versions=detect_tool_versions(
            FFMPEG, FFPROBE, vmaf_backend=primary, timeout=30.0
        ),
    )
    body = score_request_body(
        track="compression",
        reference=str(reference),
        reference_digest=ref_digest,
        output=str(candidate),
        output_digest=cand_digest,
        params={"vmaf_threshold": 90.0},
    )
    async with _client(create_app(config, backends)) as client:
        resp = await client.post("/score", json=body)

    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert canonicalizer.fault_injected
    assert candidate.read_bytes() == _GARBAGE  # the source is now garbage
    assert item.content_digest == cand_digest  # the packet names the measured bytes
    assert item.breakdown is not None and 0.0 < item.breakdown.vmaf <= 100.0
    assert item.gate_passed and not item.violations


# --- non-regular / symlinked inputs ----------------------------------------------------


def _fake_app(tmp_path: Path):
    config, fake, reference, output = _fake_world(tmp_path)
    backends = ScoringBackends(
        probe=fake,
        vmaf_primary=fake,
        vmaf_secondary=fake,
        pieapp=fake.pieapp,
        perceptual=fake,
        canonicalizer=None,
        versions=fake.versions(),
    )
    return create_app(config, backends), config, reference, output


async def test_symlinked_output_is_422(tmp_path: Path) -> None:
    app, _config, (reference, ref_digest), (output, out_digest) = _fake_app(tmp_path)
    link = tmp_path / "link.bin"
    link.symlink_to(output)
    body = score_request_body(
        track="compression",
        reference=reference,
        reference_digest=ref_digest,
        output=str(link),
        output_digest=out_digest,  # the digest is CORRECT — the symlink is the problem
    )
    async with _client(app) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "symlink_rejected"
    assert detail["field"] == "output"


async def test_fifo_input_is_422_and_never_blocks(tmp_path: Path) -> None:
    app, _config, (reference, ref_digest), (output, out_digest) = _fake_app(tmp_path)
    fifo = tmp_path / "pipe.bin"
    os.mkfifo(fifo)
    body = score_request_body(
        track="compression",
        reference=reference,
        reference_digest=ref_digest,
        output=str(fifo),
        output_digest=out_digest,
    )
    async with _client(app) as client:
        started = time.perf_counter()
        resp = await asyncio.wait_for(client.post("/score", json=body), timeout=10.0)
        elapsed = time.perf_counter() - started
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "not_a_regular_file"
    assert detail["detail"] == "fifo"
    assert elapsed < 5.0  # a blocking open would have hung until the timeout


async def test_character_device_input_is_422(tmp_path: Path) -> None:
    """/dev/zero would otherwise hash forever — it is refused, not read."""
    app, _config, (reference, ref_digest), (output, out_digest) = _fake_app(tmp_path)
    body = score_request_body(
        track="compression",
        reference=reference,
        reference_digest=ref_digest,
        output="/dev/zero",
        output_digest=out_digest,
    )
    async with _client(app) as client:
        resp = await asyncio.wait_for(client.post("/score", json=body), timeout=10.0)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "not_a_regular_file"
    assert detail["detail"] == "character device"


async def test_directory_input_is_422(tmp_path: Path) -> None:
    app, _config, (reference, ref_digest), (output, out_digest) = _fake_app(tmp_path)
    body = score_request_body(
        track="compression",
        reference=reference,
        reference_digest=ref_digest,
        output=str(tmp_path),
        output_digest=out_digest,
    )
    async with _client(app) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] in ("not_a_regular_file", "unreadable_input")


# --- work-dir hygiene ------------------------------------------------------------------


async def test_snapshots_are_removed_when_the_request_ends(tmp_path: Path) -> None:
    app, config, (reference, ref_digest), (output, out_digest) = _fake_app(tmp_path)
    body = score_request_body(
        track="compression",
        reference=reference,
        reference_digest=ref_digest,
        output=output,
        output_digest=out_digest,
    )
    async with _client(app) as client:
        assert (await client.post("/score", json=body)).status_code == 200
    leftovers = [p for p in config.work_dir.iterdir() if p.name.startswith(WORK_PREFIX)]
    assert leftovers == []


def test_snapshot_copy_is_read_only_and_carries_the_container_suffix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "miner-output.mp4"
    source.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()

    verified = snapshot_input(
        "output", str(source), digest, dest_dir=tmp_path / "private"
    )

    copy = Path(verified.path)
    assert copy.name == "output.mp4"  # ffmpeg keeps its container hint
    assert copy.read_bytes() == b"payload"
    assert copy.stat().st_mode & 0o777 == 0o400
    assert verified.digest == digest
    assert verified.source_path == str(source)


def test_snapshot_copy_drops_an_exotic_suffix(tmp_path: Path) -> None:
    source = tmp_path / "out.this-is-not-a-container"
    source.write_bytes(b"payload")
    verified = snapshot_input(
        "output",
        str(source),
        hashlib.sha256(b"payload").hexdigest(),
        dest_dir=tmp_path / "private",
    )
    assert Path(verified.path).name == "output"


def test_snapshot_aborts_and_cleans_up_when_cancelled(tmp_path: Path) -> None:
    """A timed-out request must not keep copying gigabytes after its 504."""
    source = tmp_path / "big.bin"
    source.write_bytes(b"z" * (HASH_CHUNK * 3))
    dest_dir = tmp_path / "private"

    with pytest.raises(SnapshotCancelled):
        snapshot_input(
            "output",
            str(source),
            hashlib.sha256(source.read_bytes()).hexdigest(),
            dest_dir=dest_dir,
            cancelled=lambda: True,
        )
    assert list(dest_dir.iterdir()) == []  # the partial copy is discarded


def test_sweep_removes_every_scratch_shape_the_worker_can_leave(tmp_path: Path) -> None:
    """A crash leaves more than snapshot dirs behind — all of it must be reclaimed.

    libvmaf's own temp dirs are created by :class:`tempfile.TemporaryDirectory`,
    which cleans up only if the owning process lives to do it. A sweep that knew
    just the per-request prefix therefore left `vmaf-*` (and the odd health
    probe) on the volume for the life of the deployment.
    """
    work = tmp_path / "work"
    (work / f"{WORK_PREFIX}crashed" / "inputs").mkdir(parents=True)
    stale_copy = work / f"{WORK_PREFIX}crashed" / "inputs" / "output.bin"
    stale_copy.write_bytes(b"x")
    stale_copy.chmod(0o400)  # snapshots are read-only; the sweep must still win
    (work / f"{VMAF_SCRATCH_PREFIX}abcd1234").mkdir()
    (work / f"{VMAF_SCRATCH_PREFIX}abcd1234" / "vmaf.json").write_bytes(b"{}")
    (work / f"{VMAF_VERSION_SCRATCH_PREFIX}ef56").mkdir()
    (work / f"{HEALTH_PROBE_PREFIX}deadbeef").write_bytes(b"ok")
    # Not ours: the sweep reclaims our leftovers, it does not wipe the directory.
    keep_dir = work / "operator-notes"
    keep_dir.mkdir()
    keep_file = work / "README"
    keep_file.write_bytes(b"read me")

    assert sweep_work_dir(work) == 4
    assert not (work / f"{WORK_PREFIX}crashed").exists()
    assert not (work / f"{VMAF_SCRATCH_PREFIX}abcd1234").exists()
    assert not (work / f"{VMAF_VERSION_SCRATCH_PREFIX}ef56").exists()
    assert not (work / f"{HEALTH_PROBE_PREFIX}deadbeef").exists()
    assert keep_dir.exists() and keep_file.exists()


def test_worker_startup_sweeps_the_work_dir(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / f"{WORK_PREFIX}crashed").mkdir(parents=True)
    # Crash-left libvmaf scratch: nobody but this sweep will ever remove it.
    planted_vmaf = work / f"{VMAF_SCRATCH_PREFIX}0f0f0f0f"
    planted_vmaf.mkdir(parents=True)
    (planted_vmaf / "vmaf.json").write_bytes(b'{"frames": []}')
    fake = RoleKeyedBackend()
    backends = ScoringBackends(
        probe=fake,
        vmaf_primary=fake,
        vmaf_secondary=None,
        pieapp=fake.pieapp,
        perceptual=fake,
        canonicalizer=None,
        versions=fake.versions(),
    )
    raw = {
        "core": {"metrics_port": 0},
        "scoring_worker": {
            "backend": "fake",
            "port": 0,
            "metrics_port": 0,
            "work_dir": str(work),
        },
    }
    ScoringWorker(raw, backends=backends)
    assert not (work / f"{WORK_PREFIX}crashed").exists()
    assert not planted_vmaf.exists()


# --- residual scratch the sweep could NOT delete --------------------
#
# The sweep suppresses failures so startup never dies on a permission oddity — but
# what it fails to delete is still on the volume. A fresh budget that starts at
# zero would then admit a full budget's worth of new work ON TOP of the leftovers:
# an overcommit that ends in ENOSPC mid-request. The leftovers are therefore
# measured and pre-charged, admission genuinely shrinks, and the charge is released
# only when a retry sweep observes the bytes actually gone.


def _fake_service_backends() -> ScoringBackends:
    fake = RoleKeyedBackend()
    return ScoringBackends(
        probe=fake,
        vmaf_primary=fake,
        vmaf_secondary=None,
        pieapp=fake.pieapp,
        perceptual=fake,
        canonicalizer=None,
        versions=fake.versions(),
    )


def _plant_undeletable_leftover(work: Path, *, nbytes: int) -> tuple[Path, Path]:
    """A crash-left request dir whose contents a sweep pass cannot delete.

    ``score-crashed/locked/`` is read+execute only, so unlinking the blob inside
    it fails EACCES; one rmtree pass (chmod-and-retry hook included) leaves the
    blob in place. Returns ``(entry, locked)`` — re-lock `locked` after every
    attempted sweep, because the retry hook chmods it back on its way through.
    """
    entry = work / f"{WORK_PREFIX}crashed"
    locked = entry / "locked"
    locked.mkdir(parents=True)
    (locked / "snapshot.bin").write_bytes(b"x" * nbytes)
    locked.chmod(0o500)
    return entry, locked


@pytest.mark.skipif(os.geteuid() == 0, reason="permission bits do not bind root")
def test_undeletable_leftovers_are_charged_shrink_admission_and_release(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    entry, locked = _plant_undeletable_leftover(work, nbytes=5_000)
    blob = locked / "snapshot.bin"
    try:
        sweep_work_dir(work)
        assert blob.exists()  # the sweep could not reclaim it
        locked.chmod(0o500)  # re-lock what the sweep's retry hook opened

        residual, unmeasurable = measure_scratch_entries(work)
        assert residual == {str(entry): 5_000}
        assert unmeasurable == []  # 0o500 is traversable: fully measured

        budget = ScratchBudget(
            ByteLimits(
                max_input_bytes=8_000,
                max_request_bytes=8_000,
                max_request_scratch_bytes=8_000,
                max_scratch_bytes=8_000,
            )
        )
        assert budget.charge_residual(residual) == 5_000
        assert budget.used_bytes == 5_000
        assert budget.residual_bytes == 5_000
        assert budget.charge_residual(residual) == 0  # same path never charged twice
        assert budget.used_bytes == 5_000

        # Admission SHRINKS by exactly the residual: what an empty budget would
        # have taken no longer fits...
        lease = budget.lease()
        with pytest.raises(ScoreRejected) as excinfo:
            lease.reserve(field="reference", path_text="/r", nbytes=4_000)
        assert excinfo.value.status_code == 503
        assert excinfo.value.payload["error"] == "scratch_budget_unavailable"
        # ...while what genuinely fits BESIDE the leftovers is still admitted.
        lease.reserve(field="reference", path_text="/r", nbytes=2_000)

        # Still undeletable: the retry deletes nothing and keeps the charge.
        assert budget.retry_residual_sweep() == 0
        assert budget.residual_bytes == 5_000
        assert blob.exists()
    finally:
        locked.chmod(0o700)

    # An operator fixed the permissions: the next retry deletes the leftovers
    # and releases exactly their bytes back to admission.
    assert budget.retry_residual_sweep() == 5_000
    assert budget.residual_bytes == 0
    assert not entry.exists()
    assert budget.used_bytes == 2_000  # only the live lease's reservation remains
    lease.release()
    assert budget.used_bytes == 0


def test_measuring_a_missing_work_dir_is_empty(tmp_path: Path) -> None:
    assert measure_scratch_entries(tmp_path / "never-created") == ({}, [])


@pytest.mark.skipif(os.geteuid() == 0, reason="permission bits do not bind root")
def test_untraversable_leftover_is_unmeasurable_not_zero(tmp_path: Path) -> None:
    """Round-5 an internal review: a 0o000 dir hides its contents — it must surface as
    UNMEASURABLE, never as a zero-byte measurable entry the budget would treat
    as bounded."""
    work = tmp_path / "work"
    hidden = work / "score-crashed"
    hidden.mkdir(parents=True)
    (hidden / "big.bin").write_bytes(b"x" * 4_096)
    hidden.chmod(0o000)
    try:
        measurable, unmeasurable = measure_scratch_entries(work)
        assert str(hidden) not in measurable
        assert unmeasurable == [str(hidden)]
    finally:
        hidden.chmod(0o700)


def test_worker_refuses_to_start_next_to_unmeasurable_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker's contract: an unmeasurable leftover means no admission is
    honest — fail_fatal at startup, naming the path for the operator.

    Simulated via monkeypatch: the real trigger is a leftover owned by ANOTHER
    uid, which a test cannot mint without root. A same-uid
    0o000 dir does NOT trigger this — the sweep's heal hook chmods it back to
    0o700, so it becomes measurable and is reclaimed on a later retry sweep
    (see test_untraversable_leftover_is_unmeasurable_not_zero for the raw
    measurement behavior without the sweep's healing)."""
    import vidaio.scoring_worker.service as worker_service

    work = tmp_path / "work"
    foreign = str(work / f"{WORK_PREFIX}foreign-uid")
    monkeypatch.setattr(
        worker_service, "measure_scratch_entries", lambda _wd: ({}, [foreign])
    )
    fake = RoleKeyedBackend()
    backends = ScoringBackends(
        probe=fake,
        vmaf_primary=fake,
        vmaf_secondary=None,
        pieapp=fake.pieapp,
        perceptual=fake,
        canonicalizer=None,
        versions=fake.versions(),
    )
    raw = {
        "core": {"metrics_port": 0},
        "scoring_worker": {
            "backend": "fake",
            "port": 0,
            "metrics_port": 0,
            "work_dir": str(work),
        },
    }
    worker = ScoringWorker(raw, backends=backends)
    assert worker.failed_fatally
    assert worker.fatal_reason is not None
    assert foreign in worker.fatal_reason


@pytest.mark.skipif(os.geteuid() == 0, reason="permission bits do not bind root")
def test_same_uid_locked_leftover_self_heals_through_the_sweep(tmp_path: Path) -> None:
    """A same-uid 0o000 leftover is NOT fatal: the sweep's heal hook restores
    0o700 (without crashing startup — round-5 sweep TypeError fix), after which
    the entry is measurable and reclaimable."""
    work = tmp_path / "work"
    hidden = work / f"{WORK_PREFIX}crashed"
    hidden.mkdir(parents=True)
    (hidden / "big.bin").write_bytes(b"x" * 4_096)
    hidden.chmod(0o000)
    try:
        sweep_work_dir(work)  # must not raise (the old hook TypeError'd here)
        measurable, unmeasurable = measure_scratch_entries(work)
        assert unmeasurable == []
        # Healed to traversable: either already deleted by the sweep or
        # measured at its full visible size for pre-charging.
        if str(hidden) in measurable:
            assert measurable[str(hidden)] == 4_096
    finally:
        if hidden.exists():
            hidden.chmod(0o700)


@pytest.mark.skipif(os.geteuid() == 0, reason="permission bits do not bind root")
def test_worker_startup_precharges_what_the_sweep_left_behind(tmp_path: Path) -> None:
    """The service path end to end: sweep, measure the survivors, charge, WARN."""
    work = tmp_path / "work"
    entry, locked = _plant_undeletable_leftover(work, nbytes=5_000)
    raw = {
        "core": {"metrics_port": 0},
        "scoring_worker": {
            "backend": "fake",
            "port": 0,
            "metrics_port": 0,
            "work_dir": str(work),
        },
    }
    try:
        worker = ScoringWorker(raw, backends=_fake_service_backends())
    finally:
        locked.chmod(0o700)

    budget = worker.app.state.scratch_budget
    assert budget.residual_bytes == 5_000
    assert budget.used_bytes == 5_000
    # Within the (default, huge) budget: an operator WARNING, not a fatal.
    assert worker.failed_fatally is False


@pytest.mark.skipif(os.geteuid() == 0, reason="permission bits do not bind root")
def test_residuals_larger_than_the_whole_budget_are_fatal_at_startup(
    tmp_path: Path,
) -> None:
    """Leftovers bigger than the budget mean no request can ever be admitted
    honestly — an operator problem, refused loudly instead of overcommitted."""
    work = tmp_path / "work"
    entry, locked = _plant_undeletable_leftover(work, nbytes=5_000)
    raw = {
        "core": {"metrics_port": 0},
        "scoring_worker": {
            "backend": "fake",
            "port": 0,
            "metrics_port": 0,
            "work_dir": str(work),
            "max_input_bytes": 1_000,
            "max_request_bytes": 1_000,
            "max_request_scratch_bytes": 1_000,
            "max_scratch_bytes": 1_000,
        },
    }
    try:
        worker = ScoringWorker(raw, backends=_fake_service_backends())
    finally:
        locked.chmod(0o700)

    assert worker.failed_fatally is True
    assert "reclaim" in (worker.fatal_reason or "")
    assert worker.stopping.is_set()  # fail_fatal requested the stop


async def test_a_scored_request_retries_the_residual_sweep_and_frees_the_bytes(
    tmp_path: Path,
) -> None:
    """The release half of the contract, on the real request path.

    ``create_app`` charges whatever is still under our prefixes (it does not
    sweep — the service does, before it); the leftover here is deletable, so the
    FIRST scored request's retry sweep reclaims it and the budget goes back to
    admitting those bytes. No operator action, no restart."""
    work = tmp_path / "work"
    leftover = work / f"{WORK_PREFIX}crashed"
    leftover.mkdir(parents=True)
    (leftover / "blob.bin").write_bytes(b"x" * 4_096)

    app, _config, (reference, ref_digest), (output, out_digest) = _fake_app(tmp_path)
    budget = app.state.scratch_budget
    assert budget.residual_bytes == 4_096
    assert budget.used_bytes == 4_096

    async with _client(app) as client:
        resp = await client.post(
            "/score",
            json=score_request_body(
                track="compression",
                reference=reference,
                reference_digest=ref_digest,
                output=output,
                output_digest=out_digest,
            ),
        )

    assert resp.status_code == 200, resp.text
    assert not leftover.exists()  # the retry sweep reclaimed it
    assert budget.residual_bytes == 0
    assert budget.used_bytes == 0  # ...and released it: nothing left reserved


# --- byte budgets: snapshotting must not amplify a caller into our volume -------------
#
# Verify-then-snapshot COPIES, so an unbounded input is an unbounded write to the
# scoring volume — a miner returning one enormous regular file could fill it long
# before request_timeout fires, taking down every concurrent request with it. The
# three ceilings are enforced BEFORE and DURING the copy, never after.


def _work_dir_bytes(work_dir: Path) -> int:
    """Every byte currently living under the worker's scratch root."""
    return sum(p.stat().st_size for p in work_dir.rglob("*") if p.is_file())


class _BlockingVmaf:
    """Holds a request (and therefore its scratch reservation) open on demand."""

    name = "blocking-vmaf"
    version = "1"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def compute(
        self, reference: str, candidate: str, *, deterministic_seed: int = 0
    ) -> float:
        self.entered.set()
        assert self.release.wait(30.0), "the blocking vmaf was never released"
        return 93.0


def _sized_world(
    tmp_path: Path,
    *,
    reference_bytes: int = 10_000,
    output_bytes: int = 5_000,
    vmaf: Any = None,
    **limits: int,
):
    """An app whose inputs have KNOWN sizes and whose byte budgets are tiny."""
    reference, ref_digest = _write(tmp_path / "ref.bin", b"R" * reference_bytes)
    output, out_digest = _write(tmp_path / "out.bin", b"O" * output_bytes)
    fake = RoleKeyedBackend(
        vmaf={("reference", "output"): 93.0},
        media={
            "reference": _media(reference_bytes),
            "output": _media(output_bytes),
            "miner_input": _media(reference_bytes),
        },
    )
    config = ScoringWorkerConfig(
        backend="fake",
        work_dir=tmp_path / "work",
        request_timeout=30.0,
        max_concurrent=2,
        **limits,
    )
    backends = ScoringBackends(
        probe=fake,
        vmaf_primary=vmaf if vmaf is not None else fake,
        vmaf_secondary=fake,
        pieapp=fake.pieapp,
        perceptual=fake,
        canonicalizer=None,
        versions=fake.versions(),
    )
    body = score_request_body(
        track="compression",
        reference=reference,
        reference_digest=ref_digest,
        output=output,
        output_digest=out_digest,
    )
    return create_app(config, backends), config, body


def test_oversize_input_is_refused_before_a_single_byte_is_copied(
    tmp_path: Path,
) -> None:
    payload = b"z" * (256 * 1024)
    source = tmp_path / "huge.bin"
    source.write_bytes(payload)
    dest_dir = tmp_path / "private"
    budget = ScratchBudget(
        ByteLimits(
            max_input_bytes=1024, max_request_bytes=1024, max_scratch_bytes=4096
        )
    )
    lease = budget.lease()

    with pytest.raises(ScoreRejected) as excinfo:
        snapshot_input(
            "output",
            str(source),
            hashlib.sha256(payload).hexdigest(),
            dest_dir=dest_dir,
            lease=lease,
        )

    assert excinfo.value.status_code == 422
    assert excinfo.value.payload["error"] == "input_too_large"
    assert excinfo.value.payload["input_bytes"] == len(payload)
    assert excinfo.value.payload["limit"] == 1024
    # The fstat decided it: nothing was written, so nothing has to be cleaned up.
    assert list(dest_dir.iterdir()) == []
    assert budget.used_bytes == 0 and lease.held_bytes == 0


def test_a_source_that_grows_after_its_fstat_is_cut_off_at_its_reservation(
    tmp_path: Path,
) -> None:
    """The writer we do not control keeps appending — the copy still stops."""
    source = tmp_path / "growing.bin"
    source.write_bytes(b"z" * (HASH_CHUNK * 2))
    reserved = source.stat().st_size
    dest_dir = tmp_path / "private"
    budget = ScratchBudget(
        ByteLimits(
            max_input_bytes=reserved,
            max_request_bytes=reserved,
            max_scratch_bytes=reserved * 8,
        )
    )
    lease = budget.lease()

    def keep_growing() -> bool:
        # Called once per copied chunk. It never cancels: the byte LIMIT has to
        # be the thing that stops this copy.
        with open(source, "ab") as handle:
            handle.write(b"z" * HASH_CHUNK)
        return False

    with pytest.raises(ScoreRejected) as excinfo:
        snapshot_input(
            "output",
            str(source),
            "0" * 64,
            dest_dir=dest_dir,
            cancelled=keep_growing,
            lease=lease,
        )

    assert excinfo.value.status_code == 422
    assert excinfo.value.payload["error"] == "input_grew_during_snapshot"
    assert excinfo.value.payload["limit"] == reserved
    assert source.stat().st_size > reserved  # the source really did grow
    assert list(dest_dir.iterdir()) == []  # the partial copy is gone
    assert budget.used_bytes == 0 and lease.held_bytes == 0


async def test_oversize_output_is_422_and_the_scratch_volume_stays_bounded(
    tmp_path: Path,
) -> None:
    app, config, body = _sized_world(
        tmp_path,
        reference_bytes=1_000,
        output_bytes=200_000,
        max_input_bytes=2_000,
        max_request_bytes=8_000,
        max_scratch_bytes=16_000,
    )
    async with _client(app) as client:
        resp = await client.post("/score", json=body)

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "input_too_large"
    assert detail["field"] == "output"
    assert detail["input_bytes"] == 200_000
    # The 200 KB file never reached our volume, and no partial survived.
    assert _work_dir_bytes(config.work_dir) < 8_000
    assert [p for p in config.work_dir.iterdir() if p.name.startswith(WORK_PREFIX)] == []
    assert app.state.scratch_budget.used_bytes == 0


async def test_inputs_over_the_per_request_budget_are_413(tmp_path: Path) -> None:
    """Each input fits on its own; together they do not."""
    app, config, body = _sized_world(
        tmp_path,
        reference_bytes=10_000,
        output_bytes=5_000,
        max_input_bytes=10_000,
        max_request_bytes=24_000,  # 10_000 + 10_000 + 5_000 = 25_000
        max_scratch_bytes=48_000,
    )
    async with _client(app) as client:
        resp = await client.post("/score", json=body)

    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert detail["error"] == "request_inputs_too_large"
    assert detail["field"] == "output"
    assert detail["request_bytes"] == 25_000
    assert detail["limit"] == 24_000
    assert [p for p in config.work_dir.iterdir() if p.name.startswith(WORK_PREFIX)] == []
    assert app.state.scratch_budget.used_bytes == 0


async def test_concurrent_requests_are_shed_503_instead_of_filling_the_volume(
    tmp_path: Path,
) -> None:
    """Two requests that each fit must not be allowed to fill the disk together."""
    vmaf = _BlockingVmaf()
    app, config, body = _sized_world(
        tmp_path,
        vmaf=vmaf,
        max_input_bytes=10_000,
        max_request_bytes=25_000,  # one request = 25_000 bytes exactly
        max_scratch_bytes=30_000,  # ...so a second one cannot be admitted
    )
    async with _client(app) as client:
        first = asyncio.create_task(client.post("/score", json=body))
        assert await asyncio.to_thread(
            vmaf.entered.wait, 30.0
        ), "the first request never reached the metric"
        assert app.state.scratch_budget.used_bytes == 25_000

        second = await client.post("/score", json=body)
        assert second.status_code == 503
        detail = second.json()["detail"]
        assert detail["error"] == "scratch_budget_unavailable"
        assert detail["limit"] == 30_000
        assert int(second.headers["retry-after"]) >= 1
        # Shed, not written: the volume still holds only the first request.
        assert _work_dir_bytes(config.work_dir) <= 25_000

        vmaf.release.set()
        assert (await first).status_code == 200
        # The budget frees with the request, so the retry the 503 invited works.
        third = await client.post("/score", json=body)

    assert third.status_code == 200, third.text
    assert app.state.scratch_budget.used_bytes == 0
    assert _work_dir_bytes(config.work_dir) == 0


def test_budget_config_must_be_able_to_hold_one_whole_request() -> None:
    with pytest.raises(ValueError, match="max_request_bytes must be >="):
        ScoringWorkerConfig(max_input_bytes=100, max_request_bytes=50)
    with pytest.raises(ValueError, match="max_scratch_bytes must be >="):
        ScoringWorkerConfig(
            max_input_bytes=50, max_request_bytes=100, max_scratch_bytes=80
        )


@pytest.mark.parametrize("field", ["reference", "miner_input", "output"])
async def test_every_input_field_is_verified(tmp_path: Path, field: str) -> None:
    app, _config, (reference, ref_digest), (output, out_digest) = _fake_app(tmp_path)
    body = score_request_body(
        track="compression",
        reference=reference,
        reference_digest=ref_digest,
        output=output,
        output_digest=out_digest,
    )
    body[f"{field}_digest"] = hashlib.sha256(b"not these bytes").hexdigest()
    async with _client(app) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "digest_mismatch"
    assert detail["field"] == field
