"""CPU PieAPP workers overlap while preserving measured packets and runtime identity."""
from __future__ import annotations

import asyncio
import hashlib
from collections import defaultdict
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from types import SimpleNamespace

import httpx
import pytest

from tests.scoring_worker.conftest import RoleKeyedBackend, score_request_body
from tests.scoring_worker.test_runtime_identity import _attestation
from vidaio.scoring import ItemScore, MediaInfo, ScoringConfig
from vidaio.scoring.backends_real import NotConfiguredError, PieAppTorchBackend
from vidaio.scoring_worker import ScoringBackends, ScoringWorker, ScoringWorkerConfig, create_app, effective_scorer_version
from vidaio.scoring_worker.runtime_identity import runtime_backend_stamp, runtime_commitment_digest


class RuntimeControl:
    """Counts actual loaded models and blocks inside their real backend locks."""

    def __init__(self):
        self.lock = Lock()
        self.release = Event()
        self.release.set()
        self.saturated = Event()
        self.target = 1
        self.loaded = []
        self.calls = []
        self.active = 0
        self.peak = 0

    def block(self, target):
        assert self.active == 0
        self.target = target
        self.saturated.clear()
        self.release.clear()

    def load(self, device):
        control = self

        class Runtime:
            def compute(self, reference, candidate, *, start_frame, sample_window):
                with control.lock:
                    control.active += 1
                    control.peak = max(control.peak, control.active)
                    control.calls.append((self, current_thread(), reference, candidate,
                        start_frame, sample_window))
                    if control.active >= control.target:
                        control.saturated.set()
                try:
                    if not control.release.wait(timeout=5):
                        raise TimeoutError("test did not release PieAPP inference")
                    # Measure the actual verified snapshot bytes, independent of its path.
                    return 0.1 + (Path(candidate).read_bytes()[0] % 3) / 100
                finally:
                    with control.lock:
                        control.active -= 1

        runtime = Runtime()
        with self.lock:
            self.loaded.append((runtime, device, current_thread()))
        return runtime


def _media(size, *, width=320, height=240):
    return MediaInfo(codec="h264", width=width, height=height, fps=30.0,
        frame_count=60, duration=2.0, byte_size=size)


def _world(root, *, concurrency, device="cpu", preload=False):
    root.mkdir(parents=True)
    control = RuntimeControl()
    scoring = ScoringConfig(pieapp_sample_window=7)
    pieapp = PieAppTorchBackend(device=device, sample_window=scoring.pieapp_sample_window,
        _runtime_loader=control.load, _backend_version="piq/0.8.0:pieapp")
    fake = RoleKeyedBackend(vmaf={("reference", "output"): 93.0}, media={
        "reference": _media(10_000), "output": _media(5_000),
        "miner_input": _media(1_000, width=160, height=120),
    })
    attestation = _attestation()
    attestation["payout_backends"]["pieapp"] = f"{pieapp.name}/{pieapp.version}:{device}"
    versions = dict(attestation["payout_backends"])
    versions["runtime"] = runtime_backend_stamp(attestation)
    backends = ScoringBackends(probe=fake, vmaf_primary=fake, vmaf_secondary=fake,
        pieapp=pieapp, perceptual=fake, canonicalizer=None, versions=versions,
        runtime_attestation=attestation)
    config = ScoringWorkerConfig(backend="fake", pieapp_device=device,
        work_dir=root / "work", max_concurrent=concurrency, request_timeout=30,
        queue_wait_timeout_seconds=2)
    reference, miner_input = root / "ref.bin", root / "input.bin"
    reference.write_bytes(b"R" * 10_000)
    miner_input.write_bytes(b"I" * 1_000)
    bodies = []
    for index in range(3):
        output = root / f"out-{index}.bin"
        output.write_bytes(bytes([79 + index]) * 5_000)
        bodies.append(score_request_body(track="upscaling", challenge_id="pieapp-concurrent",
            item_id=f"item-{index}", reference=str(reference),
            reference_digest=hashlib.sha256(reference.read_bytes()).hexdigest(),
            miner_input=str(miner_input),
            miner_input_digest=hashlib.sha256(miner_input.read_bytes()).hexdigest(),
            output=str(output), output_digest=hashlib.sha256(output.read_bytes()).hexdigest(),
            params={"upscale_factor": 2, "content_length": 10.0}))
    if preload:
        pieapp.preload()
    app = create_app(config, backends, scoring_config=scoring)
    return SimpleNamespace(control=control, pieapp=pieapp, backends=backends,
        attestation=attestation, config=config, scoring=scoring, bodies=bodies, app=app)


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker")


async def _wait(event):
    assert await asyncio.to_thread(event.wait, 5), "concurrent worker barrier was not reached"


def _packets(responses):
    assert all(response.status_code == 200 for response in responses), [
        (response.status_code, response.text) for response in responses]
    payloads = [response.json() for response in responses]
    scores = [ItemScore.from_json(payload["item_score_json"]) for payload in payloads]
    assert all(score.gate_passed and not score.violations and score.score > 0 for score in scores)
    assert all(score.breakdown.kind == "upscaling" for score in scores)
    return payloads


@pytest.mark.parametrize("preload", [False, True], ids=["lazy", "preloaded"])
async def test_three_cpu_requests_overlap_reuse_models_and_match_serial_packets(tmp_path, preload):
    serial = _world(tmp_path / "serial", concurrency=1, preload=preload)
    assert len(serial.control.loaded) == int(preload)
    async with serial.app.router.lifespan_context(serial.app), _client(serial.app) as client:
        serial_health = (await client.get("/healthz")).json()
        assert len(serial.control.loaded) == int(preload)
        expected = _packets([await client.post("/score", json=body) for body in serial.bodies])
    assert serial.control.peak == 1
    assert len(serial.control.loaded) == 1
    assert serial.control.calls[0][0] is serial.pieapp._runtime
    assert all(not call[1].is_alive() for call in serial.control.calls)

    parallel = _world(tmp_path / "parallel", concurrency=3, preload=preload)
    control = parallel.control
    assert len(control.loaded) == int(preload)
    async with parallel.app.router.lifespan_context(parallel.app), _client(parallel.app) as client:
        health_before = await client.get("/healthz")
        assert health_before.status_code == 200
        assert health_before.json() == serial_health
        assert len(control.loaded) == int(preload)
        first = _packets([await client.post("/score", json=parallel.bodies[0])])
        assert first == expected[:1]
        assert len(control.loaded) == 1
        assert control.calls[0][0] is parallel.pieapp._runtime
        assert control.calls[0][1] is not current_thread()
        if preload:
            assert control.loaded[0][2] is current_thread()
        else:
            assert control.loaded[0][2] is control.calls[0][1]
        for _ in range(3):
            control.block(3)
            tasks = [asyncio.create_task(client.post("/score", json=body)) for body in parallel.bodies]
            try:
                await _wait(control.saturated)
                assert control.active == control.peak == 3
                assert not any(task.done() for task in tasks)
            finally:
                control.release.set()
                responses = await asyncio.gather(*tasks)
            assert _packets(responses) == expected
            assert len(control.loaded) == parallel.config.max_concurrent
        health_after = await client.get("/healthz")
        assert health_after.status_code == 200
        assert health_after.json() == health_before.json()

    by_thread = defaultdict(set)
    for runtime, thread, reference, candidate, start, window in control.calls:
        by_thread[thread].add(id(runtime))
        assert Path(reference).stem == "reference" and Path(candidate).stem == "output"
        assert start >= 0 and window == parallel.scoring.pieapp_sample_window
    assert len(by_thread) == 3 and all(len(models) == 1 for models in by_thread.values())
    assert len({id(runtime) for runtime, *_ in control.calls}) == 3
    assert len(control.loaded) == 3 and all(device == "cpu" for _, device, _ in control.loaded)
    assert control.active == 0
    assert all(not thread.is_alive() for thread in by_thread)
    expected_identity = effective_scorer_version(parallel.config, parallel.scoring,
        runtime_attestation=parallel.attestation)
    assert health_after.json()["scorer_version"] == expected_identity
    assert health_after.json()["runtime_commitment"] == {
        "digest": runtime_commitment_digest(parallel.attestation),
        "attestation": parallel.attestation,
    }
    assert parallel.backends.versions["runtime"] == runtime_backend_stamp(parallel.attestation)
    assert parallel.app.state.scratch_budget.used_bytes == 0


async def test_cuda_requests_keep_one_shared_serialized_runtime(tmp_path, monkeypatch):
    world = _world(tmp_path / "cuda", concurrency=3, device="cuda")
    control = world.control
    original_ready = world.pieapp.ensure_ready
    attempted, attempts = Event(), []

    def ready():
        with control.lock:
            attempts.append(current_thread())
            if len(attempts) == 3:
                attempted.set()
        return original_ready()

    monkeypatch.setattr(world.pieapp, "ensure_ready", ready)
    async with world.app.router.lifespan_context(world.app), _client(world.app) as client:
        before = await client.get("/healthz")
        control.block(1)
        tasks = [asyncio.create_task(client.post("/score", json=body)) for body in world.bodies]
        try:
            await _wait(control.saturated)
            await _wait(attempted)
            assert control.active == control.peak == 1
            assert len(control.loaded) == 1
            assert not any(task.done() for task in tasks)
        finally:
            control.release.set()
            responses = await asyncio.gather(*tasks)
        concurrent = _packets(responses)
        serial = _packets([await client.post("/score", json=body) for body in world.bodies])
        assert concurrent == serial
        after = await client.get("/healthz")
        assert after.json() == before.json()
    assert control.active == 0 and control.peak == 1
    assert len(control.loaded) == 1 and control.loaded[0][1] == "cuda"
    assert len({id(runtime) for runtime, *_ in control.calls}) == 1


async def test_service_close_waits_for_cpu_scoring_after_caller_cancellation(tmp_path):
    world = _world(tmp_path / "closing", concurrency=3, preload=True)
    worker = ScoringWorker({
        "core": {"metrics_port": 0},
        "scoring_worker": world.config.model_dump(mode="json") | {"metrics_port": 0},
        "scoring": world.scoring.model_dump(mode="json"),
    }, backends=world.backends)
    control = world.control
    closing, closed = Event(), Event()
    close_errors = []

    def close():
        closing.set()
        try:
            worker.close()
        except BaseException as exc:
            close_errors.append(exc)
        finally:
            closed.set()

    closer = Thread(target=close, name="test-scoring-worker-close")
    async with _client(worker.app) as client:
        control.block(1)
        request = asyncio.create_task(client.post("/score", json=world.bodies[0]))
        try:
            await _wait(control.saturated)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert control.active == 1
            closer.start()
            await _wait(closing)
            assert not await asyncio.to_thread(closed.wait, 0.05)
            assert control.active == 1
        finally:
            control.release.set()
            await asyncio.gather(request, return_exceptions=True)
            if closer.ident is not None:
                await asyncio.to_thread(closer.join, 5)
            worker.close()

    assert closed.is_set() and not closer.is_alive()
    assert close_errors == []
    assert control.active == 0
    assert all(not call[1].is_alive() for call in control.calls)
    assert len(control.loaded) == 1
    assert worker.app.state.scratch_budget.used_bytes == 0


async def test_lifespan_shutdown_drains_cpu_workers_through_repeated_cancellation(tmp_path):
    world = _world(tmp_path / "cancelled-shutdown", concurrency=3)
    control = world.control
    lifespan = world.app.router.lifespan_context(world.app)
    await lifespan.__aenter__()
    closing = asyncio.Event()
    exit_task = None

    async def exit_lifespan():
        closing.set()
        await lifespan.__aexit__(None, None, None)

    async with _client(world.app) as client:
        control.block(1)
        request = asyncio.create_task(client.post("/score", json=world.bodies[0]))
        try:
            await _wait(control.saturated)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert control.active == 1
            exit_task = asyncio.create_task(exit_lifespan())
            await asyncio.wait_for(closing.wait(), timeout=1)
            for _ in range(3):
                # Health requests must still run while shutdown joins blocked scoring.
                response = await asyncio.wait_for(client.get("/healthz"), timeout=1)
                assert response.status_code == 200
                assert control.active == 1 and not exit_task.done()
                exit_task.cancel()
                await asyncio.sleep(0)
                assert control.active == 1 and not exit_task.done()
            control.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(exit_task, timeout=5)
        finally:
            control.release.set()
            await asyncio.gather(request, return_exceptions=True)
            if exit_task is None:
                await lifespan.__aexit__(None, None, None)
            else:
                await asyncio.gather(exit_task, return_exceptions=True)

    assert exit_task.cancelled()
    assert control.active == 0
    assert all(not call[1].is_alive() for call in control.calls)
    assert world.app.state.scratch_budget.used_bytes == 0


def test_backend_clone_keeps_recipe_and_loads_an_independent_runtime_lazily(tmp_path):
    control = RuntimeControl()
    original = PieAppTorchBackend(device="cpu", sample_window=7,
        _runtime_loader=control.load, _backend_version="pinned-test-version")
    original.ensure_ready()
    original.preload()
    assert len(control.loaded) == 1
    clone = original.clone()
    assert clone is not original
    assert clone.device == original.device == "cpu"
    assert clone.sample_window == original.sample_window == 7
    assert clone.version == original.version == "pinned-test-version"
    assert clone._runtime_loader is original._runtime_loader
    assert clone._lock is not original._lock
    assert clone._runtime is None and len(control.loaded) == 1
    media = tmp_path / "media.bin"
    media.write_bytes(b"O")
    assert original.compute(str(media), str(media), start_frame=11) == clone.compute(
        str(media), str(media), start_frame=11)
    clone.preload()
    assert len(control.loaded) == 2
    assert clone._runtime is not original._runtime
    assert all(call[-2:] == (11, 7) for call in control.calls)


def test_backend_clone_preserves_explicit_preload_failure():
    calls = []

    def unavailable(device):
        calls.append(device)
        raise NotConfiguredError("pinned PieAPP weights unavailable")

    original = PieAppTorchBackend(device="cpu", sample_window=5,
        _runtime_loader=unavailable, _backend_version="pinned-test-version")
    clone = original.clone()
    for backend in (original, clone):
        with pytest.raises(NotConfiguredError, match="pinned PieAPP weights unavailable"):
            backend.preload()
        assert backend._runtime is None
    assert calls == ["cpu", "cpu"]
