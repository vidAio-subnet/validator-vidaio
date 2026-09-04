"""Timeouts must bound WORK, not just the awaiting coroutine.

Three properties, each of which the pre-fix worker violated:

* a queued request waits a bounded time and is then shed (503 + Retry-After),
  instead of sitting behind the semaphore forever with its own budget not even
  started;
* a timed-out request's ffmpeg process GROUP is killed — no orphan keeps burning
  CPU after the caller got its 504;
* the concurrency slot belongs to the work, so ``max_concurrent`` still holds
  under a burst of requests that all time out.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest

from tests.scoring_worker.conftest import (
    FFMPEG,
    RoleKeyedBackend,
    requires_ffmpeg,
    score_request_body,
)
from vidaio.scoring import MediaInfo
from vidaio.scoring.backends_real import (
    CanonicalizeExecutor,
    MediaProcessScope,
    current_process_scope,
)
from vidaio.scoring_worker import ScoringBackends, ScoringWorkerConfig, create_app

_LONG_SOURCE = "testsrc2=size=320x240:rate=30:duration=900"


def _media(byte_size: int) -> MediaInfo:
    return MediaInfo(
        codec="h264",
        width=320,
        height=240,
        fps=30.0,
        frame_count=60,
        duration=2.0,
        byte_size=byte_size,
    )


def _fake_probe() -> RoleKeyedBackend:
    return RoleKeyedBackend(
        media={
            "reference": _media(10_000),
            "output": _media(5_000),
            "miner_input": _media(10_000),
        }
    )


class _SleepingVmaf:
    """Blocks in pure Python (uninterruptible) and records true concurrency."""

    name = "sleeping-vmaf"
    version = "1"

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._lock = threading.Lock()
        self.live = 0
        self.peak = 0
        self.calls = 0

    def compute(
        self, reference: str, candidate: str, *, deterministic_seed: int = 0
    ) -> float:
        with self._lock:
            self.calls += 1
            self.live += 1
            self.peak = max(self.peak, self.live)
        try:
            time.sleep(self._seconds)
        finally:
            with self._lock:
                self.live -= 1
        return 93.0


class _LongFfmpegVmaf:
    """First call launches a genuinely long ffmpeg through the worker's runner.

    `launched` is the test's BARRIER, and it is armed by hooking the request
    scope's own ``register`` — the method ``backends_real._run`` calls the
    instant it has adopted the child. The event therefore fires exactly when the
    process group exists and belongs to the scope, and ``pids`` is readable from
    that moment on.

    It has to be a barrier rather than a poll. The observer lives on the event
    loop, which is the same thread this request's ``request_timeout`` is counted
    on: a loop scheduled late (loaded machine, busy suite) can miss its own
    polling window entirely, and by then the scope has been cancelled AND the
    process unregistered — so ``live_pids()`` is empty forever and the test
    fails as "the long ffmpeg never started", when in fact it started, ran and
    was killed exactly as designed.
    """

    name = "long-ffmpeg-vmaf"
    version = "1"

    def __init__(self, ffmpeg: str) -> None:
        self._executor = CanonicalizeExecutor(ffmpeg, timeout=900.0)
        self.scope: MediaProcessScope | None = None
        self.launched = threading.Event()
        #: process-group ids adopted by the scope (published with `launched`)
        self.pids: list[int] = []
        self.calls = 0

    def _arm_barrier(self, scope: MediaProcessScope) -> None:
        """Publish the process group the moment the REAL runner registers it."""
        register = scope.register

        def _registered(proc) -> None:
            register(proc)  # production behaviour first (it may kill on adopt)
            self.pids = scope.live_pids()
            self.launched.set()

        scope.register = _registered  # type: ignore[method-assign]

    def compute(
        self, reference: str, candidate: str, *, deterministic_seed: int = 0
    ) -> float:
        self.calls += 1
        if self.calls > 1:  # after the timeout: prove the slot is genuinely free
            return 93.0
        scope = current_process_scope()
        assert scope is not None, "the worker must install a scope for media work"
        self.scope = scope
        self._arm_barrier(scope)
        # An honest argv plan; `-re` paces the synthetic source at its native
        # frame rate, so this genuinely occupies a core for 15 minutes unless
        # something kills it.
        self._executor.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-re",
                "-f",
                "lavfi",
                "-i",
                _LONG_SOURCE,
                "-f",
                "null",
                "-",
            ],
            timeout=900.0,
        )
        return 93.0


def _world(tmp_path: Path, vmaf, **overrides):
    reference = tmp_path / "ref.bin"
    output = tmp_path / "out.bin"
    reference.write_bytes(b"R" * 10_000)
    output.write_bytes(b"O" * 5_000)
    settings = {
        "backend": "fake",
        "work_dir": tmp_path / "work",
        "request_timeout": 0.2,
        "queue_wait_timeout_seconds": 10.0,
        "max_concurrent": 1,
    }
    settings.update(overrides)
    config = ScoringWorkerConfig(**settings)
    probe = _fake_probe()
    backends = ScoringBackends(
        probe=probe,
        vmaf_primary=vmaf,
        vmaf_secondary=None,
        pieapp=probe.pieapp,
        perceptual=probe,
        canonicalizer=None,
        versions={"vmaf": "test/1"},
    )
    body = score_request_body(
        track="compression",
        reference=str(reference),
        reference_digest=hashlib.sha256(reference.read_bytes()).hexdigest(),
        output=str(output),
        output_digest=hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    return create_app(config, backends), body


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker"
    )


def _our_ffmpeg_children() -> list[str]:
    """ffmpeg processes still parented to this test process."""
    completed = subprocess.run(
        ["ps", "-ax", "-o", "pid=,ppid=,command="],
        capture_output=True,
        text=True,
        timeout=30,
    )
    mine = str(os.getpid())
    return [
        line
        for line in completed.stdout.splitlines()
        if len(line.split(None, 2)) == 3
        and line.split(None, 2)[1] == mine
        and "ffmpeg" in line.split(None, 2)[2]
    ]


# --- queue admission -------------------------------------------------------------------


async def test_saturated_queue_sheds_with_503_and_retry_after(tmp_path: Path) -> None:
    app, body = _world(
        tmp_path,
        _SleepingVmaf(0.6),
        request_timeout=10.0,
        queue_wait_timeout_seconds=0.05,
        max_concurrent=1,
    )
    async with _client(app) as client:
        first = asyncio.create_task(client.post("/score", json=body))
        await asyncio.sleep(0.15)  # let it take the only slot
        started = time.perf_counter()
        second = await client.post("/score", json=body)
        waited = time.perf_counter() - started
        assert (await first).status_code == 200

    assert second.status_code == 503
    assert second.json()["detail"]["error"] == "queue_saturated"
    assert int(second.headers["retry-after"]) >= 1
    assert waited < 1.0  # shed, not queued behind the 0.6s scoring


async def test_queue_shedding_is_distinct_from_a_scoring_timeout(
    tmp_path: Path,
) -> None:
    """503 = never started; 504 = started and ran out of budget."""
    app, body = _world(
        tmp_path,
        _SleepingVmaf(0.5),
        request_timeout=0.1,
        queue_wait_timeout_seconds=0.05,
        max_concurrent=1,
    )
    async with _client(app) as client:
        first = asyncio.create_task(client.post("/score", json=body))
        await asyncio.sleep(0.15)
        shed = await client.post("/score", json=body)
        timed_out = await first

    assert timed_out.status_code == 504
    assert timed_out.json()["detail"]["error"] == "scoring_timeout"
    assert shed.status_code == 503
    assert shed.json()["detail"]["error"] == "queue_saturated"


# --- the slot belongs to the work ------------------------------------------------------


async def test_concurrency_never_exceeds_max_concurrent_under_a_timeout_burst(
    tmp_path: Path,
) -> None:
    """Every request times out; none of them may double up the real workload."""
    vmaf = _SleepingVmaf(0.4)
    app, body = _world(
        tmp_path,
        vmaf,
        request_timeout=0.1,  # every request 504s long before the work ends
        queue_wait_timeout_seconds=30.0,
        max_concurrent=1,
    )
    async with _client(app) as client:
        responses = await asyncio.gather(
            *(client.post("/score", json=body) for _ in range(4))
        )
    assert [r.status_code for r in responses] == [504, 504, 504, 504]
    # Releasing the slot on the AWAIT (the bug) would have run all four at once.
    assert vmaf.calls == 4
    assert vmaf.peak == 1


async def test_two_slots_are_used_but_not_exceeded(tmp_path: Path) -> None:
    vmaf = _SleepingVmaf(0.3)
    app, body = _world(
        tmp_path,
        vmaf,
        request_timeout=5.0,
        queue_wait_timeout_seconds=30.0,
        max_concurrent=2,
    )
    async with _client(app) as client:
        responses = await asyncio.gather(
            *(client.post("/score", json=body) for _ in range(4))
        )
    assert {r.status_code for r in responses} == {200}
    assert vmaf.peak == 2


# --- real cancellation -----------------------------------------------------------------


@requires_ffmpeg
async def test_timeout_kills_the_process_group_and_frees_the_slot(
    tmp_path: Path,
) -> None:
    vmaf = _LongFfmpegVmaf(FFMPEG)
    app, body = _world(
        tmp_path,
        vmaf,
        request_timeout=1.0,
        queue_wait_timeout_seconds=20.0,
        max_concurrent=1,
    )
    async with _client(app) as client:
        pending = asyncio.create_task(client.post("/score", json=body))

        # Capture the process-group leader through the backend's BARRIER, which
        # fires inside the worker's own scope.register (see _LongFfmpegVmaf).
        # Waited on a worker thread, so a busy event loop cannot make this
        # observation lose a race against the very timeout it is testing.
        assert await asyncio.to_thread(
            vmaf.launched.wait, 30.0
        ), "the long ffmpeg never started"
        pids = vmaf.pids
        assert pids, "the runner adopted no process group"

        timed_out = await pending
        assert timed_out.status_code == 504
        assert timed_out.json()["detail"]["error"] == "scoring_timeout"

        # The whole group is gone (killpg on an empty group raises).
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                os.killpg(pids[0], 0)
            except (ProcessLookupError, PermissionError):
                break
            await asyncio.sleep(0.05)
        else:  # pragma: no cover - only on a genuine leak
            pytest.fail(f"process group {pids[0]} survived the request timeout")

        assert _our_ffmpeg_children() == []

        # ...and the slot is genuinely free: the next request is served.
        started = time.perf_counter()
        after = await client.post("/score", json=body)
        assert after.status_code == 200, after.text
        assert time.perf_counter() - started < 5.0
