"""The scratch budget must cover what a request GENERATES, not just what it copies.

Every input cap measures an ENCODED file. Scoring measures the DECODED one: both
sides are canonicalized to raw y4m before any metric runs, and raw video is three
to four orders of magnitude larger than its encoding. So a small, highly
compressed, long/high-resolution clip passes every input cap and then expands
into files far larger than the worker-wide limit it was supposedly held to.

These tests pin the whole chain that closes it:

  * the projection is arithmetic, not a guess — it matches real ffmpeg output
    exactly for real clips, so reserving from it is reserving the truth;
  * a request whose projection cannot fit is refused BEFORE ffmpeg starts, with
    the measured scratch usage proving no y4m was ever written;
  * "cannot fit ever" (413) and "cannot fit right now" (503 + Retry-After) stay
    distinguishable, because shedding the former sheds it forever;
  * the projection is enforced again while it expands, since a prediction about a
    file a miner produced can be wrong;
  * libvmaf's temp dirs live inside the request's own scratch, so they are
    covered by the same accounting, the same cleanup and the same startup sweep;
  * the budget returns to zero on every path.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.scoring_worker.conftest import (
    FFMPEG,
    FFPROBE,
    requires_media_tools,
    score_request_body,
    sha256_file,
)
from vidaio.scoring import DeterministicFakeBackend, ItemScore, MediaInfo
from vidaio.scoring.backends_real import (
    SECONDARY_VMAF_MODEL,
    CanonicalizationTooLarge,
    CanonicalizeExecutor,
    FfmpegVmafBackend,
    FfprobeBackend,
    MetricLogTooLarge,
    current_media_scratch,
    current_metric_log_limit,
    use_media_scratch,
    use_metric_log_limit,
)
from vidaio.scoring.canonicalize import (
    CANONICAL_PIX_FMT,
    build_canonicalization_plan,
)
from vidaio.scoring_worker import ScoringBackends, ScoringWorkerConfig, create_app
from vidaio.scoring_worker.inputs import (
    VMAF_LOG_FLOOR_BYTES,
    WORK_PREFIX,
    Y4M_FRAME_HEADER_BYTES,
    Y4M_HEADER_BYTES,
    ScoreRejected,
    projected_canonical_bytes,
    projected_frame_count,
    projected_metric_log_bytes,
    y4m_frame_bytes,
)

pytestmark = requires_media_tools


# --- fixtures ------------------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    path: str
    digest: str
    info: MediaInfo


def _make_clip(path: Path, *, size: str, rate: int, duration: float) -> Clip:
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={duration}",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
            "-y", str(path),
        ],
        check=True, capture_output=True, timeout=120,
    )
    info = FfprobeBackend(FFPROBE, timeout=60.0).probe(str(path))
    return Clip(path=str(path), digest=sha256_file(path), info=info)


@pytest.fixture(scope="module")
def tiny_clip(tmp_path_factory: pytest.TempPathFactory) -> Clip:
    """~1 s of 160x120 — the honest case; its y4m is a few hundred kilobytes."""
    return _make_clip(
        tmp_path_factory.mktemp("tiny") / "tiny.mp4",
        size="160x120", rate=10, duration=1,
    )


@pytest.fixture(scope="module")
def dense_clip(tmp_path_factory: pytest.TempPathFactory) -> Clip:
    """Small file, big decode: ~200 KB of h264 that becomes ~11 MB of y4m."""
    return _make_clip(
        tmp_path_factory.mktemp("dense") / "dense.mp4",
        size="320x240", rate=25, duration=4,
    )


def _work_dir_bytes(work_dir: Path) -> int:
    return sum(p.stat().st_size for p in work_dir.rglob("*") if p.is_file())


class _PeakScratchWatcher:
    """Samples the work dir while a request runs and keeps the high-water mark.

    Asserting on the work dir AFTER a request proves nothing: the scratch
    directory is deleted either way. The claim under test is that a refused
    expansion never reaches the volume at all, which is a statement about the
    peak, so the peak is what gets measured.
    """

    def __init__(self, work_dir: Path, *, interval: float = 0.005) -> None:
        self._work_dir = work_dir
        self._interval = interval
        self._stop = threading.Event()
        self.peak_bytes = 0
        self.seen_names: set[str] = set()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self._stop.is_set():
            if self._work_dir.is_dir():
                total = 0
                for path in self._work_dir.rglob("*"):
                    try:
                        if path.is_file():
                            total += path.stat().st_size
                            self.seen_names.add(path.name)
                    except OSError:  # the request is deleting it under us
                        continue
                self.peak_bytes = max(self.peak_bytes, total)
            self._stop.wait(self._interval)

    def __enter__(self) -> "_PeakScratchWatcher":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


class _BlockingVmaf:
    """Holds a request — and therefore its whole scratch reservation — open."""

    name = "blocking-vmaf"
    version = "1"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def compute(
        self, reference: str, candidate: str, *, deterministic_seed: int = 0
    ) -> float:
        self.entered.set()
        assert self.release.wait(60.0), "the blocking vmaf was never released"
        return 93.0


class _LogPathRecordingVmaf(FfmpegVmafBackend):
    """Real libvmaf, but it publishes where it put its JSON log."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_paths: list[str] = []

    def _argv(self, reference: str, candidate: str, log_path: str) -> list[str]:
        # Called INSIDE the backend's TemporaryDirectory, so this is a direct
        # observation of where that directory was created.
        self.log_paths.append(log_path)
        return super()._argv(reference, candidate, log_path)


class _UnderProjectingCanonicalizer(CanonicalizeExecutor):
    """A canonicalizer whose caller reserved far too little — the wrong-guess case.

    Stands in for any way the projection could be wrong about a file the miner
    produced. The point is that being wrong costs the volume the cap, not the disk.
    """

    def __init__(self, *args: Any, cap: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cap = cap

    def run(self, plan, timeout=None, *, output_path=None, max_output_bytes=None):
        return super().run(
            plan,
            timeout=timeout,
            output_path=output_path,
            max_output_bytes=self._cap if max_output_bytes is not None else None,
        )


def _real_world(
    tmp_path: Path,
    *,
    vmaf: Any = None,
    secondary: bool = False,
    canonicalizer: CanonicalizeExecutor | None = None,
    **limits: Any,
):
    """A worker on REAL media tools whose scratch ceilings are the test's own."""
    settings: dict[str, Any] = {
        "work_dir": tmp_path / "work",
        "ffmpeg_path": FFMPEG,
        "ffprobe_path": FFPROBE,
        "request_timeout": 180.0,
        "subprocess_timeout": 90.0,
        "max_concurrent": 2,
    }
    settings.update(limits)
    config = ScoringWorkerConfig(**settings)
    fake = DeterministicFakeBackend()
    backends = ScoringBackends(
        probe=FfprobeBackend(FFPROBE, timeout=60.0),
        vmaf_primary=vmaf
        if vmaf is not None
        else FfmpegVmafBackend(FFMPEG, work_dir=config.work_dir, timeout=60.0),
        vmaf_secondary=(
            FfmpegVmafBackend(
                FFMPEG,
                model=SECONDARY_VMAF_MODEL,
                work_dir=config.work_dir,
                timeout=60.0,
            )
            if secondary
            else None
        ),
        pieapp=fake.pieapp,
        perceptual=fake,
        canonicalizer=(
            canonicalizer
            if canonicalizer is not None
            else CanonicalizeExecutor(FFMPEG, timeout=90.0)
        ),
        versions={"ffmpeg": "test", "ffprobe": "test", "libvmaf": "test"},
    )
    return create_app(config, backends), config, backends


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker"
    )


def _body(clip: Clip) -> dict:
    return score_request_body(
        track="compression",
        reference=clip.path,
        reference_digest=clip.digest,
        output=clip.path,
        output_digest=clip.digest,
        params={"vmaf_threshold": 90.0},
    )


def _one_request_projection(clip: Clip, *, runs: int = 1) -> int:
    """Everything one request generates (ref, output, delta-input + logs)."""
    frames = projected_frame_count(clip.info)
    return (
        3 * projected_canonical_bytes("reference", clip.info)
        + projected_metric_log_bytes(frames=frames, runs=runs)
    )


# --- the projection is arithmetic, not a guess ----------------------------------------


@pytest.mark.parametrize(
    ("size", "rate", "duration"),
    [("160x120", 10, 1), ("320x240", 25, 2), ("176x144", 30, 1.5)],
)
def test_the_projection_matches_real_ffmpeg_y4m_output(
    tmp_path: Path, size: str, rate: int, duration: float
) -> None:
    """Reserving from the projection is only honest if the projection is exact.

    y4m is a header line then, per frame, ``FRAME\\n`` plus the raw planes — so
    the size is computable from the probed geometry alone. With the slack
    disabled the prediction must be exact up to the one variable-length field
    (the header), and with it the prediction must never come in under the truth.
    """
    clip = _make_clip(tmp_path / "src.mp4", size=size, rate=rate, duration=duration)
    out = tmp_path / "canon.y4m"
    CanonicalizeExecutor(FFMPEG, timeout=90.0).run(
        build_canonicalization_plan(clip.path, str(out))
    )
    actual = out.stat().st_size

    exact = projected_canonical_bytes("reference", clip.info, slack_frames=0)
    # The only slop is the header line, which the constant over-states.
    assert 0 <= exact - actual <= Y4M_HEADER_BYTES, (
        f"projected {exact} vs actual {actual} for {size}@{rate}"
    )
    # The shipped call (with CFR slack) is an upper bound, and a tight one.
    reserved = projected_canonical_bytes("reference", clip.info)
    assert reserved >= actual
    assert reserved - actual <= 3 * (
        Y4M_FRAME_HEADER_BYTES + y4m_frame_bytes(clip.info.width, clip.info.height)
    )


def test_frame_bytes_follow_the_pixel_format_plane_geometry() -> None:
    """1.5 B/px for 4:2:0, 2 for 4:2:2, 3 for 4:4:4, 1 for gray, x2 at 10 bits."""
    assert y4m_frame_bytes(320, 240, "yuv420p") == 320 * 240 * 3 // 2
    assert y4m_frame_bytes(320, 240, "yuv422p") == 320 * 240 * 2
    assert y4m_frame_bytes(320, 240, "yuv444p") == 320 * 240 * 3
    assert y4m_frame_bytes(320, 240, "gray") == 320 * 240
    assert y4m_frame_bytes(320, 240, "yuv420p10le") == 320 * 240 * 3
    # Odd dimensions round the chroma planes UP: a decoder never allocates a
    # partial chroma row, so under-counting here would under-reserve.
    assert y4m_frame_bytes(5, 3, "yuv420p") == 5 * 3 + 2 * (3 * 2)
    assert y4m_frame_bytes(1, 1, CANONICAL_PIX_FMT) == 1 + 2


def test_an_unknown_canonical_pixel_format_is_refused_not_guessed() -> None:
    with pytest.raises(ScoreRejected) as excinfo:
        y4m_frame_bytes(320, 240, "rgb24")
    assert excinfo.value.status_code == 422
    assert excinfo.value.payload["error"] == "unsupported_canonical_pix_fmt"


def test_projection_uses_the_frame_count_cfr_conversion_will_actually_produce() -> None:
    """A VFR container that stores 10 frames over a minute at 30 fps expands x180.

    Canonicalization runs ``-fps_mode cfr``, so those frames get DUPLICATED into
    the output. Trusting ``nb_frames`` would under-project exactly the file that
    expands the most.
    """
    sparse = MediaInfo(
        codec="h264", width=1920, height=1080, fps=30.0, frame_count=10,
        duration=60.0, byte_size=50_000, bit_depth=8, pix_fmt="yuv420p",
    )
    assert projected_frame_count(sparse) == 1800
    honest = MediaInfo(
        codec="h264", width=1920, height=1080, fps=30.0, frame_count=1800,
        duration=60.0, byte_size=50_000, bit_depth=8, pix_fmt="yuv420p",
    )
    assert projected_frame_count(honest) == 1800


def test_a_stream_whose_size_cannot_be_bounded_is_refused() -> None:
    """No geometry means no reservation, and no reservation means no expansion."""
    unbounded = MediaInfo(
        codec="h264", width=0, height=0, fps=0.0, frame_count=0,
        duration=0.0, byte_size=1, bit_depth=8, pix_fmt="yuv420p",
    )
    with pytest.raises(ScoreRejected) as excinfo:
        projected_canonical_bytes("output", unbounded)
    assert excinfo.value.status_code == 422
    assert excinfo.value.payload["error"] == "unprojectable_stream"
    assert excinfo.value.payload["field"] == "output"


def test_the_log_reservation_scales_with_frames_and_runs() -> None:
    small = projected_metric_log_bytes(frames=1, runs=1)
    assert small == VMAF_LOG_FLOOR_BYTES  # the floor covers the JSON envelope
    one_model = projected_metric_log_bytes(frames=100_000, runs=1)
    assert projected_metric_log_bytes(frames=100_000, runs=2) == 2 * one_model


# --- refusal BEFORE anything large is written -----------------------------------------


async def test_a_tiny_input_that_would_decode_huge_is_refused_before_expanding(
    tmp_path: Path, dense_clip: Clip
) -> None:
    """The whole point: pass every INPUT cap, then be refused on the DECODE.

    ~200 KB of h264 that becomes ~11 MB of y4m per side. The worker is given a
    scratch volume that comfortably holds the inputs and cannot hold the
    expansion, and must say so with a 413 — deterministically, because a request
    that exceeds the whole budget can never fit, so a 503 would shed it forever.
    """
    encoded = Path(dense_clip.path).stat().st_size
    projection = _one_request_projection(dense_clip)
    assert projection > 20 * encoded, "fixture is not compressive enough to test this"

    app, config, _ = _real_world(
        tmp_path,
        max_input_bytes=4 * encoded,
        max_request_bytes=4 * encoded,
        max_scratch_bytes=8 * encoded,  # holds every input, holds no y4m
    )
    assert 8 * encoded < projection

    with _PeakScratchWatcher(config.work_dir) as watcher:
        async with _client(app) as client:
            resp = await client.post("/score", json=_body(dense_clip))

    assert resp.status_code == 413, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "request_scratch_too_large"
    assert detail["kind"] == "canonicalization"
    assert detail["projected_bytes"] == projection
    assert detail["limit"] == 8 * encoded
    # Not one byte of y4m ever reached the volume: the peak the work dir ever
    # held is the three snapshots, nowhere near a single canonicalized side.
    assert watcher.peak_bytes <= 4 * encoded
    assert not any(name.endswith(".y4m") for name in watcher.seen_names)
    assert list(config.work_dir.iterdir()) == []
    assert app.state.scratch_budget.used_bytes == 0


async def test_a_request_that_fits_is_scored_normally(
    tmp_path: Path, tiny_clip: Clip
) -> None:
    """The bound must not be a wall: a legitimate clip still expands and scores."""
    app, config, _ = _real_world(tmp_path, secondary=True)
    async with _client(app) as client:
        resp = await client.post("/score", json=_body(tiny_clip))

    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.metrics["vmaf"] is not None
    assert item.canonicalization_plan_digest is not None
    assert app.state.scratch_budget.used_bytes == 0
    assert _work_dir_bytes(config.work_dir) == 0
    assert list(config.work_dir.iterdir()) == []


async def test_two_requests_whose_projections_collide_shed_the_second(
    tmp_path: Path, tiny_clip: Clip
) -> None:
    """Two requests that each fit must not expand into the volume together.

    503 + Retry-After, not 413: unlike an over-large projection this is
    transient, and the retry the response invites has to actually work.
    """
    vmaf = _BlockingVmaf()
    snapshots = 3 * Path(tiny_clip.path).stat().st_size
    projection = _one_request_projection(tiny_clip)
    app, config, _ = _real_world(
        tmp_path,
        vmaf=vmaf,
        max_input_bytes=snapshots,
        max_request_bytes=snapshots,
        # Exactly one request's worth of everything, plus slack for the
        # filesystem — so the first fits and the second cannot.
        max_scratch_bytes=snapshots + projection + 4096,
    )

    async with _client(app) as client:
        first = asyncio.create_task(client.post("/score", json=_body(tiny_clip)))
        assert await asyncio.to_thread(
            vmaf.entered.wait, 60.0
        ), "the first request never reached the metric"
        # The first request holds real y4m on the volume right now.
        assert app.state.scratch_budget.used_bytes > projection // 2
        assert _work_dir_bytes(config.work_dir) > 0

        second = await client.post("/score", json=_body(tiny_clip))
        assert second.status_code == 503, second.text
        detail = second.json()["detail"]
        assert detail["error"] == "scratch_budget_unavailable"
        assert detail["kind"] == "canonicalization"
        assert int(second.headers["retry-after"]) >= 1
        # Shed, not written: only the first request's files are on the volume.
        assert len(
            [p for p in config.work_dir.iterdir() if p.name.startswith(WORK_PREFIX)]
        ) == 1

        vmaf.release.set()
        assert (await first).status_code == 200
        # ...and the retry the 503 invited genuinely succeeds.
        third = await client.post("/score", json=_body(tiny_clip))

    assert third.status_code == 200, third.text
    assert app.state.scratch_budget.used_bytes == 0
    assert _work_dir_bytes(config.work_dir) == 0


# --- enforcement DURING the expansion --------------------------------------------------


def test_a_writer_that_blows_past_its_cap_is_killed_mid_write(tmp_path: Path) -> None:
    """The reservation is a bound, not a hope: the process group dies at the cap.

    Driven by a paced writer rather than ffmpeg so the kill path — not the
    post-exit check — is what is under test: the file must stop growing long
    before the writer intended to finish.
    """
    out = tmp_path / "grows.y4m"
    script = (
        "import sys, time\n"
        "with open(sys.argv[1], 'wb') as handle:\n"
        "    for _ in range(400):\n"
        "        handle.write(b'x' * 65536)\n"
        "        handle.flush()\n"
        "        time.sleep(0.02)\n"
    )
    plan = [sys.executable, "-c", script, str(out)]

    with pytest.raises(CanonicalizationTooLarge) as excinfo:
        CanonicalizeExecutor(FFMPEG, timeout=60.0).run(
            plan, output_path=str(out), max_output_bytes=200_000
        )

    assert excinfo.value.limit_bytes == 200_000
    assert excinfo.value.observed_bytes > 200_000
    # Killed early: the writer wanted 26 MB and never got near it.
    stopped_at = out.stat().st_size
    assert stopped_at < 4_000_000, stopped_at
    # And it really is dead — the file is not still growing.
    time.sleep(0.3)
    assert out.stat().st_size == stopped_at


def test_a_run_that_finishes_between_two_polls_is_still_caught(
    tmp_path: Path, tiny_clip: Clip
) -> None:
    """Polling bounds the overshoot; the post-exit stat makes the verdict exact."""
    out = tmp_path / "canon.y4m"
    with pytest.raises(CanonicalizationTooLarge) as excinfo:
        CanonicalizeExecutor(FFMPEG, timeout=90.0).run(
            build_canonicalization_plan(tiny_clip.path, str(out)),
            output_path=str(out),
            max_output_bytes=1024,
        )
    assert excinfo.value.limit_bytes == 1024
    assert excinfo.value.observed_bytes > 1024
    assert excinfo.value.output_path == str(out)


async def test_a_wrong_projection_costs_the_cap_and_not_the_disk(
    tmp_path: Path, tiny_clip: Clip
) -> None:
    """If the estimate were wrong the request is refused, not the volume filled."""
    app, config, _ = _real_world(
        tmp_path,
        canonicalizer=_UnderProjectingCanonicalizer(FFMPEG, timeout=90.0, cap=4096),
    )
    with _PeakScratchWatcher(config.work_dir) as watcher:
        async with _client(app) as client:
            resp = await client.post("/score", json=_body(tiny_clip))

    assert resp.status_code == 413, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "canonicalized_output_too_large"
    assert detail["field"] == "reference"
    assert detail["projected_bytes"] == 4096
    assert detail["observed_bytes"] > 4096
    # The partial y4m is gone and the budget is whole again.
    assert list(config.work_dir.iterdir()) == []
    assert app.state.scratch_budget.used_bytes == 0
    assert watcher.peak_bytes < _one_request_projection(tiny_clip)


# --- the log reservation is ENFORCED, not just taken ----------------


def test_the_vmaf_log_bound_is_enforced_when_installed(
    tmp_path: Path, tiny_clip: Clip
) -> None:
    """A reservation that is not enforced is an estimate.

    The worker reserves scratch for the JSON log from a 1 KiB/frame heuristic;
    a libvmaf configuration that logs more (a wider model's feature set) used to
    write past that reservation unchecked for the length of the clip. With the
    per-run bound installed the same watchdog machinery that bounds a
    canonicalization holds the log to it — typed error, run refused.
    """
    backend = FfmpegVmafBackend(FFMPEG, work_dir=tmp_path, timeout=60.0)
    with use_metric_log_limit(64):
        assert current_metric_log_limit() == 64
        with pytest.raises(MetricLogTooLarge) as excinfo:
            backend.compute(tiny_clip.path, tiny_clip.path)
    assert excinfo.value.limit_bytes == 64
    assert excinfo.value.observed_bytes > 64
    assert current_metric_log_limit() is None  # restored on exit
    # The partial log died with the backend's temp dir: nothing left behind.
    assert list(tmp_path.iterdir()) == []
    # Without an installed bound (composition-time probes, fake mode) the same
    # run is the historical unbounded one — and an honest log passes anyway.
    assert 0.0 <= backend.compute(tiny_clip.path, tiny_clip.path) <= 100.0


async def test_a_log_that_outgrows_its_reservation_is_a_typed_413(
    tmp_path: Path, tiny_clip: Clip, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the reservation the worker took is the bound the run gets.

    The estimate constants are shrunk so the honest clip's REAL log overruns
    its reservation — standing in for any libvmaf output wider than the
    heuristic. The refusal is the deterministic 413 (a retry would reserve and
    overrun identically), the budget is whole again and the volume is clean.
    """
    monkeypatch.setattr("vidaio.scoring_worker.inputs.VMAF_LOG_FLOOR_BYTES", 32)
    monkeypatch.setattr("vidaio.scoring_worker.inputs.VMAF_LOG_BYTES_PER_FRAME", 1)
    app, config, _ = _real_world(tmp_path)
    async with _client(app) as client:
        resp = await client.post("/score", json=_body(tiny_clip))

    assert resp.status_code == 413, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "metric_log_too_large"
    assert detail["observed_bytes"] > detail["reserved_bytes"]
    # Refused AND released: budget back to zero, nothing left on the volume.
    assert app.state.scratch_budget.used_bytes == 0
    assert list(config.work_dir.iterdir()) == []


async def test_normal_runs_are_unaffected_by_the_enforced_log_bound(
    tmp_path: Path, tiny_clip: Clip
) -> None:
    """The shipped reservation comfortably holds a real log — enforcement is
    invisible to honest requests, secondary model run included."""
    app, config, _ = _real_world(tmp_path, secondary=True)
    async with _client(app) as client:
        resp = await client.post("/score", json=_body(tiny_clip))
    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.metrics["vmaf"] is not None
    assert item.metrics["vmaf_secondary"] is not None
    assert app.state.scratch_budget.used_bytes == 0


# --- libvmaf's own scratch belongs to the request --------------------------------------


def test_the_vmaf_backend_puts_its_temp_dirs_where_the_caller_says(
    tmp_path: Path,
) -> None:
    """Backends are shared across requests, so placement is thread-local, not state."""
    backend = FfmpegVmafBackend(FFMPEG, work_dir=tmp_path / "worker")
    assert backend.scratch_root() == str(tmp_path / "worker")
    assert current_media_scratch() is None

    request_scratch = tmp_path / "score-abc" / "metrics"
    with use_media_scratch(request_scratch):
        assert backend.scratch_root() == str(request_scratch)
        with use_media_scratch(None):  # nested override wins, then restores
            assert backend.scratch_root() == str(tmp_path / "worker")
        assert backend.scratch_root() == str(request_scratch)
    assert backend.scratch_root() == str(tmp_path / "worker")


async def test_libvmaf_logs_land_inside_the_request_scratch_and_die_with_it(
    tmp_path: Path, tiny_clip: Clip
) -> None:
    """Accounted, cleaned up and swept only if they live where everything else does.

    The backend is deliberately configured with a DIFFERENT work dir, so the
    assertion is about the request installing its own scratch and not about the
    backend happening to be pointed at the right place already.
    """
    recorder = _LogPathRecordingVmaf(FFMPEG, work_dir=tmp_path / "elsewhere")
    app, config, _ = _real_world(tmp_path, vmaf=recorder)
    async with _client(app) as client:
        resp = await client.post("/score", json=_body(tiny_clip))
    assert resp.status_code == 200, resp.text

    assert recorder.log_paths, "libvmaf never ran"
    for log_path in recorder.log_paths:
        relative = Path(log_path).relative_to(config.work_dir)
        # <work_dir>/score-XXXX/metrics/vmaf-YYYY/vmaf.json
        assert relative.parts[0].startswith(WORK_PREFIX)
        assert relative.parts[1] == "metrics"
        assert relative.parts[2].startswith("vmaf-")
        assert not Path(log_path).exists()  # died with the request directory
    assert not (tmp_path / "elsewhere").exists()  # never used the shared work dir
    assert list(config.work_dir.iterdir()) == []
    assert app.state.scratch_budget.used_bytes == 0


# --- the budget returns to zero on every path ------------------------------------------


@pytest.mark.parametrize(
    "corrupt_field", ["reference_digest", "output_digest", None]
)
async def test_the_budget_returns_to_zero_on_every_path(
    tmp_path: Path, tiny_clip: Clip, corrupt_field: str | None
) -> None:
    app, config, _ = _real_world(tmp_path)
    body = _body(tiny_clip)
    if corrupt_field is not None:
        body[corrupt_field] = "0" * 64
    async with _client(app) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == (200 if corrupt_field is None else 422), resp.text
    assert app.state.scratch_budget.used_bytes == 0
    assert list(config.work_dir.iterdir()) == []


async def test_an_abandoned_request_gives_its_expansion_back(
    tmp_path: Path, tiny_clip: Clip
) -> None:
    """A withdrawn caller must release the y4m reservation, not strand it.

    Cancellation and ``request_timeout`` are the same code path in the worker
    (both cancel the request's :class:`MediaProcessScope` and let the thread
    unwind); cancellation is the one a test can trigger deterministically.

    The release is bound to the SCRATCH DIRECTORY's lifetime, not to the HTTP
    response, so it lands when the worker thread genuinely unwinds — which is
    the only moment at which the bytes are actually off the volume. Until then
    the budget must keep counting them, or a second request would be admitted
    against space that is still occupied.
    """
    vmaf = _BlockingVmaf()
    app, config, _ = _real_world(tmp_path, vmaf=vmaf)
    async with _client(app) as client:
        pending = asyncio.create_task(client.post("/score", json=_body(tiny_clip)))
        assert await asyncio.to_thread(vmaf.entered.wait, 60.0)
        held = app.state.scratch_budget.used_bytes
        assert held > 0

        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending
        # The worker thread is still inside the metric, so the y4m is still on
        # the volume and the budget still owes it.
        assert app.state.scratch_budget.used_bytes == held
        vmaf.release.set()

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and app.state.scratch_budget.used_bytes:
        await asyncio.sleep(0.02)
    assert app.state.scratch_budget.used_bytes == 0
    assert _work_dir_bytes(config.work_dir) == 0
    assert list(config.work_dir.iterdir()) == []


# --- config: the ceilings must nest ----------------------------------------------------


def test_the_scratch_ceilings_must_nest_widest_last() -> None:
    with pytest.raises(ValueError, match="max_request_bytes must be >="):
        ScoringWorkerConfig(max_input_bytes=100, max_request_bytes=50)
    with pytest.raises(ValueError, match="max_scratch_bytes must be >= max_request_bytes"):
        ScoringWorkerConfig(
            max_input_bytes=50, max_request_bytes=100, max_scratch_bytes=80
        )
    with pytest.raises(
        ValueError, match="max_scratch_bytes must be >= max_request_scratch_bytes"
    ):
        ScoringWorkerConfig(
            max_input_bytes=50,
            max_request_bytes=100,
            max_request_scratch_bytes=1_000,
            max_scratch_bytes=500,
        )
    with pytest.raises(
        ValueError, match="max_request_scratch_bytes must be >= max_request_bytes"
    ):
        ScoringWorkerConfig(
            max_input_bytes=50,
            max_request_bytes=400,
            max_request_scratch_bytes=200,
            max_scratch_bytes=10_000,
        )


def test_an_unset_per_request_ceiling_clamps_to_the_worker_budget() -> None:
    """"Bigger than the whole volume" must be a 413, however the budget was tuned.

    An operator who only lowers max_scratch_bytes has not thereby raised the
    per-request allowance above it — otherwise an over-large request would be
    shed (503) on every retry instead of refused once.
    """
    config = ScoringWorkerConfig(
        max_input_bytes=1_000, max_request_bytes=2_000, max_scratch_bytes=5_000
    )
    assert config.max_request_scratch_bytes > config.max_scratch_bytes
    assert config.request_scratch_ceiling == 5_000


def test_the_shipped_defaults_hold_max_concurrent_full_requests() -> None:
    config = ScoringWorkerConfig()
    assert (
        config.max_scratch_bytes
        >= config.max_concurrent * config.request_scratch_ceiling
    )
    assert config.request_scratch_ceiling >= config.max_request_bytes
    assert config.max_request_bytes >= config.max_input_bytes
    # The expansion allowance is the majority of a request's scratch — that is
    # the whole reason the key exists.
    assert (
        config.request_scratch_ceiling - config.max_request_bytes
        > config.max_request_bytes
    )


def test_projection_scale_matches_the_documented_expansion_ratio() -> None:
    """A 30 MB ten-minute 4K clip really is hundreds of gigabytes of y4m."""
    uhd = MediaInfo(
        codec="h265", width=3840, height=2160, fps=60.0, frame_count=36_000,
        duration=600.0, byte_size=30 * 1024 * 1024, bit_depth=8, pix_fmt="yuv420p",
    )
    projected = projected_canonical_bytes("output", uhd)
    assert projected > 400 * 1024**3
    # ...and it is inside every input cap, which is exactly the bug.
    defaults = ScoringWorkerConfig()
    assert uhd.byte_size < defaults.max_input_bytes
    assert projected > defaults.max_scratch_bytes
    assert projected == Y4M_HEADER_BYTES + (36_000 + 2) * (
        Y4M_FRAME_HEADER_BYTES + 3840 * 2160 * 3 // 2
    )
