"""Full /score rounds on injected deterministic fakes — CI without media tools.

Passthrough mode (``canonicalizer=None``): no ffmpeg exists, the original files
stand in for the canonical ones and the packet records
``canonicalization_plan_digest=None`` (nothing was normalized — the omission is
honest). Digest verification still reads the real files on disk.
"""

from __future__ import annotations

import asyncio
import hashlib
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from prometheus_client import CollectorRegistry

from pydantic import ValidationError

from tests.scoring_worker.conftest import (
    RoleKeyedBackend,
    score_request_body,
)
from vidaio.scoring import (
    PERCEPTUAL_GATE_NAMES,
    ItemScore,
    MediaInfo,
    PerceptualCheckResult,
    ReasonCode,
    ScoringConfig,
    derive_pieapp_start_frame,
    score_upscaling,
    usable_frames,
)
from vidaio.scoring.backends_real import (
    PieAppTorchBackend,
    UnconfiguredPerceptualCheckBackend,
)
from vidaio.scoring_worker import (
    ScoringBackends,
    ScoringWorker,
    ScoringWorkerConfig,
    create_app,
)
from vidaio.scoring_worker.service import ApiServerFailed


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


@dataclass
class FakeWorld:
    config: ScoringWorkerConfig
    backends: ScoringBackends
    fake: RoleKeyedBackend
    reference: str
    reference_digest: str
    miner_input: str
    miner_input_digest: str
    output: str
    output_digest: str


@pytest.fixture()
def world(tmp_path: Path) -> FakeWorld:
    """Compression-shaped world: input == reference, candidate at half the bytes;
    the media map also carries an upscaling-shaped input (small payload)."""
    reference, reference_digest = _write(tmp_path / "ref.bin", b"R" * 10_000)
    output, output_digest = _write(tmp_path / "out.bin", b"O" * 5_000)
    miner_input, miner_input_digest = _write(tmp_path / "input.bin", b"I" * 1_000)
    # Keyed by ROLE: the pipeline measures private snapshots, never these paths.
    fake = RoleKeyedBackend(
        vmaf={("reference", "output"): 93.0},
        pieapp={("reference", "output"): 0.2},
        media={
            "reference": _media(10_000),
            "output": _media(5_000),
            # Compression shape: the miner's input IS the reference payload. The
            # upscaling test restates this role as the small degraded input.
            "miner_input": _media(10_000),
        },
    )
    config = ScoringWorkerConfig(
        backend="fake",
        work_dir=tmp_path / "work",
        request_timeout=10.0,
        subprocess_timeout=5.0,
    )
    backends = ScoringBackends(
        probe=fake,
        vmaf_primary=fake,
        vmaf_secondary=fake,  # same deterministic mapping -> model delta 0
        pieapp=fake.pieapp,
        perceptual=fake,
        canonicalizer=None,
        versions=fake.versions(),
    )
    return FakeWorld(
        config=config,
        backends=backends,
        fake=fake,
        reference=reference,
        reference_digest=reference_digest,
        miner_input=miner_input,
        miner_input_digest=miner_input_digest,
        output=output,
        output_digest=output_digest,
    )


def _client(
    world: FakeWorld,
    *,
    backends: ScoringBackends | None = None,
    scoring_config: ScoringConfig | None = None,
    config: ScoringWorkerConfig | None = None,
    registry: CollectorRegistry | None = None,
) -> httpx.AsyncClient:
    app = create_app(
        config if config is not None else world.config,
        backends if backends is not None else world.backends,
        scoring_config=scoring_config,
        registry=registry,
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker"
    )


def _compression_body(world: FakeWorld) -> dict:
    return score_request_body(
        track="compression",
        reference=world.reference,
        reference_digest=world.reference_digest,
        output=world.output,
        output_digest=world.output_digest,
    )


async def test_fake_compression_round_trip_and_metrics(world: FakeWorld) -> None:
    registry = CollectorRegistry()
    async with _client(world, registry=registry) as client:
        resp = await client.post("/score", json=_compression_body(world))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (
        hashlib.sha256(body["item_score_json"].encode("utf-8")).hexdigest()
        == body["packet_digest"]
    )
    item = ItemScore.from_json(body["item_score_json"])
    assert item.gate_passed and not item.violations
    assert item.breakdown is not None
    assert item.breakdown.compression_rate == pytest.approx(0.5)
    assert item.metrics["vmaf"] == 93.0
    assert item.metrics["vmaf_model_delta"] == 0.0
    assert item.metrics["vmaf_model_delta_basis"] == "miner_input"
    assert ("reference", "output") in world.fake.vmaf_calls
    assert ("miner_input", "output") in world.fake.vmaf_calls
    assert world.fake.perceptual_calls == [
        ("tone", "miner_input", "output"),
        ("grayscale", "miner_input", "output"),
        ("chroma", "miner_input", "output"),
    ]
    assert item.metrics["perceptual_gate_basis"] == "miner_input"
    assert item.canonicalization_plan_digest is None  # nothing was normalized
    assert item.pieapp_start_frame is None
    assert item.backend_versions == world.fake.versions()
    # prometheus wiring: one scored item, no gate failures
    scored = registry.get_sample_value(
        "scoring_worker_scorings_total",
        {"track": "compression", "outcome": "scored"},
    )
    assert scored == 1.0


async def test_model_delta_gate_uses_two_miner_input_runs(world: FakeWorld) -> None:
    primary = RoleKeyedBackend(
        vmaf={
            ("reference", "output"): 93.0,  # scored quality remains pristine-based
            ("miner_input", "output"): 96.0,
        },
        media={
            "reference": _media(10_000),
            "miner_input": _media(10_000),
            "output": _media(5_000),
        },
    )
    secondary = RoleKeyedBackend(
        vmaf={
            ("reference", "output"): 93.0,
            ("miner_input", "output"): 90.0,
        }
    )
    backends = ScoringBackends(
        probe=primary,
        vmaf_primary=primary,
        vmaf_secondary=secondary,
        pieapp=world.fake.pieapp,
        perceptual=world.fake,
        canonicalizer=None,
        versions={},
    )
    async with _client(world, backends=backends) as client:
        resp = await client.post("/score", json=_compression_body(world))
    assert resp.status_code == 200
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.metrics["vmaf"] == 93.0
    assert item.metrics["vmaf_model_delta"] == 6.0
    assert not item.gate_passed
    assert any(v.code is ReasonCode.VMAF_MODEL_DELTA_EXCEEDED for v in item.violations)
    assert primary.vmaf_calls == [
        ("reference", "output"),
        ("miner_input", "output"),
    ]
    assert secondary.vmaf_calls == [("miner_input", "output")]


async def test_fake_upscaling_full_round_with_derived_start_frame(
    world: FakeWorld,
) -> None:
    body = score_request_body(
        track="upscaling",
        reference=world.reference,
        reference_digest=world.reference_digest,
        output=world.output,
        output_digest=world.output_digest,
        miner_input=world.miner_input,
        miner_input_digest=world.miner_input_digest,
        params={"upscale_factor": 2, "content_length": 10.0},
        challenge_id="chal-up",
    )
    world.fake.set_media("miner_input", _media(1_000, width=160, height=120))
    async with _client(world) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.gate_passed and not item.violations
    assert item.breakdown is not None and item.breakdown.kind == "upscaling"
    # start frame is the verifier-recomputable derivation from the held-out
    # reference digest — and the fake recorded the exact call
    window = usable_frames(60, ScoringConfig().pieapp_sample_window)
    expected_start = derive_pieapp_start_frame(
        world.reference_digest, "chal-up", window
    )
    assert item.pieapp_start_frame == expected_start
    assert world.fake.pieapp_calls == [("reference", "output", expected_start)]
    expected = score_upscaling(pieapp=0.2, content_length=10.0, config=ScoringConfig())
    assert item.score == pytest.approx(expected.final)
    assert world.fake.perceptual_calls == [
        ("tone", "miner_input", "output"),
        ("grayscale", "miner_input", "output"),
        ("chroma", "miner_input", "output"),
    ]
    assert item.metrics["perceptual_gate_basis"] == "miner_input"


async def test_honest_upscaler_is_not_zeroed_by_pristine_perceptual_basis(
    world: FakeWorld,
) -> None:
    """Regression: the DAG may alter tone/chroma before the miner sees input.

    This backend models an honest output that matches the miner input closely
    enough to pass, while the same output would fail every check against the
    hidden pristine holdout. The launch worker must use the former basis.
    """

    class BasisSensitivePerceptual:
        name = "basis-sensitive"
        version = "1"

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def _check(self, reference: str, candidate: str) -> PerceptualCheckResult:
            roles = (RoleKeyedBackend.role(reference), RoleKeyedBackend.role(candidate))
            self.calls.append(roles)
            return PerceptualCheckResult(
                passed=roles == ("miner_input", "output"),
                measure=0.2 if roles[0] == "miner_input" else 2.0,
                limit=1.0,
                comparison="maximum",
            )

        check_tone_manipulation = _check
        check_color_grayscale = _check
        check_chroma_uv = _check

    perceptual = BasisSensitivePerceptual()
    backends = ScoringBackends(
        probe=world.fake,
        vmaf_primary=world.fake,
        vmaf_secondary=world.fake,
        pieapp=world.fake.pieapp,
        perceptual=perceptual,
        canonicalizer=None,
        versions=world.fake.versions(),
    )
    body = score_request_body(
        track="upscaling",
        reference=world.reference,
        reference_digest=world.reference_digest,
        output=world.output,
        output_digest=world.output_digest,
        miner_input=world.miner_input,
        miner_input_digest=world.miner_input_digest,
        params={"upscale_factor": 2, "content_length": 10.0},
        challenge_id="chal-up-basis",
    )
    world.fake.set_media("miner_input", _media(1_000, width=160, height=120))
    async with _client(world, backends=backends) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.gate_passed and item.score > 0.0
    assert perceptual.calls == [("miner_input", "output")] * 3


async def test_digest_mismatch_is_422(world: FakeWorld) -> None:
    body = _compression_body(world)
    body["output_digest"] = hashlib.sha256(b"someone else's bytes").hexdigest()
    async with _client(world) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "digest_mismatch"
    assert detail["field"] == "output"
    assert detail["actual"] == world.output_digest


async def test_missing_file_is_422(world: FakeWorld, tmp_path: Path) -> None:
    body = _compression_body(world)
    body["reference_path"] = str(tmp_path / "gone.bin")
    async with _client(world) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "file_missing"


async def test_unsupported_track_is_422(world: FakeWorld) -> None:
    body = _compression_body(world)
    body["track"] = "interpolation"
    async with _client(world) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "unsupported_track"


async def test_upscaling_with_unconfigured_pieapp_is_501(world: FakeWorld) -> None:
    backends = ScoringBackends(
        probe=world.fake,
        vmaf_primary=world.fake,
        vmaf_secondary=world.fake,
        # Keep the refusal seam independent of whichever optional packages are
        # installed on the test host.
        pieapp=PieAppTorchBackend(_backend_version="not-configured"),
        perceptual=world.fake,
        canonicalizer=None,
        versions=world.fake.versions(),
    )
    body = score_request_body(
        track="upscaling",
        reference=world.reference,
        reference_digest=world.reference_digest,
        output=world.output,
        output_digest=world.output_digest,
        params={"upscale_factor": 2},
    )
    async with _client(world, backends=backends) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 501
    assert resp.json()["detail"]["error"] == "backend_not_configured"


class _SlowVmaf:
    name = "slow-vmaf"
    version = "1"

    def compute(
        self, reference: str, candidate: str, *, deterministic_seed: int = 0
    ) -> float:
        time.sleep(0.5)
        return 90.0


async def test_request_timeout_is_typed_504(world: FakeWorld) -> None:
    config = world.config.model_copy(update={"request_timeout": 0.05})
    backends = ScoringBackends(
        probe=world.fake,
        vmaf_primary=_SlowVmaf(),
        vmaf_secondary=None,
        pieapp=world.fake.pieapp,
        perceptual=world.fake,
        canonicalizer=None,
        versions=world.fake.versions(),
    )
    async with _client(world, backends=backends, config=config) as client:
        resp = await client.post("/score", json=_compression_body(world))
    assert resp.status_code == 504
    assert resp.json()["detail"]["error"] == "scoring_timeout"


async def test_missing_secondary_vmaf_fails_closed(world: FakeWorld) -> None:
    backends = ScoringBackends(
        probe=world.fake,
        vmaf_primary=world.fake,
        vmaf_secondary=None,  # no second model run exists
        pieapp=world.fake.pieapp,
        perceptual=world.fake,
        canonicalizer=None,
        versions=world.fake.versions(),
    )
    async with _client(world, backends=backends) as client:
        resp = await client.post("/score", json=_compression_body(world))
    assert resp.status_code == 200  # a measured zero with reasons, not an error
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert not item.gate_passed
    assert item.score == 0.0
    assert any(v.code == ReasonCode.METRIC_MISSING for v in item.violations)


async def test_secondary_vmaf_disabled_by_config_records_skip(world: FakeWorld) -> None:
    backends = ScoringBackends(
        probe=world.fake,
        vmaf_primary=world.fake,
        vmaf_secondary=None,
        pieapp=world.fake.pieapp,
        perceptual=world.fake,
        canonicalizer=None,
        versions=world.fake.versions(),
    )
    scoring_config = ScoringConfig(require_secondary_vmaf=False)
    async with _client(
        world, backends=backends, scoring_config=scoring_config
    ) as client:
        resp = await client.post("/score", json=_compression_body(world))
    assert resp.status_code == 200
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.gate_passed  # explicitly disabled, not silently passed:
    assert [skip.gate for skip in item.skips] == ["vmaf_model_delta"]


# --- perceptual_checks: "required" (default) vs the auditable "skip" mode -------------


def _unconfigured_perceptual_backends(world: FakeWorld) -> ScoringBackends:
    """The shipped non-GPU composition: perceptual checks raise on every call."""
    return ScoringBackends(
        probe=world.fake,
        vmaf_primary=world.fake,
        vmaf_secondary=world.fake,
        pieapp=world.fake.pieapp,
        perceptual=UnconfiguredPerceptualCheckBackend(),
        canonicalizer=None,
        versions=world.fake.versions(),
    )


def test_perceptual_checks_defaults_to_required() -> None:
    assert ScoringWorkerConfig().perceptual_checks == "required"
    assert ScoringWorkerConfig().pieapp_device == "cpu"


def test_perceptual_checks_rejects_any_other_value() -> None:
    with pytest.raises(ValidationError):
        ScoringWorkerConfig(perceptual_checks="fake-them")


async def test_required_mode_without_perceptual_backend_is_an_honest_501(
    world: FakeWorld,
) -> None:
    """A missing required CPU perceptual backend produces a typed refusal."""
    async with _client(
        world, backends=_unconfigured_perceptual_backends(world)
    ) as client:
        resp = await client.post("/score", json=_compression_body(world))
    assert resp.status_code == 501
    detail = resp.json()["detail"]
    assert detail["error"] == "backend_not_configured"
    assert "perceptual" in detail["detail"].lower()


async def test_skip_mode_records_the_skips_and_never_fakes_a_verdict(
    world: FakeWorld,
) -> None:
    config = world.config.model_copy(update={"perceptual_checks": "skip"})
    async with _client(
        world, backends=_unconfigured_perceptual_backends(world), config=config
    ) as client:
        resp = await client.post("/score", json=_compression_body(world))
    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    # The rest of the pipeline really ran and really measured.
    assert item.gate_passed and not item.violations
    assert item.metrics["vmaf"] == 93.0
    assert item.score > 0.0
    # ...and the omission is IN the audit packet, naming the flag that caused it.
    assert [skip.gate for skip in item.skips] == list(PERCEPTUAL_GATE_NAMES)
    for skip in item.skips:
        assert 'perceptual_checks="skip"' in skip.detail
    # No perceptual value was invented anywhere in the packet.
    assert not any(
        v.code
        in (
            ReasonCode.TONE_MANIPULATION,
            ReasonCode.COLOR_GRAYSCALE,
            ReasonCode.CHROMA_UV_MANIPULATION,
        )
        for v in item.violations
    )


async def test_skip_mode_never_calls_the_perceptual_backend(world: FakeWorld) -> None:
    """A skipped gate must not touch the backend at all (not even to ignore it)."""

    class ExplodingPerceptual:
        def check_tone_manipulation(self, reference: str, candidate: str):
            raise AssertionError("skipped gate called the backend")

        check_color_grayscale = check_tone_manipulation
        check_chroma_uv = check_tone_manipulation

    backends = ScoringBackends(
        probe=world.fake,
        vmaf_primary=world.fake,
        vmaf_secondary=world.fake,
        pieapp=world.fake.pieapp,
        perceptual=ExplodingPerceptual(),
        canonicalizer=None,
        versions=world.fake.versions(),
    )
    config = world.config.model_copy(update={"perceptual_checks": "skip"})
    async with _client(world, backends=backends, config=config) as client:
        resp = await client.post("/score", json=_compression_body(world))
    assert resp.status_code == 200, resp.text


async def test_required_mode_with_a_working_backend_records_no_skips(
    world: FakeWorld,
) -> None:
    async with _client(world) as client:  # world.backends: perceptual == the fake
        resp = await client.post("/score", json=_compression_body(world))
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.gate_passed and item.skips == []


async def test_healthz_in_fake_mode_needs_no_media_tools(world: FakeWorld) -> None:
    config = world.config.model_copy(
        update={
            "ffmpeg_path": "/nonexistent/ffmpeg",
            "ffprobe_path": "/nonexistent/ffprobe",
        }
    )
    async with _client(world, config=config) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["checks"]["media_tools_present"] is True  # passthrough mode


def test_worker_refuses_fake_backend_without_injection(tmp_path: Path) -> None:
    raw = {
        "core": {"metrics_port": 0},
        "scoring_worker": {"backend": "fake", "work_dir": str(tmp_path / "w")},
    }
    with pytest.raises(ValueError, match="injected backends"):
        ScoringWorker(raw)


async def test_worker_lifecycle_serves_and_stops(
    world: FakeWorld, tmp_path: Path
) -> None:
    raw = {
        "core": {"metrics_port": 0},
        "scoring_worker": {
            "backend": "fake",
            "port": 0,
            "metrics_port": 0,
            "work_dir": str(tmp_path / "work"),
        },
    }
    worker = ScoringWorker(raw, backends=world.backends)
    serve_task = asyncio.create_task(worker.serve())
    await asyncio.sleep(0.2)  # uvicorn + health server up
    assert not serve_task.done()
    ok, payload = worker.health.health_payload()
    assert ok and payload["checks"]["http_api_serving"] is True
    worker.request_stop()
    await asyncio.wait_for(serve_task, timeout=5.0)


async def test_health_checks_are_stateless_across_threads(
    world: FakeWorld, tmp_path: Path
) -> None:
    """The HealthServer answers on its own thread: concurrent checks must not
    tread on each other (a shared fixed probe path used to make them flap)."""
    raw = {
        "core": {"metrics_port": 0},
        "scoring_worker": {
            "backend": "fake",
            "port": 0,
            "metrics_port": 0,
            "work_dir": str(tmp_path / "work"),
        },
    }
    worker = ScoringWorker(raw, backends=world.backends)
    results = await asyncio.gather(
        *(asyncio.to_thread(worker.health.health_payload) for _ in range(16))
    )
    assert all(ok for ok, _ in results)


async def test_uvicorn_failure_flips_health_and_requests_stop(
    world: FakeWorld, tmp_path: Path
) -> None:
    """A live process with a dead API must not report healthy: the supervisor
    would never replace it and every caller would just time out."""
    occupied = socket.socket()
    occupied.bind(("127.0.0.1", 0))
    occupied.listen(1)
    port = occupied.getsockname()[1]
    raw = {
        "core": {"metrics_port": 0},
        "scoring_worker": {
            "backend": "fake",
            "port": port,  # already bound -> uvicorn startup fails
            "metrics_port": 0,
            "work_dir": str(tmp_path / "work"),
        },
    }
    worker = ScoringWorker(raw, backends=world.backends)
    assert worker.health.health_payload()[0] is True  # healthy before we start
    try:
        with pytest.raises(ApiServerFailed):
            await asyncio.wait_for(worker.serve(), timeout=10.0)
    finally:
        occupied.close()

    assert worker.stopping.is_set()  # asked to be stopped, not left running
    ok, payload = worker.health.health_payload()
    assert ok is False
    assert payload["checks"]["http_api_serving"] is False
