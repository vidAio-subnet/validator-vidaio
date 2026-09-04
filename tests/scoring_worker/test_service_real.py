"""End-to-end /score over real media and the configured CPU PieAPP backend.

The focused compression fixture keeps the manipulation checks deterministic. PieAPP
uses the shipped CPU backend: an environment with the media extra must score, while
an intentionally small test environment must return the typed unconfigured response.
"""

from __future__ import annotations

import hashlib
import math
import shutil

import httpx
import pytest

from tests.scoring_worker.conftest import (
    FFMPEG,
    FFPROBE,
    ClipPair,
    requires_media_tools,
    score_request_body,
    sha256_file,
    worker_scorer_version,
)
from vidaio.scoring import PERCEPTUAL_GATE_NAMES, DeterministicFakeBackend, ItemScore
from vidaio.scoring.backends_real import (
    CanonicalizeExecutor,
    FfmpegVmafBackend,
    FfprobeBackend,
    PieAppTorchBackend,
    SECONDARY_VMAF_MODEL,
    detect_tool_versions,
)
from vidaio.scoring_worker import ScoringBackends, ScoringWorkerConfig, create_app
from vidaio.scoring_worker.service import real_backends
from vidaio.services.protocol import ScoreResponse

pytestmark = requires_media_tools


@pytest.fixture(scope="module")
def worker_config(tmp_path_factory: pytest.TempPathFactory) -> ScoringWorkerConfig:
    return ScoringWorkerConfig(
        work_dir=tmp_path_factory.mktemp("scoring-work"),
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        request_timeout=120.0,
        subprocess_timeout=60.0,
    )


@pytest.fixture(scope="module")
def backends(worker_config: ScoringWorkerConfig) -> ScoringBackends:
    primary = FfmpegVmafBackend(FFMPEG, work_dir=worker_config.work_dir, timeout=60.0)
    secondary = FfmpegVmafBackend(
        FFMPEG,
        model=SECONDARY_VMAF_MODEL,
        work_dir=worker_config.work_dir,
        timeout=60.0,
    )
    return ScoringBackends(
        probe=FfprobeBackend(FFPROBE, timeout=60.0),
        vmaf_primary=primary,
        vmaf_secondary=secondary,
        pieapp=PieAppTorchBackend(device="cpu"),
        perceptual=DeterministicFakeBackend(),
        canonicalizer=CanonicalizeExecutor(FFMPEG, timeout=60.0),
        versions=detect_tool_versions(
            FFMPEG, FFPROBE, vmaf_backend=primary, timeout=30.0
        ),
    )


@pytest.fixture()
async def client(worker_config: ScoringWorkerConfig, backends: ScoringBackends):
    app = create_app(worker_config, backends)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as c:
        yield c


def _compression_body(clips: ClipPair) -> dict:
    return score_request_body(
        track="compression",
        reference=clips.reference,
        reference_digest=clips.reference_digest,
        output=clips.candidate,
        output_digest=clips.candidate_digest,
        params={"vmaf_threshold": 90.0},
    )


async def test_compression_scores_end_to_end(
    client: httpx.AsyncClient, clips: ClipPair, worker_config: ScoringWorkerConfig
) -> None:
    resp = await client.post("/score", json=_compression_body(clips))
    assert resp.status_code == 200, resp.text
    response = ScoreResponse.model_validate(resp.json())
    assert (
        hashlib.sha256(response.item_score_json.encode("utf-8")).hexdigest()
        == response.packet_digest
    )
    item = ItemScore.from_json(response.item_score_json)
    assert item.gate_passed and not item.violations
    assert item.breakdown is not None and item.breakdown.kind == "compression"
    assert 0.0 < item.breakdown.vmaf <= 100.0
    assert 0.0 < item.breakdown.compression_rate < 1.0
    assert 0.0 < item.score <= 1.0
    # secondary-model run happened and agreed within the gate tolerance
    delta = item.metrics["vmaf_model_delta"]
    assert delta is not None and 0.0 <= delta <= 3.0
    assert item.metrics["vmaf_secondary"] is not None
    # provenance for the audit recompute: the WORKER's version, not the caller's
    assert item.scorer_version == worker_scorer_version(worker_config)
    assert set(item.backend_versions) == {"ffmpeg", "ffprobe", "libvmaf"}
    assert item.canonicalization_plan_digest is not None
    assert len(item.canonicalization_plan_digest) == 64
    assert item.content_digest == clips.candidate_digest
    assert item.skips == []


async def test_same_pair_scores_byte_identical_packets(
    client: httpx.AsyncClient, clips: ClipPair
) -> None:
    # The recomputability property: nothing time-, path- or host-dependent may
    # reach the packet, so re-scoring is byte-for-byte identical.
    first = await client.post("/score", json=_compression_body(clips))
    second = await client.post("/score", json=_compression_body(clips))
    assert first.status_code == second.status_code == 200
    assert first.json()["item_score_json"] == second.json()["item_score_json"]
    assert first.json()["packet_digest"] == second.json()["packet_digest"]


async def test_tampered_output_after_digest_is_422(
    client: httpx.AsyncClient, clips: ClipPair, tmp_path
) -> None:
    tampered = tmp_path / "cand-tampered.mp4"
    shutil.copyfile(clips.candidate, tampered)
    digest_before_tamper = sha256_file(tampered)
    with tampered.open("ab") as handle:
        handle.write(b"tampered-after-digest")
    body = score_request_body(
        track="compression",
        reference=clips.reference,
        reference_digest=clips.reference_digest,
        output=str(tampered),
        output_digest=digest_before_tamper,
    )
    resp = await client.post("/score", json=body)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "digest_mismatch"
    assert detail["field"] == "output"


async def test_upscaling_cpu_pieapp_or_typed_unconfigured_response(
    client: httpx.AsyncClient, clips: ClipPair, backends: ScoringBackends
) -> None:
    body = score_request_body(
        track="upscaling",
        reference=clips.reference,
        reference_digest=clips.reference_digest,
        output=clips.candidate,
        output_digest=clips.candidate_digest,
        params={"upscale_factor": 2},
    )
    resp = await client.post("/score", json=body)
    if backends.pieapp.version == "not-configured":
        assert resp.status_code == 501  # typed refusal, never a substituted score
        detail = resp.json()["detail"]
        assert detail["error"] == "backend_not_configured"
        assert "PieAPP" in detail["detail"]
        return

    assert backends.pieapp.device == "cpu"
    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.breakdown is not None and item.breakdown.kind == "upscaling"
    assert math.isfinite(item.breakdown.pieapp)
    assert item.metrics["pieapp"] == item.breakdown.pieapp
    assert item.pieapp_start_frame is not None
    assert item.skips == []


# --- the shipped CPU composition, plus the explicit diagnostic skip mode ---------------


@pytest.fixture()
def shipped_backends(worker_config: ScoringWorkerConfig) -> ScoringBackends:
    """Exactly what `real_backends()` composes, including CPU perceptual gates."""
    return real_backends(worker_config)


async def test_shipped_composition_required_mode_scores_on_a_cpu_host(
    worker_config: ScoringWorkerConfig,
    shipped_backends: ScoringBackends,
    clips: ClipPair,
) -> None:
    if shipped_backends.perceptual.version == "not-configured":
        pytest.skip("optional media/OpenCV dependency group is not installed")
    app = create_app(worker_config, shipped_backends)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker"
    ) as client:
        resp = await client.post("/score", json=_compression_body(clips))
    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.skips == []
    assert item.backend_versions["perceptual"].startswith("cpu-perceptual-checks/")
    assert (
        item.metrics["perceptual_config_digest"]
        == worker_config.perceptual_cpu.digest()
    )
    assert item.metrics["tone_manipulation_passed"] == "true"


async def test_shipped_composition_skip_mode_scores_with_audited_skips(
    worker_config: ScoringWorkerConfig,
    shipped_backends: ScoringBackends,
    clips: ClipPair,
) -> None:
    """The runnable local mode: REAL ffmpeg/libvmaf measurement, gates skipped
    consciously and recorded — no perceptual verdict is ever invented."""
    config = worker_config.model_copy(update={"perceptual_checks": "skip"})
    app = create_app(config, real_backends(config))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker"
    ) as client:
        resp = await client.post("/score", json=_compression_body(clips))
    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.gate_passed and not item.violations
    assert (
        item.breakdown is not None and 0.0 < item.breakdown.vmaf <= 100.0
    )  # real VMAF
    assert [skip.gate for skip in item.skips] == list(PERCEPTUAL_GATE_NAMES)
    for skip in item.skips:
        assert 'perceptual_checks="skip"' in skip.detail


async def test_healthz_reports_media_tools_and_work_dir(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["service"] == "scoring-worker"
    assert payload["status"] == "ok"
    assert payload["checks"] == {
        "work_dir_writable": True,
        "media_tools_present": True,
    }
