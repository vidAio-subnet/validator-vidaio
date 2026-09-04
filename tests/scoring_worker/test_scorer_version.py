"""The worker stamps ITS OWN scorer version and refuses to answer for another.

``ScoreRequest.scorer_version`` is an assertion about which scorer the caller
expects, never an instruction. Blindly echoing it back into the packet (the old
behaviour) let any caller — or anything on the path — mint packets that claim to
come from a scorer that never ran, which the audit bundle later cross-checks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from tests.scoring_worker.conftest import RoleKeyedBackend, score_request_body
from vidaio.scoring import CpuPerceptualConfig, ItemScore, MediaInfo, ScoringConfig
from vidaio.scoring_worker import (
    ScoringBackends,
    ScoringWorkerConfig,
    check_scorer_version,
    create_app,
    effective_scorer_version,
    scorer_identity_digest,
)
from vidaio.scoring_worker.inputs import ScoreRejected
from vidaio.scoring_worker.service import SCORING_PIPELINE_VERSION


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


@pytest.fixture()
def world(tmp_path: Path):
    reference = tmp_path / "ref.bin"
    output = tmp_path / "out.bin"
    reference.write_bytes(b"R" * 10_000)
    output.write_bytes(b"O" * 5_000)
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
        reference=str(reference),
        reference_digest=hashlib.sha256(reference.read_bytes()).hexdigest(),
        output=str(output),
        output_digest=hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    return config, backends, body


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker"
    )


# --- what the worker stamps ------------------------------------------------------------


def test_effective_version_is_the_name_plus_a_config_digest() -> None:
    config = ScoringWorkerConfig(scorer_version="vidaio-scorer/7")
    scoring = ScoringConfig()
    version = effective_scorer_version(config, scoring)
    name, _, digest = version.partition("+")
    assert name == "vidaio-scorer/7"
    assert digest == scorer_identity_digest(config, scoring)[:12]
    assert len(digest) == 12


def test_pipeline_contract_version_pins_miner_input_perceptual_basis() -> None:
    assert SCORING_PIPELINE_VERSION == 4


def test_version_moves_when_a_scoring_lever_moves() -> None:
    config = ScoringWorkerConfig()
    base = effective_scorer_version(config, ScoringConfig())
    changed = effective_scorer_version(config, ScoringConfig(vmaf_model_delta_max=2.0))
    assert base != changed


def test_version_moves_when_a_measurement_lever_moves() -> None:
    scoring = ScoringConfig()
    base = effective_scorer_version(ScoringWorkerConfig(), scoring)
    skipped = effective_scorer_version(
        ScoringWorkerConfig(perceptual_checks="skip"), scoring
    )
    other_model = effective_scorer_version(
        ScoringWorkerConfig(vmaf_model_secondary="version=vmaf_4k_v0.6.1"), scoring
    )
    other_perceptual = effective_scorer_version(
        ScoringWorkerConfig(
            perceptual_cpu=CpuPerceptualConfig(tone_mean_delta_max=0.07)
        ),
        scoring,
    )
    other_device = effective_scorer_version(
        ScoringWorkerConfig(pieapp_device="cuda"), scoring
    )
    assert base != skipped != other_model
    assert base != other_model
    assert base != other_perceptual
    assert base != other_device


def test_version_ignores_levers_that_cannot_change_a_score() -> None:
    """Two identically-scoring deployments must not refuse each other's work."""
    scoring = ScoringConfig()
    base = effective_scorer_version(ScoringWorkerConfig(), scoring)
    elsewhere = effective_scorer_version(
        ScoringWorkerConfig(
            port=9999,
            metrics_port=0,
            work_dir=Path("/somewhere/else"),
            max_concurrent=8,
            request_timeout=17.0,
            queue_wait_timeout_seconds=1.5,
        ),
        scoring,
    )
    assert base == elsewhere


# --- the request contract --------------------------------------------------------------


def test_check_accepts_absence_and_agreement() -> None:
    check_scorer_version(None, "scorer/1+abc")
    check_scorer_version("", "scorer/1+abc")
    check_scorer_version("   ", "scorer/1+abc")
    check_scorer_version("scorer/1+abc", "scorer/1+abc")


def test_check_rejects_disagreement_with_409() -> None:
    with pytest.raises(ScoreRejected) as excinfo:
        check_scorer_version("scorer/9+zzz", "scorer/1+abc")
    assert excinfo.value.status_code == 409
    assert excinfo.value.payload["error"] == "scorer_version_mismatch"
    assert excinfo.value.payload["requested"] == "scorer/9+zzz"
    assert excinfo.value.payload["scorer_version"] == "scorer/1+abc"


async def test_absent_scorer_version_gets_the_workers_own(world) -> None:
    config, backends, body = world
    assert body["scorer_version"] is None
    async with _client(create_app(config, backends)) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.scorer_version == effective_scorer_version(config, ScoringConfig())


async def test_agreeing_scorer_version_is_accepted_and_restamped(world) -> None:
    config, backends, body = world
    expected = effective_scorer_version(config, ScoringConfig())
    body["scorer_version"] = expected
    async with _client(create_app(config, backends)) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.scorer_version == expected


async def test_disagreeing_scorer_version_is_409_and_scores_nothing(world) -> None:
    config, backends, body = world
    body["scorer_version"] = "some-other-scorer/1"
    fake = backends.probe
    async with _client(create_app(config, backends)) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "scorer_version_mismatch"
    assert detail["requested"] == "some-other-scorer/1"
    assert detail["scorer_version"] == effective_scorer_version(config, ScoringConfig())
    # Refused before any work: nothing was even probed.
    assert fake.probed_paths == []


async def test_healthz_publishes_the_scorer_version(world) -> None:
    config, backends, _body = world
    async with _client(create_app(config, backends)) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["scorer_version"] == effective_scorer_version(
        config, ScoringConfig()
    )
    runtime = resp.json()["runtime_commitment"]
    assert len(runtime["digest"]) == 64
    assert runtime["attestation"]["schema"] == "vidaio-payout-runtime/1"


async def test_scorer_version_follows_the_injected_scoring_config(world) -> None:
    """The stamp must describe the config that actually scored, not the default."""
    config, backends, body = world
    scoring = ScoringConfig(vmaf_model_delta_max=1.0)
    app = create_app(config, backends, scoring_config=scoring)
    async with _client(app) as client:
        resp = await client.post("/score", json=body)
    assert resp.status_code == 200, resp.text
    item = ItemScore.from_json(resp.json()["item_score_json"])
    assert item.scorer_version == effective_scorer_version(config, scoring)
    assert item.scorer_version != effective_scorer_version(config, ScoringConfig())
