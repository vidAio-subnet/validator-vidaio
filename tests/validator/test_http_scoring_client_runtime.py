"""Production inference HTTP scoring is bound to the local release runtime."""

from __future__ import annotations

import copy
import hashlib
import json

import httpx
import pytest

from vidaio.scoring import ItemScore
from vidaio.services.protocol import (
    SCORER_CANONICAL_RUNTIME_MARKER_SHA256,
    SCORER_PIEAPP_WEIGHTS_SHA256,
    ScoreRequest,
    ScorerRuntimeMismatch,
    scorer_runtime_contract_from_healthz,
)
from vidaio.validator.inference import HttpScoringClient

IDENTITY = "vidaio-scorer/1+0123456789ab"


def _attestation() -> dict:
    return {
        "schema": "vidaio-payout-runtime/1",
        "release": {
            "manifest_verified": True,
            "manifest_sha256": "a" * 64,
            "release_version": "0.3.1",
            "source_sha256": "b" * 64,
            "runtime_sha256": "c" * 64,
            "marker_sha256": SCORER_CANONICAL_RUNTIME_MARKER_SHA256,
            "marker_verified": True,
        },
        "execution_policy": {
            "required_os": "linux",
            "required_arch": "amd64",
            "actual_os": "linux",
            "actual_arch": "amd64",
            "libc": "glibc/2.36",
            "torch_intraop_threads": "1",
            "torch_interop_threads": "1",
            "mkl_threads": "1",
            "openblas_threads": "1",
            "omp_dynamic": "FALSE",
            "mkl_dynamic": "FALSE",
            "mkl_cbwr": "COMPATIBLE",
            "aten_cpu_capability_override": "default",
            "actual_torch_intraop_threads": 1,
            "actual_torch_interop_threads": 1,
            "actual_torch_deterministic_algorithms": True,
            "actual_torch_deterministic_warn_only": False,
            "actual_torch_mkldnn_enabled": False,
            "actual_torch_mkldnn_deterministic": True,
            "actual_torch_nnpack_enabled": False,
            "actual_torch_cpu_capability": "NO AVX",
            "actual_openmp_threads": 1,
            "actual_mkl_threads": 1,
            "actual_mkl_cbwr": "COMPATIBLE",
            "actual_mkl_dynamic": False,
        },
        "payout_backends": {
            "ffmpeg": "ffmpeg/9.0",
            "ffprobe": "ffprobe/9.0",
            "libvmaf": "libvmaf/3.0.0",
            "pieapp": "pieapp-torch/piq/0.8.0:pieapp:cpu",
            "perceptual": "cpu-perceptual-checks/opencv/4.12.0:algorithm/2",
            "pieapp_weights": f"sha256:{SCORER_PIEAPP_WEIGHTS_SHA256}",
            "torch": "torch/2.8.0+cpu",
            "torchvision": "torchvision/0.23.0+cpu",
            "piq": "piq/0.8.0",
            "opencv": "opencv/4.12.0.88",
            "numpy": "numpy/2.2.6",
            "python": "cpython/3.13.15",
        },
    }


def _health(attestation: dict) -> dict:
    raw = json.dumps(
        attestation,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return {
        "scorer_version": IDENTITY,
        "runtime_commitment": {
            "digest": hashlib.sha256(raw).hexdigest(),
            "attestation": attestation,
        },
    }


def _request() -> ScoreRequest:
    return ScoreRequest(
        track="compression",
        challenge_id="challenge",
        item_id="item",
        miner_hotkey="miner",
        reference_path="/input",
        reference_digest="a" * 64,
        miner_input_path="/input",
        miner_input_digest="a" * 64,
        output_path="/output",
        output_digest="b" * 64,
        scorer_version=IDENTITY,
    )


async def test_inference_http_client_rejects_local_expected_contract_mismatch():
    expected_health = _health(_attestation())
    expected = scorer_runtime_contract_from_healthz(expected_health)
    moved = _attestation()
    moved["release"]["source_sha256"] = "f" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/healthz"
        return httpx.Response(200, json=_health(moved))

    client = HttpScoringClient(
        "http://scoring.test",
        1.0,
        expected_runtime_contract=expected,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ScorerRuntimeMismatch, match="attestation"):
            await client.scorer_identity()
    finally:
        await client._client.aclose()


async def test_inference_http_client_rejects_packet_backend_map_mismatch():
    health = _health(_attestation())
    expected = scorer_runtime_contract_from_healthz(health)
    moved_backends = copy.deepcopy(expected.backend_versions)
    moved_backends["torch"] = "torch/2.8.0+cu128"
    request = _request()
    packet = ItemScore(
        item_id=request.item_id,
        challenge_id=request.challenge_id,
        track=request.track,
        miner_hotkey=request.miner_hotkey,
        content_digest=request.output_digest,
        score=0.5,
        gate_passed=True,
        scorer_version=IDENTITY,
        backend_versions=moved_backends,
    ).to_json()

    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/healthz":
            return httpx.Response(200, json=health)
        assert http_request.url.path == "/score"
        return httpx.Response(
            200,
            json={
                "item_score_json": packet,
                "packet_digest": hashlib.sha256(packet.encode()).hexdigest(),
            },
        )

    client = HttpScoringClient(
        "http://scoring.test",
        1.0,
        expected_runtime_contract=expected,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(RuntimeError, match="backend_versions"):
            await client.score(request)
    finally:
        await client._client.aclose()
