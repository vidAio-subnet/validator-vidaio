"""HttpScoringClient wire behavior against a mock scoring worker."""

from __future__ import annotations

import copy
import hashlib
import json

import httpx
import pytest

from vidaio.competition import LifecycleEngine, migrate
from vidaio.competition import repository as repo
from vidaio.competition.interfaces import BatchItem, BatchOutput
from vidaio.competition.orchestrator import HttpScoringClient, ScoringClientError
from vidaio.core.db import connect
from vidaio.services.protocol import (
    SCORER_RUNTIME_BACKEND_KEY,
    SCORER_RUNTIME_BACKEND_PREFIX,
    SCORER_CANONICAL_RUNTIME_MARKER_SHA256,
    SCORER_RUNTIME_COMMITMENT_SCHEMA,
    SCORER_PIEAPP_WEIGHTS_SHA256,
    ScoreRequest,
    ScorerRuntimeMismatch,
    ScorerRuntimeUnavailable,
    scorer_runtime_contract_from_healthz,
)

from orchestrator_support import COMMITMENT_ROOT, START, T0, build_manifest


SCORER_IDENTITY = "v1.0.0"
PAYOUT_BACKENDS = {
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
}
RUNTIME_ATTESTATION = {
    "schema": SCORER_RUNTIME_COMMITMENT_SCHEMA,
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
    "payout_backends": PAYOUT_BACKENDS,
}


def _canonical_digest(value) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


RUNTIME_DIGEST = _canonical_digest(RUNTIME_ATTESTATION)
EXPECTED_BACKENDS = {
    **PAYOUT_BACKENDS,
    SCORER_RUNTIME_BACKEND_KEY: SCORER_RUNTIME_BACKEND_PREFIX + RUNTIME_DIGEST,
}


def _health_payload(*, digest: str = RUNTIME_DIGEST) -> dict:
    return {
        "scorer_version": SCORER_IDENTITY,
        "runtime_commitment": {
            "digest": digest,
            "attestation": copy.deepcopy(RUNTIME_ATTESTATION),
        },
    }


def _health_for_attestation(attestation: dict) -> dict:
    return {
        "scorer_version": SCORER_IDENTITY,
        "runtime_commitment": {
            "digest": _canonical_digest(attestation),
            "attestation": attestation,
        },
    }


EXPECTED_RUNTIME_CONTRACT = scorer_runtime_contract_from_healthz(_health_payload())


def _packet_json(**updates) -> str:
    packet = {
        "item_id": "0" * 64,
        "challenge_id": "chal-x",
        "track": "compression",
        "miner_hotkey": "hk-a",
        "content_digest": "1" * 64,
        "score": 0.5,
        "gate_passed": True,
        "violations": [],
        "skips": [],
        "breakdown": None,
        "metrics": {},
        "scorer_version": SCORER_IDENTITY,
        "backend_versions": dict(EXPECTED_BACKENDS),
        "canonicalization_plan_digest": None,
        "pieapp_start_frame": None,
        "scoring_config_digest": "a" * 64,
    }
    packet.update(updates)
    return json.dumps(packet, sort_keys=True, separators=(",", ":"))


@pytest.fixture
def seeded_conn():
    conn = connect(":memory:")
    migrate(conn)
    engine = LifecycleEngine()
    manifest = build_manifest()
    engine.create_competition(conn, manifest, T0)
    engine.mark_commitment_anchored(conn, manifest.competition_id, COMMITMENT_ROOT, T0)
    engine.tick(conn, START)
    contender_id = repo.enroll_contender(
        conn,
        manifest.competition_id,
        hotkey="hk-a",
        repo_url="local://hk-a",
        commit_sha="1a" * 20,
        tree_sha="2a" * 20,
        stake=1000.0,
        now=START,
    )
    item_id = repo.add_evaluation_item(
        conn,
        manifest.competition_id,
        item_index=0,
        input_sha256="0" * 64,
        input_bytes=4096,
        threshold_commitment="f" * 64,
        challenge_id="chal-x",
        length_seconds=10.0,
        now=START,
    )
    yield conn, manifest.competition_id, contender_id, item_id
    conn.close()


def _wire(
    seeded_conn,
    tmp_path,
    handler,
    *,
    health_payload: dict | None = None,
) -> tuple[HttpScoringClient, BatchItem, BatchOutput]:
    conn, cid, contender_id, item_id = seeded_conn

    def routed(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json=health_payload or _health_payload())
        return handler(request)

    client = HttpScoringClient(
        "http://scoring.test",
        conn,
        inputs_dir=tmp_path / "in",
        outputs_dir=tmp_path / "out",
        transport=httpx.MockTransport(routed),
        expected_runtime_contract=EXPECTED_RUNTIME_CONTRACT,
    )
    item = BatchItem(
        item_id=item_id, item_index=0, input_sha256="0" * 64, input_bytes=4096
    )
    output = BatchOutput(item_id=item_id, output_sha256="1" * 64, output_bytes=512)
    return client, item, output


def test_returns_verbatim_packet_and_verifies_digest(seeded_conn, tmp_path):
    conn, cid, contender_id, item_id = seeded_conn
    packet_json = _packet_json()

    def handler(request: httpx.Request) -> httpx.Response:
        body = ScoreRequest.model_validate_json(request.content)
        # The client sent the full identity + paths + digests of the artifacts.
        assert body.challenge_id == "chal-x"
        assert body.miner_hotkey == "hk-a"
        assert body.item_id == "0" * 64  # scoring_item_id defaults to input sha
        assert body.output_digest == "1" * 64
        assert body.scorer_version == "v1.0.0"
        return httpx.Response(
            200,
            json={
                "item_score_json": packet_json,
                "packet_digest": hashlib.sha256(packet_json.encode()).hexdigest(),
            },
        )

    client, item, output = _wire(seeded_conn, tmp_path, handler)
    packet = client.score_item(cid, contender_id, item, output)
    assert packet.packet_bytes == packet_json.encode()
    assert packet.contender_id == contender_id and packet.item_id == item_id


def test_rejects_packet_with_mismatched_digest(seeded_conn, tmp_path):
    conn, cid, contender_id, item_id = seeded_conn

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"item_score_json": "{}", "packet_digest": "a" * 64}
        )

    client, item, output = _wire(seeded_conn, tmp_path, handler)
    with pytest.raises(ScoringClientError, match="do not hash"):
        client.score_item(cid, contender_id, item, output)


def test_http_error_is_typed(seeded_conn, tmp_path):
    conn, cid, contender_id, item_id = seeded_conn

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="worker down")

    client, item, output = _wire(seeded_conn, tmp_path, handler)
    with pytest.raises(ScoringClientError, match="call failed"):
        client.score_item(cid, contender_id, item, output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("challenge_id", "chal-moved"),
        ("item_id", "item-moved"),
        ("track", "upscaling"),
        ("miner_hotkey", "hk-moved"),
        ("content_digest", "2" * 64),
        ("scorer_version", "vidaio-scorer/1+moved0000000"),
    ],
)
def test_rejects_self_hashed_packet_moved_from_request(
    seeded_conn, tmp_path, field, value
):
    conn, cid, contender_id, item_id = seeded_conn
    packet_json = _packet_json(**{field: value})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "item_score_json": packet_json,
                "packet_digest": hashlib.sha256(packet_json.encode()).hexdigest(),
            },
        )

    client, item, output = _wire(seeded_conn, tmp_path, handler)
    with pytest.raises(ScoringClientError, match=field):
        client.score_item(cid, contender_id, item, output)


def test_rejects_packet_with_missing_scorer_stamp(seeded_conn, tmp_path):
    conn, cid, contender_id, item_id = seeded_conn
    packet = json.loads(_packet_json())
    packet.pop("scorer_version")
    packet_json = json.dumps(packet, sort_keys=True, separators=(",", ":"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "item_score_json": packet_json,
                "packet_digest": hashlib.sha256(packet_json.encode()).hexdigest(),
            },
        )

    client, item, output = _wire(seeded_conn, tmp_path, handler)
    with pytest.raises(ScoringClientError, match="scorer_version"):
        client.score_item(cid, contender_id, item, output)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_runtime", "runtime stamp"),
        ("moved_runtime", "runtime stamp"),
        ("missing_backend", "backend_versions"),
        ("moved_backend", "backend_versions"),
        ("cuda_backend", "backend_versions"),
    ],
)
def test_rejects_packet_runtime_or_complete_backend_map_drift(
    seeded_conn, tmp_path, mutation, match
):
    conn, cid, contender_id, item_id = seeded_conn
    backends = dict(EXPECTED_BACKENDS)
    if mutation == "missing_runtime":
        backends.pop(SCORER_RUNTIME_BACKEND_KEY)
    elif mutation == "moved_runtime":
        backends[SCORER_RUNTIME_BACKEND_KEY] = SCORER_RUNTIME_BACKEND_PREFIX + "f" * 64
    elif mutation == "missing_backend":
        backends.pop("libvmaf")
    elif mutation == "moved_backend":
        backends["libvmaf"] = "libvmaf/9.9.9"
    else:
        # Reusing the correct health runtime stamp cannot conceal a CUDA scorer.
        backends["pieapp"] = "piq/0.8.0:cuda"
    packet_json = _packet_json(backend_versions=backends)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "item_score_json": packet_json,
                "packet_digest": hashlib.sha256(packet_json.encode()).hexdigest(),
            },
        )

    client, item, output = _wire(seeded_conn, tmp_path, handler)
    with pytest.raises(ScoringClientError, match=match):
        client.score_item(cid, contender_id, item, output)


def test_rejects_inconsistent_health_runtime_before_score_call(seeded_conn, tmp_path):
    conn, cid, contender_id, item_id = seeded_conn
    score_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal score_called
        score_called = True
        return httpx.Response(500)

    client, item, output = _wire(
        seeded_conn,
        tmp_path,
        handler,
        health_payload=_health_payload(digest="f" * 64),
    )
    with pytest.raises(ScoringClientError, match="does not hash"):
        client.score_item(cid, contender_id, item, output)
    assert score_called is False


@pytest.mark.parametrize(
    "mutation",
    [
        "marker",
        "manifest",
        "os",
        "arch",
        "threads",
        "kernels",
        "missing_backend",
        "cuda_pieapp",
        "cuda_torch",
        "wrong_weights",
        "local_expected_digest",
    ],
)
def test_production_client_rejects_runtime_policy_mutations_before_score(
    seeded_conn, tmp_path, mutation
):
    attestation = copy.deepcopy(RUNTIME_ATTESTATION)
    if mutation == "marker":
        attestation["release"]["marker_verified"] = False
    elif mutation == "manifest":
        attestation["release"]["manifest_verified"] = False
    elif mutation == "os":
        attestation["execution_policy"]["actual_os"] = "darwin"
    elif mutation == "arch":
        attestation["execution_policy"]["actual_arch"] = "arm64"
    elif mutation == "threads":
        attestation["execution_policy"]["actual_torch_interop_threads"] = 8
    elif mutation == "kernels":
        attestation["execution_policy"]["actual_torch_mkldnn_enabled"] = True
    elif mutation == "missing_backend":
        del attestation["payout_backends"]["libvmaf"]
    elif mutation == "cuda_pieapp":
        attestation["payout_backends"]["pieapp"] = "pieapp/piq:cuda"
    elif mutation == "cuda_torch":
        attestation["payout_backends"]["torch"] = "torch/2.8.0+cu128"
    elif mutation == "wrong_weights":
        attestation["payout_backends"]["pieapp_weights"] = "sha256:" + "f" * 64
    else:
        # Still a canonical-shaped claim: only the exact locally derived contract
        # catches this self-consistent but different release digest.
        attestation["release"]["source_sha256"] = "f" * 64

    score_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal score_called
        score_called = True
        return httpx.Response(500)

    client, item, output = _wire(
        seeded_conn,
        tmp_path,
        handler,
        health_payload=_health_for_attestation(attestation),
    )
    conn, cid, contender_id, item_id = seeded_conn
    with pytest.raises(ScorerRuntimeMismatch):
        client.score_item(cid, contender_id, item, output)
    assert score_called is False


def test_failed_health_refresh_clears_prior_runtime_contract(seeded_conn, tmp_path):
    conn, cid, contender_id, item_id = seeded_conn
    health_calls = 0
    score_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal health_calls, score_called
        if request.url.path == "/healthz":
            health_calls += 1
            if health_calls == 1:
                return httpx.Response(200, json=_health_payload())
            return httpx.Response(200, json={"scorer_version": SCORER_IDENTITY})
        score_called = True
        return httpx.Response(500)

    client = HttpScoringClient(
        "http://scoring.test",
        conn,
        inputs_dir=tmp_path / "in",
        outputs_dir=tmp_path / "out",
        transport=httpx.MockTransport(handler),
        expected_runtime_contract=EXPECTED_RUNTIME_CONTRACT,
    )
    assert client.scorer_identity() == SCORER_IDENTITY
    with pytest.raises(ScorerRuntimeUnavailable):
        client.scorer_identity()

    item = BatchItem(
        item_id=item_id,
        item_index=0,
        input_sha256="0" * 64,
        input_bytes=4096,
    )
    output = BatchOutput(
        item_id=item_id,
        output_sha256="1" * 64,
        output_bytes=512,
    )
    with pytest.raises(ScoringClientError, match="runtime identity is unavailable"):
        client.score_item(cid, contender_id, item, output)
    assert health_calls == 3
    assert score_called is False
