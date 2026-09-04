"""Remote miner byte transport: bounded streams, binding, auth, and cleanup."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import shutil
from pathlib import Path
from typing import AsyncIterator

import httpx
import pytest
from pydantic import ValidationError
from prometheus_client import CollectorRegistry

from vidaio.miner.config import MinerConfig
from vidaio.miner.service import _DiscardingFileResponse, _Metrics, create_app
from vidaio.services.artifact_auth import (
    ArtifactClientAuth,
    ArtifactServerAuth,
    CallableHotkeySigner,
    FrozenValidatorRegistry,
)
from vidaio.services.miner_artifacts import submit_miner_artifact
from vidaio.services.protocol import (
    MINER_ARTIFACT_AUTH_VERSION,
    MINER_ARTIFACT_ROUTE,
    MINER_ARTIFACT_VERSION,
    MINER_ARTIFACT_VERSION_HEADER,
    MINER_OUTPUT_DIGEST_HEADER,
    MINER_TASK_ID_HEADER,
    MINER_TASK_METADATA_HEADER,
    MinerArtifactTaskRequest,
    MinerTaskRequest,
    encode_miner_task_metadata,
)

from miner_support import FFMPEG, generate_clip

_AUTH_KEYS = {
    "validator-a": b"validator-secret",
    "miner-a": b"miner-secret",
}


def _auth_signature(hotkey: str, payload: bytes) -> str:
    return hashlib.sha512(_AUTH_KEYS[hotkey] + b"\x00" + payload).hexdigest()


def _auth_verify(hotkey: str, payload: bytes, signature: str) -> bool:
    try:
        return signature == _auth_signature(hotkey, payload)
    except KeyError:
        return False


def _auth_pair(
    *, nonce: str = "01" * 16
) -> tuple[ArtifactClientAuth, ArtifactServerAuth]:
    server_clock_calls = 0

    def server_clock() -> float:
        nonlocal server_clock_calls
        server_clock_calls += 1
        return 994.0 if server_clock_calls == 1 else 1_000.0

    client = ArtifactClientAuth(
        CallableHotkeySigner(
            "validator-a", lambda payload: _auth_signature("validator-a", payload)
        ),
        verify_fn=_auth_verify,
        clock=lambda: 1_000.0,
        nonce_factory=lambda: nonce,
    )
    server = ArtifactServerAuth(
        CallableHotkeySigner(
            "miner-a", lambda payload: _auth_signature("miner-a", payload)
        ),
        FrozenValidatorRegistry(("validator-a",)),
        verify_fn=_auth_verify,
        clock=server_clock,
    )
    return client, server


def _headers(metadata: MinerArtifactTaskRequest, **extra: str) -> dict[str, str]:
    return {
        "Content-Type": "application/octet-stream",
        MINER_ARTIFACT_VERSION_HEADER: MINER_ARTIFACT_VERSION,
        MINER_TASK_METADATA_HEADER: encode_miner_task_metadata(metadata),
        **extra,
    }


def _metadata(
    path: Path, task_id: str = "remote-1", **over
) -> MinerArtifactTaskRequest:
    values = {
        "task_id": task_id,
        "track": "compression",
        "input_digest": hashlib.sha256(path.read_bytes()).hexdigest(),
        "deadline_seconds": 120,
        **over,
    }
    return MinerArtifactTaskRequest(**values)


def test_remote_ingress_timeout_has_a_bounded_server_default() -> None:
    assert MinerConfig().artifact_ingress_timeout_seconds == 60.0
    bounded = MinerConfig(artifact_ingress_timeout_seconds=300)
    assert bounded.artifact_ingress_timeout_seconds == 300
    with pytest.raises(ValidationError):
        MinerConfig(artifact_ingress_timeout_seconds=300.0001)
    with pytest.raises(ValidationError, match="per_validator"):
        MinerConfig(
            artifact_replay_cache_entries=10,
            artifact_replay_cache_entries_per_validator=11,
        )


async def test_unsigned_v1_is_rejected_by_default_without_reading_body(
    tmp_path: Path,
) -> None:
    cfg = MinerConfig(work_dir=tmp_path / "work", ffmpeg_path=FFMPEG)
    app = create_app(cfg, _Metrics(CollectorRegistry()))
    metadata = MinerArtifactTaskRequest(
        task_id="unsigned",
        track="compression",
        input_digest=hashlib.sha256(b"input").hexdigest(),
    )
    consumed = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal consumed
        consumed += 1
        yield b"input"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        response = await client.post(
            MINER_ARTIFACT_ROUTE,
            headers=_headers(metadata),
            content=body(),
        )
    assert response.status_code == 426
    assert response.json()["detail"]["code"] == "artifact_v2_required"
    assert consumed == 0


async def test_artifact_v2_streams_and_verifies_both_hotkey_signatures(
    tmp_path: Path,
) -> None:
    client_auth, server_auth = _auth_pair()
    cfg = MinerConfig(
        work_dir=tmp_path / "work",
        ffmpeg_path=FFMPEG,
        artifact_hotkey="miner-a",
    )
    app = create_app(cfg, _Metrics(CollectorRegistry()), artifact_auth=server_auth)
    clip = generate_clip(tmp_path / "input.mp4")
    request = MinerTaskRequest(
        task_id="signed-v2",
        track="compression",
        input_path=str(clip),
        input_digest=hashlib.sha256(clip.read_bytes()).hexdigest(),
        deadline_seconds=120,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner", timeout=180
    ) as client:
        result = await submit_miner_artifact(
            client,
            "http://miner",
            request,
            output_dir=tmp_path / "downloads",
            max_input_bytes=cfg.max_input_bytes,
            max_output_bytes=cfg.max_output_bytes,
            timeout=120,
            artifact_auth=client_auth,
            expected_miner_hotkey="miner-a",
        )
    assert Path(result.output_path).is_file()
    assert (
        result.output_digest
        == hashlib.sha256(Path(result.output_path).read_bytes()).hexdigest()
    )
    assert not (cfg.work_dir / request.task_id).exists()


async def test_artifact_v2_cold_start_fence_rejects_before_a_body_byte(
    tmp_path: Path,
) -> None:
    now = [1_000.25]
    server_auth = ArtifactServerAuth(
        CallableHotkeySigner(
            "miner-a", lambda payload: _auth_signature("miner-a", payload)
        ),
        FrozenValidatorRegistry(("validator-a",)),
        verify_fn=_auth_verify,
        clock=lambda: now[0],
        request_future_skew_seconds=5,
    )
    client_auth = ArtifactClientAuth(
        CallableHotkeySigner(
            "validator-a", lambda payload: _auth_signature("validator-a", payload)
        ),
        verify_fn=_auth_verify,
        # A request captured immediately before restart may legitimately have
        # used the server's positive clock-skew allowance.
        clock=lambda: 1_005.0,
        nonce_factory=lambda: "01" * 16,
    )
    cfg = MinerConfig(
        work_dir=tmp_path / "work",
        ffmpeg_path=FFMPEG,
        warrant_track="compression",
        artifact_hotkey="miner-a",
    )
    app = create_app(cfg, _Metrics(CollectorRegistry()), artifact_auth=server_auth)
    metadata = MinerArtifactTaskRequest(
        task_id="cold-start-replay",
        track="upscaling",
        input_digest=hashlib.sha256(b"input").hexdigest(),
        deadline_seconds=30,
    )
    _, signed = client_auth.sign_request(
        metadata,
        input_size=5,
        intended_miner_hotkey="miner-a",
    )
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": "5",
        MINER_TASK_METADATA_HEADER: encode_miner_task_metadata(metadata),
        **signed,
    }
    consumed = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal consumed
        consumed += 1
        yield b"input"

    now[0] = 1_005.0
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        response = await client.post(
            MINER_ARTIFACT_ROUTE, headers=headers, content=body()
        )
    assert response.status_code == 425
    assert response.json()["detail"]["code"] == "artifact_auth_starting"
    assert consumed == 0


async def test_artifact_v2_wallet_failure_cleans_output_and_returns_typed_503(
    tmp_path: Path,
) -> None:
    client_auth, _ = _auth_pair()

    def signer_failure(_payload: bytes) -> str:
        raise LookupError("wallet locked")

    server_clock_calls = 0

    def server_clock() -> float:
        nonlocal server_clock_calls
        server_clock_calls += 1
        return 994.0 if server_clock_calls == 1 else 1_000.0

    server_auth = ArtifactServerAuth(
        CallableHotkeySigner("miner-a", signer_failure),
        FrozenValidatorRegistry(("validator-a",)),
        verify_fn=_auth_verify,
        clock=server_clock,
    )
    cfg = MinerConfig(
        work_dir=tmp_path / "work",
        ffmpeg_path=FFMPEG,
        artifact_hotkey="miner-a",
    )
    app = create_app(cfg, _Metrics(CollectorRegistry()), artifact_auth=server_auth)
    clip = generate_clip(tmp_path / "input.mp4")
    request = MinerTaskRequest(
        task_id="wallet-failure",
        track="compression",
        input_path=str(clip),
        input_digest=hashlib.sha256(clip.read_bytes()).hexdigest(),
        deadline_seconds=120,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner", timeout=180
    ) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc:
            await submit_miner_artifact(
                client,
                "http://miner",
                request,
                output_dir=tmp_path / "downloads",
                max_input_bytes=cfg.max_input_bytes,
                max_output_bytes=cfg.max_output_bytes,
                timeout=120,
                artifact_auth=client_auth,
                expected_miner_hotkey="miner-a",
            )
    assert exc.value.response.status_code == 503
    assert exc.value.response.json()["detail"]["code"] == "artifact_signing_failed"
    assert not (cfg.work_dir / request.task_id).exists()


async def test_artifact_v2_replay_is_rejected_before_a_body_byte(
    tmp_path: Path,
) -> None:
    client_auth, server_auth = _auth_pair()
    cfg = MinerConfig(
        work_dir=tmp_path / "work",
        ffmpeg_path=FFMPEG,
        warrant_track="compression",
        artifact_hotkey="miner-a",
    )
    app = create_app(cfg, _Metrics(CollectorRegistry()), artifact_auth=server_auth)
    metadata = MinerArtifactTaskRequest(
        task_id="replay",
        # A signed wrong-track request stops after auth without invoking ffmpeg.
        track="upscaling",
        input_digest=hashlib.sha256(b"input").hexdigest(),
        deadline_seconds=30,
    )
    _, signed = client_auth.sign_request(
        metadata,
        input_size=5,
        intended_miner_hotkey="miner-a",
    )
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": "5",
        MINER_TASK_METADATA_HEADER: encode_miner_task_metadata(metadata),
        **signed,
    }
    consumed = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal consumed
        consumed += 1
        yield b"input"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        first = await client.post(MINER_ARTIFACT_ROUTE, headers=headers, content=body())
        second = await client.post(
            MINER_ARTIFACT_ROUTE, headers=headers, content=body()
        )
    assert first.status_code == 422
    assert first.json()["detail"]["code"] == "warrant_track_mismatch"
    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "artifact_replay"
    assert consumed == 0
    assert first.request.headers[MINER_ARTIFACT_VERSION_HEADER] == (
        MINER_ARTIFACT_AUTH_VERSION
    )


async def test_remote_task_streams_output_and_removes_miner_copy(
    miner_client: httpx.AsyncClient, miner, tmp_path: Path
) -> None:
    clip = generate_clip(tmp_path / "input.mp4")
    metadata = _metadata(clip)

    response = await miner_client.post(
        MINER_ARTIFACT_ROUTE,
        headers=_headers(metadata),
        content=clip.read_bytes(),
    )

    assert response.status_code == 200, response.text
    assert response.headers[MINER_ARTIFACT_VERSION_HEADER] == MINER_ARTIFACT_VERSION
    assert response.headers[MINER_TASK_ID_HEADER] == metadata.task_id
    assert (
        hashlib.sha256(response.content).hexdigest()
        == response.headers[MINER_OUTPUT_DIGEST_HEADER]
    )
    assert response.content
    # FileResponse's background cleanup runs after the last response chunk. The
    # caller owns the bytes now; no cross-host path has to survive on the miner.
    assert not (Path(miner.cfg.work_dir) / metadata.task_id).exists()


async def test_remote_input_digest_mismatch_is_typed_and_leaves_no_task_dir(
    miner_client: httpx.AsyncClient, miner, tmp_path: Path
) -> None:
    clip = generate_clip(tmp_path / "input.mp4")
    metadata = _metadata(clip).model_copy(update={"input_digest": "0" * 64})
    response = await miner_client.post(
        MINER_ARTIFACT_ROUTE, headers=_headers(metadata), content=clip.read_bytes()
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "input_digest_mismatch"
    assert not (Path(miner.cfg.work_dir) / metadata.task_id).exists()


async def test_chunked_oversize_upload_is_cut_off_and_partial_is_deleted(
    tmp_path: Path,
) -> None:
    cfg = MinerConfig(
        work_dir=tmp_path / "work",
        ffmpeg_path=FFMPEG,
        max_input_bytes=1024,
        allow_unsigned_artifact_v1=True,
    )
    app = create_app(cfg, _Metrics(CollectorRegistry()))
    payload = b"x" * 8192
    metadata = MinerArtifactTaskRequest(
        task_id="too-large",
        track="compression",
        input_digest=hashlib.sha256(payload).hexdigest(),
        deadline_seconds=120,
    )
    consumed = 0

    async def chunks() -> AsyncIterator[bytes]:
        nonlocal consumed
        for _ in range(256):
            consumed += 1
            yield payload
            await asyncio.sleep(0)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        response = await client.post(
            MINER_ARTIFACT_ROUTE, headers=_headers(metadata), content=chunks()
        )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "input_too_large"
    assert consumed < 256
    assert not (cfg.work_dir / metadata.task_id).exists()


async def test_slow_remote_upload_hits_server_ingress_timeout_and_releases_slot(
    tmp_path: Path,
) -> None:
    cfg = MinerConfig(
        work_dir=tmp_path / "work",
        ffmpeg_path=FFMPEG,
        max_concurrent_tasks=1,
        artifact_ingress_timeout_seconds=0.05,
        allow_unsigned_artifact_v1=True,
    )
    app = create_app(cfg, _Metrics(CollectorRegistry()))
    payload = b"first chunk"
    metadata = MinerArtifactTaskRequest(
        task_id="slow-upload",
        track="compression",
        input_digest=hashlib.sha256(payload).hexdigest(),
        # An attacker-controlled deadline cannot enlarge the miner's cap.
        deadline_seconds=10_000,
    )

    async def stalled_body() -> AsyncIterator[bytes]:
        yield payload
        await asyncio.Event().wait()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        response = await asyncio.wait_for(
            client.post(
                MINER_ARTIFACT_ROUTE,
                headers=_headers(metadata),
                content=stalled_body(),
            ),
            timeout=1.0,
        )

    assert response.status_code == 408
    assert response.json()["detail"]["code"] == "ingress_timeout"
    assert not (cfg.work_dir / metadata.task_id).exists()
    assert not app.state.task_slots.locked()


async def test_auth_rejects_before_reading_remote_body(tmp_path: Path) -> None:
    cfg = MinerConfig(
        work_dir=tmp_path / "work", ffmpeg_path=FFMPEG, api_token="secret"
    )
    app = create_app(cfg, _Metrics(CollectorRegistry()))
    metadata = MinerArtifactTaskRequest(
        task_id="unauthorized",
        track="compression",
        input_digest=hashlib.sha256(b"x").hexdigest(),
    )
    consumed = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal consumed
        consumed += 1
        yield b"x"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        response = await client.post(
            MINER_ARTIFACT_ROUTE, headers=_headers(metadata), content=body()
        )
    assert response.status_code == 401
    assert consumed == 0
    assert not cfg.work_dir.exists()


async def test_remote_metadata_rejects_url_and_path_fields_without_reading_body(
    miner_client: httpx.AsyncClient, miner, tmp_path: Path
) -> None:
    raw = json.dumps(
        {
            "task_id": "ssrf-attempt",
            "track": "compression",
            "input_digest": "0" * 64,
            "input_url": "http://169.254.169.254/latest/meta-data/",
            "input_path": "/etc/passwd",
        },
        separators=(",", ":"),
    ).encode()
    encoded = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    consumed = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal consumed
        consumed += 1
        yield b"not read"

    response = await miner_client.post(
        MINER_ARTIFACT_ROUTE,
        headers={
            "Content-Type": "application/octet-stream",
            MINER_ARTIFACT_VERSION_HEADER: MINER_ARTIFACT_VERSION,
            MINER_TASK_METADATA_HEADER: encoded,
        },
        content=body(),
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_metadata"
    assert consumed == 0
    assert not (Path(miner.cfg.work_dir) / "ssrf-attempt").exists()


async def test_backend_output_cap_fails_closed_and_cleans_task(tmp_path: Path) -> None:
    cfg = MinerConfig(
        work_dir=tmp_path / "work",
        ffmpeg_path=FFMPEG,
        max_output_bytes=1,
        allow_unsigned_artifact_v1=True,
    )
    app = create_app(cfg, _Metrics(CollectorRegistry()))
    clip = generate_clip(tmp_path / "input.mp4")
    metadata = _metadata(clip, task_id="bounded-output")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner", timeout=180
    ) as client:
        response = await client.post(
            MINER_ARTIFACT_ROUTE, headers=_headers(metadata), content=clip.read_bytes()
        )
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "output_too_large"
    assert not (cfg.work_dir / metadata.task_id).exists()


async def test_output_send_failure_still_removes_private_task_dir(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "work" / "disconnect"
    task_dir.mkdir(parents=True)
    output = task_dir / "output.mp4"
    output.write_bytes(b"streamed output")
    response = _DiscardingFileResponse(
        output,
        media_type="application/octet-stream",
        cleanup=lambda: shutil.rmtree(task_dir, ignore_errors=True),
    )

    async def receive():  # type: ignore[no-untyped-def]
        return {"type": "http.disconnect"}

    async def send(message):  # type: ignore[no-untyped-def]
        if message["type"] == "http.response.body":
            raise ConnectionError("client disconnected")

    with pytest.raises(ConnectionError, match="client disconnected"):
        await response(
            {"type": "http", "method": "GET", "headers": [], "extensions": {}},
            receive,
            send,
        )
    assert not task_dir.exists()
