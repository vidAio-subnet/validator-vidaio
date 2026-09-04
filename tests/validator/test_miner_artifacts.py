"""Caller-side remote miner transfer: streaming, bounds, and fail-closed files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from vidaio.services.artifact_auth import (
    ArtifactClientAuth,
    ArtifactServerAuth,
    CallableHotkeySigner,
    FrozenValidatorRegistry,
)
from vidaio.services.miner_artifacts import (
    MinerArtifactColdStart,
    MinerArtifactInputError,
    MinerArtifactIntegrityError,
    MinerArtifactProtocolError,
    MinerArtifactTooLarge,
    submit_miner_artifact,
)
from vidaio.services.protocol import (
    MINER_ARTIFACT_ROUTE,
    MINER_ARTIFACT_VERSION,
    MINER_ARTIFACT_VERSION_HEADER,
    MINER_OUTPUT_DIGEST_HEADER,
    MINER_HOTKEY_HEADER,
    MINER_PROCESSING_SECONDS_HEADER,
    MINER_REQUEST_NONCE_HEADER,
    MINER_REQUEST_TIMESTAMP_HEADER,
    MINER_TASK_ID_HEADER,
    MINER_TASK_METADATA_HEADER,
    MINER_RESPONSE_SIGNATURE_HEADER,
    MinerTaskRequest,
    decode_miner_task_metadata,
)


def _test_signature(hotkey: str, payload: bytes) -> str:
    return hashlib.sha512(hotkey.encode() + b"\x00" + payload).hexdigest()


def _test_verify(hotkey: str, payload: bytes, signature: str) -> bool:
    return signature == _test_signature(hotkey, payload)


def _auth_pair() -> tuple[ArtifactClientAuth, ArtifactServerAuth]:
    server_clock_calls = 0

    def server_clock() -> float:
        nonlocal server_clock_calls
        server_clock_calls += 1
        return 994.0 if server_clock_calls == 1 else 1_000.0

    client = ArtifactClientAuth(
        CallableHotkeySigner(
            "validator-a", lambda payload: _test_signature("validator-a", payload)
        ),
        verify_fn=_test_verify,
        clock=lambda: 1_000.0,
        nonce_factory=lambda: "01" * 16,
    )
    server = ArtifactServerAuth(
        CallableHotkeySigner(
            "miner-a", lambda payload: _test_signature("miner-a", payload)
        ),
        FrozenValidatorRegistry(("validator-a",)),
        verify_fn=_test_verify,
        clock=server_clock,
    )
    return client, server


def _response_headers(task_id: str, data: bytes, **over: str) -> dict[str, str]:
    return {
        MINER_ARTIFACT_VERSION_HEADER: MINER_ARTIFACT_VERSION,
        MINER_TASK_ID_HEADER: task_id,
        MINER_OUTPUT_DIGEST_HEADER: hashlib.sha256(data).hexdigest(),
        MINER_PROCESSING_SECONDS_HEADER: "0.125",
        **over,
    }


def _request(path: Path, task_id: str = "challenge-1:7") -> MinerTaskRequest:
    h = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            h.update(chunk)
    return MinerTaskRequest(
        task_id=task_id,
        track="compression",
        input_path=str(path),
        input_digest=h.hexdigest(),
        deadline_seconds=30,
    )


async def test_streams_both_directions_and_ignores_miner_url(tmp_path: Path) -> None:
    source = tmp_path / "validator-private.mp4"
    source.write_bytes((b"0123456789abcdef" * (1 << 17)) + b"tail")
    output = b"remote-output" * (1 << 14)
    seen: dict[str, object] = {}
    app = FastAPI()

    @app.post(MINER_ARTIFACT_ROUTE)
    async def artifact(request: Request):  # type: ignore[no-untyped-def]
        metadata = decode_miner_task_metadata(
            request.headers.get(MINER_TASK_METADATA_HEADER)
        )
        seen["metadata"] = metadata
        seen["auth"] = request.headers.get("x-miner-token")
        h = hashlib.sha256()
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            h.update(chunk)
        seen["input_size"] = size
        seen["input_digest"] = h.hexdigest()

        async def chunks() -> AsyncIterator[bytes]:
            for offset in range(0, len(output), 8192):
                yield output[offset : offset + 8192]

        return StreamingResponse(
            chunks(),
            media_type="application/octet-stream",
            headers={
                **_response_headers(metadata.task_id, output),
                # A malicious/obsolete URL is inert: the client consumes only
                # this response body and never follows response locations.
                "X-Vidaio-Output-Url": "http://169.254.169.254/latest/meta-data/",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        result = await submit_miner_artifact(
            client,
            "http://miner",
            _request(source),
            output_dir=tmp_path / "downloads",
            max_input_bytes=source.stat().st_size,
            max_output_bytes=len(output),
            headers={"X-Miner-Token": "shared-secret"},
            timeout=10,
            allow_unsigned_v1=True,
        )

    assert Path(result.output_path).read_bytes() == output
    assert Path(result.output_path).is_relative_to(tmp_path / "downloads")
    assert result.output_digest == hashlib.sha256(output).hexdigest()
    assert result.processing_seconds == 0.125
    metadata = seen["metadata"]
    assert metadata.input_digest == seen["input_digest"]  # type: ignore[union-attr]
    assert seen["input_size"] == source.stat().st_size
    assert seen["auth"] == "shared-secret"
    # The private absolute path was never serialized into remote metadata.
    assert str(source) not in request_header_json(metadata)  # type: ignore[arg-type]


def request_header_json(metadata) -> str:  # type: ignore[no-untyped-def]
    return metadata.model_dump_json()


async def test_restart_fence_retries_with_fresh_auth_until_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vidaio.services import miner_artifacts as transport_module

    source = tmp_path / "input"
    source.write_bytes(b"challenge input")
    output = b"signed output after restart"
    timestamps = iter((1_000.0, 1_001.0, 1_002.0))
    nonces = iter(("01" * 16, "02" * 16, "03" * 16))
    client_auth = ArtifactClientAuth(
        CallableHotkeySigner(
            "validator-a", lambda payload: _test_signature("validator-a", payload)
        ),
        verify_fn=_test_verify,
        clock=lambda: next(timestamps),
        nonce_factory=lambda: next(nonces),
    )
    server_clock_calls = 0

    def server_clock() -> float:
        nonlocal server_clock_calls
        server_clock_calls += 1
        return 994.0 if server_clock_calls == 1 else 1_002.0

    server_auth = ArtifactServerAuth(
        CallableHotkeySigner(
            "miner-a", lambda payload: _test_signature("miner-a", payload)
        ),
        FrozenValidatorRegistry(("validator-a",)),
        verify_fn=_test_verify,
        clock=server_clock,
    )
    attempts: list[tuple[str, str]] = []
    app = FastAPI()

    @app.post(MINER_ARTIFACT_ROUTE)
    async def artifact(request: Request):  # type: ignore[no-untyped-def]
        attempts.append(
            (
                request.headers[MINER_REQUEST_TIMESTAMP_HEADER],
                request.headers[MINER_REQUEST_NONCE_HEADER],
            )
        )
        if len(attempts) < 3:
            return Response(status_code=425, content=b"restart fence")
        metadata = decode_miner_task_metadata(
            request.headers.get(MINER_TASK_METADATA_HEADER)
        )
        claims = server_auth.verify_request(
            request.headers,
            metadata,
            content_length=int(request.headers["content-length"]),
        )
        await request.body()
        signed = server_auth.response_headers(
            claims,
            metadata,
            output_digest=hashlib.sha256(output).hexdigest(),
            output_size=len(output),
            processing_seconds="0.125",
        )
        return Response(
            output,
            headers={**_response_headers(metadata.task_id, output), **signed},
        )

    backoffs: list[float] = []

    async def no_wait(seconds: float) -> None:
        backoffs.append(seconds)

    monkeypatch.setattr(transport_module, "_restart_fence_backoff", no_wait)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        result = await submit_miner_artifact(
            client,
            "http://miner",
            _request(source, "restart-success"),
            output_dir=tmp_path / "downloads",
            max_input_bytes=1024,
            max_output_bytes=1024,
            timeout=10,
            artifact_auth=client_auth,
            expected_miner_hotkey="miner-a",
        )

    assert Path(result.output_path).read_bytes() == output
    assert attempts == [
        ("1000", "01" * 16),
        ("1001", "02" * 16),
        ("1002", "03" * 16),
    ]
    assert backoffs == [1.0, 1.0]


async def test_restart_fence_deadline_exhaustion_is_typed_and_cleans_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"challenge input")
    client_auth, _ = _auth_pair()
    attempts = 0
    app = FastAPI()

    @app.post(MINER_ARTIFACT_ROUTE)
    async def artifact(_request: Request):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        return Response(status_code=425, content=b"restart fence")

    downloads = tmp_path / "downloads"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        with pytest.raises(MinerArtifactColdStart, match="restart fence") as excinfo:
            await submit_miner_artifact(
                client,
                "http://miner",
                _request(source, "restart-exhausted"),
                output_dir=downloads,
                max_input_bytes=1024,
                max_output_bytes=1024,
                timeout=0.01,
                artifact_auth=client_auth,
                expected_miner_hotkey="miner-a",
            )

    assert attempts == 1
    receipt = excinfo.value.artifact_request_receipt
    assert receipt.metadata.task_id == "restart-exhausted"
    assert receipt.miner_hotkey == "miner-a"
    assert excinfo.value.artifact_target_endpoint == "http://miner"
    assert not [path for path in downloads.rglob("*") if path.is_file()]


@pytest.mark.parametrize("mutation", ("downgrade", "foreign", "signature"))
async def test_v2_response_identity_failures_never_publish_a_download(
    tmp_path: Path, mutation: str
) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"signed input")
    output = b"signed miner output"
    client_auth, server_auth = _auth_pair()
    app = FastAPI()

    @app.post(MINER_ARTIFACT_ROUTE)
    async def artifact(request: Request):  # type: ignore[no-untyped-def]
        metadata = decode_miner_task_metadata(
            request.headers.get(MINER_TASK_METADATA_HEADER)
        )
        claims = server_auth.verify_request(
            request.headers,
            metadata,
            content_length=int(request.headers["content-length"]),
        )
        await request.body()
        signed = server_auth.response_headers(
            claims,
            metadata,
            output_digest=hashlib.sha256(output).hexdigest(),
            output_size=len(output),
            processing_seconds="0.125",
        )
        if mutation == "downgrade":
            signed[MINER_ARTIFACT_VERSION_HEADER] = MINER_ARTIFACT_VERSION
        elif mutation == "foreign":
            signed[MINER_HOTKEY_HEADER] = "miner-b"
        else:
            signed[MINER_RESPONSE_SIGNATURE_HEADER] = "00" * 64
        return Response(
            output,
            headers={
                **_response_headers(metadata.task_id, output),
                **signed,
            },
        )

    downloads = tmp_path / "downloads"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        with pytest.raises(MinerArtifactProtocolError):
            await submit_miner_artifact(
                client,
                "http://miner",
                _request(source, f"v2-{mutation}"),
                output_dir=downloads,
                max_input_bytes=1024,
                max_output_bytes=1024,
                timeout=10,
                artifact_auth=client_auth,
                expected_miner_hotkey="miner-a",
            )
    assert not [path for path in downloads.rglob("*") if path.is_file()]


async def test_digest_mismatch_removes_partial_download(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"input")
    output = b"untrusted output"
    app = FastAPI()

    @app.post(MINER_ARTIFACT_ROUTE)
    async def artifact(request: Request):  # type: ignore[no-untyped-def]
        metadata = decode_miner_task_metadata(
            request.headers.get(MINER_TASK_METADATA_HEADER)
        )
        await request.body()
        return Response(
            output,
            headers=_response_headers(
                metadata.task_id,
                output,
                **{MINER_OUTPUT_DIGEST_HEADER: hashlib.sha256(b"lie").hexdigest()},
            ),
        )

    downloads = tmp_path / "downloads"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        with pytest.raises(MinerArtifactIntegrityError, match="digest mismatch"):
            await submit_miner_artifact(
                client,
                "http://miner",
                _request(source, "bad-digest"),
                output_dir=downloads,
                max_input_bytes=1024,
                max_output_bytes=1024,
                timeout=10,
                allow_unsigned_v1=True,
            )
    assert not [p for p in downloads.rglob("*") if p.is_file()]


async def test_early_success_response_cannot_bypass_complete_input_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"the exact challenge bytes that must reach the miner")
    output = b"answer returned without reading the challenge"
    app = FastAPI()

    @app.post(MINER_ARTIFACT_ROUTE)
    async def artifact(request: Request):  # type: ignore[no-untyped-def]
        metadata = decode_miner_task_metadata(
            request.headers.get(MINER_TASK_METADATA_HEADER)
        )
        # Deliberately do not read request.stream()/request.body(). ASGI permits
        # an early response, and httpx then never exhausts the upload iterator.
        return Response(output, headers=_response_headers(metadata.task_id, output))

    downloads = tmp_path / "downloads"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        with pytest.raises(
            MinerArtifactProtocolError, match="complete input was uploaded"
        ):
            await submit_miner_artifact(
                client,
                "http://miner",
                _request(source, "early-success"),
                output_dir=downloads,
                max_input_bytes=1024,
                max_output_bytes=1024,
                timeout=10,
                allow_unsigned_v1=True,
            )
    assert not [p for p in downloads.rglob("*") if p.is_file()]


async def test_chunked_output_is_cut_off_at_hard_cap_and_cleaned(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"input")
    output = b"x" * 4096
    app = FastAPI()

    @app.post(MINER_ARTIFACT_ROUTE)
    async def artifact(request: Request):  # type: ignore[no-untyped-def]
        metadata = decode_miner_task_metadata(
            request.headers.get(MINER_TASK_METADATA_HEADER)
        )
        await request.body()

        async def chunks() -> AsyncIterator[bytes]:
            for offset in range(0, len(output), 256):
                yield output[offset : offset + 256]

        return StreamingResponse(
            chunks(), headers=_response_headers(metadata.task_id, output)
        )

    downloads = tmp_path / "downloads"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        with pytest.raises(MinerArtifactTooLarge, match="crossed 1024"):
            await submit_miner_artifact(
                client,
                "http://miner",
                _request(source, "too-big"),
                output_dir=downloads,
                max_input_bytes=1024,
                max_output_bytes=1024,
                timeout=10,
                allow_unsigned_v1=True,
            )
    assert not [p for p in downloads.rglob("*") if p.is_file()]


async def test_wrong_task_binding_is_rejected_before_any_file_is_created(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"input")
    output = b"output"
    app = FastAPI()

    @app.post(MINER_ARTIFACT_ROUTE)
    async def artifact(request: Request):  # type: ignore[no-untyped-def]
        await request.body()
        return Response(output, headers=_response_headers("another-task", output))

    downloads = tmp_path / "downloads"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        with pytest.raises(MinerArtifactProtocolError, match="another-task"):
            await submit_miner_artifact(
                client,
                "http://miner",
                _request(source, "expected-task"),
                output_dir=downloads,
                max_input_bytes=1024,
                max_output_bytes=1024,
                timeout=10,
                allow_unsigned_v1=True,
            )
    assert not downloads.exists()


async def test_symlink_input_is_refused_without_contacting_miner(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "actual"
    actual.write_bytes(b"input")
    link = tmp_path / "link"
    link.symlink_to(actual)
    called = False
    app = FastAPI()

    @app.post(MINER_ARTIFACT_ROUTE)
    async def artifact():  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        return Response(b"never")

    request = MinerTaskRequest(
        task_id="symlink",
        track="compression",
        input_path=str(link),
        input_digest=hashlib.sha256(actual.read_bytes()).hexdigest(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        with pytest.raises(MinerArtifactInputError, match="cannot open"):
            await submit_miner_artifact(
                client,
                "http://miner",
                request,
                output_dir=tmp_path / "downloads",
                max_input_bytes=1024,
                max_output_bytes=1024,
                timeout=10,
                allow_unsigned_v1=True,
            )
    assert called is False


async def test_wallet_signer_failure_closes_the_preopened_input_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vidaio.services import miner_artifacts as transport_module

    source = tmp_path / "input"
    source.write_bytes(b"input")
    opened: list[int] = []
    real_open = transport_module._open_regular_input

    def capture_open(path, max_bytes):  # type: ignore[no-untyped-def]
        fd, size = real_open(path, max_bytes)
        opened.append(fd)
        return fd, size

    def explode(_payload: bytes) -> str:
        raise RuntimeError("wallet unavailable")

    monkeypatch.setattr(transport_module, "_open_regular_input", capture_open)
    auth = ArtifactClientAuth(CallableHotkeySigner("validator-a", explode))
    async with httpx.AsyncClient(base_url="http://unused") as client:
        with pytest.raises(RuntimeError, match="wallet unavailable"):
            await submit_miner_artifact(
                client,
                "http://unused",
                _request(source, "signer-failure"),
                output_dir=tmp_path / "downloads",
                max_input_bytes=1024,
                max_output_bytes=1024,
                timeout=10,
                artifact_auth=auth,
                expected_miner_hotkey="miner-a",
            )
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])
