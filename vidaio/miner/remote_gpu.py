"""Bounded authenticated client for a dedicated remote GPU transform worker.

The chain-facing miner remains the protocol authority: it validates the signed
validator request, stages and hashes the input, applies its own deadline and
output cap, then signs the returned bytes. This client is only the private
transform hop. It never follows redirects, never places credentials in a URL,
and requires the worker to bind its response to input digest, track, solution
variant and a CUDA execution assertion.

The GPU is allowed to *produce media only*. Scoring remains the shipped CPU
scoring worker, and audit bundles retain these exact output bytes so an ordinary
CPU auditor can recompute every awarded score.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from vidaio.miner.backends import BackendError, BackendTimeoutError

GPU_PROTOCOL_VERSION = "vidaio-next-gpu-transform/v1"
GPU_METADATA_HEADER = "x-vidaio-gpu-metadata"
GPU_PROTOCOL_HEADER = "x-vidaio-gpu-protocol"
GPU_INPUT_DIGEST_HEADER = "x-vidaio-input-digest"
GPU_OUTPUT_DIGEST_HEADER = "x-vidaio-output-digest"
GPU_TRACK_HEADER = "x-vidaio-track"
GPU_VARIANT_HEADER = "x-vidaio-solution-variant"
GPU_ACCELERATED_HEADER = "x-vidaio-gpu-accelerated"
GPU_DEVICE_HEADER = "x-vidaio-gpu-device"
MAX_ERROR_BODY_BYTES = 4 * 1024
MAX_HEALTH_BODY_BYTES = 64 * 1024
SUPPORTED_GPU_VARIANTS = ("quality", "balanced", "compact", "premium")
CPU_FALLBACK_DEVICE = "cpu:ffmpeg-fallback"
LOGGER = logging.getLogger(__name__)


def _metadata_header(payload: Mapping[str, object]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _response_error(response: httpx.Response) -> str:
    """Return a small printable error without trusting a remote-sized body."""
    data = bytearray()
    for chunk in response.iter_bytes():
        remaining = MAX_ERROR_BODY_BYTES - len(data)
        if remaining <= 0:
            break
        data.extend(chunk[:remaining])
    text = bytes(data).decode("utf-8", errors="replace").replace("\n", " ").strip()
    return text or response.reason_phrase


class RemoteGpuBackend:
    """One track/profile binding to an authenticated HTTPS GPU endpoint."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        auth_token: str,
        track: str,
        solution_variant: str,
        max_output_bytes: int,
        request_timeout_seconds: float,
        connect_timeout_seconds: float = 10.0,
        allow_cpu_fallback: bool = False,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(endpoint_url)
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("remote GPU endpoint contains an invalid port") from exc
        if (
            parsed.scheme.lower() != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "remote GPU endpoint must be absolute HTTPS without credentials, "
                "query, or fragment"
            )
        if (
            not auth_token
            or auth_token != auth_token.strip()
            or any(character in auth_token for character in "\r\n\x00")
        ):
            raise ValueError(
                "remote GPU auth token must be non-empty, trimmed and single-line"
            )
        if track not in {"compression", "upscaling"}:
            raise ValueError(f"unsupported remote GPU track {track!r}")
        if solution_variant not in SUPPORTED_GPU_VARIANTS:
            raise ValueError(
                f"unsupported remote GPU solution variant {solution_variant!r}"
            )
        self.endpoint_url = endpoint_url.rstrip("/")
        self.track = track
        self.solution_variant = solution_variant
        self.max_output_bytes = int(max_output_bytes)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.allow_cpu_fallback = allow_cpu_fallback
        if self.max_output_bytes <= 0:
            raise ValueError("remote GPU output cap must be positive")
        if not (
            math.isfinite(self.request_timeout_seconds)
            and self.request_timeout_seconds > 0
            and math.isfinite(self.connect_timeout_seconds)
            and self.connect_timeout_seconds > 0
        ):
            raise ValueError("remote GPU request/connect timeouts must be finite and positive")
        self._auth_token = auth_token
        self._transport = transport

    def __repr__(self) -> str:
        return (
            "RemoteGpuBackend("
            f"endpoint_url={self.endpoint_url!r}, track={self.track!r}, "
            f"solution_variant={self.solution_variant!r}, auth_token=<redacted>)"
        )

    def _effective_timeout(self, timeout: float | None) -> float:
        whole = self.request_timeout_seconds
        if timeout is not None:
            requested = float(timeout)
            if not math.isfinite(requested):
                raise BackendTimeoutError("remote GPU deadline must be finite")
            whole = min(whole, requested)
        if whole <= 0 or not math.isfinite(whole):
            raise BackendTimeoutError("remote GPU deadline was already exhausted")
        return whole

    def _timeout(self, timeout: float | None) -> httpx.Timeout:
        whole = self._effective_timeout(timeout)
        return httpx.Timeout(
            timeout=whole,
            connect=min(whole, self.connect_timeout_seconds),
        )

    def process(
        self,
        input_path: str,
        output_path: str,
        params: Mapping[str, float | int | str],
        *,
        timeout: float | None = None,
    ) -> None:
        source_path = Path(input_path)
        destination = Path(output_path)
        if not source_path.is_file():
            raise BackendError(f"remote GPU input does not exist: {source_path}")
        input_size = source_path.stat().st_size
        input_digest = _sha256(source_path)
        effective_timeout = self._effective_timeout(timeout)
        metadata = {
            "protocol": GPU_PROTOCOL_VERSION,
            "track": self.track,
            "solution_variant": self.solution_variant,
            "input_digest": input_digest,
            "input_size": input_size,
            "deadline_seconds": effective_timeout,
            "params": dict(params),
        }
        headers = {
            "Authorization": f"Bearer {self._auth_token}",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(input_size),
            GPU_PROTOCOL_HEADER: GPU_PROTOCOL_VERSION,
            GPU_METADATA_HEADER: _metadata_header(metadata),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(
            f".{destination.name}.vidaio-next-{uuid.uuid4().hex}.part"
        )
        try:
            with (
                source_path.open("rb") as source,
                httpx.Client(
                    transport=self._transport,
                    follow_redirects=False,
                    timeout=self._timeout(effective_timeout),
                ) as client,
                client.stream(
                    "POST",
                    f"{self.endpoint_url}/process",
                    headers=headers,
                    content=source,
                ) as response,
            ):
                if response.status_code != 200:
                    detail = _response_error(response).replace(
                        self._auth_token, "<redacted>"
                    )
                    detail = "".join(
                        character if character.isprintable() else " "
                        for character in detail
                    )
                    LOGGER.warning(
                        "remote GPU refused transform status=%s track=%s variant=%s input=%s detail=%s",
                        response.status_code, self.track, self.solution_variant,
                        input_digest[:12], detail,
                    )
                    raise BackendError(
                        "remote GPU worker refused transform "
                        f"(HTTP {response.status_code}): {detail}"
                    )
                self._validate_binding(response, input_digest)
                media_type = response.headers.get("content-type", "").split(";", 1)[
                    0
                ].strip().lower()
                if media_type not in {"video/mp4", "application/octet-stream"}:
                    raise BackendError(
                        f"remote GPU response has unsupported media type {media_type!r}"
                    )
                declared = response.headers.get("content-length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError as exc:
                        raise BackendError(
                            "remote GPU response has invalid Content-Length"
                        ) from exc
                    if declared_size < 1 or declared_size > self.max_output_bytes:
                        raise BackendError(
                            "remote GPU response size is outside the configured bound: "
                            f"{declared_size}"
                        )
                digest = hashlib.sha256()
                received = 0
                with partial.open("xb") as sink:
                    os.chmod(partial, 0o600)
                    for chunk in response.iter_bytes():
                        received += len(chunk)
                        if received > self.max_output_bytes:
                            raise BackendError(
                                "remote GPU response crossed the configured output cap "
                                f"of {self.max_output_bytes} bytes"
                            )
                        digest.update(chunk)
                        sink.write(chunk)
                    sink.flush()
                    os.fsync(sink.fileno())
                if received < 1:
                    raise BackendError("remote GPU worker returned an empty output")
                expected = response.headers.get(GPU_OUTPUT_DIGEST_HEADER, "")
                actual = digest.hexdigest()
                if expected != actual:
                    raise BackendError(
                        "remote GPU output digest mismatch: "
                        f"received {actual}, worker claimed {expected or '<missing>'}"
                    )
            os.replace(partial, destination)
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError(f"remote GPU request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"remote GPU transport failed: {exc}") from exc
        finally:
            partial.unlink(missing_ok=True)

    def _validate_binding(self, response: httpx.Response, input_digest: str) -> None:
        device = response.headers.get(GPU_DEVICE_HEADER, "")
        cpu_fallback = self.allow_cpu_fallback and device == CPU_FALLBACK_DEVICE
        expected = {
            GPU_PROTOCOL_HEADER: GPU_PROTOCOL_VERSION,
            GPU_INPUT_DIGEST_HEADER: input_digest,
            GPU_TRACK_HEADER: self.track,
            GPU_VARIANT_HEADER: self.solution_variant,
            GPU_ACCELERATED_HEADER: "false" if cpu_fallback else "true",
        }
        for name, value in expected.items():
            actual = response.headers.get(name, "")
            if actual != value:
                raise BackendError(
                    f"remote GPU response binding {name}={actual!r}, expected {value!r}"
                )
        cuda = device.lower().startswith("cuda:")
        # The premium variant's compression path is deliberate CPU work
        # (ab-av1 VMAF search) and attests `abav1:<encoder>` instead of
        # claiming CUDA; every other variant/track must still prove CUDA.
        premium_cpu = (
            self.solution_variant == "premium"
            and self.track == "compression"
            and device.lower().startswith("abav1:")
        )
        if not (cuda or premium_cpu or cpu_fallback):
            raise BackendError(
                "remote worker did not attest an acceptable device in "
                f"{GPU_DEVICE_HEADER}: {device!r}"
            )

    def health(self) -> dict[str, object]:
        """Perform the explicit live probe (not used by passive local health)."""
        headers = {"Authorization": f"Bearer {self._auth_token}"}
        try:
            with (
                httpx.Client(
                    transport=self._transport,
                    follow_redirects=False,
                    # A scaled-to-zero Modal Function may need substantially more
                    # than the socket-connect budget to cold-start its GPU container.
                    # Keep the short connect phase, but give this explicit live probe
                    # the same bounded whole-request budget as a transform.
                    timeout=self._timeout(None),
                ) as client,
                client.stream(
                    "GET", f"{self.endpoint_url}/healthz", headers=headers
                ) as response,
            ):
                response.raise_for_status()
                declared = response.headers.get("content-length")
                if declared is not None and int(declared) > MAX_HEALTH_BODY_BYTES:
                    raise BackendError("remote GPU health response is too large")
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > MAX_HEALTH_BODY_BYTES:
                        raise BackendError("remote GPU health response is too large")
                payload = json.loads(raw)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise BackendError(f"remote GPU health probe failed: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("gpu_available") is not True:
            raise BackendError("remote GPU health probe did not report gpu_available=true")
        if payload.get("protocol") != GPU_PROTOCOL_VERSION:
            raise BackendError("remote GPU health probe has the wrong protocol")
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
