"""Bounded, path-free HTTP artifact exchange with inference miners.

The caller opens one regular local input through a symlink-refusing descriptor,
streams exactly those bytes to the miner, and streams the response into a
caller-owned directory.  Task binding, byte count, and sha256 are checked before
the downloaded file is published.  A miner never gets a validator/gateway path,
and a validator/gateway never follows a miner URL or path (the response body is
the output), closing both the shared-filesystem dependency and the SSRF seam.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import math
import os
import re
import stat
import tempfile
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Mapping

import httpx

from vidaio.services.artifact_auth import (
    ArtifactAuthError,
    ArtifactClientAuth,
    ArtifactRequestClaims,
    MinerArtifactReceipt,
    ValidatorArtifactRequestReceipt,
)
from vidaio.services.protocol import (
    MINER_ARTIFACT_AUTH_VERSION,
    MINER_ARTIFACT_ROUTE,
    MINER_ARTIFACT_VERSION,
    MINER_ARTIFACT_VERSION_HEADER,
    MINER_OUTPUT_DIGEST_HEADER,
    MINER_PROCESSING_SECONDS_HEADER,
    MINER_REQUEST_SIGNATURE_HEADER,
    MINER_RESPONSE_SIGNATURE_HEADER,
    MINER_TASK_ID_HEADER,
    MINER_TASK_METADATA_HEADER,
    MinerArtifactTaskRequest,
    MinerTaskRequest,
    MinerTaskResponse,
    encode_miner_task_metadata,
)

_CHUNK = 1 << 20
_MAX_ERROR_BODY = 64 << 10
_RESTART_FENCE_BACKOFF_SECONDS = 1.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_:-]{1,128}$")
_SAFE_HOSTNAME = re.compile(
    r"^(?=.{1,253}\.?$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.?$"
)


class MinerArtifactError(RuntimeError):
    """Base class for a rejected or incomplete remote artifact exchange."""


class MinerArtifactInputError(MinerArtifactError):
    """The caller's local input is unsafe, missing, changed, or over its bound."""


class MinerArtifactProtocolError(MinerArtifactError):
    """The miner's response is missing/invalid or bound to another task."""


class MinerArtifactIntegrityError(MinerArtifactProtocolError):
    """Streamed output bytes disagree with the miner's digest or byte count."""


class MinerArtifactTooLarge(MinerArtifactProtocolError):
    """A declared or streamed miner output crossed the caller's hard cap."""


class MinerArtifactColdStart(MinerArtifactError):
    """A miner's authenticated restart fence did not clear before the deadline."""


class MinerPeerAddressError(MinerArtifactError):
    """A chain-provided miner address is not a permitted HTTP destination."""


def _attach_request_evidence(
    exc: BaseException,
    request_receipt: ValidatorArtifactRequestReceipt | None,
    base_url: str,
) -> BaseException:
    """Attach an already-signed dispatch receipt to a typed transfer failure."""
    if request_receipt is not None:
        exc.artifact_request_receipt = request_receipt
        exc.artifact_target_endpoint = base_url
    return exc


async def _restart_fence_backoff(seconds: float) -> None:
    """Narrow sleep seam for bounded cold-start retry tests."""
    await asyncio.sleep(seconds)


def format_miner_peer_host(value: str, *, allow_non_public: bool = False) -> str:
    """Validate one chain-provided host and format it safely for an HTTP URL.

    Production accepts only globally routable literal IP addresses. This avoids
    DNS rebinding and prevents metagraph data from becoming an SSRF primitive.
    Report/local/compose environments may explicitly opt into private/loopback/
    link-local literals and simple DNS labels. Unspecified and multicast
    destinations are never dialable. IPv6 literals are returned bracketed.
    """
    raw = value.strip()
    if not raw or any(ch in raw for ch in ("/", "\\", "@", "?", "#")):
        raise MinerPeerAddressError(f"unsafe miner address {value!r}")
    if raw.startswith("[") or raw.endswith("]"):
        if not (raw.startswith("[") and raw.endswith("]")):
            raise MinerPeerAddressError(f"malformed bracketed miner address {value!r}")
        raw = raw[1:-1]
    if "%" in raw:  # scoped IPv6 zone ids are host-local and ambiguous in URLs
        raise MinerPeerAddressError(
            f"scoped IPv6 miner address is not allowed: {value!r}"
        )
    try:
        address = ipaddress.ip_address(raw)
    except ValueError:
        if not allow_non_public or not _SAFE_HOSTNAME.fullmatch(raw):
            raise MinerPeerAddressError(
                f"miner address {value!r} is not a public IP literal"
            )
        return raw.rstrip(".").lower()
    # Python's IPv6 classification has varied across releases for IPv4-mapped
    # literals. Apply the policy to the embedded IPv4 address explicitly so
    # ``::ffff:127.0.0.1`` cannot bypass the production loopback rule (and
    # mapped unspecified/multicast cannot bypass the never-dial rule locally).
    classified = (
        address.ipv4_mapped
        if isinstance(address, ipaddress.IPv6Address)
        and address.ipv4_mapped is not None
        else address
    )
    if classified.is_unspecified or classified.is_multicast:
        raise MinerPeerAddressError(f"miner address {value!r} is not dialable")
    if not allow_non_public and not classified.is_global:
        raise MinerPeerAddressError(f"miner address {value!r} is not globally routable")
    rendered = address.compressed
    return f"[{rendered}]" if isinstance(address, ipaddress.IPv6Address) else rendered


def _safe_task_id(task_id: str) -> str:
    if not _SAFE_TASK_ID.fullmatch(task_id):
        raise MinerArtifactInputError(
            f"task_id {task_id!r} is not a safe artifact directory name"
        )
    return task_id


def _open_regular_input(path: str | Path, max_bytes: int) -> tuple[int, int]:
    """Open one immutable handle; refuse symlinks and every non-regular type."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(os.fspath(path), flags)
    except OSError as exc:
        raise MinerArtifactInputError(
            f"cannot open miner input {path!s}: {exc}"
        ) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise MinerArtifactInputError(f"miner input {path!s} is not a regular file")
        if info.st_size > max_bytes:
            raise MinerArtifactInputError(
                f"miner input is {info.st_size} bytes; maximum is {max_bytes}"
            )
        if info.st_size < 1:
            raise MinerArtifactInputError("miner input is empty")
        with contextlib.suppress(OSError):
            os.set_blocking(fd, True)
        return fd, info.st_size
    except BaseException:
        os.close(fd)
        raise


def sign_miner_artifact_request_receipt(
    request: MinerTaskRequest,
    *,
    max_input_bytes: int,
    artifact_auth: ArtifactClientAuth,
    expected_miner_hotkey: str,
) -> ValidatorArtifactRequestReceipt:
    """Build an exact v2 request receipt without contacting a peer.

    This narrow helper exists for a chain-advertised endpoint that cannot even be
    represented as a safe HTTP destination. It performs the same regular-file,
    size and digest checks as a streamed request before signing anything, so a
    local input failure can never be converted into a miner-attributable zero.
    The caller records the invalid chain endpoint alongside this receipt as an
    auditable ``unreachable_endpoint`` observation.

    The function is synchronous because wallet-backed signing and descriptor IO
    are synchronous; async callers must run it off the event loop.
    """
    if max_input_bytes <= 0:
        raise ValueError("artifact input byte bound must be positive")
    if not expected_miner_hotkey:
        raise ValueError("artifact v2 requires the chain-attributed miner hotkey")
    _safe_task_id(request.task_id)
    metadata = MinerArtifactTaskRequest.from_local_request(request)
    fd, input_size = _open_regular_input(request.input_path, max_input_bytes)
    try:
        remaining = input_size
        digest = hashlib.sha256()
        while remaining:
            chunk = os.read(fd, min(_CHUNK, remaining))
            if not chunk:
                raise MinerArtifactInputError(
                    "miner input changed while preparing the signed request: "
                    f"expected {input_size} bytes"
                )
            remaining -= len(chunk)
            digest.update(chunk)
        actual = digest.hexdigest()
        if actual != request.input_digest:
            raise MinerArtifactInputError(
                "miner input digest changed while preparing the signed request: "
                f"declared {request.input_digest}, read {actual}"
            )
        try:
            claims, auth_headers = artifact_auth.sign_request(
                metadata,
                input_size=input_size,
                intended_miner_hotkey=expected_miner_hotkey,
            )
        except ArtifactAuthError as exc:
            raise MinerArtifactInputError(
                f"could not sign artifact v2 request: {exc}"
            ) from exc
        return ValidatorArtifactRequestReceipt(
            version=claims.version,
            validator_hotkey=claims.validator_hotkey,
            miner_hotkey=claims.miner_hotkey,
            timestamp=claims.timestamp,
            nonce=claims.nonce,
            input_size=claims.input_size,
            metadata=metadata,
            request_signature=auth_headers[MINER_REQUEST_SIGNATURE_HEADER],
        )
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


async def _input_chunks(
    fd: int, size: int, expected_digest: str, digest_box: list[str]
) -> AsyncIterator[bytes]:
    """Yield exactly the opened file's declared size, hashing the wire bytes."""
    remaining = size
    h = hashlib.sha256()
    while remaining:
        chunk = await asyncio.to_thread(os.read, fd, min(_CHUNK, remaining))
        if not chunk:
            raise MinerArtifactInputError(
                f"miner input changed while streaming: expected {size} bytes"
            )
        remaining -= len(chunk)
        h.update(chunk)
        yield chunk
    actual = h.hexdigest()
    digest_box.append(actual)
    if actual != expected_digest:
        raise MinerArtifactInputError(
            f"miner input digest changed while streaming: declared "
            f"{expected_digest}, streamed {actual}"
        )


def _positive_content_length(response: httpx.Response) -> int | None:
    value = response.headers.get("content-length")
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError as exc:
        raise MinerArtifactProtocolError(
            f"miner returned invalid Content-Length {value!r}"
        ) from exc
    if length < 0:
        raise MinerArtifactProtocolError(
            f"miner returned invalid Content-Length {value!r}"
        )
    return length


def _response_binding(
    response: httpx.Response, expected_task_id: str, *, expected_version: str
) -> tuple[str, float | None, int | None]:
    version = response.headers.get(MINER_ARTIFACT_VERSION_HEADER)
    if version != expected_version:
        raise MinerArtifactProtocolError(
            f"miner artifact version {version!r} != {expected_version!r}"
        )
    task_id = response.headers.get(MINER_TASK_ID_HEADER, "")
    if task_id != expected_task_id:
        raise MinerArtifactProtocolError(
            f"miner response task {task_id!r} != dispatched task {expected_task_id!r}"
        )
    digest = response.headers.get(MINER_OUTPUT_DIGEST_HEADER, "")
    if not _SHA256.fullmatch(digest):
        raise MinerArtifactProtocolError(
            f"miner returned invalid {MINER_OUTPUT_DIGEST_HEADER} header"
        )
    raw_seconds = response.headers.get(MINER_PROCESSING_SECONDS_HEADER)
    processing_seconds: float | None = None
    if raw_seconds not in (None, ""):
        try:
            processing_seconds = float(raw_seconds)
        except ValueError as exc:
            raise MinerArtifactProtocolError(
                f"miner returned invalid processing seconds {raw_seconds!r}"
            ) from exc
        if not math.isfinite(processing_seconds) or processing_seconds < 0:
            raise MinerArtifactProtocolError(
                f"miner returned invalid processing seconds {raw_seconds!r}"
            )
    return digest, processing_seconds, _positive_content_length(response)


async def _raise_for_status_bounded(response: httpx.Response) -> None:
    """Preserve a useful peer error without allowing an unbounded error body."""
    if 200 <= response.status_code < 300:
        return
    body = bytearray()
    async for chunk in response.aiter_bytes(_CHUNK):
        remaining = _MAX_ERROR_BODY - len(body)
        if remaining > 0:
            body.extend(chunk[:remaining])
        if len(body) >= _MAX_ERROR_BODY:
            break
    bounded = httpx.Response(
        response.status_code,
        headers=response.headers,
        content=bytes(body),
        request=response.request,
    )
    bounded.raise_for_status()


def _staging_file(output_root: Path, task_id: str) -> tuple[int, Path, Path]:
    """Create a private, task-scoped staging file below a caller-owned root."""
    safe_id = _safe_task_id(task_id)
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    task_dir = (root / safe_id).resolve()
    if task_dir == root or root not in task_dir.parents:
        raise MinerArtifactInputError(f"task_id {task_id!r} escapes output root")
    task_dir.mkdir(parents=False, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=task_dir)
    os.chmod(name, 0o600)
    return fd, Path(name), task_dir


def discard_downloaded_artifact(path: str | Path, output_root: str | Path) -> None:
    """Best-effort cleanup of one transfer known to belong to ``output_root``.

    The resolved-parent proof prevents a peer-supplied path from turning this
    helper into an arbitrary unlink.  It is used when a gateway loses its lease
    after a valid response arrived.
    """
    root = Path(output_root).resolve()
    target = Path(path)
    try:
        parent = target.parent.resolve()
    except OSError:
        return
    if parent == root or root not in parent.parents:
        return
    with contextlib.suppress(FileNotFoundError, OSError):
        target.unlink()
    with contextlib.suppress(OSError):
        parent.rmdir()


async def submit_miner_artifact(
    client: httpx.AsyncClient,
    base_url: str,
    request: MinerTaskRequest,
    *,
    output_dir: str | Path,
    max_input_bytes: int,
    max_output_bytes: int,
    headers: Mapping[str, str] | None = None,
    timeout: float,
    artifact_auth: ArtifactClientAuth | None = None,
    expected_miner_hotkey: str | None = None,
    allow_unsigned_v1: bool = False,
    _deadline: float | None = None,
    _after_restart_fence: bool = False,
) -> MinerTaskResponse:
    """Upload one file and atomically publish one verified downloaded output.

    No whole artifact is loaded into memory.  All partial files are removed on
    HTTP failure, disconnect, timeout, digest/size mismatch, or cancellation.
    ``base_url`` is operator/chain configuration; no URL from the miner response
    is ever followed. Production supplies ``artifact_auth`` plus the miner hotkey
    attributed by the chain. Unsigned v1 is available only through the explicit
    ``allow_unsigned_v1`` compatibility switch for report/dev fixtures.
    """
    if max_input_bytes <= 0 or max_output_bytes <= 0:
        raise ValueError("artifact byte bounds must be positive")
    if timeout <= 0:
        raise ValueError("artifact timeout must be positive")
    now = time.monotonic()
    deadline = now + timeout if _deadline is None else _deadline
    attempt_timeout = deadline - now
    if attempt_timeout <= 0:
        if _after_restart_fence:
            raise MinerArtifactColdStart(
                "miner restart fence remained active until the artifact deadline"
            )
        raise TimeoutError("miner artifact deadline expired before transfer")
    _safe_task_id(request.task_id)
    metadata = MinerArtifactTaskRequest.from_local_request(request)
    metadata_value = encode_miner_task_metadata(metadata)
    fd, input_size = _open_regular_input(request.input_path, max_input_bytes)
    digest_box: list[str] = []
    auth_claims: ArtifactRequestClaims | None = None
    request_receipt: ValidatorArtifactRequestReceipt | None = None
    if artifact_auth is None:
        if not allow_unsigned_v1:
            os.close(fd)
            raise ValueError(
                "artifact v2 authentication is required; unsigned v1 needs the "
                "explicit report/dev allow_unsigned_v1 opt-in"
            )
        if expected_miner_hotkey:
            os.close(fd)
            raise ValueError(
                "an expected miner hotkey cannot be verified in unsigned v1"
            )
        request_version = MINER_ARTIFACT_VERSION
        auth_headers: dict[str, str] = {}
    else:
        if not expected_miner_hotkey:
            os.close(fd)
            raise ValueError("artifact v2 requires the chain-attributed miner hotkey")
        try:
            auth_claims, auth_headers = artifact_auth.sign_request(
                metadata,
                input_size=input_size,
                intended_miner_hotkey=expected_miner_hotkey,
            )
        except ArtifactAuthError as exc:
            os.close(fd)
            raise MinerArtifactInputError(
                f"could not sign artifact v2 request: {exc}"
            ) from exc
        except BaseException:
            # A wallet/transport-backed signer can fail independently of the
            # canonical auth helpers. The streamed-input descriptor was opened
            # before signing so its size is part of the signature; never leak
            # that descriptor when the injected signer raises or cancellation
            # lands at this seam.
            with contextlib.suppress(OSError):
                os.close(fd)
            raise
        request_version = MINER_ARTIFACT_AUTH_VERSION
        request_receipt = ValidatorArtifactRequestReceipt(
            version=auth_claims.version,
            validator_hotkey=auth_claims.validator_hotkey,
            miner_hotkey=auth_claims.miner_hotkey,
            timestamp=auth_claims.timestamp,
            nonce=auth_claims.nonce,
            input_size=auth_claims.input_size,
            metadata=metadata,
            request_signature=auth_headers[MINER_REQUEST_SIGNATURE_HEADER],
        )
    request_headers = dict(headers or {})
    request_headers.update(
        {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(input_size),
            MINER_TASK_METADATA_HEADER: metadata_value,
            MINER_ARTIFACT_VERSION_HEADER: request_version,
            **auth_headers,
        }
    )
    stage: Path | None = None
    stage_fd: int | None = None
    try:
        async with asyncio.timeout(attempt_timeout):
            async with client.stream(
                "POST",
                base_url.rstrip("/") + MINER_ARTIFACT_ROUTE,
                headers=request_headers,
                content=_input_chunks(fd, input_size, request.input_digest, digest_box),
                timeout=attempt_timeout,
            ) as response:
                await _raise_for_status_bounded(response)
                # A peer is allowed to answer before it has consumed the request
                # body (for example an HTTP/1.1 early response).  In that case
                # httpx may close/cancel the request iterator and ``digest_box``
                # remains empty.  Never accept an output unless the exact opened
                # input was completely put on the wire and its final digest was
                # checked by ``_input_chunks``.
                if not digest_box:
                    raise MinerArtifactProtocolError(
                        "miner responded before the complete input was uploaded "
                        "and verified"
                    )
                if digest_box != [request.input_digest]:
                    observed = digest_box[0]
                    raise MinerArtifactInputError(
                        "miner input verification did not complete successfully: "
                        f"expected {request.input_digest}, got {observed}"
                    )
                claimed, processing_seconds, declared_size = _response_binding(
                    response, request.task_id, expected_version=request_version
                )
                if declared_size is not None and declared_size > max_output_bytes:
                    raise MinerArtifactTooLarge(
                        f"miner declared {declared_size} output bytes; maximum is "
                        f"{max_output_bytes}"
                    )
                stage_fd, stage, task_dir = _staging_file(
                    Path(output_dir), request.task_id
                )
                h = hashlib.sha256()
                received = 0
                with os.fdopen(stage_fd, "wb") as sink:
                    stage_fd = None
                    async for chunk in response.aiter_bytes(_CHUNK):
                        received += len(chunk)
                        if received > max_output_bytes:
                            raise MinerArtifactTooLarge(
                                f"miner output crossed {max_output_bytes} bytes"
                            )
                        h.update(chunk)
                        sink.write(chunk)
                    sink.flush()
                    os.fsync(sink.fileno())
                if received < 1:
                    raise MinerArtifactIntegrityError("miner returned an empty output")
                if declared_size is not None and received != declared_size:
                    raise MinerArtifactIntegrityError(
                        f"miner Content-Length was {declared_size}, received {received}"
                    )
                actual = h.hexdigest()
                if actual != claimed:
                    raise MinerArtifactIntegrityError(
                        f"miner output digest mismatch: claimed {claimed}, got {actual}"
                    )
                if artifact_auth is not None:
                    assert auth_claims is not None
                    raw_processing = response.headers.get(
                        MINER_PROCESSING_SECONDS_HEADER, ""
                    )
                    try:
                        artifact_auth.verify_response(
                            response.headers,
                            auth_claims,
                            metadata,
                            output_digest=actual,
                            output_size=received,
                            processing_seconds=raw_processing,
                        )
                    except ArtifactAuthError as exc:
                        raise MinerArtifactProtocolError(
                            f"miner artifact v2 authentication failed: {exc}"
                        ) from exc
                    signature = response.headers.get(
                        MINER_RESPONSE_SIGNATURE_HEADER, ""
                    )
                    receipt = MinerArtifactReceipt(
                        version=auth_claims.version,
                        validator_hotkey=auth_claims.validator_hotkey,
                        miner_hotkey=auth_claims.miner_hotkey,
                        timestamp=auth_claims.timestamp,
                        nonce=auth_claims.nonce,
                        input_size=auth_claims.input_size,
                        metadata=metadata,
                        request_signature=auth_headers[MINER_REQUEST_SIGNATURE_HEADER],
                        output_digest=actual,
                        output_size=received,
                        processing_seconds=raw_processing,
                        response_signature=signature,
                    )
                    receipt_json = receipt.model_dump(mode="json")
                else:
                    receipt_json = None
                # One unique immutable filename per transfer: a stale gateway
                # attempt cannot overwrite a later attempt's still-live bytes.
                final = task_dir / f"output-{uuid.uuid4().hex}.mp4"
                os.replace(stage, final)
                stage = None
                return MinerTaskResponse(
                    task_id=request.task_id,
                    output_path=str(final),
                    output_digest=actual,
                    processing_seconds=processing_seconds,
                    artifact_receipt=receipt_json,
                )
    except httpx.HTTPStatusError as exc:
        # HTTP 425 is the authenticated miner's cold-start/replay-cache fence.
        # The original timestamp/nonce must never be replayed: close this input
        # descriptor and recursively start a fresh signed attempt, while passing
        # only the remainder of the caller's single overall deadline.
        if exc.response.status_code != 425 or artifact_auth is None:
            _attach_request_evidence(exc, request_receipt, base_url)
            raise
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            cold_start = MinerArtifactColdStart(
                "miner restart fence remained active until the artifact deadline"
            )
            raise _attach_request_evidence(
                cold_start, request_receipt, base_url
            ) from exc
        await _restart_fence_backoff(min(_RESTART_FENCE_BACKOFF_SECONDS, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            cold_start = MinerArtifactColdStart(
                "miner restart fence remained active until the artifact deadline"
            )
            raise _attach_request_evidence(
                cold_start, request_receipt, base_url
            ) from exc
        # Avoid retaining one opened input descriptor per retry frame. The next
        # call reopens and revalidates the same caller-owned regular file.
        with contextlib.suppress(OSError):
            os.close(fd)
        fd = -1
        return await submit_miner_artifact(
            client,
            base_url,
            request,
            output_dir=output_dir,
            max_input_bytes=max_input_bytes,
            max_output_bytes=max_output_bytes,
            headers=headers,
            timeout=remaining,
            artifact_auth=artifact_auth,
            expected_miner_hotkey=expected_miner_hotkey,
            allow_unsigned_v1=allow_unsigned_v1,
            _deadline=deadline,
            _after_restart_fence=True,
        )
    except TimeoutError as exc:
        if _after_restart_fence:
            cold_start = MinerArtifactColdStart(
                "miner restart fence retry exhausted the artifact deadline"
            )
            raise _attach_request_evidence(
                cold_start, request_receipt, base_url
            ) from exc
        _attach_request_evidence(exc, request_receipt, base_url)
        raise
    except BaseException as exc:
        # Preserve cancellation semantics while attaching the already-signed request
        # facts. The caller decides whether this exception class is miner-attributable;
        # merely having a receipt must never turn local/scorer/storage trouble punitive.
        _attach_request_evidence(exc, request_receipt, base_url)
        raise
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
        if stage_fd is not None:
            with contextlib.suppress(OSError):
                os.close(stage_fd)
        if stage is not None:
            with contextlib.suppress(FileNotFoundError, OSError):
                stage.unlink()
            with contextlib.suppress(OSError):
                stage.parent.rmdir()
