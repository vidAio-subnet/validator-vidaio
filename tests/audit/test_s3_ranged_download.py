"""Large S3 objects are fetched as concurrent, ETag-pinned byte ranges."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest

from vidaio.audit import store as store_module
from vidaio.audit.store import ArtifactTooLargeError, IntegrityError, _RealS3Transport


class _Body:
    def __init__(self, payload: bytes, *, fail_after: int | None = None) -> None:
        self._payload = payload
        self._fail_after = fail_after
        self.closed = False

    def iter_chunks(self, chunk_size: int):  # noqa: ANN201 - SDK shaped double
        sent = 0
        for start in range(0, len(self._payload), chunk_size):
            chunk = self._payload[start : start + chunk_size]
            if self._fail_after is not None and sent + len(chunk) > self._fail_after:
                keep = self._fail_after - sent
                if keep > 0:
                    yield chunk[:keep]
                raise ConnectionError("stream reset")
            sent += len(chunk)
            yield chunk

    def close(self) -> None:
        self.closed = True


class _PreconditionFailed(Exception):
    response = {
        "ResponseMetadata": {"HTTPStatusCode": 412},
        "Error": {"Code": "PreconditionFailed"},
    }


class RangedS3Client:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.calls: list[dict[str, object]] = []
        self.threads: set[str] = set()
        self.fail_once_at: dict[int, int] = {}  # range start -> bytes before reset
        self.always_fail_start: int | None = None
        self.etag_override: str | None = None
        self._lock = threading.Lock()

    def _etag(self, key: str) -> str:
        return '"' + hashlib.md5(self.objects[key]).hexdigest() + '"'  # noqa: S324

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:  # noqa: N803
        payload = self.objects[Key]
        return {"ContentLength": len(payload), "ETag": self._etag(Key)}

    def get_object(self, **kwargs: object) -> dict[str, object]:
        key = str(kwargs["Key"])
        payload = self.objects[key]
        with self._lock:
            self.calls.append(kwargs)
            self.threads.add(threading.current_thread().name)
        spec = kwargs.get("Range")
        if spec is None:
            return {"Body": _Body(payload)}
        current = self.etag_override or self._etag(key)
        if kwargs.get("IfMatch") != current:
            raise _PreconditionFailed()
        first, last = (int(v) for v in str(spec).removeprefix("bytes=").split("-"))
        fail_after = None
        with self._lock:
            if first == self.always_fail_start:
                fail_after = 0
            elif first in self.fail_once_at:
                fail_after = self.fail_once_at.pop(first)
        return {"Body": _Body(payload[first : last + 1], fail_after=fail_after)}


def _transport(client: object) -> _RealS3Transport:
    transport = object.__new__(_RealS3Transport)
    transport._bucket = "audit-bucket"
    transport._prefix = "launch"
    transport._client = client
    return transport


@pytest.fixture(autouse=True)
def _small_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_module, "_S3_PARALLEL_MIN_BYTES", 1000)
    monkeypatch.setattr(store_module, "_S3_PARALLEL_PART_BYTES", 300)
    monkeypatch.setattr(store_module, "_CHUNK", 64)
    monkeypatch.delenv("VIDAIO_S3_PARALLEL_STREAMS", raising=False)


def _payload(size: int) -> bytes:
    return bytes((i * 31 + 7) % 251 for i in range(size))


def test_large_object_is_fetched_as_pinned_ranges(tmp_path: Path) -> None:
    data = _payload(2050)
    client = RangedS3Client({"launch/reference/x": data})
    target = tmp_path / "out"

    _transport(client).get_file("reference/x", target, max_bytes=len(data))

    assert target.read_bytes() == data
    ranges = sorted(str(call["Range"]) for call in client.calls)
    assert len(ranges) == 7  # ceil(2050 / 300)
    assert "bytes=0-299" in ranges and "bytes=1800-2049" in ranges
    etag = client._etag("launch/reference/x")
    assert {call["IfMatch"] for call in client.calls} == {etag}
    assert all(name.startswith("s3-range") for name in client.threads)


def test_small_object_keeps_the_single_stream_read(tmp_path: Path) -> None:
    data = _payload(999)
    client = RangedS3Client({"launch/k": data})
    target = tmp_path / "out"

    _transport(client).get_file("k", target, max_bytes=5000)

    assert target.read_bytes() == data
    assert [call.get("Range") for call in client.calls] == [None]


def test_streams_env_one_disables_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIDAIO_S3_PARALLEL_STREAMS", "1")
    data = _payload(4000)
    client = RangedS3Client({"launch/k": data})
    target = tmp_path / "out"

    _transport(client).get_file("k", target, max_bytes=5000)

    assert target.read_bytes() == data
    assert [call.get("Range") for call in client.calls] == [None]


@pytest.mark.parametrize("raw", ["", "abc", "0", "-4", "999"])
def test_streams_env_is_clamped(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("VIDAIO_S3_PARALLEL_STREAMS", raw)
    assert 1 <= store_module._parallel_streams() <= 32


def test_oversized_object_is_refused_before_any_download(tmp_path: Path) -> None:
    client = RangedS3Client({"launch/k": _payload(3000)})
    target = tmp_path / "out"

    with pytest.raises(ArtifactTooLargeError):
        _transport(client).get_file("k", target, max_bytes=2999)

    assert client.calls == []
    assert not target.exists()


def test_a_reset_range_resumes_from_the_bytes_already_written(tmp_path: Path) -> None:
    data = _payload(2050)
    client = RangedS3Client({"launch/k": data})
    client.fail_once_at[600] = 130
    target = tmp_path / "out"

    _transport(client).get_file("k", target, max_bytes=len(data))

    assert target.read_bytes() == data
    assert "bytes=730-899" in {str(call["Range"]) for call in client.calls}


def test_a_range_that_keeps_failing_fails_the_download_and_leaves_no_file(
    tmp_path: Path,
) -> None:
    client = RangedS3Client({"launch/k": _payload(2050)})
    client.always_fail_start = 900
    target = tmp_path / "out"

    with pytest.raises(ConnectionError):
        _transport(client).get_file("k", target, max_bytes=5000)

    assert not target.exists()


def test_object_replaced_mid_download_is_an_integrity_error(tmp_path: Path) -> None:
    client = RangedS3Client({"launch/k": _payload(2050)})
    client.etag_override = '"someone-else"'
    target = tmp_path / "out"

    with pytest.raises(IntegrityError, match="changed while"):
        _transport(client).get_file("k", target, max_bytes=5000)

    assert not target.exists()


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    client = RangedS3Client({"launch/k": _payload(2050)})
    target = tmp_path / "out"
    target.write_bytes(b"keep")

    with pytest.raises(FileExistsError):
        _transport(client).get_file("k", target, max_bytes=5000)

    assert target.read_bytes() == b"keep"


def test_client_without_head_falls_back_to_the_single_stream(tmp_path: Path) -> None:
    class NoHead:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload
            self.calls = 0

        def get_object(self, **kwargs: object) -> dict[str, object]:
            self.calls += 1
            assert "Range" not in kwargs
            return {"Body": _Body(self.payload)}

    data = _payload(4000)
    client = NoHead(data)
    target = tmp_path / "out"

    _transport(client).get_file("k", target, max_bytes=5000)

    assert target.read_bytes() == data and client.calls == 1
