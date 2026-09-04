from pathlib import Path
import sys
from types import ModuleType

import pytest

from vidaio.audit.canonical import sha256_hex
from vidaio.audit.config import AuditConfig
from vidaio.audit.store import (
    AesGcmEnvelope,
    ArtifactTooLargeError,
    FINALIZED_MARKER,
    ArtifactKind,
    ArtifactRef,
    HippiusStore,
    IntegrityError,
    LocalFsStore,
    NotConfiguredError,
    PassthroughEnvelope,
    ReadOnlyStoreError,
    SEALED_KINDS,
    SealedArtifactAccessError,
    S3Store,
    SetAlreadyFinalizedError,
    SetNotFinalizedError,
    WriteOnceConflictError,
    _RealS3Transport,
    backend_key,
    finalized_marker_key,
    make_store,
    make_public_store,
    make_unsealed_writer_store,
    released_backend_key,
    set_member_key,
)


def test_put_get_round_trip(store: LocalFsStore) -> None:
    data = b"some video bytes"
    ref = store.put(data, ArtifactKind.MINER_OUTPUT)
    assert ref.kind is ArtifactKind.MINER_OUTPUT
    assert ref.byte_size == len(data)
    assert ref.backend_key == backend_key(ArtifactKind.MINER_OUTPUT, ref.digest)
    assert store.exists(ref)
    assert store.get(ref) == data
    with store.open_stream(ref) as stream:
        assert stream.read() == data


def test_sharded_layout(tmp_path: Path) -> None:
    store = LocalFsStore(tmp_path)
    ref = store.put(b"x", ArtifactKind.MANIFEST)
    expected = tmp_path / "manifest" / ref.digest[:2] / ref.digest[2:4] / ref.digest
    assert expected.is_file()


def test_write_once_is_noop(tmp_path: Path) -> None:
    store = LocalFsStore(tmp_path)
    ref1 = store.put(b"same bytes", ArtifactKind.CHALLENGE_INPUT)
    path = tmp_path / ref1.backend_key
    before = path.stat().st_mtime_ns
    ref2 = store.put(b"same bytes", ArtifactKind.CHALLENGE_INPUT)
    assert ref1 == ref2
    assert path.stat().st_mtime_ns == before  # file untouched


def test_same_bytes_different_kinds_are_distinct(store: LocalFsStore) -> None:
    a = store.put(b"payload", ArtifactKind.CHALLENGE_INPUT)
    b = store.put(b"payload", ArtifactKind.MINER_OUTPUT)
    assert a.digest == b.digest
    assert a.backend_key != b.backend_key


def test_put_file_round_trip(tmp_path: Path, store: LocalFsStore) -> None:
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake mp4 payload" * 1000)
    ref = store.put_file(src, ArtifactKind.CHALLENGE_INPUT)
    assert ref.byte_size == src.stat().st_size
    assert store.get(ref) == src.read_bytes()
    # putting the same file again is a no-op returning the same ref
    assert store.put_file(src, ArtifactKind.CHALLENGE_INPUT) == ref


def test_corruption_detected_on_read(tmp_path: Path) -> None:
    store = LocalFsStore(tmp_path)
    ref = store.put(b"pristine artifact bytes", ArtifactKind.SCORE_PACKET)
    path = tmp_path / ref.backend_key
    raw = bytearray(path.read_bytes())
    raw[3] ^= 0xFF  # flip one byte on disk
    path.write_bytes(bytes(raw))
    with pytest.raises(IntegrityError):
        store.get(ref)


def test_missing_artifact_raises(store: LocalFsStore) -> None:
    ref = ArtifactRef(
        digest="0" * 64,
        kind=ArtifactKind.MANIFEST,
        byte_size=1,
        backend_key=backend_key(ArtifactKind.MANIFEST, "0" * 64),
    )
    assert not store.exists(ref)
    with pytest.raises(FileNotFoundError):
        store.get(ref)


class XorEnvelope:
    """Toy seal: proves sealed-at-rest bytes differ from plaintext."""

    def seal(self, data: bytes) -> bytes:
        return bytes(b ^ 0x5A for b in data)

    def unseal(self, sealed: bytes) -> bytes:
        return bytes(b ^ 0x5A for b in sealed)


def test_reference_original_sealed_at_rest(tmp_path: Path) -> None:
    store = LocalFsStore(tmp_path, envelope=XorEnvelope())
    plaintext = b"holdout original -- sealed until asset retirement"
    ref = store.put(plaintext, ArtifactKind.REFERENCE_ORIGINAL)
    on_disk = (tmp_path / ref.backend_key).read_bytes()
    assert on_disk != plaintext  # encrypted at rest
    assert store.get(ref) == plaintext  # digest covers the plaintext
    with store.open_stream(ref) as stream:
        assert stream.read() == plaintext
    # non-sealed kinds stay plaintext on disk
    ref2 = store.put(plaintext, ArtifactKind.MINER_OUTPUT)
    assert (tmp_path / ref2.backend_key).read_bytes() == plaintext


def test_submission_archive_is_private_until_crowned_release(tmp_path: Path) -> None:
    """Executable disclosure uses its own sealed, releasable namespace."""
    private = LocalFsStore(tmp_path, envelope=XorEnvelope())
    public = LocalFsStore(
        tmp_path,
        envelope=PassthroughEnvelope(),
        public_read_only=True,
        allow_sealed_operations=False,
    )
    payload = b"exact contender source archive"
    ref = private.put(payload, ArtifactKind.SUBMISSION_ARCHIVE)

    assert (tmp_path / ref.backend_key).read_bytes() != payload
    assert not private.is_released(ref)
    with pytest.raises(FileNotFoundError, match="has not been released"):
        public.get(ref)

    private.release(ref)

    assert private.is_released(ref)
    assert public.get(ref) == payload
    assert (
        tmp_path / released_backend_key(ArtifactKind.SUBMISSION_ARCHIVE, ref.digest)
    ).read_bytes() == payload


def test_is_released_rejects_corrupt_or_wrong_sized_public_copy(tmp_path: Path) -> None:
    store = LocalFsStore(tmp_path, envelope=XorEnvelope())
    ref = store.put(b"winning source archive", ArtifactKind.SUBMISSION_ARCHIVE)
    store.release(ref)
    release_path = tmp_path / released_backend_key(ref.kind, ref.digest)

    release_path.write_bytes(b"corrupt source archive")
    assert store.is_released(ref) is False

    release_path.write_bytes(b"x" * ref.byte_size)
    assert store.is_released(ref) is False


def test_retired_reference_is_public_without_master_envelope(tmp_path: Path) -> None:
    plaintext = b"holdout becomes public only after retirement"
    sealed = LocalFsStore(tmp_path, envelope=XorEnvelope())
    ref = sealed.put(plaintext, ArtifactKind.REFERENCE_ORIGINAL)
    public = LocalFsStore(tmp_path, envelope=PassthroughEnvelope())
    assert not sealed.is_released(ref)
    with pytest.raises(IntegrityError):
        public.get(ref)  # encrypted object is not public/decryptable yet

    sealed.release(ref)
    assert sealed.is_released(ref)
    assert (
        tmp_path / released_backend_key(ref.kind, ref.digest)
    ).read_bytes() == plaintext
    assert public.get(ref) == plaintext
    sealed.release(ref)  # idempotent and verify-on-read


def test_hippius_store_raises_not_configured() -> None:
    cfg = AuditConfig(backend="hippius")
    store = HippiusStore(cfg)
    with pytest.raises(NotConfiguredError, match="storage|hippius_bucket"):
        store.put(b"data", ArtifactKind.MANIFEST)
    ref = ArtifactRef(
        digest="a" * 64,
        kind=ArtifactKind.MANIFEST,
        byte_size=4,
        backend_key=backend_key(ArtifactKind.MANIFEST, "a" * 64),
    )
    with pytest.raises(NotConfiguredError):
        store.get(ref)
    with pytest.raises(NotConfiguredError):
        store.exists(ref)


def test_make_store_dispatches(tmp_path: Path) -> None:
    local = make_store(
        AuditConfig(backend="local", local_root=tmp_path), envelope=XorEnvelope()
    )
    assert isinstance(local, LocalFsStore)
    hippius = make_store(AuditConfig(backend="hippius"), envelope=XorEnvelope())
    assert isinstance(hippius, HippiusStore)


def test_make_store_refuses_plaintext_holdout_by_default(tmp_path: Path) -> None:
    cfg = AuditConfig(backend="local", local_root=tmp_path)
    with pytest.raises(NotConfiguredError, match="plaintext holdout"):
        make_store(cfg)  # no envelope -> implicit PassthroughEnvelope
    with pytest.raises(NotConfiguredError, match="plaintext holdout"):
        make_store(cfg, envelope=PassthroughEnvelope())  # explicit is no better
    with pytest.raises(NotConfiguredError, match="plaintext holdout"):
        make_store(AuditConfig(backend="hippius"))  # gated for every backend


def test_aes_gcm_envelope_round_trip_and_tamper(monkeypatch) -> None:
    pytest.importorskip("cryptography")
    monkeypatch.setenv("VIDAIO_AUDIT_HOLDOUT_KEY", "42" * 32)
    envelope = AesGcmEnvelope.from_env("VIDAIO_AUDIT_HOLDOUT_KEY")
    sealed = envelope.seal(b"private holdout")
    assert b"private holdout" not in sealed
    assert envelope.unseal(sealed) == b"private holdout"
    tampered = bytearray(sealed)
    tampered[-1] ^= 1
    with pytest.raises(IntegrityError):
        envelope.unseal(bytes(tampered))


def test_aes_gcm_file_envelope_streams_and_matches_byte_format(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("cryptography")
    monkeypatch.setenv("VIDAIO_AUDIT_HOLDOUT_KEY", "24" * 32)
    envelope = AesGcmEnvelope.from_env("VIDAIO_AUDIT_HOLDOUT_KEY")
    source = tmp_path / "reference.mp4"
    source.write_bytes((b"large-reference-chunk" * 100_000) + b"tail")
    sealed = tmp_path / "reference.aesgcm"
    restored = tmp_path / "restored.mp4"

    envelope.seal_file(source, sealed)
    assert source.read_bytes() not in sealed.read_bytes()
    assert envelope.unseal(sealed.read_bytes()) == source.read_bytes()
    envelope.unseal_file(sealed, restored)
    assert restored.read_bytes() == source.read_bytes()


def test_make_store_plaintext_holdout_requires_explicit_opt_in(tmp_path: Path) -> None:
    cfg = AuditConfig(
        backend="local", local_root=tmp_path, allow_plaintext_holdout=True
    )
    assert isinstance(make_store(cfg), LocalFsStore)
    assert isinstance(make_store(cfg, envelope=PassthroughEnvelope()), LocalFsStore)
    # a real envelope never needs the opt-in
    sealed = make_store(
        AuditConfig(backend="local", local_root=tmp_path), envelope=XorEnvelope()
    )
    assert isinstance(sealed, LocalFsStore)


# --------------------------------------------------------------------------------------
# The new EPOCH_LOG kind.
# --------------------------------------------------------------------------------------


def test_epoch_log_is_a_first_class_kind(store: LocalFsStore) -> None:
    assert ArtifactKind.EPOCH_LOG.value == "epoch_log"
    data = b'{"epoch_id": 41822, "weight_u16": []}'
    ref = store.put(data, ArtifactKind.EPOCH_LOG)
    assert ref.kind is ArtifactKind.EPOCH_LOG
    assert ref.backend_key.startswith("epoch_log/")
    assert (
        store.get(ref) == data
    )  # content-addressed + verify-on-read like any artifact


# --------------------------------------------------------------------------------------
# The S3 object-store backend, exercised against a FAKE in-memory transport (no boto3).
# --------------------------------------------------------------------------------------


class FakeObjectTransport:
    """In-memory stand-in for _ObjectTransport — the S3Store logic runs against
    this so the whole backend is testable WITHOUT boto3 installed."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.put_calls: list[str] = []
        self.put_file_calls: list[str] = []
        self.get_bytes_calls: list[str] = []
        self.get_file_calls: list[str] = []

    def put_bytes(self, key: str, payload: bytes) -> None:
        self.put_calls.append(key)
        if key in self.objects:
            raise WriteOnceConflictError(f"fake conditional create conflict for {key}")
        self.objects[key] = payload

    def put_file(self, key: str, path: Path) -> None:
        self.put_file_calls.append(key)
        self.put_bytes(key, path.read_bytes())

    def get_bytes(self, key: str) -> bytes:
        self.get_bytes_calls.append(key)
        return self.objects[key]

    def get_file(self, key: str, path: Path, *, max_bytes: int) -> None:
        self.get_file_calls.append(key)
        payload = self.objects[key]
        if len(payload) > max_bytes:
            raise ArtifactTooLargeError("fake download crossed bound")
        path.write_bytes(payload)

    def head(self, key: str) -> int | None:
        obj = self.objects.get(key)
        return None if obj is None else len(obj)

    def exists(self, key: str) -> bool:
        return key in self.objects


class _S3PreconditionFailed(Exception):
    response = {
        "ResponseMetadata": {"HTTPStatusCode": 412},
        "Error": {"Code": "PreconditionFailed"},
    }


class ConditionalS3Client:
    """Small PutObject double that enforces ``If-None-Match: *``."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls: list[dict[str, object]] = []

    def put_object(self, **kwargs: object) -> None:
        self.calls.append(kwargs)
        assert kwargs["IfNoneMatch"] == "*"
        key = str(kwargs["Key"])
        if key in self.objects:
            raise _S3PreconditionFailed()
        body = kwargs["Body"]
        payload = body.read() if hasattr(body, "read") else body
        assert isinstance(payload, bytes)
        self.objects[key] = payload


class InjectedRealS3Transport(_RealS3Transport):
    """Real conditional-write code plus dependency-free readback for store tests."""

    def get_bytes(self, key: str) -> bytes:
        return self._client.objects[self._full(key)]

    def head(self, key: str) -> int | None:
        payload = self._client.objects.get(self._full(key))
        return None if payload is None else len(payload)

    def exists(self, key: str) -> bool:
        return self._full(key) in self._client.objects


def _real_s3_transport(client: object) -> _RealS3Transport:
    """Inject a client without importing boto3 or opening a network connection."""

    transport = object.__new__(InjectedRealS3Transport)
    transport._bucket = "audit-bucket"
    transport._prefix = "launch"
    transport._client = client
    return transport


def _capture_real_s3_client_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    path_style: bool,
    anonymous: bool,
) -> tuple[dict[str, object], object]:
    """Construct the real transport seam with dependency-only SDK doubles."""

    captured: dict[str, object] = {}
    unsigned = object()

    class CapturingConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    boto3 = ModuleType("boto3")

    def client(service_name: str, **kwargs: object) -> object:
        captured["service_name"] = service_name
        captured["client_kwargs"] = kwargs
        return object()

    boto3.client = client  # type: ignore[attr-defined]
    botocore = ModuleType("botocore")
    botocore.UNSIGNED = unsigned  # type: ignore[attr-defined]
    botocore_config = ModuleType("botocore.config")
    botocore_config.Config = CapturingConfig  # type: ignore[attr-defined]
    botocore.config = botocore_config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)

    _RealS3Transport(
        bucket="audit-bucket",
        prefix="launch",
        region="eu-west-3",
        endpoint="https://objects.example.invalid",
        access_key=None if anonymous else "access",
        secret_key=None if anonymous else "secret",
        path_style=path_style,
        anonymous=anonymous,
    )
    return captured, unsigned


@pytest.mark.parametrize(
    ("path_style", "anonymous"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_real_s3_client_has_bounded_timeout_and_retry_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    path_style: bool,
    anonymous: bool,
) -> None:
    captured, unsigned = _capture_real_s3_client_config(
        monkeypatch, path_style=path_style, anonymous=anonymous
    )

    assert captured["service_name"] == "s3"
    client_kwargs = captured["client_kwargs"]
    assert isinstance(client_kwargs, dict)
    sdk_config = client_kwargs["config"]
    config_kwargs = sdk_config.kwargs  # type: ignore[attr-defined]
    assert config_kwargs["connect_timeout"] == 10
    assert config_kwargs["read_timeout"] == 30
    assert config_kwargs["retries"] == {
        "mode": "standard",
        "total_max_attempts": 3,
    }
    if anonymous:
        assert config_kwargs["signature_version"] is unsigned
    elif path_style:
        assert config_kwargs["signature_version"] == "s3v4"
    else:
        assert "signature_version" not in config_kwargs
    if path_style:
        assert config_kwargs["s3"] == {"addressing_style": "path"}
    else:
        assert "s3" not in config_kwargs


def test_real_s3_bytes_use_atomic_conditional_create() -> None:
    client = ConditionalS3Client()
    transport = _real_s3_transport(client)

    transport.put_bytes("finalized/epoch=7/log.json", b"winner")
    with pytest.raises(WriteOnceConflictError):
        transport.put_bytes("finalized/epoch=7/log.json", b"must-not-overwrite")

    assert client.objects["launch/finalized/epoch=7/log.json"] == b"winner"
    assert [call["IfNoneMatch"] for call in client.calls] == ["*", "*"]


def test_real_s3_files_use_atomic_conditional_create(tmp_path: Path) -> None:
    client = ConditionalS3Client()
    transport = _real_s3_transport(client)
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first-file")
    second.write_bytes(b"must-not-overwrite")

    transport.put_file("miner_output/object", first)
    with pytest.raises(WriteOnceConflictError):
        transport.put_file("miner_output/object", second)

    assert client.objects["launch/miner_output/object"] == b"first-file"
    assert client.calls[0]["ContentLength"] == len(b"first-file")
    assert [call["IfNoneMatch"] for call in client.calls] == ["*", "*"]


def test_real_s3_does_not_hide_non_precondition_failures() -> None:
    class ConcurrentWriteClient:
        def put_object(self, **_kwargs: object) -> None:
            error = RuntimeError("concurrent write must be retried")
            error.response = {  # type: ignore[attr-defined]
                "ResponseMetadata": {"HTTPStatusCode": 409},
                "Error": {"Code": "ConditionalRequestConflict"},
            }
            raise error

    with pytest.raises(RuntimeError, match="must be retried"):
        _real_s3_transport(ConcurrentWriteClient()).put_bytes("key", b"payload")


def test_s3_set_member_repeat_is_idempotent_but_different_bytes_conflict() -> None:
    client = ConditionalS3Client()
    store = S3Store(
        AuditConfig(backend="s3", allow_plaintext_holdout=True),
        transport=_real_s3_transport(client),
    )
    prefix = "finalized/epoch=19"

    store.put_set_member(prefix, "log.json", b"first")
    store.put_set_member(prefix, "log.json", b"first")
    with pytest.raises(WriteOnceConflictError):
        store.put_set_member(prefix, "log.json", b"different")

    assert client.objects["launch/finalized/epoch=19/log.json"] == b"first"
    assert all(call["IfNoneMatch"] == "*" for call in client.calls)


def s3_store(**cfg: object) -> tuple[S3Store, FakeObjectTransport]:
    transport = FakeObjectTransport()
    store = S3Store(
        AuditConfig(backend="s3", allow_plaintext_holdout=True, **cfg),  # type: ignore[arg-type]
        transport=transport,
    )
    return store, transport


def test_s3store_put_get_round_trip() -> None:
    store, transport = s3_store()
    data = b"some video bytes"
    ref = store.put(data, ArtifactKind.MINER_OUTPUT)
    assert ref.backend_key == backend_key(ArtifactKind.MINER_OUTPUT, ref.digest)
    assert ref.backend_key in transport.objects  # keyed by (kind, digest)
    assert store.exists(ref)
    assert store.get(ref) == data
    with store.open_stream(ref) as stream:
        assert stream.read() == data


def test_s3store_releases_verified_plaintext_under_public_prefix() -> None:
    transport = FakeObjectTransport()
    sealed = S3Store(
        AuditConfig(backend="s3"), envelope=XorEnvelope(), transport=transport
    )
    ref = sealed.put(b"retired reference", ArtifactKind.REFERENCE_ORIGINAL)
    assert transport.objects[ref.backend_key] != b"retired reference"
    sealed.release(ref)
    release_key = released_backend_key(ref.kind, ref.digest)
    assert transport.objects[release_key] == b"retired reference"
    public = S3Store(
        AuditConfig(backend="s3", allow_plaintext_holdout=True),
        envelope=PassthroughEnvelope(),
        transport=transport,
    )
    assert public.get(ref) == b"retired reference"


def test_s3_is_released_requires_bounded_read_and_content_verification() -> None:
    transport = FakeObjectTransport()
    sealed = S3Store(
        AuditConfig(backend="s3"), envelope=XorEnvelope(), transport=transport
    )
    ref = sealed.put(b"winning source archive", ArtifactKind.SUBMISSION_ARCHIVE)
    sealed.release(ref)
    key = released_backend_key(ref.kind, ref.digest)
    assert sealed.is_released(ref) is True

    transport.objects[key] = b"x" * ref.byte_size
    assert sealed.is_released(ref) is False


def test_s3_public_is_released_fails_when_anonymous_body_read_is_denied() -> None:
    class AnonymousReadDeniedTransport(FakeObjectTransport):
        def get_file(self, key: str, path: Path, *, max_bytes: int) -> None:
            if key.startswith("released/submission_archive/"):
                raise PermissionError("anonymous read denied")
            super().get_file(key, path, max_bytes=max_bytes)

    transport = AnonymousReadDeniedTransport()
    writer = S3Store(
        AuditConfig(backend="s3"), envelope=XorEnvelope(), transport=transport
    )
    ref = writer.put(b"winning source archive", ArtifactKind.SUBMISSION_ARCHIVE)
    writer.release(ref)
    public = S3Store(
        AuditConfig(backend="s3"),
        envelope=PassthroughEnvelope(),
        transport=transport,
        public_read_only=True,
        allow_sealed_operations=False,
    )

    assert public.is_released(ref) is False


def test_public_store_never_loads_holdout_key_or_private_object(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("VIDAIO_AUDIT_HOLDOUT_KEY", raising=False)
    cfg = AuditConfig(backend="local", local_root=tmp_path)
    sealed = LocalFsStore(tmp_path, envelope=XorEnvelope())
    ref = sealed.put(b"private until retirement", ArtifactKind.REFERENCE_ORIGINAL)

    public = make_public_store(cfg)
    assert public.exists(ref) is False
    with pytest.raises(FileNotFoundError, match="has not been released"):
        public.get(ref)
    with pytest.raises(ReadOnlyStoreError):
        public.put(b"no writes", ArtifactKind.MANIFEST)

    sealed.release(ref)
    assert public.exists(ref) is True
    assert public.get(ref) == b"private until retirement"


def test_keyless_writer_publishes_public_evidence_but_cannot_touch_holdouts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("VIDAIO_AUDIT_HOLDOUT_KEY", raising=False)
    store = make_unsealed_writer_store(
        AuditConfig(backend="local", local_root=tmp_path)
    )
    public = store.put(b"weight evidence", ArtifactKind.WEIGHT_VECTOR)
    assert store.get(public) == b"weight evidence"
    with pytest.raises(SealedArtifactAccessError):
        store.put(b"holdout", ArtifactKind.REFERENCE_ORIGINAL)
    sealed_ref = ArtifactRef(
        digest="0" * 64,
        kind=ArtifactKind.REFERENCE_ORIGINAL,
        byte_size=1,
        backend_key=backend_key(ArtifactKind.REFERENCE_ORIGINAL, "0" * 64),
    )
    with pytest.raises(SealedArtifactAccessError):
        store.release(sealed_ref)


def test_keyless_writer_allows_only_validator_publication_kinds(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("VIDAIO_AUDIT_HOLDOUT_KEY", raising=False)
    store = make_unsealed_writer_store(
        AuditConfig(backend="local", local_root=tmp_path)
    )

    manifest = store.put(b"publication manifest", ArtifactKind.MANIFEST)
    vector = store.put(b"weight vector", ArtifactKind.WEIGHT_VECTOR)
    assert store.get(manifest) == b"publication manifest"
    assert store.get(vector) == b"weight vector"

    for kind in ArtifactKind:
        if kind in {ArtifactKind.MANIFEST, ArtifactKind.WEIGHT_VECTOR}:
            continue
        expected_error = (
            SealedArtifactAccessError if kind in SEALED_KINDS else ReadOnlyStoreError
        )
        with pytest.raises(expected_error):
            store.put(b"forbidden", kind)
        assert not (tmp_path / backend_key(kind, sha256_hex(b"forbidden"))).exists()

    source = tmp_path / "forbidden-source"
    source.write_bytes(b"file upload bypass")
    with pytest.raises(ReadOnlyStoreError, match="may write only"):
        store.put_file(source, ArtifactKind.SCORE_PACKET)
    assert not (
        tmp_path
        / backend_key(ArtifactKind.SCORE_PACKET, sha256_hex(b"file upload bypass"))
    ).exists()


def test_keyless_writer_cannot_create_or_finalize_epoch_sets(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("VIDAIO_AUDIT_HOLDOUT_KEY", raising=False)
    authority = LocalFsStore(tmp_path)
    prefix = "finalized/epoch=41"
    expected = authority.put_set_member(
        prefix, "log.json", b"authority epoch", ArtifactKind.EPOCH_LOG
    )
    authority.finalize_set(prefix)

    thin = make_unsealed_writer_store(
        AuditConfig(backend="local", local_root=tmp_path)
    )
    assert thin.is_finalized(prefix) is True
    assert thin.get_set_member(
        prefix, "log.json", expected_digest=expected.digest
    ) == b"authority epoch"

    with pytest.raises(ReadOnlyStoreError, match="cannot mutate"):
        thin.put_set_member(
            "finalized/epoch=42",
            "log.json",
            b"forged epoch",
            ArtifactKind.EPOCH_LOG,
        )
    with pytest.raises(ReadOnlyStoreError, match="cannot mutate"):
        thin.put_set_member(
            prefix,
            "replacement.json",
            b"forged replacement",
            ArtifactKind.EPOCH_LOG,
        )
    with pytest.raises(ReadOnlyStoreError, match="cannot finalize"):
        thin.finalize_set("finalized/epoch=42")

    assert not (tmp_path / "finalized/epoch=42/log.json").exists()
    assert not (tmp_path / "finalized/epoch=42/_FINALIZED").exists()


def test_s3_public_view_is_release_only_and_read_only() -> None:
    transport = FakeObjectTransport()
    sealed = S3Store(
        AuditConfig(backend="s3"), envelope=XorEnvelope(), transport=transport
    )
    ref = sealed.put(b"holdout", ArtifactKind.REFERENCE_ORIGINAL)
    public = S3Store(
        AuditConfig(backend="s3"),
        envelope=PassthroughEnvelope(),
        transport=transport,
        public_read_only=True,
        allow_sealed_operations=False,
    )
    assert not public.exists(ref)
    with pytest.raises(FileNotFoundError):
        public.get(ref)
    with pytest.raises(ReadOnlyStoreError):
        public.finalize_set("finalized/epoch=9")
    sealed.release(ref)
    assert public.get(ref) == b"holdout"


def test_s3_materialize_streams_to_verified_file_without_get_bytes(
    tmp_path: Path,
) -> None:
    store, transport = s3_store()
    payload = b"video" * 100_000
    ref = store.put(payload, ArtifactKind.MINER_OUTPUT)
    transport.get_bytes_calls.clear()

    path = store.materialize(ref, tmp_path / "materialized", max_bytes=len(payload))

    assert path.read_bytes() == payload
    assert transport.get_file_calls == [ref.backend_key]
    assert transport.get_bytes_calls == []


def test_s3_bounded_metadata_reads_stream_without_get_bytes() -> None:
    store, transport = s3_store()
    payload = b'{"score": 1}'
    ref = store.put(payload, ArtifactKind.SCORE_PACKET)
    transport.get_bytes_calls.clear()

    assert store.get_limited(ref, max_bytes=1024) == payload
    assert store.get_digest_limited(ref.kind, ref.digest, max_bytes=1024) == payload

    assert transport.get_file_calls == [ref.backend_key, ref.backend_key]
    assert transport.get_bytes_calls == []


def test_s3_bounded_read_enforces_body_cap_after_stale_head(monkeypatch) -> None:
    store, transport = s3_store()
    ref = store.put(b"ok", ArtifactKind.SCORE_PACKET)
    transport.objects[ref.backend_key] = b"attacker-controlled replacement"
    original_head = transport.head

    def stale_head(key: str) -> int | None:
        if key == ref.backend_key:
            return ref.byte_size
        return original_head(key)

    monkeypatch.setattr(transport, "head", stale_head)
    transport.get_bytes_calls.clear()

    with pytest.raises(ArtifactTooLargeError, match="crossed bound"):
        store.get_limited(ref, max_bytes=ref.byte_size)
    assert transport.get_bytes_calls == []


def test_materialize_refuses_declared_oversize_before_remote_download(
    tmp_path: Path,
) -> None:
    store, transport = s3_store()
    ref = store.put(b"12345", ArtifactKind.MINER_OUTPUT)
    with pytest.raises(ArtifactTooLargeError, match="declares 5 bytes"):
        store.materialize(ref, tmp_path, max_bytes=4)
    assert transport.get_file_calls == []


def test_hippius_store_uses_the_verified_transport_layer() -> None:
    transport = FakeObjectTransport()
    store = HippiusStore(
        AuditConfig(backend="hippius"),
        envelope=XorEnvelope(),
        transport=transport,
    )
    ref = store.put(b"data", ArtifactKind.MANIFEST)
    assert store.get(ref) == b"data"
    assert transport.objects[ref.backend_key] == b"data"


def test_s3store_write_once_is_noop() -> None:
    store, transport = s3_store()
    ref1 = store.put(b"same bytes", ArtifactKind.CHALLENGE_INPUT)
    ref2 = store.put(b"same bytes", ArtifactKind.CHALLENGE_INPUT)
    assert ref1 == ref2
    # the second put must NOT re-upload (write-once guard via exists probe)
    assert transport.put_calls.count(ref1.backend_key) == 1


def test_s3store_corruption_detected_on_read() -> None:
    store, transport = s3_store()
    ref = store.put(b"pristine artifact bytes", ArtifactKind.SCORE_PACKET)
    raw = bytearray(transport.objects[ref.backend_key])
    raw[3] ^= 0xFF  # tamper the stored object
    transport.objects[ref.backend_key] = bytes(raw)
    with pytest.raises(IntegrityError):
        store.get(ref)


def test_s3store_seals_holdout_at_rest() -> None:
    transport = FakeObjectTransport()
    store = S3Store(
        AuditConfig(backend="s3"), envelope=XorEnvelope(), transport=transport
    )
    plaintext = b"holdout original -- sealed until asset retirement"
    ref = store.put(plaintext, ArtifactKind.REFERENCE_ORIGINAL)
    assert transport.objects[ref.backend_key] != plaintext  # encrypted at rest
    assert store.get(ref) == plaintext  # digest covers the plaintext
    # non-sealed kinds stay plaintext
    ref2 = store.put(plaintext, ArtifactKind.MINER_OUTPUT)
    assert transport.objects[ref2.backend_key] == plaintext


def test_s3store_streams_sealed_file_archive_and_release(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    transport = FakeObjectTransport()
    envelope = AesGcmEnvelope(bytes.fromhex("33" * 32))
    store = S3Store(AuditConfig(backend="s3"), envelope=envelope, transport=transport)
    source = tmp_path / "holdout.mp4"
    source.write_bytes((b"holdout-frame" * 100_000) + b"tail")

    ref = store.put_file(source, ArtifactKind.REFERENCE_ORIGINAL)
    assert ref.backend_key in transport.put_file_calls
    assert source.read_bytes() not in transport.objects[ref.backend_key]

    transport.get_bytes_calls.clear()
    store.release(ref)
    assert (
        transport.objects[released_backend_key(ref.kind, ref.digest)]
        == source.read_bytes()
    )
    assert transport.get_bytes_calls == []


def test_s3store_head_reports_size_or_none() -> None:
    _, transport = s3_store()
    transport.put_bytes("k", b"abcd")
    assert transport.head("k") == 4
    assert transport.head("missing") is None


def test_s3store_epoch_log_round_trip() -> None:
    store, _ = s3_store()
    ref = store.put(b"epoch log bytes", ArtifactKind.EPOCH_LOG)
    assert store.get(ref) == b"epoch log bytes"


# --------------------------------------------------------------------------------------
# The _FINALIZED set convention (the half-write guard) — on BOTH backends.
# --------------------------------------------------------------------------------------


@pytest.fixture(params=["local", "s3"])
def guarded_store(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "local":
        return LocalFsStore(tmp_path / "audit")
    store, _ = s3_store()
    return store


def test_key_scheme_matches_documented_layout() -> None:
    assert set_member_key("finalized/epoch=41822", "log.json") == (
        "finalized/epoch=41822/log.json"
    )
    assert finalized_marker_key("finalized/epoch=41822") == (
        "finalized/epoch=41822/_FINALIZED"
    )


def test_set_is_unreadable_until_finalized(guarded_store) -> None:
    prefix = "finalized/epoch=41822"
    ref = guarded_store.put_set_member(prefix, "log.json", b"the epoch log")
    assert ref.kind is ArtifactKind.EPOCH_LOG
    assert ref.backend_key == set_member_key(prefix, "log.json")
    # not finalized yet: a mirror MUST refuse to read a half-written set
    assert not guarded_store.is_finalized(prefix)
    with pytest.raises(SetNotFinalizedError):
        guarded_store.get_set_member(prefix, "log.json")
    # marker written LAST makes the whole set readable
    guarded_store.finalize_set(prefix)
    assert guarded_store.is_finalized(prefix)
    assert guarded_store.get_set_member(prefix, "log.json") == b"the epoch log"


def test_set_member_verify_on_read(guarded_store) -> None:
    prefix = "finalized/epoch=1"
    ref = guarded_store.put_set_member(prefix, "log.json", b"payload")
    guarded_store.finalize_set(prefix)
    # correct digest passes
    got = guarded_store.get_set_member(
        prefix, "log.json", expected_digest=ref.digest, byte_size=ref.byte_size
    )
    assert got == b"payload"
    # wrong digest is rejected (verify-on-read)
    with pytest.raises(IntegrityError):
        guarded_store.get_set_member(prefix, "log.json", expected_digest="0" * 64)


def test_s3_bounded_set_member_streams_without_get_bytes() -> None:
    store, transport = s3_store()
    prefix = "finalized/epoch=3"
    ref = store.put_set_member(prefix, "log.json", b"bounded log")
    store.finalize_set(prefix)
    transport.get_bytes_calls.clear()
    transport.get_file_calls.clear()

    assert (
        store.get_set_member(
            prefix,
            "log.json",
            expected_digest=ref.digest,
            byte_size=ref.byte_size,
            max_bytes=64,
        )
        == b"bounded log"
    )
    assert transport.get_file_calls == [ref.backend_key]
    assert transport.get_bytes_calls == []


def test_finalized_set_is_immutable(guarded_store) -> None:
    prefix = "finalized/epoch=2"
    guarded_store.put_set_member(prefix, "log.json", b"x")
    guarded_store.finalize_set(prefix)
    with pytest.raises(SetAlreadyFinalizedError):
        guarded_store.put_set_member(prefix, "extra.json", b"y")
    guarded_store.finalize_set(prefix)  # idempotent, no raise


def test_reserved_marker_name_rejected(guarded_store) -> None:
    with pytest.raises(ValueError):
        guarded_store.put_set_member("finalized/epoch=3", FINALIZED_MARKER, b"")


# --------------------------------------------------------------------------------------
# make_store backend selection + the storage/hippius transport stubs.
# --------------------------------------------------------------------------------------


def test_make_store_selects_s3_without_importing_boto3(tmp_path: Path) -> None:
    # make_store must dispatch to S3Store lazily (no SDK import / no network on build).
    store = make_store(
        AuditConfig(backend="s3", s3_bucket="b", allow_plaintext_holdout=True)
    )
    assert isinstance(store, S3Store)


def test_s3_transport_not_configured_without_deps_or_bucket() -> None:
    # No injected transport: first op triggers the lazy real transport, which fails
    # fast (boto3 missing -> '.[storage]' hint, or empty bucket) — a NotConfiguredError
    # either way. The suite runs with boto3 NOT installed.
    store = S3Store(AuditConfig(backend="s3", allow_plaintext_holdout=True))
    with pytest.raises(NotConfiguredError):
        store.put(b"data", ArtifactKind.EPOCH_LOG)


def test_put_file_sealed_pre_upload_round_trip_guard(tmp_path: Path) -> None:
    """A sealed holdout that cannot be opened must never reach storage.

    Regression for the 2026-09-04 incident: a corrupted sealed upload was only
    discovered when its release attempt held epoch finalization. put_file now
    round-trips the sealed file locally before uploading.
    """

    class CorruptingSealEnvelope(XorEnvelope):
        def seal(self, data: bytes) -> bytes:
            sealed = bytearray(super().seal(data))
            sealed[len(sealed) // 2] ^= 0x01  # a single flipped bit
            return bytes(sealed)

    transport = FakeObjectTransport()
    store = S3Store(
        AuditConfig(backend="s3"),
        envelope=CorruptingSealEnvelope(),
        transport=transport,
    )
    source = tmp_path / "pristine.mkv"
    source.write_bytes(b"pristine reference bytes" * 1024)
    with pytest.raises(IntegrityError):
        store.put_file(source, ArtifactKind.REFERENCE_ORIGINAL)
    assert not transport.objects  # the corrupt seal was never uploaded
