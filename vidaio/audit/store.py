"""Content-addressed artifact store — the physical layer of auditability.

Every scored input/output is preserved here keyed by its sha256 digest, so a
third party holding only the digests (from a bundle or an on-chain commitment)
can fetch the exact bytes and recompute every metric. This is the addition
design spec §08 calls out as the audit blocker: video bytes are never persisted today.

Properties enforced:
- content addressing: key is derived from (kind, sha256(plaintext)) only;
  `backend_key` on a ref is informational and never trusted for path lookup.
- write-once: putting bytes whose digest already exists is a no-op returning
  the same ref — artifacts can never be replaced in place.
- verify-on-read: `get()` recomputes the sha256 and raises IntegrityError on
  any mismatch, so silent on-disk tampering/corruption cannot go unnoticed.

Reference originals (holdout ground truth) and competition submission archives
pass through an `Envelope` — an encryption-at-rest hook. Digests always cover
the PLAINTEXT, so post-release recompute matches the refs committed before
enrollment/crowning.

When a challenge becomes terminal, ``release()`` publishes a verified plaintext
copy under ``released/<canonical-key>``.  Bucket policy may expose only that
prefix publicly; live sealed objects remain private and third-party auditors do
not need the master holdout key.
"""

from __future__ import annotations

import contextlib
import base64
import binascii
import hashlib
import io
import os
import tempfile
import uuid
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Callable, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from vidaio.audit.canonical import SHA256_HEX_PATTERN, sha256_hex
from vidaio.audit.config import AuditConfig

_CHUNK = 1 << 20
_S3_CONNECT_TIMEOUT_SECONDS = 10
_S3_READ_TIMEOUT_SECONDS = 30
_S3_TOTAL_MAX_ATTEMPTS = 3


class IntegrityError(Exception):
    """Stored bytes do not hash to the digest they are addressed by."""


class NotConfiguredError(RuntimeError):
    """A backend whose transport is not yet wired up was used."""


class ReadOnlyStoreError(RuntimeError):
    """A write was attempted through the public, read-only audit view."""


class SealedArtifactAccessError(RuntimeError):
    """A keyless store role attempted to access a live sealed holdout."""


class ArtifactTooLargeError(OSError):
    """An artifact exceeds a caller-owned materialization/read bound."""


class ArtifactKind(StrEnum):
    CHALLENGE_INPUT = "challenge_input"
    MINER_OUTPUT = "miner_output"
    REFERENCE_ORIGINAL = "reference_original"  # holdout — sealed at rest
    MANIFEST = "manifest"
    SCORE_PACKET = "score_packet"
    WEIGHT_VECTOR = "weight_vector"
    DAG_REVEAL = "dag_reveal"
    #: The per-epoch "epoch log" snapshot (EpochResultsSnapshot) — the one
    #: finalized artifact that drives both convergence and audit
    #: (the project design record §1, §3.1). A first-class content-addressed
    #: artifact so a validator can mirror it verify-on-read like any other; it is
    #: NOT sealed (it is public by design). The epoch-log MODEL + finalizer are a
    #: later wave — this kind lets the store hold and serve it now.
    EPOCH_LOG = "epoch_log"
    #: A persisted `AuditBundle` JSON (vidaio.audit.bundle). Content-addressed like
    #: any other artifact, so its store digest IS its `bundle_digest()` — the auditor's
    #: `StoredBundleSource` resolves a manifest AUDIT_BUNDLE ref straight back to the
    #: bundle by that digest, verify-on-read (the project design record §1c). Public
    #: (not sealed): a bundle names sealed holdouts by ref but carries no holdout bytes.
    AUDIT_BUNDLE = "audit_bundle"
    #: Exact contender source archive captured before evaluation. It remains sealed
    #: for ordinary contenders and is released only when an anchored CROWN promotion
    #: selects it as the next executable baseline. A distinct kind makes source-code
    #: disclosure explicit in both commitments and bucket policy.
    SUBMISSION_ARCHIVE = "submission_archive"


#: Kinds stored encrypted-at-rest via the Envelope hook. Public/keyless readers can
#: fetch them only through the content-addressed ``released/`` namespace.
SEALED_KINDS = frozenset(
    {ArtifactKind.REFERENCE_ORIGINAL, ArtifactKind.SUBMISSION_ARCHIVE}
)


class ArtifactRef(BaseModel):
    """Digest-first handle to a stored artifact. The digest is the identity."""

    model_config = ConfigDict(frozen=True)

    digest: str = Field(pattern=SHA256_HEX_PATTERN)  # sha256 of the PLAINTEXT
    kind: ArtifactKind
    byte_size: int = Field(ge=0)  # plaintext size
    backend_key: str


@runtime_checkable
class Envelope(Protocol):
    """Encryption-at-rest hook for sealed-holdout artifacts.

    Sealed-holdout release policy: reference originals are the hidden ground
    truth of live assets. While an asset is live they are stored sealed so no
    party with storage access can read the holdout mid-competition; only after
    the asset is retired is the original published (key revealed / artifact
    re-served unsealed) per commit-reveal. Because ArtifactRef digests cover
    the plaintext, the pre-enrollment commitments made while the artifact was
    sealed remain verifiable against the post-retirement release.

    `PassthroughEnvelope` is the default no-op implementation; a key-managed
    implementation is a later-phase adapter and only needs these two methods.
    """

    def seal(self, data: bytes) -> bytes: ...

    def unseal(self, sealed: bytes) -> bytes: ...


class PassthroughEnvelope:
    """No-op Envelope (plaintext at rest). See Envelope for the release policy."""

    def seal(self, data: bytes) -> bytes:
        return data

    def unseal(self, sealed: bytes) -> bytes:
        return sealed


class AesGcmEnvelope:
    """Authenticated encryption for sealed holdout artifacts.

    The key is deployment-owned and supplied through an environment variable;
    object-store credentials alone therefore cannot reveal live holdouts. A
    fresh nonce is safe here because content-addressed objects are write-once.
    """

    _MAGIC = b"VIDAIO-AESGCM-1\x00"

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("the audit holdout AES-GCM key must be exactly 32 bytes")
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise NotConfiguredError(
                "encrypted audit holdouts require the optional storage dependencies; "
                "install vidaio-next[storage]"
            ) from exc
        self._key = bytes(key)
        self._aesgcm = AESGCM(key)

    @classmethod
    def from_env(cls, env_name: str) -> "AesGcmEnvelope":
        encoded = os.environ.get(env_name, "").strip()
        if not encoded:
            raise NotConfiguredError(
                "refusing a plaintext holdout: no encryption envelope was supplied and "
                f"${env_name} is empty. Set it to a random 32-byte hex/base64 key, or "
                "set audit.allow_plaintext_holdout=true only for local development."
            )
        try:
            key = bytes.fromhex(encoded)
        except ValueError:
            try:
                key = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise NotConfiguredError(
                    f"${env_name} must contain a 32-byte hex or base64 AES key"
                ) from exc
        try:
            return cls(key)
        except ValueError as exc:
            raise NotConfiguredError(f"invalid ${env_name}: {exc}") from exc

    def seal(self, data: bytes) -> bytes:
        nonce = os.urandom(12)
        return self._MAGIC + nonce + self._aesgcm.encrypt(nonce, data, self._MAGIC)

    def unseal(self, sealed: bytes) -> bytes:
        if not sealed.startswith(self._MAGIC) or len(sealed) < len(self._MAGIC) + 28:
            raise IntegrityError("sealed holdout has an invalid AES-GCM envelope")
        offset = len(self._MAGIC)
        nonce = sealed[offset : offset + 12]
        try:
            return self._aesgcm.decrypt(nonce, sealed[offset + 12 :], self._MAGIC)
        except Exception as exc:
            raise IntegrityError("sealed holdout authentication failed") from exc

    def seal_file(self, source: Path, destination: Path) -> None:
        """Stream an AES-GCM envelope without buffering a large holdout in RAM."""
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        nonce = os.urandom(12)
        encryptor = Cipher(algorithms.AES(self._key), modes.GCM(nonce)).encryptor()
        encryptor.authenticate_additional_data(self._MAGIC)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with source.open("rb") as plain, destination.open("xb") as sealed:
                sealed.write(self._MAGIC)
                sealed.write(nonce)
                while chunk := plain.read(_CHUNK):
                    sealed.write(encryptor.update(chunk))
                sealed.write(encryptor.finalize())
                sealed.write(encryptor.tag)
                sealed.flush()
                os.fsync(sealed.fileno())
        except BaseException:
            with contextlib.suppress(FileNotFoundError, OSError):
                destination.unlink()
            raise

    def unseal_file(self, source: Path, destination: Path) -> None:
        """Stream/decrypt the file format emitted by :meth:`seal_file`."""
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        header_size = len(self._MAGIC) + 12
        size = source.stat().st_size
        if size < header_size + 16:
            raise IntegrityError("sealed holdout has an invalid AES-GCM envelope")
        with source.open("rb") as sealed:
            magic = sealed.read(len(self._MAGIC))
            if magic != self._MAGIC:
                raise IntegrityError("sealed holdout has an invalid AES-GCM envelope")
            nonce = sealed.read(12)
            sealed.seek(-16, os.SEEK_END)
            tag = sealed.read(16)
            cipher_size = size - header_size - 16
            sealed.seek(header_size)
            decryptor = Cipher(
                algorithms.AES(self._key), modes.GCM(nonce, tag)
            ).decryptor()
            decryptor.authenticate_additional_data(self._MAGIC)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with destination.open("xb") as plain:
                    remaining = cipher_size
                    while remaining:
                        chunk = sealed.read(min(_CHUNK, remaining))
                        if not chunk:
                            raise IntegrityError(
                                "sealed holdout ended before its authenticated tag"
                            )
                        remaining -= len(chunk)
                        plain.write(decryptor.update(chunk))
                    plain.write(decryptor.finalize())
                    plain.flush()
                    os.fsync(plain.fileno())
            except BaseException as exc:
                with contextlib.suppress(FileNotFoundError, OSError):
                    destination.unlink()
                if isinstance(exc, IntegrityError):
                    raise
                raise IntegrityError("sealed holdout authentication failed") from exc


class AuditStore(Protocol):
    """Backend-agnostic content-addressed artifact store."""

    @property
    def public_read_only(self) -> bool: ...

    def put(self, data: bytes, kind: ArtifactKind) -> ArtifactRef: ...

    def put_file(self, path: str | Path, kind: ArtifactKind) -> ArtifactRef: ...

    def get(self, ref: ArtifactRef) -> bytes: ...

    def get_limited(self, ref: ArtifactRef, max_bytes: int) -> bytes: ...

    def get_digest_limited(
        self, kind: ArtifactKind, digest: str, *, max_bytes: int
    ) -> bytes: ...

    def exists(self, ref: ArtifactRef) -> bool: ...

    def open_stream(self, ref: ArtifactRef) -> BinaryIO: ...

    def materialize(
        self, ref: ArtifactRef, directory: str | Path, *, max_bytes: int
    ) -> Path: ...

    def release(self, ref: ArtifactRef) -> None: ...

    def is_released(self, ref: ArtifactRef) -> bool: ...


def backend_key(kind: ArtifactKind, digest: str) -> str:
    """Canonical sharded key: <kind>/<aa>/<bb>/<digest>."""
    return f"{kind.value}/{digest[:2]}/{digest[2:4]}/{digest}"


def released_backend_key(kind: ArtifactKind, digest: str) -> str:
    """Public post-retirement key for a formerly sealed artifact."""
    return f"released/{backend_key(kind, digest)}"


# --------------------------------------------------------------------------------------
# The _FINALIZED "set" convention — a production-proven half-write guard.
# --------------------------------------------------------------------------------------
#
# A multi-object "set" (e.g. one epoch's files) is grouped under a key PREFIX and
# becomes readable only once a `_FINALIZED` marker object is written LAST. A reader
# probes for the marker before reading any member, so a validator can never mirror a
# half-written epoch (the project design record §4 / the project design record §1.1 — a
# pattern proven in production: `_FINALIZED` is written last, the mirror probes it).
#
# Key/naming scheme (backend-independent; the same relative keys on LocalFs and S3):
#
#     finalized/epoch={N}/log.json     # a set MEMBER (the EpochResultsSnapshot bytes)
#     finalized/epoch={N}/_FINALIZED   # the empty marker, written LAST
#
# Content-addressed audit files keep their OWN scheme, unchanged:
#
#     <kind>/<aa>/<bb>/<digest>        # backend_key(kind, digest)
#
# Set members are addressed by their (prefix, name) key; verify-on-read is still
# available by passing the member's expected sha256 digest to get_set_member.

#: The marker object name written last to make a set readable.
FINALIZED_MARKER = "_FINALIZED"


def set_member_key(prefix: str, name: str) -> str:
    """Key of a named member inside a set: `<prefix>/<name>`."""
    return f"{prefix.rstrip('/')}/{name}"


def finalized_marker_key(prefix: str) -> str:
    """Key of a set's `_FINALIZED` marker."""
    return set_member_key(prefix, FINALIZED_MARKER)


class SetNotFinalizedError(Exception):
    """A set was read before its `_FINALIZED` marker was written (half-write guard)."""


class SetAlreadyFinalizedError(Exception):
    """A finalized set is immutable: its members cannot be rewritten."""


class WriteOnceConflictError(Exception):
    """A conditional create found different bytes already stored at the key."""


class _SetConventionMixin:
    """The `_FINALIZED` half-write guard, shared by every backend.

    Requires the host to provide three raw key primitives: `_raw_put(key, data)`,
    `_raw_get(key) -> bytes`, `_raw_exists(key) -> bool`. The convention (marker
    written LAST, no member readable before the marker, a finalized set is
    immutable) is identical across LocalFs / S3 / Hippius and lives here once.
    """

    # Provided by the concrete store.
    def _raw_put(self, key: str, data: bytes) -> None: ...  # pragma: no cover
    def _raw_get(self, key: str) -> bytes: ...  # pragma: no cover
    def _raw_get_limited(
        self, key: str, *, max_bytes: int
    ) -> bytes: ...  # pragma: no cover
    def _raw_exists(self, key: str) -> bool: ...  # pragma: no cover
    def _raw_head(self, key: str) -> int | None: ...  # pragma: no cover

    def is_finalized(self, prefix: str) -> bool:
        """True once the set's `_FINALIZED` marker is present."""
        return self._raw_exists(finalized_marker_key(prefix))

    def put_set_member(
        self,
        prefix: str,
        name: str,
        data: bytes,
        kind: ArtifactKind = ArtifactKind.EPOCH_LOG,
    ) -> ArtifactRef:
        """Write one member into an in-progress set. Refuses once finalized.

        Returns an ArtifactRef whose `backend_key` is the member key and whose
        `digest`/`byte_size` let a later reader verify-on-read.
        """
        if name == FINALIZED_MARKER:
            raise ValueError(f"{FINALIZED_MARKER!r} is the reserved marker name")
        if self.is_finalized(prefix):
            raise SetAlreadyFinalizedError(
                f"set {prefix!r} is finalized and immutable; cannot write member {name!r}"
            )
        key = set_member_key(prefix, name)
        self._raw_put(key, data)
        return ArtifactRef(
            digest=sha256_hex(data), kind=kind, byte_size=len(data), backend_key=key
        )

    def finalize_set(self, prefix: str) -> None:
        """Write the `_FINALIZED` marker LAST, making the set readable. Idempotent."""
        if self.is_finalized(prefix):
            return
        self._raw_put(finalized_marker_key(prefix), b"")

    def get_set_member(
        self,
        prefix: str,
        name: str,
        *,
        expected_digest: str | None = None,
        byte_size: int | None = None,
        max_bytes: int | None = None,
    ) -> bytes:
        """Read a member — but ONLY once the set is finalized.

        Raises SetNotFinalizedError if the marker is absent (never mirror a
        half-written set). When `expected_digest` is supplied, verifies the bytes
        against it (verify-on-read), raising IntegrityError on mismatch.
        """
        if not self.is_finalized(prefix):
            raise SetNotFinalizedError(
                f"set {prefix!r} has no {FINALIZED_MARKER} marker: refusing to read a "
                "possibly half-written set"
            )
        key = set_member_key(prefix, name)
        if max_bytes is not None:
            if max_bytes <= 0:
                raise ValueError("set-member read bound must be positive")
            size = self._raw_head(key)
            if size is None:
                raise FileNotFoundError(key)
            if size > max_bytes:
                raise ArtifactTooLargeError(
                    f"set member {prefix}/{name} is {size} bytes; maximum is {max_bytes}"
                )
        data = (
            self._raw_get(key)
            if max_bytes is None
            else self._raw_get_limited(key, max_bytes=max_bytes)
        )
        if expected_digest is not None:
            if (byte_size is not None and len(data) != byte_size) or (
                sha256_hex(data) != expected_digest
            ):
                raise IntegrityError(
                    f"set member {prefix}/{name} failed verify-on-read: bytes do not "
                    "match the expected content digest"
                )
        return data


def _hash_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _read_path_limited(path: Path, *, max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` from disk, detecting growth after a HEAD/stat.

    A separate size probe is not a sufficient memory bound: an object or local
    file can be replaced between the probe and the body read. Reading one byte
    past the cap makes the bound authoritative without ever allocating the full
    attacker-controlled object.
    """
    if max_bytes <= 0:
        raise ValueError("artifact read bound must be positive")
    data = bytearray()
    with path.open("rb") as source:
        while len(data) <= max_bytes:
            chunk = source.read(min(_CHUNK, max_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
    if len(data) > max_bytes:
        raise ArtifactTooLargeError(
            f"object {path.name} crossed the {max_bytes}-byte maximum"
        )
    return bytes(data)


def _require_download_bound(ref: ArtifactRef, max_bytes: int) -> None:
    if max_bytes <= 0:
        raise ValueError("artifact download bound must be positive")
    if ref.byte_size > max_bytes:
        raise ArtifactTooLargeError(
            f"artifact {ref.kind.value}/{ref.digest} declares {ref.byte_size} bytes; "
            f"auditor maximum is {max_bytes}"
        )


def _copy_verified(
    source: BinaryIO, destination: Path, ref: ArtifactRef, max_bytes: int
) -> None:
    """Copy plaintext bytes to disk with a hard cap and content-address check."""
    _require_download_bound(ref, max_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    size = 0
    try:
        with destination.open("xb") as sink:
            while chunk := source.read(_CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise ArtifactTooLargeError(
                        f"artifact {ref.kind.value}/{ref.digest} crossed the "
                        f"{max_bytes}-byte auditor maximum"
                    )
                h.update(chunk)
                sink.write(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        if size != ref.byte_size or h.hexdigest() != ref.digest:
            raise IntegrityError(
                f"artifact {ref.kind.value}/{ref.digest} failed verify-on-read: "
                "materialized bytes do not match the content address"
            )
    except BaseException:
        with contextlib.suppress(FileNotFoundError, OSError):
            destination.unlink()
        raise


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _seal_file(envelope: Envelope, source: Path, destination: Path) -> None:
    stream = getattr(envelope, "seal_file", None)
    if callable(stream):
        stream(source, destination)
        return
    _atomic_write(destination, envelope.seal(source.read_bytes()))


def _unseal_file(envelope: Envelope, source: Path, destination: Path) -> None:
    stream = getattr(envelope, "unseal_file", None)
    if callable(stream):
        stream(source, destination)
        return
    _atomic_write(destination, envelope.unseal(source.read_bytes()))


def _verify_path(path: Path, ref: ArtifactRef) -> None:
    digest, size = _hash_file(path)
    if size != ref.byte_size or digest != ref.digest:
        raise IntegrityError(
            f"artifact {ref.kind.value}/{ref.digest} failed verify-on-read: "
            "file bytes do not match the content address"
        )


class LocalFsStore(_SetConventionMixin):
    """Filesystem store: local_root/<kind>/<aa>/<bb>/<digest>, atomic writes.

    Honors the `_FINALIZED` set convention (via _SetConventionMixin) so tests and
    dev exercise the half-write guard on the always-available local backend; set
    members and markers live under `local_root/<prefix>/...`.
    """

    def __init__(
        self,
        root: str | Path,
        envelope: Envelope | None = None,
        *,
        public_read_only: bool = False,
        allow_sealed_operations: bool = True,
    ) -> None:
        self._root = Path(root)
        self._envelope: Envelope = envelope or PassthroughEnvelope()
        self._public_read_only = public_read_only
        self._allow_sealed_operations = allow_sealed_operations

    @property
    def public_read_only(self) -> bool:
        """Whether this is the keyless, release-only independent-auditor view."""
        return self._public_read_only

    # -- raw key primitives backing the set convention -----------------------------

    def _raw_put(self, key: str, data: bytes) -> None:
        if self._public_read_only:
            raise ReadOnlyStoreError("public audit store is read-only")
        _atomic_write(self._root / key, data)

    def _raw_get(self, key: str) -> bytes:
        return (self._root / key).read_bytes()

    def _raw_get_limited(self, key: str, *, max_bytes: int) -> bytes:
        return _read_path_limited(self._root / key, max_bytes=max_bytes)

    def _raw_exists(self, key: str) -> bool:
        return (self._root / key).exists()

    def _raw_head(self, key: str) -> int | None:
        try:
            return (self._root / key).stat().st_size
        except FileNotFoundError:
            return None

    def _path(self, kind: ArtifactKind, digest: str) -> Path:
        # Derived strictly from (kind, digest) — never from ref.backend_key —
        # so a doctored ref cannot address outside the store layout.
        return self._root / kind.value / digest[:2] / digest[2:4] / digest

    def _released_path(self, kind: ArtifactKind, digest: str) -> Path:
        return self._root / released_backend_key(kind, digest)

    def put(self, data: bytes, kind: ArtifactKind) -> ArtifactRef:
        if self._public_read_only:
            raise ReadOnlyStoreError("public audit store is read-only")
        if kind in SEALED_KINDS and not self._allow_sealed_operations:
            raise SealedArtifactAccessError(
                "this store role cannot write sealed holdouts"
            )
        digest = sha256_hex(data)
        ref = ArtifactRef(
            digest=digest,
            kind=kind,
            byte_size=len(data),
            backend_key=backend_key(kind, digest),
        )
        path = self._path(kind, digest)
        if path.exists():  # write-once: identical content, nothing to do
            return ref
        stored = self._envelope.seal(data) if kind in SEALED_KINDS else data
        _atomic_write(path, stored)
        return ref

    def put_file(self, path: str | Path, kind: ArtifactKind) -> ArtifactRef:
        if self._public_read_only:
            raise ReadOnlyStoreError("public audit store is read-only")
        if kind in SEALED_KINDS and not self._allow_sealed_operations:
            raise SealedArtifactAccessError(
                "this store role cannot write sealed holdouts"
            )
        src = Path(path)
        digest, size = _hash_file(src)
        ref = ArtifactRef(
            digest=digest,
            kind=kind,
            byte_size=size,
            backend_key=backend_key(kind, digest),
        )
        dst = self._path(kind, digest)
        if dst.exists():
            return ref
        dst.parent.mkdir(parents=True, exist_ok=True)
        if kind in SEALED_KINDS:
            stage = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.part")
            try:
                _seal_file(self._envelope, src, stage)
                os.replace(stage, dst)
            finally:
                with contextlib.suppress(FileNotFoundError, OSError):
                    stage.unlink()
            return ref
        fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".tmp-")
        try:
            with src.open("rb") as fin, os.fdopen(fd, "wb") as fout:
                while chunk := fin.read(_CHUNK):
                    fout.write(chunk)
            os.replace(tmp, dst)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise
        return ref

    def get(self, ref: ArtifactRef) -> bytes:
        released = self._released_path(ref.kind, ref.digest)
        if ref.kind in SEALED_KINDS and released.is_file():
            data = released.read_bytes()
        elif ref.kind in SEALED_KINDS and not self._allow_sealed_operations:
            raise FileNotFoundError(
                f"sealed artifact {ref.kind.value}/{ref.digest} has not been released"
            )
        else:
            raw = self._path(ref.kind, ref.digest).read_bytes()
            data = self._envelope.unseal(raw) if ref.kind in SEALED_KINDS else raw
        if len(data) != ref.byte_size or sha256_hex(data) != ref.digest:
            raise IntegrityError(
                f"artifact {ref.kind.value}/{ref.digest} failed verify-on-read: "
                "stored bytes do not match the content address"
            )
        return data

    def get_limited(self, ref: ArtifactRef, max_bytes: int) -> bytes:
        _require_download_bound(ref, max_bytes)
        released = self._released_path(ref.kind, ref.digest)
        if ref.kind in SEALED_KINDS and released.is_file():
            source = released
        elif ref.kind in SEALED_KINDS and not self._allow_sealed_operations:
            raise FileNotFoundError(
                f"sealed artifact {ref.kind.value}/{ref.digest} has not been released"
            )
        else:
            source = self._path(ref.kind, ref.digest)
        # Encrypted canonical objects have envelope overhead and need unsealing;
        # public auditors never take this privileged compatibility path.
        if ref.kind in SEALED_KINDS and source == self._path(ref.kind, ref.digest):
            return self.get(ref)
        size = source.stat().st_size
        if size != ref.byte_size:
            raise IntegrityError(
                f"artifact {ref.kind.value}/{ref.digest} failed verify-on-read: "
                f"stored size {size} != committed plaintext size {ref.byte_size}"
            )
        if size > max_bytes:
            raise ArtifactTooLargeError(
                f"artifact {ref.kind.value}/{ref.digest} is {size} bytes; "
                f"auditor maximum is {max_bytes}"
            )
        data = _read_path_limited(source, max_bytes=max_bytes)
        if len(data) != ref.byte_size or sha256_hex(data) != ref.digest:
            raise IntegrityError(
                f"artifact {ref.kind.value}/{ref.digest} failed verify-on-read: "
                "stored bytes do not match the content address"
            )
        return data

    def get_digest_limited(
        self, kind: ArtifactKind, digest: str, *, max_bytes: int
    ) -> bytes:
        if max_bytes <= 0:
            raise ValueError("artifact read bound must be positive")
        if kind in SEALED_KINDS:
            source = self._released_path(kind, digest)
            if not source.is_file():
                if not self._allow_sealed_operations:
                    raise FileNotFoundError(
                        f"sealed artifact {kind.value}/{digest} has not been released"
                    )
                raise ValueError(
                    "digest-only reads of encrypted canonical holdouts are unsupported"
                )
        else:
            source = self._path(kind, digest)
        size = source.stat().st_size
        if size > max_bytes:
            raise ArtifactTooLargeError(
                f"artifact {kind.value}/{digest} is {size} bytes; maximum is {max_bytes}"
            )
        data = _read_path_limited(source, max_bytes=max_bytes)
        if sha256_hex(data) != digest:
            raise IntegrityError(
                f"artifact {kind.value}/{digest} failed digest-only verify-on-read"
            )
        return data

    def exists(self, ref: ArtifactRef) -> bool:
        if ref.kind in SEALED_KINDS and not self._allow_sealed_operations:
            return self._released_path(ref.kind, ref.digest).is_file()
        return self._path(ref.kind, ref.digest).exists()

    def open_stream(self, ref: ArtifactRef) -> BinaryIO:
        """Raw read stream. NOT integrity-verified — use get() for verified reads.

        Sealed kinds are unsealed (and therefore verified) in memory first.
        """
        if ref.kind in SEALED_KINDS:
            return io.BytesIO(self.get(ref))
        return self._path(ref.kind, ref.digest).open("rb")

    def materialize(
        self, ref: ArtifactRef, directory: str | Path, *, max_bytes: int
    ) -> Path:
        _require_download_bound(ref, max_bytes)
        released = self._released_path(ref.kind, ref.digest)
        if ref.kind in SEALED_KINDS and released.is_file():
            source_path = released
        elif ref.kind in SEALED_KINDS and not self._allow_sealed_operations:
            raise FileNotFoundError(
                f"sealed artifact {ref.kind.value}/{ref.digest} has not been released"
            )
        elif ref.kind in SEALED_KINDS:
            destination = Path(directory) / f"{ref.kind.value}-{ref.digest}"
            encrypted = self._path(ref.kind, ref.digest)
            if encrypted.stat().st_size > max_bytes + 1024:
                raise ArtifactTooLargeError(
                    f"sealed artifact {ref.kind.value}/{ref.digest} exceeds its "
                    "plaintext auditor bound plus envelope overhead"
                )
            try:
                _unseal_file(self._envelope, encrypted, destination)
                _verify_path(destination, ref)
            except BaseException:
                with contextlib.suppress(FileNotFoundError, OSError):
                    destination.unlink()
                raise
            return destination
        else:
            source_path = self._path(ref.kind, ref.digest)
        destination = Path(directory) / f"{ref.kind.value}-{ref.digest}"
        with source_path.open("rb") as source:
            _copy_verified(source, destination, ref, max_bytes)
        return destination

    def release(self, ref: ArtifactRef) -> None:
        """Publish a sealed artifact only after its challenge is terminal."""
        if self._public_read_only:
            raise ReadOnlyStoreError("public audit store is read-only")
        if not self._allow_sealed_operations:
            raise SealedArtifactAccessError(
                "this store role cannot release sealed holdouts"
            )
        if ref.kind not in SEALED_KINDS:
            raise ValueError(f"{ref.kind.value} is not a sealed artifact")
        destination = self._released_path(ref.kind, ref.digest)
        if destination.exists():
            _verify_path(destination, ref)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        try:
            _unseal_file(self._envelope, self._path(ref.kind, ref.digest), stage)
            _verify_path(stage, ref)
            os.replace(stage, destination)
        finally:
            with contextlib.suppress(FileNotFoundError, OSError):
                stage.unlink()

    def is_released(self, ref: ArtifactRef) -> bool:
        """Whether the released copy is readable and matches the committed ref.

        A path existing is not a release proof: it may be truncated, corrupt, or
        unreadable to this store role.  In particular, calling this method through
        :func:`make_public_store` proves the same keyless/anonymous read path an
        independent auditor uses, not merely that a privileged writer can stat a
        key in the release namespace.
        """
        if ref.kind not in SEALED_KINDS:
            return False
        try:
            _verify_path(self._released_path(ref.kind, ref.digest), ref)
        except (OSError, IntegrityError):
            return False
        return True


# --------------------------------------------------------------------------------------
# The remote object-store transport seam.
# --------------------------------------------------------------------------------------
#
# Following the SAME shape as the bittensor chain adapter
# (vidaio/chain/bittensor_adapter.py): ALL store logic — content-addressed keying,
# verify-on-read, write-once, the `_FINALIZED` set convention — lives in
# `_TransportBackedStore` and is FULLY unit-tested against a FAKE in-memory
# `_ObjectTransport` (tests/audit), with NO SDK installed. The real, boto3-backed
# transport (`_RealS3Transport`) is the ONLY part that imports boto3. Its conditional
# create contract is covered with an injected client; live provider compatibility is
# still a deployment-preflight concern. boto3 is an OPTIONAL dependency (`.[storage]`),
# imported lazily.


@runtime_checkable
class _ObjectTransport(Protocol):
    """Every actual object-store call, behind one thin key/bytes interface.

    Implementations own their own client/socket; the store layers content
    addressing, sealing, verify-on-read, and the `_FINALIZED` guard on top. Keys
    are opaque strings (the store derives them from (kind, digest) or (prefix, name)).
    """

    def put_bytes(self, key: str, payload: bytes) -> None: ...

    def put_file(self, key: str, path: Path) -> None: ...

    def get_bytes(self, key: str) -> bytes: ...

    def get_file(self, key: str, path: Path, *, max_bytes: int) -> None: ...

    def head(self, key: str) -> int | None:
        """Byte size of the object at `key`, or None if absent (cheap existence probe)."""
        ...

    def exists(self, key: str) -> bool: ...


class _TransportBackedStore(_SetConventionMixin):
    """Content-addressed AuditStore + `_FINALIZED` set convention over an
    `_ObjectTransport`. Backend-agnostic; the transport is the only variable part.

    The transport is connected LAZILY (on first use) so a store can be constructed
    — and `make_store` can dispatch to it — without importing the SDK or reaching
    the network; unit tests inject a fake transport instead.
    """

    def __init__(
        self,
        *,
        transport: _ObjectTransport | None = None,
        connect: "Callable[[], _ObjectTransport] | None" = None,
        envelope: Envelope | None = None,
        public_read_only: bool = False,
        allow_sealed_operations: bool = True,
    ) -> None:
        self._transport = transport
        self._connect = connect
        self._envelope: Envelope = envelope or PassthroughEnvelope()
        self._public_read_only = public_read_only
        self._allow_sealed_operations = allow_sealed_operations

    @property
    def public_read_only(self) -> bool:
        """Whether this transport was constructed unsigned and read-only."""
        return self._public_read_only

    @property
    def _t(self) -> _ObjectTransport:
        if self._transport is None:
            if self._connect is None:  # pragma: no cover - guarded by every subclass
                raise NotConfiguredError("no object-store transport configured")
            self._transport = self._connect()
        return self._transport

    # -- raw key primitives backing the set convention -----------------------------

    def _raw_put(self, key: str, data: bytes) -> None:
        if self._public_read_only:
            raise ReadOnlyStoreError("public audit store is read-only")
        try:
            self._t.put_bytes(key, data)
        except WriteOnceConflictError:
            # Repeating the exact same member/marker is idempotent. A different
            # payload at one logical set key is a typed conflict, never success.
            if self._t.get_bytes(key) != data:
                raise

    def _raw_get(self, key: str) -> bytes:
        return self._t.get_bytes(key)

    def _raw_get_limited(self, key: str, *, max_bytes: int) -> bytes:
        with tempfile.TemporaryDirectory(prefix="vidaio-bounded-read-") as tmp:
            path = Path(tmp) / "object"
            self._t.get_file(key, path, max_bytes=max_bytes)
            return path.read_bytes()

    def _raw_exists(self, key: str) -> bool:
        return self._t.exists(key)

    def _raw_head(self, key: str) -> int | None:
        return self._t.head(key)

    # -- content-addressed AuditStore interface ------------------------------------

    def put(self, data: bytes, kind: ArtifactKind) -> ArtifactRef:
        if self._public_read_only:
            raise ReadOnlyStoreError("public audit store is read-only")
        if kind in SEALED_KINDS and not self._allow_sealed_operations:
            raise SealedArtifactAccessError(
                "this store role cannot write sealed holdouts"
            )
        digest = sha256_hex(data)
        key = backend_key(kind, digest)
        if not self._t.exists(key):  # write-once: identical content is a no-op
            stored = self._envelope.seal(data) if kind in SEALED_KINDS else data
            try:
                self._t.put_bytes(key, stored)
            except WriteOnceConflictError:
                # A racing writer won the content-addressed key. Preserve it;
                # normal verify-on-read remains authoritative for its bytes.
                pass
        return ArtifactRef(
            digest=digest, kind=kind, byte_size=len(data), backend_key=key
        )

    def put_file(self, path: str | Path, kind: ArtifactKind) -> ArtifactRef:
        if self._public_read_only:
            raise ReadOnlyStoreError("public audit store is read-only")
        if kind in SEALED_KINDS and not self._allow_sealed_operations:
            raise SealedArtifactAccessError(
                "this store role cannot write sealed holdouts"
            )
        source = Path(path)
        digest, size = _hash_file(source)
        key = backend_key(kind, digest)
        if not self._t.exists(key):
            if kind in SEALED_KINDS:
                with tempfile.TemporaryDirectory(prefix="vidaio-seal-") as tmp:
                    sealed = Path(tmp) / "holdout.aesgcm"
                    _seal_file(self._envelope, source, sealed)
                    try:
                        self._t.put_file(key, sealed)
                    except WriteOnceConflictError:
                        pass
            else:
                try:
                    self._t.put_file(key, source)
                except WriteOnceConflictError:
                    pass
        return ArtifactRef(digest=digest, kind=kind, byte_size=size, backend_key=key)

    def get(self, ref: ArtifactRef) -> bytes:
        release_key = released_backend_key(ref.kind, ref.digest)
        if ref.kind in SEALED_KINDS and self._t.exists(release_key):
            data = self._t.get_bytes(release_key)
        elif ref.kind in SEALED_KINDS and not self._allow_sealed_operations:
            raise FileNotFoundError(
                f"sealed artifact {ref.kind.value}/{ref.digest} has not been released"
            )
        else:
            raw = self._t.get_bytes(backend_key(ref.kind, ref.digest))
            data = self._envelope.unseal(raw) if ref.kind in SEALED_KINDS else raw
        if len(data) != ref.byte_size or sha256_hex(data) != ref.digest:
            raise IntegrityError(
                f"artifact {ref.kind.value}/{ref.digest} failed verify-on-read: "
                "downloaded bytes do not match the content address"
            )
        return data

    def get_limited(self, ref: ArtifactRef, max_bytes: int) -> bytes:
        _require_download_bound(ref, max_bytes)
        release_key = released_backend_key(ref.kind, ref.digest)
        if ref.kind in SEALED_KINDS and self._t.exists(release_key):
            key = release_key
            sealed = False
        elif ref.kind in SEALED_KINDS and not self._allow_sealed_operations:
            raise FileNotFoundError(
                f"sealed artifact {ref.kind.value}/{ref.digest} has not been released"
            )
        else:
            key = backend_key(ref.kind, ref.digest)
            sealed = ref.kind in SEALED_KINDS
        size = self._t.head(key)
        if size is None:
            raise FileNotFoundError(key)
        if not sealed and size != ref.byte_size:
            raise IntegrityError(
                f"artifact {ref.kind.value}/{ref.digest} failed verify-on-read: "
                f"object size {size} != committed plaintext size {ref.byte_size}"
            )
        if not sealed and size > max_bytes:
            raise ArtifactTooLargeError(
                f"artifact {ref.kind.value}/{ref.digest} is {size} bytes; "
                f"auditor maximum is {max_bytes}"
            )
        # A privileged encrypted holdout has envelope overhead. It is never used
        # by a public auditor; retain authenticated one-shot decryption there.
        if sealed:
            return self.get(ref)
        data = self._raw_get_limited(key, max_bytes=max_bytes)
        if len(data) != ref.byte_size or sha256_hex(data) != ref.digest:
            raise IntegrityError(
                f"artifact {ref.kind.value}/{ref.digest} failed verify-on-read: "
                "downloaded bytes do not match the content address"
            )
        return data

    def get_digest_limited(
        self, kind: ArtifactKind, digest: str, *, max_bytes: int
    ) -> bytes:
        if max_bytes <= 0:
            raise ValueError("artifact read bound must be positive")
        if kind in SEALED_KINDS:
            key = released_backend_key(kind, digest)
            if not self._t.exists(key):
                if not self._allow_sealed_operations:
                    raise FileNotFoundError(
                        f"sealed artifact {kind.value}/{digest} has not been released"
                    )
                raise ValueError(
                    "digest-only reads of encrypted canonical holdouts are unsupported"
                )
        else:
            key = backend_key(kind, digest)
        size = self._t.head(key)
        if size is None:
            raise FileNotFoundError(key)
        if size > max_bytes:
            raise ArtifactTooLargeError(
                f"artifact {kind.value}/{digest} is {size} bytes; maximum is {max_bytes}"
            )
        data = self._raw_get_limited(key, max_bytes=max_bytes)
        if sha256_hex(data) != digest:
            raise IntegrityError(
                f"artifact {kind.value}/{digest} failed digest-only verify-on-read"
            )
        return data

    def exists(self, ref: ArtifactRef) -> bool:
        if ref.kind in SEALED_KINDS and not self._allow_sealed_operations:
            return self._t.exists(released_backend_key(ref.kind, ref.digest))
        return self._t.exists(backend_key(ref.kind, ref.digest))

    def open_stream(self, ref: ArtifactRef) -> BinaryIO:
        return io.BytesIO(self.get(ref))

    def materialize(
        self, ref: ArtifactRef, directory: str | Path, *, max_bytes: int
    ) -> Path:
        _require_download_bound(ref, max_bytes)
        release_key = released_backend_key(ref.kind, ref.digest)
        if ref.kind in SEALED_KINDS and self._t.exists(release_key):
            key = release_key
            sealed = False
        elif ref.kind in SEALED_KINDS and not self._allow_sealed_operations:
            raise FileNotFoundError(
                f"sealed artifact {ref.kind.value}/{ref.digest} has not been released"
            )
        else:
            key = backend_key(ref.kind, ref.digest)
            sealed = ref.kind in SEALED_KINDS
        if sealed:
            destination = Path(directory) / f"{ref.kind.value}-{ref.digest}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="vidaio-unseal-", dir=destination.parent
            ) as tmp:
                encrypted = Path(tmp) / "holdout.aesgcm"
                size = self._t.head(key)
                if size is None:
                    raise FileNotFoundError(key)
                if size > max_bytes + 1024:
                    raise ArtifactTooLargeError(
                        f"sealed artifact {ref.kind.value}/{ref.digest} exceeds its "
                        "plaintext auditor bound plus envelope overhead"
                    )
                self._t.get_file(key, encrypted, max_bytes=max_bytes + 1024)
                try:
                    _unseal_file(self._envelope, encrypted, destination)
                    _verify_path(destination, ref)
                except BaseException:
                    with contextlib.suppress(FileNotFoundError, OSError):
                        destination.unlink()
                    raise
            return destination
        size = self._t.head(key)
        if size is None:
            raise FileNotFoundError(key)
        if size != ref.byte_size:
            raise IntegrityError(
                f"artifact {ref.kind.value}/{ref.digest} failed verify-on-read: "
                f"object size {size} != committed plaintext size {ref.byte_size}"
            )
        if size > max_bytes:
            raise ArtifactTooLargeError(
                f"artifact {ref.kind.value}/{ref.digest} is {size} bytes; "
                f"auditor maximum is {max_bytes}"
            )
        destination = Path(directory) / f"{ref.kind.value}-{ref.digest}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        try:
            self._t.get_file(key, stage, max_bytes=max_bytes)
            digest, downloaded_size = _hash_file(stage)
            if downloaded_size != ref.byte_size or digest != ref.digest:
                raise IntegrityError(
                    f"artifact {ref.kind.value}/{ref.digest} failed verify-on-read: "
                    "downloaded bytes do not match the content address"
                )
            os.replace(stage, destination)
        finally:
            with contextlib.suppress(FileNotFoundError, OSError):
                stage.unlink()
        return destination

    def release(self, ref: ArtifactRef) -> None:
        if self._public_read_only:
            raise ReadOnlyStoreError("public audit store is read-only")
        if not self._allow_sealed_operations:
            raise SealedArtifactAccessError(
                "this store role cannot release sealed holdouts"
            )
        if ref.kind not in SEALED_KINDS:
            raise ValueError(f"{ref.kind.value} is not a sealed artifact")
        key = released_backend_key(ref.kind, ref.digest)
        if self._t.exists(key):
            with tempfile.TemporaryDirectory(prefix="vidaio-release-verify-") as tmp:
                path = Path(tmp) / "released"
                self._t.get_file(key, path, max_bytes=ref.byte_size)
                _verify_path(path, ref)
            return
        canonical = backend_key(ref.kind, ref.digest)
        with tempfile.TemporaryDirectory(prefix="vidaio-release-") as tmp:
            encrypted = Path(tmp) / "holdout.aesgcm"
            plaintext = Path(tmp) / "holdout"
            sealed_size = self._t.head(canonical)
            if sealed_size is None:
                raise FileNotFoundError(canonical)
            if sealed_size > ref.byte_size + 1024:
                raise IntegrityError(
                    f"sealed artifact {ref.kind.value}/{ref.digest} has impossible "
                    "envelope overhead"
                )
            self._t.get_file(canonical, encrypted, max_bytes=ref.byte_size + 1024)
            _unseal_file(self._envelope, encrypted, plaintext)
            _verify_path(plaintext, ref)
            try:
                self._t.put_file(key, plaintext)
            except WriteOnceConflictError:
                # A concurrent release is idempotent only if its plaintext is
                # the committed object; verify the winner before returning.
                existing = Path(tmp) / "existing-release"
                self._t.get_file(key, existing, max_bytes=ref.byte_size)
                _verify_path(existing, ref)

    def is_released(self, ref: ArtifactRef) -> bool:
        """Verify the released plaintext through this role's transport.

        For a public store the transport is explicitly unsigned, so success is a
        bounded anonymous read plus content/size verification.  A privileged store
        proves only its own access; CROWN publication therefore checks a separate
        public-store instance before finalizing the earning epoch.
        """
        if ref.kind not in SEALED_KINDS:
            return False
        key = released_backend_key(ref.kind, ref.digest)
        try:
            if self._t.head(key) != ref.byte_size:
                return False
            with tempfile.TemporaryDirectory(prefix="vidaio-release-check-") as tmp:
                path = Path(tmp) / "released"
                self._t.get_file(key, path, max_bytes=ref.byte_size)
                _verify_path(path, ref)
        except Exception:  # noqa: BLE001 - bool probe includes IAM/network failures
            return False
        return True


class S3Store(_TransportBackedStore):
    """S3 / S3-compatible object store (the production content layer).

    All logic is inherited from `_TransportBackedStore` and unit-tested with a fake
    transport; the real transport is `_RealS3Transport` (boto3), built lazily on
    first use. Install the SDK with:  uv pip install -e '.[storage]'
    Inject `transport=` to unit-test without boto3.
    """

    def __init__(
        self,
        config: AuditConfig,
        envelope: Envelope | None = None,
        *,
        transport: _ObjectTransport | None = None,
        public_read_only: bool = False,
        allow_sealed_operations: bool = True,
    ) -> None:
        super().__init__(
            transport=transport,
            connect=lambda: _connect_s3_transport(config, anonymous=public_read_only),
            envelope=envelope,
            public_read_only=public_read_only,
            allow_sealed_operations=allow_sealed_operations,
        )


class HippiusStore(_TransportBackedStore):
    """Hippius decentralized storage through its production S3 gateway."""

    def __init__(
        self,
        config: AuditConfig,
        envelope: Envelope | None = None,
        *,
        transport: _ObjectTransport | None = None,
        public_read_only: bool = False,
        allow_sealed_operations: bool = True,
    ) -> None:
        super().__init__(
            transport=transport,
            connect=lambda: _connect_hippius_transport(
                config, anonymous=public_read_only
            ),
            envelope=envelope,
            public_read_only=public_read_only,
            allow_sealed_operations=allow_sealed_operations,
        )


# ---- the real, boto3-backed S3 transport: the ONLY part not unit-tested --------------


def _connect_s3_transport(
    config: AuditConfig, *, anonymous: bool = False
) -> _ObjectTransport:
    """Build the real, boto3-backed S3 transport. Lazy import; fail fast.

    Raises NotConfiguredError (pointing at the '.[storage]' extra) if boto3 is not
    installed, or if the bucket is unset. Credentials come from the env vars NAMED
    in config, never from config values.
    """
    try:
        import boto3  # noqa: F401 - imported for its side of the seam
    except ImportError as exc:
        raise NotConfiguredError(
            "the S3 object-store backend needs the optional 'storage' dependency group "
            "— install it with:  uv pip install -e '.[storage]'  (boto3>=1.34)"
        ) from exc
    if not config.s3_bucket:
        raise NotConfiguredError(
            "S3 backend selected but audit.s3_bucket is empty — set the bucket name "
            "(credentials come from the env vars named by audit.s3_*_env)."
        )
    return _RealS3Transport(
        bucket=config.s3_bucket,
        prefix=config.s3_prefix,
        region=config.s3_region,
        endpoint=os.environ.get(config.s3_endpoint_url_env, "").strip() or None,
        access_key=None
        if anonymous
        else os.environ.get(config.s3_access_key_env) or None,
        secret_key=None
        if anonymous
        else os.environ.get(config.s3_secret_key_env) or None,
        anonymous=anonymous,
    )


def _connect_hippius_transport(
    config: AuditConfig, *, anonymous: bool = False
) -> _ObjectTransport:
    """Connect to Hippius' S3-compatible gateway with path-style SigV4."""
    try:
        import boto3  # noqa: F401
    except ImportError as exc:
        raise NotConfiguredError(
            "the Hippius backend needs the optional 'storage' dependency group"
        ) from exc
    if not config.hippius_bucket.strip():
        raise NotConfiguredError(
            "Hippius backend selected but audit.hippius_bucket is empty"
        )
    access_key = os.environ.get(config.hippius_access_key_env, "").strip()
    secret_key = os.environ.get(config.hippius_secret_key_env, "").strip()
    if not anonymous and (not access_key or not secret_key):
        raise NotConfiguredError(
            "Hippius credentials are missing; set "
            f"${config.hippius_access_key_env} and ${config.hippius_secret_key_env}"
        )
    endpoint = os.environ.get(config.hippius_endpoint_env, "").strip()
    return _RealS3Transport(
        bucket=config.hippius_bucket,
        prefix=config.hippius_prefix,
        region=config.hippius_region or "decentralized",
        endpoint=endpoint or "https://s3.hippius.com",
        access_key=None if anonymous else access_key,
        secret_key=None if anonymous else secret_key,
        path_style=True,
        anonymous=anonymous,
    )


def _is_s3_precondition_failure(exc: BaseException) -> bool:
    """Whether an S3 error proves a conditional create lost to an existing key.

    Keep this independent of ``botocore`` so the transport contract can be tested
    without the optional storage dependency. A 409 concurrent-write response is
    deliberately *not* treated as success: callers must retry it, whereas 412 is
    positive proof that ``If-None-Match: *`` protected an already-created object.
    """

    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    metadata = response.get("ResponseMetadata", {})
    error = response.get("Error", {})
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    code = error.get("Code") if isinstance(error, dict) else None
    return status == 412 or str(code) in {"412", "PreconditionFailed"}


class _RealS3Transport:
    """The real seam: one boto3 S3 client using atomic conditional creates.

    Every write sends ``If-None-Match: *``. A precondition loser raises the typed
    ``WriteOnceConflictError`` and never overwrites the winning object. Store-level
    content-addressed repeats and byte-identical set writes remain idempotent; a
    different payload at one logical set key stays a visible conflict.
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str,
        region: str,
        endpoint: str | None,
        access_key: str | None,
        secret_key: str | None,
        path_style: bool = False,
        anonymous: bool = False,
    ) -> None:  # pragma: no cover - needs boto3
        import boto3

        kwargs: dict[str, object] = {}
        from botocore.config import Config

        # Do not inherit botocore's comparatively long/unbounded-by-policy
        # defaults here. Audit publication is best-effort after a weight write;
        # a stalled object-store endpoint must return control to the scheduler.
        # ``total_max_attempts`` includes the initial request, unlike the older
        # ``max_attempts`` spelling whose interpretation differs by config source.
        config_kwargs: dict[str, object] = {
            "connect_timeout": _S3_CONNECT_TIMEOUT_SECONDS,
            "read_timeout": _S3_READ_TIMEOUT_SECONDS,
            "retries": {
                "mode": "standard",
                "total_max_attempts": _S3_TOTAL_MAX_ATTEMPTS,
            },
        }
        if anonymous:
            from botocore import UNSIGNED

            config_kwargs["signature_version"] = UNSIGNED
        elif path_style:
            config_kwargs["signature_version"] = "s3v4"
        if path_style:
            config_kwargs["s3"] = {"addressing_style": "path"}
        kwargs["config"] = Config(**config_kwargs)
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = boto3.client(
            "s3",
            region_name=region or None,
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            **kwargs,
        )

    def _full(self, key: str) -> str:  # pragma: no cover - needs boto3
        return f"{self._prefix}/{key}" if self._prefix else key

    def put_bytes(self, key: str, payload: bytes) -> None:
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=self._full(key),
                Body=payload,
                IfNoneMatch="*",
            )
        except Exception as exc:  # noqa: BLE001 - optional SDK error type
            if _is_s3_precondition_failure(exc):
                raise WriteOnceConflictError(
                    f"conditional S3 create lost for {key!r}"
                ) from exc
            raise

    def put_file(self, key: str, path: Path) -> None:
        # ``upload_file`` does not expose PutObject's conditional-create contract
        # consistently across boto3 releases/providers, so use PutObject directly.
        # botocore streams this file object without loading it into memory.
        with path.open("rb") as source:
            try:
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=self._full(key),
                    Body=source,
                    ContentLength=path.stat().st_size,
                    IfNoneMatch="*",
                )
            except Exception as exc:  # noqa: BLE001 - optional SDK error type
                if _is_s3_precondition_failure(exc):
                    raise WriteOnceConflictError(
                        f"conditional S3 create lost for {key!r}"
                    ) from exc
                raise

    def get_bytes(self, key: str) -> bytes:  # pragma: no cover - needs boto3
        resp = self._client.get_object(Bucket=self._bucket, Key=self._full(key))
        body = resp["Body"]
        try:
            return body.read()
        finally:
            body.close()

    def get_file(
        self, key: str, path: Path, *, max_bytes: int
    ) -> None:  # pragma: no cover - needs boto3
        response = self._client.get_object(Bucket=self._bucket, Key=self._full(key))
        body = response["Body"]
        written = 0
        try:
            with path.open("xb") as sink:
                for chunk in body.iter_chunks(chunk_size=_CHUNK):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        raise ArtifactTooLargeError(
                            f"object {key} crossed the {max_bytes}-byte download maximum"
                        )
                    sink.write(chunk)
                sink.flush()
                os.fsync(sink.fileno())
        except BaseException:
            with contextlib.suppress(FileNotFoundError, OSError):
                path.unlink()
            raise
        finally:
            body.close()

    def head(self, key: str) -> int | None:  # pragma: no cover - needs boto3
        from botocore.exceptions import ClientError

        try:
            resp = self._client.head_object(Bucket=self._bucket, Key=self._full(key))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in (
                "404",
                "NoSuchKey",
                "NotFound",
            ):
                return None
            raise
        return int(resp["ContentLength"])

    def exists(self, key: str) -> bool:  # pragma: no cover - needs boto3
        return self.head(key) is not None


def make_store(config: AuditConfig, envelope: Envelope | None = None) -> AuditStore:
    """Instantiate the configured backend (local | s3 | hippius).

    Refuses to build a store whose reference_original envelope is the no-op
    PassthroughEnvelope (i.e. a plaintext holdout at rest) unless the config
    explicitly opts in via `allow_plaintext_holdout: true`.
    """
    if envelope is None:
        envelope = (
            PassthroughEnvelope()
            if config.allow_plaintext_holdout
            else AesGcmEnvelope.from_env(config.holdout_key_env)
        )
    elif (
        isinstance(envelope, PassthroughEnvelope) and not config.allow_plaintext_holdout
    ):
        raise NotConfiguredError(
            "refusing to build an audit store with a plaintext holdout; set "
            "audit.allow_plaintext_holdout=true only in dev/test"
        )
    if config.backend == "local":
        return LocalFsStore(config.local_root, envelope)
    if config.backend == "s3":
        return S3Store(config, envelope)
    return HippiusStore(config, envelope)


def make_public_store(config: AuditConfig) -> AuditStore:
    """Build the anonymous, read-only view used by independent auditors.

    This path never loads the live holdout AES key and never falls back from
    ``released/reference_original`` to the private canonical namespace. Remote
    clients are explicitly unsigned; local mode exists for parity tests.
    """
    if config.backend == "local":
        return LocalFsStore(
            config.local_root,
            PassthroughEnvelope(),
            public_read_only=True,
            allow_sealed_operations=False,
        )
    if config.backend == "s3":
        return S3Store(
            config,
            PassthroughEnvelope(),
            public_read_only=True,
            allow_sealed_operations=False,
        )
    return HippiusStore(
        config,
        PassthroughEnvelope(),
        public_read_only=True,
        allow_sealed_operations=False,
    )


class _UnsealedWriterStore:
    """Application boundary for the thin validator's storage credentials.

    Bucket IAM is still required as defense in depth, but the application must
    not turn a broadly configured credential into authority-like write access.
    The thin weight-setter only publishes its content-addressed weight vector and
    the manifest that anchors that publication.  It may read finalized epoch sets
    in order to converge on the authority's vector, but it never creates or
    finalizes those sets and can never release a sealed holdout.
    """

    _WRITE_KINDS = frozenset({ArtifactKind.MANIFEST, ArtifactKind.WEIGHT_VECTOR})

    def __init__(self, store: LocalFsStore | S3Store | HippiusStore) -> None:
        self._store = store

    @property
    def public_read_only(self) -> bool:
        """This role has signed write credentials; it is not an anonymous proof."""
        return False

    @classmethod
    def _require_write_kind(cls, kind: ArtifactKind) -> None:
        if kind in SEALED_KINDS:
            # Preserve the keyless-store contract and its security-specific
            # exception for callers that distinguish sealed holdouts.
            raise SealedArtifactAccessError(
                "this store role cannot write sealed holdouts"
            )
        if kind not in cls._WRITE_KINDS:
            raise ReadOnlyStoreError(
                "thin-validator audit-store credentials may write only "
                f"{ArtifactKind.MANIFEST.value} and {ArtifactKind.WEIGHT_VECTOR.value}; "
                f"refusing {kind.value}"
            )

    def put(self, data: bytes, kind: ArtifactKind) -> ArtifactRef:
        self._require_write_kind(kind)
        return self._store.put(data, kind)

    def put_file(self, path: str | Path, kind: ArtifactKind) -> ArtifactRef:
        self._require_write_kind(kind)
        return self._store.put_file(path, kind)

    # Public evidence remains readable through this role.  SharedSnapshotProvider
    # also needs the two set READ operations below to mirror finalized epoch logs.
    def get(self, ref: ArtifactRef) -> bytes:
        return self._store.get(ref)

    def get_limited(self, ref: ArtifactRef, max_bytes: int) -> bytes:
        return self._store.get_limited(ref, max_bytes)

    def get_digest_limited(
        self, kind: ArtifactKind, digest: str, *, max_bytes: int
    ) -> bytes:
        return self._store.get_digest_limited(kind, digest, max_bytes=max_bytes)

    def exists(self, ref: ArtifactRef) -> bool:
        return self._store.exists(ref)

    def open_stream(self, ref: ArtifactRef) -> BinaryIO:
        return self._store.open_stream(ref)

    def materialize(
        self, ref: ArtifactRef, directory: str | Path, *, max_bytes: int
    ) -> Path:
        return self._store.materialize(ref, directory, max_bytes=max_bytes)

    def is_released(self, ref: ArtifactRef) -> bool:
        return self._store.is_released(ref)

    def is_finalized(self, prefix: str) -> bool:
        return self._store.is_finalized(prefix)

    def get_set_member(
        self,
        prefix: str,
        name: str,
        *,
        expected_digest: str | None = None,
        byte_size: int | None = None,
        max_bytes: int | None = None,
    ) -> bytes:
        return self._store.get_set_member(
            prefix,
            name,
            expected_digest=expected_digest,
            byte_size=byte_size,
            max_bytes=max_bytes,
        )

    def put_set_member(
        self,
        prefix: str,
        name: str,
        data: bytes,
        kind: ArtifactKind = ArtifactKind.EPOCH_LOG,
    ) -> ArtifactRef:
        del prefix, name, data, kind
        raise ReadOnlyStoreError(
            "thin validators may read finalized epoch sets but cannot mutate them"
        )

    def finalize_set(self, prefix: str) -> None:
        del prefix
        raise ReadOnlyStoreError(
            "thin validators may read finalized epoch sets but cannot finalize them"
        )

    def release(self, ref: ArtifactRef) -> None:
        del ref
        raise SealedArtifactAccessError(
            "this store role cannot release sealed holdouts"
        )


def make_unsealed_writer_store(config: AuditConfig) -> AuditStore:
    """Build a keyless writer for public evidence kinds.

    Thin validators publish weight vectors/manifests, so they cannot use the
    anonymous read-only view. This composition uses normal scoped S3 credentials
    (or workload IAM), permits public-kind reads, allowlists only ``MANIFEST`` and
    ``WEIGHT_VECTOR`` writes, and rejects finalized-set mutation and every live
    sealed-holdout operation before touching storage. It therefore needs no AES
    holdout key; bucket IAM should independently apply the same write allowlist
    and deny the canonical ``reference_original/`` prefix.
    """
    if config.backend == "local":
        store: LocalFsStore | S3Store | HippiusStore = LocalFsStore(
            config.local_root,
            PassthroughEnvelope(),
            allow_sealed_operations=False,
        )
    elif config.backend == "s3":
        store = S3Store(
            config,
            PassthroughEnvelope(),
            allow_sealed_operations=False,
        )
    else:
        store = HippiusStore(
            config,
            PassthroughEnvelope(),
            allow_sealed_operations=False,
        )
    return _UnsealedWriterStore(store)
