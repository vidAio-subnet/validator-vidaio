"""Injectable metric backends behind Protocols + the deterministic fake for tests.

Metric computation (VMAF, PieAPP, decoding probes, perceptual checks) is expensive and
environment-dependent, so it lives behind these Protocols. The scoring module itself is
pure composition/gating/aggregation and never shells out. Real ffmpeg/GPU backends land
in a later phase; ``DeterministicFakeBackend`` implements every Protocol from supplied
mappings and is used in tests and golden dry-runs.

Derandomization (spec §08): legacy PieAPP sampled 4 consecutive frames from a *random*
start, which made live upscaling scores unreproducible. Here the start frame is derived
deterministically from ``sha256(content_digest || challenge_id)`` so a verifier can
recompute it, while a miner cannot predict it before the challenge_id is assigned.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel


class MediaInfo(BaseModel):
    """Probe result for a media file — every stream property the gates consume."""

    model_config = {"frozen": True}

    codec: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    byte_size: int
    bit_depth: int = 8
    pix_fmt: str = "yuv420p"


class PerceptualCheckResult(BaseModel):
    """Outcome of one backend-driven perceptual check (tone / grayscale / chroma)."""

    model_config = {"frozen": True}

    passed: bool
    measure: float | None = None
    #: Numeric decision surface for audit-boundary hysteresis. ``None`` is
    #: permitted for legacy/test backends that do not expose a boundary.
    limit: float | None = None
    comparison: Literal["maximum", "minimum"] | None = None
    detail: str = ""


@runtime_checkable
class VmafBackend(Protocol):
    """Full-reference VMAF on the 0-100 scale (NEG model, deterministic settings)."""

    name: str
    version: str

    def compute(
        self, reference: str, candidate: str, *, deterministic_seed: int = 0
    ) -> float: ...


@runtime_checkable
class PieAppBackend(Protocol):
    """PieAPP perceptual distance (lower = closer to reference), windowed sampling."""

    name: str
    version: str

    def compute(self, reference: str, candidate: str, *, start_frame: int) -> float: ...


@runtime_checkable
class ProbeBackend(Protocol):
    """Container/stream probe (ffprobe-shaped)."""

    def probe(self, path: str) -> MediaInfo: ...


@runtime_checkable
class PerceptualCheckBackend(Protocol):
    """Backend-driven manipulation checks. `passed=False` means manipulation detected."""

    def check_tone_manipulation(
        self, reference: str, candidate: str
    ) -> PerceptualCheckResult: ...

    def check_color_grayscale(
        self, reference: str, candidate: str
    ) -> PerceptualCheckResult: ...

    def check_chroma_uv(
        self, reference: str, candidate: str
    ) -> PerceptualCheckResult: ...


@runtime_checkable
class PerceptualHashBackend(Protocol):
    """CPU perceptual hash for public-corpus near-duplicate screening."""

    def compute_phash(self, path: str) -> str: ...

    def distance(self, hash_a: str, hash_b: str) -> int: ...


def usable_frames(frame_count: int, sample_window: int) -> int:
    """Number of valid start positions for a `sample_window`-frame consecutive sample."""
    return max(0, frame_count - sample_window + 1)


def derive_pieapp_start_frame(
    content_digest: str, challenge_id: str, usable_frames: int
) -> int:
    """Deterministic, verifier-recomputable PieAPP start frame (spec §08 fix).

    ``sha256(content_digest || challenge_id) mod usable_frames`` — the two inputs are
    joined with a NUL byte so no (digest, id) pair is ambiguous under concatenation.
    Deterministic given the challenge, but unpredictable to a miner before the
    challenge_id is assigned; contains no scoring-time randomness.
    """
    if usable_frames < 1:
        raise ValueError("usable_frames must be >= 1")
    digest = hashlib.sha256(
        content_digest.encode("utf-8") + b"\x00" + challenge_id.encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") % usable_frames


def _pair(reference: str, candidate: str) -> tuple[str, str]:
    return (reference, candidate)


class DeterministicFakeBackend:
    """Implements every backend Protocol from supplied mappings. Zero randomness.

    Missing keys raise ``KeyError`` loudly — a fake must never invent a metric value.
    """

    name = "deterministic-fake"
    version = "1"

    def __init__(
        self,
        *,
        vmaf: Mapping[tuple[str, str], float] | None = None,
        pieapp: Mapping[tuple[str, str], float] | None = None,
        media: Mapping[str, MediaInfo] | None = None,
        perceptual_checks: Mapping[str, PerceptualCheckResult] | None = None,
        phashes: Mapping[str, str] | None = None,
    ) -> None:
        self._vmaf = dict(vmaf or {})
        self._pieapp = dict(pieapp or {})
        self._media = dict(media or {})
        self._checks = dict(perceptual_checks or {})
        self._phashes = dict(phashes or {})
        #: (reference, candidate, start_frame) of every PieAPP call, for assertions.
        self.pieapp_calls: list[tuple[str, str, int]] = []

    # VmafBackend
    def compute(
        self, reference: str, candidate: str, *, deterministic_seed: int = 0
    ) -> float:
        return self._vmaf[_pair(reference, candidate)]

    # PieAppBackend — `compute` already serves VmafBackend, so the PieAPP view is an
    # adapter object with its own `compute(reference, candidate, *, start_frame)`.
    @property
    def pieapp(self) -> "_FakePieApp":
        return _FakePieApp(self)

    # ProbeBackend
    def probe(self, path: str) -> MediaInfo:
        return self._media[path]

    # PerceptualCheckBackend
    def check_tone_manipulation(
        self, reference: str, candidate: str
    ) -> PerceptualCheckResult:
        return self._checks.get("tone", PerceptualCheckResult(passed=True))

    def check_color_grayscale(
        self, reference: str, candidate: str
    ) -> PerceptualCheckResult:
        return self._checks.get("grayscale", PerceptualCheckResult(passed=True))

    def check_chroma_uv(self, reference: str, candidate: str) -> PerceptualCheckResult:
        return self._checks.get("chroma", PerceptualCheckResult(passed=True))

    # PerceptualHashBackend
    def compute_phash(self, path: str) -> str:
        return self._phashes[path]

    def distance(self, hash_a: str, hash_b: str) -> int:
        """Hamming distance over the hex-decoded hashes."""
        return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()

    def versions(self) -> dict[str, str]:
        """Backend-version stamp for the audit record."""
        return {
            "vmaf": f"{self.name}/{self.version}",
            "pieapp": f"{self.name}/{self.version}",
        }


class _FakePieApp:
    """PieAppBackend view over a DeterministicFakeBackend (records start frames)."""

    def __init__(self, parent: DeterministicFakeBackend) -> None:
        self._parent = parent
        self.name = parent.name
        self.version = parent.version

    def compute(self, reference: str, candidate: str, *, start_frame: int) -> float:
        self._parent.pieapp_calls.append((reference, candidate, start_frame))
        return self._parent._pieapp[_pair(reference, candidate)]
