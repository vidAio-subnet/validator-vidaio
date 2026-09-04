"""Shared fixtures for the scoring-worker suite.

Real-media fixtures generate tiny clips with ffmpeg's lavfi testsrc2 (1s, 160x120,
ultrafast) so every real test stays fast; tests that shell out are skipped cleanly
on machines without ffmpeg+libvmaf (CI-without-media runs the fake-backend suite).
"""

from __future__ import annotations

import functools
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from vidaio.scoring.backends import MediaInfo, PerceptualCheckResult

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")


@functools.lru_cache(maxsize=1)
def has_media_tools() -> bool:
    """ffmpeg + ffprobe on PATH, with the libvmaf filter compiled in."""
    if FFMPEG is None or FFPROBE is None:
        return False
    completed = subprocess.run(
        [FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=30
    )
    return completed.returncode == 0 and "libvmaf" in completed.stdout


requires_media_tools = pytest.mark.skipif(
    not has_media_tools(), reason="ffmpeg/ffprobe with libvmaf not available"
)


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RoleKeyedBackend:
    """Path-keyed fake backend, addressed by input ROLE instead of by path.

    The worker never measures the paths a request names: it snapshots every
    verified input into a private per-request directory (``reference*``,
    ``miner_input*``, ``output*``) and measures those. A fake keyed by the
    caller-supplied path therefore cannot see the files the pipeline actually
    reads — which is the entire point of the TOCTOU fix. This adapter maps any
    path back to its role (the file stem) and delegates, so tests keep stating
    "the reference scores 93" without knowing where the worker put it.
    """

    name = "role-keyed-fake"
    version = "1"

    def __init__(
        self,
        *,
        vmaf: dict[tuple[str, str], float] | None = None,
        pieapp: dict[tuple[str, str], float] | None = None,
        media: dict[str, MediaInfo] | None = None,
        perceptual_checks: dict[str, PerceptualCheckResult] | None = None,
    ) -> None:
        self._vmaf = dict(vmaf or {})
        self._pieapp = dict(pieapp or {})
        self._media = dict(media or {})
        self._checks = dict(perceptual_checks or {})
        #: (reference_role, candidate_role, start_frame) of every PieAPP call.
        self.pieapp_calls: list[tuple[str, str, int]] = []
        #: every path handed to probe(), in order — the paths really measured.
        self.probed_paths: list[str] = []
        #: (reference_role, candidate_role) for every VMAF run.
        self.vmaf_calls: list[tuple[str, str]] = []
        #: (gate, reference_role, candidate_role) for perceptual-basis assertions.
        self.perceptual_calls: list[tuple[str, str, str]] = []

    @staticmethod
    def role(path: str) -> str:
        return Path(path).stem

    def set_media(self, role: str, info: MediaInfo) -> None:
        """Restate one role's media facts (e.g. an upscaling-shaped miner input)."""
        self._media[role] = info

    # VmafBackend
    def compute(
        self, reference: str, candidate: str, *, deterministic_seed: int = 0
    ) -> float:
        key = (self.role(reference), self.role(candidate))
        self.vmaf_calls.append(key)
        # Most pre-delta-basis fixtures describe compression with one score;
        # compression miner_input == reference in bytes. Preserve that concise
        # setup while allowing dedicated tests to provide a distinct input pair.
        fallback = ("reference", key[1]) if key[0] == "miner_input" else key
        return self._vmaf[key if key in self._vmaf else fallback]

    # PieAppBackend view
    @property
    def pieapp(self) -> "_RoleKeyedPieApp":
        return _RoleKeyedPieApp(self)

    # ProbeBackend
    def probe(self, path: str) -> MediaInfo:
        self.probed_paths.append(path)
        return self._media[self.role(path)]

    # PerceptualCheckBackend
    def check_tone_manipulation(
        self, reference: str, candidate: str
    ) -> PerceptualCheckResult:
        self.perceptual_calls.append(
            ("tone", self.role(reference), self.role(candidate))
        )
        return self._checks.get("tone", PerceptualCheckResult(passed=True))

    def check_color_grayscale(
        self, reference: str, candidate: str
    ) -> PerceptualCheckResult:
        self.perceptual_calls.append(
            ("grayscale", self.role(reference), self.role(candidate))
        )
        return self._checks.get("grayscale", PerceptualCheckResult(passed=True))

    def check_chroma_uv(self, reference: str, candidate: str) -> PerceptualCheckResult:
        self.perceptual_calls.append(
            ("chroma", self.role(reference), self.role(candidate))
        )
        return self._checks.get("chroma", PerceptualCheckResult(passed=True))

    def versions(self) -> dict[str, str]:
        return {
            "vmaf": f"{self.name}/{self.version}",
            "pieapp": f"{self.name}/{self.version}",
        }


class _RoleKeyedPieApp:
    def __init__(self, parent: RoleKeyedBackend) -> None:
        self._parent = parent
        self.name = parent.name
        self.version = parent.version

    def compute(self, reference: str, candidate: str, *, start_frame: int) -> float:
        roles = (RoleKeyedBackend.role(reference), RoleKeyedBackend.role(candidate))
        self._parent.pieapp_calls.append((*roles, start_frame))
        return self._parent._pieapp[roles]


def worker_scorer_version(config: Any, scoring_config: Any = None) -> str:
    """The version the worker will stamp for `config` (tests must not guess it)."""
    from vidaio.scoring import ScoringConfig
    from vidaio.scoring_worker import effective_scorer_version

    return effective_scorer_version(
        config, scoring_config if scoring_config is not None else ScoringConfig()
    )


@dataclass(frozen=True)
class ClipPair:
    """A tiny reference clip and a lower-bitrate re-encode of it (compression case)."""

    reference: str
    reference_digest: str
    candidate: str
    candidate_digest: str


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", *args],
        check=True,
        capture_output=True,
        timeout=60,
    )


@pytest.fixture(scope="session")
def clips(tmp_path_factory: pytest.TempPathFactory) -> ClipPair:
    if not has_media_tools():
        pytest.skip("ffmpeg/ffprobe with libvmaf not available")
    root = tmp_path_factory.mktemp("clips")
    reference = root / "ref.mp4"
    candidate = root / "cand.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x120:rate=10:duration=1",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-y",
        str(reference),
    )
    _ffmpeg(
        "-i",
        str(reference),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-b:v",
        "40k",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(candidate),
    )
    ratio = candidate.stat().st_size / reference.stat().st_size
    assert ratio < 0.80, f"fixture candidate not compressive enough (rate {ratio:.2f})"
    return ClipPair(
        reference=str(reference),
        reference_digest=sha256_file(reference),
        candidate=str(candidate),
        candidate_digest=sha256_file(candidate),
    )


def score_request_body(
    *,
    track: str,
    reference: str,
    reference_digest: str,
    output: str,
    output_digest: str,
    miner_input: str | None = None,
    miner_input_digest: str | None = None,
    params: dict | None = None,
    challenge_id: str = "chal-1",
    item_id: str = "item-1",
    miner_hotkey: str | None = "hk-test",
    scorer_version: str | None = None,
) -> dict:
    """A ScoreRequest JSON body; miner_input defaults to the reference (compression).

    ``scorer_version`` defaults to absent: the worker stamps its own, and a body
    that named one would be asserting which scorer must answer (409 if wrong).
    """
    return {
        "track": track,
        "challenge_id": challenge_id,
        "item_id": item_id,
        "miner_hotkey": miner_hotkey,
        "reference_path": reference,
        "reference_digest": reference_digest,
        "miner_input_path": miner_input if miner_input is not None else reference,
        "miner_input_digest": (
            miner_input_digest if miner_input_digest is not None else reference_digest
        ),
        "output_path": output,
        "output_digest": output_digest,
        "params": params or {},
        "scorer_version": scorer_version,
    }
