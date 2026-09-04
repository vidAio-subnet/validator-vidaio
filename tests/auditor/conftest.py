"""Shared fixtures for the auditor suite.

Real-media fixtures reuse the scoring-worker suite's tiny-clip approach (ffmpeg
lavfi testsrc2) so recompute-parity tests stay fast and skip cleanly without
ffmpeg+libvmaf. The logic tests (sampling, weight re-derivation, aggregation) use
no media at all.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.scoring_worker.conftest import (
    FFMPEG,
    FFPROBE,
    has_media_tools,
    requires_media_tools,
    sha256_file,
)
from vidaio.audit.bundle import AuditBundle, LifecycleStage, build_bundle
from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind, LocalFsStore
from vidaio.scoring import ScoringConfig, DeterministicFakeBackend
from vidaio.scoring.backends_real import (
    CanonicalizeExecutor,
    FfmpegVmafBackend,
    FfprobeBackend,
    PieAppTorchBackend,
    SECONDARY_VMAF_MODEL,
    detect_tool_versions,
)
from vidaio.scoring.result import ItemScore
from vidaio.scoring_worker import (
    ScoringBackends,
    ScoringWorkerConfig,
    effective_scorer_version,
)
from vidaio.scoring_worker.service import _score_sync
from vidaio.services.protocol import ScoreRequest

NOW = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)

__all__ = ["requires_media_tools"]


@dataclass(frozen=True)
class ClipPair:
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
    root = tmp_path_factory.mktemp("auditor-clips")
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
    return ClipPair(
        reference=str(reference),
        reference_digest=sha256_file(reference),
        candidate=str(candidate),
        candidate_digest=sha256_file(candidate),
    )


@pytest.fixture(scope="module")
def worker_config(tmp_path_factory: pytest.TempPathFactory) -> ScoringWorkerConfig:
    return ScoringWorkerConfig(
        work_dir=tmp_path_factory.mktemp("auditor-scoring-work"),
        ffmpeg_path=FFMPEG or "ffmpeg",
        ffprobe_path=FFPROBE or "ffprobe",
        request_timeout=120.0,
        subprocess_timeout=60.0,
    )


@pytest.fixture(scope="module")
def scoring_config() -> ScoringConfig:
    return ScoringConfig()


@pytest.fixture(scope="module")
def real_media_backends(worker_config: ScoringWorkerConfig) -> ScoringBackends:
    """ffmpeg/libvmaf plus explicit non-PieAPP test doubles.

    Compression recomputes for real. PieAPP is deliberately marked unavailable
    because these fixtures test compression; host media extras must not alter them.
    """
    if not has_media_tools():
        pytest.skip("ffmpeg/ffprobe with libvmaf not available")
    primary = FfmpegVmafBackend(FFMPEG, work_dir=worker_config.work_dir, timeout=60.0)
    secondary = FfmpegVmafBackend(
        FFMPEG,
        model=SECONDARY_VMAF_MODEL,
        work_dir=worker_config.work_dir,
        timeout=60.0,
    )
    return ScoringBackends(
        probe=FfprobeBackend(FFPROBE, timeout=60.0),
        vmaf_primary=primary,
        vmaf_secondary=secondary,
        pieapp=PieAppTorchBackend(_backend_version="not-configured"),
        perceptual=DeterministicFakeBackend(),
        canonicalizer=CanonicalizeExecutor(FFMPEG, timeout=60.0),
        versions=detect_tool_versions(
            FFMPEG, FFPROBE, vmaf_backend=primary, timeout=30.0
        ),
    )


def score_compression_item(
    config: ScoringWorkerConfig,
    backends: ScoringBackends,
    scoring_config: ScoringConfig,
    clips: ClipPair,
    *,
    challenge_id: str = "chal-audit",
    item_id: str = "item-audit",
    miner_hotkey: str = "hk-audit",
) -> ItemScore:
    """Produce an HONEST compression ItemScore via the worker's own pipeline."""
    scorer_version = effective_scorer_version(config, scoring_config)
    request = ScoreRequest(
        track="compression",
        challenge_id=challenge_id,
        item_id=item_id,
        miner_hotkey=miner_hotkey,
        reference_path=clips.reference,
        reference_digest=clips.reference_digest,
        miner_input_path=clips.reference,
        miner_input_digest=clips.reference_digest,
        output_path=clips.candidate,
        output_digest=clips.candidate_digest,
        params={"vmaf_threshold": 90.0},
    )
    return _score_sync(request, config, scoring_config, backends, scorer_version)


def build_real_bundle(
    store: LocalFsStore,
    clips: ClipPair,
    item: ItemScore,
    *,
    packet_bytes: bytes | None = None,
    challenge_id: str = "chal-audit",
    item_id: str = "item-audit",
    miner_hotkey: str = "hk-audit",
    committed_track: str = "compression",
    dispatch_ordering_key: int = 0,
) -> AuditBundle:
    """Store the real artifact set for a scored item and return its bundle.

    ``packet_bytes`` overrides the score-packet blob (to plant a tampered packet);
    it defaults to the honest ``item.to_json()``. The DAG_REVEAL artifact is a real
    challenge-commitment preimage carrying the pre-dispatch committed ``track`` +
    ``dispatch_ordering_key``, so the auditor's earning path can bind
    the committed fold order/track to the anchored commitment.
    """
    from vidaio.challenge.commitment import ChallengeCommitment

    ref_bytes = Path(clips.reference).read_bytes()
    cand_bytes = Path(clips.candidate).read_bytes()
    dag_bytes = ChallengeCommitment.preimage_payload(
        f"asset-{item_id}",
        sha256_hex(b"dag-" + item_id.encode()),
        8675309,
        item.scorer_version or "scorer",
        committed_track,
        dispatch_ordering_key,
    )
    packet = (
        packet_bytes if packet_bytes is not None else item.to_json().encode("utf-8")
    )

    challenge_input = store.put(ref_bytes, ArtifactKind.CHALLENGE_INPUT)
    miner_output = store.put(cand_bytes, ArtifactKind.MINER_OUTPUT)
    manifest = store.put(
        canonical_json_bytes({"vmaf_threshold": 90}), ArtifactKind.MANIFEST
    )
    score_packet = store.put(packet, ArtifactKind.SCORE_PACKET)
    reference_original = store.put(ref_bytes, ArtifactKind.REFERENCE_ORIGINAL)
    dag_reveal = store.put(dag_bytes, ArtifactKind.DAG_REVEAL)

    return build_bundle(
        challenge_id=challenge_id,
        item_id=item_id,
        miner_hotkey=miner_hotkey,
        commitment_hash=sha256_hex(dag_bytes),
        stage=LifecycleStage.POST_RETIREMENT,
        challenge_input=challenge_input,
        miner_output=miner_output,
        manifest=manifest,
        score_packet=score_packet,
        reference_original=reference_original,
        dag_reveal=dag_reveal,
        scorer_version=item.scorer_version
        or effective_scorer_version(
            ScoringWorkerConfig(), scoring_config=ScoringConfig()
        ),
        backend_versions=dict(item.backend_versions),
        created_at="2026-08-21T12:00:00+00:00",
    )
