"""A released upscaling competition score is reproducible by a keyless CPU auditor."""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.scoring_worker.conftest import (
    FFMPEG,
    FFPROBE,
    requires_media_tools,
    sha256_file,
)
from vidaio.audit import (
    ArtifactKind,
    AuditConfig,
    CompetitionItemBinding,
    LifecycleStage,
    LocalFsStore,
    build_bundle,
    make_public_store,
    merkle_proof,
    merkle_root,
    verify_bundle,
)
from vidaio.audit.recompute import CompetitionAuditContext
from vidaio.competition import CompetitionManifest, evaluation_item_commitment
from vidaio.auditor.recomputer import RealScoreRecomputer
from vidaio.scoring import ScoringConfig
from vidaio.scoring.backends_real import NotConfiguredError
from vidaio.scoring_worker import ScoringWorkerConfig, effective_scorer_version, real_backends
from vidaio.scoring_worker.service import _score_sync
from vidaio.services.protocol import ScoreRequest

pytestmark = requires_media_tools

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", *args],
        check=True,
        capture_output=True,
        timeout=60,
    )


def _media_triplet(root: Path) -> tuple[Path, Path, Path]:
    """Pristine 2x reference, low-res miner input, and a genuine miner upscale."""
    reference = root / "reference.mp4"
    miner_input = root / "miner-input.mp4"
    output = root / "output.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=192x128:rate=8:duration=1",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "ultrafast",
        "-y",
        str(reference),
    )
    _ffmpeg(
        "-i",
        str(reference),
        "-vf",
        "scale=96:64:flags=lanczos",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "ultrafast",
        "-y",
        str(miner_input),
    )
    _ffmpeg(
        "-i",
        str(miner_input),
        "-vf",
        "scale=192:128:flags=lanczos",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "ultrafast",
        "-y",
        str(output),
    )
    return reference, miner_input, output


def _manifest(
    *, reference_digest: str, input_digest: str, scorer_version: str
) -> tuple[CompetitionManifest, str]:
    competition_id = "comp-public-cpu-audit"
    item_commitment = evaluation_item_commitment(
        competition_id=competition_id,
        item_index=0,
        reference_sha256=reference_digest,
        input_sha256=input_digest,
        upscale_factor=2,
        target_width=192,
        target_height=128,
    )
    manifest = CompetitionManifest.model_validate(
        {
            "competition_id": competition_id,
            "track": "upscaling",
            "start_time": T0 + timedelta(hours=1),
            "enrollment_deadline": T0 + timedelta(hours=2),
            "finalization_time": T0 + timedelta(hours=3),
            "end_time": T0 + timedelta(hours=4),
            "minimum_alpha_stake": 1.0,
            "scoring_factors": {
                "quality": 0.6,
                "cost_efficiency": 0.0,
                "length_coverage": 0.4,
            },
            "vmaf_threshold": 90.0,
            "sealed_vmaf_variants": [85.0, 89.0, 93.0],
            "allowed_gpus": ["L4"],
            "allowed_upscale_factors": [2, 4],
            "evaluation_item_commitments": [item_commitment],
            "evaluation_batch_size": {"min": 1, "max": 1},
            "scoring_seed_commitment": "a" * 64,
            "container_size_limit_gb": 25.0,
            "scoring_version": scorer_version,
        }
    )
    return manifest, item_commitment


def test_released_distinct_upscaling_pair_recomputes_from_public_store_on_cpu(
    tmp_path: Path,
) -> None:
    reference, miner_input, output = _media_triplet(tmp_path)
    reference_digest = sha256_file(reference)
    input_digest = sha256_file(miner_input)
    output_digest = sha256_file(output)
    assert len({reference_digest, input_digest, output_digest}) == 3

    scoring_config = ScoringConfig()
    worker_config = ScoringWorkerConfig(
        work_dir=tmp_path / "scorer-work",
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        pieapp_device="cpu",
        request_timeout=180.0,
        subprocess_timeout=60.0,
    )
    scoring_backends = real_backends(
        worker_config, scoring_config=scoring_config, pieapp_device="cpu"
    )
    if (
        scoring_backends.pieapp.version == "not-configured"
        or scoring_backends.perceptual.version == "not-configured"
    ):
        pytest.skip("external CPU PieAPP/OpenCV backend is not installed")
    try:
        scoring_backends.pieapp.ensure_ready()
    except NotConfiguredError as exc:
        pytest.skip(f"external CPU PieAPP weights are unavailable: {exc}")
    assert scoring_backends.pieapp.device == "cpu"

    scorer_version = effective_scorer_version(worker_config, scoring_config)
    manifest, item_commitment = _manifest(
        reference_digest=reference_digest,
        input_digest=input_digest,
        scorer_version=scorer_version,
    )
    challenge_id = "chal-upscale-public"
    miner_hotkey = "hk-upscale-public"
    request = ScoreRequest(
        track="upscaling",
        challenge_id=challenge_id,
        item_id=input_digest,
        miner_hotkey=miner_hotkey,
        reference_path=str(reference),
        reference_digest=reference_digest,
        miner_input_path=str(miner_input),
        miner_input_digest=input_digest,
        output_path=str(output),
        output_digest=output_digest,
        params={"upscale_factor": 2, "target_width": 192, "target_height": 128},
        scorer_version=scorer_version,
    )
    item = _score_sync(
        request,
        worker_config,
        scoring_config,
        scoring_backends,
        scorer_version,
    )
    assert item.gate_passed, item.violations
    assert item.score > 0.0

    private = LocalFsStore(tmp_path / "audit")
    input_ref = private.put_file(miner_input, ArtifactKind.CHALLENGE_INPUT)
    reference_ref = private.put_file(reference, ArtifactKind.REFERENCE_ORIGINAL)
    output_ref = private.put_file(output, ArtifactKind.MINER_OUTPUT)
    manifest_ref = private.put(
        manifest.canonical_json().encode("utf-8"), ArtifactKind.MANIFEST
    )
    packet_ref = private.put(
        item.to_json().encode("utf-8"), ArtifactKind.SCORE_PACKET
    )
    binding = CompetitionItemBinding(
        item_index=0,
        input_sha256=input_digest,
        reference_sha256=reference_digest,
        upscale_factor=2,
        target_width=192,
        target_height=128,
        item_commitment=item_commitment,
    )
    threshold_commitment = "f" * 64
    bundle = build_bundle(
        challenge_id=challenge_id,
        item_id=input_digest,
        miner_hotkey=miner_hotkey,
        commitment_hash=threshold_commitment,
        stage=LifecycleStage.COMPETITION_SEALED,
        challenge_input=input_ref,
        reference_original=reference_ref,
        miner_output=output_ref,
        manifest=manifest_ref,
        score_packet=packet_ref,
        competition_item=binding,
        scorer_version=scorer_version,
        backend_versions=dict(item.backend_versions),
        created_at="2026-09-01T04:00:00+00:00",
    )

    public = make_public_store(
        AuditConfig(backend="local", local_root=tmp_path / "audit")
    )
    with pytest.raises(FileNotFoundError):
        public.get(reference_ref)
    private.release(reference_ref)
    # This is the keyless contract: the public role resolves the exact committed
    # pristine bytes only after completion/release.
    assert public.get(reference_ref) == reference.read_bytes()
    assert public.get(input_ref) == miner_input.read_bytes()

    auditor_config = worker_config.model_copy(
        update={"work_dir": tmp_path / "auditor-work"}
    )
    auditor = RealScoreRecomputer.from_config(
        auditor_config,
        scoring_config=scoring_config,
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )
    assert auditor._backends.pieapp.device == "cpu"
    leaves = [packet_ref.digest]
    report = verify_bundle(
        bundle,
        public,
        auditor,
        expected_bundle_digest=bundle.bundle_digest(),
        expected_miner_hotkey=miner_hotkey,
        require_expected_miner=True,
        published_root=merkle_root(leaves),
        inclusion_proof=merkle_proof(leaves, packet_ref.digest),
        strict=True,
        competition_context=CompetitionAuditContext(
            competition_id=manifest.competition_id,
            track="upscaling",
            manifest_digest=manifest.manifest_digest(),
            threshold_commitment=threshold_commitment,
            item_index=0,
            input_sha256=input_digest,
            reference_sha256=reference_digest,
            upscale_factor=2,
            target_width=192,
            target_height=128,
            item_commitment=item_commitment,
        ),
    )
    assert report.passed, [
        (failure.name, failure.code, failure.reason) for failure in report.failures()
    ]
