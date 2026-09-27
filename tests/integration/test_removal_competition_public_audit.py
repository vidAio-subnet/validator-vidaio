"""A released object-removal competition score is reproducible by a keyless CPU auditor."""

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
from vidaio.auditor.recomputer import RealScoreRecomputer
from vidaio.competition import CompetitionManifest, removal_item_commitment
from vidaio.scoring import ScoringConfig
from vidaio.scoring_worker import ScoringWorkerConfig, effective_scorer_version, real_backends
from vidaio.scoring_worker.service import _score_sync
from vidaio.services.protocol import ScoreRequest

pytestmark = requires_media_tools

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
SIZE = "192x128"
# the occluder: a 48x32 box sliding right, gone for a while, then back
AT = "x='if(between(t,0.75,1.25),-100,20+40*t)':y=40:eval=frame"


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", *args],
        check=True,
        capture_output=True,
        timeout=120,
    )


def removal_media(root: Path) -> tuple[Path, Path, Path]:
    """Clean reference, the two-stream served input (frames with the object + its
    mask) and a genuine repair (the clean frames, delivered as a normal H.264 file)."""
    reference = root / "reference.mkv"
    served = root / "input.mkv"
    output = root / "output.mp4"
    src = f"testsrc2=size={SIZE}:rate=8:duration=2"
    _ffmpeg("-f", "lavfi", "-i", src, "-pix_fmt", "yuv420p", "-c:v", "ffv1", "-y", str(reference))
    _ffmpeg(
        "-i", str(reference),
        "-f", "lavfi", "-i", f"color=black:size={SIZE}:rate=8:duration=2",
        "-f", "lavfi", "-i", "color=red:size=48x32:rate=8:duration=2",
        "-f", "lavfi", "-i", "color=white:size=48x32:rate=8:duration=2",
        "-filter_complex",
        f"[0:v][2:v]overlay={AT},format=yuv420p[v];"
        f"[1:v][3:v]overlay={AT},format=gray[m]",
        "-map", "[v]", "-map", "[m]",
        "-c:v", "ffv1", "-y", str(served),
    )
    _ffmpeg(
        "-i", str(reference), "-c:v", "libx264", "-crf", "8", "-g", "1",
        "-pix_fmt", "yuv420p", "-y", str(output),
    )
    return reference, served, output


def _manifest(*, commitment: str, scorer_version: str) -> CompetitionManifest:
    return CompetitionManifest.model_validate(
        {
            "competition_id": "comp-removal-public-audit",
            "track": "removal",
            "start_time": T0 + timedelta(hours=1),
            "enrollment_deadline": T0 + timedelta(hours=2),
            "finalization_time": T0 + timedelta(hours=3),
            "end_time": T0 + timedelta(hours=4),
            "minimum_alpha_stake": 1.0,
            "scoring_factors": {"quality": 1.0, "cost_efficiency": 0.0, "length_coverage": 0.0},
            "vmaf_threshold": 90.0,
            "sealed_vmaf_variants": [90.0],
            "allowed_gpus": ["L4"],
            "evaluation_item_commitments": [commitment],
            "evaluation_batch_size": {"min": 1, "max": 1},
            "scoring_seed_commitment": "a" * 64,
            "container_size_limit_gb": 25.0,
            "scoring_version": scorer_version,
        }
    )


def test_removal_manifest_requires_precommitted_items() -> None:
    with pytest.raises(ValueError, match="removal manifest requires"):
        _manifest(commitment="1" * 64, scorer_version="s").model_copy(
            update={"evaluation_item_commitments": None}
        ).model_validate(
            {
                **_manifest(commitment="1" * 64, scorer_version="s").model_dump(),
                "evaluation_item_commitments": None,
            }
        )


def test_released_removal_pair_recomputes_from_public_store_on_cpu(tmp_path: Path) -> None:
    reference, served, output = removal_media(tmp_path)
    reference_digest = sha256_file(reference)
    input_digest = sha256_file(served)
    output_digest = sha256_file(output)
    assert len({reference_digest, input_digest, output_digest}) == 3

    scoring_config = ScoringConfig()
    worker_config = ScoringWorkerConfig(
        work_dir=tmp_path / "scorer-work",
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        pieapp_device="cpu",
        request_timeout=300.0,
        subprocess_timeout=120.0,
    )
    scoring_backends = real_backends(
        worker_config, scoring_config=scoring_config, pieapp_device="cpu"
    )
    if scoring_backends.removal is None:
        pytest.skip("removal metrics backend is not installed")
    scorer_version = effective_scorer_version(worker_config, scoring_config)
    competition_id = "comp-removal-public-audit"
    commitment = removal_item_commitment(
        competition_id=competition_id,
        item_index=0,
        input_sha256=input_digest,
        reference_sha256=reference_digest,
    )
    manifest = _manifest(commitment=commitment, scorer_version=scorer_version)
    challenge_id = "chal-removal-public"
    miner_hotkey = "hk-removal-public"
    request = ScoreRequest(
        track="removal",
        challenge_id=challenge_id,
        item_id=input_digest,
        miner_hotkey=miner_hotkey,
        reference_path=str(reference),
        reference_digest=reference_digest,
        miner_input_path=str(served),
        miner_input_digest=input_digest,
        output_path=str(output),
        output_digest=output_digest,
        params={"mask_stream_index": 1},
        scorer_version=scorer_version,
    )
    item = _score_sync(request, worker_config, scoring_config, scoring_backends, scorer_version)
    assert item.gate_passed, item.violations
    assert item.score > 0.5
    assert item.metrics["mask_frames"] > 0

    private = LocalFsStore(tmp_path / "audit")
    input_ref = private.put_file(served, ArtifactKind.CHALLENGE_INPUT)
    reference_ref = private.put_file(reference, ArtifactKind.REFERENCE_ORIGINAL)
    output_ref = private.put_file(output, ArtifactKind.MINER_OUTPUT)
    manifest_ref = private.put(manifest.canonical_json().encode("utf-8"), ArtifactKind.MANIFEST)
    packet_ref = private.put(item.to_json().encode("utf-8"), ArtifactKind.SCORE_PACKET)
    binding = CompetitionItemBinding(
        item_index=0,
        input_sha256=input_digest,
        reference_sha256=reference_digest,
        mask_stream_index=1,
        item_commitment=commitment,
    )
    # upscaling keys never appear in a removal binding's canonical bytes
    assert "upscale_factor" not in binding.model_dump(mode="json")
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

    public = make_public_store(AuditConfig(backend="local", local_root=tmp_path / "audit"))
    with pytest.raises(FileNotFoundError):
        public.get(reference_ref)
    private.release(reference_ref)
    assert public.get(reference_ref) == reference.read_bytes()

    auditor = RealScoreRecomputer.from_config(
        worker_config.model_copy(update={"work_dir": tmp_path / "auditor-work"}),
        scoring_config=scoring_config,
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )
    leaves = [packet_ref.digest]
    context = CompetitionAuditContext(
        competition_id=competition_id,
        track="removal",
        manifest_digest=manifest.manifest_digest(),
        threshold_commitment=threshold_commitment,
        item_index=0,
        input_sha256=input_digest,
        reference_sha256=reference_digest,
        item_commitment=commitment,
        mask_stream_index=1,
    )
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
        competition_context=context,
    )
    assert report.passed, [(f.name, f.code, f.reason) for f in report.failures()]

    # a context naming another reference must fail the manifest commitment check
    forged = CompetitionAuditContext(
        **{**context.__dict__, "reference_sha256": "e" * 64}  # type: ignore[arg-type]
    ) if hasattr(context, "__dict__") else None
    if forged is None:
        from dataclasses import replace

        forged = replace(context, reference_sha256="e" * 64)
    forged_report = verify_bundle(
        bundle,
        public,
        auditor,
        expected_bundle_digest=bundle.bundle_digest(),
        expected_miner_hotkey=miner_hotkey,
        require_expected_miner=True,
        published_root=merkle_root(leaves),
        inclusion_proof=merkle_proof(leaves, packet_ref.digest),
        strict=True,
        competition_context=forged,
    )
    assert not forged_report.passed
    assert any(f.code == "COMPETITION_MANIFEST_INVALID" for f in forged_report.failures())
