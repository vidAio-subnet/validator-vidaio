#!/usr/bin/env python3
"""Fail-fast probe for the dependency set baked into the release image.

This is intentionally stronger than an import smoke test.  It verifies the exact
Bittensor SDK API surface the adapter calls, proves that the installed PyTorch
build is CPU-only, executes PIQ's PieAPP model over real decoded frames, and runs
one committed upscaling item through score -> post-retirement persistence ->
strict public-store audit recompute.  A validator must never discover a broken
runtime or missing weight file while auditing a live round.

Run during the pre-marker image build (with ``--preload-media``) and again in the
final container with both ``--require-runtime-manifest`` and
``--require-canonical-runtime``. It performs no chain RPCs and needs no wallet or
object-store secret.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any


EXACT_DISTRIBUTIONS = {
    "bittensor": "10.5.0",
    "async-substrate-interface": "2.2.1",
    "bittensor-drand": "2.0.0",
    "bittensor-wallet": "4.1.1",
    "modal": "1.5.4",
    "piq": "0.8.0",
    "torch": "2.8.0",
    "torchvision": "0.23.0",
}
ROOT = Path(__file__).resolve().parent.parent
CPU_UPSCALING_SMOKE_FPS = 4
CPU_UPSCALING_SMOKE_FRAME_COUNT = 40
CPU_UPSCALING_SMOKE_DURATION_SECONDS = 10.0


def _verify_launch_calibration_contract() -> dict[str, Any]:
    """Lock the code-side half of the real-media v7 calibration.

    The measured vectors live in the internal launch-calibration record.
    This release-image gate prevents an operator pool, factor, codec input, or
    shipped quality baseline from drifting away from those measured vectors
    without making the image qualification red.
    """
    import random

    from vidaio.challenge import (
        DAG_VERSION,
        LAUNCH_MAX_ELIGIBILITY_SCAN_ASSETS,
        LAUNCH_UPSCALING_MIN_CLIP_SECONDS,
        LAUNCH_UPSCALE_FACTORS,
        TRACK_RULES,
        UPSCALE_FACTORS,
        ChallengeConfig,
        build_dag,
    )
    from vidaio.miner.backends import FfmpegCompressBackend
    from vidaio.miner.config import MinerConfig
    from vidaio.miner.gpu_worker import _VARIANTS

    if DAG_VERSION != 7:
        raise RuntimeError(f"launch calibration is for DAG v7, got v{DAG_VERSION}")
    if LAUNCH_UPSCALE_FACTORS != (2,) or not set(LAUNCH_UPSCALE_FACTORS) < set(
        UPSCALE_FACTORS
    ):
        raise RuntimeError(
            "launch upscaling must mint only calibrated 2x while protocol support "
            f"retains future/historical factors: {LAUNCH_UPSCALE_FACTORS!r}, "
            f"{UPSCALE_FACTORS!r}"
        )
    challenge_cfg = ChallengeConfig()
    if challenge_cfg.upscaling_min_clip_seconds < LAUNCH_UPSCALING_MIN_CLIP_SECONDS:
        raise RuntimeError(
            "launch upscaling clip floor drifted below calibration: "
            f"{challenge_cfg.upscaling_min_clip_seconds:g}s < "
            f"{LAUNCH_UPSCALING_MIN_CLIP_SECONDS:g}s"
        )
    if challenge_cfg.max_eligibility_scan_assets != LAUNCH_MAX_ELIGIBILITY_SCAN_ASSETS:
        raise RuntimeError(
            "launch eligibility scan bound drifted from the certified corpus: "
            f"{challenge_cfg.max_eligibility_scan_assets} != "
            f"{LAUNCH_MAX_ELIGIBILITY_SCAN_ASSETS}"
        )
    expected_rules = {
        "compression": (("codec_compress",), ()),
        "upscaling": (("downscale",), ()),
    }
    actual_rules = {
        track: (rule.required, rule.optional) for track, rule in TRACK_RULES.items()
    }
    if actual_rules != expected_rules:
        raise RuntimeError(f"launch task pools drifted: {actual_rules!r}")
    for seed in range(256):
        compression = build_dag("compression", random.Random(seed))
        upscaling = build_dag("upscaling", random.Random(seed))
        if len(compression.ops) != 1 or compression.ops[0].op != "codec_compress":
            raise RuntimeError(f"seed {seed} left the codec-only launch pool")
        codec = compression.ops[0]
        if (
            codec.codec,
            codec.rate_mode,
            codec.crf,
            codec.gop,
            codec.chroma,
            codec.bit_depth,
        ) not in {
            ("h264", "crf", 8, 1, "420", 8),
            ("h264", "crf", 10, 1, "420", 8),
            ("h264", "crf", 12, 1, "420", 8),
        }:
            raise RuntimeError(f"seed {seed} drifted from calibrated codec input")
        if (
            len(upscaling.ops) != 1
            or upscaling.ops[0].op != "downscale"
            or round(1 / upscaling.ops[0].scale_factor) != 2
        ):
            raise RuntimeError(f"seed {seed} left the calibrated 2x launch pool")
    baselines = {
        "cpu_reference_crf": MinerConfig().compress_crf,
        "backend_default_crf": FfmpegCompressBackend().crf,
        "gpu_quality_crf": _VARIANTS["quality"].compression_crf,
    }
    if set(baselines.values()) != {22}:
        raise RuntimeError(f"shipped CRF-22 launch baseline drifted: {baselines!r}")
    quality_profile = (
        ROOT / "examples/competition_contenders/profiles/compression-quality.env"
    )
    if (
        "VIDAIO_NEXT_CRF=22"
        not in quality_profile.read_text(encoding="utf-8").splitlines()
    ):
        raise RuntimeError("competition quality example drifted from calibrated CRF 22")
    return {
        "dag_version": DAG_VERSION,
        "launch_upscale_factors": list(LAUNCH_UPSCALE_FACTORS),
        "max_eligibility_scan_assets": challenge_cfg.max_eligibility_scan_assets,
        "upscaling_min_clip_seconds": challenge_cfg.upscaling_min_clip_seconds,
        **baselines,
    }


def _base_version(version: str) -> str:
    """Strip a wheel local suffix such as ``+cpu`` before pin comparison."""
    return version.split("+", 1)[0]


def _require_parameter(callable_: Any, name: str) -> None:
    if not callable(callable_):
        raise RuntimeError(
            f"release SDK contract changed: required callable for {name!r} is absent"
        )
    try:
        parameters = inspect.signature(callable_).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "release SDK contract changed: cannot inspect "
            f"{getattr(callable_, '__qualname__', repr(callable_))}: {exc}"
        ) from exc
    if name not in parameters:
        raise RuntimeError(
            "release SDK contract changed: "
            f"{getattr(callable_, '__qualname__', repr(callable_))} has no {name!r} "
            f"parameter (found {list(parameters)})"
        )


def _verify_modal_contract() -> dict[str, Any]:
    """Verify the exact offline Modal surface used by the Sandbox adapter.

    Importing classes and inspecting signatures does not authenticate, resolve,
    list, create, or otherwise contact Modal. The release build runs this before
    it is allowed to ship so an SDK drift cannot first appear during a billable
    competition round.
    """
    import modal
    from modal.container_process import ContainerProcess
    from modal.sandbox_fs import _SandboxFilesystem
    from modal.stream_type import StreamType

    required_parameters: tuple[tuple[Any, tuple[str, ...]], ...] = (
        (modal.App, ("name", "tags")),
        (modal.App.run, ("name", "detach", "environment_name")),
        (
            modal.Image.from_dockerfile,
            ("path", "force_build", "context_dir", "secrets", "gpu"),
        ),
        (modal.Image.from_scratch, ("force_build",)),
        (modal.Image.debian_slim, ("python_version", "force_build")),
        (modal.Image.add_local_dir, ("local_path", "remote_path", "copy")),
        (modal.Image.entrypoint, ("entrypoint_commands",)),
        (modal.Image.cmd, ("cmd",)),
        (modal.Image.build, ("app",)),
        # Restart recovery rehydrates only a competition-owned immutable Image
        # id into a new App/runtime. Pin this SDK seam just as strictly as build
        # and Sandbox creation so drift fails the image build, not an earning run.
        # Modal 1.5.4 names this parameter ``image_id``. The runner passes the
        # persisted provider object id positionally, but the spelling is still a
        # useful pinned-SDK drift fence.
        (modal.Image.from_id, ("image_id",)),
        (
            modal.Sandbox.create,
            (
                "args",
                "app",
                "name",
                "tags",
                "image",
                "env",
                "secrets",
                "network_file_systems",
                "timeout",
                "idle_timeout",
                "gpu",
                "cpu",
                "memory",
                "block_network",
                "volumes",
                "encrypted_ports",
                "h2_ports",
                "unencrypted_ports",
                "include_oidc_identity_token",
            ),
        ),
        (modal.Sandbox.mount_image, ("path", "image")),
        (
            modal.Sandbox.exec,
            (
                "args",
                "stdout",
                "stderr",
                "timeout",
                "env",
                "secrets",
                "text",
                "bufsize",
            ),
        ),
        (modal.Sandbox.snapshot_directory, ("path", "timeout", "ttl")),
        (modal.Sandbox.terminate, ("wait",)),
        (_SandboxFilesystem.make_directory, ("remote_path",)),
        (_SandboxFilesystem.write_text, ("data", "remote_path")),
        (_SandboxFilesystem.list_files, ("remote_path",)),
        (_SandboxFilesystem.stat, ("remote_path",)),
        (_SandboxFilesystem.copy_to_local, ("remote_path", "local_path")),
    )
    for callable_, parameters in required_parameters:
        for parameter in parameters:
            _require_parameter(callable_, parameter)

    for owner, attribute in (
        (modal.Image, "object_id"),
        (modal.Sandbox, "object_id"),
        (modal.Sandbox, "filesystem"),
        (modal.Sandbox, "detach"),
        (ContainerProcess, "stdout"),
        (ContainerProcess, "stderr"),
        (ContainerProcess, "poll"),
        (modal.exception, "ImageBuildError"),
        (modal.exception, "SandboxFilesystemNotFoundError"),
        (StreamType, "PIPE"),
    ):
        if not hasattr(owner, attribute):
            raise RuntimeError(
                "release SDK contract changed: "
                f"{getattr(owner, '__name__', type(owner).__name__)}.{attribute} "
                "is absent"
            )

    return {
        "create_only_signature_check": True,
        "filesystem_signature_check": True,
        "image_restore_signature_check": True,
        "process_stream_signature_check": True,
    }


def _verify_git_executable() -> str:
    """Prove that the release can perform its bounded HTTPS Git checkout."""
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("release image is missing the required git executable")
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"release git executable is unusable: {type(exc).__name__}: {exc}"
        ) from exc
    version = completed.stdout.strip()
    if not version.startswith("git version ") or len(version) > 200:
        raise RuntimeError(f"release git returned an invalid version: {version!r}")
    return version


def _prove_cpu_pieapp_inference(pieapp: Any) -> float:
    """Decode real video frames and execute the packaged PieAPP model on CPU.

    The release image build already preloads the pinned weights.  A preload alone
    does not prove that OpenCV can decode media or that the installed PIQ/Torch
    combination can execute the network, so the offline container smoke runs one
    tiny, deterministic identity comparison end to end.
    """
    with tempfile.TemporaryDirectory(prefix="vidaio-pieapp-smoke-") as raw_root:
        clip = Path(raw_root) / "identity.y4m"
        argv = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x96:rate=4:duration=2",
            "-frames:v",
            "8",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "yuv4mpegpipe",
            "-y",
            str(clip),
        ]
        try:
            subprocess.run(argv, check=True, capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            stderr = getattr(exc, "stderr", b"")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"could not generate CPU PieAPP smoke clip: {type(exc).__name__}: "
                f"{str(stderr)[-1000:]}"
            ) from exc

        # Canonical scoring inputs are Y4M and upscaling offsets need not be
        # zero. Exercise both facts rather than merely proving an MP4 decoder.
        distance = float(pieapp.compute(str(clip), str(clip), start_frame=2))
        if not math.isfinite(distance) or abs(distance) > 1e-6:
            raise RuntimeError(
                "CPU PieAPP identity inference did not return zero distance: "
                f"{distance!r}"
            )
        return distance


def _run_smoke_ffmpeg(argv: list[str], *, label: str) -> None:
    """Run one deterministic media-fixture step with a useful fail-closed error."""
    try:
        subprocess.run(argv, check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        stderr = getattr(exc, "stderr", b"")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"CPU upscaling audit smoke could not {label}: {type(exc).__name__}: "
            f"{str(stderr)[-1000:]}"
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _prove_cpu_video_phash() -> dict[str, Any]:
    """Prove the locked ffmpeg runtime catches a real codec re-encode on CPU."""
    from vidaio.scoring import CPU_VIDEO_PHASH_VERSION, CpuVideoPhash
    from vidaio.scoring.phash_cpu import FFMPEG_PHASH_MAX_ALLOC_BYTES

    with tempfile.TemporaryDirectory(prefix="vidaio-phash-smoke-") as raw_root:
        root = Path(raw_root)
        first = root / "first.mp4"
        second = root / "second.mkv"
        common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
        _run_smoke_ffmpeg(
            [
                *common,
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x96:rate=8:duration=2",
                "-an",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "17",
                "-y",
                str(first),
            ],
            label="generate the CPU pHash source",
        )
        _run_smoke_ffmpeg(
            [
                *common,
                "-i",
                str(first),
                "-an",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "31",
                "-y",
                str(second),
            ],
            label="re-encode the CPU pHash source",
        )
        if _sha256_file(first) == _sha256_file(second):
            raise RuntimeError(
                "CPU pHash smoke re-encode was unexpectedly byte-identical"
            )
        backend = CpuVideoPhash(timeout_seconds=30.0)
        first_hash = backend.compute_phash(str(first))
        second_hash = backend.compute_phash(str(second))
        distance = backend.distance(first_hash, second_hash)
        if distance > 8:
            raise RuntimeError(
                "CPU video pHash failed to recognize the real re-encode: "
                f"distance={distance}, first={first_hash}, second={second_hash}"
            )
        return {
            "algorithm": CPU_VIDEO_PHASH_VERSION,
            "first": first_hash,
            "second": second_hash,
            "hamming_distance": distance,
            "near_duplicate_threshold": 8,
            "decoder_max_alloc_bytes": FFMPEG_PHASH_MAX_ALLOC_BYTES,
            "decoder_threads": 1,
            "filter_threads": 1,
        }


def _prove_cpu_upscaling_score_audit(
    pieapp: Any,
    *,
    allow_noncanonical_pre_marker_build_or_test_runtime: bool,
) -> dict[str, Any]:
    """Prove the release image can independently audit a real upscaling score.

    This exercises substantially more than a direct PieAPP call: the normal
    scorer canonicalizes and probes three encoded inputs, runs both libvmaf
    models plus the required CPU perceptual gates and CPU PieAPP, persists every
    audit artifact, releases the sealed reference, resolves the persisted bundle
    through a keyless/read-only store view, and strictly recomputes it through
    :class:`RealScoreRecomputer`.

    The scale factor is recovered from a real seed-derived challenge commitment,
    not selected independently for the recompute.  The strict audit also proves
    that reveal regeneration, packet identity, bundle digest, packet merkle
    inclusion, gate outcome, metrics, and final score all agree.
    """
    from vidaio.audit import (
        ArtifactKind,
        LifecycleStage,
        LocalFsStore,
        build_bundle,
        canonical_json_bytes,
        merkle_proof,
        merkle_root,
        verify_bundle,
    )
    from vidaio.auditor import RealScoreRecomputer, StoredBundleSource, persist_bundle
    from vidaio.challenge import (
        ChallengeCommitment,
        ChallengeConfig,
        MAX_CLIP_DURATION_OVERSHOOT_SECONDS,
        UPSCALE_FACTORS,
        build_dag,
        dag_rng_from_seed,
        deep_reveal_verifier,
    )
    from vidaio.epoch.log import AuditFileKind, AuditFileRef
    from vidaio.scoring import ScoringConfig
    from vidaio.scoring_worker import (
        ScoringWorkerConfig,
        effective_scorer_version,
        real_backends,
    )
    from vidaio.scoring_worker.service import _score_sync
    from vidaio.services.protocol import ScoreRequest
    from vidaio.tokenomics import TokenomicsConfig

    challenge_id = "release-cpu-upscaling-smoke"
    item_id = "release-cpu-upscaling-item"
    miner_hotkey = "release-cpu-smoke-miner"
    # A production-shaped 256-bit private seed with a stable, known 2x
    # downscale in DAG v7. Any RNG/DAG drift must break this release proof.
    private_seed = 1 << 255

    with tempfile.TemporaryDirectory(
        prefix="vidaio-upscaling-audit-smoke-"
    ) as raw_root:
        root = Path(raw_root)
        reference = root / "reference.y4m"
        miner_input = root / "miner-input.mkv"
        miner_output = root / "miner-output.mp4"
        ffmpeg = "ffmpeg"
        common = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
        # Build an attainable held-out reference from a deterministic synthetic
        # low-resolution signal. The previous two-second high-frequency testsrc was shorter
        # than the launch eligibility floor and Lanczos scored only ~0.02, so it
        # proved a real scorer path but not a *winnable* launch challenge.  This
        # fixture is ten seconds long and its high-resolution ground truth is the
        # deterministic 2x reconstruction of the same low-resolution signal.
        # The miner still receives only a separately encoded 48x48 input and its
        # output still traverses every real codec, VMAF, perceptual, PieAPP,
        # persistence, retirement and strict-audit path below.
        _run_smoke_ffmpeg(
            [
                *common,
                "-f",
                "lavfi",
                "-i",
                (
                    "testsrc2=size=48x48:"
                    f"rate={CPU_UPSCALING_SMOKE_FPS}:"
                    f"duration={CPU_UPSCALING_SMOKE_DURATION_SECONDS:g}"
                ),
                "-vf",
                "scale=96:96:flags=lanczos",
                "-frames:v",
                str(CPU_UPSCALING_SMOKE_FRAME_COUNT),
                "-pix_fmt",
                "yuv420p",
                "-f",
                "yuv4mpegpipe",
                "-y",
                str(reference),
            ],
            label="generate the held-out reference",
        )
        _run_smoke_ffmpeg(
            [
                *common,
                "-i",
                str(reference),
                "-vf",
                "scale=48:48:flags=lanczos",
                "-frames:v",
                str(CPU_UPSCALING_SMOKE_FRAME_COUNT),
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-y",
                str(miner_input),
            ],
            label="generate the committed 2x challenge input",
        )
        _run_smoke_ffmpeg(
            [
                *common,
                "-i",
                str(miner_input),
                "-vf",
                "scale=96:96:flags=lanczos",
                "-frames:v",
                str(CPU_UPSCALING_SMOKE_FRAME_COUNT),
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-y",
                str(miner_output),
            ],
            label="generate the miner's 2x output",
        )

        scoring_config = ScoringConfig()
        if (
            int(getattr(pieapp, "sample_window", -1))
            != scoring_config.pieapp_sample_window
        ):
            raise RuntimeError(
                "preloaded PieAPP sample window does not match the locked scoring config: "
                f"{getattr(pieapp, 'sample_window', None)!r} != "
                f"{scoring_config.pieapp_sample_window}"
            )
        worker_config = ScoringWorkerConfig(
            work_dir=root / "scoring-work",
            ffmpeg_path=ffmpeg,
            ffprobe_path="ffprobe",
            pieapp_device="cpu",
            request_timeout=120.0,
            subprocess_timeout=60.0,
        )
        backends = replace(
            real_backends(
                worker_config,
                scoring_config=scoring_config,
                pieapp_device="cpu",
            ),
            # Reuse the already-preloaded model. This is still an independent
            # second scoring invocation below; it merely avoids downloading or
            # loading the same immutable weights twice during the image build.
            pieapp=pieapp,
        )
        scorer_version = effective_scorer_version(worker_config, scoring_config)
        dag = build_dag("upscaling", dag_rng_from_seed(private_seed))
        commitment = ChallengeCommitment.create(
            "release-cpu-smoke-asset",
            dag,
            private_seed,
            scorer_version,
            "upscaling",
            dispatch_ordering_key=1,
        )
        downscales = [op for op in dag.ops if getattr(op, "op", "") == "downscale"]
        if len(downscales) != 1:
            raise RuntimeError(
                "CPU upscaling audit smoke commitment did not yield one downscale"
            )
        upscale_factor = round(1.0 / float(downscales[0].scale_factor))
        if upscale_factor != 2 or upscale_factor not in UPSCALE_FACTORS:
            raise RuntimeError(
                "CPU upscaling audit smoke seed/DAG drifted from its locked 2x factor: "
                f"{upscale_factor!r}"
            )

        request = ScoreRequest(
            track="upscaling",
            challenge_id=challenge_id,
            item_id=item_id,
            miner_hotkey=miner_hotkey,
            reference_path=str(reference),
            reference_digest=_sha256_file(reference),
            miner_input_path=str(miner_input),
            miner_input_digest=_sha256_file(miner_input),
            output_path=str(miner_output),
            output_digest=_sha256_file(miner_output),
            params={"upscale_factor": upscale_factor},
        )
        item = _score_sync(
            request,
            worker_config,
            scoring_config,
            backends,
            scorer_version,
        )
        minimum_payout_score = TokenomicsConfig().minimum_payout_score
        content_length = float(item.metrics.get("content_length", math.nan))
        challenge_config = ChallengeConfig()
        minimum_content_length = challenge_config.upscaling_min_clip_seconds
        maximum_content_length = (
            challenge_config.max_clip_seconds + MAX_CLIP_DURATION_OVERSHOOT_SECONDS
        )
        if (
            not item.gate_passed
            or item.violations
            or item.skips
            or item.score < minimum_payout_score
            or not math.isfinite(content_length)
            or content_length < minimum_content_length
            or content_length > maximum_content_length
            or not math.isfinite(float(item.metrics.get("pieapp", math.nan)))
            or item.pieapp_start_frame is None
            or not str(item.backend_versions.get("pieapp", "")).endswith(":cpu")
        ):
            raise RuntimeError(
                "real CPU upscaling score did not produce a payout-eligible packet: "
                f"gate_passed={item.gate_passed}, score={item.score}, "
                f"minimum_payout_score={minimum_payout_score}, "
                f"content_length={content_length}, "
                f"minimum_content_length={minimum_content_length}, "
                f"maximum_content_length={maximum_content_length}, "
                f"violations={[v.model_dump(mode='json') for v in item.violations]}, "
                f"skips={[s.model_dump(mode='json') for s in item.skips]}"
            )

        store_root = root / "audit-store"
        writer = LocalFsStore(store_root)
        challenge_input_ref = writer.put_file(miner_input, ArtifactKind.CHALLENGE_INPUT)
        miner_output_ref = writer.put_file(miner_output, ArtifactKind.MINER_OUTPUT)
        reference_ref = writer.put_file(reference, ArtifactKind.REFERENCE_ORIGINAL)
        manifest_ref = writer.put(
            canonical_json_bytes(
                {
                    "challenge_id": challenge_id,
                    "committed_upscale_factor": upscale_factor,
                    "dag_digest": dag.canonical_digest(),
                    "track": "upscaling",
                }
            ),
            ArtifactKind.MANIFEST,
        )
        packet_ref = writer.put(
            item.to_json().encode("utf-8"), ArtifactKind.SCORE_PACKET
        )
        reveal_ref = writer.put(commitment.preimage_bytes(), ArtifactKind.DAG_REVEAL)
        bundle = build_bundle(
            challenge_id=challenge_id,
            item_id=item_id,
            miner_hotkey=miner_hotkey,
            commitment_hash=commitment.commit_hash,
            stage=LifecycleStage.POST_RETIREMENT,
            challenge_input=challenge_input_ref,
            miner_output=miner_output_ref,
            manifest=manifest_ref,
            score_packet=packet_ref,
            reference_original=reference_ref,
            dag_reveal=reveal_ref,
            scorer_version=scorer_version,
            backend_versions=dict(item.backend_versions),
            created_at="2026-08-23T00:00:00+00:00",
        )
        persisted_ref = persist_bundle(writer, bundle)
        writer.release(reference_ref)

        # Model the anonymous/public post-retirement reader: no write access and
        # no sealed-object privilege. It can succeed only through the released
        # reference copy plus public content-addressed metadata/media.
        public_store = LocalFsStore(
            store_root,
            public_read_only=True,
            allow_sealed_operations=False,
        )
        resolved = StoredBundleSource(public_store).bundle_for(
            AuditFileRef(
                kind=AuditFileKind.AUDIT_BUNDLE,
                digest=persisted_ref.digest,
                challenge_id=challenge_id,
                item_id=item_id,
                source="inference",
            )
        )
        if resolved != bundle:
            raise RuntimeError(
                "persisted CPU upscaling audit bundle did not round-trip"
            )

        packet_root = merkle_root([packet_ref.digest])
        report = verify_bundle(
            resolved,
            public_store,
            RealScoreRecomputer(
                worker_config,
                backends,
                scoring_config=scoring_config,
                allow_noncanonical_pre_marker_build_or_test_runtime=(
                    allow_noncanonical_pre_marker_build_or_test_runtime
                ),
            ),
            expected_bundle_digest=persisted_ref.digest,
            expected_miner_hotkey=miner_hotkey,
            require_expected_miner=True,
            published_root=packet_root,
            inclusion_proof=merkle_proof([packet_ref.digest], packet_ref.digest),
            reveal_verifier=deep_reveal_verifier,
            strict=True,
        )
        if not report.passed:
            raise RuntimeError(
                "strict public-store CPU upscaling audit failed: "
                + json.dumps(
                    [failure.model_dump(mode="json") for failure in report.failures()],
                    sort_keys=True,
                )
            )
        passed_checks = {check.name for check in report.checks if check.passed}
        required_checks = {
            "bundle_digest",
            "commitment_reveal",
            "dag_reveal_generation",
            "merkle_inclusion",
            "score_recompute:chroma_uv_measure",
            "score_recompute:color_grayscale_measure",
            "score_recompute:pieapp",
            "score_recompute:score",
            "score_recompute:tone_manipulation_measure",
            "score_recompute:vmaf",
            "score_recompute:vmaf_model_delta",
            "gate_recompute",
        }
        missing = required_checks - passed_checks
        if missing:
            raise RuntimeError(
                "strict CPU upscaling audit omitted required checks: "
                + ", ".join(sorted(missing))
            )

        return {
            "bundle_digest": persisted_ref.digest,
            "commitment_hash": commitment.commit_hash,
            "content_length_seconds": content_length,
            "dag_digest": dag.canonical_digest(),
            "packet_digest": packet_ref.digest,
            "pieapp": float(item.metrics["pieapp"]),
            "pieapp_start_frame": item.pieapp_start_frame,
            "post_retirement_public_read": True,
            "score": item.score,
            "scorer_version": scorer_version,
            "strict_check_count": len(report.checks),
            "upscale_factor": upscale_factor,
            "upscale_factor_source": "committed_dag_reveal",
            "vmaf": float(item.metrics["vmaf"]),
        }


def verify(
    *,
    preload_media: bool,
    require_runtime_manifest: bool = False,
    require_canonical_runtime: bool = False,
) -> dict[str, Any]:
    launch_calibration = _verify_launch_calibration_contract()
    versions: dict[str, str] = {}
    for distribution, expected in EXACT_DISTRIBUTIONS.items():
        actual = importlib.metadata.version(distribution)
        if _base_version(actual) != expected:
            raise RuntimeError(
                f"{distribution} drifted: installed {actual}, release pin is {expected}"
            )
        versions[distribution] = actual

    modal_contract = _verify_modal_contract()
    git_version = _verify_git_executable()

    import bittensor as bt
    import boto3  # noqa: F401 - import is the storage-extra smoke test
    import cryptography  # noqa: F401 - AES-GCM envelope dependency
    import cv2  # noqa: F401 - CPU perceptual backend dependency
    import piq  # noqa: F401 - CPU PieAPP implementation
    import torch
    from async_substrate_interface import SubstrateInterface

    # These are adapter load-bearing, not optional conveniences. A signature
    # drift must turn the image build red instead of every live submit failing.
    for parameter in (
        "wallet",
        "netuid",
        "uids",
        "weights",
        "commit_reveal_version",
        "max_attempts",
        "version_key",
        "mev_protection",
        "raise_error",
        "wait_for_inclusion",
        "wait_for_finalization",
        "wait_for_revealed_execution",
    ):
        _require_parameter(bt.Subtensor.set_weights, parameter)
    for parameter in (
        "wallet",
        "netuid",
        "data",
        "mev_protection",
        "raise_error",
        "wait_for_inclusion",
        "wait_for_finalization",
        "wait_for_revealed_execution",
    ):
        _require_parameter(bt.Subtensor.set_commitment, parameter)
    for parameter in ("netuid", "block"):
        _require_parameter(bt.Subtensor.commit_reveal_enabled, parameter)
    _require_parameter(bt.Subtensor.tx_rate_limit, "block")
    _require_parameter(bt.Subtensor.get_commitment_metadata, "block")
    _require_parameter(bt.Subtensor, "fallback_endpoints")
    _require_parameter(bt.Subtensor, "archive_endpoints")
    # Runtime epochs are stateful in Bittensor 10.5: production derives exact
    # closes from historical SubnetEpochIndex transitions and verifies the
    # transition block's LastEpochBlock.  Missing block-pinned scheduler calls
    # must fail the image build, never first appear as a deployment-time HOLD.
    for method_name in ("get_epoch_schedule_state", "get_subnet_epoch_index"):
        method = getattr(bt.Subtensor, method_name, None)
        if not callable(method):
            raise RuntimeError(
                f"release SDK contract changed: Subtensor.{method_name} is absent"
            )
        _require_parameter(method, "netuid")
        _require_parameter(method, "block")

    # The auditor beacon and epoch finalizer use GRANDPA finality, never the
    # potentially reorgable best head. Commitment capacity additionally reads
    # MaxSpace + UsedSpaceOf through block-pinned raw storage. The pinned
    # async-substrate-interface 2.2.1 API calls the positional height argument
    # ``block_id``; checking these exact spellings catches an incompatible API
    # before deployment.
    finalized_head = getattr(SubstrateInterface, "get_chain_finalised_head", None)
    if not callable(finalized_head):
        raise RuntimeError(
            "release SDK contract changed: "
            "SubstrateInterface.get_chain_finalised_head is absent"
        )
    chain_head = getattr(SubstrateInterface, "get_chain_head", None)
    if not callable(chain_head):
        raise RuntimeError(
            "release SDK contract changed: SubstrateInterface.get_chain_head is absent"
        )
    block_number = getattr(SubstrateInterface, "get_block_number", None)
    if not callable(block_number):
        raise RuntimeError(
            "release SDK contract changed: SubstrateInterface.get_block_number is absent"
        )
    _require_parameter(block_number, "block_hash")
    block_hash = getattr(SubstrateInterface, "get_block_hash", None)
    if not callable(block_hash):
        raise RuntimeError(
            "release SDK contract changed: SubstrateInterface.get_block_hash is absent"
        )
    _require_parameter(block_hash, "block_id")
    storage_query = getattr(SubstrateInterface, "query", None)
    if not callable(storage_query):
        raise RuntimeError(
            "release SDK contract changed: SubstrateInterface.query is absent"
        )
    for parameter in ("module", "storage_function", "params", "block_hash"):
        _require_parameter(storage_query, parameter)
    storage_query_map = getattr(SubstrateInterface, "query_map", None)
    if not callable(storage_query_map):
        raise RuntimeError(
            "release SDK contract changed: SubstrateInterface.query_map is absent"
        )
    for parameter in (
        "module",
        "storage_function",
        "params",
        "block_hash",
        "page_size",
        "ignore_decoding_errors",
    ):
        _require_parameter(storage_query_map, parameter)
    # Validators discover miners exclusively through the registered Axon IP/port.
    # The release image therefore has to support the one-shot advertisement
    # helper even though the miner serves vidaio's HTTP protocol rather than a
    # Bittensor Synapse.
    for parameter in (
        "netuid",
        "axon",
        "mev_protection",
        "raise_error",
        "wait_for_inclusion",
        "wait_for_finalization",
        "wait_for_revealed_execution",
    ):
        _require_parameter(bt.Subtensor.serve_axon, parameter)
    for parameter in ("hotkey_ss58", "netuid", "block"):
        _require_parameter(bt.Subtensor.get_neuron_for_pubkey_and_subnet, parameter)
    for parameter in ("wallet", "port", "external_ip", "external_port"):
        _require_parameter(bt.Axon, parameter)
    _require_parameter(bt.Subtensor.get_timestamp, "block")

    if torch.version.cuda is not None:
        raise RuntimeError(
            "release image contains a CUDA-bearing PyTorch wheel; auditors must be CPU-only "
            f"(torch.version.cuda={torch.version.cuda!r})"
        )
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only release unexpectedly reports CUDA as available")

    # The Docker dependency smoke runs before the immutable manifest and image
    # marker exist, so it explicitly leaves this false.  Every qualification,
    # preflight, and shipped-image CI call sets it true. Initialize before any
    # PieAPP model is constructed or preloaded, then attest the complete runtime.
    canonical_runtime_digest: str | None = None
    if require_canonical_runtime:
        from vidaio.scoring import ScoringConfig
        from vidaio.scoring_worker import ScoringWorkerConfig
        from vidaio.scoring_worker.runtime_identity import (
            initialize_canonical_torch_cpu_runtime,
            payout_runtime_attestation,
            require_canonical_release_runtime,
            runtime_commitment_digest,
        )

        initialize_canonical_torch_cpu_runtime()
        canonical_attestation = payout_runtime_attestation(
            ScoringWorkerConfig(pieapp_device="cpu"), ScoringConfig()
        )
        require_canonical_release_runtime(canonical_attestation)
        canonical_runtime_digest = runtime_commitment_digest(canonical_attestation)

    from vidaio.chain.bittensor_adapter import (
        BittensorAdapterConfig,
        BittensorChainAdapter,
    )
    from vidaio.chain.factory import ChainConfig, assert_bittensor_adapter_contract
    from vidaio.epoch import EPOCH_LOG_SCHEMA_VERSION
    from vidaio.scoring.backends_real import (
        CpuPerceptualCheckBackend,
        PieAppTorchBackend,
    )
    from vidaio.weightsetter.config import WeightSetterConfig

    assert_bittensor_adapter_contract(BittensorChainAdapter)
    version_fences = {
        "chain": ChainConfig().version_key,
        "bittensor_adapter": BittensorAdapterConfig(
            validator_hotkey="release-probe"
        ).version_key,
        "weightsetter": WeightSetterConfig().version_key,
    }
    mismatched_fences = {
        name: value
        for name, value in version_fences.items()
        if value != EPOCH_LOG_SCHEMA_VERSION
    }
    if mismatched_fences:
        raise RuntimeError(
            "release version fence(s) do not match the epoch-log schema: "
            f"{mismatched_fences!r} != {EPOCH_LOG_SCHEMA_VERSION}"
        )

    perceptual = CpuPerceptualCheckBackend()
    if perceptual.version == "not-configured":
        raise RuntimeError("deterministic CPU perceptual backend is not configured")

    pieapp = PieAppTorchBackend(device="cpu")
    if pieapp.version == "not-configured":
        raise RuntimeError("CPU PieAPP backend is not configured")
    pieapp_cpu_inference: float | None = None
    cpu_upscaling_score_audit: dict[str, Any] | None = None
    cpu_video_phash: dict[str, Any] | None = None
    if preload_media:
        pieapp.preload()
        pieapp_cpu_inference = _prove_cpu_pieapp_inference(pieapp)
        cpu_video_phash = _prove_cpu_video_phash()
        cpu_upscaling_score_audit = _prove_cpu_upscaling_score_audit(
            pieapp,
            allow_noncanonical_pre_marker_build_or_test_runtime=(
                not require_canonical_runtime
            ),
        )

    from vidaio.autoupdater.integrity import verify_runtime_manifest

    runtime_manifest_path = ROOT / "runtime-release-manifest.json"
    runtime_identity = None
    if runtime_manifest_path.is_file():
        runtime_identity = verify_runtime_manifest(
            runtime_manifest_path,
            runtime_root=ROOT,
            source_root=ROOT,
        )
    elif require_runtime_manifest:
        raise RuntimeError(
            f"release runtime manifest is missing: {runtime_manifest_path}"
        )

    return {
        "status": "ok",
        "cpu_only_auditing": True,
        "cpu_upscaling_score_audit": cpu_upscaling_score_audit is not None,
        "cpu_upscaling_score_audit_proof": cpu_upscaling_score_audit,
        "cpu_video_phash_proof": cpu_video_phash,
        "epoch_log_schema": EPOCH_LOG_SCHEMA_VERSION,
        "git": git_version,
        "launch_calibration_contract": launch_calibration,
        "grandpa_finality_contract": True,
        "commit_reveal_contract": True,
        "commitment_capacity_contract": True,
        "commitment_rate_limit_contract": True,
        "canonical_payout_runtime_sha256": canonical_runtime_digest,
        "canonical_payout_runtime_verified": canonical_runtime_digest is not None,
        "pieapp": pieapp.version,
        "pieapp_cpu_identity_distance": pieapp_cpu_inference,
        "pieapp_cpu_inference": pieapp_cpu_inference is not None,
        "pieapp_preloaded": preload_media,
        "modal_contract": modal_contract,
        "perceptual": perceptual.version,
        "runtime_manifest_file_count": (
            runtime_identity.file_count if runtime_identity is not None else None
        ),
        "runtime_manifest_sha256": (
            runtime_identity.runtime_sha256 if runtime_identity is not None else None
        ),
        "runtime_manifest_verified": runtime_identity is not None,
        "stateful_epoch_schedule_contract": True,
        "serve_axon_contract": True,
        "version_fences": version_fences,
        "versions": versions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preload-media",
        action="store_true",
        help=(
            "cache pinned PieAPP weights, run offline CPU inference, and prove a "
            "real upscaling score survives strict public-store CPU audit recompute"
        ),
    )
    parser.add_argument(
        "--require-runtime-manifest",
        action="store_true",
        help="fail unless the immutable shipped-runtime manifest exists and verifies",
    )
    parser.add_argument(
        "--require-canonical-runtime",
        action="store_true",
        help=(
            "fail unless the complete marker-qualified Linux/amd64 CPU scoring "
            "runtime is canonical"
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                preload_media=args.preload_media,
                require_runtime_manifest=args.require_runtime_manifest,
                require_canonical_runtime=args.require_canonical_runtime,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
