r"""Opt-in full-pipeline PieAPP RSS and packet comparison; never collected by pytest.

Run from the repository root, with the media extras and pinned weights already
installed. Each mode scores the SAME generated inputs in a fresh process. Fixture
generation, input hashing, and result collection happen in the parent process.
All media and scoring scratch live in a TemporaryDirectory and are removed.

Small native smoke (explicitly selects production-like Torch kernel settings,
without creating or claiming the qualified Linux release identity)::

    .venv/bin/python -m tests.scoring_worker.profile_upscaling_memory \
        --width 160 --height 120 --frames 8 --mode compare \
        --cpu-policy canonical-kernels

On the qualified release runtime, use its existing CPU policy and verify its
attestation. Run these sequentially; the legacy mode intentionally reproduces
the large allocation and needs a host with sufficient available memory::

    .venv/bin/python -m tests.scoring_worker.profile_upscaling_memory \
        --width 1920 --height 1080 --frames 360 --mode compare \
        --cpu-policy runtime --require-canonical-runtime
    .venv/bin/python -m tests.scoring_worker.profile_upscaling_memory \
        --width 2560 --height 1440 --frames 360 --mode compare \
        --cpu-policy runtime --require-canonical-runtime

JSON is printed to stdout, so evidence can be redirected to a file under /tmp.
TMPDIR can select a disk-backed temporary volume for the canonical Y4M files.
RSS includes native Torch/OpenCV allocations; tracemalloc would miss them.
Self and child high-water RSS are reported separately. Their sum is a conservative
bound, NOT a simultaneously sampled process-tree peak (scoring launches media
subprocesses sequentially). Darwin reports ru_maxrss in bytes, Linux in KiB.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch


_MODULE = "tests.scoring_worker.profile_upscaling_memory"


@dataclass(frozen=True)
class MediaFixture:
    reference: str
    reference_digest: str
    miner_input: str
    miner_input_digest: str
    candidate: str
    candidate_digest: str
    width: int
    height: int
    frames: int
    fps: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_fixture(
    root: Path,
    *,
    width: int,
    height: int,
    frames: int = 360,
    fps: int = 30,
    ffmpeg: str = "ffmpeg",
    timeout: float = 1800.0,
) -> MediaFixture:
    """Generate a held-out reference, half-size input, and Lanczos-upscaled output."""
    if min(width, height) < 64 or width % 4 or height % 4:
        raise ValueError("width and height must be multiples of four and at least 64")
    if frames < 1 or fps < 1:
        raise ValueError("frames and fps must be positive")
    root.mkdir(parents=True, exist_ok=True)
    reference = root / "reference.mp4"
    miner_input = root / "miner-input.mp4"
    candidate = root / "candidate.mp4"

    def encode(inputs: list[str], output: Path, *, scale: str | None = None) -> None:
        argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", *inputs]
        if scale is not None:
            argv.extend(["-vf", scale])
        argv.extend([
            "-frames:v", str(frames), "-an", "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-threads", "1", "-y", str(output),
        ])
        subprocess.run(argv, check=True, capture_output=True, text=True, timeout=timeout)

    encode(["-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps}"], reference)
    encode(["-i", str(reference)], miner_input,
           scale=f"scale={width // 2}:{height // 2}:flags=lanczos")
    encode(["-i", str(miner_input)], candidate,
           scale=f"scale={width}:{height}:flags=lanczos")
    return MediaFixture(
        str(reference), _sha256_file(reference),
        str(miner_input), _sha256_file(miner_input),
        str(candidate), _sha256_file(candidate), width, height, frames, fps,
    )


def _peak_rss(who: int) -> int:
    value = int(resource.getrusage(who).ru_maxrss)
    if sys.platform == "darwin":
        return value
    if sys.platform.startswith("linux"):
        return value * 1024
    raise RuntimeError("RSS units are supported only on Darwin and Linux")


def _apply_explicit_kernel_profile(torch: Any) -> None:
    """Diagnostic kernel settings only; release verification remains untouched."""
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.mkldnn.enabled = False
    torch.backends.mkldnn.deterministic = True
    torch.backends.nnpack.set_flags(False)


def _torch_report(torch: Any) -> dict[str, Any]:
    from vidaio.scoring_worker.runtime_identity import _CANONICAL_CPU_ENV

    get_nnpack = getattr(torch._C, "_get_nnpack_enabled", None)
    return {
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_config": torch.__config__.show(),
        "torch_parallel_info": torch.__config__.parallel_info(),
        "environment": {key: os.environ.get(key) for key in _CANONICAL_CPU_ENV},
        "actual_policy": {
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
            "nnpack_enabled": get_nnpack() if get_nnpack is not None else "unavailable",
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        },
    }


def score_fixture(
    fixture: MediaFixture,
    work_dir: Path,
    *,
    mode: str = "current",
    sample_window: int = 4,
    cpu_policy: str = "runtime",
    require_canonical_runtime: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    timeout: float = 1800.0,
    scope: Any = None,
) -> dict[str, Any]:
    """Score with real backends; call in a fresh process for meaningful RSS values.

    ``legacy`` bypasses only the new model wrapper. The original PIQ forward,
    decoder, backend constructor, canonicalization, metrics, and packet composer
    execute unchanged. Optional media imports stay inside this opt-in function.
    """
    if mode not in {"legacy", "current"}:
        raise ValueError("mode must be legacy or current")
    if cpu_policy not in {"runtime", "canonical-kernels"}:
        raise ValueError("unknown CPU policy")
    if not 1 <= sample_window <= fixture.frames:
        raise ValueError("sample window must be between one and the fixture frame count")
    started = time.monotonic()
    import torch

    from vidaio.scoring import ScoringConfig
    from vidaio.scoring import backends_real
    from vidaio.scoring_worker import ScoringWorkerConfig
    from vidaio.scoring_worker.runtime_identity import require_canonical_release_runtime
    from vidaio.scoring_worker.service import _score_sync, effective_scorer_version, real_backends
    from vidaio.services.protocol import ScoreRequest

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / backends_real.PIEAPP_WEIGHTS_FILENAME
    if not checkpoint.is_file():
        raise RuntimeError(f"offline profiler requires pre-cached PieAPP weights: {checkpoint}")
    backends_real._verify_pieapp_weights(checkpoint)
    if cpu_policy == "canonical-kernels":
        _apply_explicit_kernel_profile(torch)

    scoring = ScoringConfig(pieapp_sample_window=sample_window)
    config = ScoringWorkerConfig(
        work_dir=work_dir,
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        pieapp_device="cpu",
        max_concurrent=1,
        request_timeout=timeout,
        subprocess_timeout=timeout,
        max_request_scratch_bytes=32 * 1024**3,
        max_scratch_bytes=32 * 1024**3,
    )
    request = ScoreRequest(
        track="upscaling", challenge_id="memory-profile-challenge",
        item_id="memory-profile-item", miner_hotkey="memory-profile-miner",
        reference_path=fixture.reference, reference_digest=fixture.reference_digest,
        miner_input_path=fixture.miner_input, miner_input_digest=fixture.miner_input_digest,
        output_path=fixture.candidate, output_digest=fixture.candidate_digest,
        params={"upscale_factor": 2},
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            torch.hub, "download_url_to_file",
            side_effect=RuntimeError("network downloads are disabled by the offline profiler"),
        ))
        if mode == "legacy":
            stack.enter_context(patch.object(
                backends_real, "_bounded_pieapp_model", lambda model, torch: model,
            ))
        backends = real_backends(config, scoring_config=scoring)
        if require_canonical_runtime:
            require_canonical_release_runtime(backends.runtime_attestation)
        scorer_version = effective_scorer_version(
            config, scoring, runtime_attestation=backends.runtime_attestation,
        )
        score_started = time.monotonic()
        item = _score_sync(request, config, scoring, backends, scorer_version, scope=scope)
        score_seconds = time.monotonic() - score_started

    packet = item.to_json()
    runtime = _torch_report(torch)
    runtime["attestation"] = backends.runtime_attestation
    model_type = type(backends.pieapp._runtime._metric.model)
    runtime["pieapp_model_class"] = f"{model_type.__module__}.{model_type.__qualname__}"
    self_peak = _peak_rss(resource.RUSAGE_SELF)
    child_peak = _peak_rss(resource.RUSAGE_CHILDREN)
    return {
        "mode": mode,
        "width": fixture.width,
        "height": fixture.height,
        "frames": fixture.frames,
        "sample_window": sample_window,
        "cpu_policy": cpu_policy,
        "canonical_runtime_required": require_canonical_runtime,
        "elapsed_seconds": time.monotonic() - started,
        "score_seconds": score_seconds,
        "self_peak_rss_bytes": self_peak,
        "child_peak_rss_bytes": child_peak,
        "conservative_sum_peak_rss_bytes": self_peak + child_peak,
        "item_score_json": packet,
        "packet_digest": hashlib.sha256(packet.encode("utf-8")).hexdigest(),
        "runtime": runtime,
    }


def _child_main() -> int:
    payload = json.load(sys.stdin)
    from vidaio.scoring.backends_real import MediaProcessScope

    scope = MediaProcessScope()

    def cancel(signum: int, _frame: Any) -> None:
        scope.cancel()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    result = score_fixture(
        MediaFixture(**payload["fixture"]), Path(payload["work_dir"]),
        scope=scope, **payload["options"],
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


def _profile_child(
    fixture: MediaFixture, work_dir: Path, options: dict[str, Any],
) -> dict[str, Any]:
    env = os.environ.copy()
    if options["cpu_policy"] == "canonical-kernels":
        from vidaio.scoring_worker.runtime_identity import _CANONICAL_CPU_ENV

        # These kernel switches take effect at import; set them in the fresh
        # child's environment rather than changing the already-running parent.
        env.update(_CANONICAL_CPU_ENV)
    payload = {"fixture": asdict(fixture), "work_dir": str(work_dir), "options": options}
    with subprocess.Popen(
        [sys.executable, "-m", _MODULE, "--_child"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(json.dumps(payload), timeout=options["timeout"])
        except subprocess.TimeoutExpired:
            proc.terminate()  # child handler cancels registered media process groups
            try:
                proc.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
            raise RuntimeError(f"{options['mode']} profile exceeded {options['timeout']} seconds") from None
    if proc.returncode:
        raise RuntimeError(f"{options['mode']} profile failed ({proc.returncode}):\n{stderr}")
    return json.loads(stdout)


def main(argv: list[str] | None = None) -> int:
    if argv is None and sys.argv[1:] == ["--_child"]:
        return _child_main()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sample-window", type=int, default=4)
    parser.add_argument("--mode", choices=("compare", "legacy", "current"), default="compare")
    parser.add_argument("--cpu-policy", choices=("runtime", "canonical-kernels"), default="runtime")
    parser.add_argument("--require-canonical-runtime", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="timeout for each fixture encode and each measured child")
    args = parser.parse_args(argv)
    if not 1 <= args.sample_window <= args.frames:
        parser.error("sample-window must be between one and frames")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    ffmpeg, ffprobe = shutil.which(args.ffmpeg), shutil.which(args.ffprobe)
    if ffmpeg is None or ffprobe is None:
        parser.error("ffmpeg and ffprobe must be installed")
    with TemporaryDirectory(prefix="vidaio-memory-profile-") as temporary:
        root = Path(temporary)
        fixture = generate_fixture(
            root / "inputs", width=args.width, height=args.height,
            frames=args.frames, fps=args.fps, ffmpeg=ffmpeg, timeout=args.timeout,
        )
        modes = ("legacy", "current") if args.mode == "compare" else (args.mode,)
        runs = []
        for mode in modes:
            options = {
                "mode": mode, "sample_window": args.sample_window,
                "cpu_policy": args.cpu_policy,
                "require_canonical_runtime": args.require_canonical_runtime,
                "ffmpeg": ffmpeg, "ffprobe": ffprobe, "timeout": args.timeout,
            }
            runs.append(_profile_child(fixture, root / mode, options))
    result = {
        "fixture": {
            key: value for key, value in asdict(fixture).items()
            if key not in {"reference", "miner_input", "candidate"}
        },
        "runs": runs,
    }
    if len(runs) == 2:
        identical = runs[0]["item_score_json"] == runs[1]["item_score_json"]
        result["packets_identical"] = identical
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if identical else 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
