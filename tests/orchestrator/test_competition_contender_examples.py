"""CPU/static contract checks for generated GPU competition contenders."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from examples.competition_contenders.materialize import materialize

ROOT = Path(__file__).resolve().parents[2] / "examples" / "competition_contenders"


def _profile(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def test_profiles_cover_both_tracks_with_distinct_economic_outputs() -> None:
    profiles = {
        path.stem: _profile(path) for path in sorted((ROOT / "profiles").glob("*.env"))
    }
    assert set(profiles) == {
        "compression-quality",
        "compression-balanced",
        "compression-compact",
        "compression-baseline",
        "upscaling-quality",
        "upscaling-balanced",
        "upscaling-compact",
        "upscaling-baseline",
    }
    for track in ("compression", "upscaling"):
        selected = [
            value
            for value in profiles.values()
            if value["VIDAIO_NEXT_TRACK"] == track
            and value["VIDAIO_NEXT_VARIANT"] != "baseline"
        ]
        assert len(selected) == 3
        assert len(
            {
                (
                    item["VIDAIO_NEXT_INTERPOLATION"],
                    item["VIDAIO_NEXT_SHARPEN"],
                    item["VIDAIO_NEXT_CRF"],
                    item["VIDAIO_NEXT_PRESET"],
                )
                for item in selected
            }
        ) == 3
    assert {
        value["VIDAIO_NEXT_SCALE"] for value in profiles.values()
        if value["VIDAIO_NEXT_TRACK"] == "compression"
    } == {"1"}
    assert {
        value["VIDAIO_NEXT_SCALE"] for value in profiles.values()
        if value["VIDAIO_NEXT_TRACK"] == "upscaling"
    } == {"committed"}


def test_template_is_digest_pinned_cuda_execution_not_a_gpu_only_scorer() -> None:
    dockerfile = (ROOT / "template" / "Dockerfile").read_text(encoding="utf-8")
    run = (ROOT / "template" / "run.sh").read_text(encoding="utf-8")
    cuda = (ROOT / "template" / "gpu_transform.cu").read_text(encoding="utf-8")
    assert dockerfile.count("@sha256:") == 2
    assert "nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:" in dockerfile
    assert "nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:" in dockerfile
    assert '"$gpu_filter" --probe' in run
    assert '"$nvidia_smi_bin"' in run
    assert "cudaMalloc" in cuda
    assert "transform_kernel<<<blocks, threads>>>" in cuda
    assert "cudaDeviceSynchronize" in cuda
    assert "int output_width" in cuda and "int output_height" in cuda
    assert "width / output_width" in cuda and "height / output_height" in cuda
    assert "pieapp" not in (dockerfile + run + cuda).lower()
    assert "vmaf" not in (dockerfile + run + cuda).lower()


def test_materializer_creates_standalone_tree_and_refuses_reuse(tmp_path: Path) -> None:
    destination = tmp_path / "vidaio-next-upscaling-quality"
    materialize(track="upscaling", variant="quality", destination=destination)
    assert {path.name for path in destination.iterdir()} == {
        "Dockerfile",
        "run.sh",
        "gpu_transform.cu",
        "variant.env",
    }
    assert os.access(destination / "run.sh", os.X_OK)
    assert _profile(destination / "variant.env")["VIDAIO_NEXT_VARIANT"] == "quality"
    with pytest.raises(FileExistsError, match="refusing to reuse/overwrite"):
        materialize(track="upscaling", variant="compact", destination=destination)


@pytest.mark.parametrize("track", ["compression", "upscaling"])
def test_materializer_provides_one_non_earning_baseline_tree_per_track(
    tmp_path: Path, track: str
) -> None:
    destination = tmp_path / f"vidaio-next-{track}-baseline"
    materialize(track=track, variant="baseline", destination=destination)
    profile = _profile(destination / "variant.env")
    assert profile["VIDAIO_NEXT_TRACK"] == track
    assert profile["VIDAIO_NEXT_VARIANT"] == "baseline"
    assert profile["VIDAIO_NEXT_SCALE"] == (
        "1" if track == "compression" else "committed"
    )


def _fake_tools(tmp_path: Path) -> tuple[dict[str, str], Path]:
    tools = tmp_path / "fake-tools"
    tools.mkdir()
    log = tmp_path / "calls.log"
    nvidia = tools / "nvidia-smi"
    nvidia.write_text("#!/bin/sh\necho nvidia-smi >>\"$VIDAIO_TEST_LOG\"\n")
    probe = tools / "ffprobe"
    probe.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *width,height*) echo 2x2 ;;\n"
        "  *avg_frame_rate*) echo 24/1 ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n"
    )
    gpu = tools / "gpu-transform"
    gpu.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = --probe ]; then\n"
        "  echo gpu-probe >>\"$VIDAIO_TEST_LOG\"\n"
        "  exit 0\n"
        "fi\n"
        "echo gpu-transform \"$*\" >>\"$VIDAIO_TEST_LOG\"\n"
        "cat\n"
    )
    ffmpeg = tools / "ffmpeg"
    ffmpeg.write_text(
        "#!/bin/sh\n"
        "input=\n"
        "overwrite=0\n"
        "previous=\n"
        "last=\n"
        "for argument in \"$@\"; do\n"
        "  [ \"$previous\" = -i ] && input=$argument\n"
        "  [ \"$argument\" = -y ] && overwrite=1\n"
        "  previous=$argument\n"
        "  last=$argument\n"
        "done\n"
        "echo ffmpeg \"$*\" >>\"$VIDAIO_TEST_LOG\"\n"
        "[ -n \"$input\" ] && [ -n \"$last\" ]\n"
        "[ ! -e \"$last\" ] || [ \"$overwrite\" -eq 1 ] || exit 1\n"
        "cat \"$input\" >\"$last\"\n"
    )
    for executable in (nvidia, probe, gpu, ffmpeg):
        executable.chmod(0o755)
    return (
        {
            "VIDAIO_NEXT_NVIDIA_SMI_BIN": str(nvidia),
            "VIDAIO_NEXT_FFPROBE_BIN": str(probe),
            "VIDAIO_NEXT_GPU_FILTER_BIN": str(gpu),
            "VIDAIO_NEXT_FFMPEG_BIN": str(ffmpeg),
            "VIDAIO_TEST_LOG": str(log),
        },
        log,
    )


@pytest.mark.parametrize(
    ("track", "variant", "upscale_factor", "gpu_args", "encode_args"),
    [
        (
            "compression",
            "balanced",
            None,
            "2 2 2 2 bilinear 0.06",
            "-crf 28 -pix_fmt",
        ),
        ("upscaling", "compact", 4, "2 2 8 8 nearest 0.00", "-crf 30 -pix_fmt"),
    ],
)
def test_generated_run_contract_with_cpu_fakes(
    tmp_path: Path,
    track: str,
    variant: str,
    upscale_factor: int | None,
    gpu_args: str,
    encode_args: str,
) -> None:
    solution = materialize(
        track=track,
        variant=variant,
        destination=tmp_path / f"vidaio-next-{track}-{variant}",
    )
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    payload = b"one-rgb-frame"  # fake tools test binding/lifecycle, not media quality
    digest = hashlib.sha256(payload).hexdigest()
    (inputs / digest).write_bytes(payload)
    if upscale_factor is not None:
        (inputs / f".vidaio-next-upscale-task-{digest}").write_text(
            '{"target_height":8,"target_width":8,'
            f'"upscale_factor":{upscale_factor}}}\n',
            encoding="ascii",
        )
    fake_env, log = _fake_tools(tmp_path)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "VIDAIO_NEXT_APP_DIR": str(solution),
        **fake_env,
    }

    result = subprocess.run(
        ["/bin/sh", str(solution / "run.sh"), str(inputs), str(outputs)],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (outputs / digest).read_bytes() == payload
    calls = log.read_text(encoding="utf-8")
    assert "gpu-probe" in calls
    assert f"gpu-transform {gpu_args}" in calls
    assert encode_args in calls
    assert f"track={track} variant={variant}" in result.stderr


def test_upscaling_template_consumes_mixed_committed_factors(tmp_path: Path) -> None:
    solution = materialize(
        track="upscaling",
        variant="balanced",
        destination=tmp_path / "vidaio-next-upscaling-balanced",
    )
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    expected: dict[str, bytes] = {}
    for payload, factor in ((b"mixed-two", 2), (b"mixed-four", 4)):
        digest = hashlib.sha256(payload).hexdigest()
        (inputs / digest).write_bytes(payload)
        target = 2 * factor
        (inputs / f".vidaio-next-upscale-task-{digest}").write_text(
            f'{{"target_height":{target},"target_width":{target},'
            f'"upscale_factor":{factor}}}\n', encoding="ascii"
        )
        expected[digest] = payload
    fake_env, log = _fake_tools(tmp_path)
    result = subprocess.run(
        ["/bin/sh", str(solution / "run.sh"), str(inputs), str(outputs)],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "VIDAIO_NEXT_APP_DIR": str(solution),
            **fake_env,
        },
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert {path.name: path.read_bytes() for path in outputs.iterdir()} == expected
    calls = log.read_text(encoding="utf-8")
    assert "gpu-transform 2 2 4 4 bilinear 0.08" in calls
    assert "gpu-transform 2 2 8 8 bilinear 0.08" in calls


@pytest.mark.parametrize(
    "sidecar",
    [
        None,
        b'{"target_height":4,"target_width":4,"upscale_factor":3}\n',
        b'{"target_height":4,"target_width":4,"upscale_factor":2}\nextra',
    ],
)
def test_upscaling_template_refuses_missing_or_noncanonical_factor(
    tmp_path: Path, sidecar: bytes | None
) -> None:
    solution = materialize(
        track="upscaling",
        variant="quality",
        destination=tmp_path / "vidaio-next-upscaling-quality",
    )
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    payload = b"factor-contract"
    digest = hashlib.sha256(payload).hexdigest()
    (inputs / digest).write_bytes(payload)
    if sidecar is not None:
        (inputs / f".vidaio-next-upscale-task-{digest}").write_bytes(sidecar)
    fake_env, _log = _fake_tools(tmp_path)

    result = subprocess.run(
        ["/bin/sh", str(solution / "run.sh"), str(inputs), str(outputs)],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "VIDAIO_NEXT_APP_DIR": str(solution),
            **fake_env,
        },
        timeout=10,
    )

    assert result.returncode != 0
    assert not list(outputs.iterdir())
