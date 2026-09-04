"""Payout-runtime identity is a scorer/auditor compatibility boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from vidaio.auditor import RealScoreRecomputer
from vidaio.scoring import DeterministicFakeBackend, ScoringConfig
from vidaio.scoring.backends_real import PIEAPP_WEIGHTS_SHA256
from vidaio.scoring_worker import (
    ScoringBackends,
    ScoringWorker,
    ScoringWorkerConfig,
    effective_scorer_version,
)
from vidaio.scoring_worker.runtime_identity import (
    CANONICAL_RUNTIME_MARKER_BYTES,
    canonical_runtime_problems,
    require_canonical_release_runtime,
    runtime_backend_stamp,
    runtime_commitment_digest,
)


def _attestation(*, os_name: str = "linux", arch: str = "amd64") -> dict:
    return {
        "schema": "vidaio-payout-runtime/1",
        "release": {
            "manifest_verified": True,
            "marker_verified": True,
            "manifest_sha256": "a" * 64,
            "marker_sha256": "b" * 64,
            "release_version": "0.3.1",
            "source_sha256": "c" * 64,
            "runtime_sha256": "d" * 64,
        },
        "execution_policy": {
            "required_os": "linux",
            "required_arch": "amd64",
            "actual_os": os_name,
            "actual_arch": arch,
            "libc": "glibc/2.36",
            "torch_intraop_threads": "1",
            "torch_interop_threads": "1",
            "mkl_threads": "1",
            "openblas_threads": "1",
            "omp_dynamic": "FALSE",
            "mkl_dynamic": "FALSE",
            "mkl_cbwr": "COMPATIBLE",
            "actual_torch_intraop_threads": 1,
            "actual_torch_interop_threads": 1,
            "actual_torch_deterministic_algorithms": True,
            "actual_torch_deterministic_warn_only": False,
            "actual_torch_mkldnn_enabled": False,
            "actual_torch_mkldnn_deterministic": True,
            "actual_torch_nnpack_enabled": False,
            "actual_torch_cpu_capability": "NO AVX",
            "actual_openmp_threads": 1,
            "actual_mkl_threads": 1,
            "actual_mkl_cbwr": "COMPATIBLE",
            "actual_mkl_dynamic": False,
            "aten_cpu_capability_override": "default",
        },
        "payout_backends": {
            "ffmpeg": "ffmpeg/9.0",
            "ffprobe": "ffprobe/9.0",
            "libvmaf": "libvmaf/3.0.0",
            "pieapp": "pieapp-torch/piq/0.8.0:pieapp:cpu",
            "perceptual": "cpu-perceptual-checks/opencv/4.12.0:algorithm/2",
            "pieapp_weights": f"sha256:{PIEAPP_WEIGHTS_SHA256}",
            "torch": "torch/2.8.0+cpu",
            "torchvision": "torchvision/0.23.0+cpu",
            "piq": "piq/0.8.0",
            "opencv": "opencv/4.12.0.88",
            "numpy": "numpy/2.2.6",
            "python": "cpython/3.13.15",
        },
    }


def _fake_backends(runtime_attestation: dict) -> ScoringBackends:
    fake = DeterministicFakeBackend()
    fake.device = "cpu"
    versions = dict(runtime_attestation["payout_backends"])
    versions["runtime"] = runtime_backend_stamp(runtime_attestation)
    return ScoringBackends(
        probe=fake,
        vmaf_primary=fake,
        vmaf_secondary=fake,
        pieapp=fake,
        perceptual=fake,
        canonicalizer=None,
        versions=versions,
        runtime_attestation=runtime_attestation,
    )


def test_effective_identity_moves_with_complete_runtime_commitment() -> None:
    worker = ScoringWorkerConfig(backend="fake")
    scoring = ScoringConfig()
    release = _attestation()
    native = _attestation(os_name="darwin", arch="arm64")

    release_identity = effective_scorer_version(
        worker, scoring, runtime_attestation=release
    )
    native_identity = effective_scorer_version(
        worker, scoring, runtime_attestation=native
    )

    assert release_identity != native_identity
    assert runtime_commitment_digest(release) != runtime_commitment_digest(native)
    assert runtime_backend_stamp(release).endswith(runtime_commitment_digest(release))


def test_backend_version_drift_moves_identity() -> None:
    worker = ScoringWorkerConfig(backend="fake")
    scoring = ScoringConfig()
    expected = _attestation()
    drifted = _attestation()
    drifted["payout_backends"]["torch"] = "torch/2.9.0+cpu"

    assert effective_scorer_version(
        worker, scoring, runtime_attestation=expected
    ) != effective_scorer_version(worker, scoring, runtime_attestation=drifted)


def test_canonical_runtime_policy_accepts_only_complete_release() -> None:
    qualified = _attestation()
    assert canonical_runtime_problems(qualified) == []
    require_canonical_release_runtime(qualified)

    native = _attestation(os_name="darwin", arch="arm64")
    with pytest.raises(RuntimeError, match="OS is 'darwin'.*architecture is 'arm64'"):
        require_canonical_release_runtime(native)

    missing_backend = _attestation()
    del missing_backend["payout_backends"]["libvmaf"]
    with pytest.raises(RuntimeError, match="versions are missing: libvmaf"):
        require_canonical_release_runtime(missing_backend)

    wrong_contract = _attestation()
    wrong_contract["schema"] = "vidaio-payout-runtime/0"
    wrong_contract["execution_policy"]["required_arch"] = "arm64"
    with pytest.raises(RuntimeError, match="schema is .*required_arch is 'arm64'"):
        require_canonical_release_runtime(wrong_contract)

    gpu_metric = _attestation()
    gpu_metric["payout_backends"]["pieapp"] = "pieapp-torch/piq/0.8.0:pieapp:cuda"
    gpu_metric["payout_backends"]["torch"] = "torch/2.8.0+cu128"
    with pytest.raises(RuntimeError, match="PieAPP.*':cpu'.*locked CPU wheel"):
        require_canonical_release_runtime(gpu_metric)

    adaptive_kernels = _attestation()
    adaptive_kernels["execution_policy"].update(
        {
            "actual_torch_cpu_capability": "AVX2",
            "actual_torch_interop_threads": 8,
            "actual_torch_mkldnn_enabled": True,
            "actual_torch_nnpack_enabled": True,
            "actual_torch_deterministic_algorithms": False,
            "actual_mkl_cbwr": "AUTO",
        }
    )
    with pytest.raises(
        RuntimeError,
        match="actual_torch_interop_threads.*actual_mkl_cbwr",
    ):
        require_canonical_release_runtime(adaptive_kernels)


def test_real_chain_worker_constructor_refuses_noncanonical_runtime(tmp_path) -> None:
    native = _attestation(os_name="darwin", arch="arm64")
    raw = {
        "chain": {"mode": "bittensor"},
        "scoring_worker": {
            "backend": "fake",
            "work_dir": str(tmp_path / "worker"),
            "port": 0,
            "metrics_port": 0,
        },
    }

    with pytest.raises(RuntimeError, match="canonical payout runtime required"):
        ScoringWorker(raw, backends=_fake_backends(native))

    # Supplying release-looking metadata cannot turn this native process into a
    # canonical scorer; the constructor independently probes the local runtime.
    with pytest.raises(RuntimeError, match="canonical payout runtime required"):
        ScoringWorker(raw, backends=_fake_backends(_attestation()))


def test_real_chain_worker_binds_attestation_to_cpu_backend_map(
    tmp_path, monkeypatch
) -> None:
    qualified = _attestation()
    monkeypatch.setattr(
        "vidaio.scoring_worker.service.payout_runtime_attestation",
        lambda config, scoring_config: qualified,
    )
    raw = {
        "chain": {"mode": "bittensor"},
        "scoring_worker": {
            "backend": "fake",
            "work_dir": str(tmp_path / "worker"),
            "port": 0,
            "metrics_port": 0,
        },
    }

    moved = _fake_backends(qualified)
    moved.versions["torch"] = "torch/2.9.0+cpu"
    with pytest.raises(RuntimeError, match=r"moved=\['torch'\]"):
        ScoringWorker(raw, backends=moved)

    cuda = _fake_backends(qualified)
    cuda.pieapp.device = "cuda"
    with pytest.raises(RuntimeError, match="PieAPP backend must execute on CPU"):
        ScoringWorker(raw, backends=cuda)


def test_recomputer_is_strict_by_default_with_narrow_build_test_opt_out(
    tmp_path,
) -> None:
    config = ScoringWorkerConfig(
        backend="fake", work_dir=tmp_path / "audit-work", metrics_port=0
    )
    native = _attestation(os_name="darwin", arch="arm64")
    with pytest.raises(RuntimeError, match="canonical payout runtime required"):
        RealScoreRecomputer(config, _fake_backends(native))

    with pytest.raises(RuntimeError, match="canonical payout runtime required"):
        RealScoreRecomputer(config, _fake_backends(_attestation()))

    build_test_only = RealScoreRecomputer(
        config,
        _fake_backends(native),
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )
    assert build_test_only.scorer_version


def test_strict_recomputer_binds_attestation_to_cpu_backend_map(
    tmp_path, monkeypatch
) -> None:
    qualified = _attestation()
    monkeypatch.setattr(
        "vidaio.auditor.recomputer.payout_runtime_attestation",
        lambda config, scoring_config: qualified,
    )
    config = ScoringWorkerConfig(
        backend="fake", work_dir=tmp_path / "audit-work", metrics_port=0
    )

    moved = _fake_backends(qualified)
    moved.versions["torch"] = "torch/2.9.0+cpu"
    with pytest.raises(RuntimeError, match=r"moved=\['torch'\]"):
        RealScoreRecomputer(config, moved)

    cuda = _fake_backends(qualified)
    cuda.pieapp.device = "cuda"
    with pytest.raises(RuntimeError, match="PieAPP backend must execute on CPU"):
        RealScoreRecomputer(config, cuda)


def test_dockerfile_builds_the_exact_image_only_runtime_marker() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    for line in CANONICAL_RUNTIME_MARKER_BYTES.decode("ascii").splitlines():
        assert f"'{line}'" in dockerfile
