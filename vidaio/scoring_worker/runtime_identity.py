"""Canonical payout-runtime commitment for scorer/auditor compatibility.

The scoring configuration alone is not a complete scorer identity.  PieAPP,
libvmaf and the canonical media decode are also functions of the executable
runtime: OS/architecture, locked Python wheels and native media binaries.  A
macOS/arm64 developer process and the Linux/amd64 release image must therefore
never advertise the same scorer identity even when they load the same YAML.

This module builds one canonical commitment over those runtime inputs.  Real
workers put the full commitment digest in ``backend_versions["runtime"]`` and
also fold the complete payload into their effective scorer identity.  Auditors
construct the commitment independently, so a different runtime fails the
existing strict backend/scorer compatibility checks before a score can be
mistaken for a recomputable one.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from vidaio.audit.canonical import canonical_json_bytes
from vidaio.autoupdater.integrity import runtime_digest, verify_runtime_manifest
from vidaio.scoring.backends_real import (
    PIEAPP_WEIGHTS_SHA256,
    CpuPerceptualCheckBackend,
    FfmpegVmafBackend,
    PieAppTorchBackend,
    detect_tool_versions,
)

RUNTIME_COMMITMENT_SCHEMA = "vidaio-payout-runtime/1"
RUNTIME_BACKEND_KEY = "runtime"
RUNTIME_BACKEND_PREFIX = "vidaio-payout-runtime/1+"

# Created only by the digest-pinned release Dockerfile.  A checkout can possess
# the same source/runtime manifest, but it cannot accidentally claim to be the
# qualified image runtime: the marker, platform policy and manifest verification
# below all have to agree.
CANONICAL_RUNTIME_MARKER_PATH = ".vidaio-release-runtime"
CANONICAL_RUNTIME_MARKER_BYTES = (
    b"vidaio-release-runtime/1\nos=linux\narch=amd64\n"
    b"aten_cpu_capability=default\ntorch_intraop_threads=1\n"
    b"torch_interop_threads=1\ntorch_deterministic_algorithms=error\n"
    b"torch_mkldnn=disabled\ntorch_nnpack=disabled\nmkl_cbwr=COMPATIBLE\n"
)

_CANONICAL_CPU_ENV = {
    "ATEN_CPU_CAPABILITY": "default",
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "MKL_CBWR": "COMPATIBLE",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_CANONICAL_TORCH_CPU_CAPABILITY = "NO AVX"
_TORCH_POLICY_LOCK = threading.Lock()

_REQUIRED_PAYOUT_BACKENDS = frozenset(
    {
        "ffmpeg",
        "ffprobe",
        "libvmaf",
        "pieapp",
        "perceptual",
        "pieapp_weights",
        "torch",
        "torchvision",
        "piq",
        "opencv",
        "numpy",
        "python",
    }
)


def _runtime_root() -> Path:
    # Source checkout and the editable release image both put ``vidaio/`` two
    # levels below the runtime root (repo root or /app respectively).
    return Path(__file__).resolve().parents[2]


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not-configured"


def complete_payout_backend_versions(
    base: Mapping[str, str],
    *,
    pieapp: PieAppTorchBackend,
    perceptual: CpuPerceptualCheckBackend,
    device: str,
) -> dict[str, str]:
    """Return every executable/package/model version that affects payout.

    ``base`` is the native ffmpeg/ffprobe/libvmaf probe.  The extra entries are
    intentionally present in every real score packet, not merely hidden inside
    a short scorer suffix, so an auditor can name the exact incompatible input.
    """

    versions = {str(name): str(version) for name, version in base.items()}
    versions.update(
        {
            "pieapp": f"{pieapp.name}/{pieapp.version}:{device}",
            "perceptual": f"{perceptual.name}/{perceptual.version}",
            "pieapp_weights": f"sha256:{PIEAPP_WEIGHTS_SHA256}",
            "torch": f"torch/{_distribution_version('torch')}",
            "torchvision": f"torchvision/{_distribution_version('torchvision')}",
            "piq": f"piq/{_distribution_version('piq')}",
            "opencv": (
                "opencv/"
                + _distribution_version("opencv-python-headless", "opencv-python")
            ),
            "numpy": f"numpy/{_distribution_version('numpy')}",
            "python": (
                f"{platform.python_implementation().lower()}/"
                f"{platform.python_version()}"
            ),
        }
    )
    return versions


def probe_payout_backend_versions(
    config: Any, scoring_config: Any, *, pieapp_device: str | None = None
) -> dict[str, str]:
    """Probe the same complete version set that :func:`real_backends` stamps.

    This path exists for configuration owners that must derive the worker's
    identity before starting its HTTP service.  Failure is represented as an
    explicitly unconfigured runtime; an actual real worker still raises its
    typed startup error while a preflight can explain why qualification failed.
    """

    if getattr(config, "backend", "real") != "real":
        return {"backend": "deterministic-injected-fake"}
    try:
        primary = FfmpegVmafBackend(
            config.ffmpeg_path,
            model=config.vmaf_model_primary,
            # Use the system temporary directory: deriving an identity must not
            # mutate or require ownership of a service role's scoring volume.
            work_dir=None,
            timeout=config.subprocess_timeout,
        )
        versions = detect_tool_versions(
            config.ffmpeg_path,
            config.ffprobe_path,
            vmaf_backend=primary,
            timeout=config.subprocess_timeout,
        )
    except Exception:  # noqa: BLE001 - becomes an honest unqualified identity
        versions = {
            "ffmpeg": "ffmpeg/not-configured",
            "ffprobe": "ffprobe/not-configured",
            "libvmaf": "libvmaf/not-configured",
        }
    device = pieapp_device or config.pieapp_device
    pieapp = PieAppTorchBackend(
        device=device, sample_window=scoring_config.pieapp_sample_window
    )
    perceptual = CpuPerceptualCheckBackend(config.perceptual_cpu)
    return complete_payout_backend_versions(
        versions, pieapp=pieapp, perceptual=perceptual, device=device
    )


def _normal_arch(machine: str) -> str:
    value = machine.strip().lower()
    if value in {"x86_64", "amd64"}:
        return "amd64"
    if value in {"aarch64", "arm64"}:
        return "arm64"
    return value or "unknown"


def _effective_torch_policy(torch: Any) -> dict[str, Any]:
    parallel = str(torch.__config__.parallel_info())

    def _parallel_int(pattern: str) -> int | str:
        match = re.search(pattern, parallel)
        return int(match.group(1)) if match else "unavailable"

    return {
        "actual_torch_intraop_threads": int(torch.get_num_threads()),
        "actual_torch_interop_threads": int(torch.get_num_interop_threads()),
        "actual_torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "actual_torch_deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "actual_torch_mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "actual_torch_mkldnn_deterministic": bool(
            torch.backends.mkldnn.deterministic
        ),
        # ``set_flags`` returns the previous state. Initialization has already
        # disabled NNPACK; setting False again both verifies and preserves it.
        "actual_torch_nnpack_enabled": bool(
            torch.backends.nnpack.set_flags(False)[0]
        ),
        "actual_torch_cpu_capability": str(
            torch.backends.cpu.get_cpu_capability()
        ),
        "actual_openmp_threads": _parallel_int(r"omp_get_max_threads\(\)\s*:\s*(\d+)"),
        "actual_mkl_threads": _parallel_int(r"mkl_get_max_threads\(\)\s*:\s*(\d+)"),
    }


def _torch_policy_problems(policy: Mapping[str, Any]) -> list[str]:
    expected: dict[str, Any] = {
        "actual_torch_intraop_threads": 1,
        "actual_torch_interop_threads": 1,
        "actual_torch_deterministic_algorithms": True,
        "actual_torch_deterministic_warn_only": False,
        "actual_torch_mkldnn_enabled": False,
        "actual_torch_mkldnn_deterministic": True,
        "actual_torch_nnpack_enabled": False,
        "actual_torch_cpu_capability": _CANONICAL_TORCH_CPU_CAPABILITY,
        "actual_openmp_threads": 1,
        "actual_mkl_threads": 1,
        "actual_mkl_cbwr": "COMPATIBLE",
        "actual_mkl_dynamic": False,
    }
    return [
        f"{name} is {policy.get(name)!r}, expected {wanted!r}"
        for name, wanted in expected.items()
        if policy.get(name) != wanted
    ]


@lru_cache(maxsize=1)
def _isolated_torch_policy_probe() -> dict[str, Any]:
    """Probe the locked wheel, including oneMKL's effective CNR branch.

    ``MKL_CBWR`` is not useful as an attestation if it is merely copied from the
    environment.  oneMKL's verbose record for a real SGEMM names the selected
    CNR branch, dynamic-thread state and thread count.  The bounded child avoids
    retaining Torch in lightweight roles that only derive scorer identity.
    """

    script = """
import json
import torch
torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.mkldnn.enabled = False
torch.backends.mkldnn.deterministic = True
torch.backends.nnpack.set_flags(False)
parallel = torch.__config__.parallel_info()
x = torch.ones((32, 32), dtype=torch.float32)
_ = x @ x
print(json.dumps({
    "actual_torch_intraop_threads": torch.get_num_threads(),
    "actual_torch_interop_threads": torch.get_num_interop_threads(),
    "actual_torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "actual_torch_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    "actual_torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
    "actual_torch_mkldnn_deterministic": torch.backends.mkldnn.deterministic,
    "actual_torch_nnpack_enabled": torch.backends.nnpack.set_flags(False)[0],
    "actual_torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
    "parallel_info": parallel,
}, sort_keys=True))
"""
    env = dict(os.environ)
    env["MKL_VERBOSE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        json_line = next(
            line for line in reversed(result.stdout.splitlines()) if line.startswith("{")
        )
        policy = json.loads(json_line)
        parallel = str(policy.pop("parallel_info"))
        omp = re.search(r"omp_get_max_threads\(\)\s*:\s*(\d+)", parallel)
        mkl = re.search(r"mkl_get_max_threads\(\)\s*:\s*(\d+)", parallel)
        cbwr = re.search(r"\bCNR:([A-Z0-9_]+)\b", result.stdout)
        dynamic = re.search(r"\bDyn:([01])\b", result.stdout)
        policy.update(
            {
                "actual_openmp_threads": int(omp.group(1)) if omp else "unavailable",
                "actual_mkl_threads": int(mkl.group(1)) if mkl else "unavailable",
                "actual_mkl_cbwr": cbwr.group(1) if cbwr else "unavailable",
                "actual_mkl_dynamic": (
                    dynamic.group(1) == "1" if dynamic else "unavailable"
                ),
            }
        )
        return policy
    except (OSError, StopIteration, ValueError, subprocess.SubprocessError):
        return {
            name: "unavailable"
            for name in (
                "actual_torch_intraop_threads",
                "actual_torch_interop_threads",
                "actual_torch_deterministic_algorithms",
                "actual_torch_deterministic_warn_only",
                "actual_torch_mkldnn_enabled",
                "actual_torch_mkldnn_deterministic",
                "actual_torch_nnpack_enabled",
                "actual_torch_cpu_capability",
                "actual_openmp_threads",
                "actual_mkl_threads",
                "actual_mkl_cbwr",
                "actual_mkl_dynamic",
            )
        }


def initialize_canonical_torch_cpu_runtime() -> dict[str, Any]:
    """Force, inspect and fail closed on the canonical CPU metric policy."""

    wrong_env = [
        f"{name}={os.environ.get(name)!r}, expected {wanted!r}"
        for name, wanted in _CANONICAL_CPU_ENV.items()
        if os.environ.get(name) not in (None, wanted)
    ]
    if wrong_env:
        raise RuntimeError("non-canonical CPU environment: " + "; ".join(wrong_env))
    for name, wanted in _CANONICAL_CPU_ENV.items():
        os.environ[name] = wanted

    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"canonical CPU PyTorch is unavailable: {exc}") from exc

    with _TORCH_POLICY_LOCK:
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.mkldnn.enabled = False
        torch.backends.mkldnn.deterministic = True
        torch.backends.nnpack.set_flags(False)
        effective = _effective_torch_policy(torch)
        # The isolated SGEMM is the public wheel's only inspectable proof that
        # oneMKL honored CBWR rather than silently selecting AUTO.
        isolated = _isolated_torch_policy_probe()
        effective["actual_mkl_cbwr"] = isolated["actual_mkl_cbwr"]
        effective["actual_mkl_dynamic"] = isolated["actual_mkl_dynamic"]
    problems = _torch_policy_problems(effective)
    if problems:
        raise RuntimeError("canonical CPU kernel policy unavailable: " + "; ".join(problems))
    return effective


@lru_cache(maxsize=1)
def _release_identity() -> dict[str, Any]:
    root = _runtime_root()
    manifest_path = root / "runtime-release-manifest.json"
    marker_path = root / CANONICAL_RUNTIME_MARKER_PATH
    try:
        marker_bytes = marker_path.read_bytes()
    except OSError:
        marker_bytes = b""
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError:
        manifest_bytes = b""

    verified = None
    try:
        verified = verify_runtime_manifest(
            manifest_path,
            runtime_root=root,
            allow_ignored_caches=True,
        )
    except (OSError, ValueError):
        pass

    recorded: dict[str, Any] = {}
    if manifest_bytes:
        try:
            parsed = json.loads(manifest_bytes)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            for key in ("version", "source_sha256", "runtime_sha256"):
                value = parsed.get(key)
                if isinstance(value, str):
                    recorded[key] = value

    current_runtime = "unavailable"
    try:
        current_runtime = runtime_digest(root, allow_ignored_caches=True)
    except (OSError, ValueError):
        pass

    return {
        "manifest_verified": verified is not None,
        "manifest_sha256": (
            hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes else "absent"
        ),
        "release_version": (
            verified.version
            if verified is not None
            else recorded.get("version", "unknown")
        ),
        "source_sha256": (
            verified.source_sha256
            if verified is not None
            else recorded.get("source_sha256", "unverified")
        ),
        "runtime_sha256": (
            verified.runtime_sha256 if verified is not None else current_runtime
        ),
        "marker_sha256": (
            hashlib.sha256(marker_bytes).hexdigest() if marker_bytes else "absent"
        ),
        "marker_verified": marker_bytes == CANONICAL_RUNTIME_MARKER_BYTES,
    }


def canonical_release_marker_present() -> bool:
    """Whether this process is inside the final marker-qualified image stage.

    Native developer/test compositions remain usable but carry a noncanonical
    runtime identity. Production and competition acceptance separately require
    this marker plus the verified manifest and full effective CPU policy.
    """

    return _release_identity().get("marker_verified") is True


def payout_runtime_attestation(
    config: Any,
    scoring_config: Any,
    *,
    backend_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Complete canonical payload committed by scorer identity and packets."""

    versions = dict(
        backend_versions
        if backend_versions is not None
        else probe_payout_backend_versions(config, scoring_config)
    )
    # The runtime backend stamp is the digest OF this payload and therefore is
    # never itself one of the payload inputs.
    versions.pop(RUNTIME_BACKEND_KEY, None)
    actual_os = "linux" if sys.platform.startswith("linux") else sys.platform
    torch_policy = (
        _isolated_torch_policy_probe()
        if getattr(config, "backend", "real") == "real"
        else {
            name: "not-applicable"
            for name in (
                "actual_torch_intraop_threads",
                "actual_torch_interop_threads",
                "actual_torch_deterministic_algorithms",
                "actual_torch_deterministic_warn_only",
                "actual_torch_mkldnn_enabled",
                "actual_torch_mkldnn_deterministic",
                "actual_torch_nnpack_enabled",
                "actual_torch_cpu_capability",
                "actual_openmp_threads",
                "actual_mkl_threads",
                "actual_mkl_cbwr",
                "actual_mkl_dynamic",
            )
        }
    )
    return {
        "schema": RUNTIME_COMMITMENT_SCHEMA,
        "release": _release_identity(),
        "execution_policy": {
            "required_os": "linux",
            "required_arch": "amd64",
            "actual_os": actual_os,
            "actual_arch": _normal_arch(platform.machine()),
            "libc": "/".join(part or "unknown" for part in platform.libc_ver()),
            "torch_intraop_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "torch_interop_threads": "1",
            "mkl_threads": os.environ.get("MKL_NUM_THREADS", "unset"),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC", "unset"),
            "mkl_dynamic": os.environ.get("MKL_DYNAMIC", "unset"),
            "mkl_cbwr": os.environ.get("MKL_CBWR", "unset"),
            "aten_cpu_capability_override": os.environ.get(
                "ATEN_CPU_CAPABILITY", "unset"
            ),
            **torch_policy,
        },
        "payout_backends": dict(sorted(versions.items())),
    }


def runtime_commitment_digest(attestation: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(attestation))).hexdigest()


def runtime_backend_stamp(attestation: Mapping[str, Any]) -> str:
    return RUNTIME_BACKEND_PREFIX + runtime_commitment_digest(attestation)


def require_attested_backend_versions(
    attestation: Mapping[str, Any], versions: Mapping[str, str]
) -> None:
    """Require the packet backend map to be the exact attestation projection."""

    payout_backends = attestation.get("payout_backends")
    if not isinstance(payout_backends, Mapping):
        raise RuntimeError("payout runtime attestation has no backend-version map")
    expected = dict(payout_backends)
    expected[RUNTIME_BACKEND_KEY] = runtime_backend_stamp(attestation)
    actual = dict(versions)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        moved = sorted(
            key
            for key in set(expected) & set(actual)
            if actual[key] != expected[key]
        )
        raise RuntimeError(
            "backend versions differ from the canonical payout-runtime "
            f"attestation (missing={missing}, unexpected={unexpected}, moved={moved})"
        )


def canonical_runtime_problems(attestation: Mapping[str, Any]) -> list[str]:
    """Explain why ``attestation`` is not the qualified release CPU runtime."""

    problems: list[str] = []
    if attestation.get("schema") != RUNTIME_COMMITMENT_SCHEMA:
        problems.append(
            f"schema is {attestation.get('schema')!r}, expected "
            f"{RUNTIME_COMMITMENT_SCHEMA!r}"
        )
    release = attestation.get("release")
    policy = attestation.get("execution_policy")
    backends = attestation.get("payout_backends")
    if not isinstance(release, Mapping):
        problems.append("release identity is missing")
    else:
        if release.get("manifest_verified") is not True:
            problems.append(
                "runtime-release-manifest.json is absent, stale, or invalid"
            )
        if release.get("marker_verified") is not True:
            problems.append(
                "digest-pinned release-image runtime marker is absent or invalid"
            )
    if not isinstance(policy, Mapping):
        problems.append("execution policy is missing")
    else:
        if policy.get("required_os") != "linux":
            problems.append(
                f"required_os is {policy.get('required_os')!r}, expected 'linux'"
            )
        if policy.get("required_arch") != "amd64":
            problems.append(
                f"required_arch is {policy.get('required_arch')!r}, expected 'amd64'"
            )
        if policy.get("actual_os") != "linux":
            problems.append(f"OS is {policy.get('actual_os')!r}, expected 'linux'")
        if policy.get("actual_arch") != "amd64":
            problems.append(
                f"architecture is {policy.get('actual_arch')!r}, expected 'amd64'"
            )
        for field in (
            "torch_intraop_threads",
            "torch_interop_threads",
            "mkl_threads",
            "openblas_threads",
        ):
            if policy.get(field) != "1":
                problems.append(f"{field} is {policy.get(field)!r}, expected '1'")
        if policy.get("omp_dynamic") != "FALSE":
            problems.append(
                f"omp_dynamic is {policy.get('omp_dynamic')!r}, expected 'FALSE'"
            )
        if policy.get("mkl_dynamic") != "FALSE":
            problems.append(
                f"mkl_dynamic is {policy.get('mkl_dynamic')!r}, expected 'FALSE'"
            )
        if policy.get("mkl_cbwr") != "COMPATIBLE":
            problems.append(
                f"mkl_cbwr is {policy.get('mkl_cbwr')!r}, expected 'COMPATIBLE'"
            )
        if policy.get("aten_cpu_capability_override") != "default":
            problems.append("ATEN_CPU_CAPABILITY override must be 'default'")
        problems.extend(_torch_policy_problems(policy))
    if not isinstance(backends, Mapping):
        problems.append("payout backend versions are missing")
    else:
        missing = sorted(_REQUIRED_PAYOUT_BACKENDS - set(backends))
        if missing:
            problems.append(
                "payout backend versions are missing: " + ", ".join(missing)
            )
        unavailable = sorted(
            name
            for name, value in backends.items()
            if "not-configured" in str(value) or str(value).endswith("/unknown")
        )
        if unavailable:
            problems.append(
                "payout backends are not fully configured: " + ", ".join(unavailable)
            )
        pieapp = str(backends.get("pieapp", ""))
        if not pieapp.endswith(":cpu"):
            problems.append("PieAPP payout backend must attest ':cpu'")
        torch_version = str(backends.get("torch", "")).lower()
        if (
            "+cpu" not in torch_version
            or "+cu" in torch_version
            or "cuda" in torch_version
        ):
            problems.append("PyTorch payout backend must be the locked CPU wheel")
        if backends.get("pieapp_weights") != f"sha256:{PIEAPP_WEIGHTS_SHA256}":
            problems.append(
                "PieAPP weights digest differs from the pinned release asset"
            )
    return problems


def require_canonical_release_runtime(attestation: Mapping[str, Any]) -> None:
    """Fail closed unless scoring/auditing runs in the qualified image policy."""

    problems = canonical_runtime_problems(attestation)
    if problems:
        raise RuntimeError(
            "canonical payout runtime required (digest-pinned Linux/amd64 release "
            "image; strict CPU recomputation): " + "; ".join(problems)
        )


__all__ = [
    "CANONICAL_RUNTIME_MARKER_BYTES",
    "CANONICAL_RUNTIME_MARKER_PATH",
    "RUNTIME_BACKEND_KEY",
    "RUNTIME_COMMITMENT_SCHEMA",
    "canonical_release_marker_present",
    "canonical_runtime_problems",
    "complete_payout_backend_versions",
    "initialize_canonical_torch_cpu_runtime",
    "payout_runtime_attestation",
    "require_attested_backend_versions",
    "probe_payout_backend_versions",
    "require_canonical_release_runtime",
    "runtime_backend_stamp",
    "runtime_commitment_digest",
]
