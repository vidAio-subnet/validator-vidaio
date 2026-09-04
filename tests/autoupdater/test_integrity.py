"""Release-manifest identity: full CI source + exact shipped runtime bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vidaio.autoupdater.integrity import (
    RUNTIME_DIRS,
    build_runtime_manifest,
    runtime_digest,
    verify_ci_release,
    verify_runtime_manifest,
    write_runtime_manifest,
)


def _runtime(root: Path, *, version: str = "0.1.0") -> Path:
    for directory in RUNTIME_DIRS:
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "vidaio" / "service.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "scripts" / "entry.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "config" / "default.yaml").write_text("chain: {}\n", encoding="utf-8")
    contender_profile = (
        root
        / "examples"
        / "competition_contenders"
        / "profiles"
        / "compression-quality.env"
    )
    contender_profile.parent.mkdir(parents=True, exist_ok=True)
    contender_profile.write_text("VIDAIO_NEXT_CRF=22\n", encoding="utf-8")
    (root / "VERSION").write_text(version + "\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='vidaio'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return root


def test_checkout_manifest_reproves_full_source_and_runtime(tmp_path: Path) -> None:
    root = _runtime(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "release.md").write_text("tested\n", encoding="utf-8")
    path = root / "runtime-release-manifest.json"
    write_runtime_manifest(path, source_root=root, runtime_root=root)

    verified = verify_runtime_manifest(
        path,
        source_root=root,
        runtime_root=root,
        verify_source_tree=True,
    )
    assert verified.version == "0.1.0"
    assert verified.runtime_sha256 == runtime_digest(root)

    # A source-only CI input is not allowed to drift in a checkout artifact.
    (root / "docs" / "release.md").write_text("changed after CI\n", encoding="utf-8")
    with pytest.raises(ValueError, match="full source digest differs"):
        verify_runtime_manifest(
            path,
            source_root=root,
            runtime_root=root,
            verify_source_tree=True,
        )


@pytest.mark.parametrize("changed_path", ("docs/release.md", "vidaio/service.py"))
def test_gate_start_snapshot_refuses_tree_changes_before_publication(
    tmp_path: Path, changed_path: str
) -> None:
    root = _runtime(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "release.md").write_text("tested\n", encoding="utf-8")
    snapshot = tmp_path.parent / f"{tmp_path.name}-gate-start.json"
    write_runtime_manifest(snapshot, source_root=root, runtime_root=root)

    (root / changed_path).write_text("changed during gate\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest differs|runtime input differs"):
        verify_runtime_manifest(
            snapshot,
            source_root=root,
            runtime_root=root,
            verify_source_tree=True,
        )


def test_ci_release_refuses_stale_marker_after_source_only_change(
    tmp_path: Path,
) -> None:
    root = _runtime(tmp_path)
    (root / "deploy").mkdir()
    deployment = root / "deploy" / "release.py"
    deployment.write_text("REVIEWED = True\n", encoding="utf-8")
    path = root / "runtime-release-manifest.json"
    manifest = write_runtime_manifest(path, source_root=root, runtime_root=root)
    marker = root / "ci-pass"
    marker.write_text(
        "0.1.0\n"
        f"source-sha256 {manifest['source_sha256']}\n"
        f"runtime-sha256 {manifest['runtime_sha256']}\n"
        f"manifest-sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}\n",
        encoding="utf-8",
    )

    deployment.write_text("REVIEWED = False\n", encoding="utf-8")

    with pytest.raises(ValueError, match="full source digest differs"):
        verify_ci_release(
            marker,
            path,
            source_root=root,
            runtime_root=root,
            expected_version="0.1.0",
        )


def test_gate_start_snapshot_refuses_source_executable_bit_change(
    tmp_path: Path,
) -> None:
    root = _runtime(tmp_path)
    (root / "deploy").mkdir()
    entrypoint = root / "deploy" / "release.py"
    entrypoint.write_text("print('release')\n", encoding="utf-8")
    entrypoint.chmod(0o644)
    snapshot = tmp_path.parent / f"{tmp_path.name}-mode-gate-start.json"
    write_runtime_manifest(snapshot, source_root=root, runtime_root=root)

    entrypoint.chmod(0o755)

    with pytest.raises(ValueError, match="full source digest differs"):
        verify_runtime_manifest(
            snapshot,
            source_root=root,
            runtime_root=root,
            verify_source_tree=True,
        )


def test_release_source_identity_rejects_symlinked_input(tmp_path: Path) -> None:
    root = _runtime(tmp_path)
    (root / "deploy").mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("UNBOUND = True\n", encoding="utf-8")
    (root / "deploy" / "release.py").symlink_to(outside)

    with pytest.raises(ValueError, match="source tree must not contain symlinks"):
        write_runtime_manifest(
            root / "runtime-release-manifest.json",
            source_root=root,
            runtime_root=root,
        )


@pytest.mark.parametrize(
    "dockerignore",
    (
        (
            ".venv/\ndata/\n.git/\nruntime-release-manifest.json\n"
            ".env\n*.env\n!.env.example\n"
            "!examples/competition_contenders/profiles/*.env\n"
        ),
        (
            ".venv/\ndata/\n.git/\nruntime-release-manifest.json\n"
            ".env\n*.env\n!.env.example\n.env.*\n"
            "!examples/competition_contenders/profiles/*.env\n"
        ),
        (
            ".venv/\ndata/\n.git/\nruntime-release-manifest.json\n"
            ".env\n.env.*\n*.env\n!.env.example\n"
            "!examples/competition_contenders/profiles/*.env\n!**\n"
        ),
        (
            ".venv/\n.git/\nruntime-release-manifest.json\n"
            ".env\n.env.*\n*.env\n!.env.example\n"
            "!examples/competition_contenders/profiles/*.env\n"
        ),
    ),
)
def test_release_source_identity_rejects_unsafe_docker_secret_boundary(
    tmp_path: Path,
    dockerignore: str,
) -> None:
    root = _runtime(tmp_path)
    (root / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (root / ".dockerignore").write_text(dockerignore, encoding="utf-8")

    with pytest.raises(ValueError, match="dockerignore|Docker build"):
        write_runtime_manifest(
            root / "runtime-release-manifest.json",
            source_root=root,
            runtime_root=root,
        )


def test_release_source_identity_rejects_undeclared_root_input(
    tmp_path: Path,
) -> None:
    root = _runtime(tmp_path)
    (root / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (root / ".dockerignore").write_text(
        ".venv/\ndata/\n.git/\nruntime-release-manifest.json\n"
        ".env\n.env.*\n*.env\n!.env.example\n"
        "!examples/competition_contenders/profiles/*.env\n",
        encoding="utf-8",
    )
    (root / "unreviewed-release.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="undeclared root input.*unreviewed-release"):
        write_runtime_manifest(
            root / "runtime-release-manifest.json",
            source_root=root,
            runtime_root=root,
        )


def test_release_source_identity_rejects_private_role_env_file(
    tmp_path: Path,
) -> None:
    root = _runtime(tmp_path)
    role_env = root / "deploy" / "testnet" / "env" / "authority.env"
    role_env.parent.mkdir(parents=True)
    role_env.write_text("VIDAIO_HOTKEY_SEED=secret\n", encoding="utf-8")

    with pytest.raises(ValueError, match="private environment file"):
        write_runtime_manifest(
            root / "runtime-release-manifest.json",
            source_root=root,
            runtime_root=root,
        )


@pytest.mark.parametrize(
    ("directory", "existing_path", "added_path", "expected_error"),
    (
        (
            "deploy",
            "deploy/modal/vidaio_next_gpu_miner.py",
            "deploy/modal/unreviewed_gpu_entrypoint.py",
            "full source digest differs",
        ),
        (
            "examples",
            "examples/competition_contenders/README.md",
            "examples/competition_contenders/unreviewed/run.sh",
            "runtime input",
        ),
    ),
)
@pytest.mark.parametrize("mutation", ("changed", "added"))
def test_operator_release_artifacts_are_bound_by_full_source_identity(
    tmp_path: Path,
    directory: str,
    existing_path: str,
    added_path: str,
    expected_error: str,
    mutation: str,
) -> None:
    """Modal deploy/example drift must revoke an otherwise-green CI identity.

    Modal deployment code is source-only, while contender examples are also
    runtime-bound. Both are part of the release/testnet procedure and must
    remain frozen for the complete ``ci.sh`` run.
    """
    root = _runtime(tmp_path)
    existing = root / existing_path
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("reviewed release artifact\n", encoding="utf-8")
    snapshot = tmp_path.parent / (
        f"{tmp_path.name}-{directory}-{mutation}-gate-start.json"
    )
    write_runtime_manifest(snapshot, source_root=root, runtime_root=root)

    if mutation == "changed":
        existing.write_text("changed during gate\n", encoding="utf-8")
    else:
        added = root / added_path
        added.parent.mkdir(parents=True, exist_ok=True)
        added.write_text("unreviewed addition\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        verify_runtime_manifest(
            snapshot,
            source_root=root,
            runtime_root=root,
            verify_source_tree=True,
        )


def test_lean_image_manifest_binds_build_source_but_verifies_runtime(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    runtime = tmp_path / "runtime"
    _runtime(source)
    _runtime(runtime)
    (source / "tests").mkdir()
    (source / "tests" / "test_release.py").write_text("assert True\n", encoding="utf-8")
    (source / "deploy").mkdir()
    (source / "deploy" / "gpu.py").write_text("GPU = True\n", encoding="utf-8")
    (source / "examples" / "contender.txt").write_text("baseline\n", encoding="utf-8")
    (runtime / "examples" / "contender.txt").write_text("baseline\n", encoding="utf-8")
    path = runtime / "runtime-release-manifest.json"
    manifest = write_runtime_manifest(path, source_root=source, runtime_root=runtime)

    checkout_manifest = build_runtime_manifest(
        source_root=source,
        runtime_root=source,
    )
    assert manifest == checkout_manifest
    assert verify_runtime_manifest(path, runtime_root=runtime).file_count > 3


@pytest.mark.parametrize("mutation", ("changed", "added", "removed"))
def test_manifest_rejects_any_runtime_tree_drift(tmp_path: Path, mutation: str) -> None:
    root = _runtime(tmp_path)
    path = root / "runtime-release-manifest.json"
    write_runtime_manifest(path, source_root=root, runtime_root=root)
    if mutation == "changed":
        (root / "vidaio" / "service.py").write_text("VALUE = 2\n", encoding="utf-8")
    elif mutation == "added":
        (root / "scripts" / "unreviewed.py").write_text("pass\n", encoding="utf-8")
    else:
        (root / "config" / "default.yaml").unlink()
    with pytest.raises(ValueError, match="runtime input|runtime digest"):
        verify_runtime_manifest(path, source_root=root, runtime_root=root)


def test_runtime_identity_rejects_untracked_sourceless_bytecode(tmp_path: Path) -> None:
    root = _runtime(tmp_path)
    cache = root / "vidaio" / "__pycache__"
    cache.mkdir()
    (cache / "payload.cpython-313.pyc").write_bytes(b"not-reviewed-bytecode")
    with pytest.raises(ValueError, match="unverified cache/bytecode"):
        write_runtime_manifest(
            root / "runtime-release-manifest.json",
            source_root=root,
            runtime_root=root,
        )


def test_manifest_rejects_unsafe_or_duplicate_paths(tmp_path: Path) -> None:
    root = _runtime(tmp_path)
    path = root / "runtime-release-manifest.json"
    write_runtime_manifest(path, source_root=root, runtime_root=root)
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["files"][0]["path"] = "../outside.py"
    path.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe path"):
        verify_runtime_manifest(path, source_root=root, runtime_root=root)


def test_ci_release_requires_marker_and_manifest_digests_to_agree(
    tmp_path: Path,
) -> None:
    root = _runtime(tmp_path)
    path = root / "runtime-release-manifest.json"
    manifest = write_runtime_manifest(path, source_root=root, runtime_root=root)
    marker = root / "ci-pass"
    manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    marker.write_text(
        "0.1.0\n"
        f"source-sha256 {manifest['source_sha256']}\n"
        f"runtime-sha256 {manifest['runtime_sha256']}\n"
        f"manifest-sha256 {manifest_sha}\n",
        encoding="utf-8",
    )
    verified = verify_ci_release(
        marker,
        path,
        source_root=root,
        runtime_root=root,
        expected_version="0.1.0",
    )
    assert verified.runtime_sha256 == manifest["runtime_sha256"]

    marker.write_text(
        marker.read_text(encoding="utf-8")
        + f"runtime-sha256 {manifest['runtime_sha256']}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one runtime-sha256"):
        verify_ci_release(marker, path, source_root=root, runtime_root=root)


def test_ci_marker_hash_binds_manifest_policy_fields(tmp_path: Path) -> None:
    root = _runtime(tmp_path)
    path = root / "runtime-release-manifest.json"
    manifest = write_runtime_manifest(path, source_root=root, runtime_root=root)
    marker = root / "ci-pass"
    marker.write_text(
        "0.1.0\n"
        f"source-sha256 {manifest['source_sha256']}\n"
        f"runtime-sha256 {manifest['runtime_sha256']}\n"
        f"manifest-sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}\n",
        encoding="utf-8",
    )
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["schema_version"] = 2
    path.write_text(json.dumps(obj), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest bytes differ"):
        verify_ci_release(marker, path, source_root=root, runtime_root=root)
