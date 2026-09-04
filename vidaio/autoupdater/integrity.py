"""CI-source and shipped-runtime identities for release activation.

The CI source digest covers everything that can influence a release decision,
including tests, docs and the Dockerfile. A lean runtime image intentionally
does not contain that whole tree, so it cannot recompute the source digest.
Instead CI/build also emits a deterministic runtime manifest containing:

* the full source digest from the build context;
* the exact VERSION of the artifact;
* every shipped runtime input (``vidaio/``, ``scripts/``, ``config/``,
  ``examples/``, the dependency lock and project metadata), with size and
  sha256;
* a framed digest over those exact runtime paths and bytes.

The broader CI source identity also binds operator-run release artifacts that
do not belong in the lean service image: ``deploy/`` contains the fresh Modal
GPU deployment entrypoint. The pinned competition contender baselines under
``examples/`` are shipped because the release dependency probe and testnet
ladder consume them directly.

The updater requires the CI marker and manifest to agree, then re-hashes the
runtime tree and rejects missing, added, symlinked, cached-bytecode or changed
inputs before an activation command can run. The manifest bytes are identical
whether generated from the checkout or inside the lean image. CI separately
requests a full-source-tree recheck; a lean runtime records that source identity
while re-verifying every byte it actually ships.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SOURCE_DIRS = (
    "vidaio",
    "scripts",
    "config",
    "tests",
    "docs",
    "deploy",
    "examples",
)
SOURCE_FILES = (
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "DEPS.md",
    "Dockerfile",
    "Makefile",
    "README.md",
    "STATUS.md",
    "VERSION",
    "docker-compose.yml",
    "pyproject.toml",
    "uv.lock",
)

# Exactly what Dockerfile retains as release identity inputs. Native binaries
# and installed wheels are separately pinned/probed by the dependency stage;
# pyproject + uv.lock bind their requested and resolved dependency graph.
RUNTIME_DIRS = ("vidaio", "scripts", "config", "examples")
RUNTIME_FILES = ("VERSION", "pyproject.toml", "uv.lock")
RUNTIME_MANIFEST_SCHEMA = 1
RUNTIME_MANIFEST_NAME = "runtime-release-manifest.json"
IGNORED_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
LOCAL_ROOT_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "data",
        "node_modules",
    }
)


@dataclass(frozen=True, slots=True)
class VerifiedRuntimeManifest:
    """Validated identity passed from the gate to the activation command."""

    version: str
    source_sha256: str
    runtime_sha256: str
    file_count: int


def _verify_dockerignore_secret_boundary(base: Path) -> None:
    """Fail before a Docker build can send local environment secrets as context."""
    dockerfile = base / "Dockerfile"
    if not dockerfile.exists():
        return  # Minimal staged-runtime fixtures do not define an image build.
    ignore_file = base / ".dockerignore"
    if ignore_file.is_symlink() or not ignore_file.is_file():
        raise ValueError("release Docker build requires a regular .dockerignore")
    patterns = [
        line.strip()
        for line in ignore_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    required = (
        ".venv/",
        "data/",
        ".git/",
        "runtime-release-manifest.json",
        ".env",
        ".env.*",
        "*.env",
        "!.env.example",
        "!examples/competition_contenders/profiles/*.env",
    )
    missing = [pattern for pattern in required if pattern not in patterns]
    if missing:
        raise ValueError(
            ".dockerignore is missing the release secret boundary pattern(s): "
            + ", ".join(missing)
        )
    env_wildcard = patterns.index(".env.*")
    public_template = patterns.index("!.env.example")
    if public_template < env_wildcard:
        raise ValueError(
            ".dockerignore must re-include .env.example after excluding .env.*"
        )
    profile_wildcard = patterns.index("*.env")
    public_profiles = patterns.index("!examples/competition_contenders/profiles/*.env")
    if public_profiles < profile_wildcard:
        raise ValueError(
            ".dockerignore must re-include public contender profiles after excluding "
            "*.env"
        )
    allowed_reincludes = {
        "!.env.example",
        "!examples/competition_contenders/profiles/*.env",
    }
    unsafe_reincludes = [
        pattern
        for pattern in patterns
        if pattern.startswith("!") and pattern not in allowed_reincludes
    ]
    if unsafe_reincludes:
        raise ValueError(
            ".dockerignore has a re-include that can expose operator files: "
            + ", ".join(unsafe_reincludes)
        )


def _verify_declared_source_closure(base: Path) -> None:
    """Reject undeclared root inputs in an official Docker release checkout."""
    if not (base / "Dockerfile").exists():
        return
    declared = {*SOURCE_FILES, *SOURCE_DIRS}
    undeclared: list[str] = []
    for path in base.iterdir():
        name = path.name
        if name in declared or name in LOCAL_ROOT_DIRS:
            continue
        if name == RUNTIME_MANIFEST_NAME or name == ".env" or name.endswith(".log"):
            continue
        if name.startswith(".env.") and name != ".env.example":
            continue
        undeclared.append(name)
    if undeclared:
        raise ValueError(
            "release checkout has undeclared root input(s); add them to the source/image "
            "contract or remove them: " + ", ".join(sorted(undeclared))
        )


def release_files(root: str | Path) -> tuple[Path, ...]:
    """Return the regular full-source gate inputs in stable path order.

    Source symlinks are forbidden. Docker can retain or dereference them while
    the old identity walk silently omitted them, leaving executable/operator
    inputs outside the digest. Ignored cache directories remain exempt because
    they are also excluded from the Docker build context.
    """
    base = Path(root).resolve()
    _verify_dockerignore_secret_boundary(base)
    _verify_declared_source_closure(base)
    found: set[Path] = set()
    for name in SOURCE_FILES:
        path = base / name
        if path.is_symlink():
            raise ValueError(f"release source input must not be a symlink: {name}")
        if path.is_file() and not path.is_symlink():
            found.add(path)
    for dirname in SOURCE_DIRS:
        directory = base / dirname
        if directory.is_symlink():
            raise ValueError(
                f"release source directory must not be a symlink: {dirname}"
            )
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            relative = path.relative_to(base)
            if any(part in IGNORED_PARTS for part in relative.parts):
                continue
            public_profile = relative.parts[:3] == (
                "examples",
                "competition_contenders",
                "profiles",
            )
            private_environment = (
                path.name == ".env"
                or path.name.endswith(".env")
                or (path.name.startswith(".env.") and path.name != ".env.example")
            )
            if private_environment and not public_profile:
                raise ValueError(
                    "release source tree contains a private environment file: "
                    + relative.as_posix()
                )
            if path.is_symlink():
                raise ValueError(
                    "release source tree must not contain symlinks: "
                    + relative.as_posix()
                )
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(
                    "release source tree contains an unsupported filesystem entry: "
                    + relative.as_posix()
                )
            if not path.name.endswith((".pyc", ".pyo")):
                found.add(path)
    return tuple(sorted(found, key=lambda path: path.relative_to(base).as_posix()))


def _framed(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(len(chunk).to_bytes(8, "big"))
        digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(
    base: Path,
    files: Iterable[Path],
    *,
    domain: bytes,
    bind_executable_bit: bool = False,
) -> str:
    chunks: list[bytes] = [domain]
    for path in files:
        chunks.append(path.relative_to(base).as_posix().encode("utf-8"))
        if bind_executable_bit:
            chunks.append(b"executable" if path.stat().st_mode & 0o111 else b"regular")
        chunks.append(path.read_bytes())
    return _framed(chunks)


def source_digest(root: str | Path = ".") -> str:
    """Hash paths, executable bits and bytes for every release-source input."""
    base = Path(root).resolve()
    files = release_files(base)
    if not files:
        raise ValueError(f"no release source files found under {base}")
    return _tree_digest(
        base,
        files,
        domain=b"vidaio-release-source-v2",
        bind_executable_bit=True,
    )


def runtime_release_files(
    root: str | Path, *, allow_ignored_caches: bool = False
) -> tuple[Path, ...]:
    """Return every exact runtime identity input, rejecting ambiguous trees."""
    base = Path(root).resolve()
    found: set[Path] = set()
    missing: list[str] = []
    for name in RUNTIME_FILES:
        path = base / name
        if path.is_symlink():
            raise ValueError(f"runtime identity input must not be a symlink: {name}")
        if not path.is_file():
            missing.append(name)
        else:
            found.add(path)
    for dirname in RUNTIME_DIRS:
        directory = base / dirname
        if directory.is_symlink():
            raise ValueError(
                f"runtime identity directory must not be a symlink: {dirname}"
            )
        if not directory.is_dir():
            missing.append(dirname + "/")
            continue
        for path in directory.rglob("*"):
            relative = path.relative_to(base)
            if path.is_symlink():
                raise ValueError(
                    "runtime identity tree must not contain symlinks: "
                    + relative.as_posix()
                )
            if any(part in IGNORED_PARTS for part in relative.parts):
                if allow_ignored_caches:
                    continue
                raise ValueError(
                    "runtime identity tree contains an unverified cache/bytecode "
                    f"entry: {relative.as_posix()}"
                )
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(
                    "runtime identity tree contains an unsupported filesystem entry: "
                    + relative.as_posix()
                )
            if path.name.endswith((".pyc", ".pyo")):
                raise ValueError(
                    "runtime identity tree contains executable bytecode outside an "
                    f"ignored cache directory: {relative.as_posix()}"
                )
            else:
                found.add(path)
    if missing:
        raise ValueError(
            f"runtime root {base} is missing required input(s): " + ", ".join(missing)
        )
    return tuple(sorted(found, key=lambda path: path.relative_to(base).as_posix()))


def runtime_digest(
    root: str | Path = ".", *, allow_ignored_caches: bool = False
) -> str:
    """Hash the exact paths and bytes retained in the release runtime."""
    base = Path(root).resolve()
    return _tree_digest(
        base,
        runtime_release_files(base, allow_ignored_caches=allow_ignored_caches),
        domain=b"vidaio-release-runtime-v1",
    )


def build_runtime_manifest(
    *,
    source_root: str | Path,
    runtime_root: str | Path,
    allow_ignored_caches: bool = False,
) -> dict[str, Any]:
    """Build a reproducible manifest without embedding machine-specific paths."""
    source = Path(source_root).resolve()
    runtime = Path(runtime_root).resolve()
    files = runtime_release_files(runtime, allow_ignored_caches=allow_ignored_caches)
    version_lines = (runtime / "VERSION").read_text(encoding="utf-8").splitlines()
    version = version_lines[0].strip() if version_lines else ""
    if not version:
        raise ValueError(f"runtime VERSION is empty under {runtime}")
    entries = []
    for path in files:
        data = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(runtime).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
    return {
        "schema_version": RUNTIME_MANIFEST_SCHEMA,
        "version": version,
        "source_sha256": source_digest(source),
        "runtime_sha256": runtime_digest(
            runtime, allow_ignored_caches=allow_ignored_caches
        ),
        "files": entries,
    }


def write_runtime_manifest(
    path: str | Path,
    *,
    source_root: str | Path,
    runtime_root: str | Path,
    allow_ignored_caches: bool = False,
) -> dict[str, Any]:
    """Atomically write canonical JSON and return the manifest object."""
    manifest = build_runtime_manifest(
        source_root=source_root,
        runtime_root=runtime_root,
        allow_ignored_caches=allow_ignored_caches,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(destination)
    return manifest


def _sha(value: object, *, field: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"runtime manifest {field} is not a lowercase sha256")
    return text


def verify_runtime_manifest(
    path: str | Path,
    *,
    runtime_root: str | Path,
    source_root: str | Path | None = None,
    verify_source_tree: bool = False,
    allow_ignored_caches: bool = False,
) -> VerifiedRuntimeManifest:
    """Verify manifest structure plus every exact current runtime input."""
    manifest_path = Path(path)
    try:
        obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"runtime manifest {manifest_path} is unreadable: {exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise ValueError("runtime manifest root must be an object")
    expected_keys = {
        "schema_version",
        "version",
        "source_sha256",
        "runtime_sha256",
        "files",
    }
    if set(obj) != expected_keys:
        raise ValueError(
            "runtime manifest fields differ from schema: "
            f"expected {sorted(expected_keys)}, got {sorted(obj)}"
        )
    if obj["schema_version"] != RUNTIME_MANIFEST_SCHEMA:
        raise ValueError(
            f"runtime manifest schema {obj['schema_version']!r} != "
            f"{RUNTIME_MANIFEST_SCHEMA}"
        )
    version = str(obj["version"]).strip()
    if not version:
        raise ValueError("runtime manifest version is empty")
    source_sha = _sha(obj["source_sha256"], field="source_sha256")
    runtime_sha = _sha(obj["runtime_sha256"], field="runtime_sha256")
    entries = obj["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("runtime manifest files must be a non-empty list")

    runtime = Path(runtime_root).resolve()
    current_files = runtime_release_files(
        runtime, allow_ignored_caches=allow_ignored_caches
    )
    current_by_name = {
        path.relative_to(runtime).as_posix(): path for path in current_files
    }
    recorded: dict[str, tuple[int, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}:
            raise ValueError("runtime manifest file entry has invalid fields")
        relative = str(entry["path"])
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or relative.startswith("./")
        ):
            raise ValueError(f"runtime manifest contains unsafe path {relative!r}")
        if relative in recorded:
            raise ValueError(f"runtime manifest duplicates path {relative!r}")
        size = entry["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"runtime manifest size for {relative!r} is invalid")
        recorded[relative] = (size, _sha(entry["sha256"], field=f"files[{relative}]"))
    if set(recorded) != set(current_by_name):
        missing = sorted(set(recorded) - set(current_by_name))
        added = sorted(set(current_by_name) - set(recorded))
        raise ValueError(
            "runtime input set differs from manifest"
            + (f"; missing={missing}" if missing else "")
            + (f"; added={added}" if added else "")
        )
    for relative, current in current_by_name.items():
        data = current.read_bytes()
        expected_size, expected_sha = recorded[relative]
        if (
            len(data) != expected_size
            or hashlib.sha256(data).hexdigest() != expected_sha
        ):
            raise ValueError(f"runtime input differs from manifest: {relative}")
    current_runtime_sha = runtime_digest(
        runtime, allow_ignored_caches=allow_ignored_caches
    )
    if current_runtime_sha != runtime_sha:
        raise ValueError(
            "runtime digest differs from manifest: "
            f"expected {runtime_sha}, got {current_runtime_sha}"
        )
    version_lines = (runtime / "VERSION").read_text(encoding="utf-8").splitlines()
    current_version = version_lines[0].strip() if version_lines else ""
    if current_version != version:
        raise ValueError(
            f"runtime VERSION {current_version!r} differs from manifest {version!r}"
        )
    if verify_source_tree:
        source = Path(source_root if source_root is not None else runtime).resolve()
        current_source_sha = source_digest(source)
        if current_source_sha != source_sha:
            raise ValueError(
                "full source digest differs from manifest: "
                f"expected {source_sha}, got {current_source_sha}"
            )
    return VerifiedRuntimeManifest(
        version=version,
        source_sha256=source_sha,
        runtime_sha256=runtime_sha,
        file_count=len(recorded),
    )


def verify_ci_release(
    marker_path: str | Path,
    manifest_path: str | Path,
    *,
    runtime_root: str | Path,
    source_root: str | Path | None = None,
    expected_version: str | None = None,
) -> VerifiedRuntimeManifest:
    """Verify one CI marker against its immutable staged runtime artifact.

    The marker is deliberately small and operator-readable, but neither it nor
    the manifest is trusted independently.  A valid release needs the marker's
    version, full-source digest and shipped-runtime digest to agree with the
    manifest, and the manifest in turn has to agree with every byte currently
    present in the staged runtime tree.
    """
    marker = Path(marker_path)
    try:
        lines = marker.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"CI pass marker {marker} is unreadable: {exc}") from exc
    version = lines[0].strip() if lines else ""
    if not version:
        raise ValueError(f"CI pass marker {marker} has no version")
    if expected_version is not None and version != expected_version:
        raise ValueError(
            f"CI pass marker version {version!r} != expected {expected_version!r}"
        )

    def marker_digest(prefix: str) -> str:
        values = [
            line.removeprefix(prefix).strip()
            for line in lines[1:]
            if line.startswith(prefix)
        ]
        if len(values) != 1:
            raise ValueError(
                f"CI pass marker must contain exactly one {prefix.strip()} entry"
            )
        return _sha(values[0], field=f"CI marker {prefix.strip()}")

    source_sha = marker_digest("source-sha256 ")
    runtime_sha = marker_digest("runtime-sha256 ")
    manifest_sha = marker_digest("manifest-sha256 ")
    manifest = Path(manifest_path)
    try:
        actual_manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"runtime manifest {manifest} is unreadable: {exc}") from exc
    if actual_manifest_sha != manifest_sha:
        raise ValueError("runtime manifest bytes differ from CI marker")
    verified = verify_runtime_manifest(
        manifest,
        runtime_root=runtime_root,
        source_root=source_root,
        # A staged checkout has the complete source tree. Re-hash it here so a
        # source-only change (deploy/docs/examples/tests/Dockerfile) invalidates
        # the old marker just as a runtime-byte change does. A lean image may
        # intentionally omit ``source_root`` and can only re-prove shipped bytes.
        verify_source_tree=source_root is not None,
    )
    # The verifier parses the file independently. Re-hash afterward so a
    # concurrently replaced manifest cannot validate unbound policy fields
    # between the marker check and the parse.
    try:
        final_manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(
            f"runtime manifest {manifest} became unreadable: {exc}"
        ) from exc
    if final_manifest_sha != manifest_sha:
        raise ValueError("runtime manifest changed while it was being verified")
    if verified.version != version:
        raise ValueError(
            f"runtime manifest version {verified.version!r} != CI marker {version!r}"
        )
    if verified.source_sha256 != source_sha:
        raise ValueError("runtime manifest source digest differs from CI marker")
    if verified.runtime_sha256 != runtime_sha:
        raise ValueError("runtime manifest runtime digest differs from CI marker")
    return verified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--write-runtime-manifest", type=Path)
    parser.add_argument("--verify-runtime-manifest", type=Path)
    parser.add_argument(
        "--verify-source-tree",
        action="store_true",
        help="also recompute the full CI source digest while verifying",
    )
    parser.add_argument(
        "--allow-ignored-caches",
        action="store_true",
        help="CI-checkout only: exclude local cache dirs that are never shipped",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    runtime_root = args.runtime_root or args.source_root
    if args.write_runtime_manifest is not None:
        manifest = write_runtime_manifest(
            args.write_runtime_manifest,
            source_root=args.source_root,
            runtime_root=runtime_root,
            allow_ignored_caches=args.allow_ignored_caches,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.verify_runtime_manifest is not None:
        verified = verify_runtime_manifest(
            args.verify_runtime_manifest,
            source_root=args.source_root,
            runtime_root=runtime_root,
            verify_source_tree=args.verify_source_tree,
            allow_ignored_caches=args.allow_ignored_caches,
        )
        print(json.dumps(asdict(verified), sort_keys=True))
        return 0
    print(source_digest(args.source_root))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the release CI gate (development tree)/Docker
    raise SystemExit(main())
