"""Offline contracts for the exact release-image and CI gate boundary."""

from __future__ import annotations

import shlex
import tomllib
from pathlib import Path

from scripts.check_doc_links import check_doc_links
from vidaio import __version__
from vidaio.autoupdater.integrity import RUNTIME_DIRS, SOURCE_DIRS, SOURCE_FILES

ROOT = Path(__file__).resolve().parents[2]


def test_release_version_sources_are_identical() -> None:
    marker_version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert marker_version == project["project"]["version"] == __version__


def test_manifest_builder_copies_only_declared_release_inputs() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    manifest_stage = dockerfile.split("FROM runtime AS runtime-manifest", 1)[1].split(
        "FROM runtime AS test", 1
    )[0]
    copy_inputs = {
        source
        for line in manifest_stage.splitlines()
        if line.startswith("COPY ")
        for source in shlex.split(line)[1:-1]
    }

    assert copy_inputs == {*SOURCE_FILES, *SOURCE_DIRS}
    for directory in SOURCE_DIRS:
        assert directory in copy_inputs
        assert f"COPY {directory} /release-source/{directory}" in manifest_stage


def test_runtime_image_copies_every_manifest_bound_directory() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime_stage = dockerfile.split("FROM runtime AS runtime-manifest", 1)[0]

    assert "examples" in RUNTIME_DIRS
    assert "COPY examples ./examples" in runtime_stage
    for directory in RUNTIME_DIRS:
        assert f"COPY {directory} ./{directory}" in runtime_stage


def test_dependency_gate_proves_final_image_default_uid() -> None:
    script = (ROOT / "scripts" / "ci.sh").read_text(encoding="utf-8")

    assert "docker build --platform linux/amd64 --target release" in script
    assert "'{{.Os}}/{{.Architecture}}'" in script
    assert '"linux/amd64"' in script
    assert "docker image inspect --format '{{.Config.User}}'" in script
    assert '"10001:10001"' in script
    assert "--user 10001:10001" not in script
    assert "os.getuid(), os.geteuid(), os.getgid(), os.getegid()" in script
    assert "0 not in identity[4]" in script


def test_final_image_and_production_preflight_require_canonical_payout_runtime() -> None:
    ci_script = (ROOT / "scripts" / "ci.sh").read_text(encoding="utf-8")
    preflight = (ROOT / "scripts" / "production_preflight.py").read_text(
        encoding="utf-8"
    )
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "--require-runtime-manifest --require-canonical-runtime" in ci_script
    assert "require_runtime_manifest=True" in preflight
    assert "require_canonical_runtime=True" in preflight
    # The runtime-stage dependency smoke occurs before either release marker is
    # created and must not falsely claim final-image qualification.
    runtime_stage = dockerfile.split("FROM runtime AS runtime-manifest", 1)[0]
    assert "--require-canonical-runtime" not in runtime_stage


def test_full_gate_binds_and_rechecks_the_exact_release_image() -> None:
    script = (ROOT / "scripts" / "ci.sh").read_text(encoding="utf-8")

    assert "RELEASE_IMAGE_ID=$(docker image inspect --format '{{.Id}}'" in script
    assert "^sha256:[0-9a-f]{64}$" in script
    assert script.count('"$RELEASE_IMAGE_ID" python') == 3
    assert (
        '"$RELEASE_IMAGE_ID" \\\n    python scripts/verify_release_dependencies.py'
        in script
    )
    assert script.count("CURRENT_RELEASE_IMAGE_ID=$(docker image inspect") == 2
    assert 'echo "image-id $RELEASE_IMAGE_ID"' in script
    assert script.index('echo "image-id $RELEASE_IMAGE_ID"') < script.index(
        'mv "$MARKER_TMP" data/ci-pass'
    )


def test_doc_link_checker_rejects_existing_paths_outside_repository(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    docs = repository / "docs"
    docs.mkdir(parents=True)
    outside = tmp_path / "host-only.md"
    outside.write_text("not part of the release\n", encoding="utf-8")
    (docs / "README.md").write_text(
        "[host-only](../../host-only.md)\n", encoding="utf-8"
    )

    _files, _checked, broken = check_doc_links(repository)

    assert broken == ("docs/README.md: ../../host-only.md (escapes repository)",)


def test_doc_link_checker_accepts_encoded_path_query_and_fragment(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Release Notes.md").write_text("# Ready\n", encoding="utf-8")
    (docs / "README.md").write_text(
        "[notes](Release%20Notes.md?view=full#ready)\n", encoding="utf-8"
    )

    files, checked, broken = check_doc_links(tmp_path)

    assert (files, checked, broken) == (2, 1, ())


def test_doc_link_checker_rejects_missing_markdown_anchor(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text(
        "# Existing\n\n[missing](#not-a-heading)\n", encoding="utf-8"
    )

    _files, _checked, broken = check_doc_links(tmp_path)

    assert broken == ("docs/README.md: #not-a-heading (missing Markdown anchor)",)
