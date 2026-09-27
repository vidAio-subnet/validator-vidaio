"""The sealed baseline archive is accepted when it carries exactly the registry's source.

A registry artifact made by ``git archive`` and the orchestrator's canonical tar of the
same pinned tree differ in container metadata (pax comment, modes, owners, mtimes) but
hold the same files. Any difference in the files themselves is still refused.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from vidaio.audit.store import ArtifactKind, LocalFsStore, PassthroughEnvelope
from vidaio.competition.epoch_evidence import CompetitionEvidenceError, _same_baseline_source

FILES = {
    "Dockerfile": b"FROM scratch\nCOPY run.sh /app/run.sh\n",
    "run.sh": b"#!/bin/sh\nexec /app/encode \"$1\" \"$2\"\n",
    "variant.env": b"PRESET=6\n",
    ".dockerignore": b".git\n",
}


def _git_archive_style(files: dict[str, bytes]) -> bytes:
    """pax global comment with the commit, modes 664/775, owner root, commit mtime."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT,
                      pax_headers={"comment": "3f23f8c990b18640b57d0e4514e1bdbf32f3f790"}) as tar:
        for name, body in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.uname, info.gname = len(body), 1787741197, "root", "root"
            info.mode = 0o775 if name.endswith(".sh") else 0o664
            tar.addfile(info, io.BytesIO(body))
    return out.getvalue()


def _canonical_style(files: dict[str, bytes], *, prefix: str = "") -> bytes:
    """mode 644, mtime 0, no owner names, sorted names."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(prefix + name)
            info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
            tar.addfile(info, io.BytesIO(files[name]))
    return out.getvalue()


@pytest.fixture()
def store(tmp_path: Path) -> LocalFsStore:
    return LocalFsStore(tmp_path, envelope=PassthroughEnvelope())


def test_same_tree_from_two_archivers_is_the_same_baseline(store: LocalFsStore) -> None:
    registry = store.put(_git_archive_style(FILES), ArtifactKind.SUBMISSION_ARCHIVE)
    sealed = store.put(_canonical_style(FILES), ArtifactKind.SUBMISSION_ARCHIVE)
    assert registry.digest != sealed.digest
    assert _same_baseline_source(store, sealed, registry)
    # a leading "./" is a path spelling, not a different file (dotfiles keep their dot)
    dotted = store.put(_canonical_style(FILES, prefix="./"), ArtifactKind.SUBMISSION_ARCHIVE)
    assert _same_baseline_source(store, dotted, registry)


def test_identical_bytes_are_accepted_without_opening_them(store: LocalFsStore) -> None:
    registry = store.put(_git_archive_style(FILES), ArtifactKind.SUBMISSION_ARCHIVE)
    assert _same_baseline_source(store, registry, registry)


@pytest.mark.parametrize(
    "changed",
    [
        {**FILES, "run.sh": b"#!/bin/sh\nexec /app/other \"$1\" \"$2\"\n"},  # content
        {**FILES, "extra.py": b"print(1)\n"},  # an added file
        {k: v for k, v in FILES.items() if k != "variant.env"},  # a removed file
        {("sub/" + k if k == "run.sh" else k): v for k, v in FILES.items()},  # a moved file
        {("dockerignore" if k == ".dockerignore" else k): v for k, v in FILES.items()},  # renamed dotfile
    ],
)
def test_any_difference_in_the_files_is_refused(store: LocalFsStore, changed) -> None:
    registry = store.put(_git_archive_style(FILES), ArtifactKind.SUBMISSION_ARCHIVE)
    sealed = store.put(_canonical_style(changed), ArtifactKind.SUBMISSION_ARCHIVE)
    assert not _same_baseline_source(store, sealed, registry)


def test_links_and_escaping_paths_refuse_the_archive(store: LocalFsStore) -> None:
    registry = store.put(_git_archive_style(FILES), ArtifactKind.SUBMISSION_ARCHIVE)
    for build in ("symlink", "escape"):
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode="w") as tar:
            for name in sorted(FILES):
                info = tarfile.TarInfo(name)
                info.size = len(FILES[name])
                tar.addfile(info, io.BytesIO(FILES[name]))
            if build == "symlink":
                link = tarfile.TarInfo("run2.sh")
                link.type, link.linkname = tarfile.SYMTYPE, "run.sh"
                tar.addfile(link)
            else:
                evil = tarfile.TarInfo("../outside")
                evil.size = 1
                tar.addfile(evil, io.BytesIO(b"x"))
        sealed = store.put(out.getvalue(), ArtifactKind.SUBMISSION_ARCHIVE)
        with pytest.raises(CompetitionEvidenceError):
            _same_baseline_source(store, sealed, registry)


def test_a_reference_whose_bytes_do_not_match_is_refused(store: LocalFsStore) -> None:
    from dataclasses import replace

    registry = store.put(_git_archive_style(FILES), ArtifactKind.SUBMISSION_ARCHIVE)
    sealed = store.put(_canonical_style(FILES), ArtifactKind.SUBMISSION_ARCHIVE)
    lying = replace(sealed, byte_size=sealed.byte_size - 1) if hasattr(sealed, "__dataclass_fields__") else sealed.model_copy(update={"byte_size": sealed.byte_size - 1})
    with pytest.raises(CompetitionEvidenceError):
        _same_baseline_source(store, lying, registry)
