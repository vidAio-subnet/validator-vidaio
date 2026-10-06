"""The miner-side pre-check catches what the sandbox image builder refuses."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "competition_precheck", ROOT / "scripts" / "competition_precheck.py"
)
precheck = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(precheck)

DIGEST = "a" * 64


def test_tag_plus_digest_add_and_user_are_reported() -> None:
    problems, notes = precheck.check_dockerfile(
        f"FROM python@sha256:{DIGEST}\n"
        f"COPY --from=mwader/static-ffmpeg:9.0@sha256:{DIGEST} /ffmpeg /usr/local/bin/\n"
        "ADD weights.tar /w\n"
        "USER 1000\n"
        "COPY run.sh /app/run.sh\n"
    )
    assert any("tag AND a digest" in p for p in problems)
    assert any("ADD is rejected" in p for p in problems)
    assert any("USER is ignored" in n for n in notes)


def test_the_shipped_examples_pass() -> None:
    for example in ("removal_example", "interpolation_example", "cpu_compression"):
        assert precheck.main([str(ROOT / "examples" / "competition_contenders" / example)]) == 0
    # the interpolation example is a valid start for a commercial-use competition
    interpolation = ROOT / "examples" / "competition_contenders" / "interpolation_example"
    assert precheck.main([str(interpolation), "--commercial"]) == 0


def test_symlinks_and_missing_run_sh_are_blocking(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text(f"FROM python@sha256:{DIGEST}\nCOPY . /app\n")
    (tmp_path / "link").symlink_to("/etc/passwd")
    assert precheck.main([str(tmp_path)]) == 1


def test_commercial_competitions_require_the_licence_files(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text(f"FROM python@sha256:{DIGEST}\nCOPY . /app\n")
    (tmp_path / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    assert precheck.main([str(tmp_path)]) == 0
    assert precheck.main([str(tmp_path), "--commercial"]) == 1
    (tmp_path / "LICENSE").write_text("Apache License 2.0\n")
    assert precheck.main([str(tmp_path), "--commercial"]) == 1
    (tmp_path / "LICENSES.md").write_text("| component | source | licence |\n")
    assert precheck.main([str(tmp_path), "--commercial"]) == 0



def test_commercial_downloads_must_be_pinned_by_content() -> None:
    commit = "b" * 40
    unpinned = (
        f"FROM python@sha256:{DIGEST}\n"
        "RUN wget -O /w/model.pth https://example.com/model.pth\n"
        "RUN git clone https://github.com/x/y /y\n"
        "RUN hf download org/model --local-dir /w\n"
        "RUN pip install git+https://github.com/x/y.git torch\n"
    )
    problems, notes = precheck.check_downloads(unpinned)
    assert len(problems) == 4
    assert any("without checking its sha256" in p for p in problems)
    assert any("git clone" in p for p in problems)
    assert any("--revision" in p for p in problems)
    assert any("git+https" in p for p in problems)
    assert any("'torch' has no == version" in n for n in notes)  # advice, not blocking

    pinned = (
        f"FROM python@sha256:{DIGEST}\n"
        "RUN wget -O /w/model.pth https://example.com/model.pth \\\n"
        f" && echo \"{DIGEST}  /w/model.pth\" | sha256sum -c -\n"
        f"RUN git clone https://github.com/x/y /y && git -C /y checkout {commit}\n"
        f"RUN hf download org/model --revision {commit} --local-dir /w\n"
        f"RUN pip install git+https://github.com/x/y.git@{commit} torch==2.4.1\n"
    )
    assert precheck.check_downloads(pinned) == ([], [])


def test_unpinned_downloads_block_only_commercial_competitions(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text(
        f"FROM python@sha256:{DIGEST}\n"
        "RUN wget -O /app/w.pth https://example.com/w.pth\n"
        "COPY run.sh /app/run.sh\n"
    )
    (tmp_path / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    (tmp_path / "LICENSE").write_text("MIT License\n")
    (tmp_path / "LICENSES.md").write_text("| component | source | licence |\n")
    assert precheck.main([str(tmp_path)]) == 0
    assert precheck.main([str(tmp_path), "--commercial"]) == 1
