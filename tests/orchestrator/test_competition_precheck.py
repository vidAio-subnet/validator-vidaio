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
    for example in ("removal_example", "cpu_compression"):
        assert precheck.main([str(ROOT / "examples" / "competition_contenders" / example)]) == 0


def test_symlinks_and_missing_run_sh_are_blocking(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text(f"FROM python@sha256:{DIGEST}\nCOPY . /app\n")
    (tmp_path / "link").symlink_to("/etc/passwd")
    assert precheck.main([str(tmp_path)]) == 1
