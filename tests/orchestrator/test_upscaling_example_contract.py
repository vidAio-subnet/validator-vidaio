"""The upscaling example and its docs use the task file the evaluator actually writes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from vidaio.competition.interfaces import upscale_task_sidecar_bytes, upscale_task_sidecar_name

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "competition_contenders" / "upscaling_example"
LEGACY_PREFIX = ".vidaio-next-upscale-factor-"


def _example_module():
    spec = importlib.util.spec_from_file_location("upscaling_example", EXAMPLE / "upscaling_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_reads_the_task_file_the_runner_stages(tmp_path):
    name = "ab" * 32
    (tmp_path / name).write_bytes(b"input")
    (tmp_path / upscale_task_sidecar_name(name)).write_bytes(upscale_task_sidecar_bytes(4, 1080, 1920))

    assert _example_module().task(str(tmp_path / name)) == (4, 1080, 1920)


def test_example_and_docs_name_the_runner_task_file():
    prefix = upscale_task_sidecar_name("0" * 64)[:-64]
    paths = [EXAMPLE / f for f in ("upscaling_example.py", "run.sh", "README.md", "test_local.py")]
    mining = ROOT / "docs" / "MINING.md"
    if mining.exists():  # the validator mirror ships without the mining guide
        paths.append(mining)
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert prefix in text, path
        assert LEGACY_PREFIX not in text, path
