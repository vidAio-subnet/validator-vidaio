"""CPU compression example contenders materialize into self-contained trees."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "examples" / "competition_contenders"


def _module():
    spec = importlib.util.spec_from_file_location("materialize_cpu", ROOT / "materialize_cpu.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_every_variant_materializes_a_distinct_complete_tree(tmp_path: Path) -> None:
    tool = _module()
    names = tool.variants()
    assert {"x264-medium", "x265-medium", "vp9-good", "svtav1-fixed", "svtav1-search"} <= set(names)
    profiles = set()
    for name in names:
        tree = tool.materialize(variant=name, destination=tmp_path / name)
        assert sorted(p.name for p in tree.iterdir()) == ["Dockerfile", "run.sh", "variant.env"]
        assert (tree / "run.sh").stat().st_mode & 0o111
        profiles.add((tree / "variant.env").read_text())
    assert len(profiles) == len(names)
    with pytest.raises(FileExistsError):
        tool.materialize(variant=names[0], destination=tmp_path / names[0])
    with pytest.raises(ValueError):
        tool.materialize(variant="nope", destination=tmp_path / "nope")


def test_run_script_is_valid_posix_shell_and_never_needs_network() -> None:
    script = ROOT / "cpu_compression" / "run.sh"
    subprocess.run(["sh", "-n", str(script)], check=True)
    text = script.read_text()
    for forbidden in ("curl", "wget", "apt-get", "pip install", "http://", "https://"):
        assert forbidden not in text
