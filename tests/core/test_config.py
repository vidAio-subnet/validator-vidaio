from pathlib import Path

from pydantic import BaseModel

from vidaio.core.config import CoreConfig, load_raw_config, section


class DemoConfig(BaseModel):
    factor: float = 1.0
    name: str = "x"


def test_load_yaml_and_section(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("core:\n  netuid: 292\ndemo:\n  factor: 2.5\n")
    raw = load_raw_config(cfg)
    core = section(raw, "core", CoreConfig)
    demo = section(raw, "demo", DemoConfig)
    assert core.netuid == 292
    assert demo.factor == 2.5
    assert demo.name == "x"


def test_env_override(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("demo:\n  factor: 2.5\n")
    monkeypatch.setenv("VIDAIO__DEMO__FACTOR", "7.5")
    monkeypatch.setenv("VIDAIO__DEMO__NAME", "override")
    raw = load_raw_config(cfg)
    demo = section(raw, "demo", DemoConfig)
    assert demo.factor == 7.5
    assert demo.name == "override"


def test_missing_file_and_missing_section() -> None:
    raw = load_raw_config(None)
    core = section(raw, "core", CoreConfig)
    assert core.netuid == 85
    assert core.db_path.name == "vidaio.db"
