"""Layered configuration: YAML file -> VIDAIO__SECTION__KEY env overrides.

Each service module owns its config model and pulls its section with `section()`;
core only defines the process-level settings every service shares.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

ENV_PREFIX = "VIDAIO__"

M = TypeVar("M", bound=BaseModel)


class CoreConfig(BaseModel):
    data_dir: Path = Path("./data")
    db_filename: str = "vidaio.db"
    log_level: str = "INFO"
    metrics_port: int = 9100
    network: str = "finney"
    netuid: int = 85

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in key[len(ENV_PREFIX) :].split("__") if p]
        if not path:
            continue
        node: Any = raw
        ok = True
        for part in path[:-1]:
            nxt = node.get(part)
            if nxt is None:
                nxt = {}
                node[part] = nxt
            if not isinstance(nxt, dict):
                ok = False
                break
            node = nxt
        if ok:
            node[path[-1]] = yaml.safe_load(value)
    return raw


def load_raw_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config (if present) and apply env overrides on top."""
    raw: dict[str, Any] = {}
    if path is not None and Path(path).exists():
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config root must be a mapping: {path}")
        raw = loaded
    return _apply_env_overrides(raw)


def section(raw: dict[str, Any], name: str, model: type[M]) -> M:
    """Validate one named section of the raw config into a module's config model."""
    return model.model_validate(raw.get(name) or {})
