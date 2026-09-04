"""Public service entrypoint — the roles a source build can run today.

The development tree's full entrypoint drives every fleet role through its
private orchestration module. This public build runs the MINER natively; the
validator/auditor roles currently require the canonical release image (their
runners and the audit loop live in the development tree, and audit identity
binds the canonical runtime anyway. Extraction into the shared package is on
the roadmap).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_config(path: str | None) -> dict[str, Any]:
    from vidaio.core import load_raw_config

    return load_raw_config(Path(path) if path else ROOT / "config" / "default.yaml")


def run_miner(raw: dict[str, Any]) -> None:
    from vidaio.chain.factory import ChainConfig, make_chain_adapter
    from vidaio.core import CoreConfig, section, setup_logging
    from vidaio.miner.config import MinerConfig
    from vidaio.miner.service import Miner
    from vidaio.services import run_service
    from vidaio.services.artifact_auth import (
        ArtifactServerAuth,
        CallableHotkeySigner,
        ChainValidatorRegistry,
    )

    core = section(raw, "core", CoreConfig)
    core.data_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(core.log_level)

    chain_cfg = section(raw, "chain", ChainConfig)
    if chain_cfg.mode != "bittensor":
        raise SystemExit(
            "the public build runs the miner against the real chain "
            "(chain.mode: bittensor); report/sim mode is a development-tree tool"
        )
    miner_cfg = section(raw, "miner", MinerConfig)
    chain = make_chain_adapter(raw)
    auth = ArtifactServerAuth(
        CallableHotkeySigner(miner_cfg.artifact_hotkey, chain.sign),
        ChainValidatorRegistry(
            chain,
            max_snapshot_age_seconds=(
                miner_cfg.artifact_validator_snapshot_max_age_seconds
            ),
        ),
        request_max_age_seconds=miner_cfg.artifact_request_max_age_seconds,
        request_future_skew_seconds=miner_cfg.artifact_request_future_skew_seconds,
    )
    run_service(Miner(raw, artifact_auth=auth))


SERVICES = {"miner": run_miner, "reference-miner": run_miner}
_IMAGE_ONLY = (
    "thin-validator-node", "weight-setter", "auditor", "own-auditor",
    "inference-validator", "authority-node", "audit-results-api",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", help="role to run (miner)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    if args.service in _IMAGE_ONLY:
        raise SystemExit(
            "role " + repr(args.service) + " runs from the canonical release "
            "image in this build generation (see the validator repository "
            "README); the source runners are being extracted into the package"
        )
    runner = SERVICES.get(args.service)
    if runner is None:
        raise SystemExit("unknown role " + repr(args.service) + "; available: miner / reference-miner")
    runner(_load_config(args.config))


if __name__ == "__main__":
    main()
