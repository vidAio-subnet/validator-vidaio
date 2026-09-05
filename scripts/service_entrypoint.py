"""Public miner and thin-validator entrypoints."""

import argparse
import json
from pathlib import Path

from vidaio.core import load_raw_config


def run_miner(raw):
    from vidaio.chain.factory import ChainConfig, make_chain_adapter
    from vidaio.core import CoreConfig, section, setup_logging
    from vidaio.miner.config import MinerConfig
    from vidaio.miner.service import Miner
    from vidaio.services import run_service
    from vidaio.services.artifact_auth import ArtifactServerAuth, CallableHotkeySigner, ChainValidatorRegistry

    core = section(raw, "core", CoreConfig)
    core.data_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(core.log_level)
    chain_config = section(raw, "chain", ChainConfig)
    if chain_config.mode != "bittensor":
        raise SystemExit("the public miner requires chain.mode=bittensor")
    config = section(raw, "miner", MinerConfig)
    chain = make_chain_adapter(raw)
    auth = ArtifactServerAuth(CallableHotkeySigner(config.artifact_hotkey, chain.sign),
                              ChainValidatorRegistry(chain, max_snapshot_age_seconds=config.artifact_validator_snapshot_max_age_seconds),
                              request_max_age_seconds=config.artifact_request_max_age_seconds,
                              request_future_skew_seconds=config.artifact_request_future_skew_seconds)
    run_service(Miner(raw, artifact_auth=auth))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=("miner", "reference-miner", "thin-validator-node", "weight-setter"))
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    raw = load_raw_config(args.config)
    if args.service in ("thin-validator-node", "weight-setter"):
        from vidaio.weightsetter.public_runtime import run_thin_validator, static_preflight

        if args.check_config:
            print(json.dumps(static_preflight(raw), indent=2, default=str))
        else:
            run_thin_validator(raw)
    elif args.check_config:
        raise SystemExit("--check-config is supported for thin validators; miner setup is documented in MINING.md")
    else:
        run_miner(raw)


if __name__ == "__main__":
    main()
