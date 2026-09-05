"""Public thin-validator preflight; never submits weights or commitments."""

import argparse
import json
from pathlib import Path

from vidaio.core import load_raw_config
from vidaio.weightsetter.public_runtime import live_preflight, static_preflight, suggested_floor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--print-floor", action="store_true")
    args = parser.parse_args()
    raw = load_raw_config(args.config)
    try:
        if args.print_floor:
            result = suggested_floor(raw)
        else:
            result = {"static": static_preflight(raw)}
            if args.live:
                result["live"] = live_preflight(raw)
        print(json.dumps(result, indent=2, default=str))
    except Exception as error:
        print(json.dumps({"status": "PREFLIGHT_HOLD", "error_type": type(error).__name__,
                          "detail": str(error) if isinstance(error, ValueError) else "Check chain, identity, authority and public-evidence permissions; no chain write performed"}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
