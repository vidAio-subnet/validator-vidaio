#!/usr/bin/env python3
"""Materialize one standalone, fresh-Git-ready competition contender tree."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "template"
PROFILES = ROOT / "profiles"
FILES = ("Dockerfile", "run.sh", "gpu_transform.cu")


def materialize(*, track: str, variant: str, destination: Path) -> Path:
    profile = PROFILES / f"{track}-{variant}.env"
    if not profile.is_file():
        raise ValueError(f"unknown contender profile: {track}/{variant}")
    if destination.exists():
        raise FileExistsError(f"refusing to reuse/overwrite destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(mode=0o700)
    try:
        for name in FILES:
            source = TEMPLATE / name
            target = destination / name
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
                dst.flush()
                os.fsync(dst.fileno())
        with profile.open("rb") as src, (destination / "variant.env").open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
            dst.flush()
            os.fsync(dst.fileno())
        (destination / "run.sh").chmod(0o755)
        for name in ("Dockerfile", "gpu_transform.cu", "variant.env"):
            (destination / name).chmod(0o644)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("compression", "upscaling"), required=True)
    parser.add_argument(
        "--variant",
        choices=("quality", "balanced", "compact", "baseline"),
        required=True,
    )
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        path = materialize(
            track=args.track,
            variant=args.variant,
            destination=args.destination.resolve(),
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
