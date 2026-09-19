#!/usr/bin/env python3
"""Materialize one standalone CPU compression contender tree, ready for `git init`.

    python materialize_cpu.py --variant svtav1-search --destination /tmp/my-contender
    cd /tmp/my-contender && git init -q && git add -A && git commit -qm submission
    git rev-parse HEAD 'HEAD^{tree}'      # the commit_sha and tree_sha you enroll

`x264-baseline` is the reference the examples are compared against; never enroll it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "cpu_compression"


def variants() -> list[str]:
    return sorted(path.stem for path in (ROOT / "profiles").glob("*.profile"))


def materialize(*, variant: str, destination: Path) -> Path:
    profile = ROOT / "profiles" / f"{variant}.profile"
    if not profile.is_file():
        raise ValueError(f"unknown CPU contender variant: {variant}")
    if destination.exists():
        raise FileExistsError(f"refusing to reuse/overwrite destination: {destination}")
    destination.mkdir(parents=True, mode=0o700)
    try:
        for name in ("Dockerfile", "run.sh"):
            shutil.copyfile(ROOT / name, destination / name)
        shutil.copyfile(profile, destination / "variant.env")
        (destination / "run.sh").chmod(0o755)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=variants(), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    print(materialize(variant=args.variant, destination=args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
