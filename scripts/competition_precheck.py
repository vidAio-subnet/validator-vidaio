#!/usr/bin/env python3
"""Check a competition entry locally BEFORE enrolling it (no network, no Docker needed).

    python scripts/competition_precheck.py /path/to/your/repo [--commit <sha>]

It applies the rules the evaluator enforces on a submission tree and the limits of the
sandbox image builder (docs/MINING.md, "The competition track"), so a Dockerfile that
builds with Docker locally but not in the sandbox is caught before the deadline:

- the tree is at most 512 MiB, holds only regular files and directories (no symlinks,
  no special files), and has no .gitmodules / Git LFS pointers;
- ``Dockerfile`` sits at the repository root and ``run.sh`` is copied to ``/app/run.sh``;
- the builder accepts only FROM, COPY, RUN, ENV, WORKDIR (plus ARG, LABEL, EXPOSE,
  which are harmless): ``ADD`` is rejected, ``USER`` is ignored, ``ENTRYPOINT``/``CMD``
  are ignored at run time, and an image reference may carry a tag OR a digest, never
  both (``name:tag@sha256:...`` fails; ``name@sha256:...`` works).

With ``--commit`` the check runs on exactly that commit (``git archive``), which is what
the evaluator fetches; without it, the working tree is checked. Exit status 0 = no
blocking problem found; 1 = at least one problem that would lose the entry.
"""

from __future__ import annotations

import argparse
import io
import re
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

MAX_TREE_BYTES = 512 * 1024 * 1024
ALLOWED = {"FROM", "COPY", "RUN", "ENV", "WORKDIR", "ARG", "LABEL", "EXPOSE"}
IGNORED = {"USER", "ENTRYPOINT", "CMD", "HEALTHCHECK", "STOPSIGNAL", "SHELL", "VOLUME"}
REJECTED = {"ADD", "ONBUILD"}
_TAG_AND_DIGEST = re.compile(r"^[^\s@]+:[^\s@/]+@sha256:[0-9a-f]{64}$")
_LFS = b"version https://git-lfs.github.com/spec/"


def _instructions(text: str) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    buf, start = "", 0
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not buf:
            start = number
        if not buf and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        word, _, rest = buf.strip().partition(" ")
        out.append((start, word.upper(), rest.strip()))
        buf = ""
    return out


def check_dockerfile(text: str) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    notes: list[str] = []
    copies_run_sh = False
    for line, word, rest in _instructions(text):
        if word in REJECTED:
            problems.append(f"line {line}: {word} is rejected by the sandbox builder (use COPY, or download in a RUN)")
        elif word in IGNORED:
            notes.append(f"line {line}: {word} is ignored (the run is always /bin/sh /app/run.sh <in> <out>)")
        elif word not in ALLOWED:
            problems.append(f"line {line}: unsupported instruction {word}")
        refs: list[str] = []
        if word == "FROM":
            refs = [tok for tok in rest.split() if not tok.startswith("--")][:1]
        if word == "COPY":
            refs = [tok.split("=", 1)[1] for tok in rest.split() if tok.startswith("--from=")]
            if re.search(r"(^|\s)(\./)?run\.sh(\s|$)", rest) and "/app" in rest:
                copies_run_sh = True
            if re.search(r"(^|\s)\.(\s|$)", rest) and re.search(r"\s/app/?\s*$", rest):
                copies_run_sh = True
        for ref in refs:
            if _TAG_AND_DIGEST.match(ref):
                problems.append(
                    f"line {line}: {ref!r} carries a tag AND a digest; keep only the digest "
                    "(name@sha256:...)"
                )
    if not copies_run_sh:
        notes.append("could not see run.sh being copied to /app/run.sh; make sure the image has it")
    return problems, notes


def _tree_from_commit(repo: Path, commit: str) -> dict[str, tuple[int, bytes | None]]:
    data = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", commit],
        check=True, capture_output=True,
    ).stdout
    files: dict[str, tuple[int, bytes | None]] = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            if member.isdir() or member.type == tarfile.XGLTYPE or member.name == "pax_global_header":
                continue
            kind = 0 if member.isfile() else 1
            head = tar.extractfile(member).read(200) if member.isfile() else None  # type: ignore[union-attr]
            files[member.name] = (member.size if kind == 0 else -1, head)
    return files


def _tree_from_dir(root: Path) -> dict[str, tuple[int, bytes | None]]:
    files: dict[str, tuple[int, bytes | None]] = {}
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if rel.split("/", 1)[0] == ".git":
            continue
        st = path.lstat()
        if stat.S_ISDIR(st.st_mode):
            continue
        if stat.S_ISREG(st.st_mode):
            with path.open("rb") as handle:
                files[rel] = (st.st_size, handle.read(200))
        else:
            files[rel] = (-1, None)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", type=Path)
    ap.add_argument("--commit", help="check exactly this commit (what the evaluator fetches)")
    args = ap.parse_args(argv)

    files = _tree_from_commit(args.repo, args.commit) if args.commit else _tree_from_dir(args.repo)
    problems: list[str] = []
    notes: list[str] = []
    total = sum(size for size, _ in files.values() if size > 0)
    if total > MAX_TREE_BYTES:
        problems.append(f"tree is {total} bytes, over the 512 MiB limit; download large weights in the Dockerfile")
    for name, (size, head) in sorted(files.items()):
        if size < 0:
            problems.append(f"{name}: symlinks and special files are not accepted")
        elif head is not None and head.startswith(_LFS):
            problems.append(f"{name}: Git LFS pointer; LFS is not fetched")
    if ".gitmodules" in files:
        problems.append(".gitmodules: submodules are not supported")
    if "Dockerfile" not in files:
        problems.append("no Dockerfile at the repository root")
    else:
        if args.commit:
            text = subprocess.run(
                ["git", "-C", str(args.repo), "show", f"{args.commit}:Dockerfile"],
                check=True, capture_output=True, text=True,
            ).stdout
        else:
            text = (args.repo / "Dockerfile").read_text(errors="replace")
        found, hints = check_dockerfile(text)
        problems += found
        notes += hints
    if not any(name == "run.sh" or name.endswith("/run.sh") for name in files):
        problems.append("no run.sh in the tree")
    for note in notes:
        print(f"note: {note}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if args.commit:
        tree = subprocess.run(
            ["git", "-C", str(args.repo), "rev-parse", f"{args.commit}^{{tree}}"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        print(f"commit {args.commit} tree {tree}")
    print("OK: no blocking problem found" if not problems else f"{len(problems)} blocking problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
