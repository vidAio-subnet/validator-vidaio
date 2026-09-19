"""Hardening for every media-tool invocation that can see submission bytes."""

from __future__ import annotations

from typing import Sequence

# --- untrusted media inputs ------------------------------------------------------------
#
# A submission is attacker-controlled bytes. libavformat picks a demuxer from CONTENT, and
# several demuxers are not containers at all but scripts that open OTHER files: an
# ``ffconcat`` playlist (or an HLS/DASH manifest) naming a sibling file decodes that file
# instead. A 42-byte "video" pointing at the served input therefore measured as a perfect,
# zero-byte encode. Every tool invocation that can see submission bytes is pinned to
# self-contained containers and to the local-file protocol. The list is deliberately
# short: what the scorer itself writes (y4m) plus the containers miners actually use.
TRUSTED_DEMUXERS: tuple[str, ...] = (
    "matroska",
    "webm",
    "mov",
    "mp4",
    "m4a",
    "3gp",
    "3g2",
    "mj2",
    "ivf",
    "yuv4mpegpipe",
)
UNTRUSTED_INPUT_ARGS: tuple[str, ...] = (
    "-protocol_whitelist",
    "file",
    "-format_whitelist",
    ",".join(TRUSTED_DEMUXERS),
)


def harden_media_inputs(argv: Sequence[str]) -> list[str]:
    """Return ``argv`` with :data:`UNTRUSTED_INPUT_ARGS` in front of every file input.

    Applied at EXECUTION time, never to a recorded plan: the canonicalization plan digest
    committed in score packets stays what it always was, while the process that actually
    runs can no longer be steered to another demuxer, protocol or file. ``-f lavfi``
    sources (the scorer's own synthetic inputs) are left alone. Idempotent.
    """
    out: list[str] = []
    items = list(argv)
    for index, token in enumerate(items):
        if token == "-i":
            synthetic = index >= 2 and items[index - 2] == "-f" and items[index - 1] == "lavfi"
            already = tuple(out[-len(UNTRUSTED_INPUT_ARGS):]) == UNTRUSTED_INPUT_ARGS
            if not synthetic and not already:
                out.extend(UNTRUSTED_INPUT_ARGS)
        out.append(token)
    return out


__all__ = ["TRUSTED_DEMUXERS", "UNTRUSTED_INPUT_ARGS", "harden_media_inputs"]
