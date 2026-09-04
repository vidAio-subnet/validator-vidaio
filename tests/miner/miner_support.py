"""Test helpers for the miner suite: tool lookup + tiny real clip generation.

Deliberately self-contained (no import from tests/challenge_service): the miner
is split into its own repo later and its suite must stand alone.
"""

import shutil
import subprocess
from pathlib import Path

AT = "2026-08-20T12:00:00Z"


def find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    brew = Path("/opt/homebrew/bin") / name
    return str(brew) if brew.exists() else None


FFMPEG = find_tool("ffmpeg")
FFPROBE = find_tool("ffprobe")


def generate_clip(
    out: Path,
    *,
    duration: float = 1.5,
    size: str = "160x120",
    crf: int = 23,
) -> Path:
    """One tiny real h264 clip via lavfi (testsrc2, ultrafast)."""
    subprocess.run(
        [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=12:duration={duration}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", str(out),
        ],
        check=True,
    )
    return out
