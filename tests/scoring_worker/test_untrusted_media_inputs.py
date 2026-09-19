"""A submission is attacker-controlled bytes: it must never steer the scorer to other files.

Reported by a miner: a 42-byte ``ffconcat`` playlist naming the served input was probed as
a valid h264 video and decoded byte-identical to that input — a perfect score for zero
bytes. Two independent layers now stop the whole class: every tool invocation is pinned
to self-contained containers + the local-file protocol, and each input is staged in its
own directory so a relative name cannot reach a sibling.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from vidaio.scoring.backends_real import (
    CanonicalizationError,
    FfprobeBackend,
    MediaToolError,
    CanonicalizeExecutor,
)
from vidaio.scoring.canonicalize import build_canonicalization_plan, plan_template_digest
from vidaio.scoring.media_inputs import UNTRUSTED_INPUT_ARGS, harden_media_inputs
from vidaio.scoring.phash_cpu import CpuVideoPhash
from vidaio.scoring_worker.inputs import snapshot_request_inputs

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg/ffprobe",
)

PLAYLIST = b"ffconcat version 1.0\nfile miner_input.mkv\n"
HLS = b"#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:1.0,\nminer_input.mkv\n#EXT-X-ENDLIST\n"


def _clip(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         "testsrc2=size=160x90:rate=10", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-y", str(path)],
        check=True,
    )
    return path


def test_the_reported_playlist_really_was_a_video_without_the_pin(tmp_path: Path) -> None:
    """Documents the bug: plain libavformat follows the playlist to the sibling input."""
    _clip(tmp_path / "miner_input.mkv")
    (tmp_path / "output.mp4").write_bytes(PLAYLIST)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name:stream=codec_name",
         "-of", "csv=p=0", str(tmp_path / "output.mp4")],
        capture_output=True, text=True,
    )
    assert out.returncode == 0 and "concat" in out.stdout and "h264" in out.stdout


@pytest.mark.parametrize(("name", "payload"), [("output.mp4", PLAYLIST), ("output.m3u8", HLS)])
def test_playlists_are_refused_by_every_tool_even_next_to_their_target(
    tmp_path: Path, name: str, payload: bytes
) -> None:
    _clip(tmp_path / "miner_input.mkv")
    candidate = tmp_path / name
    candidate.write_bytes(payload)
    with pytest.raises(MediaToolError):
        FfprobeBackend().probe(str(candidate))
    plan = build_canonicalization_plan(str(candidate), str(tmp_path / "canon.y4m"))
    with pytest.raises(CanonicalizationError):
        CanonicalizeExecutor().run(plan)
    assert not (tmp_path / "canon.y4m").exists() or (tmp_path / "canon.y4m").stat().st_size == 0
    with pytest.raises(Exception):
        CpuVideoPhash().compute_phash(str(candidate))


def test_honest_containers_still_probe_and_canonicalize(tmp_path: Path) -> None:
    for suffix in (".mkv", ".mp4"):
        clip = _clip(tmp_path / f"honest{suffix}")
        info = FfprobeBackend().probe(str(clip))
        assert info.codec == "h264" and (info.width, info.height) == (160, 90)
        canon = tmp_path / f"honest{suffix}.y4m"
        CanonicalizeExecutor().run(build_canonicalization_plan(str(clip), str(canon)))
        assert FfprobeBackend().probe(str(canon)).frame_count == info.frame_count


def test_the_committed_plan_digest_is_unchanged_by_the_execution_pin() -> None:
    plan = build_canonicalization_plan("/in/output.mp4", "/tmp/c.y4m")
    assert "-format_whitelist" not in plan  # the recorded plan is what it always was
    hardened = harden_media_inputs(plan)
    assert tuple(hardened[hardened.index("-i") - 4 : hardened.index("-i")]) == UNTRUSTED_INPUT_ARGS
    assert harden_media_inputs(hardened) == hardened
    assert plan_template_digest(plan, "/in/output.mp4", "/tmp/c.y4m") == plan_template_digest(
        build_canonicalization_plan("/x/y.mp4", "/z.y4m"), "/x/y.mp4", "/z.y4m"
    )
    synthetic = ["ffmpeg", "-f", "lavfi", "-i", "color=c=gray"]
    assert harden_media_inputs(synthetic) == synthetic


def test_each_input_is_staged_in_its_own_directory(tmp_path: Path) -> None:
    sources = {}
    for field in ("reference", "miner_input", "output"):
        path = tmp_path / f"{field}-src.mkv"
        path.write_bytes(field.encode() * 50)
        sources[field] = (str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    snapshot = snapshot_request_inputs(
        reference=sources["reference"], miner_input=sources["miner_input"],
        output=sources["output"], dest_dir=tmp_path / "private",
    )
    parents = {Path(v.path).parent for v in (snapshot.reference, snapshot.miner_input, snapshot.output)}
    assert len(parents) == 3
    output_dir = Path(snapshot.output.path).parent
    assert [p.name for p in output_dir.iterdir()] == [Path(snapshot.output.path).name]
    assert not (output_dir / "miner_input.mkv").exists()
