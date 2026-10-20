"""An output that does not state its colour matrix/range canonicalizes to the same pixels
as the same frames with tags (comp 003b: untagged outputs read as BT.601 were converted
to BT.709, moving every pixel ~3.7 levels and failing pixel-exact kept frames)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.scoring_worker.conftest import FFMPEG, FFPROBE, requires_ffmpeg
from vidaio.scoring import MediaInfo, build_canonicalization_plan, untagged_color_fill
from vidaio.scoring.backends_real import CanonicalizeExecutor, FfprobeBackend


def _info(**tags) -> MediaInfo:
    return MediaInfo(codec="h264", width=64, height=48, fps=30.0, frame_count=3, duration=0.1,
                     byte_size=100, **tags)


def test_only_unstated_tags_of_yuv_streams_are_filled() -> None:
    assert untagged_color_fill(_info(color_space="bt709", color_range="tv")) == {}
    assert untagged_color_fill(_info(color_space="unknown", color_range="tv")) == {"fill_color_space": True}
    assert untagged_color_fill(_info(color_space="unknown", color_range="unknown")) == {
        "fill_color_space": True,
        "fill_color_range": True,
    }
    assert untagged_color_fill(_info(color_space="bt470bg", color_range="pc")) == {}  # stated: honoured
    assert untagged_color_fill(_info()) == {}  # the probe does not report tags
    assert untagged_color_fill(_info(pix_fmt="gbrp", color_space="unknown", color_range="unknown")) == {}


def test_plan_tags_before_any_conversion_and_is_unchanged_otherwise() -> None:
    plain = build_canonicalization_plan("in.mp4", "out.y4m")
    filled = build_canonicalization_plan(
        "in.mp4", "out.y4m", scale_width=64, scale_height=48, fill_color_space=True, fill_color_range=True
    )
    vf = filled[filled.index("-vf") + 1].split(",")
    assert vf[0] == "setparams=colorspace=bt709:range=tv"
    assert vf[1].startswith("scale=")
    assert build_canonicalization_plan("in.mp4", "out.y4m", fill_color_space=False) == plain
    assert not any("setparams" in a for a in plain)


def _frames(path: Path) -> np.ndarray:
    rest = path.read_bytes().partition(b"\n")[2]
    return np.frombuffer(b"".join(rest.split(b"FRAME\n")[1:]), np.uint8)


@requires_ffmpeg
@pytest.mark.skipif(FFPROBE is None, reason="ffprobe not available")
def test_untagged_output_canonicalizes_to_the_tagged_pixels(tmp_path: Path) -> None:
    def ff(*args: str) -> None:
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)

    tagged = tmp_path / "tagged.mkv"
    ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=0.5",
       "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
       "-c:v", "ffv1", "-color_range", "tv", "-colorspace", "bt709", str(tagged))
    untagged = tmp_path / "untagged.mkv"
    ff("-i", str(tagged), "-vf", "setparams=range=unknown:colorspace=unknown",
       "-c:v", "ffv1", str(untagged))
    probe = FfprobeBackend(FFPROBE, timeout=30.0)
    executor = CanonicalizeExecutor(FFMPEG, timeout=60.0)
    outputs = {}
    for name, src in (("tagged", tagged), ("untagged", untagged)):
        info = probe.probe(str(src))
        out = tmp_path / f"{name}.y4m"
        executor.run(build_canonicalization_plan(str(src), str(out), **untagged_color_fill(info)))
        outputs[name] = _frames(out)
    assert probe.probe(str(untagged)).color_space == "unknown"
    assert probe.probe(str(tagged)).color_space == "bt709"
    assert np.array_equal(outputs["tagged"], outputs["untagged"])
