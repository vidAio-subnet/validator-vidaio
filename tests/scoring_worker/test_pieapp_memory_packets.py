"""The bounded CPU backbone must preserve the complete real upscaling packet."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from tests.scoring_worker.conftest import (
    FFMPEG,
    FFPROBE,
    ClipPair,
    _ffmpeg,
    requires_media_tools,
    score_request_body,
    sha256_file,
    worker_scorer_version,
)
from vidaio.scoring import ScoringConfig
from vidaio.scoring import backends_real
from vidaio.scoring_worker import ScoringWorkerConfig
from vidaio.scoring_worker.service import _score_sync, real_backends
from vidaio.services.protocol import ScoreRequest


@requires_media_tools
def test_upscaling_packet_matches_unchunked_piq(
    clips: ClipPair, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("piq")
    pytest.importorskip("cv2")
    weights = Path(torch.hub.get_dir()) / "checkpoints" / backends_real.PIEAPP_WEIGHTS_FILENAME
    if not weights.is_file():
        pytest.skip("pinned PieAPP weights must already be cached")
    backends_real._verify_pieapp_weights(weights)

    # Eight patches per fixture frame exercise two chunks without making the
    # legacy comparison allocate the full-resolution convolution workspace.
    monkeypatch.setattr(backends_real, "_PIEAPP_PATCH_BATCH_SIZE", 4)
    previous_threads = torch.get_num_threads()
    previous_nnpack = torch.backends.nnpack.set_flags(False)
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", False)
    torch.set_num_threads(1)
    try:
        _compare_packets(clips, tmp_path, monkeypatch)
    finally:
        torch.backends.nnpack.set_flags(*previous_nnpack)
        torch.set_num_threads(previous_threads)


def _compare_packets(
    clips: ClipPair, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    miner_input = tmp_path / "miner-input.mp4"
    _ffmpeg(
        "-i", clips.reference, "-vf", "scale=80:60:flags=bicubic",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-y", str(miner_input),
    )
    request = ScoreRequest.model_validate(score_request_body(
        track="upscaling",
        reference=clips.reference,
        reference_digest=clips.reference_digest,
        output=clips.candidate,
        output_digest=clips.candidate_digest,
        miner_input=str(miner_input),
        miner_input_digest=sha256_file(miner_input),
        params={"upscale_factor": 2},
    ))
    config = ScoringWorkerConfig(
        work_dir=tmp_path / "work", ffmpeg_path=FFMPEG, ffprobe_path=FFPROBE,
        request_timeout=120, subprocess_timeout=60,
    )
    scoring = ScoringConfig()
    current = real_backends(config, scoring_config=scoring)
    legacy_pieapp = current.pieapp.clone()
    with monkeypatch.context() as old:
        old.setattr(backends_real, "_bounded_pieapp_model", lambda model, torch: model)
        legacy_pieapp.preload()
    current.pieapp.preload()
    legacy = replace(current, pieapp=legacy_pieapp)
    version = worker_scorer_version(config, scoring)

    before = _score_sync(request, config, scoring, legacy, version)
    after = _score_sync(request, config, scoring, current, version)
    assert before.breakdown is not None and before.breakdown.kind == "upscaling"
    assert before.metrics["pieapp"] is not None
    assert before.metrics["vmaf"] is not None
    assert before.metrics["vmaf_secondary"] is not None
    assert before.canonical_content_digest is not None
    assert before.content_fingerprint is not None
    assert before.pieapp_start_frame is not None
    assert before.skips == []
    before_json, after_json = before.to_json(), after.to_json()
    assert before_json == after_json
    assert hashlib.sha256(before_json.encode()).digest() == hashlib.sha256(after_json.encode()).digest()
