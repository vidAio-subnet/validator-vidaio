"""Modal worker's pure contract: bounded metadata and fresh-resource source shape."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

import pytest

from vidaio.miner.gpu_worker import (
    SUPPORTED_VARIANTS,
    GpuTaskMetadata,
    GpuTransformError,
    VideoInfo,
    _detail_filter,
    _target,
    decode_gpu_metadata,
)
from vidaio.miner.remote_gpu import GPU_PROTOCOL_VERSION


def _header(**updates: object) -> str:
    payload = {
        "protocol": GPU_PROTOCOL_VERSION,
        "track": "upscaling",
        "solution_variant": "balanced",
        "input_digest": hashlib.sha256(b"input").hexdigest(),
        "input_size": 5,
        "deadline_seconds": 30.0,
        "params": {"upscale_factor": 2},
    }
    payload.update(updates)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def test_gpu_metadata_accepts_both_tracks_and_all_rank_variants() -> None:
    for track in ("compression", "upscaling"):
        for variant in SUPPORTED_VARIANTS:
            parsed = decode_gpu_metadata(
                _header(track=track, solution_variant=variant)
            )
            assert parsed.track == track
            assert parsed.solution_variant == variant


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol", "v0", "protocol"),
        ("track", "interpolation", "track"),
        ("solution_variant", "secret-fourth-profile", "variant"),
        ("input_digest", "ABC", "sha256"),
        ("input_size", 0, "greater than 0"),
        ("deadline_seconds", 601, "less than or equal to 600"),
    ],
)
def test_gpu_metadata_fails_closed(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode_gpu_metadata(_header(**{field: value}))


def test_gpu_metadata_header_has_a_hard_decode_bound() -> None:
    with pytest.raises(ValueError, match="too large"):
        decode_gpu_metadata("A" * 100_000)


def test_upscaling_target_contract_and_pixel_budget() -> None:
    info = VideoInfo(width=320, height=180, fps=24.0, frame_count=48)
    metadata = GpuTaskMetadata.model_validate(
        {
            "protocol": GPU_PROTOCOL_VERSION,
            "track": "upscaling",
            "solution_variant": "quality",
            "input_digest": "0" * 64,
            "input_size": 1,
            "deadline_seconds": 30,
            "params": {"upscale_factor": 4},
        }
    )
    assert _target(info, metadata) == (1280, 720)
    bad = metadata.model_copy(update={"params": {"upscale_factor": 3}})
    with pytest.raises(GpuTransformError, match="upscale_factor"):
        _target(info, bad)


def test_profile_detail_kernel_is_deterministic_and_materially_distinct() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.tensor(
        [[[[0.0, 0.25, 1.0], [0.5, 0.75, 0.25], [1.0, 0.0, 0.5]]]],
        dtype=torch.float32,
    )
    quality = _detail_filter(tensor, 0.10)
    compact = _detail_filter(tensor, -0.04)
    assert torch.equal(quality, _detail_filter(tensor, 0.10))
    assert quality.shape == compact.shape == tensor.shape
    assert not torch.equal(quality, compact)
    assert float(quality.min()) >= 0.0 and float(quality.max()) <= 1.0
    assert float(compact.min()) >= 0.0 and float(compact.max()) <= 1.0


def test_modal_source_is_gpu_only_prefixed_and_has_no_persistent_lookup() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "deploy"
        / "modal"
        / "vidaio_next_gpu_miner.py"
    )
    source = path.read_text()
    ast.parse(source)
    assert "APP_NAME = DEPLOYMENT_LABEL" in source
    assert 'RESOURCE_PREFIX = "vidaio-next-"' in source
    assert "VIDAIO_NEXT_FRESH_CREATION_CONFIRMATION" in source
    assert 'name="vidaio-next-gpu-miner-worker"' in source
    assert '"L4"' in source
    assert "gpu=GPU_TYPE" in source
    assert "CUDA is unavailable; CPU fallback is forbidden" in source
    assert "@modal.asgi_app()" in source
    assert "@modal.concurrent(max_inputs=1)" in source
    assert "modal.Volume.from_name" not in source
    assert "modal.Dict.from_name" not in source
    assert "modal.Queue.from_name" not in source
    assert "modal.Function.from_name" not in source
    assert "CPU-only external scoring worker/auditor" in source
