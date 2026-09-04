"""Reference-miner unit tests: /v1/task contract (+ the deprecated /task alias),
the GET /warrant TaskWarrant probe, typed errors, backend behavior."""

import pytest

from vidaio.miner import (
    BackendError,
    FfmpegCompressBackend,
    FfmpegUpscaleBackend,
    sha256_file,
)
from vidaio.miner.config import MinerConfig

from miner_support import FFMPEG, FFPROBE, generate_clip

GOOD_DIGEST = "0" * 64  # valid pattern, wrong value


async def test_legacy_path_routes_are_absent_by_default(tmp_path) -> None:
    import httpx
    from prometheus_client import CollectorRegistry

    from vidaio.miner.service import _Metrics, create_app

    cfg = MinerConfig(work_dir=tmp_path / "work", ffmpeg_path=FFMPEG)
    app = create_app(cfg, _Metrics(CollectorRegistry()))
    paths = {route.path for route in app.routes}
    assert "/v1/task/artifact" in paths
    assert "/v1/task" not in paths
    assert "/task" not in paths
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as client:
        assert (await client.post("/v1/task", json={})).status_code == 404
        assert (await client.post("/task", json={})).status_code == 404


async def test_task_digest_mismatch_422(miner_client, tmp_path) -> None:
    clip = generate_clip(tmp_path / "in.mp4")
    resp = await miner_client.post(
        "/v1/task",
        json={
            "task_id": "t1",
            "track": "compression",
            "input_path": str(clip),
            "input_digest": GOOD_DIGEST,
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "input_digest_mismatch"


async def test_task_missing_input_422(miner_client, tmp_path) -> None:
    resp = await miner_client.post(
        "/v1/task",
        json={
            "task_id": "t2",
            "track": "compression",
            "input_path": str(tmp_path / "nope.mp4"),
            "input_digest": GOOD_DIGEST,
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "input_not_found"


async def test_task_unknown_track_422(miner_client, tmp_path) -> None:
    clip = generate_clip(tmp_path / "in.mp4")
    resp = await miner_client.post(
        "/v1/task",
        json={
            "task_id": "t3",
            "track": "interpolation",
            "input_path": str(clip),
            "input_digest": sha256_file(clip),
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "unknown_track"


async def test_task_deadline_exceeded_504(miner_client, tmp_path) -> None:
    clip = generate_clip(tmp_path / "in.mp4")
    resp = await miner_client.post(
        "/v1/task",
        json={
            "task_id": "t4",
            "track": "compression",
            "input_path": str(clip),
            "input_digest": sha256_file(clip),
            "deadline_seconds": 0.001,  # impossible: typed timeout, not a crash
        },
    )
    assert resp.status_code == 504
    assert resp.json()["detail"]["code"] == "deadline_exceeded"


async def test_task_bad_upscale_params_422(miner_client, tmp_path) -> None:
    clip = generate_clip(tmp_path / "in.mp4")
    resp = await miner_client.post(
        "/v1/task",
        json={
            "task_id": "t5",
            "track": "upscaling",
            "input_path": str(clip),
            "input_digest": sha256_file(clip),
            "params": {"upscale_factor": 3},
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "bad_params"


# --- route canonicalization + warrant probe -------------------------------------------


async def test_deprecated_task_alias_responds_identically(miner_client, tmp_path) -> None:
    """POST /task is the deprecated alias of POST /v1/task: same body, same code."""
    clip = generate_clip(tmp_path / "in.mp4")
    body = {
        "task_id": "alias-1",
        "track": "compression",
        "input_path": str(clip),
        "input_digest": sha256_file(clip),
        "deadline_seconds": 120,
    }
    canonical = await miner_client.post("/v1/task", json=body)
    alias = await miner_client.post("/task", json={**body, "task_id": "alias-2"})
    assert canonical.status_code == alias.status_code == 200, alias.text
    # Same handler: identical output bytes for identical work (only the task id
    # and the advisory processing time differ).
    a, b = canonical.json(), alias.json()
    assert a["output_digest"] == b["output_digest"]
    assert (a["task_id"], b["task_id"]) == ("alias-1", "alias-2")

    # Typed errors travel the alias unchanged too.
    bad = await miner_client.post(
        "/task",
        json={
            "task_id": "alias-3",
            "track": "interpolation",
            "input_path": str(clip),
            "input_digest": sha256_file(clip),
        },
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "unknown_track"


async def test_warrant_declares_the_configured_pool(miner_client, miner) -> None:
    resp = await miner_client.get("/warrant")
    assert resp.status_code == 200
    assert resp.json() == {"track": miner.cfg.warrant_track}
    assert miner.cfg.warrant_track == "compression"  # the config default


async def test_warrant_serves_the_declared_track(tmp_path) -> None:
    """One pool per identity: the instance declares it, the probe reports it."""
    import httpx
    from prometheus_client import CollectorRegistry

    from vidaio.miner.service import _Metrics, create_app

    cfg = MinerConfig(
        work_dir=tmp_path / "work", ffmpeg_path=FFMPEG, warrant_track="upscaling"
    )
    app = create_app(cfg, _Metrics(CollectorRegistry()))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://miner") as c:
        assert (await c.get("/warrant")).json() == {"track": "upscaling"}


def test_unknown_warrant_track_is_refused_at_config_load() -> None:
    # The validator never defaults an unclassified miner into a track, so a
    # typo'd pool must fail loudly at startup rather than silently skip rounds.
    with pytest.raises(ValueError, match="not a known pool"):
        MinerConfig(warrant_track="upscalling-v2-turbo")


# --- backends directly ----------------------------------------------------------------


def test_compress_backend_shrinks_high_bitrate_input(tmp_path) -> None:
    # near-lossless input (crf 0) so the shipped CRF-22 re-encode is guaranteed smaller
    fat = generate_clip(tmp_path / "fat.mp4", crf=0)
    out = tmp_path / "out.mp4"
    FfmpegCompressBackend(FFMPEG, 120).process(str(fat), str(out), {})
    assert out.is_file()
    assert out.stat().st_size < fat.stat().st_size


def test_upscale_backend_rejects_unsupported_factor(tmp_path) -> None:
    clip = generate_clip(tmp_path / "in.mp4")
    backend = FfmpegUpscaleBackend(FFMPEG, 120)
    with pytest.raises(BackendError):
        backend.process(str(clip), str(tmp_path / "o.mp4"), {"upscale_factor": 3})
    with pytest.raises(BackendError):
        backend.process(str(clip), str(tmp_path / "o.mp4"), {})


def _probe_codec(path) -> tuple[str, str]:
    import json
    import subprocess

    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return stream["codec_name"], stream["pix_fmt"]


@pytest.mark.parametrize("codec,expected", [("H264", "h264"), ("HEVC", "hevc"), ("AV1", "av1"), ("VP9", "vp9")])
def test_compress_backend_honours_target_codec(tmp_path, codec, expected) -> None:
    clip = generate_clip(tmp_path / "in.mp4")
    out = tmp_path / f"{expected}.mp4"
    FfmpegCompressBackend(FFMPEG, 120, preset="ultrafast").process(
        str(clip), str(out), {"target_codec": codec, "compression_type": "Medium", "codec_mode": "CRF"}
    )
    assert _probe_codec(out) == (expected, "yuv420p")


def test_compress_backend_vbr_and_tiers_change_the_output(tmp_path) -> None:
    fat = generate_clip(tmp_path / "fat.mp4", crf=0, size="320x240")
    high, low, vbr = (tmp_path / n for n in ("high.mp4", "low.mp4", "vbr.mp4"))
    be = FfmpegCompressBackend(FFMPEG, 120, preset="ultrafast")
    be.process(str(fat), str(high), {"target_codec": "H264", "compression_type": "High"})
    be.process(str(fat), str(low), {"target_codec": "H264", "compression_type": "Low"})
    be.process(str(fat), str(vbr), {"target_codec": "H264", "codec_mode": "VBR", "target_bitrate": 150_000})
    assert low.stat().st_size < high.stat().st_size < fat.stat().st_size
    assert vbr.stat().st_size < fat.stat().st_size


def test_compress_backend_rejects_unknown_codec(tmp_path) -> None:
    clip = generate_clip(tmp_path / "in.mp4")
    with pytest.raises(BackendError, match="unsupported target_codec"):
        FfmpegCompressBackend(FFMPEG, 120).process(str(clip), str(tmp_path / "o.mp4"), {"target_codec": "PRORES"})
