"""Public-endpoint hardening for the reference miner.

The miner port is reachable by anyone: an unauthenticated caller must not be
able to write outside work_dir, start unbounded compute, or grow the disk
forever — and a miner whose HTTP server died must not keep reporting healthy.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import httpx
import pytest
from prometheus_client import CollectorRegistry

from vidaio.miner import Miner, sha256_file
from vidaio.miner.config import MinerConfig
from vidaio.miner.service import (
    TaskDirEscape,
    _Metrics,
    create_app,
    resolve_task_dir,
    sweep_task_dirs,
)

from miner_support import FFMPEG, generate_clip


def body(clip: Path, task_id: str, **over) -> dict:
    return {
        "task_id": task_id,
        "track": "compression",
        "input_path": str(clip),
        "input_digest": sha256_file(clip),
        "deadline_seconds": 120,
        **over,
    }


# ---- path escape ---------------------------------------------------------------

#: Everything an adversary can put in a JSON string to try to leave work_dir.
ESCAPES = [
    "../escaped",
    "../../../../tmp/escaped",
    "..",
    ".",
    "/tmp/escaped-absolute",
    "/etc/passwd",
    "sub/dir",
    "sub\\dir",
    "",
    "a" * 129,
    "\uff0e\uff0e/escaped",  # FULLWIDTH FULL STOP look-alikes
    "\u2024\u2024/escaped",  # ONE DOT LEADER look-alikes
    "task\x00id",  # NUL truncation attempt
    "task\nid",
    "task id",
    "caf\u00e9",  # non-ASCII, even when harmless, is refused
]


@pytest.mark.parametrize("task_id", ESCAPES)
async def test_task_id_path_escape_is_422_and_writes_nothing(
    miner_client, miner: Miner, tmp_path: Path, task_id: str
) -> None:
    clip = generate_clip(tmp_path / "in.mp4")
    outside = sorted(p.name for p in tmp_path.iterdir())
    work = Path(miner.cfg.work_dir)

    resp = await miner_client.post("/v1/task", json=body(clip, task_id))

    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["code"] == "invalid_task_id"
    # nothing created anywhere: not under work_dir, not beside it
    assert list(work.iterdir()) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == outside
    assert not Path("/tmp/escaped-absolute").exists()


def test_resolve_task_dir_rejects_escapes_and_accepts_ids(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    for bad in ESCAPES:
        with pytest.raises(TaskDirEscape):
            resolve_task_dir(work, bad)
    for good in ("t1", "job_42", "a-b-c", "A" * 128, "uuid-1234:5HotKey"):
        assert resolve_task_dir(work, good).parent == work.resolve()


def test_resolve_task_dir_survives_a_symlinked_work_dir(tmp_path: Path) -> None:
    """The prefix check is on RESOLVED paths, so a symlinked root still holds."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    assert resolve_task_dir(link, "t1") == real.resolve() / "t1"
    with pytest.raises(TaskDirEscape):
        resolve_task_dir(link, "../real2")


# ---- concurrency ---------------------------------------------------------------


async def test_saturated_miner_answers_429_without_queueing(
    tmp_path: Path,
) -> None:
    cfg = MinerConfig(
        work_dir=tmp_path / "work", ffmpeg_path=FFMPEG, max_concurrent_tasks=1,
        enable_legacy_path_routes=True,
    )
    app = create_app(cfg, _Metrics(CollectorRegistry()))
    clip = generate_clip(tmp_path / "in.mp4", duration=4.0, crf=0)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner", timeout=180
    ) as c:
        first = asyncio.create_task(c.post("/v1/task", json=body(clip, "slot-one")))
        # Wait until the first request has actually ACQUIRED the only slot,
        # not a fixed number of yields (which raced): once locked, a second
        # request is guaranteed to be shed.
        for _ in range(10_000):
            if app.state.task_slots.locked():
                break
            await asyncio.sleep(0)
        assert app.state.task_slots.locked()
        second = await c.post("/v1/task", json=body(clip, "slot-two"))
        assert second.status_code == 429
        assert second.json()["detail"]["code"] == "busy"
        assert (await first).status_code == 200
        # the slot is released again once the first task finished
        assert (await c.post("/v1/task", json=body(clip, "slot-three"))).status_code == 200


# ---- input bound ---------------------------------------------------------------


async def test_oversize_input_is_refused_before_any_work(tmp_path: Path) -> None:
    cfg = MinerConfig(
        work_dir=tmp_path / "work", ffmpeg_path=FFMPEG, max_input_bytes=1024,
        enable_legacy_path_routes=True,
    )
    app = create_app(cfg, _Metrics(CollectorRegistry()))
    clip = generate_clip(tmp_path / "in.mp4")
    assert clip.stat().st_size > 1024
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner", timeout=60
    ) as c:
        r = await c.post("/v1/task", json=body(clip, "too-big"))
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "input_too_large"
    assert not (tmp_path / "work").exists()  # refused before any dir was made


# ---- auth ----------------------------------------------------------------------


async def test_api_token_is_required_when_configured(tmp_path: Path) -> None:
    cfg = MinerConfig(
        work_dir=tmp_path / "work", ffmpeg_path=FFMPEG, api_token="s3cret-token",
        enable_legacy_path_routes=True,
    )
    app = create_app(cfg, _Metrics(CollectorRegistry()))
    clip = generate_clip(tmp_path / "in.mp4")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner", timeout=180
    ) as c:
        anon = await c.post("/v1/task", json=body(clip, "no-token"))
        assert anon.status_code == 401
        assert anon.json()["detail"]["code"] == "unauthorized"
        wrong = await c.post(
            "/v1/task",
            json=body(clip, "bad-token"),
            headers={"X-Miner-Token": "s3cret-toked"},
        )
        assert wrong.status_code == 401
        ok = await c.post(
            "/v1/task",
            json=body(clip, "good-token"),
            headers={"X-Miner-Token": "s3cret-token"},
        )
        assert ok.status_code == 200, ok.text


async def test_no_token_configured_stays_open_for_local_runs(miner_client, tmp_path) -> None:
    clip = generate_clip(tmp_path / "in.mp4")
    r = await miner_client.post("/v1/task", json=body(clip, "open-run"))
    assert r.status_code == 200


# ---- task dir lifecycle --------------------------------------------------------


async def test_failed_task_dir_is_removed_immediately(miner_client, miner, tmp_path) -> None:
    clip = generate_clip(tmp_path / "in.mp4")
    r = await miner_client.post(
        "/v1/task", json=body(clip, "doomed", track="upscaling", params={"upscale_factor": 3})
    )
    assert r.status_code == 422 and r.json()["detail"]["code"] == "bad_params"
    assert not (Path(miner.cfg.work_dir) / "doomed").exists()


async def test_successful_output_survives_its_ttl_then_is_swept(
    miner_client, miner: Miner, tmp_path: Path
) -> None:
    clip = generate_clip(tmp_path / "in.mp4")
    r = await miner_client.post("/v1/task", json=body(clip, "keepme"))
    assert r.status_code == 200, r.text
    out = Path(r.json()["output_path"])
    # readable by the caller straight after the response (the wire contract)
    assert out.is_file()
    assert miner.sweep_task_dirs() == 0  # still inside the TTL
    assert out.is_file()
    # age it past the TTL and the reaper takes it
    old = time.time() - (miner.cfg.task_dir_ttl_seconds + 60)
    os.utime(out.parent, (old, old))
    assert miner.sweep_task_dirs() == 1
    assert not out.parent.exists()


async def test_restart_keeps_an_output_the_caller_has_not_read_yet(
    miner_client, miner: Miner, tmp_path: Path
) -> None:
    """The real regression: crash-restart between the response and the read."""
    clip = generate_clip(tmp_path / "in.mp4")
    r = await miner_client.post("/v1/task", json=body(clip, "unread"))
    assert r.status_code == 200, r.text
    out = Path(r.json()["output_path"])
    digest = r.json()["output_digest"]

    restarted = Miner(
        {
            "core": {"data_dir": str(tmp_path / "data2"), "metrics_port": 0},
            "miner": {
                "work_dir": str(miner.cfg.work_dir),
                "ffmpeg_path": FFMPEG,
                "metrics_port": 0,
            },
        }
    )

    assert out.is_file(), "a restart deleted a result still inside its TTL"
    assert sha256_file(out) == digest  # still the bytes the response named
    assert restarted.sweep_task_dirs() == 0


def _age(path: Path, seconds: float) -> None:
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_startup_sweep_clears_expired_leftovers_but_keeps_live_ones(
    tmp_path: Path,
) -> None:
    """A restart must not shorten the TTL a written response already promised.

    The miner answers with a filesystem PATH the caller reads afterwards, so a
    process that dies right after responding leaves a dir that is still valid.
    Startup therefore sweeps at the configured TTL: expired leftovers go, a
    fresh output survives to be read (and dies on a later sweep).
    """
    work = tmp_path / "work"
    (work / "expired").mkdir(parents=True)
    (work / "expired" / "output.mp4").write_bytes(b"stale")
    fresh = work / "just-finished"
    fresh.mkdir()
    (fresh / "output.mp4").write_bytes(b"a result the gateway has not hashed yet")
    ttl = MinerConfig().task_dir_ttl_seconds
    _age(work / "expired", ttl + 60)

    miner = Miner(
        {
            "core": {"data_dir": str(tmp_path / "data")},
            "miner": {"work_dir": str(work), "ffmpeg_path": FFMPEG},
        }
    )

    assert not (work / "expired").exists()
    assert (fresh / "output.mp4").read_bytes().startswith(b"a result")
    assert miner.cfg.retain_task_dirs is False
    # ...and it is not immortal: once it ages out the reaper takes it.
    _age(fresh, ttl + 60)
    assert miner.sweep_task_dirs() == 1
    assert not fresh.exists()


def test_force_sweep_stays_available_but_is_never_the_startup_path(
    tmp_path: Path,
) -> None:
    """`force` (admin/tests) still ignores the TTL — startup must not use it."""
    work = tmp_path / "work"
    (work / "live").mkdir(parents=True)
    miner = Miner(
        {
            "core": {"data_dir": str(tmp_path / "data")},
            "miner": {"work_dir": str(work), "ffmpeg_path": FFMPEG},
        }
    )
    assert (work / "live").is_dir()  # startup honoured the TTL
    assert miner.sweep_task_dirs(force=True) == 1
    assert not (work / "live").exists()


def test_retain_task_dirs_disables_all_sweeping(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "leftover").mkdir(parents=True)
    miner = Miner(
        {
            "core": {"data_dir": str(tmp_path / "data")},
            "miner": {
                "work_dir": str(work),
                "ffmpeg_path": FFMPEG,
                "retain_task_dirs": True,
            },
        }
    )
    assert (work / "leftover").is_dir()
    assert miner.sweep_task_dirs(force=True) == 0


def test_sweeper_ignores_foreign_entries(tmp_path: Path) -> None:
    """It only deletes things shaped like its own task dirs, never anything else."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "task1").mkdir()
    (work / "not a task id").mkdir()
    (work / "loose-file").write_bytes(b"x")
    outside = tmp_path / "precious"
    outside.mkdir()
    (work / "evil").symlink_to(outside)
    assert sweep_task_dirs(work, ttl_seconds=0.0, now=time.time()) == 1
    assert not (work / "task1").exists()
    assert (work / "not a task id").is_dir()
    assert (work / "loose-file").is_file()
    assert outside.is_dir() and (work / "evil").is_symlink()


# ---- lifecycle / health (finding #22) ------------------------------------------


class _FailingServer:
    """Stands in for a uvicorn server whose bind fails."""

    def __init__(self) -> None:
        self.should_exit = False

    async def serve(self) -> None:
        raise OSError("[errno 48] address already in use")


async def test_bind_failure_flips_health_and_stops_the_service(tmp_path: Path) -> None:
    miner = Miner(
        {
            "core": {"data_dir": str(tmp_path / "data"), "metrics_port": 0},
            "miner": {
                "work_dir": str(tmp_path / "work"),
                "ffmpeg_path": FFMPEG,
                "metrics_port": 0,
            },
        }
    )
    ok, payload = miner.health.health_payload()
    assert payload["checks"]["http_api"] is True

    miner._create_http_server = lambda: _FailingServer()  # type: ignore[method-assign]
    await asyncio.wait_for(miner.run(), timeout=5)

    ok, payload = miner.health.health_payload()
    assert payload["checks"]["http_api"] is False
    assert ok is False and payload["status"] == "degraded"
    assert miner.stopping.is_set()  # supervisor gets its restart signal
