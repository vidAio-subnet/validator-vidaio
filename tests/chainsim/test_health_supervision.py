"""Chainsim health tells the truth — from the health THREAD, and about the API.

Two ways the simulator used to report "ok" while being useless:

  * `sim_db` executed on the service's own SQLite connection. Health checks run
    on the HealthServer thread, so that check shares a connection (and its
    transaction state) with the event loop, and it passes even when the database
    FILE has gone — sqlite keeps serving an already-open, deleted inode. A
    short-lived connection per check answers the question actually being asked.
  * The uvicorn task was created and never watched. A bind failure makes uvicorn
    exit (SystemExit) after logging; unwatched, that leaves a process reporting
    healthy with nothing listening on its API port at all.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import uvicorn


# --- sim_db: another thread, and a live answer ------------------------------------


def test_the_db_check_answers_from_another_thread(sim):
    """This is where it really runs: the HealthServer's own thread."""
    result: dict = {}

    def _probe() -> None:
        result["payload"] = sim.health.health_payload()

    thread = threading.Thread(target=_probe, name="health-probe")
    thread.start()
    thread.join(timeout=10)

    ok, payload = result["payload"]
    assert ok is True
    assert payload["checks"]["sim_db"] is True
    assert payload["checks"]["http_api"] is True


def test_the_db_check_does_not_use_the_service_connection(sim):
    """Structural: it must open its own connection rather than borrow one."""

    class _Exploding:
        def execute(self, *a, **k):
            raise AssertionError("the health check used the service connection")

    real, sim._conn = sim._conn, _Exploding()
    try:
        payload = sim.health.health_payload()[1]
    finally:
        sim._conn = real
    assert payload["checks"]["sim_db"] is True


def test_the_db_check_goes_red_when_the_database_file_is_gone(sim):
    """A LIVE check: an already-open handle would keep reporting health forever."""
    db = Path(sim.config.db_path)
    for path in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")):
        if path.exists():
            path.unlink()

    ok, payload = sim.health.health_payload()

    assert payload["checks"]["sim_db"] is False
    assert ok is False


def test_concurrent_probes_do_not_poison_each_other(sim):
    verdicts: list[bool] = []
    errors: list[BaseException] = []

    def _probe() -> None:
        try:
            verdicts.append(sim.health.health_payload()[1]["checks"]["sim_db"])
        except BaseException as exc:  # pragma: no cover - the failure we forbid
            errors.append(exc)

    threads = [threading.Thread(target=_probe) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert verdicts == [True] * 8


# --- the API task is supervised ----------------------------------------------------


async def test_a_bind_failure_flips_http_api_and_stops_the_sim(
    sim, monkeypatch, caplog
):
    async def _cannot_bind(self) -> None:
        raise SystemExit(1)  # what uvicorn does when the port is taken

    monkeypatch.setattr(uvicorn.Server, "serve", _cannot_bind)

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(sim.run(), timeout=10)

    payload = sim.health.health_payload()[1]
    assert payload["checks"]["http_api"] is False
    assert sim.stopping.is_set()
    assert any(
        r.levelno >= logging.ERROR and "HTTP API exited unexpectedly" in r.getMessage()
        for r in caplog.records
    )


async def test_a_crashed_api_task_is_treated_the_same(sim, monkeypatch):
    async def _crash(self) -> None:
        raise RuntimeError("uvicorn fell over")

    monkeypatch.setattr(uvicorn.Server, "serve", _crash)

    await asyncio.wait_for(sim.run(), timeout=10)

    assert sim.health.health_payload()[1]["checks"]["http_api"] is False


async def test_a_normal_shutdown_leaves_http_api_green(sim, monkeypatch):
    """The ordinary stop path must not look like a failure."""

    async def _serve_until_asked(self) -> None:
        while not self.should_exit:
            await asyncio.sleep(0.01)

    monkeypatch.setattr(uvicorn.Server, "serve", _serve_until_asked)

    async def _stop_soon() -> None:
        await asyncio.sleep(0.05)
        sim.request_stop()

    await asyncio.wait_for(asyncio.gather(sim.run(), _stop_soon()), timeout=10)

    assert sim.health.health_payload()[1]["checks"]["http_api"] is True
