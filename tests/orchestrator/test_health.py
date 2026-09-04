"""HEALTH MUST NOT LIE.

Two ways it did:

1. `_check_db` used the orchestrator's OWN sqlite connection. Health checks run on
   the HealthServer's HTTP thread, and that connection belongs to the event loop's
   thread — so the check either raised ProgrammingError (reporting "DB down" when
   the DB was fine) or, with check_same_thread off, poked a connection in the
   middle of another thread's transaction. Neither answers the question.

2. Nothing watched the control-API serve task. A bind failure left a live process
   with no control plane, reporting `status: ok` to every probe, with nothing to
   restart it.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest

from orchestrator_support import T0, build_manifest


def _payload(orch):
    ok, payload = orch.health.health_payload()
    return ok, payload["checks"]


def test_health_checks_answer_from_another_thread(orchestrator_factory, fixture_repos):
    """The regression: the checks run on the HealthServer thread, not this one."""
    orch = orchestrator_factory(repos=fixture_repos)
    orch.create_competition(build_manifest(), T0)

    result: dict[str, object] = {}

    def probe() -> None:
        try:
            result["value"] = _payload(orch)
        except BaseException as exc:  # noqa: BLE001 - the whole point is to see it
            result["error"] = exc

    thread = threading.Thread(target=probe, name="health-probe")
    thread.start()
    thread.join(timeout=30)

    assert "error" not in result, result.get("error")
    ok, checks = result["value"]  # type: ignore[misc]
    assert checks["db"] is True
    assert ok is True


def test_the_db_check_uses_its_own_connection_not_the_services(
    orchestrator_factory, fixture_repos
):
    """It must report on the DATABASE, not on the service's connection object."""
    orch = orchestrator_factory(repos=fixture_repos)
    orch.conn.close()  # the loop's connection is gone; the database is not

    assert orch._check_db() is True

    # And a genuinely broken database DOES fail the check.
    with open(orch.core.db_path, "wb") as fh:
        fh.write(b"not a sqlite database at all")
    with pytest.raises(sqlite3.DatabaseError):
        orch._check_db()
    _ok, checks = _payload(orch)
    assert checks["db"] is False


def test_a_db_check_never_disturbs_a_live_transaction(
    orchestrator_factory, fixture_repos
):
    """A short-lived reader must not join (or block) the service's own writes."""
    orch = orchestrator_factory(repos=fixture_repos)
    orch.create_competition(build_manifest(), T0)
    orch.conn.execute("BEGIN IMMEDIATE")
    try:
        assert orch._check_db() is True
    finally:
        orch.conn.execute("ROLLBACK")
    assert orch.conn.in_transaction is False


# ---- control-API monitoring --------------------------------------------------------


async def test_a_control_api_that_dies_flips_health_and_stops_the_service(
    orchestrator_factory, fixture_repos
):
    """A live process with no control plane is not healthy."""
    orch = orchestrator_factory(repos=fixture_repos, control_token="tok")
    assert orch.control_app is not None
    assert _payload(orch)[1]["control_api"] is True

    class DeadServer:
        should_exit = False

        async def serve(self) -> None:
            raise OSError("address already in use")

    orch._create_control_server = lambda: DeadServer()  # type: ignore[method-assign]
    await asyncio.wait_for(orch.run(), timeout=30)

    ok, checks = _payload(orch)
    assert checks["control_api"] is False
    assert ok is False
    assert orch.stopping.is_set()  # requested stop, so a supervisor restarts it


async def test_a_clean_shutdown_does_not_flip_the_control_health(
    orchestrator_factory, fixture_repos
):
    """Stopping on purpose is not a failure — only an UNEXPECTED exit is."""
    orch = orchestrator_factory(repos=fixture_repos, control_token="tok")

    class QuietServer:
        should_exit = False

        async def serve(self) -> None:
            while not self.should_exit:
                await asyncio.sleep(0.01)

    orch._create_control_server = lambda: QuietServer()  # type: ignore[method-assign]
    runner = asyncio.create_task(orch.run())
    await asyncio.sleep(0.05)
    orch.request_stop()
    await asyncio.wait_for(runner, timeout=30)

    assert _payload(orch)[1]["control_api"] is True


async def test_without_a_control_token_the_api_axis_stays_healthy(
    orchestrator_factory, fixture_repos
):
    """No control app configured means nothing to be unhealthy about."""
    orch = orchestrator_factory(repos=fixture_repos)
    assert orch.control_app is None
    assert _payload(orch)[1]["control_api"] is True

    runner = asyncio.create_task(orch.run())
    await asyncio.sleep(0.05)
    orch.request_stop()
    await asyncio.wait_for(runner, timeout=30)
    assert _payload(orch)[1]["control_api"] is True
