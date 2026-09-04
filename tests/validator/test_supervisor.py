"""Supervisor behaviour: restart on crash, backoff gating, crash-loop parking,
clean-exit handling, SIGTERM fan-out. Children are tiny dummy callables in a
generated module (spawn ctx re-imports it via the propagated sys.path).

THE EXIT-CODE CONTRACT is pinned here from both ends, with
children built on the REAL BaseService rather than bare sys.exit calls:

    exit 0        deliberate stop  -> stay down
    exit non-zero crash            -> restart, then park

That asymmetry is why a service whose HTTP API dies must not simply return: for
most of a release it did, and a validator/miner/gateway whose uvicorn task died
looked to this supervisor exactly like an operator-requested shutdown.
"""

from __future__ import annotations

import textwrap
import time

import pytest

from vidaio.services.base import FATAL_EXIT_CODE
from vidaio.validator import (
    STATE_BACKOFF,
    STATE_PARKED,
    STATE_RUNNING,
    STATE_STOPPED,
    ChildParkedError,
    ChildSpec,
    Supervisor,
)

CHILD_MODULE = textwrap.dedent(
    """
    import sys
    import time

    from vidaio.services import BaseService, run_service


    def sleep_forever(config):
        time.sleep(60)


    def crash_always(config):
        sys.exit(3)


    def exit_clean(config):
        pass


    class _Service(BaseService):
        name = "dummy-service"
        mode = "stop"

        async def run(self):
            if self.mode == "api_dies":
                # What every service's serve-task monitor now does when its
                # uvicorn/control-API task ends on its own.
                self.fail_fatal("http server exited unexpectedly; no API")
                return
            self.request_stop()
            await self.stopping.wait()


    def api_dies(config):
        svc = _Service(config or {"core": {"metrics_port": 0}})
        svc.mode = "api_dies"
        run_service(svc)


    def cooperative_stop(config):
        run_service(_Service(config or {"core": {"metrics_port": 0}}))
    """
)

#: Children built on BaseService must not fight over a fixed metrics port.
SERVICE_CONFIG = {"core": {"metrics_port": 0}}

WAIT = 30.0


@pytest.fixture
def dummy(tmp_path, monkeypatch) -> str:
    mod_dir = tmp_path / "supmods"
    mod_dir.mkdir()
    (mod_dir / "vidaio_dummy_children.py").write_text(CHILD_MODULE)
    # spawn propagates the parent's sys.path to children (preparation data)
    monkeypatch.syspath_prepend(str(mod_dir))
    return "vidaio_dummy_children"


def poll_until(sup: Supervisor, cond, timeout: float = WAIT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sup.poll()
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError(f"condition not reached within {timeout}s: states={sup.states()}")


def test_target_validation_fails_fast():
    with pytest.raises(ModuleNotFoundError):
        Supervisor([("x", "definitely_not_a_module:fn", None)])
    with pytest.raises(ValueError):
        Supervisor([("x", "no-colon-here", None)])
    with pytest.raises(ValueError):
        Supervisor([("x", "os:getpid", None), ("x", "os:getpid", None)])


def test_killed_child_restarts_others_untouched(dummy):
    sup = Supervisor(
        [
            ChildSpec("worker", f"{dummy}:sleep_forever"),
            ("weight-setter", f"{dummy}:sleep_forever", None),
        ],
        max_restarts=5,
        restart_window_seconds=60.0,
        backoff_base_seconds=0.05,
        backoff_max_seconds=0.2,
        join_timeout_seconds=10.0,
    )
    sup.start()
    try:
        poll_until(
            sup,
            lambda: sup.process("worker").is_alive() and sup.process("weight-setter").is_alive(),
        )
        weights_pid = sup.process("weight-setter").pid
        old = sup.process("worker")
        old.terminate()  # simulated crash (nonzero exitcode)
        old.join(WAIT)
        poll_until(
            sup, lambda: sup.process("worker") is not old and sup.process("worker").is_alive()
        )
        assert sup.states()["worker"] == STATE_RUNNING
        assert sup.restart_count("worker") == 1
        # the crash never touched the weight-setter
        assert sup.process("weight-setter").pid == weights_pid
        assert sup.process("weight-setter").is_alive()
    finally:
        sup.shutdown()
    assert not sup.process("worker").is_alive()
    assert not sup.process("weight-setter").is_alive()
    assert set(sup.states().values()) == {STATE_STOPPED}


def test_crash_loop_child_parked_others_keep_running(dummy):
    sup = Supervisor(
        [
            ("looper", f"{dummy}:crash_always", None),
            ("steady", f"{dummy}:sleep_forever", None),
        ],
        max_restarts=2,
        restart_window_seconds=60.0,
        backoff_base_seconds=0.01,
        backoff_max_seconds=0.05,
        join_timeout_seconds=10.0,
    )
    sup.start()
    try:
        poll_until(sup, lambda: sup.states()["looper"] == STATE_PARKED)
        assert sup.restart_count("looper") == 2  # budget spent, then parked
        assert sup.states()["steady"] == STATE_RUNNING
        assert sup.process("steady").is_alive()
        # parked stays parked — poll never resurrects it
        parked_proc = sup.process("looper")
        sup.poll()
        assert sup.process("looper") is parked_proc
    finally:
        sup.shutdown()
    assert sup.states()["looper"] == STATE_PARKED  # parking survives shutdown
    assert sup.states()["steady"] == STATE_STOPPED


def test_fail_on_park_surfaces_a_fatal_group_failure(dummy):
    """Production groups must not remain healthy with a critical child parked."""
    sup = Supervisor(
        [("critical", f"{dummy}:crash_always", None)],
        max_restarts=0,
        fail_on_park=True,
        join_timeout_seconds=10.0,
    )
    sup.start()
    try:
        proc = sup.process("critical")
        assert proc is not None
        proc.join(WAIT)
        with pytest.raises(ChildParkedError, match="critical") as caught:
            sup.poll()
        assert caught.value.child_name == "critical"
        assert caught.value.exitcode == 3
        assert sup.states()["critical"] == STATE_PARKED
    finally:
        sup.shutdown()


def test_fail_on_park_does_not_terminate_group_for_noncritical_child(dummy):
    """A report-only auditor may park without interrupting a healthy weight-setter."""
    sup = Supervisor(
        [
            ChildSpec(
                "report-only-auditor",
                f"{dummy}:crash_always",
                critical=False,
            ),
            ChildSpec("weight-setter", f"{dummy}:sleep_forever"),
        ],
        max_restarts=0,
        fail_on_park=True,
        join_timeout_seconds=10.0,
    )
    sup.start()
    try:
        auditor = sup.process("report-only-auditor")
        assert auditor is not None
        auditor.join(WAIT)
        # This poll parks the auditor.  It must not raise ChildParkedError or
        # fan out shutdown to the independently healthy weight-setter.
        sup.poll()
        assert sup.states()["report-only-auditor"] == STATE_PARKED
        assert sup.states()["weight-setter"] == STATE_RUNNING
        assert sup.process("weight-setter").is_alive()
    finally:
        sup.shutdown()


def test_backoff_gates_restart_on_injected_clock(dummy):
    t = [0.0]
    sup = Supervisor(
        [("c", f"{dummy}:crash_always", None)],
        max_restarts=5,
        restart_window_seconds=1000.0,
        backoff_base_seconds=10.0,
        backoff_max_seconds=100.0,
        join_timeout_seconds=10.0,
        clock=lambda: t[0],
    )
    sup.start()
    try:
        first = sup.process("c")
        first.join(WAIT)
        sup.poll()
        assert sup.states()["c"] == STATE_BACKOFF
        sup.poll()
        assert sup.process("c") is first  # clock unchanged: restart still gated
        t[0] = 10.0  # backoff_base * 2**0 elapsed — now due
        sup.poll()
        assert sup.states()["c"] == STATE_RUNNING
        assert sup.process("c") is not first
        assert sup.restart_count("c") == 1
    finally:
        sup.shutdown()


def test_a_service_whose_api_dies_exits_nonzero_and_is_restarted(dummy):
    """THE round-3 #3 regression, end to end.

    The child is a real BaseService that discovers its HTTP API gone. Before
    `fail_fatal` it flipped health, requested stop and returned — exit 0 — and
    this supervisor correctly filed that as a deliberate shutdown and left it
    down forever, API and all.
    """
    sup = Supervisor(
        [ChildSpec("api-death", f"{dummy}:api_dies", SERVICE_CONFIG)],
        max_restarts=2,
        restart_window_seconds=60.0,
        backoff_base_seconds=0.01,
        backoff_max_seconds=0.05,
        join_timeout_seconds=10.0,
    )
    sup.start()
    try:
        first = sup.process("api-death")
        first.join(WAIT)
        # the exit code IS the message: 70, not 0
        assert first.exitcode == FATAL_EXIT_CODE
        assert first.exitcode != 0
        sup.poll()
        assert sup.states()["api-death"] == STATE_BACKOFF  # scheduled to come back
        poll_until(sup, lambda: sup.restart_count("api-death") >= 1)
        assert sup.process("api-death") is not first
        # and because it keeps failing, it is eventually parked rather than
        # restarted forever
        poll_until(sup, lambda: sup.states()["api-death"] == STATE_PARKED)
        assert sup.restart_count("api-death") == 2
    finally:
        sup.shutdown()


def test_a_cooperative_stop_exits_zero_and_is_not_restarted(dummy):
    """The other half of the contract: request_stop is a real, respected stop."""
    sup = Supervisor(
        [ChildSpec("clean", f"{dummy}:cooperative_stop", SERVICE_CONFIG)],
        max_restarts=5,
        backoff_base_seconds=0.01,
        join_timeout_seconds=10.0,
    )
    sup.start()
    try:
        proc = sup.process("clean")
        proc.join(WAIT)
        assert proc.exitcode == 0
        poll_until(sup, lambda: sup.states()["clean"] == STATE_STOPPED)
        sup.poll()
        assert sup.process("clean") is proc  # never respawned
        assert sup.restart_count("clean") == 0
    finally:
        sup.shutdown()


def test_clean_exit_is_not_restarted(dummy):
    sup = Supervisor(
        [("oneshot", f"{dummy}:exit_clean", None)],
        backoff_base_seconds=0.01,
        join_timeout_seconds=10.0,
    )
    sup.start()
    try:
        proc = sup.process("oneshot")
        proc.join(WAIT)
        poll_until(sup, lambda: sup.states()["oneshot"] == STATE_STOPPED)
        sup.poll()
        assert sup.process("oneshot") is proc  # never respawned
        assert sup.restart_count("oneshot") == 0
    finally:
        sup.shutdown()
