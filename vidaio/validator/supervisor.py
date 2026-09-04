"""Process-isolation supervisor (spec design spec §13: SN44's core safety pattern).

Every heavy/unsafe service runs as a SEPARATE OS process (multiprocessing
'spawn' context — no forked locks/loops), so a segfault/OOM in one child can
never take down the weight-setter; the DB is the only shared state. The
supervisor monitors liveness via `Process.is_alive` + exit codes and:

- restarts a crashed child with bounded exponential backoff;
- parks a crash-looping child once its restart budget (max_restarts inside
  restart_window_seconds) is exhausted — logged CRITICAL, everything else keeps
  running;
- leaves a cleanly-exited (exitcode 0) child stopped;
- fans out SIGTERM on shutdown (terminate → join → kill stragglers).

THE EXIT-CODE CONTRACT
-----------------------------------------
The exit code is the whole conversation between a child and this supervisor, and
it has exactly two meanings:

* **exitcode 0 = DELIBERATE STOP.** The child was asked to go (SIGTERM/SIGINT
  from our own shutdown fan-out, or an operator), or it was a one-shot that
  finished. It is marked STOPPED and never restarted.
* **exitcode != 0 = CRASH.** Includes a signal death (negative exitcode) and
  `vidaio.services.base.FATAL_EXIT_CODE` (70), which a service uses to say "my
  API/loop died under me, I am not able to do my job". Restarted under the
  bounded backoff, parked if it keeps happening.

That asymmetry is why `BaseService.fail_fatal()` exists: a service that noticed
its uvicorn task die and simply returned would exit 0, and this supervisor —
correctly, per the contract — would file it as a clean shutdown and leave it down
forever. Any "we cannot continue" path in a child MUST end in a non-zero exit.

Children are generic (name, "module:callable" target, config) specs; the child
entrypoint imports the target and calls it with the config dict. Timing knobs
and the clock are injectable so tests run fast and deterministic.
"""

from __future__ import annotations

import importlib
import multiprocessing
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from vidaio.core import get_logger

STATE_RUNNING = "running"
STATE_BACKOFF = "backoff"
STATE_PARKED = "parked"
STATE_STOPPED = "stopped"


class ChildParkedError(RuntimeError):
    """A critical child exhausted its restart budget.

    Parking is useful for a developer/report stack where the remaining services
    should stay observable.  A production process group may instead set
    ``fail_on_park=True`` so its outer container supervisor receives a non-zero
    exit rather than reporting a partially-dead node as healthy.
    """

    def __init__(self, child_name: str, exitcode: int | None) -> None:
        self.child_name = child_name
        self.exitcode = exitcode
        super().__init__(
            f"critical child {child_name!r} is permanently parked after exit "
            f"code {exitcode!r}"
        )


@dataclass(frozen=True)
class ChildSpec:
    name: str
    #: import target, "package.module:callable"; called as target(config)
    target: str
    config: dict[str, Any] | None = None
    #: Whether parking this child is allowed to fail a production process group
    #: whose ``fail_on_park`` policy is enabled.  Observability-only children
    #: (notably an auditor) stay restartable/alertable but must not terminate an
    #: otherwise healthy weight-setter when their retry budget is exhausted.
    critical: bool = True


def _resolve_target(target: str) -> Callable[[dict[str, Any] | None], Any]:
    module_name, sep, attr = target.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"child target must be 'module:callable', got {target!r}")
    fn = getattr(importlib.import_module(module_name), attr)
    if not callable(fn):
        raise TypeError(f"child target {target!r} is not callable")
    return fn


def _child_entry(target: str, config: dict[str, Any] | None) -> None:
    """Top-level (spawn-picklable) child entrypoint."""
    _resolve_target(target)(config)


@dataclass
class _Child:
    spec: ChildSpec
    process: multiprocessing.process.BaseProcess | None = None
    state: str = STATE_STOPPED
    #: monotonic timestamps of recent restarts (budget window)
    restarts: deque[float] = field(default_factory=deque)
    next_restart_at: float = 0.0


class Supervisor:
    def __init__(
        self,
        children: Iterable[ChildSpec | tuple[str, str, dict[str, Any] | None]],
        *,
        max_restarts: int = 5,
        restart_window_seconds: float = 600.0,
        backoff_base_seconds: float = 0.5,
        backoff_max_seconds: float = 30.0,
        poll_interval_seconds: float = 0.5,
        join_timeout_seconds: float = 5.0,
        fail_on_park: bool = False,
        clock: Callable[[], float] = time.monotonic,
        mp_context: multiprocessing.context.BaseContext | None = None,
    ) -> None:
        specs = [c if isinstance(c, ChildSpec) else ChildSpec(*c) for c in children]
        if len({s.name for s in specs}) != len(specs):
            raise ValueError("child names must be unique")
        for spec in specs:
            _resolve_target(spec.target)  # fail fast on a typo, not at first crash
        self._children: dict[str, _Child] = {s.name: _Child(spec=s) for s in specs}
        self.max_restarts = max_restarts
        self.restart_window_seconds = restart_window_seconds
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.join_timeout_seconds = join_timeout_seconds
        self.fail_on_park = fail_on_park
        self._clock = clock
        self._ctx = mp_context or multiprocessing.get_context("spawn")
        self.log = get_logger("supervisor")

    # -- introspection ---------------------------------------------------------

    def states(self) -> dict[str, str]:
        return {name: child.state for name, child in self._children.items()}

    def process(self, name: str) -> multiprocessing.process.BaseProcess | None:
        return self._children[name].process

    def restart_count(self, name: str) -> int:
        return len(self._children[name].restarts)

    # -- lifecycle -------------------------------------------------------------

    def _spawn(self, child: _Child) -> None:
        process = self._ctx.Process(
            target=_child_entry,
            args=(child.spec.target, child.spec.config),
            name=f"vidaio-{child.spec.name}",
            daemon=False,
        )
        process.start()
        child.process = process
        child.state = STATE_RUNNING
        self.log.info(f"child started: {child.spec.name} pid={process.pid}")

    def start(self) -> None:
        for child in self._children.values():
            if child.state == STATE_STOPPED and child.process is None:
                self._spawn(child)

    def poll(self) -> None:
        """One monitoring step: reap exits, schedule/execute restarts.

        Applies the exit-code contract from the module docstring: 0 is the ONLY
        code that means "stay down". Everything else — a signal, an unhandled
        exception (1), a service's FATAL_EXIT_CODE — is a crash to be restarted.
        """
        now = self._clock()
        for child in self._children.values():
            if child.state == STATE_RUNNING:
                process = child.process
                if process is None or process.is_alive():
                    continue
                exitcode = process.exitcode
                if exitcode == 0:
                    # Deliberate stop (SIGTERM/SIGINT or an in-process
                    # request_stop). NOT restarted, by contract.
                    child.state = STATE_STOPPED
                    self.log.info(f"child exited cleanly: {child.spec.name}")
                    continue
                self._on_crash(child, exitcode, now)
            elif child.state == STATE_BACKOFF and now >= child.next_restart_at:
                child.restarts.append(now)
                self._spawn(child)

    def _on_crash(self, child: _Child, exitcode: int | None, now: float) -> None:
        while child.restarts and now - child.restarts[0] > self.restart_window_seconds:
            child.restarts.popleft()
        if len(child.restarts) >= self.max_restarts:
            child.state = STATE_PARKED
            self.log.critical(
                f"child crash-looping, PARKED (no further restarts): {child.spec.name}"
                f" exitcode={exitcode} restarts={len(child.restarts)}"
                f"/{self.restart_window_seconds}s — other children keep running"
            )
            if self.fail_on_park and child.spec.critical:
                raise ChildParkedError(child.spec.name, exitcode)
            return
        backoff = min(
            self.backoff_max_seconds,
            self.backoff_base_seconds * 2 ** len(child.restarts),
        )
        child.state = STATE_BACKOFF
        child.next_restart_at = now + backoff
        self.log.error(
            f"child crashed: {child.spec.name} exitcode={exitcode};"
            f" restart in {backoff:.2f}s"
        )

    def run(self, stop_event: threading.Event) -> None:
        """Blocking monitor loop; returns after a full shutdown fan-out."""
        self.start()
        try:
            while not stop_event.is_set():
                self.poll()
                stop_event.wait(self.poll_interval_seconds)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful SIGTERM fan-out: terminate -> join -> kill stragglers."""
        alive = [
            c for c in self._children.values() if c.process is not None and c.process.is_alive()
        ]
        for child in alive:
            assert child.process is not None
            child.process.terminate()  # SIGTERM on posix
        deadline = time.monotonic() + self.join_timeout_seconds
        for child in alive:
            assert child.process is not None
            child.process.join(max(0.0, deadline - time.monotonic()))
            if child.process.is_alive():
                self.log.error(f"child ignored SIGTERM, killing: {child.spec.name}")
                child.process.kill()
                child.process.join(self.join_timeout_seconds)
        for child in self._children.values():
            if child.state != STATE_PARKED:
                child.state = STATE_STOPPED
