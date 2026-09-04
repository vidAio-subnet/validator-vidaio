"""BaseService — the shared skeleton every long-running process extends.

Provides: config loading (core + own section), structured JSON logging, a live
/health + /metrics HealthServer, graceful shutdown (SIGINT/SIGTERM), and a
supervised async run loop. Subclasses implement `run()` and register health
checks/metrics on `self.health`.

THE EXIT-CODE CONTRACT
-----------------------------------------
The process exit code is the ONLY thing the supervisor
(`vidaio.validator.supervisor`) can read, so it has to mean something:

* **exit 0 — a deliberate stop.** SIGINT/SIGTERM, or an in-process
  `request_stop()`. The supervisor leaves the child stopped.
* **non-zero — a crash.** The supervisor restarts it with bounded backoff and
  parks it if it crash-loops.

A service that discovers it can no longer do its job (classically: its uvicorn
task died on its own, so the process is alive with nobody answering its port)
must therefore NOT return normally — that is indistinguishable from a clean
SIGTERM and the child would stay down forever while looking healthy-ish. It
calls `fail_fatal(reason)`, which records the reason, flips health, and requests
stop; `serve()` then raises `FatalServiceError` once the loop has unwound, and
`run_service()` / `main()` turn that into a NON-ZERO process exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from pathlib import Path
from typing import Any

from vidaio.core import CoreConfig, HealthServer, get_logger, load_raw_config, section, setup_logging

#: Process exit code for a fatal service failure (EX_SOFTWARE). Any non-zero
#: value means "crash, restart me"; this one is distinctive enough to grep for.
FATAL_EXIT_CODE = 70

#: The health check `fail_fatal` registers (and pins to False). It is registered
#: LAZILY — only once a service has actually failed — so a healthy service's
#: /health payload keeps exactly the checks it declared.
FATAL_CHECK_NAME = "fatal_failure"


class FatalServiceError(RuntimeError):
    """The service cannot do its job any more; the PROCESS must exit non-zero.

    Raised out of `serve()` after the run loop has unwound. Carries the exit code
    `run_service()` gives the OS, which is what makes the supervisor restart the
    child instead of recording a clean stop.
    """

    def __init__(self, service: str, reason: str, exit_code: int = FATAL_EXIT_CODE) -> None:
        super().__init__(f"{service}: {reason}")
        self.service = service
        self.reason = reason
        self.exit_code = exit_code


class BaseService:
    #: subclass override: the service's name (health payload, logger, thread names)
    name: str = "service"

    def __init__(self, raw_config: dict[str, Any], *, metrics_port: int | None = None) -> None:
        self.raw_config = raw_config
        self.core = section(raw_config, "core", CoreConfig)
        self.log = get_logger(self.name)
        self.health = HealthServer(
            self.name, metrics_port if metrics_port is not None else self.core.metrics_port
        )
        self._stop = asyncio.Event()
        self._fatal_reason: str | None = None

    # -- lifecycle -------------------------------------------------------------

    async def run(self) -> None:
        """The service's main loop. Must return promptly once `self.stopping` is set."""
        raise NotImplementedError

    @property
    def stopping(self) -> asyncio.Event:
        return self._stop

    def request_stop(self) -> None:
        """Ask for a COOPERATIVE stop: the process will exit 0 (see the contract)."""
        self._stop.set()

    # -- fatal failure ---------------------------------------

    @property
    def fatal_reason(self) -> str | None:
        """Why this service is going down hard, or None for a cooperative stop."""
        return self._fatal_reason

    @property
    def failed_fatally(self) -> bool:
        return self._fatal_reason is not None

    def fail_fatal(self, reason: str) -> None:
        """The service can no longer do its job: go down and be RESTARTED.

        Records the reason (the FIRST one wins — later cascading failures are
        symptoms), flips health so anything scraping /health sees it before the
        process is gone, logs CRITICAL, and requests stop. `serve()` then raises
        `FatalServiceError` and the process exits non-zero, which is what makes a
        supervisor restart it instead of filing it as a deliberate shutdown.

        Idempotent and safe from any task; it never raises.
        """
        first = self._fatal_reason is None
        if first:
            self._fatal_reason = reason
            # Lazily registered so a healthy service's /health payload is
            # unchanged; once registered it can never pass again.
            self.health.register_check(FATAL_CHECK_NAME, lambda: False)
            self.log.critical(
                f"FATAL: {reason} — exiting NON-ZERO so a supervisor restarts this"
                " service instead of recording a deliberate shutdown",
                extra={"fields": {"service": self.name, "reason": reason}},
            )
        self.request_stop()

    async def serve(self) -> None:
        """Start health server + signal handlers, run the loop, clean up.

        Returns normally on a cooperative stop (caller exits 0); raises
        `FatalServiceError` when `fail_fatal` was called (caller exits non-zero).
        """
        self.health.start()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):  # non-unix platforms
                loop.add_signal_handler(sig, self.request_stop)
        self.log.info("service starting")
        try:
            await self.run()
        finally:
            self.health.stop()
            self.log.info("service stopped")
        if self._fatal_reason is not None:
            raise FatalServiceError(self.name, self._fatal_reason)

    # -- entrypoint ------------------------------------------------------------

    @classmethod
    def main(cls, config_path: str | Path | None = None) -> None:
        raw = load_raw_config(config_path)
        core = section(raw, "core", CoreConfig)
        setup_logging(core.log_level)
        run_service(cls(raw))


def run_service(service: BaseService) -> None:
    """`asyncio.run(service.serve())` with the exit-code contract applied.

    THE one place a fatal failure becomes a process exit code. Every entrypoint
    (`BaseService.main`, the local-stack supervisor children) goes through it, so
    "the API died" can never reach the supervisor as exit 0.
    """
    try:
        asyncio.run(service.serve())
    except FatalServiceError as exc:
        # Already logged CRITICAL by fail_fatal; do not print a traceback for
        # what is a deliberate, described exit.
        sys.stderr.write(f"{exc.service}: FATAL: {exc.reason}\n")
        raise SystemExit(exc.exit_code) from None
