"""A sandbox runner whose remote runtime exists only while it is needed.

A remote sandbox runtime (the fresh, create-only Modal App) is an ephemeral object kept
alive by client heartbeats. Creating it once per *process* left it idle for the whole
enrollment window, where it silently died (``App state is APP_STATE_STOPPED``) while
the local availability flag still said healthy, and recovery needed new names plus a
container recreate by hand.

``RenewableRunner`` wraps a factory instead of a runner:

* nothing remote is created at construction — an idle orchestrator holds no App and
  therefore has nothing that can die;
* ``acquire()`` creates a runner from the factory when a competition is about to build
  or evaluate; every creation gets a new sequence number, from which the factory mints
  brand-new resource names (the create-only rule is preserved: names are never reused
  and nothing is discovered or attached to);
* ``release()`` closes it as soon as no phase needs sandboxes;
* an operation that fails because the remote runtime is gone marks the inner runner
  dead, so the next ``acquire()`` replaces it. The orchestrator's existing restart fence
  sees a new ``runtime_session_id`` and rehydrates exactly-bound images / resets batches,
  i.e. an in-process replacement is handled exactly like a process restart.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from vidaio.competition.runners.errors import RunnerUnavailableError

_LOG = logging.getLogger("vidaio.competition.runners.renewable")

#: Substrings (lower-case) that identify "the remote runtime no longer exists".
DEAD_RUNTIME_MARKERS: tuple[str, ...] = (
    "app_state_stopped",
    "app state is",
    "app is stopped",
    "app has stopped",
    "this runner must be replaced",
)


def is_dead_runtime_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in DEAD_RUNTIME_MARKERS)


def generation_name(base: str, sequence: int, *, stamp: str, max_length: int = 63) -> str:
    """``<base>-g<seq>-<stamp>``: a never-reused resource name for one generation.

    ``base`` is the operator-configured identity; it is shortened (never the unique
    suffix) when the result would exceed ``max_length``.
    """
    if sequence < 1:
        raise ValueError("generation sequence starts at 1")
    suffix = f"-g{sequence:03d}-{stamp}"
    room = max_length - len(suffix)
    if room < 8:
        raise ValueError("resource name budget is too small for a generation suffix")
    return base[:room].rstrip("-._") + suffix


class RenewableRunner:
    """``SandboxRunner`` facade over a lazily created, replaceable inner runner."""

    def __init__(
        self,
        factory: Callable[[int, dict[str, Any] | None], Any],
        *,
        static: dict[str, Any] | None = None,
        idle_grace_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        if idle_grace_seconds < 0:
            raise ValueError("idle_grace_seconds must be >= 0")
        self._factory = factory
        #: ``release()`` is ignored while an operation is in flight and for this long
        #: after the last acquire/operation, so a lifecycle tick can never close the
        #: runtime between a caller's acquire() and its first build.
        self._idle_grace = idle_grace_seconds
        self._clock = clock
        self._in_flight = 0
        self._last_used = clock()
        #: Compute envelope requested for the NEXT generation, and the one the live
        #: inner runner was created with. A change forces a new generation.
        self._resources: dict[str, Any] | None = None
        self._inner_resources: dict[str, Any] | None = None
        #: Attributes that must be readable while idle (e.g. ``gpu`` for the manifest
        #: policy check, the work directories). Never anything runtime-scoped.
        self._static = dict(static or {})
        self._log = logger or _LOG
        self._lock = threading.RLock()
        self._inner: Any | None = None
        self._sequence = 0
        self._closed = False
        self._last_error: str | None = None

    # -- lifecycle ------------------------------------------------------------

    @property
    def active(self) -> bool:
        with self._lock:
            return self._inner is not None

    @property
    def generations(self) -> int:
        """How many inner runners have been created so far."""
        return self._sequence

    def configure(self, resources: dict[str, Any] | None) -> None:
        """Set the compute envelope of the competition about to use the runner."""
        with self._lock:
            self._resources = None if resources is None else dict(resources)

    def acquire(self) -> None:
        """Ensure a live inner runner exists; create a fresh one when needed."""
        with self._lock:
            if self._closed:
                raise RunnerUnavailableError("the renewable runner is closed")
            self._last_used = self._clock()
            inner = self._inner
            if inner is not None:
                available = getattr(inner, "available", None)
                if self._inner_resources != self._resources:
                    if self._in_flight > 0:
                        raise RunnerUnavailableError(
                            "sandbox resources changed while an operation is in flight"
                        )
                    self._discard("compute envelope changed")
                elif not callable(available) or available():
                    return
                else:
                    self._discard("inner runner reported unavailable")
            self._sequence += 1
            try:
                self._inner = self._factory(self._sequence, self._resources)
                self._inner_resources = (
                    None if self._resources is None else dict(self._resources)
                )
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise
            self._last_error = None
            self._log.info(
                "sandbox runtime acquired",
                extra={"fields": {"generation": self._sequence}},
            )

    def release(self) -> None:
        """Close the inner runner (no-op when idle). Safe to call every tick."""
        with self._lock:
            if self._inner is None or self._in_flight > 0:
                return
            if self._clock() - self._last_used < self._idle_grace:
                return
            self._discard("released: no phase needs sandboxes")

    def close(self) -> None:
        with self._lock:
            self._discard("runner closed")
            self._closed = True

    def available(self) -> bool:
        """Idle is healthy (nothing allocated); otherwise ask the inner runner."""
        with self._lock:
            if self._closed:
                return False
            inner = self._inner
        if inner is None:
            return True
        available = getattr(inner, "available", None)
        return bool(available()) if callable(available) else True

    def _discard(self, reason: str) -> None:
        inner, self._inner = self._inner, None
        if inner is None:
            return
        self._log.info(
            "sandbox runtime discarded",
            extra={"fields": {"generation": self._sequence, "reason": reason}},
        )
        close = getattr(inner, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - a dead App cannot always close
                self._log.warning(
                    "closing a discarded sandbox runtime failed",
                    extra={"fields": {"error": f"{type(exc).__name__}: {exc}"}},
                )

    # -- delegation -----------------------------------------------------------

    def _live(self) -> Any:
        with self._lock:
            if self._inner is None:
                raise RunnerUnavailableError(
                    "no sandbox runtime is acquired; acquire() before building or "
                    "running batches"
                )
            return self._inner

    def _guarded(self, name: str) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                inner = self._live()
                self._in_flight += 1
                self._last_used = self._clock()
            try:
                return getattr(inner, name)(*args, **kwargs)
            except BaseException as exc:
                if is_dead_runtime_error(exc):
                    with self._lock:
                        if self._inner is inner:
                            self._discard(f"remote runtime is gone: {exc}")
                raise
            finally:
                with self._lock:
                    self._in_flight -= 1
                    self._last_used = self._clock()

        return call

    def build(self, contender: Any) -> str:
        return self._guarded("build")(contender)

    def run_batch(self, *args: Any, **kwargs: Any) -> Any:
        return self._guarded("run_batch")(*args, **kwargs)

    def isolation_probe(self, image_digest: str) -> Any:
        return self._guarded("isolation_probe")(image_digest)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not defined above.
        if name.startswith("_"):
            raise AttributeError(name)
        static = self.__dict__.get("_static", {})
        if name in static:
            return static[name]
        inner = self.__dict__.get("_inner")
        if inner is None:
            # Optional capability probes (`getattr(runner, x, None)`) must see "absent"
            # while idle rather than an exception type they do not expect.
            raise AttributeError(name)
        value = getattr(inner, name)
        return self._guarded(name) if callable(value) else value


__all__ = [
    "DEAD_RUNTIME_MARKERS",
    "RenewableRunner",
    "generation_name",
    "is_dead_runtime_error",
]
