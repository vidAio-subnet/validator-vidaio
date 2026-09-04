"""Suite-wide guard: NO TEST MAY BIND A FIXED PORT.

Every service in this repo ships with a real, fixed metrics/API port
(`vidaio/services/protocol.py` owns the map: 9100-9108, 8201, 8210, 8300, 8400,
8500, 29996). A test that constructs a service and starts it without overriding
those ports binds the production port on the developer's box, which means:

  * two copies of the suite cannot run at the same time — the second one dies
    with `[Errno 48] address already in use` somewhere unrelated to its subject;
  * the suite cannot run while `the development-tree stack runner` is up;
  * and the failure surfaces as a flake in whichever test lost the race, not in
    the one that bound the port.

That was a real defect: parallel test runs hit it — a test's gateway builder
passed `port: 0` but not `metrics_port`, and `GatewayConfig` defaults that to
the live 9107.

The rule this enforces is simple: a test may only bind a port it was GIVEN by
the kernel. Bind with port 0 (`HealthServer(..., port=0)`, `metrics_port: 0`,
uvicorn `port=0`) and read the assigned port back (`HealthServer.bound_port`,
`server.servers[0].sockets[0].getsockname()[1]`) — the pattern in
`tests/core/test_metrics.py`.

Binding a port that WAS handed out ephemerally is allowed, so the deliberate
"this port is already occupied" tests (which bind :0, capture the port, close
nothing, and point a service at it) keep working: the port is unique to this
process, so a concurrent suite gets a different one.

Violations are recorded rather than raised, so code that wraps its own bind in
`except OSError` cannot swallow the report; the autouse fixture fails the test
that caused it, by name.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

#: Ports the kernel handed out to THIS process in response to a port-0 bind.
#: Re-binding one of these is safe under concurrency (another process gets a
#: different port), so it is not a violation.
_ephemeral: set[int] = set()

#: (port, address) pairs bound at a fixed port since the last test started.
_violations: list[tuple[int, Any]] = []

_real_bind = socket.socket.bind


def _guarded_bind(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
    port = address[1] if isinstance(address, tuple) and len(address) > 1 else None
    result = _real_bind(self, address, *args, **kwargs)
    if isinstance(port, int):
        if port == 0:
            try:
                _ephemeral.add(self.getsockname()[1])
            except OSError:  # pragma: no cover - socket already gone
                pass
        elif port not in _ephemeral:
            _violations.append((port, address))
    return result


socket.socket.bind = _guarded_bind  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _no_fixed_port_binds() -> Any:
    del _violations[:]
    yield
    if _violations:
        bound = ", ".join(f"{addr} (port {port})" for port, addr in _violations)
        del _violations[:]
        raise AssertionError(
            "this test bound a FIXED port: "
            + bound
            + ". Fixed ports make the suite unable to run twice at once (or"
            " alongside a local stack). Pass port 0 / metrics_port 0 and read the"
            " assigned port back (see tests/conftest.py)."
        )
