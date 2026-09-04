"""Live /health + /metrics on every service (design spec §14: 'missing — build it').

HealthServer runs a small threaded HTTP server exposing:
  GET /health  -> JSON {service, status, checks}; 200 when all checks pass, else 503
  GET /metrics -> Prometheus exposition from the service's registry
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST


class HealthServer:
    def __init__(
        self,
        service: str,
        port: int,
        registry: CollectorRegistry | None = None,
        host: str = "0.0.0.0",
    ) -> None:
        self.service = service
        self.host = host
        self.port = port
        self.registry = registry or CollectorRegistry()
        self._checks: dict[str, Callable[[], bool]] = {}
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def register_check(self, name: str, fn: Callable[[], bool]) -> None:
        self._checks[name] = fn

    def health_payload(self) -> tuple[bool, dict[str, Any]]:
        checks: dict[str, bool] = {}
        ok = True
        for name, fn in self._checks.items():
            try:
                healthy = bool(fn())
            except Exception:
                healthy = False
            checks[name] = healthy
            ok = ok and healthy
        status = "ok" if ok else "degraded"
        return ok, {"service": self.service, "status": status, "checks": checks}

    @property
    def bound_port(self) -> int:
        """Actual port (useful when constructed with port=0 in tests)."""
        if self._server is None:
            return self.port
        return self._server.server_address[1]

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    ok, payload = outer.health_payload()
                    body = json.dumps(payload).encode()
                    self.send_response(200 if ok else 503)
                    self.send_header("Content-Type", "application/json")
                elif self.path == "/metrics":
                    body = generate_latest(outer.registry)
                    self.send_response(200)
                    self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                else:
                    body = b"not found"
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:  # silence stderr access log
                pass

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name=f"health-{self.service}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
