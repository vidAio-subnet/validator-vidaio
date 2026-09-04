import json
import urllib.request

from prometheus_client import Counter

from vidaio.core.metrics import HealthServer


def test_health_and_metrics_endpoints() -> None:
    server = HealthServer("test-svc", port=0)
    Counter("demo_total", "demo", registry=server.registry).inc(3)
    server.register_check("db", lambda: True)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.bound_port}"
        with urllib.request.urlopen(f"{base}/health") as resp:
            payload = json.loads(resp.read())
        assert payload == {"service": "test-svc", "status": "ok", "checks": {"db": True}}
        with urllib.request.urlopen(f"{base}/metrics") as resp:
            assert b"demo_total 3.0" in resp.read()
    finally:
        server.stop()


def test_failing_check_degrades() -> None:
    server = HealthServer("test-svc", port=0)
    server.register_check("chain", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    ok, payload = server.health_payload()
    assert not ok
    assert payload["status"] == "degraded"
    assert payload["checks"] == {"chain": False}
