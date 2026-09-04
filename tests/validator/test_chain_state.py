"""review #21 (chain refresh fails open) and #22 (health reports the truth).

#21: `HttpChainAdapter.refresh()` swallows transport failures and the validator
stamped the refresh timestamp regardless, so a startup race against the chain sim
left an EMPTY snapshot considered fresh for the whole 30-minute throttle window
and the validator reported successful empty rounds. A failed refresh must not be
recorded as a success, and a stale/unavailable snapshot must SKIP the round with
a structured reason.

The adapter's freshness surface (`has_fresh_snapshot`) is owned by the chain
layer and is FEATURE-DETECTED here, so the validator is correct whether or not it
has landed yet.

#22: the health callbacks run on the HealthServer's thread; using the round
loop's sqlite3 handle there raised ProgrammingError and made /health report the
DB unhealthy even when it was fine.
"""

from __future__ import annotations

import json
import threading
import urllib.request

from vidaio.validator import miner_manager

from validator_support import mk_neuron


def metric(validator, name: str, labels: dict[str, str] | None = None) -> float | None:
    return validator.health.registry.get_sample_value(name, labels or {})


class FreshnessChain:
    """InMemoryChain wrapper with the adapter freshness surface an internal review adds."""

    def __init__(self, inner, *, fresh: bool = True, refresh_error: Exception | None = None):
        self.inner = inner
        self.fresh = fresh
        self.refresh_error = refresh_error
        self.refresh_calls = 0
        self.freshness_calls: list[tuple] = []

    def current_block(self):
        return self.inner.current_block()

    def neurons(self):
        return self.inner.neurons()

    def refresh(self):
        self.refresh_calls += 1
        if self.refresh_error is not None:
            raise self.refresh_error

    def has_fresh_snapshot(self, now, max_age):
        self.freshness_calls.append((now, max_age))
        return self.fresh

    async def set_weights(self, weights, *, version_key, hotkeys=None):
        return await self.inner.set_weights(
            weights, version_key=version_key, hotkeys=hotkeys
        )

    async def anchor_commitment(self, payload):
        return await self.inner.anchor_commitment(payload)


class UnavailableChain(FreshnessChain):
    """A never-refreshed adapter raises instead of serving an empty snapshot."""

    def neurons(self):
        raise RuntimeError("ChainStateUnavailable: neurons() before any refresh")


async def test_failed_refresh_is_not_recorded_as_a_successful_one(
    make_validator, chain, miner_client, conn
):
    flaky = FreshnessChain(chain, refresh_error=OSError("chainsim unreachable"))
    flaky.has_fresh_snapshot = None  # adapter without the new surface (not yet landed)
    validator = make_validator(chain=flaky, config={"max_chain_snapshot_age_seconds": 60.0})
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    report = await validator.run_round()

    assert validator._last_refresh_at is None  # NOT stamped
    assert report.skipped_reason == "chain_snapshot_never_refreshed"
    assert report.scored == {} and report.round_id is None
    assert metric(validator, "vidaio_validator_chain_refresh_failures_total") == 1.0
    assert (
        metric(
            validator,
            "vidaio_validator_rounds_skipped_total",
            {"reason": "chain_snapshot_never_refreshed"},
        )
        == 1.0
    )
    # a skipped round is NOT a completed round
    assert metric(validator, "vidaio_validator_rounds_total") in (None, 0.0)
    assert miner_manager.uncommitted_rounds(conn) == []


async def test_adapter_reporting_a_stale_snapshot_skips_the_round(
    make_validator, chain, miner_client, conn
):
    stale = FreshnessChain(chain, fresh=False)
    validator = make_validator(chain=stale)
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    report = await validator.run_round()

    assert report.skipped_reason == "chain_snapshot_stale"
    assert stale.freshness_calls  # the adapter's own surface was consulted
    assert stale.freshness_calls[0][1] == validator.config.max_chain_snapshot_age_seconds
    assert miner_client.task_calls == []  # nothing was dispatched or scored
    assert miner_manager.uncommitted_rounds(conn) == []


async def test_unavailable_chain_state_skips_instead_of_scoring_an_empty_round(
    make_validator, chain, conn
):
    validator = make_validator(chain=UnavailableChain(chain))

    report = await validator.run_round()

    assert report.skipped_reason == "chain_state_unavailable"
    assert report.scored == {}
    assert (
        metric(
            validator,
            "vidaio_validator_rounds_skipped_total",
            {"reason": "chain_state_unavailable"},
        )
        == 1.0
    )


async def test_locally_stale_snapshot_skips_when_the_adapter_has_no_surface(
    make_validator, chain, miner_client
):
    """Correct even before the chain owner lands `has_fresh_snapshot`."""
    clock = [0.0]
    plain = FreshnessChain(chain)
    plain.has_fresh_snapshot = None
    validator = make_validator(
        chain=plain,
        clock=lambda: clock[0],
        config={
            "metagraph_refresh_seconds": 1800.0,
            "max_chain_snapshot_age_seconds": 60.0,
        },
    )
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    assert (await validator.run_round()).skipped_reason is None  # fresh refresh
    clock[0] = 61.0  # inside the refresh throttle, past the staleness bound
    report = await validator.run_round()

    assert report.skipped_reason == "chain_snapshot_stale"
    assert plain.refresh_calls == 1  # throttle held; the round was skipped instead


async def test_fresh_adapter_snapshot_runs_the_round_normally(
    make_validator, chain, miner_client
):
    fresh = FreshnessChain(chain, fresh=True)
    validator = make_validator(chain=fresh)
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    report = await validator.run_round()

    assert report.skipped_reason is None
    assert report.scored == {1: 0.8}


async def test_gate_can_be_disabled_with_zero(make_validator, chain, miner_client):
    stale = FreshnessChain(chain, fresh=False)
    validator = make_validator(
        chain=stale, config={"max_chain_snapshot_age_seconds": 0.0}
    )
    chain.set_neurons([mk_neuron(1)])
    miner_client.tracks = {1: "compression"}

    assert (await validator.run_round()).skipped_reason is None


# --- #22 health ----------------------------------------------------------------


async def test_health_check_from_another_thread_reports_a_healthy_db(validator, chain):
    """The real /health path: a DIFFERENT thread must not report the DB broken."""
    chain.set_neurons([])
    await validator.run_round()

    results: list[tuple[bool, dict]] = []
    thread = threading.Thread(target=lambda: results.append(validator.health.health_payload()))
    thread.start()
    thread.join(timeout=5.0)

    assert results, "health check thread did not finish"
    ok, payload = results[0]
    assert ok is True, payload
    assert payload["checks"]["db"] is True


async def test_live_health_server_reports_the_truth(validator, chain):
    chain.set_neurons([])
    await validator.run_round()
    # EPHEMERAL bind (same as tests/core/test_metrics.py and the gateway's health
    # test): the configured default is the real 9101, and binding it here collides
    # with any other test process — or a running local stack — on this box.
    # ValidatorConfig rightly refuses metrics_port 0, so the server is retargeted
    # rather than the config.
    validator.health.port = 0
    validator.health.start()
    try:
        url = f"http://127.0.0.1:{validator.health.bound_port}/health"
        with urllib.request.urlopen(url, timeout=5.0) as response:
            assert response.status == 200
            payload = json.loads(response.read())
    finally:
        validator.health.stop()

    assert payload["status"] == "ok"
    assert payload["checks"]["db"] is True


def test_in_memory_db_falls_back_to_the_loop_connection(raw_config, chain, tmp_path):
    """':memory:' cannot be reopened, so no per-thread handle is invented for it."""
    from vidaio.core import apply_migrations, connect
    from vidaio.validator import MIGRATIONS_DIR, InferenceValidator

    conn = connect(":memory:")
    apply_migrations(conn, MIGRATIONS_DIR)
    validator = InferenceValidator(raw_config, chain=chain, conn=conn)
    assert validator._conn_factory is None
    assert validator._health_conn() is conn
    assert validator._db_reachable() is True
