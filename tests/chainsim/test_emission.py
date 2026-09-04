"""Emission model: proportional to the LAST recorded weight vector, per block.

A vector recorded at block B governs blocks strictly after B until the next
vector; ungoverned blocks and unregistered-uid shares are undistributed.
"""

from __future__ import annotations

import pytest

from vidaio.chainsim.service import RegisterRequest, WeightsRequest


@pytest.fixture
def sim2(make_sim):
    sim = make_sim(tempo=2, emission_per_block=1.0)
    sim.register(RegisterRequest(hotkey="val", role="validator"))  # uid 0
    sim.register(RegisterRequest(hotkey="hk1", role="miner"))  # uid 1
    sim.register(RegisterRequest(hotkey="hk2", role="miner"))  # uid 2
    return sim


def credited(sim) -> dict[int, float]:
    return {n["uid"]: n["emission_credited"] for n in sim.state()["neurons"]}


def test_no_vector_means_no_distribution(sim2):
    sim2.advance(5)
    state = sim2.state()
    assert credited(sim2) == {0: 0.0, 1: 0.0, 2: 0.0}
    assert state["emission"]["minted"] == 6.0  # blocks 1..6
    assert state["emission"]["undistributed"] == 6.0


def test_emission_follows_the_last_recorded_vector_proportionally(sim2):
    # vector at block 1: 3:1 split between uid 1 and uid 2
    assert sim2.submit_weights(WeightsRequest(hotkey="val", vector={1: 3.0, 2: 1.0}))[
        "success"
    ]
    sim2.advance(4)  # blocks 2..5 governed by the 3:1 vector
    assert credited(sim2) == {0: 0.0, 1: 3.0, 2: 1.0}
    # block 1 itself had no prior vector -> undistributed
    assert sim2.state()["emission"]["undistributed"] == pytest.approx(1.0)

    # new vector at block 5 (tempo 2: 5 > 1 + 2): even split, governs blocks 6+
    assert sim2.submit_weights(WeightsRequest(hotkey="val", vector={1: 1.0, 2: 1.0}))[
        "success"
    ]
    sim2.advance(2)  # blocks 6..7
    assert credited(sim2) == {0: 0.0, 1: pytest.approx(4.0), 2: pytest.approx(2.0)}


def test_neurons_report_the_current_per_block_rate(sim2):
    sim2.submit_weights(WeightsRequest(hotkey="val", vector={1: 3.0, 2: 1.0}))
    rates = {n["uid"]: n["emission"] for n in sim2._neuron_dicts()}
    assert rates == {0: 0.0, 1: pytest.approx(0.75), 2: pytest.approx(0.25)}


def test_unregistered_uid_share_is_burned(sim2):
    sim2.submit_weights(WeightsRequest(hotkey="val", vector={1: 1.0, 99: 1.0}))
    sim2.advance(2)  # blocks 2..3: uid 99 does not exist, its half is burned
    state = sim2.state()
    assert credited(sim2) == {0: 0.0, 1: pytest.approx(1.0), 2: 0.0}
    # minted 3 (blocks 1..3), distributed 1, burned: block 1 (1.0) + uid99 share (1.0)
    assert state["emission"]["undistributed"] == pytest.approx(2.0)


def test_settlement_is_deterministic_wrt_when_it_runs(tmp_path, fake_time):
    """Settling lazily (late) equals settling eagerly (mid-run) — same credits."""
    from vidaio.chainsim import ChainSim

    def build(name: str) -> ChainSim:
        return ChainSim(
            {
                "core": {"metrics_port": 0},
                "chainsim": {
                    "metrics_port": 0,
                    "db_path": str(tmp_path / f"{name}.db"),
                    "report_dir": str(tmp_path / f"{name}-reports"),
                    "tempo": 2,
                },
            },
            now=fake_time,
        )

    lazy, eager = build("lazy"), build("eager")
    for sim, settle_each_step in ((lazy, False), (eager, True)):
        sim.register(RegisterRequest(hotkey="val", role="validator"))
        sim.register(RegisterRequest(hotkey="hk1", role="miner"))
        sim.register(RegisterRequest(hotkey="hk2", role="miner"))
        sim.submit_weights(WeightsRequest(hotkey="val", vector={1: 3.0, 2: 1.0}))
        sim.advance(4)
        if settle_each_step:
            sim.state()  # forces settlement mid-run
        sim.submit_weights(WeightsRequest(hotkey="val", vector={1: 1.0, 2: 1.0}))
        sim.advance(2)
    assert credited(lazy) == credited(eager) == {
        0: 0.0,
        1: pytest.approx(4.0),
        2: pytest.approx(2.0),
    }
    lazy.close()
    eager.close()
