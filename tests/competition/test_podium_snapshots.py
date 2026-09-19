"""A registered contender that runs no inference miner is still payable."""

from __future__ import annotations

from types import SimpleNamespace

from vidaio.competition.podium_snapshots import with_podium_carriers
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import EmissionState, MinerSnapshot, RewardWindowState
from vidaio.tokenomics.weights import build_weight_vector

from datetime import datetime, timedelta, timezone

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _neuron(uid: int, hotkey: str, stake: float = 600.0):
    return SimpleNamespace(uid=uid, hotkey=hotkey, coldkey=f"ck-{hotkey}", ip=f"10.0.0.{uid}", alpha_stake=stake)


def _inference_miner() -> MinerSnapshot:
    return MinerSnapshot(uid=1, hotkey="inf", coldkey="ck-inf", ip="10.0.0.1",
                         track="compression", accumulate_score=0.8)


def test_absent_registered_podium_hotkeys_get_zero_score_rows() -> None:
    neurons = [_neuron(1, "inf"), _neuron(5, "winner"), _neuron(7, "second")]
    out = with_podium_carriers(
        [_inference_miner()], chain_neurons=neurons,
        podium_hotkeys=("winner", "second", "gone"), track="compression",
    )
    assert [(m.uid, m.hotkey, m.accumulate_score, m.track) for m in out] == [
        (1, "inf", 0.8, "compression"),
        (5, "winner", 0.0, "compression"),
        (7, "second", 0.0, "compression"),
    ]  # "gone" is deregistered: never added, its share stays with the sink


def test_present_hotkeys_and_reused_uid_slots_are_left_alone() -> None:
    snapshots = [_inference_miner()]
    neurons = [_neuron(1, "winner")]  # uid 1 is already another identity's slot
    assert with_podium_carriers(
        snapshots, chain_neurons=neurons, podium_hotkeys=("winner",), track="compression"
    ) == snapshots
    assert with_podium_carriers(
        snapshots, chain_neurons=[_neuron(1, "inf")], podium_hotkeys=("inf",), track="compression"
    ) == snapshots
    assert with_podium_carriers(snapshots, chain_neurons=neurons, podium_hotkeys=(), track="compression") == snapshots


def test_competition_only_contenders_are_actually_paid() -> None:
    config = TokenomicsConfig(competition_emissions_enabled=True)
    window = RewardWindowState(
        kind=EmissionState.PODIUM, starts_at=NOW, ends_at=NOW + timedelta(hours=168),
        podium_hotkeys=("winner", "second"), winner_hotkey="winner", winner_uid=5,
        winner_score=0.5, winner_margin=0.02, baseline_score=0.49, baseline_version=0,
        baseline_artifact_digest="ab" * 32, source_competition_id="c", source_track="compression",
        source_cycle=1, last_applied_cycle=1,
    )
    neurons = [_neuron(1, "inf"), _neuron(5, "winner"), _neuron(7, "second")]
    without = build_weight_vector(config, [_inference_miner()], burn_uid=0, reward_state=window, now=NOW)
    assert without.get(5, 0.0) == 0.0 and without[0] > 0.39  # the whole podium burned
    snapshots = with_podium_carriers(
        [_inference_miner()], chain_neurons=neurons,
        podium_hotkeys=window.podium_hotkeys, track=window.source_track,
    )
    paid = build_weight_vector(config, snapshots, burn_uid=0, reward_state=window, now=NOW)
    assert abs(paid[5] - 0.40 * 0.70) < 1e-12 and abs(paid[7] - 0.40 * 0.20) < 1e-12
    assert paid[1] > 0.0  # the inference miner keeps its own share
