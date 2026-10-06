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
    # A window without its own payout policy is paid with the live v3 defaults: the
    # whole emission, podium split 50/24 redistributed over the two filled places.
    assert abs(paid[5] - 0.50 / 0.74) < 1e-12 and abs(paid[7] - 0.24 / 0.74) < 1e-12
    assert paid[1] == 0.0  # inference earns nothing under tokenomics v3


def test_removal_competition_carriers_use_a_protocol_track() -> None:
    """A removal (competition-only) track must never reach a miner snapshot: the epoch log
    refuses non-protocol tracks, which deferred the sn85-removal-002 result."""
    from vidaio.competition.podium_snapshots import CARRIER_FALLBACK_TRACK, carrier_track
    from vidaio.epoch.log import PROTOCOL_TRACKS

    assert CARRIER_FALLBACK_TRACK in PROTOCOL_TRACKS
    assert carrier_track("removal") == CARRIER_FALLBACK_TRACK
    assert all(carrier_track(t) == t for t in PROTOCOL_TRACKS)
    neurons = [_neuron(1, "inf"), _neuron(5, "winner"), _neuron(7, "second")]
    out = with_podium_carriers(
        [_inference_miner()], chain_neurons=neurons,
        podium_hotkeys=("winner", "second"), track="removal",
    )
    assert [(m.uid, m.track) for m in out] == [(1, "compression"), (5, "compression"), (7, "compression")]
    assert {m.track for m in out} <= PROTOCOL_TRACKS


def test_removal_competition_contenders_are_paid_like_any_other_track() -> None:
    config = TokenomicsConfig(competition_emissions_enabled=True)
    window = RewardWindowState(
        kind=EmissionState.CROWN, starts_at=NOW, ends_at=NOW + timedelta(hours=168),
        podium_hotkeys=("winner", "second"), winner_hotkey="winner", winner_uid=5,
        winner_score=0.79, winner_margin=0.79, baseline_score=0.0, baseline_version=0,
        baseline_artifact_digest="ab" * 32, source_competition_id="removal-c", source_track="removal",
        source_cycle=1, last_applied_cycle=1,
    )
    neurons = [_neuron(1, "inf"), _neuron(5, "winner"), _neuron(7, "second")]
    snapshots = with_podium_carriers(
        [_inference_miner()], chain_neurons=neurons,
        podium_hotkeys=window.podium_hotkeys, track=window.source_track,
    )
    paid = build_weight_vector(config, snapshots, burn_uid=0, reward_state=window, now=NOW)
    assert paid.get(5, 0.0) > 0.5 and paid.get(7, 0.0) > 0.0  # the crown and 2nd place are payable
    assert paid[1] == 0.0


def _subject(role: str, uid: int | None = None, hotkey: str | None = None):
    return SimpleNamespace(role=role, uid=uid, hotkey=hotkey)


def _result_epoch_input():
    """A baseline plus seven contenders (uids 5-11): five paid places, c10 and c11 unpaid."""
    subjects = (_subject("baseline"),) + tuple(_subject("contender", uid, f"c{uid}") for uid in range(5, 12))
    podium = tuple(f"c{uid}" for uid in range(5, 10))
    neurons = [_neuron(1, "inf", stake=0.0)] + [_neuron(uid, f"c{uid}") for uid in range(5, 12)]
    return SimpleNamespace(subjects=subjects), podium, neurons


def _census_failures(snapshots, neurons, competition_input) -> dict[int, str]:
    """The auditor's own census verdicts for a result epoch: FAIL codes by uid."""
    from vidaio.auditor.config import AuditorConfig
    from vidaio.auditor.report import ItemVerdictKind
    from vidaio.auditor.service import Auditor, InMemoryBundleSource
    from vidaio.epoch.log import MinerCensusEntry

    census = [
        MinerCensusEntry(uid=n.uid, hotkey=n.hotkey, coldkey=n.coldkey, ip=n.ip, alpha_stake=n.alpha_stake)
        for n in neurons
    ]
    log = SimpleNamespace(
        miners=list(snapshots), miner_census=census, schema_version=17,
        audit_manifest=SimpleNamespace(
            earning_inputs={}, availability_inputs=(), per_uid={}, competition_input=competition_input,
        ),
    )
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="auditor", tokenomics=TokenomicsConfig(), burn_uid=94),
        InMemoryBundleSource(),
    )
    verdicts = auditor._census_verdicts(log, {entry.uid: entry for entry in census})
    return {v.uid: v.code for v in verdicts if v.verdict is ItemVerdictKind.FAIL}


def test_result_epoch_carriers_list_the_podium_then_every_other_contender() -> None:
    from vidaio.competition.podium_snapshots import result_epoch_carrier_hotkeys

    competition_input = SimpleNamespace(subjects=(
        _subject("baseline"), _subject("contender", 9, "unpaid"),
        _subject("contender", 5, "winner"), _subject("contender", 7, "second"),
    ))
    assert result_epoch_carrier_hotkeys(("winner", "second"), competition_input) == ("winner", "second", "unpaid")
    assert result_epoch_carrier_hotkeys(("winner", "winner"), None) == ("winner",)


def test_unpaid_contenders_get_rows_so_the_result_epoch_audits_clean() -> None:
    """sn85-removal-002: unpaid contenders that run no inference miner had no economic row in
    the result epoch, and the auditor read each one as censorship (CENSUS_MISMATCH)."""
    from vidaio.auditor import CENSUS_MISMATCH
    from vidaio.competition.podium_snapshots import result_epoch_carrier_hotkeys

    competition_input, podium, neurons = _result_epoch_input()
    podium_only = with_podium_carriers(
        [_inference_miner()], chain_neurons=neurons, podium_hotkeys=podium, track="removal",
    )
    assert _census_failures(podium_only, neurons, competition_input) == {10: CENSUS_MISMATCH, 11: CENSUS_MISMATCH}

    every_contender = with_podium_carriers(
        [_inference_miner()], chain_neurons=neurons,
        podium_hotkeys=result_epoch_carrier_hotkeys(podium, competition_input), track="removal",
    )
    assert _census_failures(every_contender, neurons, competition_input) == {}
    assert [m.uid for m in every_contender] == [1, 5, 6, 7, 8, 9, 10, 11]
    assert all(
        m.accumulate_score == 0.0 and not m.excluded and m.track == "compression"
        for m in every_contender if m.uid != 1
    )


def test_unpaid_contender_rows_leave_the_weight_vector_unchanged() -> None:
    from vidaio.competition.podium_snapshots import result_epoch_carrier_hotkeys

    competition_input, podium, neurons = _result_epoch_input()
    config = TokenomicsConfig(competition_emissions_enabled=True)
    window = RewardWindowState(
        kind=EmissionState.CROWN, starts_at=NOW, ends_at=NOW + timedelta(hours=168),
        podium_hotkeys=podium, winner_hotkey="c5", winner_uid=5,
        winner_score=0.79, winner_margin=0.79, baseline_score=0.0, baseline_version=0,
        baseline_artifact_digest="ab" * 32, source_competition_id="removal-c", source_track="removal",
        source_cycle=1, last_applied_cycle=1,
    )
    podium_only = with_podium_carriers(
        [_inference_miner()], chain_neurons=neurons, podium_hotkeys=podium, track="removal",
    )
    every_contender = with_podium_carriers(
        [_inference_miner()], chain_neurons=neurons,
        podium_hotkeys=result_epoch_carrier_hotkeys(podium, competition_input), track="removal",
    )
    before = build_weight_vector(config, podium_only, burn_uid=0, reward_state=window, now=NOW)
    after = build_weight_vector(config, every_contender, burn_uid=0, reward_state=window, now=NOW)
    assert {uid: w for uid, w in after.items() if w} == {uid: w for uid, w in before.items() if w}
    assert after.get(10, 0.0) == 0.0 and after.get(11, 0.0) == 0.0
    assert abs(after[5] - 0.90) < 1e-12  # the crown split is untouched
