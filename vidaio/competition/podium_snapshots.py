"""Make every registered podium identity payable, with or without an inference miner.

The weight vector pays a podium hotkey only when that hotkey appears in the epoch's
economic miner snapshot. That snapshot is projected from inference activity (a hotkey
with a resolved inference track), so a contender that only competes was registered,
ranked, committed in the epoch evidence — and then had its share routed to the sink.

``with_podium_carriers`` appends a zero-score snapshot row for each PAID podium hotkey
that is registered at the close block but absent from the economic snapshot. The row
carries the close-block identity (uid, coldkey, ip, stake) and the competition's track.
A zero accumulator can never earn inference emissions or occupy an inference rank; the
row exists solely so the committed podium share has a payable uid. A hotkey that is no
longer registered is not added: its share still goes to the sink, exactly as before.

The row's track must be a PROTOCOL track: the epoch log refuses any miner snapshot whose
track is outside ``TRACK_RULES``. A competition-only track such as
``removal`` is therefore carried on a fixed protocol track. Competition payouts map the
podium hotkey to its uid and never read the track, and a zero accumulator earns no
inference share, so the carrier track changes nothing but the log's validity.

The epoch that APPLIES a result needs a row for every contender with committed evidence,
paid or not: the auditor binds each evidenced contender uid to the economic miners set and
reads an omitted one as censorship (``CENSUS_MISMATCH``, which disputed the
sn85-removal-002 result epoch). :func:`result_epoch_carrier_hotkeys` lists the paid podium
first, then every other contender of the committed competition input. A zero accumulator
keeps those rows out of the IP/coldkey dedup and the rank curve, so the weight vector is
unchanged.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from vidaio.challenge.dag import TRACK_RULES
from vidaio.tokenomics.state import MinerSnapshot

#: Protocol track used for carrier rows of competition-only tracks (e.g. ``removal``).
CARRIER_FALLBACK_TRACK = "compression"


def carrier_track(track: str) -> str:
    """The snapshot track for a podium carrier row: the competition track when it is a
    protocol track, else :data:`CARRIER_FALLBACK_TRACK`."""
    return track if track in TRACK_RULES else CARRIER_FALLBACK_TRACK


def with_podium_carriers(
    snapshots: Sequence[MinerSnapshot],
    *,
    chain_neurons: Iterable[object],
    podium_hotkeys: Sequence[str],
    track: str | None,
) -> list[MinerSnapshot]:
    """Return ``snapshots`` plus one zero-score row per registered, absent podium hotkey.

    In a result epoch ``podium_hotkeys`` is :func:`result_epoch_carrier_hotkeys`, which also
    names the unpaid contenders."""
    result = list(snapshots)
    if not podium_hotkeys or not track:
        return result
    row_track = carrier_track(track)
    present = {miner.hotkey for miner in result}
    taken_uids = {miner.uid for miner in result}
    by_hotkey = {str(getattr(n, "hotkey")): n for n in chain_neurons}
    for hotkey in podium_hotkeys:
        if hotkey in present:
            continue
        neuron = by_hotkey.get(hotkey)
        if neuron is None:
            continue  # deregistered: the share is withheld to the sink, never reassigned
        uid = int(getattr(neuron, "uid"))
        if uid in taken_uids:
            continue  # the uid slot now belongs to another identity
        result.append(
            MinerSnapshot(
                uid=uid,
                hotkey=hotkey,
                coldkey=str(getattr(neuron, "coldkey")),
                ip=str(getattr(neuron, "ip", "") or ""),
                track=row_track,
                accumulate_score=0.0,
                excluded=False,
                alpha_stake=float(getattr(neuron, "alpha_stake", 0.0) or 0.0),
            )
        )
        present.add(hotkey)
        taken_uids.add(uid)
    result.sort(key=lambda miner: miner.uid)
    return result


def result_epoch_carrier_hotkeys(
    podium_hotkeys: Sequence[str], competition_input: object | None
) -> tuple[str, ...]:
    """Hotkeys that need an economic row in the epoch that applies a competition result:
    the paid podium first, then every other contender with committed evidence in
    ``competition_input`` (a baseline carries no identity and is skipped)."""
    hotkeys = list(dict.fromkeys(podium_hotkeys))
    seen = set(hotkeys)
    for subject in getattr(competition_input, "subjects", None) or ():
        hotkey = getattr(subject, "hotkey", None)
        if getattr(subject, "role", None) == "contender" and hotkey and hotkey not in seen:
            hotkeys.append(hotkey)
            seen.add(hotkey)
    return tuple(hotkeys)


def with_treasury_carrier(
    snapshots: Sequence[MinerSnapshot],
    *,
    chain_neurons: Iterable[object],
    treasury_hotkey: str | None,
) -> list[MinerSnapshot]:
    """Return ``snapshots`` plus a zero-score row for the registered licensing treasury.

    The treasury (tokenomics v4) runs no miner, so like a podium carrier it has no
    economic row of its own; the row makes its policy share payable. An unset or
    deregistered treasury adds nothing, and its share goes to the sink."""
    if not treasury_hotkey:
        return list(snapshots)
    return with_podium_carriers(
        snapshots,
        chain_neurons=chain_neurons,
        podium_hotkeys=(treasury_hotkey,),
        track=CARRIER_FALLBACK_TRACK,
    )


__all__ = [
    "CARRIER_FALLBACK_TRACK",
    "carrier_track",
    "result_epoch_carrier_hotkeys",
    "with_podium_carriers",
    "with_treasury_carrier",
]
