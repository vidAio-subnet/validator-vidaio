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
"""

from __future__ import annotations

from typing import Iterable, Sequence

from vidaio.tokenomics.state import MinerSnapshot


def with_podium_carriers(
    snapshots: Sequence[MinerSnapshot],
    *,
    chain_neurons: Iterable[object],
    podium_hotkeys: Sequence[str],
    track: str | None,
) -> list[MinerSnapshot]:
    """Return ``snapshots`` plus one zero-score row per registered, absent podium hotkey."""
    result = list(snapshots)
    if not podium_hotkeys or not track:
        return result
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
                track=track,
                accumulate_score=0.0,
                excluded=False,
                alpha_stake=float(getattr(neuron, "alpha_stake", 0.0) or 0.0),
            )
        )
        present.add(hotkey)
        taken_uids.add(uid)
    result.sort(key=lambda miner: miner.uid)
    return result


__all__ = ["with_podium_carriers"]
