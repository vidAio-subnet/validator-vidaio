"""Per-track inference pools: top-N graded rank curve (design spec §03).

Emission splits across tracks by the fixed `track_weights`. Within a track, miners rank
by accumulated score; the top N (5) take deterministic descending linear rank weights
N, N-1, ..., 1, rank N+1+ take 0, and the live prefix renormalizes within its track.
An absent or below-floor track stays absent instead of donating its allocation to a
different track.

(The retention-multiplier reshaping of the top-N split was REMOVED with the retention
feature for v1 — retention removed — owner decision; an internal review — so rank weight no longer depends on any windowed retention input.)

Eligibility is fully deterministic and score-gated: excluded miners, the -1 sentinel,
and accumulated scores below the configured absolute payout floor take nothing;
selection-time dedup keeps only the first (lowest) uid per IP and per coldkey.
"""

from __future__ import annotations

import ipaddress
from typing import Iterable, Sequence

from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.ewma import is_excluded
from vidaio.tokenomics.state import MinerSnapshot


def dedup_ip_key(ip: str) -> str | None:
    """Return the usable IP identity for dedup, or ``None`` when it is unknown.

    Bittensor represents a neuron that is not serving an axon with an unspecified
    address (normally ``0.0.0.0``; IPv6 transports may expose ``::``). That means
    *no address was advertised* -- it is not a shared network identity. Coldkey
    dedup remains enforced for miners whose IP is unavailable.

    Canonicalize valid advertised addresses so alternate IPv6 spellings (and
    IPv4-mapped IPv6) cannot evade dedup. Preserve malformed non-empty values as
    exact strings; only explicit unavailable-address representations are exempt.
    """
    value = ip.strip()
    if not value:
        return None
    try:
        parsed = ipaddress.ip_address(value)
    except ValueError:
        return value
    if parsed.is_unspecified:
        return None
    # IPv4-mapped unspecified IPv6 is another unavailable-axon spelling.
    if isinstance(parsed, ipaddress.IPv6Address):
        mapped = parsed.ipv4_mapped
        if mapped is not None and mapped.is_unspecified:
            return None
        if mapped is not None:
            return str(mapped)
    return str(parsed)


def dedup_losers(candidates: Iterable[MinerSnapshot]) -> set[int]:
    """The uids that LOSE the IP/coldkey dedup among ``candidates`` (the dedup RULE).

    Deterministic: candidates are scanned in uid order and the lowest uid wins each
    advertised IP and each coldkey slot; every later miner colliding on an
    already-claimed IP OR coldkey is a loser. Unspecified/blank IPs do not occupy an
    IP slot. This is the ONE dedup implementation — ``eligible_for_ranking``
    (weight-vector selection) and the auditor's independent ``excluded`` re-derivation
    from the close-block metagraph BOTH go through it, so there is a single dedup rule
    to reason about. The caller decides which miners are candidates
    (the score/exclusion eligibility gate); this only resolves the collisions.
    """
    seen_ips: set[str] = set()
    seen_coldkeys: set[str] = set()
    losers: set[int] = set()
    for miner in sorted(candidates, key=lambda m: m.uid):
        ip_key = dedup_ip_key(miner.ip)
        if (ip_key is not None and ip_key in seen_ips) or miner.coldkey in seen_coldkeys:
            losers.add(miner.uid)
            continue
        if ip_key is not None:
            seen_ips.add(ip_key)
        seen_coldkeys.add(miner.coldkey)
    return losers


def dedup_excluded(
    miners: Iterable[MinerSnapshot], *, minimum_payout_score: float = 0.0
) -> set[int]:
    """The uids the IP/coldkey dedup would EXCLUDE — the auditor's independent
    re-derivation of the ``excluded`` flag from the close-block metagraph identities
.

    Gates on SCORE only (a non-scoring / sentinel miner never occupies or shadows a
    slot — matching ``eligible_for_ranking``), and DELIBERATELY IGNORES each miner's own
    ``excluded`` flag: that flag is exactly what is being re-derived, so consulting it
    would be circular (a correctly-excluded dup would falsely re-derive as not-a-dup).
    The IP/coldkey collisions come from the caller's metagraph-sourced identities via the
    shared ``dedup_losers`` rule, so the outcome is bound to the chain, not the log.
    """
    candidates = [
        m
        for m in miners
        if not is_excluded(m.accumulate_score)
        and m.accumulate_score >= minimum_payout_score
        and m.accumulate_score > 0.0
    ]
    return dedup_losers(candidates)


def eligible_for_ranking(
    miners: Iterable[MinerSnapshot], *, minimum_payout_score: float = 0.0
) -> list[MinerSnapshot]:
    """Exclusion + sentinel + absolute-score gate, then IP/coldkey dedup.

    Deterministic: candidates are scanned in uid order and the lowest uid wins each IP
    and each coldkey slot. A miner with no genuine positive score can never occupy (or
    shadow) a slot. The IP/coldkey resolution is the shared ``dedup_losers`` rule (an internal review) so the auditor re-derives the SAME dedup the vector is built from.
    """
    candidates = sorted(
        (
            m
            for m in miners
            if not m.excluded
            and not is_excluded(m.accumulate_score)
            and m.accumulate_score >= minimum_payout_score
            and m.accumulate_score > 0.0
        ),
        key=lambda m: m.uid,
    )
    losers = dedup_losers(candidates)
    return [m for m in candidates if m.uid not in losers]


def track_shares(config: TokenomicsConfig, track_miners: Sequence[MinerSnapshot]) -> dict[int, float]:
    """Within-track shares summing to 1.0 (empty dict if no rankable miner).

    Callers pass already-eligible miners of one track. Rank order: score descending,
    uid ascending as the deterministic tie-break. Rank ``r`` in a top-N curve receives
    the positive integer weight ``N - r + 1``. The selected live prefix is normalized
    within the track, so with five miners the shares are exactly 5/15, 4/15, 3/15,
    2/15 and 1/15. (Retention reshaping remains removed; this curve depends only on
    committed rank and uid.)
    """
    ranked = sorted(track_miners, key=lambda m: (-m.accumulate_score, m.uid))
    top = ranked[: config.top_n_per_track]
    if not top:
        return {}
    weighted = {
        miner.uid: float(config.top_n_per_track - rank)
        for rank, miner in enumerate(top)
    }
    total = sum(weighted.values())
    if total <= 0.0:
        return {}
    return {uid: share / total for uid, share in weighted.items()}


def inference_shares(config: TokenomicsConfig, miners: Iterable[MinerSnapshot]) -> dict[int, float]:
    """Cross-track inference shares summing to 1.0 (empty dict if nobody is eligible).

    Track weights are fixed, not renormalized over live tracks: an absent/below-floor
    track contributes no shares and its allocation remains unassigned at this layer.
    Miners on tracks absent from `track_weights` take nothing. Composition order is
    deterministic (config track order, uid order within tracks).
    """
    eligible = eligible_for_ranking(
        miners, minimum_payout_score=config.minimum_payout_score
    )
    by_track: dict[str, list[MinerSnapshot]] = {}
    for miner in eligible:
        if miner.track in config.track_weights:
            by_track.setdefault(miner.track, []).append(miner)
    shares: dict[int, float] = {}
    for track, pool in config.track_weights.items():
        for uid, share in track_shares(config, by_track.get(track, ())).items():
            shares[uid] = shares.get(uid, 0.0) + pool * share
    return shares
