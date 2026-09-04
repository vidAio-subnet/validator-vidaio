"""The ONE canonical deterministic float->u16 weight quantizer (the shared home).

This is the single source of the submission quantizer the whole system converges
on (the project design record rule 9, the project design record Part 3). It is the TAIL
of the weight pipeline and is therefore **dependency-free** — it imports nothing
heavy (only `math`), so the chain adapter, the (future) epoch-log finalizer, the
publication document, and the auditor can ALL reuse it without dragging any
service/SDK tree in. Consolidating it here (out of `vidaio/chain/bittensor_adapter`)
is the project design record build-wave 1 (role-remap: "CONSOLIDATE").

The determinism property is a hard contract (tested here AND in tests/chain): two
validators that hold the SAME float vector emit BYTE-IDENTICAL u16 output — that
is what lets independent validators submit the same vector and clear Yuma
consensus without clipping.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction

#: The chain's weight grid: bittensor stores u16 pairs and MAX-normalizes a
#: submitted vector (largest weight -> this value). Mirrored by the chain
#: adapter (re-exported there) and equal to
#: vidaio.weightsetter.intents.WEIGHT_QUANTIZATION_SCALE — every producer/comparator
#: puts weights on this same grid.
U16_MAX = 65535


def quantize_u16(weights: dict[int, float]) -> dict[int, int]:
    """Put a float vector on VIDAIO's authority sum-grid (sum EXACTLY 65535).

    THE deterministic quantization the whole system converges on
    (the project design record rule 9, the project design record Part 3). Two validators
    that hold the SAME float vector get BYTE-IDENTICAL SDK input. Bittensor then deterministically maps
    this sum-grid to the runtime max-grid via :func:`max_normalize_u16`.

    The scheme (a production-proven normalizer without a burn slot, since
    burn_proportion = 0.0 is locked — an empty epoch's 100%-burn convergence
    vector is handled UPSTREAM
    in build_weight_vector, which hands this a single-uid `{burn_uid: 1.0}` that
    quantizes to `{burn_uid: 65535}`):

    1. Drop non-positive entries (they carry no emission and the chain records
       none). Empty in -> empty out.
    2. Iterate uids in SORTED order and do EVERY deciding computation in EXACT
       rational arithmetic (`fractions.Fraction`), never floating point. Each input
       float is converted to its exact binary value via `Fraction(w)` — a pure,
       platform-independent function of the identical input bits — so the sum, the
       normalization, the ideal shares and the remainder ranks are bit-for-bit the
       same on ANY platform. (Floating `math.fsum` can differ by one LSB across
       non-Windows builds — Python's own docs note this — which near an integer
       quota or a remainder tie could flip a u16 allocation across heterogeneous
       validator images; exact arithmetic removes that last source of divergence, #18.)
    3. Sum-normalize and take each ideal share `p_i * 65535`; FLOOR to int but
       floor every strictly-positive miner UP to >= 1 unit, so a sub-1/65535
       share never silently falls to zero (a tiny weight must stay
       comparable on the grid).
    4. Hand the rounding remainder out by LARGEST fractional remainder
       (largest-remainder rounding), tie-broken by uid, so the vector sums to
       exactly 65535.
    5. If flooring over-allocated (many tiny weights pushed the sum past 65535),
       reclaim the overflow off the HEAVIEST miners by water-filling, never
       below 1 (overflow reclaim).

    Determinism is a tested property (tests/tokenomics + tests/chain): same floats
    -> identical u16, sum == 65535, and two independent calls are byte-identical —
    now under EXACT arithmetic, so byte-identity holds across heterogeneous builds
    (#18), not merely same-build runs.
    """
    positive = {int(uid): float(w) for uid, w in weights.items() if float(w) > 0.0}
    if not positive:
        return {}

    uids = sorted(positive)
    if len(uids) > U16_MAX:
        # Pathological: more positive miners than u16 units. There is no vector
        # that gives each >= 1 and sums to 65535; refuse rather than silently
        # drop the tail (which would make two validators disagree on WHICH tail).
        raise ValueError(
            f"cannot quantize {len(uids)} positive weights onto a {U16_MAX}-unit grid"
        )

    # EXACT rational arithmetic in the deciding path — no float sum, no float
    # division (#18). Fraction(w) is the exact value of the identical input float.
    exact = {uid: Fraction(positive[uid]) for uid in uids}
    total = sum(exact.values(), Fraction(0))
    ideal = {uid: exact[uid] / total * U16_MAX for uid in uids}  # exact Fraction
    # floor of a non-negative Fraction, exactly (math.floor(Fraction) -> exact int).
    floors = {uid: math.floor(ideal[uid]) for uid in uids}
    floored = {uid: max(1, floors[uid]) for uid in uids}
    allocated = sum(floored.values())
    remainder = U16_MAX - allocated

    if remainder > 0:
        # Largest fractional remainder first; ties by uid. The remainder is an EXACT
        # Fraction, so the rank order is identical on every platform. Cycle if
        # (rarely) the flooring-to-1 left a remainder larger than the uid count.
        order = sorted(uids, key=lambda uid: (-(ideal[uid] - floors[uid]), uid))
        for i in range(remainder):
            floored[order[i % len(order)]] += 1
    elif remainder < 0:
        # Overflow from flooring tiny weights up to 1: reclaim off the heaviest,
        # never below 1 (water-filling, deterministic order).
        deficit = -remainder
        order = sorted(uids, key=lambda uid: (-floored[uid], uid))
        i = 0
        while deficit > 0:
            uid = order[i % len(order)]
            if floored[uid] > 1:
                floored[uid] -= 1
                deficit -= 1
            i += 1

    return floored


def max_normalize_u16(weights: dict[int, float]) -> dict[int, int]:
    """Mirror Bittensor 10.5's final ``convert_weights_and_uids_for_emit`` step.

    ``quantize_u16`` is VIDAIO's convergence/evidence representation: it uses the
    complete u16 sum grid and therefore always totals :data:`U16_MAX`.  The pinned
    SDK does one more conversion before constructing the extrinsic, however: it
    divides every value by the *largest* value, multiplies by 65535, applies
    Python's :func:`round`, and removes zeros.  Consequently the vector actually
    stored by the runtime is max-normalized (largest value == 65535), not generally
    sum-normalized.

    This dependency-free helper deliberately follows that exact deciding path so
    adapters, durable intent reconciliation, and tests can name the on-chain bytes
    without importing the SDK. Non-positive inputs are omitted; callers that need
    Bittensor's public negative-value rejection must validate before calling this
    canonicalizer. Because the SDK first casts each input to binary32, two
    mathematically proportional float vectors can differ by one u16 step after
    rounding; this helper intentionally preserves that boundary instead of claiming
    idealized scale invariance.
    """
    # ``Subtensor.set_weights`` first turns Python lists into ``np.float32``.
    # Round-trip through IEEE-754 binary32 with stdlib ``struct`` so this helper
    # stays dependency-free while matching that otherwise easy-to-miss boundary.
    converted = {
        int(uid): struct.unpack("!f", struct.pack("!f", float(weight)))[0]
        for uid, weight in weights.items()
    }
    positive = {uid: weight for uid, weight in converted.items() if weight > 0.0}
    if not positive:
        return {}

    maximum = float(max(positive.values()))
    emitted: dict[int, int] = {}
    for uid in sorted(positive):
        # Keep the float division + builtin round used by bittensor==10.5.0 rather
        # than substituting Fraction/Decimal: exact differential tests below pin
        # half-even rounding at the SDK's boundary values.
        value = round(float(positive[uid]) / maximum * int(U16_MAX))
        if value != 0:
            emitted[uid] = int(value)
    return emitted
