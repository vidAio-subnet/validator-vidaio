"""Unit tests for the consolidated deterministic u16 quantizer (its shared home).

`quantize_u16` moved out of vidaio/chain/bittensor_adapter.py into the dep-free
vidaio/tokenomics/quantize.py (the project design record wave 1). The chain
adapter, the epoch-log finalizer, publication and the auditor all reuse THIS one
function, so the BYTE-IDENTICAL determinism property (DECISIONS rule 9) is proven
here at the source, not only in tests/chain.
"""

from __future__ import annotations

import pytest

from vidaio.tokenomics import max_normalize_u16, quantize_u16
from vidaio.tokenomics.quantize import U16_MAX


def test_sums_to_exactly_65535():
    for vec in ({1: 0.5, 2: 0.3, 3: 0.2}, {7: 1.0}, {1: 0.1, 2: 0.1, 3: 0.8}):
        assert sum(quantize_u16(vec).values()) == U16_MAX


def test_deterministic_and_byte_identical_across_validators():
    # Two "validators" holding the SAME float vector (built in DIFFERENT key
    # orders) must emit BYTE-IDENTICAL u16 — the DECISIONS rule-9 property.
    vec_a = {3: 0.2, 1: 0.5, 2: 0.3}
    vec_b = {1: 0.5, 2: 0.3, 3: 0.2}
    q_a = quantize_u16(vec_a)
    q_b = quantize_u16(vec_b)
    assert q_a == q_b
    assert sorted(q_a.items()) == sorted(q_b.items())
    assert quantize_u16(vec_a) == q_a  # stable across repeated calls


def test_drops_non_positive_and_empty_in_empty_out():
    assert quantize_u16({1: 0.0, 2: -3.0}) == {}
    assert quantize_u16({}) == {}
    q = quantize_u16({1: 1.0, 2: 0.0, 3: -1.0})
    assert set(q) == {1}


def test_floors_tiny_positive_share_to_at_least_one_unit():
    q = quantize_u16({1: 1e-12, 2: 1.0})
    assert q[1] >= 1
    assert sum(q.values()) == U16_MAX


def test_refuses_more_positive_miners_than_grid_units():
    with pytest.raises(ValueError):
        quantize_u16({uid: 1.0 for uid in range(U16_MAX + 1)})


def test_water_fill_reclaim_when_flooring_overflows():
    # Many tiny weights each floored up to >= 1 overshoot 65535; reclaim shaves the
    # deficit off the heaviest and the vector still sums to exactly 65535.
    weights = {0: 1000.0}
    weights.update({uid: 1e-9 for uid in range(1, 5000)})
    q = quantize_u16(weights)
    assert sum(q.values()) == U16_MAX
    assert min(q.values()) >= 1
    assert q[0] == max(q.values())


def test_single_uid_burn_vector_quantizes_to_full_grid():
    # The empty-epoch convergence vector build_weight_vector emits: {burn_uid: 1.0}
    # must land on the whole grid at the burn uid (DECISIONS rule 11).
    assert quantize_u16({94: 1.0}) == {94: U16_MAX}


def test_burn_uid_matches_the_adapters_reexported_function():
    # The chain adapter re-exports the SAME object; consolidation is single-sourced.
    from vidaio.chain.bittensor_adapter import quantize_u16 as adapter_quantize

    assert adapter_quantize is quantize_u16


def test_max_normalize_u16_matches_pinned_sdk_boundary_vectors():
    # These include a value that becomes exactly 0.5 of a u16 step (banker's
    # rounding drops it) and the neighbouring value that survives. They pin the
    # exact float-division/builtin-round path used by bittensor==10.5.0.
    vectors = (
        {1: 5.0, 2: 3.0, 3: 2.0},
        {7: 1.0},
        {1: 1.0, 2: 0.5 / U16_MAX, 3: 0.5000001 / U16_MAX},
        {9: 39321.0, 4: 26214.0},
    )
    expected = (
        {1: 65535, 2: 39321, 3: 26214},
        {7: 65535},
        {1: 65535, 3: 1},
        {4: 43690, 9: 65535},
    )
    assert tuple(max_normalize_u16(vector) for vector in vectors) == expected


def test_max_normalize_u16_differential_against_pinned_bittensor_sdk():
    bt = pytest.importorskip("bittensor", reason="optional pinned chain extra")
    pytest.importorskip("numpy", reason="bittensor emit helper input")
    from bittensor.utils.weight_utils import convert_and_normalize_weights_and_uids

    assert bt.__version__ == "10.5.0"
    for vector in (
        {1: 5.0, 2: 3.0, 3: 2.0},
        {7: 1.0},
        {1: 1.0, 2: 0.5 / U16_MAX, 3: 0.5000001 / U16_MAX},
        {9: 39321.0, 4: 26214.0},
    ):
        uids = sorted(vector)
        sdk_uids, sdk_values = convert_and_normalize_weights_and_uids(
            uids,
            [vector[uid] for uid in uids],
        )
        assert max_normalize_u16(vector) == dict(zip(sdk_uids, sdk_values, strict=True))
