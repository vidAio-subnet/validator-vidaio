"""The version ORDERING RULE (vidaio.autoupdater.service module docstring).

Only the leading dotted numeric spine orders; equal spines are lateral, never a
downgrade. This file is the rule's executable definition — change them together.
"""

from __future__ import annotations

import pytest

from vidaio.autoupdater import compare_versions, version_key


def test_version_key_is_the_leading_numeric_spine() -> None:
    assert version_key("1.2.10") == (1, 2, 10)
    assert version_key("0.1.0") == (0, 1, 0)
    assert version_key(" 2.0 ") == (2, 0)
    # an optional leading v and any suffix are ignored — only the spine orders
    assert version_key("v0.3.1-rc2") == (0, 3, 1)
    assert version_key("1.2.3+build.7") == (1, 2, 3)
    # no digits at all -> the empty spine
    assert version_key("main") == ()
    assert version_key("") == ()


def test_numeric_ordering_not_lexicographic() -> None:
    assert compare_versions("0.1.10", "0.1.9") == 1  # 10 > 9 numerically
    assert compare_versions("0.2.0", "0.1.9") == 1
    assert compare_versions("0.1.9", "0.1.10") == -1
    assert compare_versions("1.0.0", "0.99.99") == 1


def test_shorter_spines_are_zero_padded() -> None:
    assert compare_versions("1.2", "1.2.0") == 0
    assert compare_versions("1.2.1", "1.2") == 1
    assert compare_versions("1", "1.0.0") == 0


@pytest.mark.parametrize(
    ("candidate", "baseline"),
    [
        ("0.1.0-rc1", "0.1.0"),  # suffix change at the same spine
        ("v0.1.0", "0.1.0"),  # cosmetic v
        ("main", "trunk"),  # no digits anywhere
        ("nightly", "1.2.3-x"),  # empty spine vs padded spine... see below
    ],
)
def test_equal_or_empty_spines_are_lateral_never_a_downgrade(
    candidate: str, baseline: str
) -> None:
    # () pads to all-zeros, so "nightly" vs "1.2.3-x" is NOT lateral — split it out
    if version_key(candidate) == () and version_key(baseline) != ():
        assert compare_versions(candidate, baseline) == -1
    else:
        assert compare_versions(candidate, baseline) == 0


def test_downgrade_is_strictly_below() -> None:
    assert compare_versions("0.1.0", "0.2.0") == -1
    assert compare_versions("0.1.0-rc1", "0.1.1") == -1
