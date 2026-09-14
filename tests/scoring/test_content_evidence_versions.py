"""Content rule identities retain historical commitments across a size-band roll."""

from types import SimpleNamespace

import pytest

from vidaio.scoring.content_duplicate_evidence import (
    CONTENT_EVIDENCE_RULE,
    CONTENT_EVIDENCE_RULE_V1,
    CONTENT_EVIDENCE_RULE_V2,
    InvalidContentEvidence,
    content_duplicate_identity,
    content_duplicate_identity_v1,
    same_content,
)


def test_historical_content_scorer_identities_keep_their_committed_bytes():
    args = dict(committed_scorer_version="scorer/1+0123456789ab", track="compression",
        scoring_config_digest="f" * 64)
    # Historical values were captured from the implementation before this rule roll.
    assert content_duplicate_identity_v1(**args) == "validator-content-duplicate/1+18948b075862"
    assert content_duplicate_identity(**args, evidence_rule=CONTENT_EVIDENCE_RULE_V1) == (
        "validator-content-duplicate/2+b23db47fbc1e")
    assert content_duplicate_identity(**args, evidence_rule=CONTENT_EVIDENCE_RULE_V2) == (
        "validator-content-duplicate/3+cdf5c84293ec")
    assert content_duplicate_identity(**args) == "validator-content-duplicate/4+39d5712c414d"
    assert content_duplicate_identity(**args, evidence_rule=CONTENT_EVIDENCE_RULE) == (
        "validator-content-duplicate/4+39d5712c414d")


@pytest.mark.parametrize("size,expected", [(9901, True), (9900, True), (9899, False)])
def test_historical_predicate_keeps_its_exact_inclusive_one_percent_boundary(size, expected):
    left = SimpleNamespace(encoded_size=size, canonical_content_digest="a" * 64,
        content_fingerprint=("0" * 16,) * 32, canonicalization_plan_digest="c" * 64)
    right = SimpleNamespace(**{**vars(left), "encoded_size": 10000, "canonical_content_digest": "b" * 64})
    assert same_content(left, right, evidence_rule=CONTENT_EVIDENCE_RULE_V1) is expected
    assert same_content(right, left, evidence_rule=CONTENT_EVIDENCE_RULE_V1) is expected
    assert not same_content(left, right)


def test_unknown_rule_cannot_fall_back_to_historical_identity_or_digest_equality():
    member = SimpleNamespace(canonical_content_digest="a" * 64, canonicalization_plan_digest="c" * 64)
    with pytest.raises(InvalidContentEvidence, match="unsupported content evidence_rule"):
        same_content(member, member, evidence_rule="canonical_content/4")
    with pytest.raises(InvalidContentEvidence, match="unsupported content evidence_rule"):
        content_duplicate_identity(committed_scorer_version="scorer/1", track="compression",
            scoring_config_digest="f" * 64, evidence_rule="canonical_content/4")
